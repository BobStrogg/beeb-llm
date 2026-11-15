"""Tiny character-level transformer, trained in float and then with
quantisation-aware training (QAT) that simulates the BBC Micro's integer maths.

Architecture (all of it maps 1:1 onto the 6502 code in src/):
  * token embedding, no position embedding (position comes from a learned
    per-head relative bias, so the KV cache is a ring buffer and generation
    can run forever with a sliding window of T characters)
  * L pre-norm blocks: gainless RMSNorm -> multi-query attention (H query
    heads share one key/value head of size hd) -> RMSNorm -> ReLU MLP
  * final RMSNorm -> output projection to V logits
Every scale is a power of two, so the Beeb only ever shifts.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

EXP_UNITS = 8           # attention/logit scores are in 1/8 nat units on the Beeb
NORM_BITS = 5           # RMSNorm output: 1.0 == 32


def ste(x, q):          # straight-through estimator
    return x + (q - x).detach()


def fq(x, bits, lo, hi, mode="round"):
    """fake-quantise x onto the grid 2^-bits, clamped to [lo, hi] integers"""
    s = 2.0 ** bits
    y = x * s
    q = torch.floor(y + 0.5) if mode == "round" else torch.floor(y)
    q = q.clamp(lo, hi)
    return ste(x, q / s)


def pow2_bits(maxabs, limit=127):
    return int(math.floor(math.log2(limit / max(float(maxabs.detach() if hasattr(maxabs, "detach") else maxabs), 1e-9))))


def fq_weight(w, cap=99):
    b = min(cap, pow2_bits(w.detach().abs().max()))
    return fq(w, b, -127, 127), b


class Cfg:
    def __init__(self, d=32, H=4, L=2, F=64, T=64, V=47):
        self.d, self.H, self.L, self.F, self.T, self.V = d, H, L, F, T, V
        self.hd = d // H

    def as_dict(self):
        return dict(d=self.d, H=self.H, L=self.L, F=self.F, T=self.T, V=self.V, hd=self.hd)


class Block(nn.Module):
    def __init__(self, c: Cfg, idx: int):
        super().__init__()
        d, hd, H, T, Fh = c.d, c.hd, c.H, c.T, c.F
        self.c = c
        self.wq = nn.Linear(d, d); self.wk = nn.Linear(d, hd); self.wv = nn.Linear(d, hd)
        self.wo = nn.Linear(d, d); self.w1 = nn.Linear(d, Fh); self.w2 = nn.Linear(Fh, d)
        with torch.no_grad():
            self.wq.weight.mul_(1 / math.sqrt(hd))
            self.wo.weight.mul_(0.5 / math.sqrt(2 * c.L)); self.w2.weight.mul_(0.5 / math.sqrt(2 * c.L))
        slopes = torch.tensor([2.0 ** -(1 + 2 * h) for h in range(H)])
        self.rb = nn.Parameter(slopes[:, None] * torch.arange(T)[None, :].float())  # nats, subtracted


def rmsnorm(x, qat):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    return fq(y, NORM_BITS, -127, 127) if qat else y


class TinyLM(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.emb = nn.Parameter(torch.randn(c.V, c.d) * 0.5)
        self.blocks = nn.ModuleList([Block(c, i) for i in range(c.L)])
        self.out = nn.Linear(c.d, c.V)
        self.sc = None      # activation scales (bits) once calibrated -> QAT
        self.caps = {}      # max weight bits per matrix so 16-bit accumulators don't overflow

    def nparams(self):
        return sum(p.numel() for p in self.parameters())

    def band_mask(self, S, device):
        i = torch.arange(S, device=device)
        dist = i[:, None] - i[None, :]
        return dist, (dist >= 0) & (dist < self.c.T)

    def lin(self, layer, x, a_in, qat, name=None, collect=None):
        if collect is not None:
            collect.setdefault(name + ".pre", []).append(layer(x).detach().abs().flatten())
        if not qat: return layer(x), None
        w, bw = fq_weight(layer.weight, self.caps.get(name, 99))
        b = fq(layer.bias, a_in + bw, -32767, 32767)
        return F.linear(x, w, b), bw

    def forward(self, idx, qat=False, collect=None):
        c, sc = self.c, self.sc
        B, S = idx.shape
        dist, ok = self.band_mask(S, idx.device)
        dist_c = dist.clamp(0, c.T - 1)
        if qat:
            e, _ = fq_weight(self.emb)
            r = fq(e[idx], sc["ar"], -32767, 32767)
        else:
            r = self.emb[idx]
        for bi, blk in enumerate(self.blocks):
            s = sc[f"l{bi}"] if qat else None
            x = rmsnorm(r, qat)
            q, _ = self.lin(blk.wq, x, NORM_BITS, qat, f"l{bi}.wq", collect)
            k, _ = self.lin(blk.wk, x, NORM_BITS, qat, f"l{bi}.wk", collect)
            v, _ = self.lin(blk.wv, x, NORM_BITS, qat, f"l{bi}.wv", collect)
            if qat:
                q = fq(q, s["aq"], -127, 127); k = fq(k, s["ak"], -127, 127); v = fq(v, s["av"], -127, 127)
            if collect is not None:
                collect.setdefault(f"l{bi}.q", []).append(q.detach().abs().flatten())
                collect.setdefault(f"l{bi}.k", []).append(k.detach().abs().flatten())
                collect.setdefault(f"l{bi}.v", []).append(v.detach().abs().flatten())
            qh = q.view(B, S, c.H, c.hd).transpose(1, 2)                 # B H S hd
            att = torch.einsum("bhsd,btd->bhst", qh, k)                   # shared K
            if qat:
                att = fq(att, 3, -16384, 16383, mode="floor")
                rb = fq(blk.rb, 3, -127, 127)
            else:
                rb = blk.rb
            att = att - rb[:, dist_c][None]
            att = att.masked_fill(~ok[None, None], float("-inf"))
            if qat:   # the Beeb's exp table: e = round(127*exp(-delta)), cut off after 63/8 nats
                m = att.amax(-1, keepdim=True).detach()
                delta = (m - att)
                ew = 127 * torch.exp(-delta)
                ew = ste(ew, torch.where(delta * EXP_UNITS < 63.5, torch.floor(ew.detach() + 0.5), torch.zeros_like(ew)))
                p = ew / ew.sum(-1, keepdim=True)
            else:
                p = att.softmax(-1)
            o = torch.einsum("bhst,btd->bhsd", p, v).transpose(1, 2).reshape(B, S, c.d)
            if qat: o = fq(o, s["av"], -127, 127)
            y, _ = self.lin(blk.wo, o, s["av"] if qat else 0, qat, f"l{bi}.wo", collect)
            r = r + (fq(y, sc["ar"], -32767, 32767) if qat else y)
            x = rmsnorm(r, qat)
            h, _ = self.lin(blk.w1, x, NORM_BITS, qat, f"l{bi}.w1", collect)
            h = F.relu(h)
            if qat: h = fq(h, s["ah"], 0, 127)
            if collect is not None:
                collect.setdefault(f"l{bi}.h", []).append(h.detach().abs().flatten())
            y, _ = self.lin(blk.w2, h, s["ah"] if qat else 0, qat, f"l{bi}.w2", collect)
            r = r + (fq(y, sc["ar"], -32767, 32767) if qat else y)
            if collect is not None:
                collect.setdefault("r", []).append(r.detach().abs().flatten())
        x = rmsnorm(r, qat)
        logits, _ = self.lin(self.out, x, NORM_BITS, qat, "out", collect)
        if qat: logits = fq(logits, 3, -127, 127, mode="floor")
        return logits
