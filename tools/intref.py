#!/usr/bin/env python3
"""Bit-exact integer reference for the BBC Micro inference code (src/llm.6502).

Every operation here is exactly what the 6502 does: int8 weights and
activations, 16-bit wrapping accumulators, power-of-two shifts, a table
RMSNorm, a table exp() softmax, 8-step restoring division, xorshift16
sampling with temperature and top-k.  test/verify.mjs checks the Beeb
against this token for token.

usage:
  intref.py eval   [model.npz]                 bits/char of the integer model on the held-out text
  intref.py gen    [model.npz] SEED KT TOPK N "prompt"
"""
import json, math, sys
import numpy as np

EXP_LEN = 64
EXP = [int(math.floor(127 * math.exp(-i / 8) + 0.5)) for i in range(EXP_LEN)]
NORM_BITS = 5


def w16(x):
    return ((np.asarray(x, dtype=np.int64) + 32768) & 0xFFFF) - 32768


class IntModel:
    def __init__(self, path):
        z = np.load(path)
        self.cfg = c = json.loads(str(z["cfg"])); sc = json.loads(str(z["scales"]))
        self.vocab = str(z["vocab"])
        self.d, self.H, self.hd, self.L, self.F, self.T, self.V = (c[k] for k in ("d", "H", "hd", "L", "F", "T", "V"))
        assert self.T & (self.T - 1) == 0
        self.ar = sc["ar"]
        self.E = z["E"].astype(np.int64); self.ae = int(z["ae"])
        assert self.ae <= self.ar
        self.emb_shift = self.ar - self.ae
        # RMSNorm table: y = x8 * R[m] >> (e + NK), R[m] = 32*sqrt(d)*2^NK/sqrt(m), m in 64..255
        C = (1 << NORM_BITS) * math.sqrt(self.d)
        self.NK = max(k for k in range(12) if math.floor(C * 2 ** k / 8 + 0.5) <= 127)
        self.R = [0] * 64 + [int(math.floor(C * 2 ** self.NK / math.sqrt(m) + 0.5)) for m in range(64, 256)]
        self.layers = []
        warn = []

        def lin(prefix, a_in, a_out=None, res=False):
            W = z[prefix + ".W"].astype(np.int64); bw = int(z[prefix + ".bits"]); bf = z[prefix + ".b"]
            acc_bits = a_in + bw
            S = acc_bits - (self.ar if res else a_out)
            if not res: assert S >= 1, (prefix, S)
            b = np.floor(bf * 2.0 ** acc_bits + 0.5).astype(np.int64)
            if S > 0: b = b + (1 << (S - 1))
            if np.abs(b).max() > 32767: warn.append(prefix)
            return dict(W=W, b=np.clip(b, -32768, 32767), S=S)

        for l in range(self.L):
            s = sc[f"l{l}"]
            L = dict(q=lin(f"l{l}.wq", NORM_BITS, s["aq"]), k=lin(f"l{l}.wk", NORM_BITS, s["ak"]),
                     v=lin(f"l{l}.wv", NORM_BITS, s["av"]), o=lin(f"l{l}.wo", s["av"], res=True),
                     f1=lin(f"l{l}.w1", NORM_BITS, s["ah"]), f2=lin(f"l{l}.w2", s["ah"], res=True),
                     rb=z[f"l{l}.rb"].astype(np.int64), SQK=s["aq"] + s["ak"] - 3)
            assert L["SQK"] >= 1
            self.layers.append(L)
        self.out = lin("out", NORM_BITS, 3)
        if warn: print("warning: bias clipped in", warn, file=sys.stderr)
        self.overflow = 0

    # ---- primitives (rows = leading axes)
    def norm(self, r):
        r = np.asarray(r, dtype=np.int64)
        out = np.zeros_like(r)
        for idx in np.ndindex(r.shape[:-1]):
            v = r[idx]
            mx = int(np.abs(v).max())
            if mx == 0: continue
            if mx >= 128:
                s = 1
                while (mx >> s) > 127: s += 1
                x8 = v >> s
            else:
                s = 0
                while (mx << s) < 64: s += 1
                x8 = v << s
            ss = int((x8 * x8).sum())
            e = 0
            while ss >= 256: ss >>= 2; e += 1
            sh = e + self.NK
            y = (x8 * self.R[ss] + (1 << (sh - 1))) >> sh
            out[idx] = np.clip(y, -127, 127)
        return out

    def matvec(self, p, x):            # 16-bit wrapping accumulators
        acc = p["b"] + x @ p["W"].T
        w = w16(acc)
        self.overflow += int((w != acc).sum())
        return w

    def requant(self, acc, S, lo=-127):
        return np.clip(acc >> S, lo, 127)

    def resadd(self, r, acc, S):
        d = (acc >> S) if S >= 0 else w16(acc << -S)
        return w16(r + d)

    def attend(self, L, q, K, V, dist):
        """q: (d,) int8; K, V: (n, hd) int8 for the n visible slots; dist: (n,) 0..T-1"""
        o = np.zeros(self.d, dtype=np.int64)
        for h in range(self.H):
            qh = q[h * self.hd:(h + 1) * self.hd]
            acc = w16(K @ qh)
            s = (acc >> L["SQK"]) - L["rb"][h][dist]
            delta = s.max() - s
            e = np.array([EXP[x] if x < EXP_LEN else 0 for x in delta], dtype=np.int64)
            E = int(e.sum())
            N = e @ V
            q8 = (np.abs(N) + (E >> 1)) // E
            o[h * self.hd:(h + 1) * self.hd] = np.where(N < 0, -q8, q8)
        return o

    def logits(self, r):
        x = self.norm(r)
        return np.clip(self.matvec(self.out, x) >> self.out["S"], -127, 127)

    # ---- incremental (what the Beeb does)
    def reset(self):
        self.pos = 0
        self.K = np.zeros((self.L, self.T, self.hd), dtype=np.int64)
        self.Vc = np.zeros((self.L, self.T, self.hd), dtype=np.int64)

    def step(self, tok, want_logits=True):
        T = self.T
        r = self.E[tok] << self.emb_shift
        slot = self.pos & (T - 1); n = min(self.pos + 1, T)
        slots = np.array([(slot - i) & (T - 1) for i in range(n)])   # dist i
        dist = np.arange(n)
        for l, L in enumerate(self.layers):
            x = self.norm(r)
            q = self.requant(self.matvec(L["q"], x), L["q"]["S"])
            self.K[l, slot] = self.requant(self.matvec(L["k"], x), L["k"]["S"])
            self.Vc[l, slot] = self.requant(self.matvec(L["v"], x), L["v"]["S"])
            o = self.attend(L, q, self.K[l, slots], self.Vc[l, slots], dist)
            r = self.resadd(r, self.matvec(L["o"], o), L["o"]["S"])
            x = self.norm(r)
            h = self.requant(self.matvec(L["f1"], x), L["f1"]["S"], lo=0)
            r = self.resadd(r, self.matvec(L["f2"], h), L["f2"]["S"])
        self.pos += 1
        return self.logits(r) if want_logits else None

    # ---- whole sequence at once (for evaluation); identical maths
    def forward_seq(self, toks):
        T, S = self.T, len(toks)
        r = self.E[np.asarray(toks)] << self.emb_shift
        for L in self.layers:
            x = self.norm(r)
            q = self.requant(self.matvec(L["q"], x), L["q"]["S"])
            k = self.requant(self.matvec(L["k"], x), L["k"]["S"])
            v = self.requant(self.matvec(L["v"], x), L["v"]["S"])
            o = np.zeros_like(q)
            for p in range(S):
                lo = max(0, p - T + 1)
                ks, vs = k[lo:p + 1][::-1], v[lo:p + 1][::-1]
                o[p] = self.attend(L, q[p], ks, vs, np.arange(len(ks)))
            r = self.resadd(r, self.matvec(L["o"], o), L["o"]["S"])
            x = self.norm(r)
            h = self.requant(self.matvec(L["f1"], x), L["f1"]["S"], lo=0)
            r = self.resadd(r, self.matvec(L["f2"], h), L["f2"]["S"])
        return self.logits(r)


class Rng:
    def __init__(self, seed): self.x = seed & 0xFFFF or 1

    def next(self):
        x = self.x
        x ^= (x << 7) & 0xFFFF; x ^= x >> 9; x ^= (x << 8) & 0xFFFF
        self.x = x
        return x


def sample(lu, kT, topk, rng):
    z = (np.asarray(lu, dtype=np.int64) * kT) >> 4
    delta = z.max() - z
    e = [EXP[x] if x < EXP_LEN else 0 for x in delta]
    if topk:
        keep = [False] * len(e)
        for _ in range(topk):
            best = -1
            for i, v in enumerate(e):
                if v > 0 and not keep[i] and (best < 0 or v > e[best]): best = i
            if best < 0: break
            keep[best] = True
        e = [v if keep[i] else 0 for i, v in enumerate(e)]
    E = sum(e)
    mask = 1
    while mask < E - 1: mask = mask * 2 + 1
    while True:
        r = rng.next() & mask
        if r < E: break
    c = 0
    while r >= e[c]: r -= e[c]; c += 1
    return c


def generate(m, seed, kT, topk, n, prompt="", logits=None):
    """returns the list of sampled tokens (prompt not included)"""
    st = {ch: i for i, ch in enumerate(m.vocab)}
    toks = [st["\n"]] + [st[ch] for ch in prompt.lower() if ch in st]
    m.reset(); rng = Rng(seed)
    for t in toks[:-1]: m.step(t, want_logits=False)
    lu = m.step(toks[-1])
    out = []
    for _ in range(n):
        if logits is not None: logits.append([int(v) for v in lu])
        c = sample(lu, kT, topk, rng)
        out.append(c)
        lu = m.step(c)
    return out


def evaluate(m, text, nchunks=12, S=None):
    st = {ch: i for i, ch in enumerate(m.vocab)}
    data = [st[ch] for ch in text if ch in st]
    val = data[-len(data) // 20:]
    S = S or 2 * m.T
    rs = np.random.RandomState(123)
    tot, cnt = 0.0, 0
    for _ in range(nchunks):
        i = rs.randint(0, len(val) - S - 1)
        x, y = val[i:i + S], val[i + 1:i + S + 1]
        lu = m.forward_seq(x)[m.T:] / 8.0                  # nats
        lu = lu - lu.max(-1, keepdims=True)
        lp = lu - np.log(np.exp(lu).sum(-1, keepdims=True))
        tot -= lp[np.arange(len(lp)), y[m.T:]].sum(); cnt += len(lp)
    return tot / cnt / math.log(2)


if __name__ == "__main__":
    cmd = sys.argv[1]
    m = IntModel(sys.argv[2] if len(sys.argv) > 2 else "build/model.npz")
    if cmd == "eval":
        print(f"integer model: {evaluate(m, open('build/corpus.txt').read()):.3f} bpc, accumulator overflows: {m.overflow}")
    elif cmd == "json":       # for test/verify.mjs
        seed, kT, topk, n = map(int, sys.argv[3:7])
        prompt = sys.argv[7] if len(sys.argv) > 7 else ""
        lg = []
        out = generate(m, seed, kT, topk, n, prompt, lg)
        print(json.dumps(dict(tokens=out, logits=lg, text="".join(m.vocab[c] for c in out))))
    elif cmd == "gen":
        seed, kT, topk, n = map(int, sys.argv[3:7])
        prompt = sys.argv[7] if len(sys.argv) > 7 else ""
        out = generate(m, seed, kT, topk, n, prompt)
        print(prompt + "".join(m.vocab[c] for c in out))
