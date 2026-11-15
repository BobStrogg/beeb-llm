#!/usr/bin/env python3
"""Train the tiny transformer on build/corpus.txt, then QAT, then export
integer weights to build/model.npz (read by intref.py and export_beeb.py).

usage: train.py [--d 32 --L 2 --F 64 --T 64 --steps 8000 --qat-steps 3000 --out build/model.npz]
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from model import TinyLM, Cfg, pow2_bits, NORM_BITS
from prepare_corpus import VOCAB

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="build/corpus.txt")
ap.add_argument("--d", type=int, default=32); ap.add_argument("--H", type=int, default=4)
ap.add_argument("--L", type=int, default=2); ap.add_argument("--F", type=int, default=64)
ap.add_argument("--T", type=int, default=64)
ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--qat-steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=4e-3)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out", default="build/model.npz")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.set_num_threads(args.threads)

text = open(args.corpus).read()
stoi = {ch: i for i, ch in enumerate(VOCAB)}
data = torch.tensor([stoi[ch] for ch in text if ch in stoi], dtype=torch.long)
n = len(data); nval = n // 20
train, val = data[:-nval], data[-nval:]
c = Cfg(d=args.d, H=args.H, L=args.L, F=args.F, T=args.T, V=len(VOCAB))
S = 2 * c.T
model = TinyLM(c)
print(f"corpus {n} symbols, model {model.nparams()} params, cfg {c.as_dict()}", flush=True)


def batch(src, B):
    ix = torch.randint(len(src) - S - 1, (B,))
    x = torch.stack([src[i:i + S] for i in ix]); y = torch.stack([src[i + 1:i + S + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(qat, nb=40):
    g = torch.Generator().manual_seed(123)
    tot = 0.0
    for _ in range(nb):
        ix = torch.randint(len(val) - S - 1, (32,), generator=g)
        x = torch.stack([val[i:i + S] for i in ix]); y = torch.stack([val[i + 1:i + S + 1] for i in ix])
        lg = model(x, qat=qat)
        tot += F.cross_entropy(lg[:, c.T:].reshape(-1, c.V), y[:, c.T:].reshape(-1)).item()
    return tot / nb / math.log(2)   # bits per character (on positions with a full window)


def run(steps, lr, qat):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01 if not qat else 0.0)
    t0 = time.time()
    for it in range(steps):
        warm = min(1.0, (it + 1) / 200)
        cur = lr * warm * (0.5 * (1 + math.cos(math.pi * it / steps)) * 0.9 + 0.1)
        for gp in opt.param_groups: gp["lr"] = cur
        x, y = batch(train, args.batch)
        loss = F.cross_entropy(model(x, qat=qat).reshape(-1, c.V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 1000 == 0 or it == steps - 1:
            print(f"  {'qat' if qat else 'float'} {it:5d} loss {loss.item()/math.log(2):.3f} bpc  val {evaluate(qat):.3f} bpc  {time.time()-t0:.0f}s", flush=True)


print("float training"); run(args.steps, args.lr, False)
fbpc = evaluate(False)

# ---- calibrate activation scales (powers of two) on training data
col = {}
with torch.no_grad():
    for _ in range(20): model(batch(train, 32)[0], collect=col)
def pct(name, p=0.9995): return torch.quantile(torch.cat(col[name])[:2_000_000], p).item()
sc = {}
rmax = torch.cat(col["r"]).max().item()
sc["ar"] = min(pow2_bits(rmax * 4, 32767), pow2_bits(model.emb.abs().max() / 4, 127) + 8)
for l in range(c.L):
    sc[f"l{l}"] = {"a" + k: pow2_bits(pct(f"l{l}.{k}")) for k in "qkvh"}
print("scales", json.dumps(sc), "rmax", round(rmax, 2), flush=True)
# cap weight precision so that |accumulator| stays well inside 16 bits
caps = {"emb": 99}
for name in [f"l{l}.{nm}" for l in range(c.L) for nm in ("wq", "wk", "wv", "wo", "w1", "w2")] + ["out"]:
    a_in = sc[name.split(".")[0]]["av"] if name.endswith("wo") else sc[name.split(".")[0]]["ah"] if name.endswith("w2") else NORM_BITS
    mx = torch.cat(col[name + ".pre"]).max().item()
    if name == "out": mx = max(mx, 16.0)       # logits clamp at +-127/8 nats anyway
    caps[name] = pow2_bits(mx * 2.0, 32767) - a_in
print("weight-bit caps", json.dumps(caps), flush=True)
model.sc = sc; model.caps = caps

print("QAT"); run(args.qat_steps, args.lr * 0.25, True)
qbpc = evaluate(True)
print(f"val bpc: float {fbpc:.3f}  qat {qbpc:.3f}")

# ---- export integers
def wq(w, name):
    w = w.detach().double(); b = min(caps[name], pow2_bits(w.abs().max()))
    return np.clip(np.floor(w.numpy() * 2.0 ** b + 0.5), -127, 127).astype(np.int64), b

ex = {"cfg": json.dumps(c.as_dict()), "scales": json.dumps(sc), "caps": json.dumps(caps), "vocab": VOCAB,
      "float_bpc": fbpc, "qat_bpc": qbpc}
E, ae = wq(model.emb, "emb"); ex["E"] = E; ex["ae"] = ae
for l, blk in enumerate(model.blocks):
    for nm in ("wq", "wk", "wv", "wo", "w1", "w2"):
        w, b = wq(getattr(blk, nm).weight, f"l{l}.{nm}")
        ex[f"l{l}.{nm}.W"] = w; ex[f"l{l}.{nm}.bits"] = b
        ex[f"l{l}.{nm}.b"] = getattr(blk, nm).bias.detach().double().numpy()
    ex[f"l{l}.rb"] = np.clip(np.floor(blk.rb.detach().double().numpy() * 8 + 0.5), -127, 127).astype(np.int64)
w, b = wq(model.out.weight, "out"); ex["out.W"] = w; ex["out.bits"] = b; ex["out.b"] = model.out.bias.detach().double().numpy()
np.savez(args.out, **ex)
torch.save(model.state_dict(), args.out.replace(".npz", ".pt"))
print("wrote", args.out)
