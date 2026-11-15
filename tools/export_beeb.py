#!/usr/bin/env python3
"""Write integer models in the BBC Micro's layout: one relocatable file per model.

  tools/export_beeb.py NAME=model.npz [NAME=model.npz ...]   (at most 4, smallest first)

writes
  build/m<i>.bin        the model file M<i> on the disc (header + weights + tables)
  build/catalog.6502    BeebAsm include for the boot menu (names, sizes, needs, speeds)
  build/models.json     the same, for the tests and the README

File layout (all offsets from the start of the file, the 6502 adds its load
address to every word marked "offset" when it loads the model):

  0  D  1 NL  2 FF  3 CTX  4 NV  5 NK  6 EMB_SHIFT  7 SL  8 log2(CTX)  9 0
  10 file size   12 emb (offset)   14 out_w (offset)   16 out_b (offset)   18 rtab (offset)
  20 NL x 30-byte layer entries:
       qkv_w qkv_bl o_w o_bl f1_w f1_bl f2_w f2_bl rb  (9 offsets)
       kcache vcache (filled in by the loader: the KV cache goes after the model)
       shifts sq sk sv so s1 s2 sqk, 0
  then per layer: qkv_w qkv_bl qkv_bh o_w o_bl o_bh f1_* f2_* rb, then emb, out_*, rtab

Matrices are stored column by column with each weight biased by +128, so the
6502 can use the weight directly as the Y index into its quarter-square table.
Biases are split into a low-byte array and a high-byte array (high = low + rows).
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from intref import IntModel

HD = 8
SPEEDS = "models/speeds.json"      # measured characters/second (test/speed.mjs), x100


def u8(a): return [int(v) & 0xFF for v in np.asarray(a).ravel()]


def export(m):
    assert m.hd == HD and m.V == 47 and m.T in (32, 64)
    hdr_len = 20 + 30 * m.L
    blob = bytearray(hdr_len); off = {}

    def put(name, data):
        off[name] = len(blob); blob.extend(bytes(data))

    def matrix(name, p):
        W = p["W"]
        put(name + "_w", [(int(v) + 128) & 0xFF for v in W.T.ravel()])
        b = p["b"]
        put(name + "_bl", [int(v) & 0xFF for v in b]); put(name + "_bh", [(int(v) >> 8) & 0xFF for v in b])

    for l, L in enumerate(m.layers):
        qkv = dict(W=np.concatenate([L["q"]["W"], L["k"]["W"], L["v"]["W"]]),
                   b=np.concatenate([L["q"]["b"], L["k"]["b"], L["v"]["b"]]))
        matrix(f"l{l}_qkv", qkv); matrix(f"l{l}_o", L["o"]); matrix(f"l{l}_f1", L["f1"]); matrix(f"l{l}_f2", L["f2"])
        put(f"l{l}_rb", u8(L["rb"]))
    put("emb", u8(m.E)); matrix("out", m.out)
    put("rtab", [m.R[i] for i in range(64, 256)])
    w = lambda o, v: blob.__setitem__(slice(o, o + 2), bytes([v & 0xFF, v >> 8]))
    blob[0:10] = bytes([m.d, m.L, m.F, m.T, m.V, m.NK, m.emb_shift, m.out["S"], m.T.bit_length() - 1, 0])
    w(10, len(blob)); w(12, off["emb"]); w(14, off["out_w"]); w(16, off["out_bl"]); w(18, off["rtab"])
    for l, L in enumerate(m.layers):
        e = 20 + 30 * l
        for i, k in enumerate(("qkv_w", "qkv_bl", "o_w", "o_bl", "f1_w", "f1_bl", "f2_w", "f2_bl", "rb")):
            w(e + 2 * i, off[f"l{l}_{k}"])
        blob[e + 22:e + 30] = bytes([L["q"]["S"], L["k"]["S"], L["v"]["S"], L["o"]["S"] & 0xFF,
                                     L["f1"]["S"], L["f2"]["S"] & 0xFF, L["SQK"], 0])
    assert len(blob) < 65536
    return blob


def nparams(m):
    return int(sum(np.asarray(v).size for L in m.layers for k in ("q", "k", "v", "o", "f1", "f2") for v in (L[k]["W"], L[k]["b"]))
               + sum(L["rb"].size for L in m.layers) + m.E.size + m.out["W"].size + m.out["b"].size)


if __name__ == "__main__":
    args = [a.split("=", 1) for a in sys.argv[1:]]
    assert 1 <= len(args) <= 4
    speeds = json.load(open(SPEEDS)) if os.path.exists(SPEEDS) else {}
    cat = []
    for i, (name, path) in enumerate(args):
        m = IntModel(path); blob = export(m)
        open(f"build/m{i}.bin", "wb").write(blob)
        z = np.load(path)
        info = dict(name=name, file=f"M{i}", source=path, d=m.d, L=m.L, F=m.F, T=m.T, params=nparams(m),
                    bytes=len(blob), cache=2 * m.L * HD * m.T, rows=max(m.d + 2 * HD, m.F, m.V),
                    qat_bpc=float(z["qat_bpc"]) if "qat_bpc" in z else None,
                    macs=m.L * (m.d * (m.d + 16) + m.d * m.d + 2 * m.d * m.F) + m.d * m.V)
        info["need"] = info["bytes"] + info["cache"]
        sp = speeds.get(name, {})
        info["cps_b"] = sp.get("b"); info["cps_2p"] = sp.get("2p")
        cat.append(info)
        print(f"M{i} {name:7s} {info['params']:6d} params {len(blob):6d} bytes + cache {info['cache']} rows {info['rows']}")
    # speeds not measured yet: estimate from the multiply-accumulate count (Small = 1.58 c/s on a Model B,
    # the 3MHz 65C02 second processor is ~1.45x faster)
    for c in cat:
        if c["cps_b"] is None: c["cps_b"] = round(1.58 * 14816 / c["macs"], 2); c["est"] = True
        if c["cps_2p"] is None: c["cps_2p"] = round(c["cps_b"] * 1.45, 2); c["est"] = True
    json.dump(cat, open("build/models.json", "w"), indent=1)
    with open("build/putfiles.6502", "w") as f:
        for i in range(len(cat)): f.write(f'PUTFILE "../build/m{i}.bin", "M{i}", 0, 0\n')
    with open("build/catalog.6502", "w") as f:
        f.write("\\ generated by tools/export_beeb.py - do not edit (included in the menu overlay)\n")
        f.write(f"NMODELS = {len(cat)}\n")
        f.write(".cat_need EQUW " + ", ".join(str(c["need"]) for c in cat) + "\n")
        f.write(".cat_rows EQUB " + ", ".join(str(c["rows"]) for c in cat) + "\n")
        f.write(".cat_cps_b EQUW " + ", ".join(str(round(c["cps_b"] * 100)) for c in cat) + "\n")
        f.write(".cat_cps_p EQUW " + ", ".join(str(round(c["cps_2p"] * 100)) for c in cat) + "\n")
        f.write(".cat_text\n")       # 18 characters each: name, params, RAM needed
        for c in cat:
            t = f"{c['name'][:6]:6s} {c['params'] / 1000:4.1f}K {c['need'] / 1024:4.1f}K"
            assert len(t) == 18, t
            f.write(f'    EQUS "{t}"\n')
        f.write(".cat_name\n")       # 20 characters each, for the title line
        for c in cat:
            t = f"{c['name']} {c['params'] / 1000:.1f}K params"[:20]
            f.write(f'    EQUS "{t:20s}"\n')
