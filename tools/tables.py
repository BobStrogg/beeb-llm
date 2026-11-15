#!/usr/bin/env python3
"""Model table for the README and the web page, from build/models.json (tools/export_beeb.py)
and models/speeds.json (measured in jsbeeb by test/verify.mjs).

  tools/tables.py            prints the markdown table and fills <!--MODELS--> rows in web/index.html
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from intref import IntModel, evaluate, generate

cat = json.load(open("build/models.json"))
speeds = json.load(open("models/speeds.json")) if os.path.exists("models/speeds.json") else {}
text = open("build/corpus.txt").read()
rows, html = [], []
for c in cat:
    m = IntModel(c["source"])
    bpc = evaluate(m, text, 40)
    out = generate(m, 1234, 20, 10, 90, "the ")
    sample = ("the " + "".join(m.vocab[t] for t in out)).replace("\n", " / ")
    sample = sample[:sample.rfind(" ")] + "…"
    sp = speeds.get(c["name"], {})
    b = f'{sp["b"]:.2f}' if "b" in sp else "doesn't fit"
    p = f'{sp["2p"]:.2f}' if "2p" in sp else "?"
    shape = f'{c["L"]} × d{c["d"]}, MLP {c["F"]}'
    rows.append(f'| **{c["name"]}** | {c["params"]:,} | {shape} | {c["need"] / 1024:.1f} KB | {bpc:.2f} | {b} | {p} | {sample} |')
    html.append(f'<tr><td>{c["name"]}</td><td>{c["params"]:,}</td><td>{shape}</td><td>{bpc:.2f}</td><td>{b}</td><td>{p}</td></tr>')
print("| Model | Params | Shape | RAM | Bits/char | Chars/s Model B | Chars/s 2P | Sample (seed 1234, temp 0.8, top-k 10, prompt \"the \") |")
print("|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
p = "web/index.html"; s = open(p).read()
s = re.sub(r"(<table id=\"models\">.*?</tr>\n).*?(</table>)", lambda mo: mo.group(1) + "\n".join(html) + "\n<!--MODELS-->" + mo.group(2), s, flags=re.S)
open(p, "w").write(s)
