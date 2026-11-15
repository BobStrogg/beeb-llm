#!/usr/bin/env python3
"""Turn a plain-text book into the model's 47-symbol alphabet.

usage: prepare_corpus.py INPUT.txt OUTPUT.txt

* strips a Project Gutenberg header/footer if there is one
* lower-cases, folds curly quotes / dashes / ligatures to ASCII
* joins the lines of each paragraph; one paragraph per line ('\n' is a symbol)
* drops anything that is not in VOCAB
"""
import re, sys

# Token order is the order the BBC Micro uses: token n <-> VOCAB[n].
VOCAB = "\n abcdefghijklmnopqrstuvwxyz.,'\"-;:!?0123456789"
assert len(VOCAB) == len(set(VOCAB))

FOLD = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2014": " - ", "\u2013": "-",
        "\u00e6": "ae", "\u00c6": "ae", "\u0153": "oe", "\u0152": "oe", "\u00e9": "e", "\u00e8": "e",
        "\u00ea": "e", "\u00e0": "a", "\u00e2": "a", "\u00f4": "o", "\u00fb": "u", "\u00ef": "i",
        "\u00e7": "c", "\u00fc": "u", "\u00f6": "o", "\u00e4": "a", "\u2026": "...", "`": "'",
        "(": ", ", ")": ", ", "[": "", "]": "", "_": "", "*": "", "\t": " "}


def prepare(text: str) -> str:
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    m = re.search(r"\*\*\*\s*START OF.*?\*\*\*", text)
    if m: text = text[m.end():]
    m = re.search(r"\*\*\*\s*END OF", text)
    if m: text = text[:m.start()]
    for a, b in FOLD.items():
        text = text.replace(a, b)
    paras = []
    for p in re.split(r"\n\s*\n", text):
        p = " ".join(p.split()).lower()
        p = "".join(ch for ch in p if ch in VOCAB)
        p = re.sub(r" +", " ", p).strip()
        p = re.sub(r" ([,.;:!?])", r"\1", p)
        p = re.sub(r",+", ",", p)
        if p: paras.append(p)
    return "\n".join(paras) + "\n"


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    out = prepare(open(src, encoding="utf-8", errors="replace").read())
    open(dst, "w").write(out)
    print(f"{src}: {len(out)} symbols, {out.count(chr(10))} paragraphs -> {dst}")
