#!/bin/sh
# Rebuild the disc from the committed stand-in weights (no training needed).
set -e
cd "$(dirname "$0")"
mkdir -p build
${PYTHON:-python3} tools/export_beeb.py Tiny=models/standin-tiny.npz Small=models/standin-small.npz \
    Medium=models/standin-medium.npz Large=models/standin-large.npz
./build.sh
tools/make_mmb.sh
cp build/beeb-llm.ssd web/jsbeeb/discs/ && cp build/beeb-llm.ssd build/BEEB-LLM.MMB.zip discs/
