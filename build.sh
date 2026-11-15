#!/bin/sh
# Assemble src/llm.6502 (+ build/model.bin from tools/export_beeb.py) into build/beeb-llm.ssd
set -e
cd "$(dirname "$0")/src"
beebasm -i llm.6502 -do ../build/beeb-llm.ssd -opt 3 -title BEEB-LLM -v > ../build/listing.txt
grep -E "^(code|model|bss|free)" ../build/listing.txt
