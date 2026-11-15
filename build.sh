#!/bin/sh
# Assemble both builds of src/llm.6502 into build/beeb-llm.ssd:
#   LLM + MENU     (Model B, PARASITE=0)     LLMP + MENUP   (6502 Second Processor, PARASITE=1)
#   M0..M3         the models, from tools/export_beeb.py
#   !BOOT          runs LLMP if a second processor is present (OSBYTE &EA), else LLM
set -e
cd "$(dirname "$0")/src"
beebasm -D PARASITE=0 -i llm.6502 -do ../build/host.ssd -opt 3 -title BEEB-LLM -v > ../build/listing.txt
beebasm -D PARASITE=1 -i llm.6502 -di ../build/host.ssd -do ../build/beeb-llm.ssd -v > ../build/listing-2p.txt
rm ../build/host.ssd
for l in listing listing-2p; do echo "$l:"; grep -E "^(code|bss|model area|menu) " ../build/$l.txt; done
