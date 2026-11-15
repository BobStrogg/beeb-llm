#!/bin/sh
# build/BEEB-LLM.MMB: a 511-slot MMFS image with BEEB-LLM in slot 0 (boots with
# SHIFT+BREAK), zipped with the .ssd as build/BEEB-LLM.MMB.zip
set -e
cd "$(dirname "$0")/.."
U=vendor/MMB_Utils
rm -f build/BEEB-LLM.MMB
perl $U/dblank_mmb.pl -f build/BEEB-LLM.MMB >/dev/null
perl $U/dput_ssd.pl -f build/BEEB-LLM.MMB 0 build/beeb-llm.ssd
perl $U/donboot.pl -f build/BEEB-LLM.MMB 0 0 >/dev/null
python3 - <<'PY'
import zipfile
with zipfile.ZipFile("build/BEEB-LLM.MMB.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("build/BEEB-LLM.MMB", "BEEB-LLM.MMB"); z.write("build/beeb-llm.ssd", "beeb-llm.ssd")
PY
ls -l build/BEEB-LLM.MMB.zip
