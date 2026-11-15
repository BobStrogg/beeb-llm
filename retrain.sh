#!/bin/sh
# One command: corpus -> train (float + QAT) -> integer export -> disc + MMFS image.
#
#   ./retrain.sh                 trains on corpus/hhgttg.txt if it exists, else the stand-in
#   ./retrain.sh path/to/book.txt
#   QUICK=1 ./retrain.sh         short run to check the pipeline (a few minutes)
#
# A private corpus (anything other than the public-domain stand-in) is built
# into build/ only: models/ and web/ are left alone so copyrighted text, or
# weights trained on it, never end up in the repository or on GitHub Pages.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
mkdir -p build
if [ -n "$1" ]; then SRC="$1"; elif [ -f corpus/hhgttg.txt ]; then SRC=corpus/hhgttg.txt; else SRC=standin; fi
if [ "$SRC" = standin ]; then
  tools/get_standin.sh
  $PY tools/prepare_corpus.py corpus/standin-threemen.txt build/c1.txt
  $PY tools/prepare_corpus.py corpus/standin-bummel.txt build/c2.txt
  cat build/c1.txt build/c2.txt > build/corpus.txt
  PUBLIC=1
else
  $PY tools/prepare_corpus.py "$SRC" build/corpus.txt
  PUBLIC=0
fi
if [ -n "$QUICK" ]; then STEPS="--steps 1500 --qat-steps 300"; else STEPS="--steps 14000 --qat-steps 3000"; fi
$PY tools/train.py --corpus build/corpus.txt --d 32 --L 2 --F 64 --T 32 $STEPS --out build/model.npz
$PY tools/intref.py eval build/model.npz
$PY tools/intref.py gen build/model.npz 1234 20 10 300 "the "
$PY tools/export_beeb.py build/model.npz
./build.sh
[ -d vendor/MMB_Utils ] && tools/make_mmb.sh || echo "(vendor/MMB_Utils missing: git submodule update --init to build the .MMB)"
if [ "$PUBLIC" = 1 ] && [ -z "$QUICK" ]; then
  mkdir -p models && cp build/model.npz models/standin.npz
  cp build/beeb-llm.ssd web/jsbeeb/discs/beeb-llm.ssd && cp build/beeb-llm.ssd build/BEEB-LLM.MMB.zip discs/
  echo "stand-in model: copied to models/standin.npz, discs/ and web/"
else
  [ "$PUBLIC" = 1 ] && echo "QUICK run: models/ and web/ left alone" || echo "private corpus: build/beeb-llm.ssd and build/BEEB-LLM.MMB.zip are for your own machines - not copied to models/ or web/"
fi
