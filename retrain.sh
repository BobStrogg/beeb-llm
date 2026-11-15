#!/bin/sh
# One command: corpus -> train all four models (float + QAT) -> integer export -> disc + MMFS image.
#
#   ./retrain.sh                 trains on corpus/hhgttg.txt if it exists, else the stand-in
#   ./retrain.sh path/to/book.txt
#   QUICK=1 ./retrain.sh         short runs to check the pipeline (a few minutes)
#   SIZES="tiny small" ./retrain.sh   retrain only some (the others come from build/, or models/ for the stand-in)
#
# A private corpus (anything other than the public-domain stand-in) is built
# into build/ only: models/, discs/ and web/ are left alone so copyrighted text,
# or weights trained on it, never end up in the repository or on GitHub Pages.
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
shape() { case $1 in
  tiny) echo "--d 24 --L 2 --F 48";;      # 10K params
  small) echo "--d 32 --L 2 --F 64";;     # 17K
  medium) echo "--d 48 --L 2 --F 96";;    # 35K (second processor)
  large) echo "--d 48 --L 3 --F 96";;     # 50K (second processor)
esac; }
# the four trainings run side by side, two threads each
for m in ${SIZES:-tiny small medium large}; do
  echo "training $m: $(shape $m)"
  $PY tools/train.py --corpus build/corpus.txt $(shape $m) --T 32 --threads 2 $STEPS --out build/$m.npz > build/train-$m.log 2>&1 &
done
wait
ARGS=""
for m in tiny small medium large; do
  f=build/$m.npz
  [ -f "$f" ] || { [ "$PUBLIC" = 1 ] && f=models/standin-$m.npz; }
  [ -f "$f" ] || { echo "no $m model (build/$m.npz)"; exit 1; }
  echo "$m: $f"; $PY tools/intref.py eval "$f"
  N=$(echo $m | sed 's/^./\U&/'); ARGS="$ARGS $N=$f"
done
$PY tools/intref.py gen build/small.npz 1234 20 10 300 "the " 2>/dev/null || true
$PY tools/export_beeb.py $ARGS
./build.sh
[ -d vendor/MMB_Utils ] && tools/make_mmb.sh || echo "(vendor/MMB_Utils missing: git submodule update --init to build the .MMB)"
if [ "$PUBLIC" = 1 ] && [ -z "$QUICK" ]; then
  mkdir -p models
  for m in ${SIZES:-tiny small medium large}; do cp build/$m.npz models/standin-$m.npz; done
  cp build/beeb-llm.ssd web/jsbeeb/discs/beeb-llm.ssd && cp build/beeb-llm.ssd build/BEEB-LLM.MMB.zip discs/
  echo "stand-in models: copied to models/, discs/ and web/"
else
  [ "$PUBLIC" = 1 ] && echo "QUICK run: models/ and web/ left alone" || echo "private corpus: build/beeb-llm.ssd and build/BEEB-LLM.MMB.zip are for your own machines - not copied to models/, discs/ or web/"
fi
