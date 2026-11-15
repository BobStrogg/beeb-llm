#!/bin/sh
# Full verification: rebuild from the committed weights, compare the Beeb with
# the Python reference token for token, boot / quit-to-BASIC checks, and the
# browser check of the page (served locally).
set -e
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-python3}
./build_disc.sh
cp models/standin.npz build/model.npz
node test/verify.mjs 200
node test/boot.mjs
python3 -m http.server 8766 --directory web >/dev/null 2>&1 & SRV=$!
sleep 1; node test/browser.mjs http://localhost:8766/; kill $SRV
