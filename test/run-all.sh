#!/bin/sh
# Full verification: rebuild from the committed weights, compare every model with
# the Python reference token for token on a Model B and on a 6502 Second Processor,
# boot / menu / quit-to-BASIC checks on both, and the browser check of the page
# (served locally) for both machines.
set -e
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-python3}
./build_disc.sh
node test/verify.mjs 200
node test/boot.mjs
python3 -m http.server 8766 --directory web >/dev/null 2>&1 & SRV=$!
sleep 1; node test/browser.mjs http://localhost:8766/; kill $SRV
