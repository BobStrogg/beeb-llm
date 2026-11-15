#!/bin/sh
# Download the public-domain stand-in corpus (Jerome K. Jerome, Project Gutenberg):
#   Three Men in a Boat (#308) and Three Men on the Bummel (#2183)
set -e
cd "$(dirname "$0")/../corpus"
get() { [ -s "$2" ] || curl -sfL -A "Mozilla/5.0" -o "$2" "$1"; echo "$2: $(wc -c < "$2") bytes"; }
get https://gutenberg.pglaf.org/3/0/308/308-0.txt standin-threemen.txt
get https://gutenberg.pglaf.org/2/1/8/2183/2183-0.txt standin-bummel.txt
