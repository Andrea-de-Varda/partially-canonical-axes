#!/bin/bash
# Download the Warriner, Kuperman & Brysbaert (2013) valence/arousal/dominance
# norms from the article's supplementary material and save them as
# data/affective.csv (the exact file used for the paper; checked by sha256).
# Cite: Warriner, Kuperman & Brysbaert (2013), Behavior Research Methods 45:1191-1207.
set -euo pipefail
cd "$(dirname "$0")/.."
URL="https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0314-x/MediaObjects/13428_2012_314_MOESM1_ESM.zip"
SHA=85f6d7e35069b0eff3d9d88c4dbe08e6ae7dbdb81eac64be3f89ed0f5079d8e3
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/w.zip" "$URL"
unzip -q "$tmp/w.zip" -d "$tmp"
echo "$SHA  $tmp/BRM-emot-submit.csv" | sha256sum -c --quiet -
mv "$tmp/BRM-emot-submit.csv" data/affective.csv
echo "wrote data/affective.csv"
