#!/bin/bash
# Download the fMRI responses used in the paper (Pereira et al., 2018, Exp. 2 and 3;
# per-session language-network responses, see README "Brain data") from the
# GitHub release of this repository, and check the sha256.
set -euo pipefail
cd "$(dirname "$0")/.."
URL="https://github.com/Andrea-de-Varda/partially-canonical-axes/releases/download/v1.0/pereira_language.npz"
SHA=d7a2f3649d59544effde6fc334945b4bd906b9a0e8ce853eca34185033d1a7ae
curl -fL -o data/pereira_language.npz.part "$URL"
echo "$SHA  data/pereira_language.npz.part" | sha256sum -c --quiet -
mv data/pereira_language.npz.part data/pereira_language.npz
echo "wrote data/pereira_language.npz"
