"""
One-off migration: relabel the layers of the Pereira result CSVs of the three
Mamba models, which were computed with HF's raw Mamba hidden-state tuple, to the
canonical convention of layer_index.py used by every other result.

The relabelling is exact (only the layer labels change): HF's raw Mamba tuple holds
[block 1, ..., block L pre-norm, block L post-norm], so raw r -> canonical r+1
for r <= L-2, the pre-norm duplicate (raw L-1) is dropped, raw L -> canonical L.
Canonical layer 0 (embedding) does not exist for these files; it is excluded
from every statistic anyway.

Idempotent: a file without a layer-0 row is already canonical and is skipped.
The released CSVs are already relabelled; running it again changes nothing.

    python other/migrate_mamba_layers.py            # dry run
    python other/migrate_mamba_layers.py --write
"""
import argparse
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from layer_index import canonicalize_mamba_layers  # noqa: E402

PATTERN = os.path.join(ROOT, "data/results/pereira_voxavg_*/pereira_expt*_state-spaces_mamba-*-hf.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    files = sorted(glob.glob(PATTERN))
    if not files:
        sys.exit("no Mamba Pereira CSVs found")
    for p in files:
        df = pd.read_csv(p)
        layers = sorted(df["layer"].unique())
        if 0 not in layers:
            print(f"[skip, already canonical] {os.path.relpath(p, ROOT)}")
            continue
        n_raw = int(max(layers)) + 1
        if layers != list(range(n_raw)):
            sys.exit(f"unexpected layer set in {p}: {layers}")
        out = canonicalize_mamba_layers(df, n_raw=n_raw)
        print(f"[{'write' if args.write else 'dry'}] {os.path.relpath(p, ROOT)}: raw layers 0..{n_raw-1} "
              f"({len(df)} rows) -> canonical 1..{n_raw-1} ({len(out)} rows)")
        if args.write:
            out.to_csv(p, index=False)


if __name__ == "__main__":
    main()
