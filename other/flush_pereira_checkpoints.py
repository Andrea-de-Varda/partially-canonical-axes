"""
Rebuild Pereira result CSVs from their *_passagecv4_*partial.pkl checkpoints,
in every voxel-set results directory (pereira_targets.RESULTS_DIRS).

WHY THIS EXISTS
---------------
run_parallel_alignment() appends every finished (layer, participant) result to
a checkpoint pickle immediately, but the stage CSV is only rewritten
periodically. A task killed hard (OOM, walltime without SIGTERM) can therefore
leave results in the pickle that are not yet in the CSV.

This script converts checkpoints -> CSVs with NO recomputation. It needs no GPU
and no model, so it can run on any machine.

IT ONLY READS CHECKPOINTS WRITTEN UNDER THE CURRENT POOLING + METRIC TAGS.
Checkpoints from an older contract map to the SAME CSV stem (the stem is
everything before "_passagecv4_"), so a bare glob would rebuild a CSV from
whichever file happened to sort last. Older checkpoints are listed and skipped.

Usage:
  python other/flush_pereira_checkpoints.py            # report + write
  python other/flush_pereira_checkpoints.py --dry-run  # report only
"""
import argparse
import glob
import os
import pickle

import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pooling import POOLING_TAG          # noqa: E402
from metric_config import METRIC_TAG     # noqa: E402
import pereira_targets as PT             # noqa: E402

CONTRACT = f"_{PT.CKPT_VERSION}_{POOLING_TAG}_{METRIC_TAG}_partial.pkl"


def load_checkpoint(path):
    """Read every record; tolerate a tail truncated by a mid-write kill."""
    records, truncated = [], False
    with open(path, "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
            except EOFError:
                break
            except Exception:
                truncated = True   # partial final record — keep what we have
                break
            records.extend(batch)
    return records, truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None,
                    help="one directory only (default: every voxel set in pereira_targets.RESULTS_DIRS)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dirs = [args.results_dir] if args.results_dir else [PT.RESULTS_DIRS[v] for v in PT.VOXSETS]
    total_new = 0
    for d in dirs:
        print(f"==== {d}")
        total_new += flush_dir(d, args.dry_run)
    print(f"\n{'Would recover' if args.dry_run else 'Recovered'} {total_new} "
          f"result rows from checkpoints.")
    if not args.dry_run and total_new:
        print("Now re-run: python other/check_pereira_completeness.py")


def flush_dir(results_dir, dry_run):
    all_pkls = sorted(glob.glob(f"{results_dir}/*_{PT.CKPT_VERSION}_*partial.pkl"))
    pkls = [p for p in all_pkls if os.path.basename(p).endswith(CONTRACT)]
    stale = [p for p in all_pkls if p not in pkls]
    if stale:
        print(f"Ignoring {len(stale)} checkpoint(s) from an older contract "
              f"(current: *{CONTRACT}):")
        for p in stale:
            print(f"  [SKIP] {os.path.basename(p)}")
        print()
    if not pkls:
        print(f"No checkpoints matching *{CONTRACT} in {results_dir}/")
        return 0

    total_new = 0
    for pkl in pkls:
        stem = os.path.basename(pkl)[:-len(CONTRACT)]
        csv = f"{results_dir}/{stem}.csv"
        records, truncated = load_checkpoint(pkl)
        if not records:
            print(f"[EMPTY]  {stem}")
            continue

        # Last write wins per (layer, part) — a retried task supersedes the old one.
        dedup = {(r["layer"], r["part"]): r for r in records}
        df = pd.DataFrame(list(dedup.values())).sort_values(["layer", "part"])

        n_before = len(pd.read_csv(csv)) if os.path.exists(csv) else 0
        gained = len(df) - n_before
        tag = "TRUNCATED-TAIL " if truncated else ""

        if gained > 0:
            total_new += gained
            action = "would write" if dry_run else "WROTE"
            if not dry_run:
                df.to_csv(csv, index=False)
            print(f"[{action}] {tag}{stem}: {n_before} -> {len(df)} rows (+{gained})")
        else:
            print(f"[ok]     {tag}{stem}: {len(df)} rows, CSV already current")
    return total_new


if __name__ == "__main__":
    main()
