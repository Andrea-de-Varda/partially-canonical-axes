"""
Completeness audit for the Pereira encoding result CSVs.

Every pereira_expt*.csv in each voxel-set results directory
(pereira_targets.RESULTS_DIRS: pereira_voxavg_top20rel/, pereira_voxavg_all/)
should contain one row per (layer, participant) pair: layers 0..L (block
streams have L = n_blocks + 1 entries, pre/post-act streams have n_blocks
entries) and the participants kept by pereira_targets.build_targets (original
struct indices; single-session participants are dropped, so Exp 2 has 0-4 and
6-8). Workers killed by the kernel OOM killer drop tasks
silently, so run this before trusting any downstream plot.

The released Mamba block-stream files have no layer-0 row: they were computed
with HF's raw Mamba hidden-state tuple and relabelled to the canonical layer
convention by other/migrate_mamba_layers.py (see layer_index.py), which has no
embedding entry. For those files layers 1..L-1 are expected.

The expected layer count comes from EXPECTED_LAYERS (n_blocks + 1 per model);
pre/post-act streams are expected to have one layer fewer. Inferring it from
the CSV's own max layer would be circular: a run killed part-way up the stack
writes a file whose max layer IS the truncation point.

Models absent from the table fall back to the inferred bound, which is a
lower bound only, and are marked "(inferred)". A model missing one experiment
entirely is reported too -- the per-file loop cannot see a file that is not
there.

Usage:
  python other/check_pereira_completeness.py            # audit + exit 1 if incomplete
"""

import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pereira_targets as PT  # noqa: E402

# n_blocks + 1 (the block stream emits the embedding output plus one per block).
EXPECTED_LAYERS = {
    "gpt2": 13,
    "gpt2-medium": 25,
    "gpt2-large": 37,
    "gpt2-xl": 49,
    "meta-llama_Llama-3.2-1B": 17,
    "meta-llama_Llama-3.2-3B": 29,
    "google_gemma-3-1b-pt": 27,
    "Qwen_Qwen3-0.6B-Base": 29,
    "state-spaces_mamba-130m-hf": 25,
    "state-spaces_mamba-370m-hf": 49,
    "state-spaces_mamba-790m-hf": 49,
    "EleutherAI_pythia-160m": 13,
}


def main():
    from pereira_data import load_pereira
    d = load_pereira()
    # the kept participants do not depend on the voxel set
    parts = {
        "expt2": [t["part"] for t in PT.build_targets(d["net_structs"]["PereiraE2_96pass"], "all")],
        "expt3": [t["part"] for t in PT.build_targets(d["net_structs"]["PereiraE3_72pass"], "all")],
    }
    print(f"Expected participants: {parts}\n")
    bad = 0
    for voxset in PT.VOXSETS:
        print(f"==== voxel set {voxset}: {PT.results_dir(voxset)}")
        bad += audit_dir(PT.results_dir(voxset), parts)
        print()
    if bad:
        print(f"\n{bad} file(s) incomplete or duplicated — resubmit the affected "
              f"array indices (checkpoints resume exactly the missing tasks).")
        sys.exit(1)
    print("\nAll Pereira result CSVs complete.")


def audit_dir(RESULTS_DIR, parts):
    files = sorted(glob.glob(f"{RESULTS_DIR}/pereira_expt*.csv"))
    if not files:
        print("  (no result CSVs)")
        return 1
    parsed = []  # (path, exp, stream, alias, df)
    for path in files:
        m = re.match(r"pereira_(expt[23])_(pre_act_|post_act_)?(.+)\.csv",
                     os.path.basename(path))
        if not m:
            continue
        exp, stream, alias = m.group(1), (m.group(2) or "block").rstrip("_"), m.group(3)
        parsed.append((path, exp, stream, alias, pd.read_csv(path)))

    # expected block-stream layer count per alias: the table where we have it,
    # otherwise the (circular, lower-bound) inference from the data itself.
    block_layers = {}
    inferred = set()
    for _, exp, stream, alias, df in parsed:
        if stream == "block":
            if alias in EXPECTED_LAYERS:
                block_layers[alias] = EXPECTED_LAYERS[alias]
            else:
                block_layers[alias] = max(block_layers.get(alias, 0), df.layer.max() + 1)
                inferred.add(alias)

    bad = 0

    # A model missing a whole experiment produces no file, so the loop below
    # would never mention it.
    present = {(exp, alias) for _, exp, stream, alias, _ in parsed if stream == "block"}
    for alias in sorted({a for _, a in present}):
        for exp in ("expt2", "expt3"):
            if (exp, alias) not in present:
                print(f"pereira_{exp}_{alias}.csv  <-- MISSING ENTIRELY")
                bad += 1
    for path, exp, stream, alias, df in parsed:
        plist = parts[exp]
        if alias in block_layers:
            L = block_layers[alias] if stream == "block" else block_layers[alias] - 1
        else:
            L = df.layer.max() + 1  # fallback: lower bound only
        # relabelled Mamba files (migrate_mamba_layers.py) start at layer 1
        first = 1 if (alias.startswith("state-spaces_mamba") and stream == "block"
                      and not (df.layer == 0).any()) else 0
        missing = [(l, p) for l in range(first, L) for p in plist
                   if not ((df.layer == l) & (df.part == p)).any()]
        extra = sorted(set(df.part.astype(int)) - set(plist))
        dup = len(df) - len(df.drop_duplicates(["layer", "part"]))
        status = ""
        if missing:
            status = f"  <-- INCOMPLETE, first missing: {missing[:6]}"
            bad += 1
        elif extra:
            status = f"  <-- UNEXPECTED participants {extra}"
            bad += 1
        elif dup:
            status = f"  <-- {dup} DUPLICATE (layer, part) rows"
            bad += 1
        tag = "  (inferred)" if alias in inferred and stream == "block" else ""
        print(f"{os.path.basename(path):58s} rows={len(df):4d}/{(L - first) * len(plist):4d}{status}{tag}")
    return bad


if __name__ == "__main__":
    main()
