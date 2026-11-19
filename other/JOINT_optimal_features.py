"""
Joint optimal rotation across ALL feature tasks.

The per-task analyses fit each feature task (Wug, BLiMP variants, Hypernymy,
Perceptual, Affective) with its OWN orthogonal rotation, which makes the
model→feature optimal score not directly comparable to the model→model and
model→brain optimal scores (one rotation each).

This script fits ONE shared orthogonal rotation per (model, layer) jointly across
all 16 feature tasks (equal weight per task), then writes per-task CSVs with the
same format as the existing per-task results — only the `optimal` column changes.
The `mean` and `sem` columns (random-rotation curves) are copied verbatim from
the existing per-task CSVs.

Outputs land in `data/results/joint/` so they don't overwrite the existing
per-task results.

Usage (from the repo root):
  python other/JOINT_optimal_features.py --modelname gpt2

Per-model output (16 CSVs):
  data/results/joint/lang_wug_{alias}.csv
  data/results/joint/lang_hypernymy_{alias}.csv
  data/results/joint/lang_<blimp_subtask>_{alias}.csv   (12 of these)
  data/results/joint/perceptual_{alias}_semantic.csv
  data/results/joint/affective_{alias}_semantic.csv
"""

import os
import sys
import argparse
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch

sys.path.append(".")
sys.path.append("./privileged_axes")
# absolute repo root too, so the import works regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooling import POOLING_TAG
from metric_config import METRIC_TAG

# Reuse loading helpers from the existing per-task scripts
from LANG_selectivity import load_task_data, get_all_layer_activations as get_lang_activations
from feature_data import load_norms, pair_split, DATA_TAG
from layer_index import LAYER_TAG
from SEM_alignment import get_all_layer_activations as get_sem_activations

from privileged_axes.get_optimal_joint import find_optimal_joint


# ---------------- Y CLEANUP (matches existing scripts) ----------------

def remove_constant_columns(Y, tol=1e-5):
    stds = Y.std(axis=0)
    keep = stds > tol
    return Y[:, keep], keep


# ---------------- existing-CSV row lookup ----------------

def _existing_csv_path(task_name, alias):
    if task_name in ("perceptual", "affective"):
        return f"data/results/{task_name}_{alias}_semantic.csv"
    if task_name == "hypernymy":
        return f"data/results/lang_hypernymy_{alias}.csv"
    # All other LANG tasks (wug, blimp_*) use lang_<task>_{alias}.csv
    return f"data/results/lang_{task_name}_{alias}.csv"


def _joint_csv_path(task_name, alias):
    if task_name in ("perceptual", "affective"):
        return f"data/results/joint/{task_name}_{alias}_semantic.csv"
    if task_name == "hypernymy":
        return f"data/results/joint/lang_hypernymy_{alias}.csv"
    return f"data/results/joint/lang_{task_name}_{alias}.csv"


def _existing_mean_sem(task_name, alias, layer):
    """
    Return (mean_str, sem_str) from the existing per-task CSV for the given
    (task, model, layer). Returns None if the row is not found — caller fills
    with NaN strings as fallback.
    """
    path = _existing_csv_path(task_name, alias)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        row = df[df["layer"] == layer]
        if row.empty:
            return None
        r = row.iloc[0]
        return str(r["mean"]), str(r["sem"])
    except Exception:
        return None


# ---------------- MAIN ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Joint optimal rotation across all feature tasks")
    parser.add_argument("--modelname", default="gpt2", type=str)
    parser.add_argument("--n_epochs", default=50, type=int)
    args = parser.parse_args()

    modelname = args.modelname
    alias = modelname.replace("/", "_")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs = args.n_epochs

    print(f"\n{'='*60}", flush=True)
    print(f"  Joint Optimal Rotation — {alias}", flush=True)
    print(f"  device={device}  n_epochs={n_epochs}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ---- LANG tasks: wug, hypernymy, blimp_* ----
    lang_tasks = load_task_data()    # list of dicts: {task, sentences, Y}
    print(f"  LANG tasks: {[t['task'] for t in lang_tasks]}", flush=True)

    all_lang_sentences = []
    sent_to_idx = {}
    for t in lang_tasks:
        for s in t["sentences"]:
            if s not in sent_to_idx:
                sent_to_idx[s] = len(all_lang_sentences)
                all_lang_sentences.append(s)
    print(f"  LANG unique sentences: {len(all_lang_sentences)}", flush=True)

    layer_acts_lang = get_lang_activations(modelname, all_lang_sentences, device=device)
    n_layers_plus1 = len(layer_acts_lang)
    d_model = layer_acts_lang[0].shape[1]
    print(f"  Layers: {n_layers_plus1}  d_model: {d_model}", flush=True)

    # ---- SEM tasks: perceptual, affective ----
    # Shared loader (feature_data.py): 13,372 unique words.
    _words, sem_sentences, Y_perceptual, Y_affective = load_norms()
    print(f"  SEM sentences: {len(sem_sentences)}  "
          f"Y_perceptual={Y_perceptual.shape}  Y_affective={Y_affective.shape}",
          flush=True)

    print(f"  [SEM] Extracting activations (span='quoted')…", flush=True)
    layer_acts_sem = get_sem_activations(
        modelname, sem_sentences, device=device,
        data_name="semantics", span="quoted", verbose=True)
    if len(layer_acts_sem) != n_layers_plus1:
        raise RuntimeError(
            f"layer count mismatch: lang={n_layers_plus1} sem={len(layer_acts_sem)}")

    # ---- Output dir ----
    out_dir = "data/results/joint"
    os.makedirs(out_dir, exist_ok=True)

    # ---- Resume support ----
    # POOLING_TAG: joint optimal rotations are fit on the LANG/SEM activations,
    # so a checkpoint from a different pooling rule is not resumable (see
    # LANG_selectivity.py for the failure this prevents).
    ckpt_path = (f"{out_dir}/joint_optimal_{alias}_{POOLING_TAG}_{METRIC_TAG}"
                 f"_{DATA_TAG}_{LAYER_TAG}_partial.pkl")
    completed = set()       # {(layer, task_name)}
    rows_pickled = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            while True:
                try:
                    batch = pickle.load(f)
                except EOFError:
                    break
                for r in batch:
                    completed.add((r["layer"], r["target"]))
                    rows_pickled.append(r)
        print(f"  Resuming: {len(completed)} (layer, task) pairs already done",
              flush=True)

    # ---- per-layer joint optimisation ----
    all_results = list(rows_pickled)

    for layer in range(n_layers_plus1):
        # Build task_data for this layer
        task_data = []

        # LANG tasks
        for t in lang_tasks:
            name = t["task"]
            if (layer, name) in completed:
                continue
            idxs = [sent_to_idx[s] for s in t["sentences"]]
            X = layer_acts_lang[layer][idxs]
            Y, _ = remove_constant_columns(t["Y"])
            # Same pair-grouped split as the per-task LANG analysis.
            task_data.append((name, X, Y, pair_split(t["pair_ids"])))

        # SEM tasks
        for sem_name, Y_raw in [("perceptual", Y_perceptual),
                                 ("affective",  Y_affective)]:
            if (layer, sem_name) in completed:
                continue
            X = layer_acts_sem[layer]
            Y, _ = remove_constant_columns(Y_raw)
            task_data.append((sem_name, X, Y))

        if not task_data:
            print(f"  [layer {layer}] all tasks already done, skipping", flush=True)
            continue

        print(f"\n  [layer {layer}] training joint rotation over "
              f"{len(task_data)} tasks…", flush=True)
        test_scores, _val_curves, best_epoch, _W = find_optimal_joint(
            task_data, num_epochs=n_epochs)

        # Save each task's row (incremental pkl + collect)
        new_rows = []
        for name, *_ in task_data:
            new_rows.append({
                "target":  name,
                "layer":   layer,
                "optimal": float(test_scores[name]),
            })
        with open(ckpt_path, "ab") as f:
            pickle.dump(new_rows, f)
        all_results.extend(new_rows)

    # ---- Write per-task CSVs (one CSV per task, joint optimal score per layer) ----
    print(f"\n  Writing per-task CSVs to {out_dir}/", flush=True)
    task_names = [t["task"] for t in lang_tasks] + ["perceptual", "affective"]
    for task_name in task_names:
        task_rows = [r for r in all_results if r["target"] == task_name]
        if not task_rows:
            print(f"    [{task_name}] no rows", flush=True)
            continue

        # Build full row with mean/sem copied from the existing per-task CSV
        out_rows = []
        for r in task_rows:
            ms = _existing_mean_sem(task_name, alias, r["layer"])
            if ms is None:
                # Fallback: NaN-filled string of shape (11,) — keeps CSV parseable
                nan_str = "[" + " ".join(["nan"] * 11) + "]"
                mean_s, sem_s = nan_str, nan_str
                print(f"    [WARN] no per-task row for {task_name} layer {r['layer']}; "
                      f"writing NaN mean/sem", flush=True)
            else:
                mean_s, sem_s = ms
            out_rows.append({
                "target":  task_name,
                "layer":   r["layer"],
                "mean":    mean_s,
                "sem":     sem_s,
                "optimal": r["optimal"],
            })

        df = pd.DataFrame(out_rows).sort_values("layer")
        out_csv = _joint_csv_path(task_name, alias)
        df.to_csv(out_csv, index=False)
        print(f"    [SAVED] {out_csv}  ({len(df)} layers)", flush=True)

    print(f"\n[DONE] Joint optimal rotation for {alias}", flush=True)


if __name__ == "__main__":
    main()
