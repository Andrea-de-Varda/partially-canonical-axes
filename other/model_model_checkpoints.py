"""
Checkpoint alignment analysis: does the privileged basis emerge during training?

For each of ~50 log-spaced training steps, compares:
  EleutherAI/pythia-160m@stepN  vs  EleutherAI/pythia-160m-seed2@stepN

(same step, different random seed — symmetric comparison)

For each step, across 4 depth-selected layers (25/50/75/100%), computes:
  1. native  — alignment at α=0 (no rotation)     both directions averaged
  2. random  — alignment at α=1 (n_seeds random)  both directions averaged
  3. optimal — get_optimal_alignment               both directions averaged

Weights are deleted from the HF cache after each checkpoint to avoid accumulation.

Usage (run from project root on cluster):
  python other/model_model_checkpoints.py

Output:
  data/results/model_model_checkpoints_{alias_a}_vs_{alias_b}.csv
"""

import os
import sys
import argparse
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch

sys.path.append(".")
sys.path.append("./privileged_axes")


from privileged_axes.get_alignment_pairwise import get_alignment_axis_rotation_multiseed
from privileged_axes.get_optimal_pairwise import get_optimal_alignment

from other._checkpoint_helpers import (
    select_log_spaced_checkpoints,
    depth_selected_layers,
    load_pereira_sentences,
    get_activations_no_cache,
    evict_revision,
)


# ----------------------------- ALIGNMENT HELPERS -------------------------

def _compute_layer_metrics(X_a, X_b, n_seeds, n_epochs):
    """
    Compute native, random (α=1), and optimal alignment for one layer pair.
    All metrics averaged over both directions (A→B and B→A).
    """
    # --- random rotation: both directions, n_seeds ---
    _, _, all_AtoB = get_alignment_axis_rotation_multiseed(
        X_a, X_b, n_seeds=n_seeds, return_all=True)
    _, _, all_BtoA = get_alignment_axis_rotation_multiseed(
        X_b, X_a, n_seeds=n_seeds, return_all=True)

    # (n_seeds, 11, n_target_dims) → (n_seeds, 11) by averaging over target dims
    per_seed_AtoB = np.nanmean(all_AtoB, axis=2)
    per_seed_BtoA = np.nanmean(all_BtoA, axis=2)
    avg_both = (per_seed_AtoB + per_seed_BtoA) / 2   # (n_seeds, 11)

    native = float(avg_both[:, 0].mean())   # α=0 index
    random = float(avg_both[:, -1].mean())  # α=1 index

    # --- optimal alignment: both directions ---
    opt_AtoB = float(get_optimal_alignment(X_a, X_b, num_epochs=n_epochs))
    opt_BtoA = float(get_optimal_alignment(X_b, X_a, num_epochs=n_epochs))
    optimal  = (opt_AtoB + opt_BtoA) / 2

    return native, random, optimal


# ----------------------------- MAIN ---------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint-to-checkpoint basis alignment (same step, different seed)")
    parser.add_argument("--model_a", default="EleutherAI/pythia-160m")
    parser.add_argument("--model_b", default="EleutherAI/pythia-160m-seed2")
    parser.add_argument("--n_checkpoints", type=int, default=50,
                        help="Number of log-spaced checkpoints to sample")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of random rotation seeds")
    parser.add_argument("--n_epochs", type=int, default=50,
                        help="Epochs for optimal alignment")
    args = parser.parse_args()

    alias_a = args.model_a.replace("/", "_")
    alias_b = args.model_b.replace("/", "_")
    out_csv = f"data/results/model_model_checkpoints_{alias_a}_vs_{alias_b}.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}", flush=True)
    print(f"  Checkpoint Basis Alignment", flush=True)
    print(f"  A: {args.model_a}", flush=True)
    print(f"  B: {args.model_b}", flush=True)
    print(f"  n_checkpoints={args.n_checkpoints}  n_seeds={args.n_seeds}"
          f"  n_epochs={args.n_epochs}  device={device}", flush=True)
    print(f"{'='*60}\n", flush=True)

    os.makedirs("data/results", exist_ok=True)

    # ---- Load Pereira sentences ----
    sentences = load_pereira_sentences()

    # ---- Select checkpoint steps ----
    steps = select_log_spaced_checkpoints(args.n_checkpoints)
    print(f"  Selected {len(steps)} checkpoints: {steps[:5]} ... {steps[-3:]}", flush=True)

    # ---- Load existing results to allow resuming ----
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        done_steps = set(existing["step"].tolist())
        rows = existing.to_dict("records")
        print(f"  Resuming: {len(done_steps)} steps already done", flush=True)
    else:
        done_steps = set()
        rows = []

    # ---- n_layers: load first checkpoint to determine architecture ----
    # (Both models have the same architecture, so we only need to check once)
    n_layers = None

    for step in steps:
        if step in done_steps:
            print(f"[SKIP] step {step} already in CSV", flush=True)
            continue

        revision = f"step{step}"
        print(f"\n[step={step}  revision={revision}]", flush=True)

        # Load activations for both models at this step
        acts_a, n_a = get_activations_no_cache(args.model_a, sentences, revision, device)
        acts_b, n_b = get_activations_no_cache(args.model_b, sentences, revision, device)

        if n_layers is None:
            n_layers = n_a
            layers = depth_selected_layers(n_layers)
            depth_fracs = [0.25, 0.50, 0.75, 1.00]
            print(f"  n_layers={n_layers}  depth layers={layers}", flush=True)

        assert n_a == n_b == n_layers, f"Unexpected layer count: {n_a} vs {n_b}"

        # Evict checkpoint weights (skip step 143000 = main branch, keep in cache)
        if step != 143000:
            evict_revision(args.model_a, revision)
            evict_revision(args.model_b, revision)

        # Compute metrics for each depth layer
        for l, depth_frac in zip(layers, depth_fracs):
            X_a = acts_a[l].astype(np.float32)
            X_b = acts_b[l].astype(np.float32)

            print(f"  [depth={depth_frac:.0%}  layer={l}]", flush=True)
            native, random, optimal = _compute_layer_metrics(
                X_a, X_b, args.n_seeds, args.n_epochs)
            print(f"    native={native:.3f}  random={random:.3f}  optimal={optimal:.3f}",
                  flush=True)

            rows.append({
                "step":       step,
                "layer_a":    l,
                "layer_b":    l,
                "depth_frac": depth_frac,
                "native":     native,
                "random":     random,
                "optimal":    optimal,
            })

        # Save incrementally after each step
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"  [SAVED] {out_csv} ({len(rows)} rows)", flush=True)

        # Free activation arrays
        del acts_a, acts_b
        gc.collect()

    print(f"\n[DONE] {out_csv}", flush=True)


if __name__ == "__main__":
    main()
