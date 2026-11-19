"""
Model-to-model basis alignment.

Measures whether two models share the same representational basis using combined
Pereira Exp2 + Exp3 sentences as stimuli. For each of 4 depth-selected layers
(25%, 50%, 75%, 100%), computes:
  1. Random rotation alignment  (both directions, averaged) across n_seeds × 11 alphas
  2. Optimal alignment          (both directions, averaged) via learned orthogonal rotation

Both directions averaged: score = (A→B + B→A) / 2, giving a symmetric measure.

Usage (run from project root):
  python other/model_model_alignment.py --model_b gpt2
  python other/model_model_alignment.py --model_b EleutherAI/pythia-160m-seed2
  python other/model_model_alignment.py --model_b state-spaces/mamba-130m-hf

Output:
  data/results/model_model_{alias_A}_vs_{alias_B}.csv
"""

import os, sys, argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch

sys.path.append(".")
sys.path.append("./privileged_axes")


from privileged_axes.get_alignment_pairwise import get_alignment_axis_rotation_multiseed
from privileged_axes.get_optimal_pairwise import get_optimal_alignment
from PEREIRA_alignment_gpu import get_all_layer_activations
from pereira_data import load_pereira


# ----------------------------- UTILS -----------------------------

def load_pereira_sentences():
    """Load and combine Pereira Exp2 + Exp3 sentences (deduplicated, order preserved)."""
    data = load_pereira()   # data/pereira_language.npz (see pereira_data.py)
    net_sent = data["net_sent_orders"]

    # Remove boundary tokens — same logic as PEREIRA_alignment_gpu.py
    ex2 = [s[1:-1] for s in net_sent["PereiraE2_96pass"]]
    ex3 = [s[1:-1] for s in net_sent["PereiraE3_72pass"]]

    seen = set()
    combined = []
    for s in ex2 + ex3:
        if s not in seen:
            seen.add(s)
            combined.append(s)

    print(f"  Pereira sentences: Exp2={len(ex2)}, Exp3={len(ex3)}, combined unique={len(combined)}",
          flush=True)
    return combined


def depth_selected_layers(n_layers):
    """Return 4 layer indices at 25%, 50%, 75%, 100% depth.
    n_layers includes embedding layer (index 0), so max index = n_layers - 1.
    """
    max_idx = n_layers - 1
    return [round(frac * max_idx) for frac in (0.25, 0.50, 0.75, 1.00)]


def matched_layer_b(l_a, n_layers_a, n_layers_b):
    """Proportionally map a layer index from model A to model B."""
    frac = l_a / max(n_layers_a - 1, 1)
    return round(frac * (n_layers_b - 1))


# ----------------------------- ALIGNMENT -------------------------

def _per_seed_means(all_scores):
    """all_scores: (n_seeds, n_rotations=11, n_target_dims) → (n_seeds, 11)."""
    return np.nanmean(all_scores, axis=2)


def compute_random_both_directions(X_a, X_b, n_seeds):
    """
    Random rotation alignment in both directions, averaged.

    Returns
    -------
    mean_arr : (11,) — average alignment for alpha = 0, 0.1, …, 1.0
    sem_arr  : (11,) — SEM across seeds
    """
    _, _, all_AtoB = get_alignment_axis_rotation_multiseed(
        X_a, X_b, n_seeds=n_seeds, return_all=True)
    _, _, all_BtoA = get_alignment_axis_rotation_multiseed(
        X_b, X_a, n_seeds=n_seeds, return_all=True)

    # Average over target dims for each seed
    per_seed_AtoB = _per_seed_means(all_AtoB)   # (n_seeds, 11)
    per_seed_BtoA = _per_seed_means(all_BtoA)   # (n_seeds, 11)
    per_seed_avg  = (per_seed_AtoB + per_seed_BtoA) / 2   # (n_seeds, 11)

    mean_arr = per_seed_avg.mean(axis=0)
    sem_arr  = per_seed_avg.std(axis=0, ddof=1) / np.sqrt(n_seeds)
    return mean_arr, sem_arr


def compute_optimal_both_directions(X_a, X_b, n_epochs):
    """
    Optimal alignment in both directions, averaged.
    Returns a scalar.
    """
    opt_AtoB = get_optimal_alignment(X_a, X_b, num_epochs=n_epochs)
    opt_BtoA = get_optimal_alignment(X_b, X_a, num_epochs=n_epochs)
    return float((opt_AtoB + opt_BtoA) / 2)


# ----------------------------- MAIN ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute model-to-model basis alignment")
    parser.add_argument("--model_a", default="EleutherAI/pythia-160m",
                        help="Reference model (default: Pythia-160m)")
    parser.add_argument("--model_b", required=True,
                        help="Comparison model")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of random rotation seeds")
    parser.add_argument("--n_epochs", type=int, default=50,
                        help="Epochs for optimal alignment training (50, as in every other analysis)")
    args = parser.parse_args()

    alias_a = args.model_a.replace("/", "_")
    alias_b = args.model_b.replace("/", "_")
    out_csv = f"data/results/model_model_{alias_a}_vs_{alias_b}.csv"

    if os.path.exists(out_csv):
        print(f"[SKIP] {out_csv} already exists.", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}", flush=True)
    print(f"  Model-to-Model Alignment", flush=True)
    print(f"  A: {args.model_a}", flush=True)
    print(f"  B: {args.model_b}", flush=True)
    print(f"  n_seeds={args.n_seeds}  n_epochs={args.n_epochs}  device={device}", flush=True)
    print(f"{'='*60}\n", flush=True)

    if device == "cpu":
        print("[WARNING] No GPU detected — optimal alignment will be slow or fail.", flush=True)

    os.makedirs("data/results", exist_ok=True)

    # ---- Sentences ----
    sentences = load_pereira_sentences()

    # ---- Activations ----
    print(f"Loading activations for {args.model_a}...", flush=True)
    repr_a, _ = get_all_layer_activations(
        args.model_a, sentences, device=device,
        data_name="pereira_combined", residual_only=True)

    print(f"Loading activations for {args.model_b}...", flush=True)
    repr_b, _ = get_all_layer_activations(
        args.model_b, sentences, device=device,
        data_name="pereira_combined", residual_only=True)

    acts_a = repr_a["block_outputs"]   # {layer_idx: (n_sent, d_a)}
    acts_b = repr_b["block_outputs"]   # {layer_idx: (n_sent, d_b)}

    n_layers_a = len(acts_a)
    n_layers_b = len(acts_b)
    print(f"  Model A: {n_layers_a} layers | Model B: {n_layers_b} layers", flush=True)

    # ---- Depth-selected layer pairs ----
    layers_a    = depth_selected_layers(n_layers_a)
    depth_fracs = [0.25, 0.50, 0.75, 1.00]

    rows = []
    for l_a, depth_frac in zip(layers_a, depth_fracs):
        l_b = matched_layer_b(l_a, n_layers_a, n_layers_b)
        print(f"\n[depth={depth_frac:.0%}  layer_a={l_a}  layer_b={l_b}]", flush=True)

        X_a = acts_a[l_a].astype(np.float32)
        X_b = acts_b[l_b].astype(np.float32)

        # Random rotation (both directions)
        print("  Random rotation...", flush=True)
        mean_arr, sem_arr = compute_random_both_directions(X_a, X_b, args.n_seeds)

        # Optimal alignment (both directions)
        print("  Optimal alignment...", flush=True)
        optimal = compute_optimal_both_directions(X_a, X_b, args.n_epochs)

        print(
            f"  native={mean_arr[0]:.3f}  random={mean_arr[-1]:.3f}  optimal={optimal:.3f}",
            flush=True,
        )

        rows.append({
            "layer_a":    l_a,
            "layer_b":    l_b,
            "depth_frac": depth_frac,
            "mean":       mean_arr,
            "sem":        sem_arr,
            "optimal":    optimal,
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[SAVED] {out_csv}", flush=True)


if __name__ == "__main__":
    main()
