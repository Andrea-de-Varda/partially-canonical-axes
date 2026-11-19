"""
RSA-based checkpoint analysis (rotation-invariant control for model_model_checkpoints.py).

For each of ~50 log-spaced training steps, compares:
  EleutherAI/pythia-160m@stepN  vs  EleutherAI/pythia-160m-seed2@stepN

For each (step, depth-selected layer), computes a single RSA score:
  1. Compute RDM_a[i,j] = 1 - cosine_similarity(X_a[i], X_a[j]) for the n Pereira sentences
  2. Same for RDM_b
  3. RSA = Spearman correlation between the upper triangles of the two RDMs

Cosine similarity between rows IS exactly invariant under any orthogonal rotation
of the columns (||X[i]@Q|| = ||X[i]|| and (X[i]@Q)·(X[j]@Q) = X[i]·X[j] for
orthogonal Q). So this RSA score serves as a control: if the peak-then-decline
at step ~1000 also appears here, the pattern reflects representational similarity
rather than basis alignment.

Weights are deleted from the HF cache after each checkpoint to avoid accumulation.

Usage (cluster):
  python other/model_model_checkpoints_rsa.py --model_a EleutherAI/pythia-160m \\
                                              --model_b EleutherAI/pythia-160m-seed2

Output:
  data/results/model_model_checkpoints_rsa_{alias_a}_vs_{alias_b}.csv
"""

import os
import sys
import argparse
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.append(".")

from other._checkpoint_helpers import (
    select_log_spaced_checkpoints,
    depth_selected_layers,
    load_pereira_sentences,
    get_activations_no_cache,
    evict_revision,
)


# ---------------- RSA ----------------

def _cosine_rdm(X):
    """
    Cosine-similarity RDM for X of shape (n, d): RDM[i,j] = 1 - cos(X[i], X[j]).

    Cosine similarity is exactly rotation-invariant under orthogonal column rotation.
    Rows with zero norm produce NaN entries (handled by spearmanr later).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, np.nan, norms)
    Xn = X / safe_norms
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def compute_rsa(X_a, X_b):
    """Spearman correlation between upper triangles of the two cosine-RDMs."""
    rdm_a = _cosine_rdm(X_a)
    rdm_b = _cosine_rdm(X_b)
    n = rdm_a.shape[0]
    iu = np.triu_indices(n, k=1)
    a = rdm_a[iu]
    b = rdm_b[iu]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


# ---------------- MAIN ----------------

def main():
    parser = argparse.ArgumentParser(
        description="RSA-based checkpoint analysis (same step, different seed)")
    parser.add_argument("--model_a", default="EleutherAI/pythia-160m")
    parser.add_argument("--model_b", default="EleutherAI/pythia-160m-seed2")
    parser.add_argument("--n_checkpoints", type=int, default=50,
                        help="Number of log-spaced checkpoints to sample")
    args = parser.parse_args()

    alias_a = args.model_a.replace("/", "_")
    alias_b = args.model_b.replace("/", "_")
    out_csv = f"data/results/model_model_checkpoints_rsa_{alias_a}_vs_{alias_b}.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}", flush=True)
    print(f"  RSA Checkpoint Analysis", flush=True)
    print(f"  A: {args.model_a}", flush=True)
    print(f"  B: {args.model_b}", flush=True)
    print(f"  n_checkpoints={args.n_checkpoints}  device={device}", flush=True)
    print(f"{'='*60}\n", flush=True)

    os.makedirs("data/results", exist_ok=True)

    sentences = load_pereira_sentences()
    steps = select_log_spaced_checkpoints(args.n_checkpoints)
    print(f"  Selected {len(steps)} checkpoints: {steps[:5]} ... {steps[-3:]}",
          flush=True)

    # Resume support
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        done_steps = set(existing["step"].tolist())
        rows = existing.to_dict("records")
        print(f"  Resuming: {len(done_steps)} steps already done", flush=True)
    else:
        done_steps = set()
        rows = []

    n_layers = None
    layers = None
    depth_fracs = [0.25, 0.50, 0.75, 1.00]

    for step in steps:
        if step in done_steps:
            print(f"[SKIP] step {step} already in CSV", flush=True)
            continue

        revision = f"step{step}"
        print(f"\n[step={step}  revision={revision}]", flush=True)

        acts_a, n_a = get_activations_no_cache(args.model_a, sentences, revision, device)
        acts_b, n_b = get_activations_no_cache(args.model_b, sentences, revision, device)

        if n_layers is None:
            n_layers = n_a
            layers = depth_selected_layers(n_layers)
            print(f"  n_layers={n_layers}  depth layers={layers}", flush=True)

        assert n_a == n_b == n_layers, f"Unexpected layer count: {n_a} vs {n_b}"

        # Evict before alignment computation (weights are no longer needed)
        if step != 143000:
            evict_revision(args.model_a, revision)
            evict_revision(args.model_b, revision)

        for l, depth_frac in zip(layers, depth_fracs):
            X_a = acts_a[l].astype(np.float64)
            X_b = acts_b[l].astype(np.float64)
            rsa = compute_rsa(X_a, X_b)
            print(f"  [depth={depth_frac:.0%}  layer={l}]  rsa={rsa:.3f}",
                  flush=True)
            rows.append({
                "step":       step,
                "layer":      l,
                "depth_frac": depth_frac,
                "rsa":        rsa,
            })

        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"  [SAVED] {out_csv} ({len(rows)} rows)", flush=True)

        del acts_a, acts_b
        gc.collect()

    print(f"\n[DONE] {out_csv}", flush=True)


if __name__ == "__main__":
    main()
