"""
Sparsity analyses for privileged axes in LLMs.

Analysis 1 — Lifetime Sparseness (Khosla et al. Fig 8A):
    Excess kurtosis per dimension (column-wise), averaged across dimensions.
    Computed on all linguistic stimuli (SEM words + LANG sentences).
    Compared: native basis vs. random rotations.
    Units are z-scored BEFORE rotation: the native value is invariant to this,
    but rotating raw activations lets a few high-variance units dominate every
    rotated unit.

Analysis 2 — Readout Weight Sparsity (Khosla et al. Fig 8B):
    Ridge model trained to predict each linguistic target.
    Kurtosis of learned weight vector: how sparse is the readout?
    Units are z-scored on the ridge training rows BEFORE fitting. Fitting on
    raw activations would make the weight of each unit scale with its variance,
    so the very unequal unit variances of LLMs (outlier dimensions) would produce
    heavy-tailed native weights even for a readout with no
    preferred units, while a random rotation equalises variances -- a native >
    random kurtosis gap without any sparse readout. Because the z-scoring is
    applied before the rotation, the rotated readout is still exactly W @ Q.
    Targets: SEM (perceptual + affective norms, Ridge regression),
             LANG (BLiMP / wug / hypernymy, RidgeClassifier).

Run from project root:
    python other/sparsity_analysis.py --modelname gpt2 --n_seeds 10
"""

import os, sys, argparse, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

sys.path.append(".")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from privileged_axes.rotation_code import randomRotationPowers
from pooling import POOLING_TAG
from feature_data import load_lang_tasks as _load_lang_tasks, load_norms, pair_split
from layer_index import canonical_hidden_states
from LANG_selectivity import lang_cache_path


# ----------------------------- UTILS -----------------------------

def _get_n_layers(cfg):
    return (getattr(cfg, "num_hidden_layers", None)
            or getattr(cfg, "n_layer", None)
            or getattr(cfg, "n_layers", None))


def mean_kurtosis_torch(X):
    """Excess kurtosis per dimension, averaged. X: torch tensor (n, d) on any device.
    Computed in float32; avoids CPU bottleneck for large matrices.
    """
    mu = X.mean(0)
    d = X - mu
    m2 = (d ** 2).mean(0)
    m4 = (d ** 4).mean(0)
    k = m4 / (m2 ** 2 + 1e-10) - 3.0
    return float(k.nanmean().item())


def weight_kurtosis_single(w):
    """Kurtosis of a 1-D weight vector (numpy)."""
    return float(scipy_kurtosis(w, fisher=True))


def weight_kurtosis_multioutput(W):
    """Mean kurtosis across rows of a (n_targets, d) weight matrix (numpy)."""
    return float(np.nanmean([scipy_kurtosis(row, fisher=True) for row in W]))


def load_rot_mats(dims, seed, alpha_steps):
    """Load or generate rotation matrices for a given seed.
    Returns ndarray of shape (n_alphas+1, dims, dims):
      index 0 = identity, index 1..n_alphas = alpha 0.1..1.0.
    """
    os.makedirs("rotations", exist_ok=True)
    rot_path = f"rotations/rot_mats{dims}_seed{seed}.npy"
    if os.path.exists(rot_path):
        mats = np.load(rot_path, allow_pickle=True)
        if mats.dtype == object:  # tolerate a file saved as a list of matrices
            mats = np.stack(list(mats))
        return mats
    rot_mats = np.stack(
        [np.identity(dims)]
        + list(randomRotationPowers(dims, alpha_steps, seed=seed, unpermute=True))
    )
    np.save(rot_path, rot_mats)
    return rot_mats


# ----------------------------- DATA LOADING ----------------------

def load_sem_data(data_dir="data"):
    """SEM sentences and targets from feature_data.load_norms.
    Returns (sentences, Y_sem) where Y_sem is (n_words, 9): 3 affective + 6 perceptual.
    """
    _words, sentences, Y_per, Y_aff = load_norms(data_dir)
    return sentences, np.concatenate([Y_aff, Y_per], axis=1).astype(float)


def load_lang_tasks(data_dir="data/language_data"):
    """LANG tasks from feature_data.load_lang_tasks (deduplicated).
    Returns (tasks, all_sentences, sentence_to_idx); each task dict has
    task, sentences, Y (1-D), pair_ids and idxs (into all_sentences).
    The global sentence order matches LANG_selectivity.py, whose activation
    cache is read here.
    """
    tasks = _load_lang_tasks(data_dir)
    for t in tasks:
        t["Y"] = t["Y"].ravel()
    all_sentences, sentence_to_idx = [], {}
    for task in tasks:
        for s in task["sentences"]:
            if s not in sentence_to_idx:
                sentence_to_idx[s] = len(all_sentences)
                all_sentences.append(s)
    for task in tasks:
        task["idxs"] = [sentence_to_idx[s] for s in task["sentences"]]
    return tasks, all_sentences, sentence_to_idx


# ----------------------------- ACTIVATIONS -----------------------

def compute_sem_activations(model_name, sentences, device):
    """Extract quoted-span pooled activations for SEM sentences (no caching).
    Returns {layer_idx: np.ndarray (n_sentences, d_model)}.
    """
    print(f"  [SEM] Computing activations ({len(sentences)} sentences, not saved)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if "gemma" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    else:
        model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    n_layers = _get_n_layers(model.config)
    layer_outputs = {i: [] for i in range(n_layers + 1)}

    for s in tqdm(sentences, desc="  SEM forward pass"):
        # Pool only tokens inside the quoted word: "The meaning of the word 'X'"
        q1 = s.index("'") + 1
        q2 = s.index("'", q1)

        tok = tokenizer(s, return_tensors="pt", truncation=True, return_offsets_mapping=True)
        offsets = tok.pop("offset_mapping")[0]
        tok = {k: v.to(device) for k, v in tok.items()}

        with torch.no_grad():
            outs = model(**tok, output_hidden_states=True)

        idx = [i for i, (st, en) in enumerate(offsets.tolist())
               if st >= q1 and en <= q2 and (st, en) != (0, 0)]
        if not idx:  # fallback: full sentence
            idx = [i for i, (st, en) in enumerate(offsets.tolist()) if (st, en) != (0, 0)]

        hs = canonical_hidden_states(model, outs, tok["input_ids"])
        for i, layer in enumerate(hs):
            vec = layer.squeeze(0)[idx].mean(dim=0).cpu().numpy()
            layer_outputs[i].append(vec)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    for i in layer_outputs:
        layer_outputs[i] = np.stack(layer_outputs[i])
    return layer_outputs


def load_lang_activations(alias, cache_dir="cached_activations"):
    """Load LANG activation cache (sentence-level pooled).
    Returns {layer: ndarray(n, d)} or None if cache missing.
    """
    fname = lang_cache_path(alias, cache_dir)   # same cache LANG_selectivity.py writes
    if os.path.exists(fname):
        print(f"  [LANG] Loading cache: {fname}", flush=True)
        return np.load(fname, allow_pickle=True).item()
    print(f"  [LANG] Cache not found: {fname}. Run LANG_selectivity.py first.", flush=True)
    return None


# ----------------------------- RIDGE UTILS -----------------------

def fit_ridge_weights(X, Y, is_clf, test_size=0.2, alpha=1.0, random_state=42, groups=None):
    """Fit Ridge once on native X. Returns weight matrix W of shape (n_targets, d).

    X is z-scored per unit with the TRAINING rows' mean/sd before fitting (see
    module docstring). groups (e.g. minimal-pair ids) keeps both
    members of a pair on the same side of the split.

    Key insight: for rotated data X_rot = X @ Q the Ridge solution satisfies
    w_rot = w_native @ Q  (rotation counter-rotates the weights, no refit needed).
    This reduces 45k Ridge fits to ~480 by applying Q to weights analytically.
    """
    if groups is not None:
        tr_idx, _, _ = pair_split(groups, train_fract=1 - test_size, dev_fract=0.0,
                                  test_fract=test_size, seed=random_state)
        X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    else:
        X_tr, _, Y_tr, _ = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd == 0] = 1.0
    X_tr = (X_tr - mu) / sd
    if is_clf:
        clf = RidgeClassifier(alpha=alpha)
        clf.fit(X_tr, Y_tr)
        W = clf.coef_
        if W.ndim == 1:
            W = W[None, :]  # binary clf returns (d,) — reshape to (1, d)
        return W
    else:
        reg = Ridge(alpha=alpha)
        reg.fit(X_tr, Y_tr)
        W = reg.coef_
        if W.ndim == 1:
            W = W[None, :]  # ensure (1, d)
        return W  # (n_targets, d)


def kurtosis_of_weights(W):
    """Mean kurtosis across rows of W (n_targets, d) or (1, d)."""
    if W.shape[0] == 1:
        return weight_kurtosis_single(W.ravel())
    return weight_kurtosis_multioutput(W)


def rotate_weights(W, Q):
    """Apply rotation Q to weight matrix W: w_rot = w_native @ Q.
    W: (n_targets, d), Q: (d, d) — no Ridge refit needed.
    """
    return W @ Q


# ----------------------------- MAIN ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sparsity analyses for LLM representations")
    parser.add_argument("--modelname", default="gpt2", type=str)
    parser.add_argument("--n_seeds", default=10, type=int)
    args = parser.parse_args()

    modelname = args.modelname
    alias = modelname.replace("/", "_")
    n_seeds = args.n_seeds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha_steps = np.arange(0.1, 1.0 + 1e-9, 0.1)

    print(f"\n{'='*60}", flush=True)
    print(f"  Sparsity Analyses — {alias}", flush=True)
    print(f"  Seeds: {n_seeds}", flush=True)
    print(f"{'='*60}\n", flush=True)

    os.makedirs("data/results", exist_ok=True)

    # ---- Load data ----
    sem_sentences, Y_sem = load_sem_data()
    lang_tasks, all_lang_sentences, _ = load_lang_tasks()
    print(f"  SEM: {len(sem_sentences)} words, {Y_sem.shape[1]} targets", flush=True)
    print(f"  LANG: {len(lang_tasks)} tasks, {len(all_lang_sentences)} unique sentences", flush=True)

    # ---- Activations ----
    sem_acts = compute_sem_activations(modelname, sem_sentences, device)
    lang_acts = load_lang_activations(alias)

    n_layers = len(sem_acts)
    d_model = sem_acts[0].shape[1]
    dims = (d_model // 2) * 2

    print(f"\n  Layers: {n_layers} (0..{n_layers-1}), d_model: {d_model}, dims: {dims}\n", flush=True)

    # ---- Preload rotation matrices as GPU tensors ----
    # Keeping them on GPU avoids repeated CPU→GPU transfers in the inner loop.
    # Memory: n_seeds × n_alphas × dims² × 4 bytes = 10 × 10 × 2048² × 4 ≈ 1.6 GB for 3B model.
    print(f"  Preloading rotation matrices ({n_seeds} seeds × {len(alpha_steps)} alphas)...", flush=True)
    rot_cache_np = {}   # numpy (n_alphas+1, dims, dims) — needed for weight rotation in Analysis 2
    rot_cache_gpu = {}  # torch tensors on device, shape (n_alphas, dims, dims), alpha 0.1..1.0 only
    for seed in range(n_seeds):
        mats = load_rot_mats(dims, seed, alpha_steps)  # (n_alphas+1, dims, dims)
        rot_cache_np[seed] = mats
        rot_cache_gpu[seed] = torch.tensor(
            mats[1:], dtype=torch.float32, device=device)  # skip identity at index 0
    print(f"  Done.\n", flush=True)

    # ====================================================================
    # ANALYSIS 1: Lifetime Kurtosis — GPU matmuls
    # ====================================================================
    print(f"{'='*60}", flush=True)
    print(f"  Analysis 1: Lifetime Sparseness  [GPU rotations]", flush=True)
    print(f"{'='*60}", flush=True)

    kurtosis_csv = f"data/results/sparsity_kurtosis_{alias}.csv"
    if os.path.exists(kurtosis_csv):
        print(f"  [SKIP] {kurtosis_csv} already exists.\n", flush=True)
    else:
        kurtosis_rows = []

        for layer in range(n_layers):
            # Pool SEM + LANG activations, send to GPU as float32
            arrs = [sem_acts[layer]]
            if lang_acts is not None and layer in lang_acts:
                arrs.append(lang_acts[layer])
            X_np = np.concatenate(arrs, axis=0)[:, :dims]
            X_gpu = torch.tensor(X_np, dtype=torch.float32, device=device)

            # Standardise every unit BEFORE rotating. Kurtosis is invariant to a
            # per-unit affine rescaling, so the native value is unchanged; but a
            # rotated unit is a mixture of native units weighted by their
            # standard deviations, so rotating RAW activations makes the rotated
            # average reflect the few highest-variance native directions rather
            # than all units equally. In Pythia-160M (max/median unit variance
            # ~680 at layer 6) that inflated rotated/native to 6.45; with
            # standardisation it is 0.90. GPT-2, whose unit variances are far
            # more homogeneous, is unaffected. This matches the alignment metric,
            # which z-scores units on the training split before any rotation.
            X_gpu = (X_gpu - X_gpu.mean(0)) / (X_gpu.std(0) + 1e-8)

            # Native (alpha=0) — cheap, compute on GPU too
            k_native = mean_kurtosis_torch(X_gpu)
            kurtosis_rows.append({
                "layer": layer, "path": "native", "alpha": 0.0, "seed": -1,
                "kurtosis": k_native,
            })

            # Random rotations — all matmuls on GPU
            kurt_random = []
            for seed in range(n_seeds):
                Q_batch = rot_cache_gpu[seed]  # (n_alphas, dims, dims) on device
                for ai, alpha_val in enumerate(alpha_steps):
                    X_rot = X_gpu @ Q_batch[ai]  # (n, dims) @ (dims, dims) on GPU
                    k = mean_kurtosis_torch(X_rot)
                    kurtosis_rows.append({
                        "layer": layer, "path": "random",
                        "alpha": round(float(alpha_val), 1),
                        "seed": seed, "kurtosis": k,
                    })
                    kurt_random.append(k)

            print(
                f"  Layer {layer:2d} | kurtosis: native={k_native:.3f}  "
                f"mean_random={np.mean(kurt_random):.3f}",
                flush=True,
            )

        pd.DataFrame(kurtosis_rows).to_csv(kurtosis_csv, index=False)
        print(f"\n  [SAVED] {kurtosis_csv}\n", flush=True)

    # ====================================================================
    # ANALYSIS 2: Readout Weight Sparsity — fit once, rotate weights analytically
    # ====================================================================
    # Key identity: for X_rot = X @ Q, the Ridge solution satisfies w_rot = w_native @ Q.
    # Proof: w_rot = (Q^T X^T X Q + αI)^{-1} Q^T X^T y = Q^T (X^T X + αI)^{-1} X^T y = Q^T w_native.
    # In numpy broadcasting: W_rot = W_native @ Q  (applies Q^T to each row).
    # This replaces 45k Ridge fits with ~480 fits + cheap matmuls.
    print(f"{'='*60}", flush=True)
    print(f"  Analysis 2: Readout Weight Sparsity  [fit-once + weight rotation]", flush=True)
    print(f"{'='*60}", flush=True)

    # --- 2a: SEM (regression) ---
    sem_csv = f"data/results/sparsity_readout_sem_{alias}.csv"
    if os.path.exists(sem_csv):
        print(f"  [SKIP] {sem_csv} already exists.\n", flush=True)
    else:
        sem_rows = []
        for layer in range(n_layers):
            X = sem_acts[layer][:, :dims]

            # Fit once on native X
            W_native = fit_ridge_weights(X, Y_sem, is_clf=False)   # (n_targets, dims)
            k_native = kurtosis_of_weights(W_native)
            sem_rows.append({
                "layer": layer, "path": "native", "alpha": 0.0, "seed": -1,
                "kurtosis_weights": k_native,
            })

            # Rotations: just apply Q to weights — no refit
            k_random = []
            for seed in range(n_seeds):
                rot_mats = rot_cache_np[seed]
                for ai, alpha_val in enumerate(alpha_steps):
                    Q = rot_mats[ai + 1]
                    k = kurtosis_of_weights(rotate_weights(W_native, Q))
                    sem_rows.append({
                        "layer": layer, "path": "random",
                        "alpha": round(float(alpha_val), 1),
                        "seed": seed, "kurtosis_weights": k,
                    })
                    k_random.append(k)

            print(
                f"  [sem] Layer {layer:2d} | w-kurtosis: native={k_native:.3f}  "
                f"mean_random={np.mean(k_random):.3f}",
                flush=True,
            )

        pd.DataFrame(sem_rows).to_csv(sem_csv, index=False)
        print(f"  [SAVED] {sem_csv}\n", flush=True)

    # Free SEM activations
    del sem_acts

    # --- 2b: LANG tasks (classification) ---
    if lang_acts is None:
        print("  [SKIP] LANG cache not available — skipping LANG readout sparsity.\n", flush=True)
    else:
        for task in lang_tasks:
            task_name = task["task"]
            idxs = task["idxs"]
            Y_task = task["Y"]

            out_csv = f"data/results/sparsity_readout_{task_name}_{alias}.csv"
            if os.path.exists(out_csv):
                print(f"  [SKIP] {task_name}: already exists.", flush=True)
                continue

            task_rows = []

            for layer in range(n_layers):
                X = lang_acts[layer][idxs][:, :dims]

                # Fit once on native X
                W_native = fit_ridge_weights(X, Y_task, is_clf=True,
                                             groups=task["pair_ids"])  # (1, dims)
                k_native = kurtosis_of_weights(W_native)
                task_rows.append({
                    "layer": layer, "path": "native", "alpha": 0.0, "seed": -1,
                    "kurtosis_weights": k_native,
                })

                # Rotations: apply Q to weights — no refit
                k_random = []
                for seed in range(n_seeds):
                    rot_mats = rot_cache_np[seed]
                    for ai, alpha_val in enumerate(alpha_steps):
                        Q = rot_mats[ai + 1]
                        k = kurtosis_of_weights(rotate_weights(W_native, Q))
                        task_rows.append({
                            "layer": layer, "path": "random",
                            "alpha": round(float(alpha_val), 1),
                            "seed": seed, "kurtosis_weights": k,
                        })
                        k_random.append(k)

                print(
                    f"  [{task_name}] Layer {layer:2d} | "
                    f"w-kurtosis: native={k_native:.3f}  "
                    f"mean_random={np.mean(k_random):.3f}",
                    flush=True,
                )

            pd.DataFrame(task_rows).to_csv(out_csv, index=False)
            print(f"  [SAVED] {out_csv}\n", flush=True)

    print(f"{'='*60}", flush=True)
    print(f"  Done — {alias}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
