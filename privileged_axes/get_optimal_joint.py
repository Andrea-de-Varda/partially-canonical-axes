"""
Joint optimal rotation across multiple feature datasets.

Instead of fitting a separate orthogonal rotation per task (as
get_optimal_alignment does), this module fits ONE shared orthogonal rotation W
that minimises the equal-weighted sum of OnetoOneLoss across all tasks.

This makes the model→feature optimal score directly comparable to model→model
and model→brain optimal scores (which use a single rotation each).

Key reuse:
- `Model`           class from get_optimal_pairwise.py (geotorch orthogonal nn.Linear)
- `OnetoOneLoss`    class from get_optimal_pairwise.py (1 - max-cosine-similarity)
- `compute_sim`     helper from get_optimal_pairwise.py (per-voxel test correlations)
- `split_data`, `normalize_datasets` from get_alignment_pairwise.py
"""

import numpy as np
import torch

from get_optimal_pairwise import Model, OnetoOneLoss, compute_sim, match_units
from get_alignment_pairwise import split_data, split_data_by_indices, normalize_datasets
from torchmetrics.functional import pairwise_cosine_similarity


def _prepare_task_data(X, Y, ri=0, train_fract=0.6, dev_fract=0.2, test_fract=0.2,
                       split_indices=None):
    """Reproduce the per-task split/normalise/clean steps from get_optimal_alignment.

    Returns (xtrain, ytrain, xval, yval, xtest, ytest) numpy float32 arrays,
    cleaned of NaN/Inf, with X NOT yet truncated (truncation happens after we
    determine the global even d).
    """
    if split_indices is not None:
        # Explicit (train, dev, test) indices, e.g. the pair-grouped split of the
        # minimal-pair tasks (feature_data.pair_split) -- identical to the split
        # the per-task analysis uses.
        X_split = split_data_by_indices(X, split_indices)
        Y_split = split_data_by_indices(Y, split_indices)
    else:
        if dev_fract == 0:
            raise ValueError("dev_fract must be >0 to avoid test leakage")
        X_split = split_data(X, ri, train_fract, dev_fract, test_fract)
        Y_split = split_data(Y, ri, train_fract, dev_fract, test_fract)
    X_split = normalize_datasets(X_split)
    Y_split = normalize_datasets(Y_split)

    train_x, val_x, test_x = X_split
    train_y, val_y, test_y = Y_split

    for arr in (train_x, val_x, test_x, train_y, val_y, test_y):
        arr[~np.isfinite(arr)] = 0

    # Empty val: fall back to test (matches get_optimal_alignment behaviour)
    if len(val_x) == 0:
        val_x, val_y = test_x, test_y

    return train_x, train_y, val_x, val_y, test_x, test_y


def find_optimal_joint(task_data, num_epochs=50, initialize=True):
    """
    Train one orthogonal rotation W shared across all tasks.

    Parameters
    ----------
    task_data : list of (task_name, X, Y) tuples
        X has shape (n_t, d). All tasks must share the same d.
        Y has shape (n_t, k_t) — k_t (number of voxels/dims) can differ across tasks.
    num_epochs : int
    initialize : bool
        If True, init W to identity (matching get_optimal_pairwise.find_optimal).

    Returns
    -------
    test_scores : dict {task_name: float}
        Per-task test mean-over-voxels Pearson r at the joint best_epoch.
    val_curves  : dict {task_name: np.ndarray of length (num_epochs+1)}
        Per-task per-epoch val mean-over-voxels score (for diagnostics).
    best_epoch  : int
    rotation    : np.ndarray of shape (d, d)
        The learned orthogonal rotation at best_epoch.
    """
    if not task_data:
        raise ValueError("task_data is empty")

    # ---- per-task split & normalise ----
    prepared = []
    d_full = None
    for item in task_data:
        # (name, X, Y) or (name, X, Y, split_indices)
        name, X, Y = item[:3]
        split_indices = item[3] if len(item) > 3 else None
        if d_full is None:
            d_full = X.shape[1]
        elif X.shape[1] != d_full:
            raise ValueError(
                f"task {name} has X with {X.shape[1]} columns; expected {d_full}")
        xtr, ytr, xv, yv, xte, yte = _prepare_task_data(X, Y, split_indices=split_indices)
        prepared.append((name, xtr, ytr, xv, yv, xte, yte))

    # Truncate X to even dims (matching find_optimal in get_optimal_pairwise.py)
    dims = (d_full // 2) * 2
    prepared = [
        (n, xtr[:, :dims], ytr, xv[:, :dims], yv, xte[:, :dims], yte)
        for (n, xtr, ytr, xv, yv, xte, yte) in prepared
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-build CUDA tensors (kept on GPU across epochs — same pattern as find_optimal)
    task_tensors = []
    for n, xtr, ytr, xv, yv, xte, yte in prepared:
        task_tensors.append({
            "name":  n,
            "xtr":   torch.tensor(xtr, dtype=torch.float32, device=device),
            "ytr":   torch.tensor(ytr, dtype=torch.float32, device=device),
            "xv_np": xv,  "yv_np": yv,    # compute_sim expects numpy
            "xte_np": xte, "yte_np": yte,
        })

    model = Model(dims, initialize=initialize).to(device)
    criterion = OnetoOneLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=1, epochs=max(num_epochs, 1),
        anneal_strategy="cos")

    n_tasks = len(task_tensors)

    # Per-task curves: length num_epochs+1 (initial identity + post-epoch evals)
    val_curves  = {t["name"]: [] for t in task_tensors}
    test_curves = {t["name"]: [] for t in task_tensors}
    rotations   = []

    # ---- evaluate at A = current weight (initial identity if initialize=True) ----
    def _evaluate(A_tensor):
        A_np_keep = A_tensor.detach()
        for t in task_tensors:
            xtr, ytr = t["xtr"], t["ytr"]
            x_tr = (xtr @ A_np_keep).T
            y_tr = ytr.T
            out = pairwise_cosine_similarity(
                x_tr - torch.nanmean(x_tr, 1)[:, None],
                y_tr - torch.nanmean(y_tr, 1)[:, None],
            )
            idxs, sgn = match_units(out)          # project metric (metric_config.MATCH_ABS)
            v_corrs = compute_sim(t["xv_np"].copy(),  t["yv_np"].copy(),  A_np_keep, idxs, sgn)
            te_corrs = compute_sim(t["xte_np"].copy(), t["yte_np"].copy(), A_np_keep, idxs, sgn)
            val_curves[t["name"]].append(float(np.nanmean(v_corrs)))
            test_curves[t["name"]].append(float(np.nanmean(te_corrs)))
        rotations.append(A_np_keep.cpu().numpy())

    A_init = model.linear.weight.detach()
    _evaluate(A_init)

    # ---- training loop ----
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        cayley_failed = False
        loss_total = None
        for t in task_tensors:
            try:
                out = model(t["xtr"])
            except Exception as e:
                print(f"  [WARNING] forward failed for task {t['name']} "
                      f"at epoch {epoch}: {e}")
                cayley_failed = True
                break
            loss_t = criterion(out, t["ytr"])
            loss_total = loss_t if loss_total is None else (loss_total + loss_t)

        if cayley_failed or (loss_total is None):
            # Replicate previous epoch's evaluations
            for t in task_tensors:
                val_curves[t["name"]].append(
                    val_curves[t["name"]][-1] if val_curves[t["name"]] else 0.0)
                test_curves[t["name"]].append(
                    test_curves[t["name"]][-1] if test_curves[t["name"]] else 0.0)
            rotations.append(rotations[-1] if rotations else np.eye(dims))
            continue

        # Equal-weight: divide by n_tasks (Adam direction unaffected, but keeps
        # learning-rate scale comparable to single-task case)
        (loss_total / n_tasks).backward()
        optimizer.step()
        scheduler.step()

        try:
            A = model.linear.weight.detach()
        except Exception as e:
            print(f"  [WARNING] Skipping eval at epoch {epoch}: {e}")
            for t in task_tensors:
                val_curves[t["name"]].append(
                    val_curves[t["name"]][-1] if val_curves[t["name"]] else 0.0)
                test_curves[t["name"]].append(
                    test_curves[t["name"]][-1] if test_curves[t["name"]] else 0.0)
            rotations.append(rotations[-1] if rotations else np.eye(dims))
            continue

        _evaluate(A)

    # ---- pick best_epoch via joint val score ----
    val_arr = np.array(
        [val_curves[t["name"]] for t in task_tensors],
        dtype=float)                            # shape (n_tasks, num_epochs+1)
    epoch_means = np.nanmean(val_arr, axis=0)   # shape (num_epochs+1,)
    best_epoch = int(np.nanargmax(epoch_means))

    test_scores = {name: float(test_curves[name][best_epoch])
                   for name in test_curves}
    val_curves_np = {name: np.array(v, dtype=float)
                     for name, v in val_curves.items()}

    print(f"  [JOINT] best_epoch={best_epoch}  joint_val={epoch_means[best_epoch]:.3f}  "
          f"test_means={ {n: round(s, 3) for n, s in test_scores.items()} }",
          flush=True)

    return test_scores, val_curves_np, best_epoch, rotations[best_epoch]

