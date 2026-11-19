from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit
from scipy.stats import pearsonr
import numpy as np
import math
import sys
import torch
from rotation_code import randomRotationPowers
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from metric_config import resolve_match_abs

def normalize(X, mean=None, std=None, remove_constant=True):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    std[std == 0] = 1  # prevent divide-by-zero

    X_norm = (X - mean) / std

    if remove_constant:
        keep_cols = ~(np.isnan(X_norm).any(axis=0) | np.isinf(X_norm).any(axis=0))
        X_norm = X_norm[:, keep_cols]
        mean = mean[keep_cols]
        std = std[keep_cols]
    return X_norm, mean, std, keep_cols if remove_constant else np.arange(X.shape[1])


def normalize_datasets(datasets):
    datasets[0], mean, std, _ = normalize(datasets[0], remove_constant=False)
    # dev can be [] (python list) or a numpy array
    if len(datasets) > 1 and datasets[1] is not None:
        if isinstance(datasets[1], (list, tuple)):
            has_dev = len(datasets[1]) > 0
        else:
            # numpy / torch-like
            has_dev = hasattr(datasets[1], "size") and datasets[1].size > 0
        if has_dev:
            datasets[1], _, _, _ = normalize(datasets[1], mean, std, remove_constant=False)
    datasets[2], _, _, _ = normalize(datasets[2], mean, std, remove_constant=False)
    return datasets

def split_data(data, r_, train_fract, dev_fract, test_fract, categories = []):
    if len(categories):
        train_idx, test_idx = train_test_split(np.arange(len(data)), test_size = test_fract, random_state = r_, stratify = categories)
    else: 
        train_idx, test_idx = train_test_split(np.arange(len(data)), test_size = test_fract, random_state = r_)
    if dev_fract != 0:
        if len(categories):
            train_idx, dev_idx = train_test_split(train_idx, test_size = dev_fract/(dev_fract + train_fract), random_state = r_, stratify = categories[train_idx])
        else: 
            train_idx, dev_idx = train_test_split(train_idx, test_size = dev_fract/(dev_fract + train_fract), random_state = r_)
        dev = data[dev_idx]
    else: 
        dev = []
    train = data[train_idx] 
    test = data[test_idx] 
    return [train, dev, test]

def columnwise_pearson(pred, y):
    """Pearson r between matching columns of pred and y, both (n, k). NaN where undefined."""
    pred = np.asarray(pred, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    pc = pred - np.nanmean(pred, axis=0); yc = y - np.nanmean(y, axis=0)
    num = np.nansum(pc * yc, axis=0)
    den = np.sqrt(np.nansum(pc ** 2, axis=0) * np.nansum(yc ** 2, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[~np.isfinite(r)] = np.nan
    return r


def concat_folds_pearson(fold_preds, folds, Y):
    """
    CV-concatenated correlation. fold_preds[k] has shape (..., n_test_k, n_vox)
    (leading dims arbitrary, e.g. rotations); folds[k] = (train, dev, test) index
    arrays. Predictions are concatenated across folds in fold order and
    correlated with the RAW held-out Y (same row order) -> shape (..., n_vox).
    Pearson r is invariant to affine changes of Y, so raw Y is the cleanest
    target; predictions keep their per-fold normalisation (train-split z-score),
    whose fold-to-fold offsets are negligible because the train splits overlap
    ~90%.
    """
    test_idx = np.concatenate([np.asarray(f[2]) for f in folds])
    P = np.concatenate([np.asarray(p) for p in fold_preds], axis=-2)   # (..., n_all, n_vox)
    Yt = np.asarray(Y)[test_idx]
    lead = P.shape[:-2]
    flat = P.reshape(-1, P.shape[-2], P.shape[-1])
    out = np.stack([columnwise_pearson(flat[i], Yt) for i in range(flat.shape[0])])
    return out.reshape(*lead, P.shape[-1]) if lead else out[0]


def make_grouped_folds(groups, n_splits=10, dev_fraction=0.25, dev_seed=0):
    """
    Group-level (e.g. passage-level) CV folds. Test sets come from
    GroupKFold(n_splits), so every group lands in exactly one test fold and no
    group ever straddles train and test. Within each fold's remaining ~90%, a
    GroupShuffleSplit holds out dev_fraction of the *training groups* as a dev
    set (used for optimal-rotation epoch selection), i.e. the same 3:1
    train:dev ratio as the original 60/20/20 random split.

    Returns a list of n_splits tuples (train_idx, dev_idx, test_idx). Fully
    deterministic (GroupKFold does not shuffle; dev_seed is fixed).
    """
    groups = np.asarray(groups)
    n = len(groups)
    folds = []
    gkf = GroupKFold(n_splits=n_splits)
    for trainval_idx, test_idx in gkf.split(np.arange(n), groups=groups):
        gss = GroupShuffleSplit(n_splits=1, test_size=dev_fraction, random_state=dev_seed)
        tr_rel, dev_rel = next(gss.split(trainval_idx, groups=groups[trainval_idx]))
        folds.append((trainval_idx[tr_rel], trainval_idx[dev_rel], test_idx))
    return folds

def split_data_by_indices(data, fold):
    """Split data into [train, dev, test] from explicit index arrays."""
    train_idx, dev_idx, test_idx = fold
    return [data[train_idx], data[dev_idx], data[test_idx]]

def get_alignment_axis_rotation(data1, data2, train_fract=0.6, dev_fract=0.2, test_fract=0.2, ri=0, rot_seed=0, categories=[], split_indices=None, return_predictions=False, match_abs=None):
    match_abs = resolve_match_abs(match_abs)
    if split_indices is not None:
        m_codes = split_data_by_indices(data1, split_indices)
        gts = split_data_by_indices(data2, split_indices)
    else:
        m_codes = split_data(data1, ri, train_fract, dev_fract, test_fract, categories)
        gts = split_data(data2, ri, train_fract, dev_fract, test_fract, categories)
    # Normalize both X and Y (to match optimal alignment procedure)
    m_codes = normalize_datasets(m_codes)
    gts = normalize_datasets(gts)
    # Clean NaN/Inf in both X and Y (train and test)
    for arr in [m_codes[0], m_codes[2], gts[0], gts[2]]:
        arr[~np.isfinite(arr)] = 0

    dims = math.floor(m_codes[0].shape[1] / 2) * 2
    rot_path = f'rotations/rot_mats{dims}_seed{rot_seed}.npy'
    
    try: 
        rot_mats = np.load(rot_path)
        print(f"Loaded rotations from {rot_path}")
    except FileNotFoundError:
        alpha = np.arange(0.1, 1.0 + 1e-15, 0.1)
        rot_mats = [np.identity(dims)] + randomRotationPowers(dims, alpha, seed=rot_seed, unpermute=True)
        np.save(rot_path, rot_mats)
        print(f"Saved rotations to {rot_path}")
    results = []
    preds_out = []
    for Q in rot_mats: 
        A = m_codes[0].copy()[:, :dims] @ Q 
        B = gts[0].copy()
        N = B.shape[0]

        sA = A.sum(0)
        sB = B.sum(0)
        p1 = N * np.einsum('ij,ik->kj', A, B)
        p2 = sA * sB[:, None]
        p3 = N * ((B ** 2).sum(0)) - (sB ** 2)
        p4 = N * ((A ** 2).sum(0)) - (sA ** 2)
        pcorr = ((p1 - p2) / np.sqrt(p4 * p3[:, None]))
        pcorr[np.isnan(pcorr)] = 0
        
        test_mat = (m_codes[2].copy()[:, :dims] @ Q)
        test_mat[np.isinf(test_mat)] = 0
        if match_abs:
            indices = np.argmax(np.abs(pcorr), 1)
            sgn = np.sign(pcorr[np.arange(len(indices)), indices]); sgn[sgn == 0] = 1
        else:
            indices = np.argmax(pcorr, 1); sgn = np.ones(len(indices))
        pred_mat = test_mat[:, indices] * sgn
        test_corr = [pearsonr(gts[2][:, v], pred_mat[:, v])[0] for v in range(gts[2].shape[1])]
        results.append(test_corr)
        if return_predictions:
            preds_out.append(pred_mat)
    
    if return_predictions:
        return np.asarray(results), np.stack(preds_out)
    return np.asarray(results)

def get_alignment_axis_rotation_gpu(
    data1, data2,
    train_fract=0.6, dev_fract=0.2, test_fract=0.2,
    ri=0, rot_seed=0, categories=None, split_indices=None,
    return_predictions=False, match_abs=None
):
    """
    return_predictions=True additionally returns the held-out predictions,
    shape (n_rotations, n_test, n_voxels): for each rotation, the (fold-
    normalised) activation of the unit matched to each voxel on the train
    split. Used to CV-concatenate predictions across folds so that a single
    Pearson r per voxel is computed on ALL held-out sentences, instead of
    averaging ten r's each estimated on 24-38 sentences.

    match_abs=True matches each voxel to the unit with the largest |train r|
    and flips the unit's sign when that r is negative, i.e. axes are matched
    up to orientation (project default, metric_config.MATCH_ABS). The signed
    argmax (match_abs=False) is asymmetric: a random
    rotation mixes native units with random-signed weights, so any component
    shared by the native units with a consistent sign is available to the
    rotated basis with BOTH signs but to the native basis with only one. In
    layers with a dominant same-sign shared component (gpt2-xl layers 36-46:
    effective rank 5-7, 100% of PC1 loadings of one sign) this lets a random
    rotation beat the native basis for every voxel anticorrelated with that
    component -- a metric artefact, not a basis-privilege fact (see metric_config.py).
    """
    # ---------- split & normalise ----------
    match_abs = resolve_match_abs(match_abs)
    if categories is None:
        categories = []
    if split_indices is not None:
        m_codes_ = split_data_by_indices(data1, split_indices)
        gts_     = split_data_by_indices(data2, split_indices)
    else:
        m_codes_ = split_data(data1, ri, train_fract, dev_fract, test_fract, categories)
        gts_     = split_data(data2, ri, train_fract, dev_fract, test_fract, categories)
    # Normalize both X and Y (to match optimal alignment procedure)
    m_codes_ = normalize_datasets(m_codes_)
    gts_ = normalize_datasets(gts_)

    # clean NaN / Inf in‑place for both X and Y (train and test)
    for idx in (0, 2):
        m_codes_[idx][~np.isfinite(m_codes_[idx])] = 0.0
        gts_[idx][~np.isfinite(gts_[idx])] = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_codes = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in m_codes_]
    gts     = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in gts_]

    # ---------- rotation matrices ----------
    dims = (m_codes[0].shape[1] // 2) * 2
    rot_path = f"rotations/rot_mats{dims}_seed{rot_seed}.npy"
    try:
        rot_mats_np = np.load(rot_path)
    except FileNotFoundError:
        alpha = np.arange(0.1, 1.0 + 1e-15, 0.1)
        rot_mats_np = [np.eye(dims)] + randomRotationPowers(dims, alpha, seed=rot_seed, unpermute=True)
        np.save(rot_path, rot_mats_np)
    rot_mats = [torch.as_tensor(R, dtype=torch.float32, device=device) for R in rot_mats_np]

    train_X, test_X = m_codes[0][:, :dims], m_codes[2][:, :dims]
    train_Y, test_Y = gts[0], gts[2]
    Ntrain, Ntest   = train_Y.shape[0], test_Y.shape[0]

    out = []
    preds_out = []
    for Q in rot_mats:
        # --- train‑split correlation matrix pcorr (voxels × dims)
        A_tr = train_X @ Q
        B_tr = train_Y
        sA, sB = A_tr.sum(0), B_tr.sum(0)

        p1 = Ntrain * torch.einsum("ij,ik->kj", A_tr, B_tr)
        p2 = sA * sB[:, None]
        p3 = Ntrain * (B_tr.square().sum(0)) - sB.square()
        p4 = Ntrain * (A_tr.square().sum(0)) - sA.square()
        pcorr = (p1 - p2) / torch.sqrt(p4 * p3[:, None])
        pcorr = torch.nan_to_num(pcorr, nan=0.0)

        if match_abs:
            best_idx = torch.argmax(pcorr.abs(), dim=1)     # (n_voxels,)
            sgn = torch.sign(pcorr.gather(1, best_idx[:, None])[:, 0]); sgn[sgn == 0] = 1
        else:
            best_idx = torch.argmax(pcorr, dim=1)           # (n_voxels,)
            sgn = None

        # --- test‑split predictions, gathered in one shot
        A_te = test_X @ Q
        pred = A_te.gather(1, best_idx.unsqueeze(0).expand(Ntest, -1))
        if sgn is not None:
            pred = pred * sgn

        # --- per‑voxel Pearson r (vectorised)
        gt  = test_Y
        gt_c   = gt - gt.mean(0)
        pred_c = pred - pred.mean(0)
        cov    = (gt_c * pred_c).sum(0)
        std_gt = torch.sqrt((gt_c.square()).sum(0))
        std_pr = torch.sqrt((pred_c.square()).sum(0))
        denom  = std_gt * std_pr + 1e-8
        r      = cov / denom
        r      = torch.nan_to_num(r, nan=float("nan"))

        out.append(r.cpu().numpy())
        if return_predictions:
            preds_out.append(pred.cpu().numpy())

    if return_predictions:
        return np.stack(out), np.stack(preds_out)   # (n_rot, n_vox), (n_rot, n_test, n_vox)
    return np.stack(out)          # shape: (n_rotations, n_voxels)

def get_alignment_axis_rotation_multiseed(data1, data2, n_seeds=5, return_all=False, folds=None,
                                          return_concat=False, **kwargs):
    """
    If `folds` is provided (list of (train_idx, dev_idx, test_idx), e.g. from
    make_grouped_folds), each rotation seed is evaluated on every fold and the
    per-voxel scores are averaged across folds before the across-seed
    mean/sem. Otherwise a single random split (seed ri) is used.

    return_concat=True (requires folds) additionally returns a dict with the
    second estimator, the CV-concatenated r (see concat_folds_pearson):
      concat_mean : (n_rot, n_vox)          mean over seeds
      concat_all  : (n_seeds, n_rot, n_vox)
      fold_means  : (n_seeds, n_folds, n_rot)  per-fold r, mean over voxels
    """
    # choose CPU vs GPU implementation
    align_fn = (
        get_alignment_axis_rotation_gpu
        if torch.cuda.is_available()
        else get_alignment_axis_rotation
    )

    ri_fixed = kwargs.get("ri", 0)
    all_scores = []
    concat_all, fold_means = [], []
    if return_concat and folds is None:
        raise ValueError("return_concat requires folds")

    for seed in range(n_seeds):
        local_kwargs = dict(kwargs)
        local_kwargs["ri"] = ri_fixed
        local_kwargs["rot_seed"] = seed

        if folds is not None:
            fold_scores, fold_preds = [], []
            for fold in folds:
                if return_concat:
                    sc, pr = align_fn(data1, data2, split_indices=fold, return_predictions=True, **local_kwargs)
                    fold_preds.append(pr)
                else:
                    sc = align_fn(data1, data2, split_indices=fold, **local_kwargs)
                fold_scores.append(sc)
            fold_scores = np.stack(fold_scores)                 # (n_folds, n_rot, n_vox)
            scores = np.nanmean(fold_scores, axis=0)
            if return_concat:
                concat_all.append(concat_folds_pearson(fold_preds, folds, data2))   # (n_rot, n_vox)
                fold_means.append(np.nanmean(fold_scores, axis=2))                  # (n_folds, n_rot)
        else:
            scores = align_fn(data1, data2, **local_kwargs)
        all_scores.append(scores)

    all_scores  = np.stack(all_scores)
    mean_scores = all_scores.mean(axis=0)
    sem_scores  = all_scores.std(axis=0, ddof=1) / np.sqrt(n_seeds)

    if return_concat:
        concat_all = np.stack(concat_all)
        extras = {"concat_mean": np.nanmean(concat_all, axis=0),
                  "concat_all": concat_all,
                  "fold_means": np.stack(fold_means)}
        if return_all:
            return mean_scores, sem_scores, all_scores, extras
        return mean_scores, extras
    if return_all:
        return mean_scores, sem_scores, all_scores
    return mean_scores
