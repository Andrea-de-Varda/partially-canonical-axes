from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import geotorch
from torchmetrics.functional import pairwise_cosine_similarity
from torch import optim
import numpy as np
from get_alignment_pairwise import split_data, split_data_by_indices, normalize_datasets
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from metric_config import MATCH_ABS

def match_units(out, match_abs=None):
    """
    out: (n_units, n_targets) train similarity matrix. Returns (idxs, sgn):
    the matched unit per target and the sign to apply to it. With match_abs
    (project default, metric_config.MATCH_ABS) units are matched by |sim| and
    negatively-correlated units are flipped -- axes matched up to orientation.
    """
    match_abs = MATCH_ABS if match_abs is None else match_abs
    if match_abs:
        idxs = torch.argmax(out.abs(), 0)
        sgn = torch.sign(out[idxs, torch.arange(out.shape[1], device=out.device)])
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    else:
        idxs = torch.argmax(out, 0)
        sgn = torch.ones(out.shape[1], device=out.device, dtype=out.dtype)
    return idxs, sgn


def compute_sim(x, y, mat, ids, sgn=None):
    """
    Per-voxel Pearson r (as centred cosine) between voxel i and rotated unit
    ids[i], with the unit's sign flipped by sgn[i] (None = all +1). Runs on
    mat's device (a d x d matmul per call; keeping it on the GPU matters at d ~ 5000).
    """
    dev = mat.device
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev) @ mat.float()
    y = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=dev)
    x = x.T
    y = y.T
    out = pairwise_cosine_similarity(x - torch.nanmean(x,1)[:,None], y - torch.nanmean(y,1)[:,None])
    ids_t = torch.as_tensor(ids, device=dev)
    result = out[ids_t, torch.arange(out.shape[1], device=dev)]
    if sgn is not None:
        result = result * torch.as_tensor(sgn, dtype=result.dtype, device=dev)
    return list(result.detach().cpu().numpy())


def predict_matched(x, mat, ids, sgn=None):
    """Held-out predictions: (sign-corrected) rotated activations of the unit matched to each voxel -> (n, n_vox)."""
    dev = mat.device
    xr = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev) @ mat.float()
    ids_t = torch.as_tensor(ids, device=dev)
    pred = xr[:, ids_t]
    if sgn is not None:
        pred = pred * torch.as_tensor(sgn, dtype=pred.dtype, device=dev)[None, :]
    return pred.detach().cpu().numpy() 

class Model(nn.Module):
    def __init__(self, num_neurons, initialize=True):
        super().__init__()
        self.linear = nn.Linear(num_neurons, num_neurons, bias=False)
        geotorch.orthogonal(self.linear, "weight", triv="expm")

        if initialize:
            with torch.no_grad():
                eye = torch.eye(
                    num_neurons,
                    dtype=self.linear.weight.dtype,
                    device=self.linear.weight.device,
                )
                self.linear.weight = eye
    def forward(self, x):
        # row-vector convention: x @ W
        return x @ self.linear.weight

class OnetoOneLoss(nn.Module):
    def __init__(self, match_abs=None):
        super(OnetoOneLoss, self).__init__()
        # None -> project default (metric_config.MATCH_ABS). Explicit so a per-run
        # metric override (PEREIRA --signed_match) reaches the loss too.
        self.match_abs = MATCH_ABS if match_abs is None else bool(match_abs)

    def forward(self, x, y):
        
        out = pairwise_cosine_similarity(x.T - torch.nanmean(x.T,1)[:,None], y.T - torch.nanmean(y.T,1)[:,None])
        
        safe_out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
        # Sign-invariant objective under the project metric (metric_config.MATCH_ABS):
        # the best match is the unit with the largest |cos|, so the learned rotation
        # is scored on the same scale as the |r|-matched native/random bases.
        if self.match_abs:
            safe_out = safe_out.abs()
        safe_out = 1-safe_out.max(0).values
        return torch.nanmean(safe_out)

    
def find_optimal(train_x, train_y, val_x, val_y, test_x, test_y, num_epochs = 10, initialize = True, keep_rotations = False, match_abs = None): #set initialize to True to start from identity
    """
    Returns (val_curve, test_curve, As, test_preds).
      val_curve/test_curve : per epoch, per-voxel score lists
      As                   : per-epoch rotation matrices ONLY if keep_rotations,
                             else an empty list (storing every epoch's d x d
                             matrix costs GBs per worker at d ~ 5000).
      test_preds           : per-epoch held-out predictions (n_test, n_vox), a
                             few hundred KB each, so the caller can pick the
                             dev-selected epoch and CV-concatenate across folds.
    match_abs: None -> project default (metric_config.MATCH_ABS); applied to the
    train-side matching AND the loss, so native/random/optimal share one metric.
    """
    match_abs = MATCH_ABS if match_abs is None else bool(match_abs)
    # Clean NaN/Inf in ALL splits (train, val, test) for both X and Y
    # This matches the native basis alignment procedure
    xtrain = train_x.copy()
    ytrain = train_y.copy()
    xtrain[~np.isfinite(xtrain)] = 0
    ytrain[~np.isfinite(ytrain)] = 0

    xval = val_x.copy()
    yval = val_y.copy()
    xval[~np.isfinite(xval)] = 0
    yval[~np.isfinite(yval)] = 0

    xtest = test_x.copy()
    ytest = test_y.copy()
    xtest[~np.isfinite(xtest)] = 0
    ytest[~np.isfinite(ytest)] = 0

    tensor_x = torch.Tensor(xtrain)
    tensor_y = torch.Tensor(ytrain)

    my_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(my_dataset, batch_size=len(my_dataset), shuffle=False)
    model = Model(xtrain.shape[1], initialize=initialize).cuda()


    A = model.linear.weight.detach()
    tensor_x = tensor_x.cuda()
    tensor_y = tensor_y.cuda()
    x_tr = tensor_x @ A
    x_tr = x_tr.T
    y_tr = tensor_y.T
    out = pairwise_cosine_similarity(x_tr - torch.nanmean(x_tr,1)[:,None], y_tr - torch.nanmean(y_tr,1)[:,None])
    idxs, sgn = match_units(out, match_abs)

    results, test_results, As, test_preds = [], [], [], []
    if keep_rotations:
        As.append(A.cpu().numpy())
    result = compute_sim(xval, yval, A, idxs, sgn)
    results.append(result)

    result = compute_sim(xtest, ytest, A, idxs, sgn)
    test_results.append(result)
    test_preds.append(predict_matched(xtest, A, idxs, sgn))

    criterion = OnetoOneLoss(match_abs)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    # one scheduler step per epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=1, epochs=num_epochs, anneal_strategy='cos')
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        cayley_failed = False
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            try:
                outputs = model(inputs.cuda())
            except Exception as e:
                # the orthogonal parametrization can fail numerically; skip this epoch
                print(f"  [WARNING] Forward pass failed at epoch {epoch}, skipping: {e}")
                cayley_failed = True
                break
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print statistics
            running_loss += loss.item()

        if cayley_failed:
            results.append(results[-1] if results else [0.0])
            test_results.append(test_results[-1] if test_results else [0.0])
            test_preds.append(test_preds[-1])
            if keep_rotations:
                As.append(As[-1] if As else np.eye(xtrain.shape[1]))
            continue

        try:
            A = model.linear.weight.detach()  # already on GPU
        except Exception as e:
            print(f"  [WARNING] Skipping epoch {epoch} evaluation: {e}")
            results.append(results[-1] if results else [0.0])
            test_results.append(test_results[-1] if test_results else [0.0])
            test_preds.append(test_preds[-1])
            if keep_rotations:
                As.append(As[-1] if As else np.eye(xtrain.shape[1]))
            continue
        x_tr = tensor_x @ A

        x_tr = x_tr.T
        y_tr = tensor_y.T
        out = pairwise_cosine_similarity(x_tr - torch.nanmean(x_tr,1)[:,None], y_tr - torch.nanmean(y_tr,1)[:,None])
        idxs, sgn = match_units(out, match_abs)

        result = compute_sim(xval, yval, A, idxs, sgn)
        results.append(result)

        result = compute_sim(xtest, ytest, A, idxs, sgn)
        test_results.append(result)
        test_preds.append(predict_matched(xtest, A, idxs, sgn))
        if keep_rotations:
            As.append(A.cpu().numpy())


    return results, test_results, As, test_preds


def get_optimal_alignment(
    X, Y,
    train_fract=0.6, dev_fract=0.2, test_fract=0.2,
    ri=0, num_epochs=25, return_curve=False, return_rotation=False,
    split_indices=None, return_test_pred=False, match_abs=None
):
    """
    return_test_pred=True returns (test_score, test_pred) where test_pred is the
    (n_test, n_vox) held-out prediction at the dev-selected epoch. Mutually
    exclusive with return_curve/return_rotation (used by the CV path only).
    """
    if split_indices is not None:
        if len(split_indices[1]) == 0:
            raise ValueError("split_indices must include a non-empty dev split (epoch selection)")
        X_split = split_data_by_indices(X, split_indices)
        Y_split = split_data_by_indices(Y, split_indices)
    else:
        if dev_fract == 0:
            raise ValueError("dev_fract must be >0 to avoid test leakage")
        # identical split & normalisation
        X_split = split_data(X, ri, train_fract, dev_fract, test_fract)
        Y_split = split_data(Y, ri, train_fract, dev_fract, test_fract)
    X_split = normalize_datasets(X_split)
    Y_split = normalize_datasets(Y_split)

    train_x, val_x, test_x = X_split
    train_y, val_y, test_y = Y_split

    # Truncate X to even dimensions (to match native basis alignment)
    dims = (train_x.shape[1] // 2) * 2
    train_x, val_x, test_x = train_x[:, :dims], val_x[:, :dims], test_x[:, :dims]

    # ---------- handle empty val set ----------
    if len(val_x) == 0:          # dev_fract == 0
        val_x, val_y = test_x, test_y   # just evaluate on test each epoch

    val_curve, test_curve, As, test_preds = find_optimal(
        train_x, train_y, val_x, val_y, test_x, test_y,
        num_epochs=num_epochs, keep_rotations=return_rotation, match_abs=match_abs)

    # Epoch selected on the dev split (mean over targets); the test split is never used for selection.
    val_arr  = np.array(val_curve,  dtype=float)  # shape (n_epochs+1, n_voxels)
    test_arr = np.array(test_curve, dtype=float)  # shape (n_epochs+1, n_voxels)

    epoch_means = np.nanmean(val_arr, axis=1)     # mean across voxels
    best_epoch  = int(np.nanargmax(epoch_means))  # epoch index with best mean val

    test_score = float(np.nanmean(test_arr[best_epoch]))  # mean across voxels
    print(f"Finished training. Best epoch {best_epoch}, test mean = {round(test_score, 3)}")

    if return_test_pred:
        return test_score, test_preds[best_epoch]
    if return_rotation:
        best_rotation = As[best_epoch]  # orthogonal matrix (dims x dims)
        if return_curve:
            return test_score, val_curve, test_curve, best_rotation
        return test_score, best_rotation
    return (test_score, val_curve, test_curve) if return_curve else test_score


def get_optimal_alignment_cv(X, Y, folds, num_epochs=25, return_detail=False, match_abs=None):
    """
    Grouped-CV version of get_optimal_alignment: trains one orthogonal
    rotation per fold (epoch selected on that fold's dev split, scored on the
    held-out test split, exactly as in get_optimal_alignment) and returns the
    mean test score across folds.

    return_detail=True returns a dict with both estimators:
      optimal        : mean over folds of the per-fold (mean-over-voxel) r
      optimal_folds  : the 10 per-fold values
      optimal_concat : mean over voxels of the CV-concatenated r (predictions
                       from each fold's dev-selected epoch, concatenated, one
                       r per voxel on all held-out sentences vs raw Y)
    """
    from get_alignment_pairwise import concat_folds_pearson
    fold_scores, fold_preds = [], []
    for k, fold in enumerate(folds):
        score, pred = get_optimal_alignment(X, Y, split_indices=fold, num_epochs=num_epochs,
                                            return_test_pred=True, match_abs=match_abs)
        fold_scores.append(score)
        fold_preds.append(pred)
    mean_score = float(np.nanmean(fold_scores))
    if not return_detail:
        return mean_score
    concat_r = concat_folds_pearson(fold_preds, folds, Y)          # (n_vox,)
    return {"optimal": mean_score,
            "optimal_folds": np.asarray(fold_scores, dtype=float),
            "optimal_concat": float(np.nanmean(concat_r))}


