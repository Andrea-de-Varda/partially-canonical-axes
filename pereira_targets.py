"""
Brain targets for the Pereira analyses: repetition-averaged voxels.

Each participant's `language_from90to100prcnt` matrix (pereira_data.load_pereira)
is n_sentences x (n_sessions * 1364): the 1364 language voxels, once per session,
session-major (column j*1364 + v = voxel v in session j). Every participant has
exactly 1364 unique voxels, the (session, voxel) layout is identical for every
sentence, and the column count is 1364 x the number of sessions (1-4). The
single-session columns are therefore repeated measurements of the same voxels,
not separate voxels.

WHAT THIS MODULE DOES
  * averages each voxel over its sessions -> n_sentences x 1364;
  * computes each voxel's test-retest reliability: the mean over session pairs
    of the Pearson r across ALL sentences between the voxel's two sessions
    (brain data only; no model involved);
  * voxel set "top20rel" (main text): the 20% most reliable voxels of each
    participant (exactly ceil(0.2 * 1364) = 273), ranked on that reliability;
    voxel set "all" (SI): all 1364 voxels;
  * voxel_native_counts: voxel-level native-vs-rotated summary, computed in the
    Pereira worker and stored as CSV columns (no per-voxel arrays are saved);
  * drops participants with a single session (no repetition to average and no
    reliability): Exp 2 participant index 5 (UID 343). Participant indices are
    the ORIGINAL positions in the struct, so index 5 is simply absent from Exp 2.
"""
import warnings

import numpy as np

N_VOX = 1364                  # language voxels per participant (top 10% of the fixed MNI parcels)
TOP_FRACTION = 0.2
VOXSETS = ("top20rel", "all")
MAIN_VOXSET = "top20rel"
NETWORK_KEY = "language_from90to100prcnt"
# Checkpoint contract: v4 = repetition-averaged voxel targets (+ per-voxel outputs).
CKPT_VERSION = "passagecv4"
RESULTS_DIRS = {
    "top20rel": "data/results/pereira_voxavg_top20rel",
    "all": "data/results/pereira_voxavg_all",
}


def results_dir(voxset):
    if voxset not in RESULTS_DIRS:
        raise ValueError(f"unknown voxel set {voxset!r}; expected one of {VOXSETS}")
    return RESULTS_DIRS[voxset]


def session_blocks(Y, n_vox=N_VOX):
    """(n_sent, k*n_vox) session-major columns -> (k, n_sent, n_vox)."""
    Y = np.asarray(Y, dtype=np.float64)
    n, c = Y.shape
    if c % n_vox:
        raise ValueError(f"{c} columns is not a multiple of {n_vox} voxels")
    k = c // n_vox
    return Y.reshape(n, k, n_vox).transpose(1, 0, 2)


def _colcorr(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (A * B).sum(0) / np.sqrt((A * A).sum(0) * (B * B).sum(0))
    r[~np.isfinite(r)] = np.nan
    return r


def voxel_reliability(blocks):
    """Mean pairwise between-session Pearson r per voxel; None if < 2 sessions."""
    k = blocks.shape[0]
    if k < 2:
        return None
    rs = [_colcorr(blocks[a], blocks[b]) for a in range(k) for b in range(a + 1, k)]
    with warnings.catch_warnings():            # all-NaN column (constant voxel) -> NaN, silently
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(rs), axis=0)


def top_reliable(rel, fraction=TOP_FRACTION):
    """Indices (sorted ascending) of the ceil(fraction * n) most reliable voxels; NaN ranks last."""
    n_keep = int(np.ceil(fraction * len(rel)))
    order = np.argsort(np.where(np.isfinite(rel), rel, -np.inf), kind="stable")[::-1]
    return np.sort(order[:n_keep])


def build_targets(participant_structs, voxset, n_vox=N_VOX):
    """
    participant_structs: list of per-participant dicts from
    pereira_data.load_pereira()['net_structs'][<experiment>].

    Returns a list (one entry per KEPT participant, in original order) of dicts:
      part         original participant index in the struct
      Y            (n_sent, n_selected) repetition-averaged responses
      vox_idx      (n_selected,) indices into the participant's 1364 voxels
      n_sessions   number of sessions averaged
      reliability  (1364,) test-retest reliability of every voxel
    """
    if voxset not in VOXSETS:
        raise ValueError(f"unknown voxel set {voxset!r}; expected one of {VOXSETS}")
    out = []
    for i, st in enumerate(participant_structs):
        blocks = session_blocks(st[NETWORK_KEY], n_vox)
        rel = voxel_reliability(blocks)
        if rel is None:
            continue                                  # single session: dropped
        vox = top_reliable(rel) if voxset == "top20rel" else np.arange(n_vox)
        out.append({"part": i, "Y": blocks.mean(0)[:, vox], "vox_idx": vox,
                    "n_sessions": blocks.shape[0], "reliability": rel})
    return out


def voxel_native_counts(native, random_seeds):
    """
    Voxel-level summary of the native-vs-rotated comparison, computed in the
    worker so that no per-voxel arrays have to be stored.
      native       (n_vox,)           native-basis score per voxel (alpha = 0)
      random_seeds (n_seeds, n_vox)   fully rotated score per voxel (alpha = 1), per seed
    Returns (n_valid_voxels, fraction of (voxel, seed) pairs with native > random,
    fraction of voxels where native beats every seed). The first fraction equals
    the mean over voxels of the share of random rotations the native basis beats;
    chance is 0.5 and 1 / (n_seeds + 1) respectively.
    """
    native = np.asarray(native, dtype=float); random_seeds = np.asarray(random_seeds, dtype=float)
    ok = np.isfinite(native) & np.isfinite(random_seeds).all(0)
    if not ok.any():
        return 0, np.nan, np.nan
    gt = native[ok][None, :] > random_seeds[:, ok]
    return int(ok.sum()), float(gt.mean()), float(gt.all(0).mean())
