"""
Stimuli and targets for the model-to-feature analyses.

Single source of truth for LANG_selectivity.py, SEM_alignment.py,
other/JOINT_optimal_features.py and other/sparsity_analysis.py.

Stimulus cleaning:
  * hypernymy: only 551 of the 1,000 well-formed sentences are distinct (some
    repeated 5x). We keep the FIRST pair for each distinct well-formed sentence
    -> 551 pairs.
  * wug: 2 pairs have identical well-formed and ill-formed sentences (pure label
    noise); they are dropped -> 998 pairs.
  * norms: read with NA parsing off (words such as "null"/"nan"/"none" stay
    words) and, where Warriner lists a word in two cases (AIDS/aids, TV/tv, ...),
    the lowercase entry is kept -> 13,372 unique words (the raw merge has 13,381
    rows, 9 of them case-duplicates with different affective ratings).
  * minimal-pair tasks are split BY PAIR (both members on the same side), see
    `pair_split`. With pairs kept together each split is exactly label-balanced.
"""

import json
import os
from glob import glob

import numpy as np
import pandas as pd

# Goes into activation-cache and checkpoint filenames so results computed on a
# different stimulus set / split are never resumed or reused.
DATA_TAG = "v2"

PERCEPTUAL_COLS = ["Auditory.mean", "Gustatory.mean", "Haptic.mean",
                   "Interoceptive.mean", "Olfactory.mean", "Visual.mean"]
AFFECTIVE_COLS = ["V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]


def _read_pairs(path, prefix=""):
    good, bad = [], []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            good.append(prefix + item["sentence_good"].strip())
            bad.append(prefix + item["sentence_bad"].strip())
    return good, bad


def _dedup_pairs(good, bad):
    """Drop pairs whose members are identical, and keep only the first pair for
    each distinct well-formed sentence."""
    seen, g2, b2 = set(), [], []
    for g, b in zip(good, bad):
        if g == b or g in seen:
            continue
        seen.add(g)
        g2.append(g)
        b2.append(b)
    return g2, b2


def load_lang_tasks(data_dir="data/language_data"):
    """Minimal-pair tasks (hypernymy, wug, 12 BLiMP sub-tasks).

    Returns a list of dicts with keys
      task      : name
      sentences : good + bad (good first)
      Y         : (n, 1) float array, 1 = well-formed, 0 = ill-formed
      pair_ids  : (n,) int array; good[i] and bad[i] share pair id i
    """
    sources = [("hypernymy", os.path.join(data_dir, "hypernymy_pairs.jsonl"), "Consider "),
               ("wug", os.path.join(data_dir, "wug_pairs.jsonl"), "")]
    sources += [(os.path.splitext(os.path.basename(p))[0], p, "")
                for p in sorted(glob(os.path.join(data_dir, "blimp_data", "*.jsonl")))]

    tasks = []
    for name, path, prefix in sources:
        if not os.path.exists(path):
            continue
        good, bad = _dedup_pairs(*_read_pairs(path, prefix))
        n = len(good)
        tasks.append({
            "task": name,
            "sentences": good + bad,
            "Y": np.concatenate([np.ones(n), np.zeros(n)])[:, None],
            "pair_ids": np.concatenate([np.arange(n), np.arange(n)]),
        })
    return tasks


def load_norms(data_dir="data"):
    """Perceptual (Lancaster) and affective (Warriner) norms on their shared words.

    Returns (words, sentences, Y_perceptual (n, 6), Y_affective (n, 3)).
    """
    aff = pd.read_csv(os.path.join(data_dir, "affective.csv"), low_memory=False,
                      keep_default_na=False, na_values=[])[["Word"] + AFFECTIVE_COLS]
    aff["is_lower"] = aff["Word"] == aff["Word"].str.lower()
    aff["Word"] = aff["Word"].str.lower()
    aff = (aff.sort_values(["Word", "is_lower"], ascending=[True, False])
              .drop_duplicates("Word").drop(columns="is_lower"))

    per = pd.read_csv(os.path.join(data_dir, "perceptual.csv"),
                      keep_default_na=False, na_values=[])[["Word"] + PERCEPTUAL_COLS]
    per["Word"] = per["Word"].str.lower()
    if per["Word"].duplicated().any():
        raise RuntimeError("Lancaster norms contain case-duplicate words")

    merged = pd.merge(aff, per, on="Word").sort_values("Word").reset_index(drop=True)
    words = merged["Word"].tolist()
    sentences = [f"The meaning of the word '{w}'" for w in words]
    Y_per = merged[PERCEPTUAL_COLS].to_numpy(dtype=float)
    Y_aff = merged[AFFECTIVE_COLS].to_numpy(dtype=float)
    return words, sentences, Y_per, Y_aff


def pair_split(pair_ids, train_fract=0.6, dev_fract=0.2, test_fract=0.2, seed=0):
    """Train/dev/test item indices with every pair on one side of the split.

    Pairs are shuffled with a fixed seed (numpy, independent of the sklearn
    version) and cut 60/20/20 by pair. Returns (train_idx, dev_idx, test_idx).
    """
    if abs(train_fract + dev_fract + test_fract - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    pair_ids = np.asarray(pair_ids)
    groups = np.unique(pair_ids)
    rng = np.random.default_rng(seed)
    groups = groups[rng.permutation(len(groups))]
    n_tr = int(round(train_fract * len(groups)))
    n_dev = int(round(dev_fract * len(groups)))
    g_tr, g_dev, g_te = groups[:n_tr], groups[n_tr:n_tr + n_dev], groups[n_tr + n_dev:]
    idx = lambda g: np.flatnonzero(np.isin(pair_ids, g))
    return idx(g_tr), idx(g_dev), idx(g_te)
