"""
Loader for the Pereira et al. (2018) fMRI responses used in the paper.

The pipeline needs, per experiment, the sentence order and, per participant, the
language-network response matrix (n_sentences x n_sessions*1364, session-major
columns; values are per-session GLM effect sizes from the Fedorenko lab's
re-analysis). These are distributed as data/pereira_language.npz (float32, which
is lossless for these data; see README "Brain data"). The original private file
data/processed_pereira_structs.pkl is used instead when present.

load_pereira() returns the same structure in both cases:
    {"net_structs":     {experiment: [ {NETWORK_KEY: (n_sent, n_cols) float64}, ... ]},
     "net_sent_orders": {experiment: [sentence, ...]}}
Participants keep their original order (index = position in the list), including
the Experiment 2 participant with a single session, which build_targets drops.
"""
import json
import os
import pickle

import numpy as np

from pereira_targets import NETWORK_KEY

NPZ = "data/pereira_language.npz"
PKL = "data/processed_pereira_structs.pkl"
EXPERIMENTS = ("PereiraE2_96pass", "PereiraE3_72pass")


def load_pereira(npz_path=NPZ, pkl_path=PKL):
    if os.path.exists(npz_path):
        return _from_npz(npz_path)
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        return {"net_structs": d["net_structs"], "net_sent_orders": d["net_sent_orders"]}
    raise FileNotFoundError(
        f"Neither {npz_path} nor {pkl_path} found. Download pereira_language.npz "
        "(see README, 'Brain data') into data/.")


def _from_npz(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    structs, sents = {}, {}
    for exp in EXPERIMENTS:
        sents[exp] = list(meta[exp]["sentences"])
        structs[exp] = [{NETWORK_KEY: z[f"{exp}__p{i}"].astype(np.float64)}
                        for i in range(len(meta[exp]["uids"]))]
    return {"net_structs": structs, "net_sent_orders": sents}
