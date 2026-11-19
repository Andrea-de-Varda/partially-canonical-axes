"""
Tests for stimulus loading, pair-grouped splits, Mamba layer indexing and
readout standardisation. Run from the repo root:

    HF_HUB_OFFLINE=1 python -m pytest -q other/test_core.py

The Mamba test needs state-spaces/mamba-130m-hf in the local HF cache and is
skipped otherwise; the norms test needs data/affective.csv
(scripts/fetch_warriner.sh) and is skipped otherwise.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "privileged_axes")]
os.chdir(ROOT)

from feature_data import load_lang_tasks, load_norms, pair_split  # noqa: E402
from layer_index import raw_to_canonical_mamba, canonicalize_mamba_layers  # noqa: E402


# ----------------------------- stimuli -----------------------------

def test_lang_task_counts_and_no_duplicates():
    tasks = {t["task"]: t for t in load_lang_tasks()}
    assert len(tasks) == 14
    n_pairs = {k: len(t["sentences"]) // 2 for k, t in tasks.items()}
    assert n_pairs["hypernymy"] == 551
    assert n_pairs["wug"] == 998
    for k, t in tasks.items():
        good = t["sentences"][: n_pairs[k]]
        bad = t["sentences"][n_pairs[k]:]
        assert len(set(good)) == len(good), k
        assert all(g != b for g, b in zip(good, bad)), k
        assert t["Y"].shape == (2 * n_pairs[k], 1)
        assert t["Y"][: n_pairs[k]].all() and not t["Y"][n_pairs[k]:].any()
        assert (t["pair_ids"][: n_pairs[k]] == t["pair_ids"][n_pairs[k]:]).all()
    assert all(n == 1000 for k, n in n_pairs.items() if k not in ("hypernymy", "wug"))


def test_hypernymy_prefix_kept():
    hyp = next(t for t in load_lang_tasks() if t["task"] == "hypernymy")
    assert all(s.startswith("Consider ") for s in hyp["sentences"])


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "affective.csv")),
                    reason="needs data/affective.csv: run scripts/fetch_warriner.sh")
def test_norms_unique_words():
    words, sents, Yp, Ya = load_norms()
    assert len(words) == 13372 == len(set(words))
    assert Yp.shape == (13372, 6) and Ya.shape == (13372, 3)
    assert np.isfinite(Yp).all() and np.isfinite(Ya).all()
    w = dict(zip(words, Ya[:, 0]))
    assert w["aids"] == pytest.approx(3.52)          # lowercase Warriner entry kept
    assert sents[0] == f"The meaning of the word '{words[0]}'"


# ----------------------------- split -----------------------------

def test_pair_split_keeps_pairs_together_and_is_balanced():
    n = 551
    pid = np.concatenate([np.arange(n), np.arange(n)])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    tr, dv, te = pair_split(pid)
    allidx = np.concatenate([tr, dv, te])
    assert len(allidx) == 2 * n and len(np.unique(allidx)) == 2 * n
    sets = [set(pid[i]) for i in (tr, dv, te)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    for idx in (tr, dv, te):
        assert y[idx].mean() == 0.5
    assert abs(len(tr) / (2 * n) - 0.6) < 0.01 and abs(len(te) / (2 * n) - 0.2) < 0.01
    # deterministic
    assert all((a == b).all() for a, b in zip((tr, dv, te), pair_split(pid)))


def test_pair_split_without_dev():
    pid = np.repeat(np.arange(100), 2)
    tr, dv, te = pair_split(pid, train_fract=0.8, dev_fract=0.0, test_fract=0.2)
    assert len(dv) == 0 and len(tr) == 160 and len(te) == 40


# ----------------------------- layers -----------------------------

def test_raw_to_canonical_mamba_mapping():
    # 24 blocks -> 25 raw entries: raw 0..22 = blocks 1..23, raw 23 = block 24 pre-norm, raw 24 = post-norm
    m = [raw_to_canonical_mamba(r, 25) for r in range(25)]
    assert m[:23] == list(range(1, 24))
    assert m[23] is None and m[24] == 24


def test_canonicalize_table():
    import pandas as pd
    df = pd.DataFrame({"layer": range(25), "v": range(25)})
    out = canonicalize_mamba_layers(df)
    assert list(out["layer"]) == list(range(1, 25))
    assert out.loc[out.layer == 24, "v"].item() == 24      # post-norm row kept
    assert 23 not in out["v"].tolist()                     # pre-norm duplicate dropped


def test_canonical_hidden_states_mamba():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from layer_index import canonical_hidden_states
    name = "state-spaces/mamba-130m-hf"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = transformers.AutoModel.from_pretrained(name, local_files_only=True).eval()
    except Exception:
        pytest.skip("mamba-130m not in local HF cache")
    ids = tok("The meaning of the word 'calm'", return_tensors="pt")
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    raw = out.hidden_states
    L = model.config.num_hidden_layers
    can = canonical_hidden_states(model, out, ids["input_ids"])
    assert len(raw) == L + 1 == len(can)
    assert torch.allclose(can[0], model.get_input_embeddings()(ids["input_ids"]))
    for k in range(1, L):
        assert torch.equal(can[k], raw[k - 1])
    assert torch.equal(can[L], raw[L])
    assert torch.allclose(model.norm_f(raw[L - 1]), raw[L], atol=1e-5)


# ----------------------------- readout sparsity -----------------------------

def test_ridge_on_standardised_X_rotates_exactly():
    """W @ Q equals refitting on (z-scored X) @ Q, so the weight-rotation
    shortcut in sparsity_analysis.py stays exact with per-unit standardisation."""
    from sklearn.linear_model import Ridge
    from scipy.stats import special_ortho_group
    rng = np.random.default_rng(0)
    n, d = 400, 20
    X = rng.standard_normal((n, d)) * np.exp(rng.normal(0, 1.5, d))
    Y = rng.standard_normal((n, 3))
    Xz = (X - X.mean(0)) / X.std(0)
    Q = special_ortho_group.rvs(d, random_state=1)
    W = Ridge(alpha=1.0).fit(Xz, Y).coef_
    W_refit = Ridge(alpha=1.0).fit(Xz @ Q, Y).coef_
    assert np.allclose(W @ Q, W_refit, atol=1e-8)


def test_fit_ridge_weights_scale_invariant():
    """The native readout does not depend on per-unit scale."""
    pytest.importorskip("transformers")
    sys.path.insert(0, os.path.join(ROOT, "other"))
    from sparsity_analysis import fit_ridge_weights
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 16))
    Y = rng.standard_normal((300, 2))
    scale = np.exp(rng.normal(0, 2, 16))
    W1 = fit_ridge_weights(X, Y, is_clf=False)
    W2 = fit_ridge_weights(X * scale, Y, is_clf=False)
    assert np.allclose(W1, W2, atol=1e-8)
    pid = np.repeat(np.arange(150), 2)
    yb = np.tile([1, 0], 150)
    Wc = fit_ridge_weights(X, yb, is_clf=True, groups=pid)
    assert Wc.shape == (1, 16)
