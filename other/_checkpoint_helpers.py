"""
Shared helpers for the checkpoint-based training-dynamics analyses
(model_model_checkpoints.py and model_model_checkpoints_rsa.py).

Provides:
  - select_log_spaced_checkpoints: pick ~N steps from the 154 available Pythia checkpoints
  - load_pereira_sentences:        Exp2 + Exp3 unique sentences
  - depth_selected_layers:         4 layer indices at 25/50/75/100% depth
  - get_activations_no_cache:      load model at a revision, run inference, free GPU
  - evict_revision:                delete a checkpoint revision from the HF cache on disk
"""

import os
import gc

import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from pooling import pool_tokens, last_token_is_droppable
import torch
import huggingface_hub
from transformers import AutoTokenizer, AutoModel


# ---------------- CHECKPOINT SELECTION ----------------

# All checkpoints available for Pythia models on HuggingFace
_EARLY   = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
_REGULAR = list(range(1000, 144000, 1000))   # 1000, 2000, ..., 143000
ALL_STEPS = sorted(set(_EARLY + _REGULAR))   # 154 total


def select_log_spaced_checkpoints(n=50):
    """Return ~n log-spaced checkpoint steps (always includes 0 and 143000)."""
    targets = np.geomspace(1, 143000, max(n - 1, 1))
    selected = set()
    for t in targets:
        closest = min(ALL_STEPS, key=lambda s: abs(s - t))
        selected.add(closest)
    selected.add(0)
    selected.add(143000)
    return sorted(selected)


# ---------------- LAYER SELECTION ----------------

def depth_selected_layers(n_layers):
    """Return 4 layer indices at 25%, 50%, 75%, 100% depth.
    n_layers includes embedding (index 0); max index = n_layers - 1.
    """
    max_idx = n_layers - 1
    return [round(frac * max_idx) for frac in (0.25, 0.50, 0.75, 1.00)]


# ---------------- DATA LOADING ----------------

def load_pereira_sentences():
    """Load and combine Pereira Exp2 + Exp3 sentences (deduplicated, order preserved)."""
    from pereira_data import load_pereira
    data = load_pereira()
    net_sent = data["net_sent_orders"]
    ex2 = [s[1:-1] for s in net_sent["PereiraE2_96pass"]]
    ex3 = [s[1:-1] for s in net_sent["PereiraE3_72pass"]]
    seen = set()
    combined = []
    for s in ex2 + ex3:
        if s not in seen:
            seen.add(s)
            combined.append(s)
    print(f"  Pereira sentences: Exp2={len(ex2)}, Exp3={len(ex3)}, "
          f"combined unique={len(combined)}", flush=True)
    return combined


# ---------------- ACTIVATION EXTRACTION (no cache) ----------------

def _tensor_to_sentvec(hidden_state_tensor, drop_last=True):
    """(1, T, d) -> (d,): drops position 0 always, position T-1 only when `drop_last`.

    These are the Pereira sentences, which all end in a period, so `drop_last` is
    True throughout."""
    return pool_tokens(hidden_state_tensor[0], drop_last=drop_last)


def get_activations_no_cache(model_name, sentences, revision, device="cuda"):
    """
    Load model at a specific revision, run inference, return block_outputs dict.
    No activation cache files written. Caller is responsible for evicting weights via evict_revision.

    Returns
    -------
    block_outputs : dict {layer_idx: np.ndarray [n_sentences, d_model]}
                    layer 0 = embedding, layers 1..n = transformer blocks
    n_layers      : int (= len(block_outputs) = n_transformer_blocks + 1)
    """
    print(f"  [LOAD] {model_name} @ {revision}", flush=True)
    model = AutoModel.from_pretrained(
        model_name, revision=revision, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True
    )
    model.eval()

    tok_max = getattr(tokenizer, "model_max_length", None)
    has_reasonable_max = isinstance(tok_max, int) and (0 < tok_max < 1_000_000)

    def _encode(text):
        if has_reasonable_max:
            return tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=tok_max).to(device)
        return tokenizer(text, return_tensors="pt", truncation=False).to(device)

    block_outputs = None
    expected_len = None

    for s in sentences:
        tokens = _encode(s)
        drop_last = last_token_is_droppable(tokenizer, tokens["input_ids"])
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError(f"{model_name}@{revision} did not return hidden_states.")
        if expected_len is None:
            expected_len = len(hs)
            block_outputs = {i: [] for i in range(expected_len)}
        for i, layer_hs in enumerate(hs):
            block_outputs[i].append(_tensor_to_sentvec(layer_hs, drop_last))

    block_outputs = {i: np.stack(v) for i, v in block_outputs.items()}
    n_layers = len(block_outputs)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return block_outputs, n_layers


# ---------------- HF CACHE EVICTION ----------------

def evict_revision(model_name, revision):
    """
    Remove a specific checkpoint revision from the HF cache.

    Caller should skip step=143000 (it equals the 'main' branch and shares a commit
    hash, so deleting it would also remove the main snapshot).
    """
    hub_dir = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub",
    )
    if not os.path.isdir(hub_dir):
        return
    try:
        cache_info = huggingface_hub.scan_cache_dir(hub_dir)
    except Exception as e:
        print(f"  [WARN] Could not scan cache: {e}", flush=True)
        return

    hashes = [
        rev.commit_hash
        for repo in cache_info.repos
        if repo.repo_id == model_name
        for rev in repo.revisions
        if revision in rev.refs
    ]
    if not hashes:
        return
    try:
        strategy = cache_info.delete_revisions(*hashes)
        strategy.execute()
        print(f"  [EVICT] {revision} of {model_name} "
              f"({strategy.expected_freed_size_str})", flush=True)
    except Exception as e:
        print(f"  [WARN] Could not evict {revision} of {model_name}: {e}", flush=True)
