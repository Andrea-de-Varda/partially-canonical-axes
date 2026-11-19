"""
Canonical layer indexing for HuggingFace `hidden_states`.

Every analysis treats index 0 as the embedding layer (excluded downstream) and
index i >= 1 as the residual stream after block i, with the last index being the
output of the last block after the final norm. That is what HF returns for
GPT-2, GPT-NeoX/Pythia, Llama, Gemma-3 and Qwen3.

HF's MambaModel does NOT follow this convention (transformers 4.57.x,
modeling_mamba.py; verified at runtime): it appends a hidden state AFTER each block and then once
more after `norm_f`, so its tuple is
    [block 1, block 2, ..., block L (pre-norm), block L (post-norm)]
i.e. there is no embedding entry and the last block appears twice. Using it as
is made "exclude layer 0" drop block 1 and counted block L twice.

`canonical_hidden_states` rebuilds the Mamba tuple as
    [embedding, block 1, ..., block L-1, block L (post-norm)]
so that all architectures share one convention. `canonicalize_mamba_layers`
applies the same mapping to result tables computed from the raw Mamba tuple
(used by other/migrate_mamba_layers.py: the released Pereira Mamba results were
computed with the raw tuple and relabelled this way, so they have no layer-0 row;
a fresh run of PEREIRA_alignment_gpu.py produces canonical layers directly).
"""

import pandas as pd

# Goes into activation-cache filenames so caches built with the raw Mamba tuple
# are never reused.
LAYER_TAG = "canonL"


def is_mamba(model_or_config):
    cfg = getattr(model_or_config, "config", model_or_config)
    return getattr(cfg, "model_type", None) == "mamba"


def canonical_hidden_states(model, outputs, input_ids):
    """Return the hidden-state tuple in the canonical convention (see module doc)."""
    hs = outputs.hidden_states
    if not is_mamba(model):
        return hs
    n_blocks = model.config.num_hidden_layers
    if len(hs) != n_blocks + 1:
        raise RuntimeError(
            f"Unexpected Mamba hidden_states length {len(hs)} for {n_blocks} blocks; "
            f"the transformers version may use a different convention - check layer_index.py.")
    import torch
    with torch.no_grad():   # callers run the forward pass under no_grad; match it
        emb = model.get_input_embeddings()(input_ids)
        if not getattr(model, "_layer_convention_checked", False):
            # Verify the raw-tuple convention this mapping assumes, once per model,
            # so a transformers version with a different convention fails loudly
            # instead of silently mislabelling layers.
            first_is_emb = hs[0].shape == emb.shape and torch.allclose(hs[0], emb, atol=1e-5)
            last_is_norm = torch.allclose(model.norm_f(hs[-2]), hs[-1], atol=1e-4, rtol=1e-4)
            if first_is_emb or not last_is_norm:
                raise RuntimeError(
                    "Mamba hidden_states do not follow the convention assumed by layer_index.py "
                    f"(first_is_embedding={first_is_emb}, last_is_norm_of_previous={last_is_norm}). "
                    "Do not use these results; check the transformers version.")
            model._layer_convention_checked = True
    return (emb,) + tuple(hs[:-2]) + (hs[-1],)


def raw_to_canonical_mamba(raw_layer, n_raw):
    """Map a raw Mamba hidden_states index to its canonical index (None = drop).

    n_raw = number of raw entries (= n_blocks + 1). Raw r <= n_blocks-2 is block
    r+1; raw n_blocks-1 is the pre-norm duplicate of the last block (dropped);
    raw n_blocks is the last block post-norm (canonical n_blocks).
    """
    n_blocks = n_raw - 1
    if raw_layer <= n_blocks - 2:
        return raw_layer + 1
    if raw_layer == n_blocks - 1:
        return None
    if raw_layer == n_blocks:
        return n_blocks
    raise ValueError(f"raw layer {raw_layer} out of range for {n_raw} entries")


def canonicalize_mamba_layers(df, layer_col="layer", n_raw=None):
    """Relabel a per-layer result table computed from the raw Mamba tuple."""
    if n_raw is None:
        n_raw = int(df[layer_col].max()) + 1
    out = df.copy()
    out[layer_col] = [raw_to_canonical_mamba(int(l), n_raw) for l in out[layer_col]]
    out = out[out[layer_col].notna()].copy()
    out[layer_col] = out[layer_col].astype(int)
    return out
