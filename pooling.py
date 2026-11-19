"""
Sentence-vector pooling shared by every activation-extraction pipeline.

RULE: mean over hidden states of all tokens EXCEPT the
first position, and except the last position WHEN THAT LAST TOKEN IS
PUNCTUATION OR A SPECIAL TOKEN.

Why drop the first position: a diagnostic on GPT-2, Pythia and Gemma showed that plain
all-token mean pooling is dominated by attention-sink tokens with massive norms
-- position 0 in every model (Gemma's <bos>; the first content token in GPT-2
and Pythia, whatever it is), and the sentence-final period in Pythia -- so that
in the middle layers the sentence matrix collapses to effective rank ~1-2 and
its top PC is 1/n_tokens: the vectors encoded sentence LENGTH, not content.
Dropping the sink positions restores rank 15-60 and r(PC1, 1/len) <= 0.4 in all
three tested models. A norm-threshold rule was rejected (it misses Gemma layers
2 and 20 where the sink is below 5x the median yet still dominates).

Why the last position is dropped only CONDITIONALLY: the sink that motivated dropping it
is the sentence-final *period*. Dropping the last position unconditionally
destroys any stimulus set whose final token carries the manipulation. That is
the case for the wug (morphology) and hypernymy (lexical semantics)
tasks: their minimal pairs differ ONLY in the final token, so under the
unconditional rule the good and bad member of every pair pool to the same
vector -- 1000/1000 identical for hypernymy, 842/1000 for wug (the remainder
differ only because "-ies" re-tokenizes). The label becomes unpredictable by
construction, and because the two members of a pair land on opposite sides of a
random split, held-out r goes reliably NEGATIVE (-0.06 to -0.11, identical for
native / random / optimal, and driven to -0.35 by the joint optimizer). A
duplicated-X simulation reproduces it at -0.085 +- 0.037.

The conditional rule is a no-op for punctuation-final stimuli: the Pereira
sentences and all 12 BLiMP paradigms end in punctuation 100% of the time, and
the SEM pipeline pools only the quoted-word span (tag "_spanpool") and never
uses this rule. Only wug and hypernymy are affected.

Every cache filename must carry POOLING_TAG so caches built under a different rule
can never be reused silently.
"""
import string
import unicodedata

import numpy as np
import torch

POOLING_TAG = "dropF_punctL"   # drop First; drop Last only if punctuation/special


def is_punct_token(tok_str):
    """
    True if a decoded token is punctuation (or pure whitespace / empty).

    Leading word-boundary markers are stripped first: BPE "Ġ" (GPT-2, Pythia),
    SentencePiece "▁" (Llama, Gemma, Qwen), and the ASCII space that some fast
    tokenizers emit directly. A token of several punctuation characters ("...",
    "?!", "''") counts as punctuation; a token with any letter or digit does not.
    """
    if tok_str is None:
        return False
    s = tok_str.replace("Ġ", " ").replace("▁", " ").strip()
    if not s:
        return True  # pure whitespace carries no content either
    return all(ch in string.punctuation or unicodedata.category(ch).startswith("P")
               for ch in s)


def last_token_is_droppable(tokenizer, input_ids):
    """
    Decide whether the FINAL token of a tokenized sequence should be dropped.

    Drop it when it is punctuation (the sentence-final-period sink) or a special
    token (</s>, <eos>, <pad>, ...). Keep it when it is a content token, which is
    the case that matters for the wug / hypernymy minimal pairs.

    input_ids: 1-D sequence of ids (list, np.ndarray, or torch.Tensor; a (1, T)
    batch is accepted and squeezed).
    """
    if torch.is_tensor(input_ids):
        input_ids = input_ids.detach().cpu().reshape(-1).tolist()
    else:
        input_ids = np.asarray(input_ids).reshape(-1).tolist()
    if not input_ids:
        return True
    last_id = int(input_ids[-1])

    special = set(getattr(tokenizer, "all_special_ids", None) or [])
    if last_id in special:
        return True

    tok_str = tokenizer.convert_ids_to_tokens(last_id)
    return is_punct_token(tok_str)


def token_keep_mask(n_tokens, drop_last=True):
    """
    Boolean mask of positions kept for pooling.

    Always drops position 0. Drops position T-1 as well when `drop_last`.
    Sequences of <= 2 tokens keep everything (dropping positions would leave
    nothing to average).
    """
    n_tokens = int(n_tokens)
    m = np.ones(n_tokens, dtype=bool)
    if n_tokens > 2:
        m[0] = False
        if drop_last:
            m[-1] = False
    return m


def pool_tokens(h, drop_last=True):
    """
    h: (T, d) torch tensor or ndarray of per-token hidden states -> (d,) float32.

    Mean over tokens 1..T-2 when `drop_last` (punctuation-final sequences), or
    1..T-1 when not (the final token carries content). T <= 2 falls back to all
    tokens. Callers get `drop_last` from `last_token_is_droppable(tokenizer, ids)`.
    """
    if torch.is_tensor(h):
        h = h.detach().float().cpu().numpy()
    h = np.asarray(h, dtype=np.float32)
    if h.ndim != 2:
        raise ValueError(f"pool_tokens expects (T, d), got {h.shape}")
    return h[token_keep_mask(h.shape[0], drop_last=drop_last)].mean(axis=0)
