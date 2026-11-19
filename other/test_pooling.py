"""Tests for the punctuation-conditional pooling rule (pooling.py)."""
import sys, os, json, glob, string
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
# Bound the thread pools BEFORE torch is imported (unbounded intra-op threads
# can oversubscribe a shared machine).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, torch
torch.set_num_threads(4)
from transformers import AutoTokenizer
from pooling import (POOLING_TAG, is_punct_token, last_token_is_droppable,
                     token_keep_mask, pool_tokens)

fails = []
def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + extra) if extra else ''}")
    if not cond: fails.append(name)

print(f"POOLING_TAG = {POOLING_TAG!r}\n")

print("1. is_punct_token")
for t in [".", ",", "!", "?", ";", ":", "Ġ.", "▁.", "...", "?!", "''", "’", "—", " ", ""]:
    check(f"   punct {t!r}", is_punct_token(t) is True)
for t in ["dog", "Ġdog", "▁dog", "ities", "Ġ3", "3", "a.b", "Ġelected"]:
    check(f"   content {t!r}", is_punct_token(t) is False)

print("\n2. token_keep_mask")
check("   drop_last=True,  n=5 -> keep 1..3", token_keep_mask(5, True).tolist()==[False,True,True,True,False])
check("   drop_last=False, n=5 -> keep 1..4", token_keep_mask(5, False).tolist()==[False,True,True,True,True])
check("   n=2 keeps all (guard, both ways)",
      token_keep_mask(2,True).tolist()==[True,True] and token_keep_mask(2,False).tolist()==[True,True])
check("   n=1 keeps all", token_keep_mask(1,True).tolist()==[True])
check("   n=3 drop_last=True keeps only middle", token_keep_mask(3,True).tolist()==[False,True,False])

print("\n3. pool_tokens honours drop_last")
h = np.arange(5*3, dtype=np.float32).reshape(5,3)
np.testing.assert_allclose(pool_tokens(h, True),  h[1:4].mean(0))
np.testing.assert_allclose(pool_tokens(h, False), h[1:5].mean(0))
check("   matches explicit slices", True)
np.testing.assert_allclose(pool_tokens(torch.tensor(h), False), h[1:5].mean(0))
check("   torch input agrees with numpy", True)

print("\n4. last_token_is_droppable on real tokenizers")
for name in ["gpt2", "EleutherAI/pythia-160m"]:
    tok = AutoTokenizer.from_pretrained(name)
    for text, want in [("The dog barked.", True), ("The dog barked", False),
                       (" mayors, and other elected", False),
                       ("First, there is one utetornity", False),
                       ("Is it true?", True), ("He said 'go'", True)]:
        ids = tok(text, return_tensors="pt")["input_ids"]
        got = last_token_is_droppable(tok, ids)
        check(f"   [{name}] {text!r} -> drop_last={got}", got == want, f"expected {want}")

print("\n5. Corpus-level consequence (gpt2 tokenizer)")
tok = AutoTokenizer.from_pretrained("gpt2")
def frac_drop(sents):
    return np.mean([last_token_is_droppable(tok, tok(s, return_tensors="pt")["input_ids"]) for s in sents])

blimp = []
for f in sorted(glob.glob("data/language_data/blimp_data/*.jsonl"))[:3]:
    L = [json.loads(l) for l in open(f)][:100]
    blimp += [x["sentence_good"] for x in L]
check("   BLiMP: 100% of finals droppable (rule is a no-op)", frac_drop(blimp) == 1.0,
      f"got {frac_drop(blimp):.1%}")

import pandas as pd
per = pd.read_csv("data/pereira_averaged_with_passage.csv")["Sentence"].astype(str).tolist()[:200]
check("   Pereira: 100% droppable (rule is a no-op)", frac_drop(per) == 1.0, f"got {frac_drop(per):.1%}")

for name, pre in [("wug", ""), ("hypernymy", "Consider ")]:
    L = [json.loads(l) for l in open(f"data/language_data/{name}_pairs.jsonl")][:200]
    sents = [pre + x["sentence_good"].strip() for x in L] + [pre + x["sentence_bad"].strip() for x in L]
    check(f"   {name}: 0% droppable (final token is content)", frac_drop(sents) == 0.0,
          f"got {frac_drop(sents):.1%}")

print("\n6. Content-final minimal pairs: good/bad do not pool identically")
from transformers import AutoModel
model = AutoModel.from_pretrained("gpt2").eval()
for name, pre in [("wug", ""), ("hypernymy", "Consider ")]:
    L = [json.loads(l) for l in open(f"data/language_data/{name}_pairs.jsonl")][:15]
    same_old = same_new = 0
    for x in L:
        g, b = pre + x["sentence_good"].strip(), pre + x["sentence_bad"].strip()
        vs = {}
        for tag, txt in (("g", g), ("b", b)):
            ids = tok(txt, return_tensors="pt")
            with torch.no_grad():
                hs = model(**ids, output_hidden_states=True).hidden_states[6].squeeze(0)
            dl = last_token_is_droppable(tok, ids["input_ids"])
            vs[tag] = (pool_tokens(hs, True), pool_tokens(hs, dl))
        if np.allclose(vs["g"][0], vs["b"][0], atol=1e-6): same_old += 1
        if np.allclose(vs["g"][1], vs["b"][1], atol=1e-6): same_new += 1
    check(f"   {name}: unconditional rule collapses {same_old}/15 pairs", same_old >= 12, f"{same_old}/15")
    check(f"   {name}: conditional rule collapses {same_new}/15 pairs", same_new == 0, f"{same_new}/15")

print("\n7. Bit-identity on punctuation-final text")
maxdiff = 0.0
for s in per[:10] + blimp[:10]:
    ids = tok(s, return_tensors="pt")
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states[6].squeeze(0)
    dl = last_token_is_droppable(tok, ids["input_ids"])
    maxdiff = max(maxdiff, np.abs(pool_tokens(hs, True) - pool_tokens(hs, dl)).max())
check("   Pereira+BLiMP vectors bit-identical to the unconditional rule", maxdiff == 0.0, f"max|diff|={maxdiff:g}")

print("\n" + ("ALL PASSED" if not fails else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)
