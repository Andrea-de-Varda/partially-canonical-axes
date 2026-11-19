import os, sys, argparse, json, pickle
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from pooling import pool_tokens, last_token_is_droppable, POOLING_TAG
from metric_config import METRIC_TAG
from cache_utils import atomic_save_npy
from feature_data import load_lang_tasks, pair_split, DATA_TAG
from layer_index import canonical_hidden_states, LAYER_TAG
import pandas as pd
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
from multiprocessing import Process, Queue

sys.path.append("./privileged_axes")
from privileged_axes.get_alignment_pairwise import get_alignment_axis_rotation_multiseed
from privileged_axes.get_optimal_pairwise import get_optimal_alignment


# ----------------------------- DATA LOADING -----------------------------

def load_task_data(data_dir="data/language_data"):
    """
    Minimal-pair tasks (hypernymy, wug, 12 BLiMP sub-tasks), loaded from
    feature_data.py (deduplicated). Returns list of dicts with keys:
    task, sentences (good + bad), Y (binary label column vector), pair_ids.
    """
    return load_lang_tasks(data_dir)


# ----------------------------- ACTIVATIONS -----------------------------

def _get_n_layers(cfg):
    return (getattr(cfg, "num_hidden_layers", None)
            or getattr(cfg, "n_layer", None)
            or getattr(cfg, "n_layers", None))


def lang_cache_path(model_name, cache_dir="cached_activations", data_name="lang_selectivity"):
    """Activation-cache filename; also read by other/sparsity_analysis.py.
    DATA_TAG / LAYER_TAG: the stimulus set and the layer convention
    (layer_index.py) are part of the name, so a cache built under a different
    one is never reused."""
    return os.path.join(
        cache_dir,
        f"{model_name.replace('/', '_')}_all_layers_{data_name}_sentence_"
        f"{POOLING_TAG}_{DATA_TAG}_{LAYER_TAG}.npy",
    )


def get_all_layer_activations(model_name, sentences, device="cpu",
                              cache_dir="cached_activations",
                              data_name="lang_selectivity"):
    """
    Extract mean-pooled hidden states at every layer for a list of sentences.
    Returns {layer_idx: np.ndarray (n_sentences, d_model)}.
    Caches to disk.
    """
    os.makedirs(cache_dir, exist_ok=True)
    fname = lang_cache_path(model_name, cache_dir, data_name)

    if os.path.exists(fname):
        print(f"  [CACHE] Loading {fname}", flush=True)
        return np.load(fname, allow_pickle=True).item()

    print(f"  [COMPUTE] {model_name} — {len(sentences)} sentences", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if "gemma" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    else:
        model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    n_layers = _get_n_layers(model.config)
    layer_outputs = {i: [] for i in range(n_layers + 1)}

    for s in tqdm(sentences, desc="Forward pass"):
        tok = tokenizer(s, return_tensors="pt", truncation=True,
                        return_offsets_mapping=True)
        offsets = tok.pop("offset_mapping")[0]
        tok = {k: v.to(device) for k, v in tok.items()}

        # Drop the final position only when it is punctuation or a special
        # token. The BLiMP sentences all end in a period, so they are unchanged;
        # the wug and hypernymy pairs differ ONLY in their final content token,
        # and dropping it made good and bad pool to the same vector. See
        # pooling.py.
        drop_last = last_token_is_droppable(tokenizer, tok["input_ids"])

        with torch.no_grad():
            outs = model(**tok, output_hidden_states=True)

        hs = canonical_hidden_states(model, outs, tok["input_ids"])
        for i, layer in enumerate(hs):
            layer_outputs[i].append(pool_tokens(layer.squeeze(0), drop_last=drop_last))

    for i in layer_outputs:
        layer_outputs[i] = np.stack(layer_outputs[i])

    atomic_save_npy(fname, layer_outputs)
    print(f"  [SAVED] {fname}", flush=True)
    return layer_outputs


# ----------------------------- WORKER ------------------------------------

def remove_constant_columns(Y, tol=1e-5):
    stds = Y.std(axis=0)
    keep = stds > tol
    return Y[:, keep], keep


def alignment_worker(task_q, device_rank, ckpt_path, n_epochs):
    if torch.cuda.device_count() > device_rank:
        torch.cuda.set_device(device_rank)
    while True:
        try:
            task = task_q.get(timeout=30)
        except Exception:
            break
        layer, X, task_name, Y, n_seeds, split = task
        Y, _ = remove_constant_columns(Y)
        # Pair-grouped 60/20/20 split (feature_data.pair_split), shared by the
        # native, random and optimal bases, so the two members of a minimal pair
        # never straddle the split.
        kwargs = dict(split_indices=split)

        optimal = get_optimal_alignment(X, Y, num_epochs=n_epochs, **kwargs)
        mean_s, sem_s, _ = get_alignment_axis_rotation_multiseed(
            X, Y, n_seeds=n_seeds, return_all=True, **kwargs
        )
        result = {
            "target": task_name,
            "layer": layer,
            "mean": np.nanmean(mean_s, axis=1),
            "sem":  np.nanmean(sem_s,  axis=1),
            "optimal": optimal,
        }
        with open(ckpt_path, "ab") as f:
            pickle.dump([result], f)

        native_r = result["mean"][0]
        opt_r = result["optimal"]
        ratio = native_r / opt_r if opt_r > 0 else float("nan")
        print(
            f"  [{task_name}] Layer {layer:2d} | "
            f"native r={native_r:.4f}  optimal={opt_r:.4f}  "
            f"ratio={ratio:.3f}",
            flush=True,
        )


# ----------------------------- MAIN -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Language acceptability alignment (correlation framework)"
    )
    parser.add_argument("--modelname", default="gpt2", type=str)
    parser.add_argument("--n_seeds", default=10, type=int)
    parser.add_argument("--n_workers_per_gpu", default=4, type=int)
    parser.add_argument("--n_epochs", default=50, type=int)
    args = parser.parse_args()

    modelname = args.modelname
    alias = modelname.replace("/", "_")
    n_seeds = args.n_seeds
    n_workers_per_gpu = args.n_workers_per_gpu
    n_epochs = args.n_epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: Load all tasks
    tasks = load_task_data()
    task_names = [t["task"] for t in tasks]
    print(f"{'='*60}", flush=True)
    print(f"  Language Acceptability Alignment — {alias}", flush=True)
    print(f"  Tasks ({len(tasks)}): {task_names}", flush=True)
    print(f"  Seeds: {n_seeds}, Workers/GPU: {n_workers_per_gpu}", flush=True)
    print(f"{'='*60}", flush=True)

    # Step 2: Deduplicate sentences, get activations once
    all_sentences = []
    sentence_to_idx = {}
    for task in tasks:
        for s in task["sentences"]:
            if s not in sentence_to_idx:
                sentence_to_idx[s] = len(all_sentences)
                all_sentences.append(s)
    print(f"\n  Total unique sentences: {len(all_sentences)}", flush=True)

    layer_acts = get_all_layer_activations(modelname, all_sentences, device=device)
    n_layers_plus1 = len(layer_acts)
    d_model = layer_acts[0].shape[1]
    print(f"  Layers: {n_layers_plus1} (0..{n_layers_plus1 - 1}), d_model: {d_model}\n",
          flush=True)

    # Step 3: Set up checkpointing
    os.makedirs("data/results", exist_ok=True)
    # POOLING_TAG in the name: results depend on how tokens are pooled, so a
    # checkpoint written under a different pooling rule must never be resumed.
    checkpoint_path = (f"data/results/lang_alignment_{alias}_{POOLING_TAG}_{METRIC_TAG}"
                       f"_{DATA_TAG}_{LAYER_TAG}_partial.pkl")

    completed = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            while True:
                try:
                    batch = pickle.load(f)
                except EOFError:
                    break
                for r in batch:
                    completed.add((r["layer"], r["target"]))
        print(f"  Resuming: {len(completed)} layer-task pairs already done\n", flush=True)

    # Step 4: Build task queue
    task_q = Queue()
    n_queued = 0
    for task in tasks:
        task_name = task["task"]
        idxs = [sentence_to_idx[s] for s in task["sentences"]]
        Y = task["Y"]
        split = pair_split(task["pair_ids"])

        for layer in range(n_layers_plus1):
            if (layer, task_name) in completed:
                continue
            X = layer_acts[layer][idxs]
            task_q.put((layer, X, task_name, Y, n_seeds, split))
            n_queued += 1

    print(f"  Queued {n_queued} jobs "
          f"({len(tasks)} tasks x {n_layers_plus1} layers - {len(completed)} done)\n",
          flush=True)

    if n_queued == 0:
        print("  Nothing to compute — all done!\n", flush=True)
    else:
        # Step 5: Launch workers
        n_gpus = max(1, torch.cuda.device_count())
        procs = []
        for g in range(n_gpus):
            for _ in range(n_workers_per_gpu):
                p = Process(target=alignment_worker,
                            args=(task_q, g, checkpoint_path, n_epochs))
                p.start()
                procs.append(p)
        for p in procs:
            p.join()

    # Step 6: Collect results from checkpoint pkl
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            while True:
                try:
                    batch = pickle.load(f)
                except EOFError:
                    break
                results.extend(batch)

    if not results:
        print("  [WARNING] No results collected!", flush=True)
        return

    # Step 7: Save per-task CSVs
    os.makedirs("plots/lang_alignment", exist_ok=True)
    print(f"\n{'='*60}", flush=True)
    print(f"  Results Summary — {alias}", flush=True)
    print(f"{'='*60}", flush=True)

    for task_name in task_names:
        task_results = [r for r in results if r["target"] == task_name]
        if not task_results:
            print(f"  [{task_name}] No results — skipped", flush=True)
            continue

        df = pd.DataFrame(task_results).sort_values("layer")
        output_csv = f"data/results/lang_{task_name}_{alias}.csv"
        df.to_csv(output_csv, index=False)

        # Summary: native alignment (alpha=0) and optimal, averaged across layers
        native_scores = np.array([r["mean"][0] for r in task_results])
        optimal_scores = np.array([r["optimal"] for r in task_results])
        ratios = native_scores / np.where(optimal_scores > 0, optimal_scores, np.nan)

        # Best layer
        best_layer_idx = np.argmax(native_scores)
        best_layer = task_results[best_layer_idx]["layer"]
        best_native = native_scores[best_layer_idx]
        best_optimal = optimal_scores[best_layer_idx]

        print(
            f"  [{task_name:<45s}] "
            f"mean native={np.nanmean(native_scores):.4f}  "
            f"mean optimal={np.nanmean(optimal_scores):.4f}  "
            f"mean ratio={np.nanmean(ratios):.3f}  |  "
            f"best layer={best_layer} (r={best_native:.4f}, opt={best_optimal:.4f})",
            flush=True,
        )

    print(f"\n  CSV files saved to data/results/lang_*_{alias}.csv", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
