import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import multiprocessing as mp

import argparse
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm  # used only in get_all_layer_activations
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from multiprocessing import Process, Queue
import queue
import signal

# ensure inner tqdm in alignment module is disabled to avoid multiple bars
sys.path.append("./privileged_axes")

from privileged_axes.get_alignment_pairwise import get_alignment_axis_rotation_multiseed, make_grouped_folds
from privileged_axes.get_optimal_pairwise import get_optimal_alignment, get_optimal_alignment_cv
from pooling import pool_tokens, last_token_is_droppable, POOLING_TAG
from layer_index import canonical_hidden_states, LAYER_TAG
from metric_config import MATCH_ABS, METRIC_TAG
from cache_utils import atomic_save_npy
import pereira_targets as PT
from pereira_data import load_pereira

# ---------------------------- UTILS ----------------------------
def load_passage_groups(experiment, sentences, csv_path="data/pereira_averaged_with_passage.csv"):
    """
    Return the passage_group label (e.g. 'PereiraE2_96pass|Banana') for each
    sentence, in the same order as `sentences`. Sentences are matched by text
    against data/pereira_averaged_with_passage.csv; raises if any sentence is
    missing so passage-level CV can never silently fall back.
    """
    df = pd.read_csv(csv_path)
    sub = df[df["experiment"] == experiment]
    mapping = dict(zip(sub["Sentence"], sub["passage_group"]))
    missing = [s for s in sentences if s not in mapping]
    if missing:
        raise ValueError(
            f"{len(missing)}/{len(sentences)} sentences not found in {csv_path} "
            f"for {experiment}; first missing: {missing[0]!r}"
        )
    if len(mapping) != len(sentences):
        raise ValueError(
            f"Sentence count mismatch for {experiment}: CSV has {len(mapping)} "
            f"unique sentences, pipeline has {len(sentences)}"
        )
    return np.array([mapping[s] for s in sentences])


def remove_constant_columns(Y, tol=1e-5):
    stds = Y.std(axis=0)
    keep = stds > tol
    return Y[:, keep], keep


def _get_transformer_core_and_blocks(model):
    # Blocks under `.h` (GPT-2 style) or `.layers` (LLaMA/Qwen/Gemma style)
    core = model
    if hasattr(model, "model") and (hasattr(model.model, "layers") or hasattr(model.model, "h")):
        core = model.model
    if hasattr(core, "transformer") and (hasattr(core.transformer, "h") or hasattr(core.transformer, "layers")):
        core = core.transformer
    if hasattr(core, "h"):
        return core, core.h
    if hasattr(core, "layers"):
        return core, core.layers
    raise AttributeError("Cannot find transformer blocks in the model.")


def _get_activation_modules(block, model_name):
    # Returns (pre_act_module, post_act_module): the linear layer BEFORE the activation function and the activation function's output.
    # The element-wise nonlinearity is what can break rotational symmetry (Khosla et al., 2024).
    model_name_lower = model_name.lower()

    # GPT-2 family: mlp.c_fc -> mlp.act -> mlp.c_proj
    if "gpt2" in model_name_lower or "gpt-2" in model_name_lower:
        if hasattr(block, "mlp") and hasattr(block.mlp, "c_fc") and hasattr(block.mlp, "act"):
            return block.mlp.c_fc, block.mlp.act
        raise AttributeError(f"GPT-2 block missing expected mlp.c_fc or mlp.act")

    # LLaMA family: mlp.gate_proj -> mlp.act_fn (SwiGLU architecture)
    if "llama" in model_name_lower:
        if hasattr(block, "mlp") and hasattr(block.mlp, "gate_proj") and hasattr(block.mlp, "act_fn"):
            return block.mlp.gate_proj, block.mlp.act_fn
        raise AttributeError(f"LLaMA block missing expected mlp.gate_proj or mlp.act_fn")

    # Qwen family: same as LLaMA (SwiGLU architecture)
    if "qwen" in model_name_lower:
        if hasattr(block, "mlp") and hasattr(block.mlp, "gate_proj") and hasattr(block.mlp, "act_fn"):
            return block.mlp.gate_proj, block.mlp.act_fn
        raise AttributeError(f"Qwen block missing expected mlp.gate_proj or mlp.act_fn")

    # Gemma family: same as LLaMA (SwiGLU architecture)
    if "gemma" in model_name_lower:
        if hasattr(block, "mlp") and hasattr(block.mlp, "gate_proj") and hasattr(block.mlp, "act_fn"):
            return block.mlp.gate_proj, block.mlp.act_fn
        raise AttributeError(f"Gemma block missing expected mlp.gate_proj or mlp.act_fn")

    # Mamba family: uses mixer.in_proj -> mixer.act (different architecture)
    if "mamba" in model_name_lower:
        if hasattr(block, "mixer") and hasattr(block.mixer, "in_proj") and hasattr(block.mixer, "act"):
            return block.mixer.in_proj, block.mixer.act
        raise AttributeError(f"Mamba block missing expected mixer.in_proj or mixer.act")

    # Fallback: try common patterns
    # Try GPT-2 style first
    if hasattr(block, "mlp"):
        if hasattr(block.mlp, "c_fc") and hasattr(block.mlp, "act"):
            return block.mlp.c_fc, block.mlp.act
        # Try LLaMA/Qwen style
        if hasattr(block.mlp, "gate_proj") and hasattr(block.mlp, "act_fn"):
            return block.mlp.gate_proj, block.mlp.act_fn

    raise AttributeError(
        f"Cannot determine activation modules for model '{model_name}'. "
        f"Block type: {type(block).__name__}. "
        f"Please add support for this model family in _get_activation_modules()."
    )    


def _tensor_to_sentvec(x: torch.Tensor, drop_last: bool = True) -> np.ndarray:
    """(1, T, d) / (T, d) per-token activations -> (d,) sentence vector via pooling.pool_tokens.

    Drops position 0 always and position T-1 only when `drop_last` (see pooling.py).
    Every Pereira sentence ends in a period, so `drop_last` is True throughout here."""
    if x is None:
        raise ValueError("Activation is None.")
    if not torch.is_tensor(x):
        raise TypeError(f"Expected tensor, got {type(x)}.")
    x = x.detach()
    if x.dim() == 3:
        x = x[0] if x.size(0) == 1 else x.mean(dim=0)
    elif x.dim() != 2:
        raise ValueError(f"Unexpected tensor shape: {x.shape}")
    return pool_tokens(x, drop_last=drop_last)


# Block outputs (residual stream) and, optionally, MLP pre/post-activation streams via hooks
def get_all_layer_activations(
    model_name,
    sentences,
    device="cpu",
    cache_dir="cached_activations",
    data_name="pereira",
    residual_only=False,
    trust_remote_code=True,
):
    """
    Returns:
        repr_streams: dict with keys:
            - "block_outputs": {layer_idx: np.ndarray [n_sentences, d_model]} for idx=0..n_layers
            - optionally "pre_act", "post_act": {layer_idx: np.ndarray [n_sentences, d_mlp]} for idx=0..n_layers-1
        fname: str path to the .npy cache file written/loaded

    Behavior:
        - residual_only=True:
            * DOES NOT call _get_transformer_core_and_blocks()
            * DOES NOT register hooks
            * Works for any HF model that returns hidden_states
        - residual_only=False:
            * Uses _get_transformer_core_and_blocks() + _get_activation_modules() hooks
    """
    os.makedirs(cache_dir, exist_ok=True)

    mode_suffix = "_residual_only" if residual_only else ""
    fname = os.path.join(
        cache_dir,
        # LAYER_TAG: caches hold canonical layers (layer_index.py), so a cache
        # built under a different layer convention is never reused.
        f"{model_name.replace('/', '_')}_repr_streams_v3_{POOLING_TAG}_{LAYER_TAG}_{data_name}{mode_suffix}.npy",
    )

    # ----------------- load cache (if exists) -----------------
    if os.path.exists(fname):
        print(f"[LOADING] {fname}", flush=True)
        data = np.load(fname, allow_pickle=True).item()
        if (not residual_only) and (("pre_act" not in data) or ("post_act" not in data)):
            raise RuntimeError(
                f"Cache at {fname} is missing pre_act/post_act despite residual_only=False. "
                f"Delete this cache file and rerun."
            )
        if ("block_outputs" not in data) or (not isinstance(data["block_outputs"], dict)):
            raise RuntimeError(f"Cache at {fname} is malformed (missing/invalid block_outputs). Delete and rerun.")
        return data, fname

    # ----------------- build model/tokenizer -----------------
    print(f"[COMPUTING] {model_name} repr streams (block outputs{' only' if residual_only else ', pre/post act'})", flush=True)
    if 'gemma' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False).to(device)
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model.eval()

    # Tokenization: avoid HF warning when tokenizer has no sensible max length
    # Many tokenizers use a sentinel huge int; treat that as "no truncation".
    tok_max = getattr(tokenizer, "model_max_length", None)
    has_reasonable_max = isinstance(tok_max, int) and (tok_max > 0) and (tok_max < 1_000_000)

    def _encode(text: str):
        if has_reasonable_max:
            return tokenizer(text, return_tensors="pt", truncation=True, max_length=tok_max).to(device)
        else:
            return tokenizer(text, return_tensors="pt", truncation=False).to(device)

    # ----------------- residual-only path (NO blocks, NO hooks) -----------------
    if residual_only:
        block_outputs = None
        expected_hs_len = None  # should become n_layers+1

        for s in tqdm(sentences):
            tokens = _encode(s)
            drop_last = last_token_is_droppable(tokenizer, tokens["input_ids"])
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True, return_dict=True)

            hs = (canonical_hidden_states(model, outputs, tokens["input_ids"])
                  if outputs.hidden_states is not None else None)
            if hs is None:
                raise RuntimeError(f"Model {model_name} did not return hidden_states (got None).")

            if expected_hs_len is None:
                expected_hs_len = len(hs)
                if expected_hs_len < 2:
                    raise RuntimeError(
                        f"Model {model_name} returned too few hidden_states ({expected_hs_len})."
                    )
                # initialize once we know n_layers from hs
                n_layers = expected_hs_len - 1
                block_outputs = {i: [] for i in range(n_layers + 1)}
            else:
                if len(hs) != expected_hs_len:
                    raise RuntimeError(
                        f"Inconsistent hidden_states length across sentences for {model_name}: "
                        f"expected {expected_hs_len}, got {len(hs)}. "
                        f"This usually indicates variable model behavior or a misconfigured forward pass."
                    )

            for i, layer in enumerate(hs):
                block_outputs[i].append(_tensor_to_sentvec(layer, drop_last))

        for i in block_outputs:
            block_outputs[i] = np.stack(block_outputs[i])

        repr_streams = {"block_outputs": block_outputs}
        atomic_save_npy(fname, repr_streams)
        return repr_streams, fname

    # ----------------- full path (blocks + pre/post hooks) -----------------
    core, blocks = _get_transformer_core_and_blocks(model)
    n_layers = len(blocks)

    block_outputs = {i: [] for i in range(n_layers + 1)}
    pre_act = {i: [] for i in range(n_layers)}
    post_act = {i: [] for i in range(n_layers)}

    current_pre_act = [None] * n_layers
    current_post_act = [None] * n_layers

    def make_pre_act_hook(layer_idx):
        def hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            current_pre_act[layer_idx] = output
        return hook

    def make_post_act_hook(layer_idx):
        def hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            current_post_act[layer_idx] = output
        return hook

    handles = []
    try:
        for i, block in enumerate(blocks):
            pre_act_module, post_act_module = _get_activation_modules(block, model_name)
            handles.append(pre_act_module.register_forward_hook(make_pre_act_hook(i)))
            handles.append(post_act_module.register_forward_hook(make_post_act_hook(i)))

        for s in tqdm(sentences):
            tokens = _encode(s)
            drop_last = last_token_is_droppable(tokenizer, tokens["input_ids"])

            current_pre_act[:] = [None] * n_layers
            current_post_act[:] = [None] * n_layers

            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True, return_dict=True)

            hs = (canonical_hidden_states(model, outputs, tokens["input_ids"])
                  if outputs.hidden_states is not None else None)
            if hs is None:
                raise RuntimeError(f"Model {model_name} did not return hidden_states (got None).")
            if len(hs) != (n_layers + 1):
                raise RuntimeError(
                    f"Unexpected number of hidden_states for {model_name}: got {len(hs)}, expected {n_layers + 1} "
                    f"(n_layers={n_layers})."
                )

            for i, layer in enumerate(hs):
                block_outputs[i].append(_tensor_to_sentvec(layer, drop_last))

            for layer_idx in range(n_layers):
                if current_pre_act[layer_idx] is None:
                    raise RuntimeError(f"Pre-activation hook did not capture for layer {layer_idx}.")
                if current_post_act[layer_idx] is None:
                    raise RuntimeError(f"Post-activation hook did not capture for layer {layer_idx}.")
                pre_act[layer_idx].append(_tensor_to_sentvec(current_pre_act[layer_idx], drop_last))
                post_act[layer_idx].append(_tensor_to_sentvec(current_post_act[layer_idx], drop_last))

    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    for i in block_outputs:
        block_outputs[i] = np.stack(block_outputs[i])
    for i in pre_act:
        pre_act[i] = np.stack(pre_act[i])
        post_act[i] = np.stack(post_act[i])

    repr_streams = {"block_outputs": block_outputs, "pre_act": pre_act, "post_act": post_act}
    atomic_save_npy(fname, repr_streams)
    return repr_streams, fname


def plot_layerwise(all_scores, alphas=None, title="", save_path=None, raw_scores=False):
    n_rot = all_scores.shape[1]
    if alphas is None:
        alphas = np.linspace(0, 1, n_rot)
    n_layers = all_scores.shape[0]
    mean_curve = all_scores.mean(axis=0)
    sem_curve  = all_scores.std(axis=0, ddof=1) / np.sqrt(n_layers)

    plt.figure(figsize=(4, 3), dpi=300)
    colors = cm.viridis(np.linspace(0, 1, n_layers))
    for s in range(n_layers):
        plt.plot(alphas, all_scores[s], marker='o', color=colors[s], alpha = 0.5, label=f"layer {s}")
    plt.errorbar(alphas, mean_curve, yerr=sem_curve, fmt='-ko', linewidth=2, label="mean ± SEM")
    plt.xticks(alphas)
    plt.xlabel("Rotation (α)")
    plt.ylabel("Pairwise alignment" if raw_scores else "Pairwise alignment/optimal")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def plot_repr_streams_comparison(layerwise_by_stream, alphas=None, title="", save_path=None, raw_scores=False):
    first = next(iter(layerwise_by_stream.values()))
    n_rot = first.shape[1]
    if alphas is None:
        alphas = np.linspace(0, 1, n_rot)
    plt.figure(figsize=(5, 4), dpi=300)
    for name, layerwise in layerwise_by_stream.items():
        mean_curve = layerwise.mean(axis=0)
        sem_curve  = layerwise.std(axis=0, ddof=1) / np.sqrt(layerwise.shape[0])
        plt.errorbar(alphas, mean_curve, yerr=sem_curve, fmt='-o', linewidth=2, label=name)
    plt.xticks(alphas)
    plt.xlabel("Rotation (α)")
    plt.ylabel("Pairwise alignment" if raw_scores else "Pairwise alignment/optimal")
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


# ---------------------------- WORKERS ----------------------------
# Cache for embeddings loaded by worker (to avoid reloading for each task)
_worker_embedding_cache = {}
_worker_brain_cache = {}

def alignment_worker(task_queue, result_queue, device_rank, checkpoint_path, n_epochs, stagger_idx=0):
    global _worker_embedding_cache
    # Stagger worker start-up: each worker's first task np.load()s the whole
    # activation cache file, so N workers starting together would spike host
    # RAM by ~N x filesize.
    if stagger_idx:
        import time
        time.sleep(15 * stagger_idx)
    if torch.cuda.device_count() > device_rank:
        torch.cuda.set_device(device_rank)
    # Cap CPU threads: torch defaults to the node's core count, not the job's
    # cgroup allocation, so N workers would oversubscribe a small allocation.
    try:
        _cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))
        _nw = max(1, int(os.environ.get("PER_ALIGN_N_WORKERS", "1")))
        torch.set_num_threads(max(1, _cpus // _nw))
    except Exception:
        torch.set_num_threads(1)
    while True:
        try:
            task = task_queue.get(timeout=30)  # wait up to 30s for tasks (each task takes ~seconds)
        except queue.Empty:
            break
        (layer, part_pos, part_idx, brain_path, modelname, alias, n_seeds, exp_name,
         cache_path, stream_key, folds) = task
        # A single failing task must not kill the worker: log it and move on.
        # The runner's retry rounds will re-queue it in a fresh process. (Note
        # this cannot catch kernel OOM kills — those still terminate the whole
        # process; the runner detects that case via the missing-task check.)
        try:
            # Load brain data from a per-experiment cache file (once per worker)
            # rather than shipping a ~12.6 MB copy of Y through the queue per task.
            if brain_path not in _worker_brain_cache:
                _worker_brain_cache[brain_path] = np.load(brain_path, allow_pickle=True).item()
            _bc = _worker_brain_cache[brain_path]
            if int(_bc["part"][part_pos]) != part_idx:
                raise RuntimeError(f"brain cache {brain_path}: position {part_pos} holds participant "
                                   f"{_bc['part'][part_pos]}, task expects {part_idx}")
            Y, _ = remove_constant_columns(_bc["Y"][part_pos])
            # Load embeddings from cache file (memory efficient - each worker loads once)
            if cache_path is None or stream_key is None:
                raise ValueError(f"cache_path and stream_key must be provided. Got cache_path={cache_path}, stream_key={stream_key}")
            cache_key = (cache_path, stream_key)
            if cache_key not in _worker_embedding_cache:
                data = np.load(cache_path, allow_pickle=True).item()
                _worker_embedding_cache[cache_key] = data[stream_key]
            X = _worker_embedding_cache[cache_key][layer]
            print(f"[{exp_name}-{alias}] GPU {device_rank}: Participant {part_idx}, Layer {layer}", flush = True)
            # Passage-level CV, TWO estimators for every quantity:
            #  (1) per-fold r averaged over the 10 folds  -> "mean", "optimal"
            #  (2) CV-concatenated r: held-out predictions from all folds are
            #      concatenated and ONE r per voxel is computed on all sentences
            #      -> "mean_concat", "optimal_concat".  (1) averages ten r's each
            #      estimated on 24-38 sentences (SE ~0.17-0.22 per voxel) and is
            #      dominated by noise; (2) has SE ~0.05-0.065.
            # One metric for all three bases: the optimal rotation gets the run's
            # metric too, so a --signed_match run is not scored against a |cos| ceiling.
            run_match_abs = os.environ.get("PER_ALIGN_MATCH_ABS", "1" if MATCH_ABS else "0") == "1"
            od = get_optimal_alignment_cv(X, Y, folds=folds, num_epochs=n_epochs, return_detail=True,
                                          match_abs=run_match_abs)
            opt = od["optimal"]
            mean_scores, sem_scores, all_scores, ex = get_alignment_axis_rotation_multiseed(
                X, Y, folds=folds, n_seeds=n_seeds, return_all=True, return_concat=True,
                match_abs=run_match_abs
            )
            mean_over_units = np.nanmean(mean_scores, axis=1)
            sem_over_units  = np.nanmean(sem_scores, axis=1)
            concat_over_units = np.nanmean(ex["concat_mean"], axis=1)                       # (n_rot,)
            concat_sem = np.nanstd(np.nanmean(ex["concat_all"], axis=2), axis=0, ddof=1) / np.sqrt(n_seeds)
            fold_means = np.nanmean(ex["fold_means"], axis=0)                                 # (n_folds, n_rot)
            ratio = mean_over_units / opt
            ratio_str = ", ".join(f"{r:.3f}" for r in ratio)
            print(f"[{exp_name}-{alias}] GPU {device_rank}: "
                  f"Participant {part_idx}, Layer {layer} | "
                  f"alignment/optimal: [{ratio_str}] | "
                  f"CONCAT native={concat_over_units[0]:.4f} random={concat_over_units[-1]:.4f} "
                  f"optimal={od['optimal_concat']:.4f}", flush = True)
            result = {"part": part_idx, "layer": layer,
                      "mean": mean_over_units, "sem": sem_over_units, "optimal": opt,
                      # estimator (2)
                      "mean_concat": concat_over_units, "sem_concat": concat_sem,
                      "optimal_concat": od["optimal_concat"],
                      # per-fold values (mean over voxels & seeds): shape (n_folds, n_rot) flattened row-major
                      "fold_means": fold_means.ravel(), "n_folds": fold_means.shape[0],
                      "optimal_folds": od["optimal_folds"]}
            # Voxel-level native-vs-rotated counts (alpha = 0 is the identity for
            # every seed, so the native score is seed-independent), for both estimators.
            n_ok, frac_fm, _ = PT.voxel_native_counts(all_scores[0, 0], all_scores[:, -1])
            _, frac_cc, all_cc = PT.voxel_native_counts(ex["concat_all"][0, 0], ex["concat_all"][:, -1])
            result.update({"vox_n": n_ok,
                           "vox_frac_native_gt_random": frac_fm,
                           "vox_frac_native_gt_random_concat": frac_cc,
                           "vox_frac_native_beats_all_random_concat": all_cc})
            result_queue.put(result)
            with open(checkpoint_path, "ab") as f:
                pickle.dump([result], f)
        except Exception as e:
            print(f"[WORKER-ERROR] {exp_name}-{alias} GPU {device_rank}: "
                  f"Participant {part_idx}, Layer {layer} failed: {type(e).__name__}: {e}", flush=True)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

# ---------------------------- RUNNER ----------------------------
# Stages that ended with missing (layer, participant) results despite retries.
# main() inspects this and exits nonzero so SLURM marks the task failed.
INCOMPLETE_STAGES = []


def _write_stage_csv(results, exp_name, alias, res_dir):
    """
    Materialise the stage CSV from whatever results exist right now.

    Called at stage start (from the checkpoint), periodically while workers
    run, on SIGTERM, and at stage end, so a walltime kill never loses the
    visible output of results that are already in the checkpoint pickle.
    """
    if not results:
        return 0
    df = pd.DataFrame(results)
    if df.empty or "layer" not in df.columns:
        return 0
    # last write wins per cell, so a retried task supersedes an earlier one
    df = df.drop_duplicates(subset=["layer", "part"], keep="last").sort_values(["layer", "part"])
    path = f"{res_dir}/{exp_name}_{alias}.csv"
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)   # atomic: never leave a half-written CSV behind
    return len(df)

def run_parallel_alignment(embeddings, brain_data, modelname, alias, n_seeds, n_workers_per_gpu, exp_name, n_epochs, return_layerwise=False, cache_path=None, stream_key=None, raw_scores=False, passage_groups=None, voxset=PT.MAIN_VOXSET):
    """
    brain_data: list of target dicts from pereira_targets.build_targets (one per
    kept participant; "part" is the original participant index and is what the
    results record). voxset selects the results directory and cache names.

    Run alignment in parallel. Workers load embeddings from the cache file
    (cache_path, stream_key; both required) rather than receiving them through
    the queue.

    If raw_scores=True, plot raw alignment scores instead of normalized by optimal.

    passage_groups (required): one passage label per sentence. CV is
    leave-passages-out — GroupKFold(10) for test folds, with a grouped 3:1
    train:dev split inside each fold (see make_grouped_folds).
    """
    if passage_groups is None:
        raise ValueError("passage_groups is required (passage-level CV)")
    folds = make_grouped_folds(passage_groups, n_splits=10, dev_fraction=0.25)
    fold_sizes = [(len(tr), len(dv), len(te)) for tr, dv, te in folds]
    print(f"[INFO] {exp_name}: {len(folds)} passage-level folds "
          f"(train/dev/test sizes: {fold_sizes[0]} ... {fold_sizes[-1]})", flush=True)
    # Checkpoint names carry the target version (PT.CKPT_VERSION), POOLING_TAG
    # and METRIC_TAG, so a checkpoint written under a different schema, pooling
    # rule or metric is never resumed. Runs that override the project metric
    # get their own alias so their checkpoint, CSV and plots never mix with
    # project results.
    _run_abs = os.environ.get("PER_ALIGN_MATCH_ABS", "1" if MATCH_ABS else "0") == "1"
    if _run_abs != MATCH_ABS:
        alias = f"{alias}_{'absmatch' if _run_abs else 'signed'}"
    # One results directory per voxel set (pereira_targets.py).
    res_dir = PT.results_dir(voxset)
    checkpoint_path = f"{res_dir}/{exp_name}_{alias}_{PT.CKPT_VERSION}_{POOLING_TAG}_{METRIC_TAG}_partial.pkl"
    os.environ["PER_ALIGN_N_WORKERS"] = str(n_workers_per_gpu)   # inherited by spawned workers
    os.makedirs(res_dir, exist_ok=True)
    part_ids = [int(t["part"]) for t in brain_data]
    completed = set()
    partial = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            while True:
                try:
                    batch = pickle.load(f)
                except EOFError:
                    break
                for r in batch:
                    partial.append(r)
                    completed.add((r["layer"], r["part"]))
    # Materialise whatever the checkpoint already holds BEFORE doing any work,
    # so results computed by a previously killed task become visible immediately.
    if partial:
        n = _write_stage_csv(partial, exp_name, alias, res_dir)
        print(f"[RESUME] {exp_name}: wrote {n} previously-computed results from checkpoint to CSV", flush=True)
    # Stage summary
    print(f"[INFO] {exp_name}: embeddings has {len(embeddings)} layers (keys: {sorted(embeddings.keys())})", flush=True)
    print(f"[INFO] {exp_name}: brain_data has {len(brain_data)} participants", flush=True)
    print(f"[INFO] {exp_name}: completed set has {len(completed)} items", flush=True)
    # Create tasks - workers load embeddings from cache file directly (memory efficient)
    if not cache_path or not stream_key:
        raise ValueError("cache_path and stream_key are required to avoid OOM issues")
    # One compact brain-data cache per (experiment, voxel set); tasks carry only
    # its path. The voxel set and target version are part of the name.
    os.makedirs("cached_activations", exist_ok=True)
    _expt_tag = "expt2" if "expt2" in exp_name else "expt3"
    brain_path = f"cached_activations/brain_{_expt_tag}_voxavg_{voxset}.npy"

    def _cache_matches(path):
        # Rebuilding costs ~1 s, so any mismatch with the freshly built targets
        # (a changed pereira_targets rule, a partial write) triggers a rewrite.
        try:
            c = np.load(path, allow_pickle=True).item()
            return (list(map(int, c["part"])) == part_ids and
                    all(np.array_equal(c["vox_idx"][i], t["vox_idx"]) and np.array_equal(c["Y"][i], t["Y"])
                        for i, t in enumerate(brain_data)))
        except Exception:
            return False
    if not (os.path.exists(brain_path) and _cache_matches(brain_path)):
        # Object arrays filled element-wise: np.array(list_of_2d, dtype=object)
        # tries to broadcast into a 3-D array when the leading dims match.
        _Y = np.empty(len(brain_data), dtype=object)
        _V = np.empty(len(brain_data), dtype=object)
        for _i, _t in enumerate(brain_data):
            _Y[_i] = _t["Y"]; _V[_i] = _t["vox_idx"]
        atomic_save_npy(brain_path, {"part": np.asarray(part_ids), "Y": _Y, "vox_idx": _V})
        print(f"[INFO] wrote brain cache {brain_path} ({len(brain_data)} participants)", flush=True)
    tasks = [
        (layer, pos, pid, brain_path, modelname, alias, n_seeds, exp_name, cache_path, stream_key, folds)
        for layer in embeddings for pos, pid in enumerate(part_ids)
        if (layer, pid) not in completed
    ]
    print(f"[INFO] {exp_name}: created {len(tasks)} tasks", flush=True)
    n_gpus = max(1, torch.cuda.device_count())
    results = partial.copy()
    done = set(completed)
    remaining = tasks
    # Workers can be killed silently by the kernel OOM killer. After the workers
    # exit, check which (layer, participant) tasks are still missing and respawn
    # fresh workers on just those, up to max_rounds. Anything still missing
    # afterwards is reported and makes the job exit nonzero (results stay
    # resumable via the checkpoint).
    max_rounds = 3
    for round_i in range(max_rounds):
        if not remaining:
            break
        if round_i > 0:
            print(f"[RETRY] {exp_name}: {len(remaining)} tasks unfinished — workers likely "
                  f"died (OOM-killed?); respawning fresh workers "
                  f"(round {round_i + 1}/{max_rounds})", flush=True)
        q_tasks = Queue()
        q_res = Queue()
        for t in remaining:
            q_tasks.put(t)
        procs = []
        widx = 0
        for gpu in range(n_gpus):
            for _ in range(n_workers_per_gpu):
                p = Process(target=alignment_worker, args=(q_tasks, q_res, gpu, checkpoint_path, n_epochs, widx))
                p.start()
                procs.append(p)
                widx += 1

        # IMPORTANT: drain q_res *while* the workers are still running, and join
        # only afterwards. Joining first deadlocks: a worker process cannot exit
        # until the results it queued have been flushed through the underlying
        # pipe, and that pipe blocks once enough results accumulate — so the
        # parent waits forever in join() while the child waits forever in flush().
        # SLURM sends SIGTERM before SIGKILL at walltime: flush the CSV first.
        def _on_sigterm(signum, frame):
            n = _write_stage_csv(results, exp_name, alias, res_dir)
            print(f"\n[SIGTERM] {exp_name}: walltime/cancel — flushed {n} results to CSV "
                  f"before exit; checkpoint is intact, just resubmit to resume.", flush=True)
            for pr in procs:
                try: pr.terminate()
                except Exception: pass
            sys.exit(143)
        _prev_term = signal.signal(signal.SIGTERM, _on_sigterm)

        since_flush = 0
        import time as _time
        t_start = _time.time(); t_last_result = t_start; t_last_beat = t_start
        n_round = 0
        watchdog_s = float(os.environ.get("PER_ALIGN_WATCHDOG_HOURS", "4")) * 3600
        while True:
            try:
                r = q_res.get(timeout=1)
                results.append(r)
                done.add((r["layer"], r["part"]))
                since_flush += 1; n_round += 1
                t_last_result = _time.time()
                # periodic flush so a hard kill loses at most 25 results of visibility
                if since_flush >= 25:
                    _write_stage_csv(results, exp_name, alias, res_dir)
                    since_flush = 0
            except queue.Empty:
                if not any(p.is_alive() for p in procs):
                    break
                now = _time.time()
                # heartbeat every 10 min, so a stalled round is visible in the log
                if now - t_last_beat >= 600:
                    alive = sum(p.is_alive() for p in procs)
                    print(f"[HEARTBEAT] {exp_name}: {alive}/{len(procs)} workers alive, "
                          f"{n_round} results this round, {len(done) - len(completed)} new total, "
                          f"{(now - t_last_result)/60:.0f} min since last result, "
                          f"{(now - t_start)/3600:.1f} h in round", flush=True)
                    t_last_beat = now
                # watchdog: a stall becomes a retry round
                if now - t_last_result > watchdog_s:
                    print(f"[WATCHDOG] {exp_name}: no result for {(now - t_last_result)/3600:.1f} h "
                          f"(> {watchdog_s/3600:.1f} h) -- terminating workers and retrying", flush=True)
                    for pr in procs:
                        try: pr.terminate()
                        except Exception: pass
                    break
        for p in procs:
            p.join()
        signal.signal(signal.SIGTERM, _prev_term)
        # final sweep: anything queued between the last get() and worker exit
        while True:
            try:
                r = q_res.get_nowait()
                results.append(r)
                done.add((r["layer"], r["part"]))
            except queue.Empty:
                break
        remaining = [t for t in remaining if (t[0], t[2]) not in done]

    if remaining:
        miss = sorted((t[0], t[2]) for t in remaining)
        print(f"[ERROR] {exp_name}: {len(miss)} (layer, participant) tasks STILL MISSING "
              f"after {max_rounds} rounds: {miss}", flush=True)
        print(f"[ERROR] {exp_name}: the CSV written below is INCOMPLETE and the job will "
              f"exit nonzero at the end — resubmit this array index to resume from the "
              f"checkpoint.", flush=True)
        INCOMPLETE_STAGES.append((exp_name, miss))
    else:
        print(f"[CHECK] {exp_name}: complete — {len(results)} results "
              f"({len(embeddings)} layers x {len(brain_data)} participants)", flush=True)

    df = pd.DataFrame(results)
    if df.empty or "layer" not in df.columns:
        print(f"[ERROR] {exp_name}: No results collected! Workers may have crashed (OOM?). "
              f"Try reducing --n_workers_per_gpu or check memory usage.", flush=True)
        # Return empty results to avoid crash
        if return_layerwise:
            return df, np.array([])
        return df
    _write_stage_csv(results, exp_name, alias, res_dir)
    df = df.drop_duplicates(subset=["layer", "part"], keep="last")
    layerwise = []
    for _, res in df.groupby("layer"):
        all_scores  = np.stack(res["mean"].to_numpy())
        if raw_scores:
            scores = all_scores  # raw alignment scores
        else:
            optimal = res["optimal"].to_numpy()[:, None]
            scores = (all_scores / optimal)  # normalized by optimal
        layerwise.append(scores.mean(axis=0))
    layerwise_arr = np.stack(layerwise)
    plot_layerwise(layerwise_arr, title=f"{exp_name}", save_path=f"plots/pereira/voxavg_{voxset}/{exp_name}_{alias}_layerwise.png", raw_scores=raw_scores)

    if return_layerwise:
        return df, layerwise_arr
    return df


# ---------------------------- MAIN ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", default="gpt2", type=str)
    parser.add_argument("--n_seeds", default=10, type=int)
    parser.add_argument("--n_workers_per_gpu", default=4, type=int)
    parser.add_argument("--n_epochs", default=50, type=int)
    parser.add_argument("--residual_only", action="store_true", help="Only run residual stream analysis (skip pre/post activation)")
    parser.add_argument("--normalize", action="store_true", help="Normalize scores by optimal alignment (default: raw scores)")
    # Splitting knobs: one SLURM array task per (model, experiment, stream),
    # so each stage fits in a single walltime.
    parser.add_argument("--only_expt", choices=["2", "3"], default=None,
                        help="Run only Pereira Exp 2 or Exp 3 (default: both)")
    parser.add_argument("--only_stream", choices=["block", "pre_act", "post_act"], default=None,
                        help="Run only one representation stream (default: all available)")
    parser.add_argument("--signed_match", action="store_true",
                        help="EXPERIMENTS ONLY: override the project metric (metric_config.MATCH_ABS=True, "
                             "axes matched up to orientation) with the original signed argmax of Khosla et al. "
                             "Results are aliased _signed so they never mix with project results.")
    parser.add_argument("--voxels", choices=list(PT.VOXSETS), default=PT.MAIN_VOXSET,
                        help="Brain target (pereira_targets.py): repetition-averaged voxels, either the 20%% most "
                             "reliable per participant (top20rel, main text) or all 1364 (all, SI). "
                             "Single-session participants are dropped in both.")
    parser.add_argument("--watchdog_hours", type=float, default=4.0,
                        help="Kill and respawn workers if no result arrives for this long (default 4h)")
    args = parser.parse_args()
    os.environ["PER_ALIGN_WATCHDOG_HOURS"] = str(args.watchdog_hours)
    os.environ["PER_ALIGN_MATCH_ABS"] = "0" if args.signed_match else ("1" if MATCH_ABS else "0")   # inherited by workers
    only_expt   = args.only_expt
    only_stream = args.only_stream
    def _want(expt, stream):
        return ((only_expt is None or only_expt == expt) and
                (only_stream is None or only_stream == stream))
    modelname = args.modelname
    alias = modelname.replace("/", "_")
    n_seeds = args.n_seeds
    n_workers_per_gpu = args.n_workers_per_gpu
    n_epochs = args.n_epochs
    residual_only = args.residual_only
    raw_scores = not args.normalize  # default is raw scores
    voxset = args.voxels
    print(f"[INFO] brain target: repetition-averaged voxels, set={voxset} -> {PT.results_dir(voxset)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if only_expt or only_stream:
        print(f"[INFO] SUBSET MODE: expt={only_expt or 'both'} stream={only_stream or 'all'}")
    if residual_only:
        print("[INFO] Running in residual-only mode (skipping pre/post activation analysis)")
    if raw_scores:
        print("[INFO] Plotting raw alignment scores")
    else:
        print("[INFO] Plotting scores normalized by optimal alignment")

    data = load_pereira()   # data/pereira_language.npz (see pereira_data.py)
    net_structs = data["net_structs"]
    net_sent = data["net_sent_orders"]

    # EXPT 2
    if only_expt == "3":
        print("[INFO] SUBSET MODE: skipping Exp. 2 entirely")
    else:
        print(f"[INFO] Extracting Exp. 2 representations from {alias}")
        ex2 = [s[1:-1] for s in net_sent['PereiraE2_96pass']]
        groups2 = load_passage_groups('PereiraE2_96pass', ex2)
        print(f"[INFO] Exp. 2: {len(set(groups2))} passages over {len(ex2)} sentences")
        repr_streams2, cache_path2 = get_all_layer_activations(modelname, ex2, device=device, data_name="pereira_expt2", residual_only=residual_only)
        bd2  = PT.build_targets(net_structs['PereiraE2_96pass'], voxset)
        print(f"[INFO] Exp. 2 targets: participants {[t['part'] for t in bd2]}, voxels {[t['Y'].shape[1] for t in bd2]}")

        lw_block2, lw_pre2, lw_post2 = None, None, None

        # full block outputs with residual
        if _want("2", "block"):
            print("Processing full block output (residual stream)")
            _, lw_block2 = run_parallel_alignment(repr_streams2["block_outputs"], bd2, modelname, alias, n_seeds, n_workers_per_gpu, exp_name="pereira_expt2", n_epochs = n_epochs, return_layerwise=True, cache_path=cache_path2, stream_key="block_outputs", raw_scores=raw_scores, passage_groups=groups2, voxset=voxset)

        if not residual_only and _want("2", "pre_act"):
            # pre-activation: output of first linear, BEFORE GeLU/SiLU
            print("Processing pre-activation (before GeLU/SiLU)")
            _, lw_pre2 = run_parallel_alignment(repr_streams2["pre_act"], bd2, modelname, alias, n_seeds, n_workers_per_gpu, exp_name="pereira_expt2_pre_act", n_epochs = n_epochs, return_layerwise=True, cache_path=cache_path2, stream_key="pre_act", raw_scores=raw_scores, passage_groups=groups2, voxset=voxset)
        if not residual_only and _want("2", "post_act"):
            # post-activation: output of GeLU/SiLU
            print("Processing post-activation (after GeLU/SiLU)")
            _, lw_post2 = run_parallel_alignment(repr_streams2["post_act"], bd2, modelname, alias, n_seeds, n_workers_per_gpu, exp_name="pereira_expt2_post_act", n_epochs = n_epochs, return_layerwise=True, cache_path=cache_path2, stream_key="post_act", raw_scores=raw_scores, passage_groups=groups2, voxset=voxset)

        # plot comparison (skip empty results)
        os.makedirs(f"plots/pereira/voxavg_{voxset}", exist_ok=True)
        streams2 = {k: v for k, v in {
            "Block output": lw_block2,
            "Pre-act (before GeLU/SiLU)": lw_pre2,
            "Post-act (after GeLU/SiLU)": lw_post2,
        }.items() if v is not None and len(v) > 0}
        if streams2 and only_stream is None:
            plot_repr_streams_comparison(streams2, title=f"Pereira Exp. 2 - {alias}",
                save_path=f"plots/pereira/voxavg_{voxset}/pereira_expt2_{alias}_repr_streams_comparison.png", raw_scores=raw_scores)
        else:
            print(f"[WARNING] No valid results for Exp. 2 comparison plot")

    # EXPT 3
    if only_expt == "2":
        print("[INFO] SUBSET MODE: skipping Exp. 3 entirely")
        if INCOMPLETE_STAGES:
            print(f"\n[FATAL] {alias}: {len(INCOMPLETE_STAGES)} stage(s) ended INCOMPLETE:", flush=True)
            for name, miss in INCOMPLETE_STAGES:
                print(f"  {name}: {len(miss)} missing: {miss}", flush=True)
            sys.exit(1)
        return
    print(f"[INFO] Extracting Exp. 3 representations from {alias}")
    ex3 = [s[1:-1] for s in net_sent['PereiraE3_72pass']]
    groups3 = load_passage_groups('PereiraE3_72pass', ex3)
    print(f"[INFO] Exp. 3: {len(set(groups3))} passages over {len(ex3)} sentences")
    repr_streams3, cache_path3 = get_all_layer_activations(modelname, ex3, device=device, data_name="pereira_expt3", residual_only=residual_only)

    bd3  = PT.build_targets(net_structs['PereiraE3_72pass'], voxset)
    print(f"[INFO] Exp. 3 targets: participants {[t['part'] for t in bd3]}, voxels {[t['Y'].shape[1] for t in bd3]}")

    lw_block3, lw_pre3, lw_post3 = None, None, None

    # full block outputs with residual
    if _want("3", "block"):
        _, lw_block3 = run_parallel_alignment(repr_streams3["block_outputs"], bd3, modelname, alias, n_seeds, n_workers_per_gpu, exp_name="pereira_expt3", n_epochs = n_epochs, return_layerwise=True, cache_path=cache_path3, stream_key="block_outputs", raw_scores=raw_scores, passage_groups=groups3, voxset=voxset)

    if not residual_only and _want("3", "pre_act"):
        # pre-activation: output of first linear, BEFORE GeLU/SiLU
        _, lw_pre3 = run_parallel_alignment(repr_streams3["pre_act"], bd3, modelname, alias, n_seeds, n_workers_per_gpu, exp_name="pereira_expt3_pre_act", n_epochs = n_epochs, return_layerwise=True, cache_path=cache_path3, stream_key="pre_act", raw_scores=raw_scores, passage_groups=groups3, voxset=voxset)
    if not residual_only and _want("3", "post_act"):
        # post-activation: output of GeLU/SiLU
        _, lw_post3 = run_parallel_alignment(repr_streams3["post_act"], bd3, modelname, alias, n_seeds, n_workers_per_gpu, exp_name="pereira_expt3_post_act", n_epochs = n_epochs, return_layerwise=True, cache_path=cache_path3, stream_key="post_act", raw_scores=raw_scores, passage_groups=groups3, voxset=voxset)

    # plot comparison (skip empty results)
    streams3 = {k: v for k, v in {
        "Block output": lw_block3,
        "Pre-act (before GeLU/SiLU)": lw_pre3,
        "Post-act (after GeLU/SiLU)": lw_post3,
    }.items() if v is not None and len(v) > 0}
    if streams3 and only_stream is None:
        plot_repr_streams_comparison(streams3, title=f"Pereira Exp. 3 - {alias}",
            save_path=f"plots/pereira/voxavg_{voxset}/pereira_expt3_{alias}_repr_streams_comparison.png", raw_scores=raw_scores)
    else:
        print(f"[WARNING] No valid results for Exp. 3 comparison plot")

    # Hard-fail if any stage ended with missing (layer, participant) tasks, so
    # an OOM-riddled run can never look like a clean completion in SLURM.
    if INCOMPLETE_STAGES:
        print(f"\n[FATAL] {alias}: {len(INCOMPLETE_STAGES)} stage(s) ended INCOMPLETE:", flush=True)
        for name, miss in INCOMPLETE_STAGES:
            print(f"  {name}: {len(miss)} missing (layer, participant) pairs: {miss}", flush=True)
        print("[FATAL] Resubmit this array index — checkpoints will resume exactly the missing tasks.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    mp.freeze_support()
    main()
