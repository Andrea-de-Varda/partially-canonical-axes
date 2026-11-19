import os, sys, argparse, pickle
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from pooling import POOLING_TAG
from metric_config import METRIC_TAG
from cache_utils import atomic_save_npy
from feature_data import load_norms, DATA_TAG
from layer_index import canonical_hidden_states, LAYER_TAG
import pandas as pd
import matplotlib.pyplot as plt
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


def _get_n_layers(cfg):
    return getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None) or getattr(cfg, "n_layers", None)


def get_all_layer_activations(model_name, sentences, device="cpu",
                              cache_dir="cached_activations", data_name="pereira",
                              span="sentence", verbose=False):
    os.makedirs(cache_dir, exist_ok=True)
    # Cached because JOINT_optimal_features.py reuses it. The
    # "_spanpool" tag marks that this pipeline pools ONLY the quoted-word span,
    # which already excludes the first/last sink positions (see pooling.py).
    # DATA_TAG / LAYER_TAG: word list and layer convention (layer_index.py) are
    # part of the name, so a cache built under a different one is never reused.
    fname = os.path.join(cache_dir, f"{model_name.replace('/', '_')}_all_layers_{data_name}_{span}_spanpool"
                                    f"_{DATA_TAG}_{LAYER_TAG}.npy")
    if os.path.exists(fname):
        if verbose:
            print(f"[LOADING] {fname}", flush=True)
        return np.load(fname, allow_pickle=True).item()

    if verbose:
        print(f"[COMPUTING] {model_name} all layers ({span})", flush = True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if "gemma" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    else:
        model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    n_layers = _get_n_layers(model.config)
    layer_outputs = {i: [] for i in range(n_layers + 1)}

    for s in tqdm(sentences):
        if span == "quoted":
            q1, q2 = s.index("'") + 1, s.index("'", s.index("'") + 1)
            span_start, span_end = q1, q2
        else:
            span_start = span_end = None

        tok = tokenizer(s, return_tensors="pt", truncation=True, return_offsets_mapping=True)
        offsets = tok.pop("offset_mapping")[0]
        tok = {k: v.to(device) for k, v in tok.items()}

        with torch.no_grad():
            outs = model(**tok, output_hidden_states=True)

        if span == "quoted":
            idx = [i for i, (st, en) in enumerate(offsets.tolist())
                   if st >= span_start and en <= span_end and (st, en) != (0, 0)]
        else:
            idx = [i for i, (st, en) in enumerate(offsets.tolist()) if (st, en) != (0, 0)]

        hs = canonical_hidden_states(model, outs, tok["input_ids"])
        for i, layer in enumerate(hs):
            vec = layer.squeeze(0)[idx].mean(dim=0).cpu().numpy()
            layer_outputs[i].append(vec)

    for i in layer_outputs:
        layer_outputs[i] = np.stack(layer_outputs[i])
    atomic_save_npy(fname, layer_outputs)
    return layer_outputs


def plot_seed_and_mean(all_scores, alphas=None, title="", save_path=None):
    n_seeds, n_rot, _ = all_scores.shape
    alphas = np.linspace(0, 1, n_rot) if alphas is None else alphas
    seed_means = np.nanmean(all_scores, axis=2)
    mean_curve = seed_means.mean(axis=0)
    sem_curve = seed_means.std(axis=0, ddof=1) / np.sqrt(n_seeds)

    plt.figure(figsize=(6, 3), dpi=300)
    for s in range(n_seeds):
        plt.plot(alphas, seed_means[s], marker='o', alpha=0.6)
    plt.errorbar(alphas, mean_curve, yerr=sem_curve, fmt='-ko', linewidth=2)
    plt.xticks(alphas)
    plt.xlabel("Rotation (α)")
    plt.ylabel("Mean Pairwise Alignment")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def remove_constant_columns(Y, tol=1e-5):
    stds = Y.std(axis=0)
    keep = stds > tol
    return Y[:, keep], keep


def alignment_worker(task_q, res_q, device_rank, ckpt_path):
    if torch.cuda.device_count() > device_rank:
        torch.cuda.set_device(device_rank)
    while True:
        try:
            task = task_q.get(timeout=30)
        except Exception:
            break
        layer, X, target, Y, tr, vr, te, seeds = task
        Y, _ = remove_constant_columns(Y)
        kwargs = dict(train_fract=tr, dev_fract=vr, test_fract=te)
        optimal = get_optimal_alignment(X, Y, num_epochs=50, **kwargs)
        mean_s, sem_s, _ = get_alignment_axis_rotation_multiseed(
            X, Y, n_seeds=seeds, return_all=True, **kwargs
        )
        result = {
            "target": target,
            "layer": layer,
            "mean": np.nanmean(mean_s, axis=1),
            "sem":  np.nanmean(sem_s,  axis=1),
            "optimal": optimal
        }
        res_q.put(result)
        with open(ckpt_path, "ab") as f:
            pickle.dump([result], f)
        print(f"[Layer {layer} | {target}] GPU {device_rank}", flush = True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", default="gpt2", type=str)
    parser.add_argument("--n_seeds", default=10, type=int)
    parser.add_argument("--n_workers_per_gpu", default=4, type=int)
    args = parser.parse_args()

    modelname = args.modelname
    alias = modelname.replace("/", "_")
    n_seeds = args.n_seeds
    n_workers_per_gpu = args.n_workers_per_gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Shared loader (feature_data.py): case-duplicates resolved, NA parsing off
    # -> 13,372 unique words.
    _words, sentences, Y_perceptual, Y_affective = load_norms()
    # ------------------------------------------------------------------

    print(f"[INFO] Extracting representations from {alias}", flush = True)
    embeddings_model = get_all_layer_activations(
        modelname, sentences, device=device, data_name="semantics",
        span="quoted", verbose=False
    )

    # POOLING_TAG / METRIC_TAG / DATA_TAG / LAYER_TAG in the name, as in LANG/JOINT,
    # so a checkpoint written under different settings is never resumed.
    checkpoint_path = (f"data/results/semantics_{alias}_{POOLING_TAG}_{METRIC_TAG}"
                       f"_{DATA_TAG}_{LAYER_TAG}_partial.pkl")
    os.makedirs("data/results", exist_ok=True)

    # resume
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
                    completed.add((r["layer"], r["target"]))

    tasks, task_q, res_q = [], Queue(), Queue()
    for layer in embeddings_model.keys():
        for target, Y, tr in [("perceptual", Y_perceptual, 0.6),
                              ("affective",  Y_affective,  0.6)]:
            if (layer, target) in completed:
                continue
            tasks.append((layer, embeddings_model[layer], target, Y, tr, 0.2, 0.2, n_seeds))
    for t in tasks:
        task_q.put(t)

    n_gpus = max(1, torch.cuda.device_count())
    procs = []
    for g in range(n_gpus):
        for _ in range(n_workers_per_gpu):
            p = Process(target=alignment_worker, args=(task_q, res_q, g, checkpoint_path))
            p.start()
            procs.append(p)
    for p in procs:
        p.join()

    results = partial
    while not res_q.empty():
        results.append(res_q.get())

    os.makedirs("plots/semantics", exist_ok=True)

    df_perceptual = pd.DataFrame([r for r in results if r["target"] == "perceptual"]).sort_values("layer")
    df_affective  = pd.DataFrame([r for r in results if r["target"] == "affective"]).sort_values("layer")

    df_perceptual.to_csv(f"data/results/perceptual_{alias}_semantic.csv", index=False)
    df_affective.to_csv(f"data/results/affective_{alias}_semantic.csv", index=False)

    layer_means = np.stack([r["mean"] for _, r in df_perceptual.iterrows()])
    plot_seed_and_mean(layer_means[:, :, None],
                       title="Layerwise, perceptual",
                       save_path=f"plots/semantics/layerwise_perceptual_{alias}.png")

    layer_means = np.stack([r["mean"] for _, r in df_affective.iterrows()])
    plot_seed_and_mean(layer_means[:, :, None],
                       title="Layerwise, affective",
                       save_path=f"plots/semantics/layerwise_affective_{alias}.png")


if __name__ == "__main__":
    mp.freeze_support()
    main()
