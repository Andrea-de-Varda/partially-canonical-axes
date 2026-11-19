import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from glob import glob
import os
import pereira_targets as _PT

# Pereira results live in one directory per brain-target voxel set (see
# pereira_targets.py). PEREIRA_VOXSET=all selects the SI set; default is the
# main-text set (20% most reliable repetition-averaged voxels).
PEREIRA_VOXSET = os.environ.get("PEREIRA_VOXSET", _PT.MAIN_VOXSET)
PEREIRA_RES = _PT.results_dir(PEREIRA_VOXSET)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 150,
})

model_info = {
    "gpt2":                        ("GPT-2", r"GPT-2",               "o"),
    "gpt2-medium":                 ("GPT-2", r"GPT-2$_{\mathrm{M}}$",     "s"),
    "gpt2-large":                  ("GPT-2", r"GPT-2$_{\mathrm{L}}$",     "v"),
    "gpt2-xl":                     ("GPT-2", r"GPT-2$_{\mathrm{XL}}$",    "D"),
    "meta-llama_Llama-3.2-1B":     ("LLaMA", r"LLaMA 3.2$_{\mathrm{1B}}$", "o"),
    "meta-llama_Llama-3.2-3B":     ("LLaMA", r"LLaMA 3.2$_{\mathrm{3B}}$", "s"),
    "google_gemma-3-1b-pt":        ("Gemma",  r"Gemma 3$_{\mathrm{1B}}$",   "o"),
    "state-spaces_mamba-130m-hf":  ("Mamba",    r"Mamba$_{\mathrm{130M}}$",    "o"),
    "state-spaces_mamba-370m-hf":  ("Mamba",    r"Mamba$_{\mathrm{370M}}$",    "s"),
    "state-spaces_mamba-790m-hf":  ("Mamba",    r"Mamba$_{\mathrm{790M}}$",    "v"),
    "Qwen_Qwen3-0.6B-Base":        ("Qwen",     r"Qwen 3$_{\mathrm{0.6B}}$",   "o"),
    "EleutherAI_pythia-160m":      ("Pythia",   r"Pythia$_{\mathrm{160M}}$",   "o"),
    "EleutherAI_pythia-160m-seed2":      ("Pythia",   r"Pythia$_{\mathrm{160M-Seed2}}$",   "o"),
}

model_order = [
    "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
    "meta-llama_Llama-3.2-1B",
    "meta-llama_Llama-3.2-3B",
    "google_gemma-3-1b-pt",
    "state-spaces_mamba-130m-hf", "state-spaces_mamba-370m-hf",
    "state-spaces_mamba-790m-hf",
    "Qwen_Qwen3-0.6B-Base",
    "EleutherAI_pythia-160m",
]

family_colors = {
    "GPT-2":  "#4477AA",   # blue
    "LLaMA":  "#EE6677",   # red / coral
    "Gemma":  "#AA3377",   # magenta / purple
    "Mamba":  "#228833",   # green
    "Qwen":   "#CCBB44",   # yellow / gold
    "Pythia": "#66CCEE",   # cyan
}

family_colors_light = {k: c + "88" for k, c in family_colors.items()}

def extract_key(path, to_replace="pereira_expt2_", to_replace_2=""):
    return (path.split("/")[-1]
            .replace(to_replace, "")
            .replace(to_replace_2, "")
            .replace(".csv", ""))


def _load_csv(path):
    """Load a results CSV and return (model_key, layerwise_array, n_layers)."""
    res = pd.read_csv(path)
    res["mean"] = res["mean"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    res["sem"]  = res["sem"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    return res


def _build_layerwise(res, normalize=True, skip_layer0=True):
    """
    From a loaded dataframe, build (layerwise_array, layer_indices).

    Normalisation uses a RATIO OF MEANS: within each layer the alignment and
    the optimal score are each averaged across participants FIRST, and only
    then divided. A mean of ratios is unstable under passage-level CV, where the
    learned optimal rotation can fail to generalise: `optimal` is near zero or
    negative in some (layer, participant) cells, and per-cell ratios then blow up
    and dominate the average.
    """
    if skip_layer0:
        res = res[res["layer"] > 0]
    layerwise = []
    layers = []
    for layer_id, df in res.groupby("layer"):
        all_scores = np.stack(df["mean"].to_numpy())
        if normalize:
            num = np.nanmean(all_scores, axis=0)             # mean alignment per alpha
            den = np.nanmean(df["optimal"].to_numpy())        # mean optimal, same cells
            scores_row = num / den if np.isfinite(den) and abs(den) > 1e-8 else np.full_like(num, np.nan)
        else:
            scores_row = np.nanmean(all_scores, axis=0)
        layerwise.append(scores_row)
        layers.append(layer_id)
    return np.stack(layerwise), np.array(layers)


def _is_pre_post_act(path):
    """Return True if path corresponds to pre_act or post_act results."""
    basename = os.path.basename(path)
    return "_pre_act_" in basename or "_post_act_" in basename

def plot_all(datasets, plot_title="All models", to_replace="pereira_expt2_",
             to_replace_2="", normalize=True, save_path=None):
    """
    Main rotation-alignment plot. Excludes pre_act / post_act files.
    """
    # Filter out pre/post act; sort so _semantic files come first (preferred)
    datasets = sorted(
        [d for d in datasets if not _is_pre_post_act(d)],
        key=lambda p: (0 if p.endswith("_semantic.csv") else 1),
    )

    model_keys = []
    model_avgs = []
    seen_keys = set()

    for model_path in datasets:
        try:
            model_key = extract_key(model_path, to_replace=to_replace,
                                    to_replace_2=to_replace_2)
            if model_key not in model_info:
                print(f"  skipping {model_path}: unknown model key '{model_key}'")
                continue
            if model_key in seen_keys:
                print(f"  skipping {model_path}: duplicate model key '{model_key}'")
                continue
            seen_keys.add(model_key)
            res = _load_csv(model_path)
            layerwise, _ = _build_layerwise(res, normalize=normalize)
            model_avg = layerwise.mean(axis=0)  # average across layers
            model_keys.append(model_key)
            model_avgs.append(model_avg)
        except Exception as e:
            print(f"  skipping {model_path}: {e}")

    if not model_keys:
        print(f"  No data for '{plot_title}'")
        return

    ordered = sorted(zip(model_keys, model_avgs),
                     key=lambda x: model_order.index(x[0]) if x[0] in model_order else 1e9)
    model_keys, model_avgs = zip(*ordered)
    modelwise = np.vstack(model_avgs)
    alphas = np.linspace(0, 1, modelwise.shape[1])

    fig, ax = plt.subplots(figsize=(4.8*.8, 3.2*.8))
    for key, row in zip(model_keys, modelwise):
        family, label, marker = model_info[key]
        ax.plot(alphas, row, label=label, marker=marker,
                color=family_colors[family], linewidth=1.8, markersize=4)

    ax.set_xlabel(r"Rotation fraction $\alpha$")
    ax.set_ylabel("Alignment / optimal" if normalize else "Pairwise alignment (r)")
    ax.set_title(plot_title)
    ax.grid(True, linestyle="--")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_pre_post_act(expt_prefix="pereira_expt2", results_dir=PEREIRA_RES,
                      normalize=True, save_path=None):
    """
    For each model that has pre_act and post_act data, plot three lines
    (pre-act, residual, post-act) in a single figure. One subplot per model.
    """
    # Discover models that have all three streams
    pre_files = sorted(glob(f"{results_dir}/{expt_prefix}_pre_act_*.csv"))
    post_files = sorted(glob(f"{results_dir}/{expt_prefix}_post_act_*.csv"))

    def _model_from_stream(path, stream):
        basename = os.path.basename(path).replace(".csv", "")
        return basename.replace(f"{expt_prefix}_{stream}_", "")

    pre_models = {_model_from_stream(f, "pre_act"): f for f in pre_files}
    post_models = {_model_from_stream(f, "post_act"): f for f in post_files}
    # Residual files
    all_residual = glob(f"{results_dir}/{expt_prefix}_*.csv")
    residual_models = {}
    for f in all_residual:
        if _is_pre_post_act(f):
            continue
        key = extract_key(f, to_replace=f"{expt_prefix}_")
        residual_models[key] = f

    # Include models that have residual + at least one of pre/post
    # Qwen was run residual-only (no pre/post-activation results)
    _exclude_pre_post = {"Qwen_Qwen3-0.6B-Base"}
    candidates = sorted(
        (set(residual_models) & (set(pre_models) | set(post_models))) - _exclude_pre_post,
        key=lambda x: model_order.index(x) if x in model_order else 1e9)

    if not candidates:
        print(f"  No models with pre/post act streams for {expt_prefix}")
        return

    n = len(candidates)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n * .8, 3.0*.8), sharey=True, squeeze=False)
    axes = axes[0]

    stream_styles = {
        "Pre-act":   {"linestyle": "--", "alpha": 0.85},
        "Residual":  {"linestyle": "-",  "alpha": 1.0},
        "Post-act":  {"linestyle": ":",  "alpha": 0.85},
    }

    for idx, model_key in enumerate(candidates):
        ax = axes[idx]
        family, label, marker = model_info.get(model_key, ("?", model_key, "o"))
        base_color = family_colors.get(family, "#666666")

        streams = [("Residual", residual_models[model_key])]
        if model_key in pre_models:
            streams.insert(0, ("Pre-act", pre_models[model_key]))
        if model_key in post_models:
            streams.append(("Post-act", post_models[model_key]))

        for stream_name, filepath in streams:
            res = _load_csv(filepath)
            layerwise, _ = _build_layerwise(res, normalize=normalize)
            avg = layerwise.mean(axis=0)
            alphas = np.linspace(0, 1, len(avg))
            style = stream_styles[stream_name]
            ax.plot(alphas, avg, label=stream_name, marker=marker,
                    color=base_color, markersize=3.5, linewidth=1.6, **style)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel(r"Rotation fraction $\alpha$")
        if idx == 0:
            ax.set_ylabel("Alignment / optimal" if normalize else "Pairwise alignment (r)")
        ax.grid(True, linestyle="--")
        if idx == 0:
            ax.legend(fontsize=7, frameon=False, loc="best")

    title = expt_prefix.replace("_", " ").replace("pereira ", "Pereira ").title()
    fig.suptitle(f"{title} — Pre/Post Activation Comparison", fontsize=11, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved {save_path}")
    else:
        plt.show()
    plt.close(fig)



def plot_sparsity_kurtosis(kurtosis_files, normalize=True,
                           plot_title="Lifetime Sparseness", save_path=None):
    """
    Plot kurtosis vs. rotation amount (α) for each model.
    kurtosis_files: list of data/results/sparsity_kurtosis_{alias}.csv paths.
    Averages across layers. Normalizes to native=1.0 if normalize=True.
    """
    entries = []
    for path in kurtosis_files:
        fname = os.path.basename(path).replace(".csv", "")
        model_key = None
        for mk in sorted(model_info.keys(), key=len, reverse=True):
            if fname == f"sparsity_kurtosis_{mk}":
                model_key = mk
                break
        if model_key is None:
            print(f"  skipping {path}: unknown model key")
            continue
        try:
            df = pd.read_csv(path)
            native_val = df[df["path"] == "native"]["kurtosis"].mean()
            random_by_alpha = df[df["path"] == "random"].groupby("alpha")["kurtosis"].mean()
            alphas = [0.0] + sorted(random_by_alpha.index)
            values = [native_val] + [random_by_alpha[a] for a in sorted(random_by_alpha.index)]
            if normalize and native_val > 0:
                values = [v / native_val for v in values]
            entries.append((model_key, alphas, values))
        except Exception as e:
            print(f"  skipping {path}: {e}")

    entries.sort(key=lambda x: model_order.index(x[0]) if x[0] in model_order else 1e9)
    if not entries:
        print(f"  No data for '{plot_title}'")
        return

    fig, ax = plt.subplots(figsize=(4.8 * .8, 3.2 * .8))
    for model_key, alphas, values in entries:
        family, label, marker = model_info[model_key]
        color = family_colors.get(family, "#666666")
        ax.plot(alphas, values, label=label, color=color,
                marker=marker, markersize=3, linewidth=1.2)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)
    ax.set_xlabel(r"Rotation fraction $\alpha$")
    ax.set_ylabel("Kurtosis (norm. to native)" if normalize else "Kurtosis")
    ax.set_title(plot_title)
    ax.grid(True, linestyle="--")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_sparsity_readout(readout_files, normalize=True,
                          plot_title="Readout Weight Sparsity", save_path=None):
    """
    Plot readout weight kurtosis vs. rotation amount, averaged across tasks.
    readout_files: list of data/results/sparsity_readout_{task}_{alias}.csv paths.
    Grouped and averaged per model alias.
    """
    from collections import defaultdict
    model_task_curves = defaultdict(list)

    for path in readout_files:
        fname = os.path.basename(path).replace(".csv", "")
        model_key = None
        for mk in sorted(model_info.keys(), key=len, reverse=True):
            if fname.endswith("_" + mk):
                model_key = mk
                break
        if model_key is None:
            print(f"  skipping {path}: unknown model key")
            continue
        try:
            df = pd.read_csv(path)
            native_val = df[df["path"] == "native"]["kurtosis_weights"].mean()
            random_by_alpha = (df[df["path"] == "random"]
                               .groupby("alpha")["kurtosis_weights"].mean())
            alphas = [0.0] + sorted(random_by_alpha.index)
            values = [native_val] + [random_by_alpha[a] for a in sorted(random_by_alpha.index)]
            model_task_curves[model_key].append((alphas, values))
        except Exception as e:
            print(f"  skipping {path}: {e}")

    entries = []
    for model_key, task_curves in model_task_curves.items():
        alphas = task_curves[0][0]
        all_vals = []
        for _, vals in task_curves:
            v = np.array(vals)
            if normalize:
                if v[0] > 0:
                    all_vals.append(v / v[0])
                # skip tasks with non-positive native kurtosis from normalized average
            else:
                all_vals.append(v)
        if not all_vals:
            continue
        avg_vals = np.mean(all_vals, axis=0).tolist()
        entries.append((model_key, alphas, avg_vals))
    entries.sort(key=lambda x: model_order.index(x[0]) if x[0] in model_order else 1e9)
    if not entries:
        print(f"  No data for '{plot_title}'")
        return

    fig, ax = plt.subplots(figsize=(4.8 * .8, 3.2 * .8))
    for model_key, alphas, values in entries:
        family, label, marker = model_info[model_key]
        color = family_colors.get(family, "#666666")
        ax.plot(alphas, values, label=label, color=color,
                marker=marker, markersize=3, linewidth=1.2)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)
    ax.set_xlabel(r"Rotation fraction $\alpha$")
    ax.set_ylabel("Readout kurtosis")
    ax.set_title(plot_title)
    ax.grid(True, linestyle="--")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Readout weight sparsity, reported against its random-rotation null
# ---------------------------------------------------------------------------
# Ridge is exactly rotation-equivariant, so the rotated readout is the native
# readout rotated; its sample excess kurtosis under a uniformly random rotation
# has expectation -6/(d+1) and SD ~sqrt(24/d), regardless of the native vector.
# The alpha = 1 values therefore ARE the null, and the informative quantity is
# the native kurtosis relative to it. These two figures show exactly that,
# instead of normalising every curve by a native value that may itself be noise.

READOUT_SEM_TASK = "sem"   # multi-output ridge over the 9 perceptual/affective norms


def _readout_cells(readout_files):
    """
    Tidy per-cell table from sparsity_readout_{task}_{alias}.csv files.

    One row per (model, task, layer): `native` kurtosis, `null_mean` / `null_sd`
    (mean and SD over rotation seeds at alpha = 1, i.e. the empirical null),
    `null_seeds` ({seed: kurtosis at alpha = 1}, needed to build the null of any
    statistic pooled across cells), and `curve`, the seed-averaged kurtosis at
    every alpha (index 0 = native).
    """
    rows = []
    keys = sorted(model_info.keys(), key=len, reverse=True)
    for path in readout_files:
        fname = os.path.basename(path).replace(".csv", "")
        model_key = next((mk for mk in keys if fname.endswith("_" + mk)), None)
        if model_key is None:
            continue
        task = fname[len("sparsity_readout_"):-(len(model_key) + 1)]
        df = pd.read_csv(path)
        nat = df[df["path"] == "native"].groupby("layer")["kurtosis_weights"].mean()
        rnd = df[df["path"] == "random"]
        by_alpha = rnd.groupby(["layer", "alpha"])["kurtosis_weights"].mean().unstack("alpha")
        full = rnd[np.isclose(rnd["alpha"], 1.0)]
        per_seed = full.groupby(["layer", "seed"])["kurtosis_weights"].mean()
        null_mean = full.groupby("layer")["kurtosis_weights"].mean()
        null_sd = full.groupby("layer")["kurtosis_weights"].std(ddof=1)
        alphas = [0.0] + sorted(by_alpha.columns)
        for layer in nat.index:
            if layer == 0:   # embedding layer, excluded as in every other analysis
                continue
            if layer not in by_alpha.index:
                continue
            curve = [nat[layer]] + [by_alpha.loc[layer, a] for a in sorted(by_alpha.columns)]
            rows.append(dict(model=model_key, task=task, layer=layer,
                             native=nat[layer], null_mean=null_mean[layer],
                             null_sd=null_sd[layer], null_seeds=per_seed.loc[layer].to_dict(),
                             curve=np.asarray(curve, float),
                             alphas=np.asarray(alphas, float)))
    return pd.DataFrame(rows)


def _readout_pooled(g):
    """
    Median native kurtosis over a set of cells, tested against the null of that
    same statistic: for each rotation seed, the median over the same cells of the
    alpha = 1 kurtosis. Mean and SD across seeds give the band.

    Comparing a median over many cells to the spread of a *single* cell would
    be the wrong null: it is ~sqrt(n) too wide.
    """
    seeds = sorted(set.intersection(*(set(d) for d in g["null_seeds"])))
    null_medians = np.array([np.median([d[s] for d in g["null_seeds"]]) for s in seeds])
    nat, mu = g["native"].median(), null_medians.mean()
    sd = null_medians.std(ddof=1)
    return dict(native=nat, null_mean=mu, null_sd=sd, lo=mu - 1.96 * sd, hi=mu + 1.96 * sd,
                significant=bool(nat > mu + 1.96 * sd), n_cells=len(g), n_seeds=len(seeds))


def _readout_summary(cells):
    """Per model: `_readout_pooled` over all its cells (tasks x layers)."""
    return {model: _readout_pooled(g) for model, g in cells.groupby("model")}


def _readout_layer_summary(cells):
    """Per (model, layer): `_readout_pooled` over the tasks at that layer."""
    return {key: _readout_pooled(g) for key, g in cells.groupby(["model", "layer"])}


def _readout_groups(cells):
    """Language tasks (single binary readouts) vs the multi-output semantic-norms readout."""
    return [("Discrete features\n(Wug, BLiMP, Hypernymy)", cells[cells["task"] != READOUT_SEM_TASK]),
            ("Continuous features\n(perceptual, affective)", cells[cells["task"] == READOUT_SEM_TASK])]


def _save_svg_png(fig, save_path):
    """House style: editable-text SVG next to a PNG."""
    stem = os.path.splitext(save_path)[0]
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    fig.savefig(stem + ".svg", format="svg", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
    print(f"  Saved {stem}.svg / .png")


_NS_FILL, _NS_EDGE = "#cccccc", "#999999"


def _readout_symlog(ax, axis, vmax):
    """
    Symlog with a +-1 linear region: the null band (|x| < ~0.35) and the many
    near-zero native values render proportionally, and only the large values
    (Gemma, GPT-2 semantic) are log-compressed. A narrower linear region made the
    band straddle the break and look a decade wide.
    """
    set_scale = ax.set_xscale if axis == "x" else ax.set_yscale
    set_scale("symlog", linthresh=1.0, linscale=1.6)
    # Fewer labels when the axis reaches the log decades: the linear region is then
    # compressed and 0.25 / 0.5 / 2 collide with their neighbours.
    if vmax < 5:
        candidates = (-0.25, 0, 0.25, 0.5, 1, 2)
    elif vmax < 9.5:
        candidates = (0, 0.5, 1, 5)
    else:
        candidates = (0, 0.5, 1, 10, 50)
    ticks = [t for t in candidates if t <= max(vmax * 1.05, 1)]
    (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks)
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:g}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)
    (ax.xaxis if axis == "x" else ax.yaxis).set_minor_locator(mticker.NullLocator())


def plot_readout_sparsity_ridges(readout_files, save_path=None, bandwidth=0.12,
                                 axes=None, legend=True, label_fontsize=10):
    """
    Native readout kurtosis per model against the random-rotation null.

    Per model row: shaded ridge = distribution of the per-layer native values
    (Gaussian KDE computed in the axis' symlog coordinates, so the shape is not
    distorted by the log compression; `bandwidth` is in those display units);
    small dots = per-layer medians over tasks, family colour (darker = deeper) if
    above that layer's null, faint grey otherwise; diamond = median over all
    cells, family colour + black edge if above the model's null. The null of a
    median is ~0 +- 0.03 in every model, so it is drawn once per panel as a grey
    band spanning the widest model's 95% range instead of a per-row Gaussian.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    cells = _readout_cells(readout_files)
    if cells.empty:
        print("  No data for readout ridges")
        return
    models = [m for m in model_order if m in set(cells["model"])]
    ypos = np.arange(len(models))[::-1]
    rng = np.random.default_rng(0)   # vertical jitter of the layer dots only

    # `axes`: draw into two caller-supplied axes (paper multi-panel figures) and
    # skip legend/saving; the caller owns the figure. Default: standalone figure.
    external = axes is not None
    with plt.rc_context({"svg.fonttype": "none", "font.family": "DejaVu Sans"}):
        if external:
            fig = axes[0].figure
        else:
            fig, axes = plt.subplots(1, 2, figsize=(7.4 * .75, 4.6 * .75), sharey=True, dpi=300)
        for ax, (title, sub) in zip(axes, _readout_groups(cells)):
            summ, layers = _readout_summary(sub), _readout_layer_summary(sub)
            if not summ:
                continue
            vmax = max(1.0, max(v["native"] for v in layers.values()))
            # Scale first: the KDE is computed in the transformed coordinates.
            _readout_symlog(ax, "x", vmax)
            tf = ax.xaxis.get_transform()
            xright = vmax
            ax.axvspan(min(r["lo"] for r in summ.values()), max(r["hi"] for r in summ.values()),
                       color=_NS_EDGE, alpha=0.25, linewidth=0, zorder=1)
            for y, m in zip(ypos, models):
                if m not in summ:
                    continue
                r = summ[m]
                col = family_colors.get(model_info[m][0], "#666666")
                lr = sorted((l, v) for (mm, l), v in layers.items() if mm == m)
                xs = np.array([v["native"] for _, v in lr])
                t = tf.transform(xs)
                tg = np.linspace(t.min() - 3 * bandwidth, t.max() + 3 * bandwidth, 400)
                dens = np.exp(-0.5 * ((tg[:, None] - t[None, :]) / bandwidth) ** 2).sum(1)
                dens /= dens.max()
                xg = tf.inverted().transform(tg)
                xright = max(xright, xg.max())
                ax.fill_between(xg, y, y + 0.6 * dens, color=col, alpha=0.25, linewidth=0, zorder=2)
                ax.plot(xg, y + 0.6 * dens, color=col, linewidth=0.7, zorder=2)
                depth_max = max(l for l, _ in lr) or 1
                for l, v in lr:
                    jy = y + 0.08 + 0.14 * rng.random()
                    if v["significant"]:
                        ax.scatter(v["native"], jy, s=6, color=col, alpha=0.3 + 0.6 * l / depth_max,
                                   linewidths=0, zorder=3)
                    else:
                        ax.scatter(v["native"], jy, s=6, color=_NS_EDGE, alpha=0.25,
                                   linewidths=0, zorder=3)
                sig = r["significant"]
                ax.scatter(r["native"], y + 0.15, s=32, marker="D", zorder=5,
                           color=col if sig else "white",
                           edgecolors="black" if sig else _NS_EDGE, linewidths=0.7)
            ax.set_xlim(-0.45, xright * 1.05)
            ax.set_title(title, fontsize=9, weight="bold")
            ax.set_xlabel("Readout weight kurtosis", fontsize=label_fontsize)
            ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_linewidth(1.5)
            ax.tick_params(axis="both", labelsize=9)
        axes[0].set_yticks(ypos + 0.15)
        axes[0].set_yticklabels([model_info[m][1] for m in models], fontsize=9)
        axes[0].tick_params(axis="y", length=0)
        axes[0].set_ylim(-0.5, len(models) - 0.2)
        if external:
            return cells
        handles = [
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#4477AA",
                   markeredgecolor="black", markersize=4.5, label="Median native (above null)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#4477AA", alpha=0.7,
                   markeredgecolor="none", markersize=3, label="Layer above null (darker = deeper)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=_NS_EDGE, alpha=0.4,
                   markeredgecolor="none", markersize=3, label="Layer within null"),
            Patch(facecolor=_NS_EDGE, alpha=0.25, edgecolor="none", label="Random basis (95%, all models)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                   fontsize=8, bbox_to_anchor=(0.55, -0.1))
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        if save_path:
            _save_svg_png(fig, save_path)
        else:
            plt.show()
        plt.close(fig)
    return cells


def plot_readout_sparsity_raw(readout_files, save_path=None):
    """
    Raw (un-normalised) readout kurtosis vs rotation fraction, one line per model,
    with the random-basis range shaded. Median across task x layer cells at each
    alpha, so alpha = 0 equals the ridge figure's diamond. Same two panels. Symlog y: native values span ~0.02-5.
    """
    cells = _readout_cells(readout_files)
    if cells.empty:
        print("  No data for raw readout curve")
        return
    models = [m for m in model_order if m in set(cells["model"])]
    with plt.rc_context({"svg.fonttype": "none", "font.family": "DejaVu Sans"}):
        fig, axes = plt.subplots(1, 2, figsize=(7.4 * .75, 3.0 * .75), dpi=300)
        for ax, (title, sub) in zip(axes, _readout_groups(cells)):
            summ = _readout_summary(sub)
            # Null width scales as 1/sqrt(d), so it differs by model. One band would
            # either swallow models that are above their OWN null (union) or understate
            # the noise for others (intersection). Nested: dark = inside every model's
            # null, light = inside at least one model's null.
            vmax = 1.0
            if summ:
                ax.axhspan(min(r["lo"] for r in summ.values()), max(r["hi"] for r in summ.values()),
                           color=_NS_FILL, alpha=0.45, zorder=0, linewidth=0)
                ax.axhspan(max(r["lo"] for r in summ.values()), min(r["hi"] for r in summ.values()),
                           color=_NS_EDGE, alpha=0.45, zorder=0, linewidth=0)
            for m in models:
                g = sub[sub["model"] == m]
                if g.empty:
                    continue
                alphas = g["alphas"].iloc[0]
                curve = np.median(np.stack(g["curve"].to_numpy()), axis=0)
                family, label, marker = model_info[m]
                ax.plot(alphas, curve, label=label, color=family_colors.get(family, "#666666"),
                        marker=marker, markersize=3, linewidth=1.2, zorder=3)
                vmax = max(vmax, float(np.nanmax(curve)))
            _readout_symlog(ax, "y", vmax)
            ax.set_ylim(-0.4, vmax * 1.35)
            ax.set_title(title, fontsize=9, weight="bold")
            ax.set_xlabel(r"Rotation fraction $\alpha$", fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_linewidth(1.5)
            ax.tick_params(axis="both", labelsize=9)
        axes[0].set_ylabel("Readout weight kurtosis", fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        handles += [plt.Rectangle((0, 0), 1, 1, facecolor=_NS_EDGE, alpha=0.45, edgecolor="none"),
                    plt.Rectangle((0, 0), 1, 1, facecolor=_NS_FILL, alpha=0.45, edgecolor="none")]
        labels += ["Random basis, every model (±1.96 SD)", "Random basis, some models"]
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   frameon=False, fontsize=7.5)
        fig.tight_layout()
        if save_path:
            _save_svg_png(fig, save_path)
        else:
            plt.show()
        plt.close(fig)
    return cells


def _readout_pooled_curve(cells):
    """
    One raw alpha-curve per model pooling discrete and continuous features with
    equal weight: mean of (median curve over discrete-feature cells) and (median
    curve over continuous-feature cells). A plain median over all cells would give
    the single continuous-feature readout 1/15 of the weight and effectively drop it.
    The null is built from the same statistic per rotation seed (see
    `_readout_pooled`). Models missing either feature type use the one they have.
    """
    out = {}
    for model, g in cells.groupby("model"):
        parts = [sub for _, sub in _readout_groups(g) if not sub.empty]
        curve = np.mean([np.median(np.stack(sub["curve"].to_numpy()), axis=0) for sub in parts], axis=0)
        seeds = sorted(set.intersection(*(set(d) for d in g["null_seeds"])))
        null = np.array([np.mean([np.median([d[k] for d in sub["null_seeds"]]) for sub in parts])
                         for k in seeds])
        mu, sd = null.mean(), null.std(ddof=1)
        out[model] = dict(alphas=g["alphas"].iloc[0], curve=curve, native=curve[0],
                          null_mean=mu, null_sd=sd, lo=mu - 1.96 * sd, hi=mu + 1.96 * sd,
                          significant=bool(curve[0] > mu + 1.96 * sd))
    return out


def plot_readout_sparsity_raw_pooled(readout_files, save_path=None):
    """
    Single-panel version of `plot_readout_sparsity_raw`: raw readout kurtosis vs
    rotation fraction, one line per model, discrete and continuous features pooled
    with equal weight (`_readout_pooled_curve`). Nested null bands as in the
    two-panel figure: dark = inside every model's null, light = inside at least one.
    """
    cells = _readout_cells(readout_files)
    if cells.empty:
        print("  No data for pooled raw readout curve")
        return
    pooled = _readout_pooled_curve(cells)
    models = [m for m in model_order if m in pooled]
    with plt.rc_context({"svg.fonttype": "none", "font.family": "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(4.0 * .75, 3.0 * .75), dpi=300)
        ax.axhspan(min(r["lo"] for r in pooled.values()), max(r["hi"] for r in pooled.values()),
                   color=_NS_FILL, alpha=0.45, zorder=0, linewidth=0)
        ax.axhspan(max(r["lo"] for r in pooled.values()), min(r["hi"] for r in pooled.values()),
                   color=_NS_EDGE, alpha=0.45, zorder=0, linewidth=0)
        vmax = 1.0
        for m in models:
            r = pooled[m]
            family, label, marker = model_info[m]
            ax.plot(r["alphas"], r["curve"], label=label, color=family_colors.get(family, "#666666"),
                    marker=marker, markersize=3, linewidth=1.2, zorder=3)
            vmax = max(vmax, float(np.nanmax(r["curve"])))
        _readout_symlog(ax, "y", vmax)
        ax.set_ylim(-0.4, vmax * 1.35)
        ax.set_xlabel(r"Rotation fraction $\alpha$", fontsize=10)
        ax.set_ylabel("Readout weight kurtosis", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(1.5)
        ax.tick_params(axis="both", labelsize=9)
        handles, labels = ax.get_legend_handles_labels()
        handles += [plt.Rectangle((0, 0), 1, 1, facecolor=_NS_EDGE, alpha=0.45, edgecolor="none"),
                    plt.Rectangle((0, 0), 1, 1, facecolor=_NS_FILL, alpha=0.45, edgecolor="none")]
        labels += ["Random basis, every model (±1.96 SD)", "Random basis, some models"]
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   frameon=False, fontsize=7.5)
        fig.tight_layout()
        if save_path:
            _save_svg_png(fig, save_path)
        else:
            plt.show()
        plt.close(fig)
    return pooled


def _gather_model_scores(file_groups, to_replace_2="_semantic"):
    """
    Build {model_key: (random_mean, native_mean, optimal_mean)} averaged across subtasks.

    random_mean = mean at α=1.0 (last entry of the mean array).
    native_mean = mean at α=0.0 (first entry).
    optimal_mean = the optimal rotation score.

    file_groups: list of (prefix_str, [file_paths]) pairs, one per subtask.
    Within each subtask, _semantic files are preferred over non-semantic.
    """
    from collections import defaultdict
    model_randoms  = defaultdict(list)
    model_natives  = defaultdict(list)
    model_optimals = defaultdict(list)

    for prefix, files in file_groups:
        sorted_files = sorted(files, key=lambda p: (0 if p.endswith("_semantic.csv") else 1))
        seen_in_task = set()
        for path in sorted_files:
            key = extract_key(path, to_replace=prefix, to_replace_2=to_replace_2)
            if key not in model_info or key in seen_in_task:
                continue
            seen_in_task.add(key)
            try:
                res = _load_csv(path)
                res = res[res["layer"] > 0]
                model_randoms[key].extend([r["mean"][-1] for _, r in res.iterrows()])
                model_natives[key].extend([r["mean"][0]  for _, r in res.iterrows()])
                model_optimals[key].extend(res["optimal"].tolist())
            except Exception as e:
                print(f"  warning {path}: {e}")

    return {k: (float(np.mean(model_randoms[k])),
                float(np.mean(model_natives[k])),
                float(np.mean(model_optimals[k])))
            for k in model_natives}


def plot_model_model_alignment(csv_files, normalize=True,
                               plot_title="Model-to-model alignment",
                               save_path=None):
    """
    Plot rotation-alignment curves for model-to-model comparisons.

    csv_files : list of data/results/model_model_{alias_A}_vs_{alias_B}.csv paths.

    Each CSV has columns: layer_a, layer_b, depth_frac, mean (array), sem (array), optimal.
    The 4 rows (depth 25/50/75/100%) are averaged to produce one curve per pair.

    If normalize=True, each mean array is divided by the corresponding optimal scalar,
    then averaged across depth layers.
    """
    # Pair labels and colors derived from alias_B in the filename
    def _pair_label_color(csv_path):
        fname = os.path.basename(csv_path).replace(".csv", "")
        # fname: model_model_{alias_A}_vs_{alias_B}
        # alias_B is everything after '_vs_'
        alias_b = fname.split("_vs_", 1)[-1]
        if alias_b in model_info:
            family, label, _ = model_info[alias_b]
            color = family_colors.get(family, "#888888")
            return label, color
        # Fallback: humanize the alias
        readable = alias_b.replace("EleutherAI_", "").replace("state-spaces_", "").replace("-hf", "")
        return readable, "#888888"

    entries = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            df["mean"] = df["mean"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
            df["sem"]  = df["sem"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

            curves = []
            for _, row in df.iterrows():
                v = row["mean"]
                if normalize and row["optimal"] > 0:
                    v = v / row["optimal"]
                curves.append(v)

            if not curves:
                continue
            avg_curve = np.mean(curves, axis=0)
            label, color = _pair_label_color(csv_path)
            entries.append((label, color, avg_curve))
        except Exception as e:
            print(f"  skipping {csv_path}: {e}")

    if not entries:
        print(f"  No data for '{plot_title}'")
        return

    alphas = np.linspace(0, 1, entries[0][2].shape[0])

    fig, ax = plt.subplots(figsize=(4.8 * .8, 3.2 * .8))
    for label, color, curve in entries:
        ax.plot(alphas, curve, label=label, color=color, linewidth=1.8, marker="o", markersize=4)

    ax.set_xlabel(r"Rotation fraction $\alpha$")
    ax.set_ylabel("Alignment / optimal" if normalize else "Pairwise alignment (r)")
    ax.set_title(plot_title)
    ax.grid(True, linestyle="--")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_checkpoint_alignment(csv_paths, save_path=None,
                              plot_title="Basis alignment across training",
                              rsa_dir="data/results"):
    """
    Plot how native, random, and optimal alignment evolve across training steps,
    for one or more model sizes. Overlays the rotation-invariant RSA control on a
    secondary y-axis (right) when matching model_model_checkpoints_rsa_*.csv files
    are present in `rsa_dir`. Skips RSA silently for any model size that lacks one.

    Layout: N panels side-by-side (one per model size). Each panel:
      left y-axis  → native (solid), random (dashed), optimal (dash-dot)
      right y-axis → RSA (dotted, grey) — only labelled on the rightmost panel
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    import re

    def _size_label(path):
        m = re.search(r"pythia-(\w+)-seed", os.path.basename(path))
        return m.group(1).upper() if m else os.path.basename(path)

    def _numeric(label):
        m = re.search(r"(\d+)", label)
        return int(m.group(1)) if m else 0

    def _rsa_path_for(alignment_csv_path):
        """
        Find the RSA CSV that matches an alignment CSV. The filenames may
        differ in whether `model_a` carries a `-seed1` suffix, so we try
        several reasonable variants.

        Returns the first existing path, or None.
        """
        fname = os.path.basename(alignment_csv_path)
        # Strip the alignment prefix so we can manipulate alias_a / alias_b
        suffix = fname.replace("model_model_checkpoints_", "", 1)   # X_vs_Y.csv
        if "_vs_" not in suffix:
            return None
        alias_a, _, rest = suffix.partition("_vs_")
        # Candidate alias_a forms — try the original first, then strip "-seed1"
        candidates = [alias_a]
        if alias_a.endswith("-seed1"):
            candidates.append(alias_a[: -len("-seed1")])
        else:
            candidates.append(alias_a + "-seed1")
        for cand in candidates:
            p = os.path.join(rsa_dir,
                             f"model_model_checkpoints_rsa_{cand}_vs_{rest}")
            if os.path.exists(p):
                return p
        return None

    def _rsa_series(rsa_csv):
        """Return (steps, mean, sem) averaged across layers per step, or None."""
        if rsa_csv is None or not os.path.exists(rsa_csv):
            return None
        df = pd.read_csv(rsa_csv)
        if df.empty:
            return None
        grouped = df.groupby("step")
        steps = np.array(sorted(grouped.groups.keys()))
        mean = np.zeros(len(steps))
        sem  = np.zeros(len(steps))
        for i, s in enumerate(steps):
            v = grouped.get_group(s)["rsa"].dropna().values
            mean[i] = v.mean() if len(v) else np.nan
            sem[i]  = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        plot_steps = steps.astype(float)
        if plot_steps[0] == 0:
            plot_steps[0] = 0.5
        return plot_steps, mean, sem

    entries = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"  [plot_checkpoint_alignment] not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        grouped = df.groupby("step")
        steps = np.array(sorted(grouped.groups.keys()))
        data = {col: (np.zeros(len(steps)), np.zeros(len(steps)))
                for col in ("native", "random", "optimal")}

        for i, step in enumerate(steps):
            g = grouped.get_group(step)
            for col in ("native", "random", "optimal"):
                vals = g[col].dropna().values
                data[col][0][i] = vals.mean()
                data[col][1][i] = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

        plot_steps = steps.astype(float).copy()
        if plot_steps[0] == 0:
            plot_steps[0] = 0.5

        rsa = _rsa_series(_rsa_path_for(csv_path))    # may be None
        entries.append((_size_label(csv_path), plot_steps, data, rsa))

    if not entries:
        print("  [plot_checkpoint_alignment] no data found")
        return

    entries.sort(key=lambda e: _numeric(e[0]))
    n = len(entries)

    line_styles = [("-", "#2166ac"), ("--", "#d6604d"), ("-.", "#1a9641")]
    line_labels = [r"native ($\alpha$=0)", r"random ($\alpha$=1)", "optimal"]
    cols        = ["native", "random", "optimal"]
    rsa_color   = "black"

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), sharey=True)
    if n == 1:
        axes = [axes]

    # Determine global RSA y-range across panels (for shared right axis)
    rsa_vals = [e[3][1] for e in entries if e[3] is not None]
    have_rsa = len(rsa_vals) > 0
    if have_rsa:
        all_rsa = np.concatenate(rsa_vals)
        rsa_lo  = float(np.nanmin(all_rsa))
        rsa_hi  = float(np.nanmax(all_rsa))
        # Add small padding
        rsa_pad = 0.05 * (rsa_hi - rsa_lo + 1e-9)
        rsa_ylim = (rsa_lo - rsa_pad, rsa_hi + rsa_pad)

    rsa_axes = []
    for idx, (ax, (label, plot_steps, data, rsa)) in enumerate(zip(axes, entries)):
        for (ls, color), col, line_label in zip(line_styles, cols, line_labels):
            mean, sem = data[col]
            ax.plot(plot_steps, mean, color=color, linestyle=ls,
                    linewidth=1.8, label=line_label)
            ax.fill_between(plot_steps, mean - sem, mean + sem,
                            color=color, alpha=0.15)

        # RSA twinx (always create one for visual consistency, even if no data)
        ax_r = ax.twinx()
        rsa_axes.append(ax_r)
        if rsa is not None:
            r_steps, r_mean, r_sem = rsa
            ax_r.plot(r_steps, r_mean, color=rsa_color, linestyle="-",
                      linewidth=0.7, alpha=1.0, zorder=0, label="RSA")
            ax_r.fill_between(r_steps, r_mean - r_sem, r_mean + r_sem,
                              color=rsa_color, alpha=0.04, zorder=0)

        if have_rsa:
            ax_r.set_ylim(*rsa_ylim)
        # Hide right-axis ticks except on the rightmost panel
        if idx < n - 1:
            ax_r.set_yticklabels([])
            ax_r.tick_params(axis="y", which="both", length=0)

        ax.set_xscale("log")
        ax.set_xlabel("Training step")
        ax.set_title(f"Pythia-{label}")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Pairwise alignment (r)")
    if have_rsa:
        rsa_axes[-1].set_ylabel("RSA (Spearman r)", color=rsa_color)
        rsa_axes[-1].tick_params(axis="y", labelcolor=rsa_color)

    # Combined legend on the leftmost panel
    handles_l, labels_l = axes[0].get_legend_handles_labels()
    if have_rsa:
        from matplotlib.lines import Line2D
        handles_l.append(Line2D([0], [0], color=rsa_color, linestyle="-",
                                linewidth=0.7, label="RSA"))
        labels_l.append("RSA")
    axes[0].legend(handles_l, labels_l, frameon=False, fontsize=7)

    fig.suptitle(plot_title, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _plot_checkpoint_metric(entries, metric_fn, ylabel, ref_line,
                            plot_title, save_path):
    """
    Internal helper: single-panel plot of a derived metric across training steps.

    entries   : list of (label, plot_steps, df_grouped) sorted by model size
    metric_fn : callable(g) → per-step scalar value for a grouped step DataFrame
    ylabel    : y-axis label
    ref_line  : y value for horizontal reference line (or None)
    """
    from matplotlib.lines import Line2D

    ref_steps  = entries[0][1]
    colors     = [plt.cm.Blues(v) for v in np.linspace(0.35, 0.92, len(entries))]
    avg_color  = "#1a1a2e"

    per_model = []
    for label, plot_steps, grouped, steps in entries:
        vals = np.array([metric_fn(grouped.get_group(s)) for s in steps])
        per_model.append(vals)

    avg = np.mean(per_model, axis=0)
    sem = np.std(per_model, axis=0, ddof=1) / np.sqrt(len(per_model))

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for (label, plot_steps, *_), vals, color in zip(entries, per_model, colors):
        ax.plot(plot_steps, vals, color=color, linewidth=0.9, alpha=0.35, label=label)

    ax.plot(ref_steps, avg, color=avg_color, linewidth=2.2)
    ax.fill_between(ref_steps, avg - sem, avg + sem, color=avg_color, alpha=0.15)

    if ref_line is not None:
        ax.axhline(ref_line, color="#cc0000", linewidth=0.8, linestyle="--")

    ax.set_xscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(plot_title)
    ax.grid(True, linestyle="--", alpha=0.4)

    size_handles = [Line2D([0], [0], color=c, linewidth=0.9, alpha=0.35, label=e[0])
                    for e, c in zip(entries, colors)]
    avg_handle   = Line2D([0], [0], color=avg_color, linewidth=2.2, label="average")
    ax.legend(handles=size_handles + [avg_handle], frameon=False, fontsize=7)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_checkpoint_ratios(csv_paths, save_dir=None):
    """
    Four single-panel plots of alignment metrics across training steps:
      1. native / random
      2. native / optimal
      3. native - random
      4. native - optimal

    Each plot: thin coloured line per model size + thick average ± SEM.
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    import re

    def _size_label(path):
        m = re.search(r"pythia-(\w+)-seed", os.path.basename(path))
        return m.group(1).upper() if m else os.path.basename(path)

    def _numeric(label):
        m = re.search(r"(\d+)", label)
        return int(m.group(1)) if m else 0

    entries = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        grouped = df.groupby("step")
        steps = np.array(sorted(grouped.groups.keys()))
        plot_steps = steps.astype(float).copy()
        if plot_steps[0] == 0:
            plot_steps[0] = 0.5
        entries.append((_size_label(csv_path), plot_steps, grouped, steps))

    if not entries:
        print("  [plot_checkpoint_ratios] no data")
        return

    entries.sort(key=lambda e: _numeric(e[0]))

    base = save_dir or "."
    os.makedirs(base, exist_ok=True)

    specs = [
        (lambda g: (g["native"] / g["random"].replace(0, np.nan)).dropna().mean(),
         "Native / random", 1.0, "Native / random across training",
         os.path.join(base, "model_model_checkpoints_native_over_random.png")),
        (lambda g: (g["native"] - g["random"]).dropna().mean(),
         "Native − random", 0.0, "Native − random across training",
         os.path.join(base, "model_model_checkpoints_native_minus_random.png")),
    ]

    for metric_fn, ylabel, ref_line, title, path in specs:
        _plot_checkpoint_metric(entries, metric_fn, ylabel, ref_line, title, path)


def plot_checkpoint_rsa(csv_paths, save_path=None,
                        plot_title="RSA across training (rotation-invariant control)"):
    """
    Plot RSA (Spearman r between cosine-RDMs) across training steps for one or more
    Pythia model sizes. Layout matches plot_checkpoint_metric:
      thin coloured line per model size + thick black average ± SEM, log-x.
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    import re
    from matplotlib.lines import Line2D

    def _size_label(path):
        m = re.search(r"pythia-(\w+)-seed", os.path.basename(path))
        return m.group(1).upper() if m else os.path.basename(path)

    def _numeric(label):
        m = re.search(r"(\d+)", label)
        return int(m.group(1)) if m else 0

    entries = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"  [plot_checkpoint_rsa] not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        grouped = df.groupby("step")
        steps = np.array(sorted(grouped.groups.keys()))
        rsa_mean = np.zeros(len(steps))
        for i, step in enumerate(steps):
            vals = grouped.get_group(step)["rsa"].dropna().values
            rsa_mean[i] = vals.mean() if len(vals) else np.nan
        plot_steps = steps.astype(float).copy()
        if plot_steps[0] == 0:
            plot_steps[0] = 0.5
        entries.append((_size_label(csv_path), plot_steps, rsa_mean))

    if not entries:
        print("  [plot_checkpoint_rsa] no data")
        return

    entries.sort(key=lambda e: _numeric(e[0]))
    ref_steps = entries[0][1]
    colors = [plt.cm.Blues(v) for v in np.linspace(0.35, 0.92, len(entries))]
    avg_color = "#1a1a2e"

    per_model = np.stack([e[2] for e in entries])
    avg = np.nanmean(per_model, axis=0)
    if len(entries) > 1:
        sem = np.nanstd(per_model, axis=0, ddof=1) / np.sqrt(len(entries))
    else:
        sem = np.zeros_like(avg)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for (label, plot_steps, vals), color in zip(entries, colors):
        ax.plot(plot_steps, vals, color=color, linewidth=0.9, alpha=0.35)
    ax.plot(ref_steps, avg, color=avg_color, linewidth=2.2)
    ax.fill_between(ref_steps, avg - sem, avg + sem, color=avg_color, alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("RSA (Spearman r)")
    ax.set_title(plot_title)
    ax.grid(True, linestyle="--", alpha=0.4)

    size_handles = [Line2D([0], [0], color=c, linewidth=0.9, alpha=0.35, label=e[0])
                    for e, c in zip(entries, colors)]
    avg_handle = Line2D([0], [0], color=avg_color, linewidth=2.2, label="average")
    ax.legend(handles=size_handles + [avg_handle], frameon=False, fontsize=7)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _gather_model_pair_scores(results_dir):
    """
    Per-model basis-privilege scores from model-to-model CSVs.
    The 'partner' (alias_b) is treated as the model identity for the score.

    Returns: {model_key: (random_mean, native_mean, optimal_mean)} averaged across
    the 4 depth-selected layers in each model_model_{alias_a}_vs_{alias_b}.csv.
    """
    out = {}
    for path in sorted(glob(f"{results_dir}/model_model_*.csv")):
        # Skip the checkpoint-trajectory CSVs
        if "checkpoints" in os.path.basename(path):
            continue
        try:
            fname = os.path.basename(path).replace(".csv", "")
            if "_vs_" not in fname:
                continue
            alias_b = fname.split("_vs_", 1)[-1]
            if alias_b not in model_info:
                continue
            df = pd.read_csv(path)
            df["mean"] = df["mean"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
            rnd = float(np.mean([r["mean"][-1] for _, r in df.iterrows()]))
            nat = float(np.mean([r["mean"][0]  for _, r in df.iterrows()]))
            opt = float(df["optimal"].astype(float).mean())
            out[alias_b] = (rnd, nat, opt)
        except Exception as e:
            print(f"  warning {path}: {e}")
    return out


def _significance_marker(p):
    """Standard significance markers for annotation."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def plot_inter_model_privilege(metric="difference", results_dir="data/results",
                                save_path=None,
                                plot_title=None):
    """
    Per-(model, task) basis-privilege dot-plot with summary stats annotated.

    metric : "difference" → native − random, reference line at 0
             "ratio"      → native / random, reference line at 1

    For each task we annotate:
      - the per-task one-sample t-test against the reference value (privilege > chance?)
      - the Spearman correlation with log(parameter count) (does privilege scale with model size?)
    Models within a task are arranged in family-grouped lanes (no random jitter)
    so the plot is readable.
    """
    from scipy import stats as _stats
    from matplotlib.lines import Line2D

    if metric not in ("difference", "ratio"):
        raise ValueError(f"metric must be 'difference' or 'ratio', got {metric!r}")

    blimp_groups = [(f"lang_{t}_", sorted(glob(f"{results_dir}/lang_{t}_*.csv")))
                    for t in _blimp_tasks]

    task_groups = {
        "Wug":        [("lang_wug_",       sorted(glob(f"{results_dir}/lang_wug_*.csv")))],
        "BLiMP":      blimp_groups,
        "Hypernymy":  [("lang_hypernymy_", sorted(glob(f"{results_dir}/lang_hypernymy_*.csv")))],
        "Perceptual": [("perceptual_",     sorted(glob(f"{results_dir}/perceptual_*.csv")))],
        "Affective":  [("affective_",      sorted(glob(f"{results_dir}/affective_*.csv")))],
        "Pereira E2": [("pereira_expt2_",  sorted(glob(f"{PEREIRA_RES}/pereira_expt2_*.csv")))],
        "Pereira E3": [("pereira_expt3_",  sorted(glob(f"{PEREIRA_RES}/pereira_expt3_*.csv")))],
    }

    score_by_task = {label: _gather_model_scores(grp) for label, grp in task_groups.items()}
    score_by_task["Model pairs"] = _gather_model_pair_scores(results_dir)

    task_order = ["Wug", "BLiMP", "Hypernymy", "Perceptual", "Affective",
                  "Pereira E2", "Pereira E3", "Model pairs"]
    task_order = [t for t in task_order if t in score_by_task and score_by_task[t]]

    if not task_order:
        print("  [plot_inter_model_privilege] no data found")
        return

    # Approximate parameter counts (millions) — used for size-correlation stat only
    param_counts_M = {
        "gpt2":                              124,
        "gpt2-medium":                       355,
        "gpt2-large":                        774,
        "gpt2-xl":                          1500,
        "meta-llama_Llama-3.2-1B":          1230,
        "meta-llama_Llama-3.2-1B-Instruct": 1230,
        "meta-llama_Llama-3.2-3B":          3210,
        "google_gemma-3-1b-pt":             1000,
        "google_gemma-3-4b-pt":             4000,
        "state-spaces_mamba-130m-hf":        130,
        "state-spaces_mamba-370m-hf":        370,
        "state-spaces_mamba-790m-hf":        790,
        "Qwen_Qwen3-0.6B-Base":              600,
        "EleutherAI_pythia-160m":            160,
        "EleutherAI_pythia-160m-seed2":      160,
    }

    ref_value = 0.0 if metric == "difference" else 1.0
    family_x_order = ["GPT-2", "LLaMA", "Gemma", "Mamba", "Qwen", "Pythia"]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    seen_models = []
    annotations = []   # (x_pos, task, t_p, sp_rho, sp_p, mean_priv)

    for x_pos, task in enumerate(task_order):
        scores = score_by_task[task]
        # Compute per-model privilege values for this task
        rows = []
        for k, (rnd, nat, opt) in scores.items():
            if k not in model_info:
                continue
            if metric == "difference":
                y = nat - rnd
            else:
                y = nat / rnd if (rnd is not None and rnd != 0) else np.nan
            rows.append({"key": k,
                         "family": model_info[k][0],
                         "params_M": param_counts_M.get(k, np.nan),
                         "priv": y})
        if not rows:
            continue
        df = pd.DataFrame(rows)

        # Stats
        priv_vals = df["priv"].dropna().values
        if len(priv_vals) >= 2:
            t_stat, t_p = _stats.ttest_1samp(priv_vals, ref_value)
        else:
            t_stat, t_p = np.nan, np.nan

        size_df = df.dropna(subset=["params_M", "priv"])
        if len(size_df) >= 4:
            sp_rho, sp_p = _stats.spearmanr(np.log(size_df["params_M"]),
                                             size_df["priv"])
        else:
            sp_rho, sp_p = np.nan, np.nan

        annotations.append((x_pos, task, t_p, sp_rho, sp_p, priv_vals.mean()))

        # Lay out points within the task: x = x_pos + family_offset + within_family_offset
        present_families = [f for f in family_x_order if f in set(df["family"])]
        n_fams = len(present_families)
        # Reserve ±0.42 around x_pos for the family lanes
        if n_fams > 1:
            fam_centers = np.linspace(-0.40, 0.40, n_fams)
        else:
            fam_centers = np.array([0.0])

        for fc, family in zip(fam_centers, present_families):
            sub = df[df["family"] == family]
            sub = sub.sort_values(by="key", key=lambda c: c.map(
                lambda k: model_order.index(k) if k in model_order else 1e9))
            n_in_fam = len(sub)
            if n_in_fam == 1:
                local_off = np.array([0.0])
            else:
                local_off = np.linspace(-0.07, 0.07, n_in_fam)
            for local, (_, r) in zip(local_off, sub.iterrows()):
                k = r["key"]
                family, label, marker = model_info[k]
                color = family_colors.get(family, "#888888")
                ax.scatter(x_pos + fc + local, r["priv"],
                           color=color, marker=marker,
                           s=46, edgecolor="black", linewidth=0.5,
                           zorder=3)
                if k not in seen_models:
                    seen_models.append(k)

        # Family-mean horizontal tick (small black bars)
        for fc, family in zip(fam_centers, present_families):
            fam_mean = df[df["family"] == family]["priv"].mean()
            ax.plot([x_pos + fc - 0.10, x_pos + fc + 0.10],
                    [fam_mean, fam_mean], color="black",
                    linewidth=1.2, alpha=0.55, zorder=2)

    # X-axis
    ax.set_xticks(range(len(task_order)))
    ax.set_xticklabels(task_order, rotation=15, ha="right")
    ax.set_xlim(-0.6, len(task_order) - 0.4)

    # Reference line + axis label
    ax.axhline(ref_value, color="black", linewidth=0.7, linestyle="--", zorder=1)
    if metric == "difference":
        ax.set_ylabel(r"Native $-$ random  (basis privilege)")
        default_title = "Basis privilege per model per task (additive)"
    else:
        ax.set_ylabel("Native / random  (basis privilege)")
        default_title = "Basis privilege per model per task (ratio)"
    ax.set_title(plot_title or default_title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, zorder=0)

    # Annotate per-task stats above the cluster
    y_max = ax.get_ylim()[1]
    y_anno = y_max - 0.04 * (y_max - ax.get_ylim()[0])
    for x_pos, task, t_p, sp_rho, sp_p, mean_priv in annotations:
        ann = f"{_significance_marker(t_p)}"
        if np.isfinite(sp_p):
            ann += f"\nρ(size)={sp_rho:+.2f}"
            ann += f"\np={sp_p:.2f}"
        ax.text(x_pos, y_anno, ann, ha="center", va="top",
                fontsize=6.5, color="#333333", linespacing=1.05)

    # Legend
    legend_keys = sorted(seen_models,
                         key=lambda k: model_order.index(k) if k in model_order else 1e9)
    legend_handles = []
    for k in legend_keys:
        family, label, marker = model_info[k]
        color = family_colors.get(family, "#888888")
        legend_handles.append(Line2D(
            [0], [0], marker=marker, color="white",
            markerfacecolor=color, markeredgecolor="black",
            markeredgewidth=0.5, markersize=7, label=label))
    family_mean_handle = Line2D([0], [0], color="black", linewidth=1.2,
                                 alpha=0.55, label="family mean")
    legend_handles.append(family_mean_handle)
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7,
              title="Model", title_fontsize=8)

    fig.tight_layout()

    # Footer note about the stats convention
    fig.text(0.005, 0.005,
             "*** p<0.001  ** p<0.01  * p<0.05  n.s. ≥ 0.05 "
             "(one-sample t-test vs reference); ρ(size) = Spearman with log(params)",
             ha="left", fontsize=6.5, color="#555555")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_native_vs_optimal_three(results_dir="data/results", save_path=None,
                                  include_random=False,
                                  feature_results_dir=None):
    """
    Three-panel figure: model pairs (left), linguistic features (center), brain (right).
    Each panel: boxplot of alignment per category.
    include_random=False → two boxes (native, optimal).
    include_random=True  → three boxes (random α=1, native α=0, optimal).
    Group labels drawn below tick labels.
    Y-axes are independent across panels.

    feature_results_dir : optional override for the middle (features) panel only.
        Used by plot_native_vs_optimal_three_joint to point at data/results/joint
        while the brain + pair panels still read from data/results.
    """
    from matplotlib.patches import Patch

    feat_dir = feature_results_dir or results_dir

    blimp_groups = [(f"lang_{t}_", sorted(glob(f"{feat_dir}/lang_{t}_*.csv")))
                    for t in _blimp_tasks]

    # Brain panel: two sub-experiments as separate x-positions
    brain_cat_defs = [
        ("Expt 2", [("pereira_expt2_", sorted(glob(f"{PEREIRA_RES}/pereira_expt2_*.csv")))]),
        ("Expt 3", [("pereira_expt3_", sorted(glob(f"{PEREIRA_RES}/pereira_expt3_*.csv")))]),
    ]

    # Non-brain panel: morphology -- syntax -- semantics
    nonbrain_cat_defs = [
        ("Wug",        [("lang_wug_",       sorted(glob(f"{feat_dir}/lang_wug_*.csv")))]),
        ("BLiMP",      blimp_groups),
        ("Perceptual", [("perceptual_", sorted(glob(f"{feat_dir}/perceptual_*.csv")))]),
        ("Affective",  [("affective_",  sorted(glob(f"{feat_dir}/affective_*.csv")))]),
        ("Hypernymy",  [("lang_hypernymy_", sorted(glob(f"{feat_dir}/lang_hypernymy_*.csv")))]),
    ]

    def _build(cat_defs):
        out = {}
        for label, file_groups in cat_defs:
            s = _gather_model_scores(file_groups)
            if s:
                out[label] = s
        return out

    brain_scores    = _build(brain_cat_defs)
    nonbrain_scores = _build(nonbrain_cat_defs)

    # Model-pair panel: one scalar per pair (mean across depth layers)
    pair_data   = {}   # {pair_label: (random_scalar, native_scalar, optimal_scalar)}
    pair_colors = {}   # {pair_label: color}
    for csv_path in sorted(glob(f"{results_dir}/model_model_*.csv")):
        # Skip the checkpoint-trajectory CSVs: they use a step/native/random/
        # optimal schema with no `mean` column, and their `_vs_` suffix can
        # collide with a real pair label (pythia-160m-seed2), so relying on the
        # load to fail would make the panel depend on glob ordering.
        if "checkpoints" in os.path.basename(csv_path):
            continue
        try:
            fname = os.path.basename(csv_path).replace(".csv", "")
            alias_b = fname.split("_vs_", 1)[-1]
            if alias_b in model_info:
                family, label_b, _ = model_info[alias_b]
                color = family_colors.get(family, "#888888")
                pair_label = label_b
            else:
                pair_label = (alias_b.replace("EleutherAI_", "")
                                     .replace("state-spaces_", "")
                                     .replace("-hf", ""))
                color = "#888888"
            df = pd.read_csv(csv_path)
            df["mean"] = df["mean"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
            rnd_val = float(np.mean([row["mean"][-1] for _, row in df.iterrows()]))
            nat_val = float(np.mean([row["mean"][0]  for _, row in df.iterrows()]))
            opt_val = float(np.mean(df["optimal"].astype(float)))
            pair_data[pair_label]   = (rnd_val, nat_val, opt_val)
            pair_colors[pair_label] = color
        except Exception as e:
            print(f"  warning {csv_path}: {e}")

    if not brain_scores and not nonbrain_scores and not pair_data:
        print("  No data for native vs. optimal summary")
        return

    n_b = len(brain_scores)
    n_n = len(nonbrain_scores)
    w_p = 1.5
    w_b = max(1.6, n_b * 1.1)
    w_n = max(3.0, n_n * 1.0)
    fig_w = (w_p + w_n + w_b + 0.9) * 0.9
    fig_h = (3.2 * .8) * 0.8

    # Panel order: model pairs (leftmost) — brain (middle) — features (rightmost).
    fig, (ax_p, ax_b, ax_n) = plt.subplots(
        1, 3,
        gridspec_kw={"width_ratios": [w_p, w_b, w_n], "wspace": 0.3},
        figsize=(fig_w, fig_h),
    )

    def _add_group_bracket(ax, x_lo, x_hi, label, line_y=-0.18, text_y=-0.26):
        """Horizontal line + label drawn below tick labels in blended coordinates
        (x in data units, y in axes fraction; 0=bottom spine, negative=below)."""
        trans = ax.get_xaxis_transform()
        ax.plot([x_lo, x_hi], [line_y, line_y],
                transform=trans, color="black", lw=0.8, clip_on=False)
        ax.text((x_lo + x_hi) / 2, text_y, label,
                transform=trans, ha="center", va="top",
                fontsize=7, fontweight="bold", clip_on=False)

    def _draw_panel(ax, cat_scores_dict):
        labels = list(cat_scores_dict.keys())
        n = len(labels)
        if n == 0:
            return None

        # Positions: 2 or 3 boxes per category, centered on tick
        if include_random:
            step = 0.42
            gap_local = 2.1
            pos_nat = np.arange(n) * gap_local
            pos_rnd = pos_nat - step
            pos_opt = pos_nat + step
            pos_left  = pos_rnd
            pos_right = pos_opt
        else:
            gap_local = 1.5
            pos_nat = np.arange(n) * gap_local
            pos_opt = pos_nat + 0.55
            pos_rnd   = None
            pos_left  = pos_nat
            pos_right = pos_opt

        box_rnd, box_nat, box_opt = [], [], []
        for ci, lbl in enumerate(labels):
            scores = cat_scores_dict[lbl]
            keys = sorted(scores.keys(),
                          key=lambda k: model_order.index(k) if k in model_order else 1e9)
            rnds = np.array([scores[k][0] for k in keys])
            nats = np.array([scores[k][1] for k in keys])
            opts = np.array([scores[k][2] for k in keys])
            box_rnd.append(rnds)
            box_nat.append(nats)
            box_opt.append(opts)

            for k, rnd, nat, opt in zip(keys, rnds, nats, opts):
                family, _, _ = model_info.get(k, ("?", k, "o"))
                color = family_colors.get(family, "#666666")
                if include_random:
                    xs = [pos_rnd[ci], pos_nat[ci], pos_opt[ci]]
                    ys = [rnd, nat, opt]
                else:
                    xs = [pos_nat[ci], pos_opt[ci]]
                    ys = [nat, opt]
                ax.plot(xs, ys, color=color, linewidth=0.35, alpha=0.3, zorder=2)
                ax.scatter(xs, ys, color=color, s=3, alpha=0.45, zorder=3)

        def _bp(positions, data, facecolor):
            valid = [(p, d) for p, d in zip(positions, data) if len(d) > 1]
            if not valid:
                return
            ps, ds = zip(*valid)
            bp = ax.boxplot(
                list(ds), positions=list(ps), widths=0.28, patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.2},
                boxprops={"linewidth": 1.0},
                whiskerprops={"linewidth": 1.0},
                capprops={"linewidth": 1.0},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(facecolor)
                patch.set_alpha(0.65)

        if include_random:
            _bp(pos_rnd, box_rnd, "#cccccc")   # grey = random
        _bp(pos_nat, box_nat, "#9ecae1")        # light blue = native
        _bp(pos_opt, box_opt, "#3182bd")        # dark blue = optimal

        tick_pos = pos_nat  # center of the group (= native position)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlim(pos_left[0] - 0.6, pos_right[-1] + 0.6)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return labels, pos_left, pos_right

    # --- Model-pair panel (leftmost) ---
    # One group "Model pairs": distribution over pairs, 2 or 3 boxes (random/native/optimal)
    if pair_data:
        keys_p = list(pair_data.keys())
        rnds_p = np.array([pair_data[k][0] for k in keys_p])
        nats_p = np.array([pair_data[k][1] for k in keys_p])
        opts_p = np.array([pair_data[k][2] for k in keys_p])

        if include_random:
            pos_p_rnd, pos_p_nat, pos_p_opt = -0.42, 0.0, 0.42
            xlim_lo, xlim_hi = pos_p_rnd - 0.5, pos_p_opt + 0.5
        else:
            pos_p_nat, pos_p_opt = 0.0, 0.55
            xlim_lo, xlim_hi = pos_p_nat - 0.5, pos_p_opt + 0.5

        for k, rnd, nat, opt in zip(keys_p, rnds_p, nats_p, opts_p):
            color = pair_colors[k]
            if include_random:
                xs = [pos_p_rnd, pos_p_nat, pos_p_opt]
                ys = [rnd, nat, opt]
            else:
                xs = [pos_p_nat, pos_p_opt]
                ys = [nat, opt]
            ax_p.plot(xs, ys, color=color, linewidth=0.35, alpha=0.3, zorder=2)
            ax_p.scatter(xs, ys, color=color, s=3, alpha=0.45, zorder=3)

        def _bpp(pos, vals, fc):
            if len(vals) > 1:
                bp = ax_p.boxplot(
                    [vals], positions=[pos], widths=0.28, patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.2},
                    boxprops={"linewidth": 1.0},
                    whiskerprops={"linewidth": 1.0},
                    capprops={"linewidth": 1.0},
                )
                bp["boxes"][0].set_facecolor(fc)
                bp["boxes"][0].set_alpha(0.65)

        if include_random:
            _bpp(pos_p_rnd, rnds_p, "#cccccc")
        _bpp(pos_p_nat, nats_p, "#9ecae1")
        _bpp(pos_p_opt, opts_p, "#3182bd")

        ax_p.set_xticks([pos_p_nat])
        ax_p.set_xticklabels(["Model pairs"], fontsize=7)
        ax_p.set_xlim(xlim_lo, xlim_hi)
        ax_p.set_ylabel("Alignment (r)")
        ax_n.set_ylabel("")
        ax_b.set_ylabel("")
        ax_p.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax_p.spines["top"].set_visible(False)
        ax_p.spines["right"].set_visible(False)
        _add_group_bracket(ax_p, xlim_lo + 0.25, xlim_hi - 0.25, "Model pairs")
    else:
        ax_p.set_visible(False)
        ax_n.set_ylabel("Alignment (r)")

    # --- Non-brain panel (center) ---
    if nonbrain_scores:
        res = _draw_panel(ax_n, nonbrain_scores)
        if res:
            labels_n, pos_nat_n, pos_opt_n = res

            def _bracket_for(labels_subset):
                idxs = [i for i, l in enumerate(labels_n) if l in labels_subset]
                if idxs:
                    x_lo = pos_nat_n[idxs[0]]  - 0.25
                    x_hi = pos_opt_n[idxs[-1]] + 0.25
                    return x_lo, x_hi
                return None

            for group_label, subset in [
                ("Morphology", ["Wug"]),
                ("Syntax",     ["BLiMP"]),
                ("Semantics",  ["Perceptual", "Affective", "Hypernymy"]),
            ]:
                coords = _bracket_for(subset)
                if coords:
                    _add_group_bracket(ax_n, coords[0], coords[1], group_label)

            # Legend: single row, top-left of the pairs panel
            legend_handles = []
            if include_random:
                legend_handles.append(Patch(color="#cccccc", alpha=0.8, label="Random"))
            legend_handles += [
                Patch(color="#9ecae1", alpha=0.8, label="Native"),
                Patch(color="#3182bd", alpha=0.8, label="Optimal"),
            ]
            ax_p.legend(handles=legend_handles,
                        ncol=len(legend_handles),
                        loc="upper left",
                        bbox_to_anchor=(0.0, 1.12),
                        frameon=False, fontsize=7)
    else:
        ax_n.set_visible(False)

    # --- Brain panel (right) ---
    if brain_scores:
        res = _draw_panel(ax_b, brain_scores)
        if res:
            _, pos_nat_b, pos_opt_b = res
            x_lo = pos_nat_b[0] - 0.25
            x_hi = pos_opt_b[-1] + 0.25
            _add_group_bracket(ax_b, x_lo, x_hi, "Brain")
    else:
        ax_b.set_visible(False)

    # Leave room at bottom for group labels; suppress tight_layout warning from clip_on=False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(rect=[0, 0.18, 1, 1])
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_native_vs_optimal_three_joint(results_dir="data/results",
                                        joint_dir="data/results/joint",
                                        save_path=None, include_random=False):
    """
    Variant of plot_native_vs_optimal_three where the middle (features) panel
    uses CSVs from the joint optimal rotation (data/results/joint/), while the
    brain and model-pair panels use the original per-task CSVs.
    """
    plot_native_vs_optimal_three(
        results_dir=results_dir,
        feature_results_dir=joint_dir,
        save_path=save_path,
        include_random=include_random,
    )


_blimp_tasks = [
    "adjunct_island", "anaphor_gender_agreement", "determiner_noun_agreement_1",
    "ellipsis_n_bar_1", "existential_there_object_raising",
    "existential_there_quantifiers_1", "irregular_past_participle_adjectives",
    "irregular_plural_subject_verb_agreement_1",
    "matrix_question_npi_licensor_present", "principle_A_c_command",
    "transitive", "wh_questions_subject_gap",
]


def plot_lang_syntax_combined(results_dir="data/results", normalize=True,
                              plot_title="Syntax (BLiMP)", save_path=None):
    """
    Aggregate alignment curves across all BLiMP tasks per model, then plot.
    Each model has 12 CSVs (lang_{task}_{alias}.csv). We average their
    layerwise curves to get one curve per model.
    """
    from collections import defaultdict

    # Discover model keys from BLiMP CSVs
    model_curves = defaultdict(list)  # model_key -> list of model_avg arrays

    for task_name in _blimp_tasks:
        task_files = sorted(glob(f"{results_dir}/lang_{task_name}_*.csv"))
        for path in task_files:
            model_key = (os.path.basename(path)
                         .replace(f"lang_{task_name}_", "")
                         .replace(".csv", ""))
            if model_key not in model_info:
                continue
            try:
                res = _load_csv(path)
                layerwise, _ = _build_layerwise(res, normalize=normalize)
                model_avg = layerwise.mean(axis=0)
                model_curves[model_key].append(model_avg)
            except Exception as e:
                print(f"  skipping {path}: {e}")

    if not model_curves:
        print(f"  No data for '{plot_title}'")
        return

    # Average across tasks per model
    entries = []
    for model_key, curves in model_curves.items():
        avg = np.mean(curves, axis=0)
        entries.append((model_key, avg))

    entries.sort(key=lambda x: model_order.index(x[0]) if x[0] in model_order else 1e9)

    fig, ax = plt.subplots(figsize=(4.8 * 0.8, 3.2 * 0.8))
    alphas = np.linspace(0, 1, entries[0][1].shape[0])

    for model_key, avg in entries:
        family, label, marker = model_info[model_key]
        ax.plot(alphas, avg, label=label, marker=marker,
                color=family_colors[family], linewidth=1.8, markersize=4)

    ax.set_xlabel(r"Rotation fraction $\alpha$")
    ax.set_ylabel("Alignment / optimal" if normalize else "Pairwise alignment (r)")
    ax.set_title(plot_title)
    ax.grid(True, linestyle="--")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_all_tasks_layerwise(results_dir="data/results", normalize=True,
                             alpha_idx=0, save_path=None):
    """
    Single plot showing mean layerwise alignment for ALL tasks.
    Each line = mean across models; shaded band = ±1 SEM.
    Line labels are placed at the right end of each line, outside the axes.

    Tasks:
      - Pereira Expt. 2 & 3 (brain)
      - Affective & Perceptual (semantic)
      - Morphology (Wug), Lexical (Hypernymy), Syntax (BLiMP aggregate)
    """
    from collections import defaultdict

    # ------------------------------------------------------------------ #
    # 1. Regular tasks: (display_label, glob_prefix, to_replace, to_replace_2)
    # ------------------------------------------------------------------ #
    regular_tasks = [
        ("Pereira Expt. 2",    f"{PEREIRA_RES}/pereira_expt2_*.csv",   "pereira_expt2_",  ""),
        ("Pereira Expt. 3",    f"{PEREIRA_RES}/pereira_expt3_*.csv",   "pereira_expt3_",  ""),
        ("Affective",          f"{results_dir}/affective_*.csv",        "affective_",      "_semantic"),
        ("Perceptual",         f"{results_dir}/perceptual_*.csv",       "perceptual_",     "_semantic"),
        ("Morphology (Wug)",   f"{results_dir}/lang_wug_*.csv",         "lang_wug_",       ""),
        ("Lexical (Hypernymy)",f"{results_dir}/lang_hypernymy_*.csv",   "lang_hypernymy_", ""),
    ]

    n_points = 100
    x_common = np.linspace(0, 1, n_points)

    def _load_curve(path, alpha_idx, normalize):
        """Return (curve, layer_ids) for one model CSV.

        Normalization uses the *median* optimal across layers, making it
        robust to the few near-zero optimal values that appear in early
        layers of some tasks.
        """
        res = _load_csv(path)
        layerwise, layer_ids = _build_layerwise(res, normalize=False, skip_layer0=True)
        curve = layerwise[:, alpha_idx]
        if normalize:
            # Use median optimal across layers (skip layer 0 already applied)
            res_trimmed = res[res["layer"] > 0]
            median_opt = np.median(res_trimmed["optimal"].to_numpy())
            if median_opt > 0:
                curve = curve / median_opt
        return curve, layer_ids

    # Ordered list of (label, mean_curve, sem_curve) for plotting
    task_results = []

    for label, pattern, to_replace, to_replace_2 in regular_tasks:
        files = sorted(glob(pattern))
        files = [f for f in files if not _is_pre_post_act(f)]
        # Prefer _semantic files over plain files for the same model
        files = sorted(files, key=lambda p: (0 if p.endswith("_semantic.csv") else 1))

        seen_keys = set()
        interpolated = []
        for path in files:
            model_key = extract_key(path, to_replace=to_replace, to_replace_2=to_replace_2)
            if model_key not in model_info:
                continue
            if model_key in seen_keys:
                continue
            seen_keys.add(model_key)
            try:
                curve, layer_ids = _load_curve(path, alpha_idx, normalize)
                x = np.linspace(0, 1, len(layer_ids))
                interpolated.append(np.interp(x_common, x, curve))
            except Exception as e:
                print(f"  skipping {path}: {e}")

        if not interpolated:
            print(f"  No data for task '{label}'")
            continue
        arr = np.array(interpolated)
        mean_curve = arr.mean(axis=0)
        sem_curve  = arr.std(axis=0) / np.sqrt(len(arr))
        task_results.append((label, mean_curve, sem_curve))

    # ------------------------------------------------------------------ #
    # 2. BLiMP syntax: average 12 sub-tasks per model, then across models
    # ------------------------------------------------------------------ #
    blimp_model_curves = defaultdict(list)
    for task_name in _blimp_tasks:
        for path in sorted(glob(f"{results_dir}/lang_{task_name}_*.csv")):
            model_key = (os.path.basename(path)
                         .replace(f"lang_{task_name}_", "")
                         .replace(".csv", ""))
            if model_key not in model_info:
                continue
            try:
                curve, layer_ids = _load_curve(path, alpha_idx, normalize)
                x = np.linspace(0, 1, len(layer_ids))
                blimp_model_curves[model_key].append(np.interp(x_common, x, curve))
            except Exception as e:
                print(f"  skipping {path}: {e}")

    if blimp_model_curves:
        # Average 12 sub-tasks → one curve per model
        model_avgs = [np.mean(curves, axis=0) for curves in blimp_model_curves.values()]
        arr = np.array(model_avgs)
        mean_curve = arr.mean(axis=0)
        sem_curve  = arr.std(axis=0) / np.sqrt(len(arr))
        task_results.append(("Syntax (BLiMP)", mean_curve, sem_curve))
    else:
        print("  No BLiMP data found")

    if not task_results:
        print("  No data for all-tasks layerwise plot")
        return

    # ------------------------------------------------------------------ #
    # 3. Plotting
    # ------------------------------------------------------------------ #
    # Qualitative palette — 7 visually distinct colours
    _task_colors = [
        "#1f77b4",   # blue         – Pereira 2
        "#aec7e8",   # light blue   – Pereira 3
        "#d62728",   # red          – Affective
        "#e377c2",   # pink         – Perceptual
        "#2ca02c",   # green        – Morphology
        "#8c564b",   # brown        – Lexical
        "#9467bd",   # purple       – Syntax
    ]

    # Manual vertical nudges (in data units) to separate clustered end-labels.
    # Perceptual / Syntax / Morphology / Affective all end within ~0.09 of each other.
    _label_nudge = {
        "Perceptual":        +0.030,
        "Syntax (BLiMP)":    +0.018,
        "Morphology (Wug)":  -0.015,
        "Affective":         -0.030,
    }

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    for i, (label, mean_curve, sem_curve) in enumerate(task_results):
        color = _task_colors[i % len(_task_colors)]
        ax.plot(x_common, mean_curve, color=color, linewidth=2.0)
        ax.fill_between(x_common, mean_curve - sem_curve, mean_curve + sem_curve,
                        color=color, alpha=0.15)
        # Label at the right end, outside the axes.
        # get_yaxis_transform(): x in axes fraction [0,1], y in data coordinates.
        y_label = float(mean_curve[-1]) + _label_nudge.get(label, 0.0)
        ax.text(1.02, y_label, label,
                transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=7.5, color=color, clip_on=False)

    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel("Alignment / optimal" if normalize else "Pairwise alignment (r)")
    ax.set_title("Alignment across layers — all tasks")
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle="--")
    fig.tight_layout()
    # Shrink the axes so right-side labels are not clipped when saving
    fig.subplots_adjust(right=0.68)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("plots/summary", exist_ok=True)

    for expt, label in [("pereira_expt2", "Pereira Expt. 2"),
                         ("pereira_expt3", "Pereira Expt. 3")]:
        files = glob(f"{PEREIRA_RES}/{expt}_*.csv")
        if not files:
            continue

        # 1) Main rotation plot (no pre/post act)
        print(f"[{label}] Main rotation plot")

        plot_all(files, plot_title=label, to_replace=f"{expt}_",
                 save_path=f"plots/summary/{expt}_rotation_nonorm.png", normalize=False)

        # 2) Pre/post act comparison
        print(f"[{label}] Pre/post activation comparison")
        plot_pre_post_act(expt_prefix=expt,
                          save_path=f"plots/summary/{expt}_pre_post_act.png")



    # Perceptual & affective
    for domain, label in [("perceptual", "Perceptual"), ("affective", "Affective")]:
        files = glob(f"data/results/{domain}_*.csv")
        if not files:
            continue
        print(f"[{label}] Main rotation plot")

        plot_all(files, plot_title=f"{label} alignment",
                 to_replace=f"{domain}_", to_replace_2="_semantic",
                 save_path=f"plots/summary/{domain}_rotation_nonorm.png", normalize=False)


    # Language alignment plots (correlation framework)
    for task_name, label in [("wug", "Morphology (Wug)"), ("hypernymy", "Lexical (Hypernymy)")]:
        files = glob(f"data/results/lang_{task_name}_*.csv")
        if not files:
            continue
        print(f"[Lang] {label}")
        plot_all(files, plot_title=label, to_replace=f"lang_{task_name}_",
                 save_path=f"plots/summary/lang_{task_name}_rotation_nonorm.png",
                 normalize=False)

    # Syntax: aggregate across 12 BLiMP tasks
    if glob("data/results/lang_adjunct_island_*.csv"):
        print("[Lang] Syntax (BLiMP)")
        plot_lang_syntax_combined(
            normalize=False, plot_title="Syntax (BLiMP)",
            save_path="plots/summary/lang_syntax_rotation_nonorm.png")

    # Model-to-model rotation plot
    mm_files = sorted(glob("data/results/model_model_*.csv"))
    if mm_files:
        print("[Model-model] Rotation alignment")
        plot_model_model_alignment(
            mm_files, normalize=False,
            save_path="plots/summary/model_model_rotation.png")

    # Native vs. optimal summary boxplot
    print("[Summary] Native vs. optimal alignment")
    plot_native_vs_optimal_three(
        include_random=True,
        save_path="plots/summary/native_vs_optimal_three.png")

    # Inter-model basis privilege — exploratory; per-task t-tests show privilege
    # is generally significant but does NOT scale systematically with model size
    # or differ by family. Kept in `other_plots/` to avoid cluttering the main
    # summary directory.
    print("[Other] Inter-model basis privilege (no systematic family/size effects)")
    plot_inter_model_privilege(
        metric="difference",
        save_path="plots/summary/other_plots/inter_model_privilege_difference.png")
    plot_inter_model_privilege(
        metric="ratio",
        save_path="plots/summary/other_plots/inter_model_privilege_ratio.png")

    # Joint optimal rotation variant (only if joint CSVs exist)
    if os.path.isdir("data/results/joint") and \
       any(glob("data/results/joint/lang_*.csv")):
        print("[Summary] Native vs. optimal (joint rotation)")
        plot_native_vs_optimal_three_joint(
            include_random=True,
            save_path="plots/summary/native_vs_optimal_three_joint.png")

    # Sparsity — lifetime kurtosis
    kurtosis_files = sorted(glob("data/results/sparsity_kurtosis_*.csv"))
    if kurtosis_files:
        print("[Sparsity] Lifetime kurtosis")
        plot_sparsity_kurtosis(kurtosis_files,
                               plot_title="Lifetime Sparseness",
                               save_path="plots/summary/sparsity_kurtosis.png")

    # Sparsity — readout weight kurtosis (all tasks pooled per model)
    readout_files = sorted(glob("data/results/sparsity_readout_*.csv"))
    if readout_files:
        print("[Sparsity] Readout weight kurtosis")
        plot_sparsity_readout(readout_files,
                              plot_title="Readout Weight Sparsity",
                              save_path="plots/summary/sparsity_readout.png")

    # All-tasks layerwise summary
    print("[Summary] All tasks — layerwise alignment")
    plot_all_tasks_layerwise(
        save_path="plots/summary/all_tasks_layer_depth.png")

    # Checkpoint alignment (training dynamics) — exclude the RSA control CSVs
    ckpt_files = [f for f in sorted(glob("data/results/model_model_checkpoints_*.csv"))
                  if "_rsa_" not in os.path.basename(f)]
    if ckpt_files:
        print("[Checkpoints] Training dynamics alignment")
        plot_checkpoint_alignment(
            ckpt_files,
            save_path="plots/summary/model_model_checkpoints.png")
        plot_checkpoint_ratios(
            ckpt_files,
            save_dir="plots/summary")

    # RSA standalone plot (kept available, but RSA is also overlaid in
    # plot_checkpoint_alignment above)
    rsa_files = sorted(glob("data/results/model_model_checkpoints_rsa_*.csv"))
    if rsa_files:
        print("[Checkpoints] RSA standalone (rotation-invariant control)")
        plot_checkpoint_rsa(
            rsa_files,
            save_path="plots/summary/model_model_checkpoints_rsa.png")

