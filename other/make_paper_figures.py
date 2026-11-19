"""
Paper figures (multi-panel, house style), written to plots/paper/ as SVG + PDF + PNG
and copied into the manuscript's figs/ folder by `--copy` (PAPER_FIGS_DIR=<folder>).

  Fig1.pdf   A model-to-model alignment vs alpha | B model-to-brain (Exp 2, Exp 3)
  Fig2.pdf   A-E model-to-feature alignment vs alpha (Wug, BLiMP, Hypernymy, Perceptual, Affective)
  Fig4.pdf   A readout sparsity vs the random-rotation null (discrete | continuous) | B privilege over training
  FigS1.pdf  absolute training trajectories per Pythia size + RSA control
  FigS2.pdf  BLiMP sub-tasks: native - random per model and sub-task

One legend per figure (models never repeated across panels), panel letters aligned
to the left edge of each panel's tick labels. Curves are the mean over layers > 0
(and participants / sub-tasks), un-normalised r, exactly as in other/paper_stats.py.

Run locally:
  conda activate basis-alignment
  python other/make_paper_figures.py [--copy]
"""
import os, re, sys, glob, shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "other"))
os.chdir(ROOT)
import plotting as P                      # noqa: E402
from paper_stats import targets, per_model_table, load_rows, ALPHAS, RES, PEREIRA_RES, OUT as STATS_OUT, one_sample  # noqa: E402

mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["pdf.fonttype"] = 42

OUT = "plots/paper" + ("" if P.PEREIRA_VOXSET == P._PT.MAIN_VOXSET else f"_voxavg_{P.PEREIRA_VOXSET}")
PAPER_FIGS = os.environ.get("PAPER_FIGS_DIR")   # --copy destination (the manuscript's figs/ folder)
os.makedirs(OUT, exist_ok=True)

SPINE_LW = 1.5
FS_LABEL, FS_TICK, FS_TITLE, FS_LEG, FS_LETTER = 10, 8.5, 10, 8, 13
SEED2 = "EleutherAI_pythia-160m-seed2"
NS_FILL, NS_EDGE = "#cccccc", "#999999"


# ------------------------------------------------------------------ chrome --
def chrome(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(SPINE_LW)
    ax.grid(axis=grid_axis, linestyle="--", alpha=0.5, zorder=0)
    ax.tick_params(labelsize=FS_TICK)


def panel_letters(fig, axes_letters, dx=-0.004):
    """`dx` may be a float or a {letter: dx} dict (per-letter horizontal nudge)."""
    """
    Bold letters at the left edge of each panel's tight bbox (tick labels included),
    all at ONE height: the top of the tallest panel title, so letters sit level with
    the titles instead of floating above them.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    top = max(ax.title.get_window_extent(r).transformed(inv).y1 for ax, _ in axes_letters)
    for ax, letter in axes_letters:
        bb = ax.get_tightbbox(r).transformed(inv)
        d = dx.get(letter, -0.004) if isinstance(dx, dict) else dx
        fig.text(bb.x0 + d, top, letter, fontsize=FS_LETTER, fontweight="bold",
                 ha="left", va="top")


def save(fig, stem):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{OUT}/{stem}.{ext}", format=ext, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT}/{stem}.[svg|pdf|png]")


def model_style(m):
    family, label, marker = P.model_info[m]
    if m == SEED2:
        label = r"Pythia$_{\mathrm{160M}}$ (seed 2)"
    color = P.family_colors[family]
    ls = "--" if m == SEED2 else "-"
    return dict(color=color, marker=marker, linestyle=ls, label=label)


def model_handles(models):
    return [Line2D([0], [0], lw=1.6, markersize=4, **model_style(m)) for m in models]


def curve_panel(ax, tbl, title=None, ylabel=None, xlabel=True):
    """Alignment vs alpha, one line per model (mean over layers), house chrome."""
    for m in [k for k in P.model_order + [SEED2] if k in tbl.index]:
        st = model_style(m); st.pop("label")
        ax.plot(ALPHAS, tbl.loc[m, "curve"], lw=1.6, markersize=3.5, zorder=3, **st)
    chrome(ax)
    ax.set_xticks([0, 0.5, 1])
    if title:
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
    if xlabel:
        ax.set_xlabel(r"Rotation fraction $\alpha$", fontsize=FS_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)


# ------------------------------------------------------------------- Fig 1 --
def fig1(T):
    fig, axes = plt.subplots(1, 3, figsize=(8.6 * .95, 2.7 * .95), dpi=300,
                             gridspec_kw=dict(wspace=0.5))
    curve_panel(axes[0], T["Model-to-model"], "A. Model-to-model", "Pairwise alignment (r)")
    curve_panel(axes[1], T["Pereira Exp 2"], "B. Model-to-brain, Exp. 2", "Pairwise alignment (r)")
    curve_panel(axes[2], T["Pereira Exp 3"], "C. Model-to-brain, Exp. 3")
    models = P.model_order + [SEED2]
    fig.legend(handles=model_handles(models), loc="center left", bbox_to_anchor=(0.91, 0.45),
               frameon=False, fontsize=FS_LEG, handlelength=2.2, labelspacing=0.5)
    save(fig, "Fig1")


# ------------------------------------------------------------------- Fig 2 --
def fig2(T):
    fig, axes = plt.subplots(2, 3, figsize=(8.0 * .95, 5.0 * .95), dpi=300,
                             gridspec_kw=dict(wspace=0.42, hspace=0.55))
    spec = [("Wug", "Morphology (Wug)"), ("BLiMP", "Syntax (BLiMP)"), ("Hypernymy", "Semantics (Hypernymy)"),
            ("Perceptual", "Semantics (Perceptual)"), ("Affective", "Semantics (Affective)")]
    flat = axes.ravel()
    for i, (ax, (key, title)) in enumerate(zip(flat, spec)):
        curve_panel(ax, T[key], title, "Pairwise alignment (r)" if i % 3 == 0 else None,
                    xlabel=(i >= 2))
    flat[5].axis("off")
    flat[5].legend(handles=model_handles(P.model_order), loc="center", bbox_to_anchor=(0.45, 0.5),
                   frameon=False, fontsize=FS_LEG, handlelength=2.0, labelspacing=0.45, ncol=2,
                   columnspacing=1.0)
    save(fig, "Fig2")


# ------------------------------------------------------------------- Fig 4 --
def checkpoint_gaps():
    """{size_label: (steps, gap per step (mean over layers))}, sorted by size."""
    out = {}
    for p in sorted(glob.glob(f"{RES}/model_model_checkpoints_EleutherAI_pythia-*-seed1_vs_*.csv")):
        size = re.search(r"pythia-(\w+)-seed", os.path.basename(p)).group(1).upper()
        df = pd.read_csv(p); df["gap"] = df["native"] - df["random"]
        g = df.groupby("step")["gap"].mean()
        out[size] = (g.index.values.astype(float), g.values)
    return dict(sorted(out.items(), key=lambda kv: int(re.sub(r"\D", "", kv[0]))))


def training_panel(ax):
    gaps = checkpoint_gaps()
    n = len(gaps)
    cols = [plt.cm.Blues(v) for v in np.linspace(0.4, 0.95, n)]
    steps = None
    per = []
    for (size, (st, g)), c in zip(gaps.items(), cols):
        x = st.copy(); x[x == 0] = 0.5
        ax.plot(x, g, color=c, lw=0.9, alpha=0.6, zorder=2, label=size)
        per.append(g); steps = x
    per = np.array(per)
    mean, sem = per.mean(0), per.std(0, ddof=1) / np.sqrt(n)
    ax.plot(steps, mean, color="#1a1a2e", lw=2.2, zorder=4, label="Mean")
    ax.fill_between(steps, mean - sem, mean + sem, color="#1a1a2e", alpha=0.15, lw=0, zorder=3)
    ax.axhline(0, color="#333333", lw=0.9, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.35, 3e5)
    ax.set_xticks([1, 10, 100, 1e3, 1e4, 1e5])
    ax.set_xticklabels(["0", r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
    ax.set_xlabel("Training step", fontsize=FS_LABEL)
    ax.set_ylabel("Basis privilege (native $-$ random)", fontsize=FS_LABEL)
    ax.set_title("Privilege over training", fontsize=FS_TITLE, fontweight="bold")
    chrome(ax)
    h = [Line2D([0], [0], color=c, lw=0.9, alpha=0.8, label=s) for s, c in zip(gaps, cols)]
    h.append(Line2D([0], [0], color="#1a1a2e", lw=2.2, label="Mean $\\pm$ SEM"))
    ax.legend(handles=h, frameon=False, fontsize=FS_LEG, loc="upper left", handlelength=1.8,
              labelspacing=0.4, title="Pythia size", title_fontsize=FS_LEG)


def fig4():
    fig = plt.figure(figsize=(10.0 * .95, 4.3 * .95), dpi=300)
    # A: two ridge sub-panels; B: training panel (wider).
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 0.9, 1.55], wspace=0.12)
    a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1], sharey=a1)
    gs_b = fig.add_gridspec(1, 3, width_ratios=[0.9, 0.9, 1.55], wspace=0.45)
    b = fig.add_subplot(gs_b[2])
    P.plot_readout_sparsity_ridges(sorted(glob.glob(f"{RES}/sparsity_readout_*.csv")),
                                   axes=[a1, a2], label_fontsize=FS_LABEL)
    plt.setp(a2.get_yticklabels(), visible=False)
    for ax in (a1, a2):
        ax.tick_params(labelsize=FS_TICK)
        ax.set_title(ax.get_title(), fontsize=FS_TITLE, fontweight="bold")
    training_panel(b)
    handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#4477AA", markeredgecolor="black",
               markersize=4.5, label="Median native readout"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4477AA", alpha=0.7,
               markeredgecolor="none", markersize=3, label="Single layer"),
        Patch(facecolor=NS_EDGE, alpha=0.25, edgecolor="none", label="Random basis (95%)"),
    ]
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    xa = (a1.get_position().x0 + a2.get_position().x1) / 2      # centre of panel A
    ybot = min(a1.get_tightbbox(fig.canvas.get_renderer()).transformed(inv).y0,
               a2.get_tightbbox(fig.canvas.get_renderer()).transformed(inv).y0)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(xa, ybot - 0.035), ncol=3,
               frameon=False, fontsize=FS_LEG, columnspacing=1.4, handletextpad=0.5)
    panel_letters(fig, [(a1, "A"), (b, "B")], dx={"A": -0.004, "B": 0.012})
    save(fig, "Fig4")


# ------------------------------------------------------------------ Fig S1 --
def figS1():
    ck = sorted(glob.glob(f"{RES}/model_model_checkpoints_EleutherAI_pythia-*-seed1_vs_*.csv"))
    ck = sorted(ck, key=lambda p: int(re.sub(r"\D", "", re.search(r"pythia-(\w+)-seed", p).group(1))))
    n = len(ck)
    fig, axes = plt.subplots(1, n, figsize=(11.5 * .9, 2.9 * .9), dpi=300, sharey=True,
                             gridspec_kw=dict(wspace=0.12))
    c_nat, c_rnd, c_opt, c_rsa = "#3D6FA8", "#767676", "#1F4E79", "#C08A2E"
    rsa_axes = []
    for i, (ax, p) in enumerate(zip(axes, ck)):
        size = re.search(r"pythia-(\w+)-seed", os.path.basename(p)).group(1).upper()
        df = pd.read_csv(p)
        g = df.groupby("step")
        steps = np.array(sorted(g.groups)); x = steps.astype(float); x[x == 0] = 0.5
        for col, c, ls in (("optimal", c_opt, "-."), ("native", c_nat, "-"), ("random", c_rnd, "--")):
            m = np.array([g.get_group(s)[col].mean() for s in steps])
            se = np.array([g.get_group(s)[col].std(ddof=1) / np.sqrt(len(g.get_group(s))) for s in steps])
            ax.plot(x, m, color=c, ls=ls, lw=1.6, zorder=3)
            ax.fill_between(x, m - se, m + se, color=c, alpha=0.18, lw=0, zorder=2)
        rsa_p = glob.glob(f"{RES}/model_model_checkpoints_rsa_EleutherAI_pythia-{size.lower()}_vs_*.csv")
        ax2 = ax.twinx(); rsa_axes.append(ax2)
        if rsa_p:
            r = pd.read_csv(rsa_p[0]).groupby("step")["rsa"].mean()
            xr = r.index.values.astype(float); xr[xr == 0] = 0.5
            ax2.plot(xr, r.values, color=c_rsa, lw=1.1, zorder=3)
        ax2.spines[["top", "left"]].set_visible(False)
        ax2.spines["right"].set_linewidth(SPINE_LW); ax2.spines["right"].set_color(c_rsa)
        ax2.tick_params(axis="y", labelsize=FS_TICK, colors=c_rsa, length=3)
        if i < n - 1:
            plt.setp(ax2.get_yticklabels(), visible=False); ax2.spines["right"].set_visible(False)
            ax2.tick_params(axis="y", length=0)
        ax.set_xscale("log"); ax.set_xlim(0.35, 3e5)
        ax.set_xticks([1, 100, 1e4]); ax.set_xticklabels(["0", r"$10^2$", r"$10^4$"])
        ax.set_title(f"Pythia {size}", fontsize=FS_TITLE, fontweight="bold")
        ax.set_xlabel("Training step", fontsize=FS_LABEL)
        chrome(ax)
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    axes[0].set_ylabel("Pairwise alignment (r)", fontsize=FS_LABEL)
    for a in rsa_axes:
        a.set_ylim(min(x.get_ylim()[0] for x in rsa_axes), max(x.get_ylim()[1] for x in rsa_axes))
    rsa_axes[-1].set_ylabel("RSA (Spearman $\\rho$)", fontsize=FS_LABEL, color=c_rsa)
    h = [Line2D([0], [0], color=c_nat, lw=1.6, label=r"Native ($\alpha=0$)"),
         Line2D([0], [0], color=c_rnd, lw=1.6, ls="--", label=r"Random ($\alpha=1$)"),
         Line2D([0], [0], color=c_opt, lw=1.6, ls="-.", label=r"Optimal ($Q^*$)"),
         Line2D([0], [0], color=c_rsa, lw=1.1, label="RSA (rotation-invariant)")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False, fontsize=FS_LEG)
    panel_letters(fig, list(zip(axes, "ABCDE")))
    save(fig, "FigS1")


# ------------------------------------------------------------------ Fig S2 --
BLIMP_SHORT = {
    "adjunct_island": "Adjunct island", "anaphor_gender_agreement": "Anaphor agr.",
    "determiner_noun_agreement_1": "Det.-noun agr.", "ellipsis_n_bar_1": "Ellipsis",
    "existential_there_object_raising": "Object raising", "existential_there_quantifiers_1": "Quantifiers",
    "irregular_past_participle_adjectives": "Irregular forms",
    "irregular_plural_subject_verb_agreement_1": "Subj.-verb agr.",
    "matrix_question_npi_licensor_present": "NPI licensing", "principle_A_c_command": "Binding",
    "transitive": "Argument str.", "wh_questions_subject_gap": "Filler-gap",
}


def figS2():
    rows = []
    for t in P._blimp_tasks:
        tbl = per_model_table([(f"lang_{t}_", sorted(glob.glob(f"{RES}/lang_{t}_*.csv")))])
        for m, r in tbl.iterrows():
            rows.append(dict(task=t, model=m, gap=r["native"] - r["random"]))
    d = pd.DataFrame(rows)
    summ = []
    for t in P._blimp_tasks:
        s = one_sample(d[d.task == t]["gap"]); s["task"] = t; summ.append(s)
    summ = pd.DataFrame(summ).sort_values("mean", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2 * .95, 3.0 * .95), dpi=300)
    rng = np.random.default_rng(0)
    for i, r in summ.iterrows():
        sub = d[d.task == r["task"]]
        sig = r["p_w"] < 0.05
        for _, q in sub.iterrows():
            st = model_style(q["model"])
            ax.scatter(i + rng.uniform(-0.18, 0.18), q["gap"], s=16, marker=st["marker"],
                       color=st["color"] if sig else NS_FILL, edgecolors="black" if sig else NS_EDGE,
                       linewidths=0.4, alpha=0.75, zorder=3)
        ax.errorbar(i, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                    fmt="o", color="black" if sig else NS_EDGE, markersize=5, capsize=3, lw=1.2, zorder=4)
        ax.text(i, ax.get_ylim()[1] if False else 0.145, P.stars(r["p_w"]) if hasattr(P, "stars") else
                ("***" if r["p_w"] < 0.001 else "**" if r["p_w"] < 0.01 else "*" if r["p_w"] < 0.05 else "n.s."),
                ha="center", va="bottom", fontsize=7.5, color="black" if sig else NS_EDGE)
    ax.axhline(0, color="#333333", lw=0.9, ls="--", zorder=1)
    ax.set_xticks(range(len(summ)))
    ax.set_xticklabels([BLIMP_SHORT[t] for t in summ["task"]], rotation=30, ha="right", fontsize=FS_TICK)
    ax.set_ylabel("Native $-$ random (r)", fontsize=FS_LABEL)
    ax.set_ylim(-0.06, 0.16)
    chrome(ax)
    ax.legend(handles=model_handles(P.model_order), loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=FS_LEG, handlelength=1.5, labelspacing=0.5)
    for h in ax.get_legend().legend_handles:
        h.set_linestyle("none")
    save(fig, "FigS2")   # BLiMP sub-tasks
    summ.to_csv(f"{STATS_OUT}/blimp_subtasks.csv", index=False)


# ----------------------------------------------- depth table (Table S1) --
TARGET_COLORS = {   # same palette as plotting.plot_all_tasks_layerwise
    "Model-to-model": "#444444", "Pereira Exp 2": "#1f77b4", "Pereira Exp 3": "#aec7e8",
    "Wug": "#2ca02c", "BLiMP": "#9467bd", "Hypernymy": "#8c564b",
    "Perceptual": "#e377c2", "Affective": "#d62728",
}
TARGET_LABELS = {"Model-to-model": "Model-to-model", "Pereira Exp 2": "Brain, Exp. 2",
                 "Pereira Exp 3": "Brain, Exp. 3", "Wug": "Morphology (Wug)", "BLiMP": "Syntax (BLiMP)",
                 "Hypernymy": "Semantics (Hypernymy)", "Perceptual": "Semantics (Perceptual)",
                 "Affective": "Semantics (Affective)"}
X_DEPTH = np.linspace(0, 1, 101)


def _layer_profile(path, prefix, suffix=""):
    """Per-layer (layers > 0) native, random, optimal for one CSV, participants averaged."""
    df = load_rows(path, prefix, suffix)
    if "depth_frac" in df:                       # model-to-model: 4 depth-selected layers
        g = df.groupby("depth_frac"); depth = np.array(sorted(g.groups))
    else:
        g = df.groupby("layer"); layers = np.array(sorted(g.groups)); depth = layers / layers.max()
    nat = np.array([np.mean([c[0] for c in g.get_group(k)["curve"]]) for k in sorted(g.groups)])
    rnd = np.array([np.mean([c[-1] for c in g.get_group(k)["curve"]]) for k in sorted(g.groups)])
    opt = np.array([g.get_group(k)["optimal"].astype(float).mean() for k in sorted(g.groups)])
    return df["model"].iloc[0], depth, nat, rnd, opt


def layer_curves():
    """{target: {model: dict(depth, native_raw, random_raw)}} per actual layer (sub-tasks averaged)."""
    spec = {
        "Model-to-model": [("model_model_EleutherAI_pythia-160m_vs_", f"{RES}/model_model_EleutherAI_pythia-160m_vs_*.csv", "")],
        "Pereira Exp 2": [("pereira_expt2_", f"{PEREIRA_RES}/pereira_expt2_*.csv", "")],
        "Pereira Exp 3": [("pereira_expt3_", f"{PEREIRA_RES}/pereira_expt3_*.csv", "")],
        "Wug": [("lang_wug_", f"{RES}/lang_wug_*.csv", "")],
        "BLiMP": [(f"lang_{t}_", f"{RES}/lang_{t}_*.csv", "") for t in P._blimp_tasks],
        "Hypernymy": [("lang_hypernymy_", f"{RES}/lang_hypernymy_*.csv", "")],
        "Perceptual": [("perceptual_", f"{RES}/perceptual_*_semantic.csv", "_semantic")],
        "Affective": [("affective_", f"{RES}/affective_*_semantic.csv", "_semantic")],
    }
    out = {}
    for target, groups in spec.items():
        acc = {}
        for prefix, pattern, suffix in groups:
            for path in sorted(glob.glob(pattern)):
                m, depth, nat, rnd, opt = _layer_profile(path, prefix, suffix)
                if m not in P.model_info:
                    continue
                d = acc.setdefault(m, dict(native_raw=[], random_raw=[], depth=depth))
                d["native_raw"].append(nat); d["random_raw"].append(rnd)
        out[target] = {m: dict(depth=d["depth"], native_raw=np.mean(d["native_raw"], axis=0),
                               random_raw=np.mean(d["random_raw"], axis=0)) for m, d in acc.items()}
    return out


def _depth_index(depth, dep):
    """Index of the layer nearest to normalised depth `dep`.

    Ties (e.g. Gemma-3-1B, 26 layers: 0.75 * 26 = 19.5) are broken with Python's
    round() (half to even), the rule other/model_model_alignment.py uses to pick its
    depth-matched layers, so every row of the depth table uses the same layer per model.
    """
    d = np.abs(np.asarray(depth) - dep)
    cand = np.flatnonzero(np.isclose(d, d.min(), atol=1e-9))
    if len(cand) == 1:
        return int(cand[0])
    n_layers = round(1 / (depth[1] - depth[0]))       # depth = layer / n_layers, consecutive layers
    target = round(dep * n_layers)                     # half to even, as in model_model_alignment
    return int(next(j for j in cand if round(depth[j] * n_layers) == target))


def depth_table(LC, depths=(0.25, 0.5, 0.75, 1.0)):
    """
    Native - random at fixed normalised depths, one value per model, NO interpolation:
    for each model the layer whose normalised depth (layer / n_layers) is nearest to the
    target depth is used (exact for model-to-model, whose 4 layers are depth-selected).
    Wilcoxon across models at each depth.
    """
    rows = []
    for target, models in LC.items():
        for dep in depths:
            gaps, nat = [], []
            for v in models.values():
                j = _depth_index(v["depth"], dep)
                gaps.append(v["native_raw"][j] - v["random_raw"][j]); nat.append(v["native_raw"][j])
            s = one_sample(np.array(gaps))
            rows.append(dict(target=target, depth=dep, native=np.mean(nat), gap=s["mean"], ci_lo=s["ci_lo"],
                             ci_hi=s["ci_hi"], n_pos=s["n_pos"], n=s["n"], p_w=s["p_w"]))
    t = pd.DataFrame(rows)
    t.to_csv(f"{STATS_OUT}/depth_table.csv", index=False)
    return t


if __name__ == "__main__":
    T = targets()
    fig1(T); fig2(T); fig4(); figS1(); figS2()
    dt = depth_table(layer_curves())
    print(dt.round(4).to_string())
    if "--copy" in sys.argv:
        if not PAPER_FIGS:
            sys.exit("--copy needs PAPER_FIGS_DIR=<manuscript figs/ folder>")
        if P.PEREIRA_VOXSET != P._PT.MAIN_VOXSET:
            sys.exit(f"--copy refused: PEREIRA_VOXSET={P.PEREIRA_VOXSET} is the SI variant")
        for stem in ("Fig1", "Fig2", "Fig4", "FigS1", "FigS2"):
            shutil.copy(f"{OUT}/{stem}.pdf", f"{PAPER_FIGS}/{stem}.pdf")
        shutil.copy("plots/summary/native_vs_optimal_three.png", f"{PAPER_FIGS}/Fig3.png")
        print("  copied to", PAPER_FIGS)
