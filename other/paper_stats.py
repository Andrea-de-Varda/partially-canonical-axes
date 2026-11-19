"""
Statistics for the paper, computed from the CURRENT result CSVs in data/results/.

Unit of analysis is the model (one scalar per model per alignment target; mean
over layers > 0 and, where applicable, over participants / sub-tasks), exactly as
the summary figures aggregate. n = 12 models (12 pairs for model-to-model).

Per target:
  * privilege  = native (alpha = 0) - random (alpha = 1)     -> one-sample tests vs 0
  * ceiling    = optimal - native                            -> one-sample tests vs 0
  * monotonic decline: per-model Spearman rho between alpha and alignment (11 alphas),
    combined across models with Stouffer's Z on one-sided p-values; per-model OLS
    slope tested across models as a check.
  * Spearman between privilege and log(parameter count).
Primary test: exact two-sided Wilcoxon signed-rank (robust to a single outlier
model); the one-sample t and its 95% CI are reported alongside. Holm correction
across the 8 primary privilege tests.

Pereira additionally: the weak-identification audit of the optimal ceiling
(fraction of (layer, participant) cells with optimal < native, <= 0). Participants are
averaged over: no participant-level model.

Training dynamics: per-size gap at step 0 and at the last step, Spearman between
log10(step + 1) and gap per size, paired test last vs first across the 5 sizes.

Readout sparsity: per-model median native kurtosis vs the null of the median over
10 rotation seeds (plotting._readout_pooled), discrete / continuous / pooled.

Run locally:
  conda activate basis-alignment
  python other/paper_stats.py
Writes data/results/stats/paper_stats.md and per-model CSV tables.
"""
import os, sys, glob, warnings
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import plotting as P  # noqa: E402

RES = "data/results"
PEREIRA_RES = P.PEREIRA_RES          # voxel-set directory, set by PEREIRA_VOXSET (default top20rel)
# SI variant (PEREIRA_VOXSET=all) writes to its own folder so it never overwrites the main stats.
OUT = "data/results/stats" + ("" if P.PEREIRA_VOXSET == P._PT.MAIN_VOXSET else f"_voxavg_{P.PEREIRA_VOXSET}")
os.makedirs(OUT, exist_ok=True)
ALPHAS = np.round(np.arange(0, 1.01, 0.1), 1)

PARAMS_M = {
    "gpt2": 124, "gpt2-medium": 355, "gpt2-large": 774, "gpt2-xl": 1500,
    "meta-llama_Llama-3.2-1B": 1230, "meta-llama_Llama-3.2-3B": 3210,
    "google_gemma-3-1b-pt": 1000, "state-spaces_mamba-130m-hf": 130,
    "state-spaces_mamba-370m-hf": 370, "state-spaces_mamba-790m-hf": 790,
    "Qwen_Qwen3-0.6B-Base": 600, "EleutherAI_pythia-160m": 160,
    "EleutherAI_pythia-160m-seed2": 160,
}
LABEL = {k: v[1] for k, v in P.model_info.items()}
# Residual-stream width (hidden_size in each model's Hugging Face config).
D_MODEL = {"gpt2": 768, "gpt2-medium": 1024, "gpt2-large": 1280, "gpt2-xl": 1600,
           "meta-llama_Llama-3.2-1B": 2048, "meta-llama_Llama-3.2-3B": 3072,
           "google_gemma-3-1b-pt": 1152, "state-spaces_mamba-130m-hf": 768,
           "state-spaces_mamba-370m-hf": 1024, "state-spaces_mamba-790m-hf": 1536,
           "Qwen_Qwen3-0.6B-Base": 1024, "EleutherAI_pythia-160m": 768}


# ----------------------------------------------------------------- loaders --
def _arr(s):
    return np.fromstring(s.strip("[]"), sep=" ")


def load_rows(path, prefix, suffix=""):
    """Rows with layer > 0 of one results CSV: columns curve (11), optimal (+ part)."""
    df = pd.read_csv(path)
    df["curve"] = df["mean"].apply(_arr)
    if "layer" in df:
        df = df[df["layer"] > 0]
    key = P.extract_key(path, to_replace=prefix, to_replace_2=suffix)
    df["model"] = key
    return df


def per_model_table(file_groups, suffix=""):
    """
    file_groups: list of (prefix, [paths]) -- one entry per sub-task.
    Returns per-model DataFrame: curve (mean over all rows), native, random, optimal.
    Sub-tasks are weighted equally (each contributes its own layer-mean).
    """
    rows = []
    for prefix, paths in file_groups:
        for p in paths:
            df = load_rows(p, prefix, suffix)
            key = df["model"].iloc[0]
            if key not in P.model_info:
                continue
            rows.append(dict(model=key, curve=np.mean(np.stack(df["curve"]), axis=0),
                             optimal=df["optimal"].astype(float).mean()))
    d = pd.DataFrame(rows)
    out = []
    for m, g in d.groupby("model"):
        curve = np.mean(np.stack(g["curve"]), axis=0)
        out.append(dict(model=m, label=LABEL[m], curve=curve, native=curve[0],
                        random=curve[-1], optimal=g["optimal"].mean(),
                        n_subtasks=len(g)))
    return pd.DataFrame(out).set_index("model")


def targets():
    T = {}
    T["Model-to-model"] = per_model_table(
        [("model_model_EleutherAI_pythia-160m_vs_",
          sorted(glob.glob(f"{RES}/model_model_EleutherAI_pythia-160m_vs_*.csv")))])
    T["Pereira Exp 2"] = per_model_table([("pereira_expt2_", sorted(glob.glob(f"{PEREIRA_RES}/pereira_expt2_*.csv")))])
    T["Pereira Exp 3"] = per_model_table([("pereira_expt3_", sorted(glob.glob(f"{PEREIRA_RES}/pereira_expt3_*.csv")))])
    T["Wug"] = per_model_table([("lang_wug_", sorted(glob.glob(f"{RES}/lang_wug_*.csv")))])
    T["BLiMP"] = per_model_table([(f"lang_{t}_", sorted(glob.glob(f"{RES}/lang_{t}_*.csv"))) for t in P._blimp_tasks])
    T["Hypernymy"] = per_model_table([("lang_hypernymy_", sorted(glob.glob(f"{RES}/lang_hypernymy_*.csv")))])
    T["Perceptual"] = per_model_table([("perceptual_", sorted(glob.glob(f"{RES}/perceptual_*_semantic.csv")))], "_semantic")
    T["Affective"] = per_model_table([("affective_", sorted(glob.glob(f"{RES}/affective_*_semantic.csv")))], "_semantic")
    return T


# ------------------------------------------------------------------- tests --
def one_sample(x, mu=0.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    d = x - mu
    t, p_t = stats.ttest_1samp(x, mu)
    ci = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=stats.sem(x))
    try:
        w = stats.wilcoxon(d, alternative="two-sided", method="exact")
        W, p_w = w.statistic, w.pvalue
    except ValueError:
        W, p_w = np.nan, np.nan
    return dict(n=n, mean=x.mean(), sd=x.std(ddof=1), ci_lo=ci[0], ci_hi=ci[1],
                t=t, p_t=p_t, W=W, p_w=p_w, n_pos=int((d > 0).sum()),
                dz=x.mean() / x.std(ddof=1))


def fmt_p(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def holm(pvals):
    pvals = np.asarray(pvals, float)
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, pvals[i] * (m - rank))
        adj[i] = min(1.0, running)
    return adj


def monotonic(tbl):
    """
    Per-model Kendall tau(alpha, curve) over the 11 alphas (exact p), combined across
    models with Stouffer's method on ONE-SIDED p-values (H1: tau < 0), so that models whose
    alignment increases with alpha count against the hypothesis:
        Z = sum_i z_i / sqrt(k),  z_i = Phi^-1(1 - p_i^{one-sided}).
    Models are independent, which is what Stouffer requires. Also returns the
    per-model OLS slope tested across models (a second, distribution-free check).
    """
    rhos, slopes, zs = [], [], []
    for m, r in tbl.iterrows():
        # Kendall tau with the EXACT permutation p (finite even for a perfectly
        # monotone curve: 1/11! one-sided); Spearman's t-approximation returns p = 0
        # at rho = -1 and breaks Stouffer. Spearman rho is still reported descriptively.
        kt = stats.kendalltau(ALPHAS, r["curve"], method="exact", alternative="less")
        zs.append(stats.norm.isf(kt.pvalue))
        rhos.append(stats.spearmanr(ALPHAS, r["curve"]).correlation)
        slopes.append(np.polyfit(ALPHAS, r["curve"], 1)[0])
    tbl["rho_alpha"] = rhos
    tbl["slope_alpha"] = slopes
    Z = np.sum(zs) / np.sqrt(len(zs))
    return dict(Z=Z, p=stats.norm.sf(Z), slope=one_sample(slopes), n_neg=int(np.sum(np.array(rhos) < 0)),
                n_strong=int(np.sum(np.array(rhos) <= -0.9)), mean_rho=float(np.mean(rhos)))


# ------------------------------------------------------------------ report --
L = []


def say(s=""):
    print(s)
    L.append(s)


def main():
    T = targets()
    say("# Paper statistics (from data/results)\n")
    say(f"Metric tag: {__import__('metric_config').METRIC_TAG}; pooling: "
        f"{__import__('pooling').POOLING_TAG}\n")

    # ---------------------------------------------------------- summary ----
    say("## 1. Privilege (native - random) per target, one observation per model\n")
    say("| Target | n | random | native | optimal | gap mean [95% CI] | d_z | t | p_t | W | p_W (exact) | models > 0 |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    prim_p, prim_names, summary_rows = [], [], []
    for name, tbl in T.items():
        gap = tbl["native"] - tbl["random"]
        s = one_sample(gap)
        prim_p.append(s["p_w"]); prim_names.append(name)
        say(f"| {name} | {s['n']} | {tbl['random'].mean():.4f} | {tbl['native'].mean():.4f} | "
            f"{tbl['optimal'].mean():.4f} | {s['mean']:+.4f} [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] | "
            f"{s['dz']:.2f} | {s['t']:.2f} | {fmt_p(s['p_t'])} | {s['W']:.0f} | {fmt_p(s['p_w'])} | "
            f"{s['n_pos']}/{s['n']} |")
        summary_rows.append(dict(target=name, **s, random=tbl['random'].mean(),
                                 native=tbl['native'].mean(), optimal=tbl['optimal'].mean()))
    adj = holm(prim_p)
    say("\nHolm-adjusted Wilcoxon p across the 8 targets: " +
        "; ".join(f"{n}: {fmt_p(a)}" for n, a in zip(prim_names, adj)) + "\n")

    say("## 2. Ceiling (optimal - native) per target\n")
    say("| Target | gap mean [95% CI] | native/optimal | t | p_t | W | p_W | models > 0 |")
    say("|---|---|---|---|---|---|---|---|")
    for name, tbl in T.items():
        s = one_sample(tbl["optimal"] - tbl["native"])
        ratio = tbl["native"].mean() / tbl["optimal"].mean()
        say(f"| {name} | {s['mean']:+.4f} [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] | {ratio:.2f} | "
            f"{s['t']:.2f} | {fmt_p(s['p_t'])} | {s['W']:.0f} | {fmt_p(s['p_w'])} | {s['n_pos']}/{s['n']} |")

    say("\n## 3. Monotonic decline with alpha (per-model Kendall tau over 11 alphas, exact one-sided p, Stouffer-combined; Spearman rho descriptive; OLS slope tested across models)\n")
    say("| Target | models rho<0 | models rho<=-0.9 | mean rho | Stouffer Z | p (one-sided) | slope mean | t | p_t | p_W |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for name, tbl in T.items():
        mo = monotonic(tbl); s = mo["slope"]
        say(f"| {name} | {mo['n_neg']}/{len(tbl)} | {mo['n_strong']}/{len(tbl)} | {mo['mean_rho']:.2f} | {mo['Z']:.2f} | {fmt_p(mo['p'])} | "
            f"{s['mean']:+.4f} | {s['t']:.2f} | {fmt_p(s['p_t'])} | {fmt_p(s['p_w'])} |")

    say("\n## 4. Privilege vs model size (Spearman with log10 params)\n")
    say("| Target | n | rho | p |")
    say("|---|---|---|---|")
    for name, tbl in T.items():
        d = tbl.copy()
        d["gap"] = d["native"] - d["random"]
        d["logp"] = [np.log10(PARAMS_M[m]) for m in d.index]
        d = d[d.index != "EleutherAI_pythia-160m-seed2"]
        r = stats.spearmanr(d["logp"], d["gap"])
        say(f"| {name} | {len(d)} | {r.correlation:+.2f} | {fmt_p(r.pvalue)} |")

    # ----------------------------------------------------- per-model table --
    say("\n## 5. Per-model values (native / random / optimal; gap = native - random)\n")
    wide = []
    for name, tbl in T.items():
        for m, r in tbl.iterrows():
            wide.append(dict(target=name, model=m, label=r["label"], native=r["native"],
                             random=r["random"], optimal=r["optimal"],
                             gap=r["native"] - r["random"], rho_alpha=r["rho_alpha"]))
    wide = pd.DataFrame(wide)
    wide.to_csv(f"{OUT}/per_model_scores.csv", index=False)
    piv = wide.pivot(index="label", columns="target", values="gap")
    say(piv.round(4).to_markdown())
    say(f"\n(Full per-model table with native/random/optimal: {OUT}/per_model_scores.csv)\n")

    # ------------------------------------------------- model-to-model detail --
    mm = T["Model-to-model"]
    seed = mm.loc["EleutherAI_pythia-160m-seed2"]
    others = mm.drop("EleutherAI_pythia-160m-seed2")
    say("## 6. Model-to-model detail\n")
    say(f"Seed pair: native {seed['native']:.3f}, random {seed['random']:.3f}, gap {seed['native']-seed['random']:+.4f}, optimal {seed['optimal']:.3f}")
    g = others["native"] - others["random"]
    say(f"Other 11 pairs: native {others['native'].mean():.3f}, random {others['random'].mean():.3f}, "
        f"gap {g.mean():+.4f} (range {g.min():+.4f} to {g.max():+.4f}); seed-pair gap z relative to the other pairs: "
        f"{(seed['native']-seed['random']-g.mean())/g.std(ddof=1):+.2f}\n")

    # ------------------------------------------------ Pereira ceiling audit --
    say("## 7. Pereira: ceiling audit over (layer, participant) cells\n")
    for exp in ("expt2", "expt3"):
        cells, audit = [], []
        for p in sorted(glob.glob(f"{PEREIRA_RES}/pereira_{exp}_*.csv")):
            df = load_rows(p, f"pereira_{exp}_")
            m = df["model"].iloc[0]
            if m not in P.model_info:
                continue
            nat = df["curve"].apply(lambda c: c[0]); rnd = df["curve"].apply(lambda c: c[-1])
            opt = df["optimal"].astype(float)
            audit.append(dict(model=m, n=len(df), opt_lt_nat=(opt < nat).mean(), opt_le0=(opt <= 0).mean(),
                              nat_gt_rnd=(nat > rnd).mean()))
            for part, gp in df.groupby("part"):
                cells.append(dict(model=m, part=int(part),
                                  gap=np.mean([c[0] - c[-1] for c in gp["curve"]]),
                                  ceiling=np.mean(gp["optimal"].astype(float) - gp["curve"].apply(lambda c: c[0]))))
        cells = pd.DataFrame(cells); audit = pd.DataFrame(audit)
        cells.to_csv(f"{OUT}/pereira_{exp}_cells.csv", index=False)
        say(f"Exp {exp[-1]} ceiling audit over (layer, participant) cells, mean across models: optimal < native in "
            f"{audit['opt_lt_nat'].mean()*100:.0f}% (range {audit['opt_lt_nat'].min()*100:.0f}-{audit['opt_lt_nat'].max()*100:.0f}%), "
            f"optimal <= 0 in {audit['opt_le0'].mean()*100:.0f}%; native > random in {audit['nat_gt_rnd'].mean()*100:.0f}% of cells\n")

    # ----------------------------------------------------- BLiMP sub-tasks --
    say("## 8. BLiMP per sub-task (Wilcoxon across models)\n")
    say("| Sub-task | gap mean | models > 0 | p_W |")
    say("|---|---|---|---|")
    n_sig = 0
    for t in P._blimp_tasks:
        tbl = per_model_table([(f"lang_{t}_", sorted(glob.glob(f"{RES}/lang_{t}_*.csv")))])
        s = one_sample(tbl["native"] - tbl["random"])
        n_sig += s["p_w"] < 0.05
        say(f"| {t} | {s['mean']:+.4f} | {s['n_pos']}/{s['n']} | {fmt_p(s['p_w'])} |")
    say(f"\n{n_sig}/12 sub-tasks significant at p < 0.05 (uncorrected)\n")

    # ------------------------------------------------------- checkpoints ----
    say("## 9. Training dynamics (Pythia seed pairs; gap = native - random, mean over layers)\n")
    ck = sorted(glob.glob(f"{RES}/model_model_checkpoints_EleutherAI_pythia-*-seed1_vs_*.csv"))
    import re
    rows = []
    say("| Size | steps | gap step 0 | layers > 0 at step 0 | gap last step | rho(log step, gap) | p | step of max gap |")
    say("|---|---|---|---|---|---|---|---|")
    for p in ck:
        size = re.search(r"pythia-(\w+)-seed", os.path.basename(p)).group(1)
        df = pd.read_csv(p); df["gap"] = df["native"] - df["random"]
        g = df.groupby("step")["gap"].mean()
        steps = g.index.values
        r = stats.spearmanr(np.log10(steps + 1), g.values)
        cells0 = df[df["step"] == 0]["gap"]
        rows.append(dict(size=size, gap0=g.iloc[0], gap_last=g.iloc[-1], rho=r.correlation, p=r.pvalue,
                         n_layers=len(cells0), n_pos0=int((cells0 > 0).sum()), num=int(re.sub(r"\D", "", size))))
        say(f"| {size} | {len(steps)} | {g.iloc[0]:+.4f} | {int((cells0>0).sum())}/{len(cells0)} | {g.iloc[-1]:+.4f} | "
            f"{r.correlation:+.2f} | {fmt_p(r.pvalue)} | {g.idxmax()} |")
    cr = pd.DataFrame(rows).sort_values("num")
    s0 = one_sample(cr["gap0"]); sl = one_sample(cr["gap_last"]); sd = one_sample(cr["gap_last"] - cr["gap0"])
    say(f"\nStep 0 across sizes: mean {s0['mean']:+.4f} [{s0['ci_lo']:+.4f}, {s0['ci_hi']:+.4f}], t(4) = {s0['t']:.2f}, "
        f"p_t = {fmt_p(s0['p_t'])}, Wilcoxon p = {fmt_p(s0['p_w'])}, {s0['n_pos']}/5 > 0; "
        f"all (size, layer) cells at step 0: {cr['n_pos0'].sum()}/{cr['n_layers'].sum()} > 0 "
        f"(binomial p = {fmt_p(stats.binomtest(int(cr['n_pos0'].sum()), int(cr['n_layers'].sum())).pvalue)})")
    say(f"Last step across sizes: mean {sl['mean']:+.4f} [{sl['ci_lo']:+.4f}, {sl['ci_hi']:+.4f}], t(4) = {sl['t']:.2f}, p_t = {fmt_p(sl['p_t'])}")
    say(f"Last - first: mean {sd['mean']:+.4f}, t(4) = {sd['t']:.2f}, p_t = {fmt_p(sd['p_t'])}, Wilcoxon p = {fmt_p(sd['p_w'])}, {sd['n_pos']}/5\n")
    cr.to_csv(f"{OUT}/checkpoints_summary.csv", index=False)

    # ---------------------------------------------------- readout sparsity --
    say("## 10. Readout sparsity: median native kurtosis vs null of the median (10 rotation seeds)\n")
    cells = P._readout_cells(sorted(glob.glob(f"{RES}/sparsity_readout_*.csv")))
    say("| Model | d | discrete native | null mean +- SD | z | sig | continuous native | null | z | sig | pooled native | z | sig |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    groups = dict(P._readout_groups(cells))
    disc = P._readout_summary(list(groups.values())[0]); cont = P._readout_summary(list(groups.values())[1])
    pooled = P._readout_pooled_curve(cells)
    dims = {}
    for m in P.model_order:
        a, b, c = disc[m], cont[m], pooled[m]
        za = (a["native"] - a["null_mean"]) / a["null_sd"]; zb = (b["native"] - b["null_mean"]) / b["null_sd"]
        zc = (c["native"] - c["null_mean"]) / c["null_sd"]
        say(f"| {LABEL[m]} | {D_MODEL[m]} | {a['native']:.3f} | {a['null_mean']:+.3f} +- {a['null_sd']:.3f} | {za:.1f} | {'*' if a['significant'] else ''} | "
            f"{b['native']:.3f} | {b['null_mean']:+.3f} +- {b['null_sd']:.3f} | {zb:.1f} | {'*' if b['significant'] else ''} | "
            f"{c['native']:.3f} | {zc:.1f} | {'*' if c['significant'] else ''} |")
    say(f"\nDiscrete: {sum(v['significant'] for v in disc.values())}/12 above null; continuous: "
        f"{sum(v['significant'] for v in cont.values())}/12; pooled: {sum(v['significant'] for v in pooled.values())}/12")
    # alpha=1 endpoint check: mean over cells of null_mean. Under a uniformly random
    # rotation each readout vector is uniform on the sphere, so its sample excess
    # kurtosis has expectation -6/(d+1) (that of d i.i.d. Gaussian coordinates).
    pred = np.mean([-6 / (D_MODEL[m] + 1) for m in cells["model"]])
    say(f"alpha = 1 endpoint: per-cell null mean over all cells = {cells['null_mean'].mean():+.4f} "
        f"(SD across cells {cells['null_mean'].std():.3f}); expected under a uniformly random rotation "
        f"-6/(d+1) per cell, i.e. {pred:+.4f} averaged over the same cells (a median is more negative).\n")

    with open(f"{OUT}/paper_stats.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nWritten {OUT}/paper_stats.md")


if __name__ == "__main__":
    main()
