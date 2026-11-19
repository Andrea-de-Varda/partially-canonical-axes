"""Tests for the readout-sparsity-vs-null aggregation and figures (plotting.py)."""
import os, sys, tempfile
import numpy as np, pandas as pd
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import matplotlib; matplotlib.use("Agg")
import plotting as P

fails = []
def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + extra) if extra else ''}")
    if not cond: fails.append(name)

def write_csv(d, task, model, native, null_mu=0.0, null_sd=0.1, layers=3, seeds=10, rng=None):
    rows = []
    for l in range(1, layers + 1):   # layer 0 (embedding) is excluded by _readout_cells
        rows.append(dict(layer=l, path="native", alpha=0.0, seed=-1, kurtosis_weights=native))
        for s in range(seeds):
            for a in np.round(np.arange(0.1, 1.01, 0.1), 1):
                val = null_mu + null_sd * rng.standard_normal() if np.isclose(a, 1.0) \
                      else native * (1 - a) + null_mu
                rows.append(dict(layer=l, path="random", alpha=a, seed=s, kurtosis_weights=val))
    p = os.path.join(d, f"sparsity_readout_{task}_{model}.csv")
    pd.DataFrame(rows).to_csv(p, index=False); return p

rng = np.random.default_rng(0)
with tempfile.TemporaryDirectory() as d:
    files = [
        write_csv(d, "wug", "gpt2", native=2.0, rng=rng),          # far above null -> significant
        write_csv(d, "wug", "gpt2-xl", native=0.02, rng=rng),      # inside null -> not
        write_csv(d, "sem", "gpt2", native=0.02, rng=rng),         # sem: inside null
        write_csv(d, "sem", "gpt2-xl", native=1.5, rng=rng),       # sem: above null
        # "gpt2" must NOT swallow "gpt2-xl" files (longest-suffix match)
    ]
    cells = P._readout_cells(files)

    print("1. _readout_cells")
    check("   one row per (model, task, layer)", len(cells) == 4 * 3, f"{len(cells)} rows")
    check("   longest-suffix model match (gpt2 vs gpt2-xl)",
          set(cells[cells.task == "wug"].model) == {"gpt2", "gpt2-xl"})
    check("   native recovered", np.allclose(cells[(cells.model == "gpt2") & (cells.task == "wug")].native, 2.0))
    check("   curve starts at native", all(np.isclose(c[0], n) for c, n in zip(cells.curve, cells.native)))
    check("   null_sd is the empirical SD over seeds at alpha=1",
          0.05 < cells.null_sd.median() < 0.2, f"{cells.null_sd.median():.3f}")

    print("\n2. _readout_groups separates sem from language tasks")
    (_, lang), (_, sem) = P._readout_groups(cells)
    check("   language group has no sem rows", (lang.task != "sem").all() and len(lang) == 6)
    check("   sem group has only sem rows", (sem.task == "sem").all() and len(sem) == 6)

    print("\n3. _readout_summary significance == native outside null band")
    sl, ss = P._readout_summary(lang), P._readout_summary(sem)
    check("   language: gpt2 native 2.0 is significant", sl["gpt2"]["significant"])
    check("   language: gpt2-xl native 0.02 is NOT", not sl["gpt2-xl"]["significant"])
    check("   sem: gpt2 0.02 NOT, gpt2-xl 1.5 IS", (not ss["gpt2"]["significant"]) and ss["gpt2-xl"]["significant"])
    for g in (sl, ss):
        for m, r in g.items():
            check(f"   band consistent for {m}", np.isclose(r["hi"] - r["null_mean"], 1.96 * r["null_sd"])
                  and r["significant"] == (r["native"] > r["hi"]))

    print("\n3b. null of the median, not of a single cell")
    check("   null_sd is the SD over seeds of the per-seed median across cells",
          all(np.isclose(r["null_sd"], np.std([np.median([dd[k] for dd in g.null_seeds]) for k in range(10)], ddof=1))
              for (m, g), r in ((mg, sl[mg[0]]) for mg in lang.groupby("model"))))
    ls = P._readout_layer_summary(lang)
    check("   layer summary has one entry per (model, layer)", len(ls) == 2 * 3)
    check("   layer-level significance: gpt2 wug layers above, gpt2-xl not",
          all(ls[("gpt2", l)]["significant"] for l in range(1, 4)) and not any(ls[("gpt2-xl", l)]["significant"] for l in range(1, 4)))
    # The GPT-2 case: native ~1.3 single-cell SDs above the null in every one of many
    # cells. Inside the single-cell band, but far outside the null of the median.
    many = [write_csv(d, f"t{i}", "gpt2-medium", native=0.13, null_sd=0.1, layers=10, rng=rng) for i in range(14)]
    r = P._readout_summary(P._readout_cells(many))["gpt2-medium"]
    check("   modest shift over 140 cells is significant", r["significant"] and r["null_sd"] < 0.05,
          f"native {r['native']:.3f}, hi {r['hi']:.3f}, sd {r['null_sd']:.3f}")
    many0 = [write_csv(d, f"u{i}", "gpt2-large", native=0.0, null_sd=0.1, layers=10, rng=rng) for i in range(14)]
    check("   native at the null over 140 cells is not", not P._readout_summary(P._readout_cells(many0))["gpt2-large"]["significant"])

    print("\n3c. pooled curve weights discrete and continuous features equally")
    pc = P._readout_pooled_curve(cells)
    check("   gpt2: native = mean(discrete 2.0, continuous 0.02)", np.isclose(pc["gpt2"]["native"], 1.01))
    check("   gpt2-xl: native = mean(0.02, 1.5)", np.isclose(pc["gpt2-xl"]["native"], 0.76))
    check("   curve[0] == native and band consistent",
          all(np.isclose(r["curve"][0], r["native"]) and np.isclose(r["hi"] - r["null_mean"], 1.96 * r["null_sd"]) for r in pc.values()))
    P.plot_readout_sparsity_raw_pooled(files, save_path=os.path.join(d, "figs", "pooled.png"))
    check("   pooled figure writes svg + png", all(os.path.exists(os.path.join(d, "figs", f)) for f in ("pooled.svg", "pooled.png")))
    check("   pooled returns None on no files", P.plot_readout_sparsity_raw_pooled([]) is None)

    print("\n4. both figures write SVG + PNG")
    out = os.path.join(d, "figs")
    P.plot_readout_sparsity_ridges(files, save_path=os.path.join(out, "dumb.png"))
    P.plot_readout_sparsity_raw(files, save_path=os.path.join(out, "raw.png"))
    for f in ("dumb.svg", "dumb.png", "raw.svg", "raw.png"):
        check(f"   {f} exists", os.path.exists(os.path.join(out, f)))
    svg = open(os.path.join(out, "dumb.svg")).read()
    check("   SVG text is editable (not paths)", "<text" in svg)

    print("\n5. empty input degrades gracefully")
    check("   ridges returns None on no files", P.plot_readout_sparsity_ridges([]) is None)

print("\n" + ("ALL PASSED" if not fails else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)
