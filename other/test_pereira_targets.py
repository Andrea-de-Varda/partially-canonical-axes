"""Tests for the repetition-averaged Pereira brain targets (pereira_targets.py) and
the Pereira runner (voxel-set directories, voxel-level count columns).

  python other/test_pereira_targets.py            # unit tests + real-data tests (if the brain data is present)
  python other/test_pereira_targets.py --parquet  # + cross-check against the lab's raw parquet (not distributed)
  python other/test_pereira_targets.py --smoke    # + end-to-end run of the runner on gpt2 (needs a CUDA GPU, ~minutes)

Section 4 needs data/pereira_language.npz (see README, "Brain data") and is
skipped without it. The smoke test needs geotorch/torchmetrics only for the
optimal rotation; when they are missing it writes functional stand-ins (torch's
matrix-exp orthogonal parametrization; cosine similarity) into a temp dir on
PYTHONPATH, so spawned workers see them too.
"""
import os, sys, tempfile, subprocess, textwrap, glob
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
import numpy as np
import pereira_targets as PT

fails = []
def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + extra) if extra else ''}")
    if not cond: fails.append(name)

rng = np.random.default_rng(0)

print("1. session_blocks: column j*n_vox + v is voxel v of session j")
n, nv, k = 7, 5, 3
Y = np.arange(n * k * nv, dtype=float).reshape(n, k * nv)
B = PT.session_blocks(Y, nv)
check("   shape (k, n, n_vox)", B.shape == (k, n, nv))
check("   block j == columns j*nv:(j+1)*nv",
      all(np.array_equal(B[j], Y[:, j*nv:(j+1)*nv]) for j in range(k)))
try:
    PT.session_blocks(np.zeros((3, 11)), 5); check("   non-multiple column count raises", False)
except ValueError:
    check("   non-multiple column count raises", True)

print("\n2. voxel_reliability / top_reliable on synthetic data")
n, nv, k = 300, 100, 3
signal = rng.standard_normal((n, nv))
gain = np.zeros(nv); gain[:30] = 1.0                      # voxels 0-29 carry a repeatable signal
sessions = np.stack([signal * gain + rng.standard_normal((n, nv)) for _ in range(k)])
sessions[:, :, 99] = 0.0                                  # a constant voxel -> NaN reliability
rel = PT.voxel_reliability(sessions)
check("   signal voxels reliable (~0.5)", np.nanmean(rel[:30]) > 0.4, f"{np.nanmean(rel[:30]):.3f}")
check("   noise voxels ~0", abs(np.nanmean(rel[30:99])) < 0.05, f"{np.nanmean(rel[30:99]):.3f}")
check("   constant voxel -> NaN", np.isnan(rel[99]))
check("   single session -> None", PT.voxel_reliability(sessions[:1]) is None)
top = PT.top_reliable(rel)
check("   keeps ceil(0.2 n) voxels", len(top) == 20)
check("   all kept voxels are signal voxels", set(top) <= set(range(30)))
check("   indices sorted", np.all(np.diff(top) > 0))
check("   NaN ranks last", 99 not in PT.top_reliable(rel, fraction=0.99))

print("\n2b. voxel_native_counts")
nat = np.array([1.0, 0.0, 0.5, np.nan])
rnd = np.array([[0.0, 1.0, 0.4, 0.0],
                [0.5, 1.0, 0.6, 0.0]])
n_ok, frac, beats = PT.voxel_native_counts(nat, rnd)
check("   NaN voxel excluded", n_ok == 3)
check("   fraction of (voxel, seed) pairs native > random", np.isclose(frac, 3 / 6), f"{frac}")
check("   fraction of voxels beating every seed", np.isclose(beats, 1 / 3), f"{beats}")
check("   ties are not wins", PT.voxel_native_counts(np.zeros(4), np.zeros((2, 4)))[1] == 0.0)
check("   all-NaN -> (0, nan, nan)", PT.voxel_native_counts(np.full(3, np.nan), np.zeros((2, 3)))[0] == 0)
null = PT.voxel_native_counts(rng.standard_normal(5000), rng.standard_normal((10, 5000)))
check("   chance levels 0.5 and 1/(n_seeds+1)", abs(null[1] - 0.5) < 0.02 and abs(null[2] - 1 / 11) < 0.02, str(null))

print("\n3. build_targets")
structs = [{PT.NETWORK_KEY: np.concatenate(list(sessions), axis=1)},          # 3 sessions
           {PT.NETWORK_KEY: sessions[0]},                                      # 1 session -> dropped
           {PT.NETWORK_KEY: np.concatenate(list(sessions[:2]), axis=1)}]      # 2 sessions
ta = PT.build_targets(structs, "all", n_vox=nv)
tt = PT.build_targets(structs, "top20rel", n_vox=nv)
check("   single-session participant dropped, original indices kept", [t["part"] for t in ta] == [0, 2])
check("   'all' = session mean of every voxel", np.allclose(ta[0]["Y"], sessions.mean(0)) and ta[0]["Y"].shape[1] == nv)
check("   2-session participant averages its 2 sessions", np.allclose(ta[1]["Y"], sessions[:2].mean(0)))
check("   'top20rel' = mean of the selected voxels",
      np.allclose(tt[0]["Y"], sessions.mean(0)[:, tt[0]["vox_idx"]]) and tt[0]["Y"].shape[1] == 20)
check("   n_sessions recorded", [t["n_sessions"] for t in ta] == [3, 2])
try:
    PT.build_targets(structs, "bogus", n_vox=nv); check("   unknown voxel set raises", False)
except ValueError:
    check("   unknown voxel set raises", True)
check("   results dirs distinct per voxel set", len({PT.results_dir(v) for v in PT.VOXSETS}) == len(PT.VOXSETS))

print("\n4. real data (pereira_data.load_pereira)")
from pereira_data import load_pereira, NPZ, PKL
HAVE_DATA = os.path.exists(NPZ) or os.path.exists(PKL)
if not HAVE_DATA:
    print(f"  SKIP  no brain data ({NPZ}); download it to run this section")
    if "--parquet" in sys.argv or "--smoke" in sys.argv:
        sys.exit("--parquet / --smoke need the brain data too")
    print(f"\n{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else ""))
    sys.exit(1 if fails else 0)
DATA = load_pereira()
D = DATA["net_structs"]
t2a = PT.build_targets(D["PereiraE2_96pass"], "all"); t3a = PT.build_targets(D["PereiraE3_72pass"], "all")
t2t = PT.build_targets(D["PereiraE2_96pass"], "top20rel"); t3t = PT.build_targets(D["PereiraE3_72pass"], "top20rel")
check("   Exp 2 keeps participants 0-4, 6-8 (index 5 = UID 343, 1 session)", [t["part"] for t in t2a] == [0, 1, 2, 3, 4, 6, 7, 8])
check("   Exp 3 keeps all 6", [t["part"] for t in t3a] == list(range(6)))
check("   'all' has 1364 voxels, 384/243 sentences",
      all(t["Y"].shape == (384, 1364) for t in t2a) and all(t["Y"].shape == (243, 1364) for t in t3a))
check("   'top20rel' has 273 voxels", all(t["Y"].shape[1] == 273 for t in t2t + t3t))
check("   top20rel selection is more reliable than all",
      all(np.nanmean(t["reliability"][tt["vox_idx"]]) > np.nanmean(t["reliability"]) for t, tt in zip(t2a + t3a, t2t + t3t)))
# the blocks really are the same voxels in different sessions: session-mean spatial patterns agree
sp = [np.corrcoef(PT.session_blocks(D["PereiraE2_96pass"][i][PT.NETWORK_KEY])[:, :, :].mean(1))[0, 1] for i in (0, 2, 8)]
check("   session blocks share spatial pattern (same voxels)", min(sp) > 0.5, str(np.round(sp, 2)))

if "--parquet" in sys.argv:
    print("\n5. cross-check against the raw parquet (Session x Voxel)")
    if not os.path.exists("data/Pereira.parquet.gzip"):
        sys.exit("--parquet needs the lab's raw parquet data/Pereira.parquet.gzip (not distributed)")
    import pandas as pd
    for exp, uid_pos, uid in [("PereiraE2_96pass", 0, 18), ("PereiraE3_72pass", 3, 289)]:
        df = pd.read_parquet("data/Pereira.parquet.gzip", columns=["UID", "Session", "Experiment", "Sentence", "Network", "Voxel", "EffectSize"],
                             filters=[("Network", "==", PT.NETWORK_KEY), ("UID", "==", uid), ("Experiment", "==", exp)])
        df = df[df.EffectSize.notna()]
        for c in ("Sentence", "Session"):
            df[c] = df[c].astype(str)
        avg = df.groupby(["Sentence", "Voxel"]).EffectSize.mean().unstack("Voxel").sort_index()   # sentences sorted as in the struct
        tgt = [t for t in PT.build_targets(D[exp], "all") if t["part"] == uid_pos][0]
        # struct voxel order is the parquet row order; match by correlating columns is overkill:
        # compare the sorted multiset per sentence and the exact matrix after aligning voxel order
        order = df.drop_duplicates(["Session"]).Session.tolist()
        first = df[(df.Session == order[0]) & (df.Sentence == avg.index[0])].Voxel.to_numpy()
        check(f"   {exp} UID {uid}: {len(order)} sessions, {avg.shape[1]} voxels", avg.shape[1] == PT.N_VOX)
        check(f"   {exp} UID {uid}: struct average == parquet session average",
              np.allclose(avg[first].to_numpy(), tgt["Y"], atol=1e-5))

if "--smoke" in sys.argv:
    print("\n6. end-to-end smoke run of the runner (gpt2, 2 layers, 2 seeds, 2 epochs; CUDA GPU)")
    stub = tempfile.mkdtemp(prefix="pt_stubs_")
    try:
        import geotorch, torchmetrics  # noqa: F401
    except ImportError:
        os.makedirs(f"{stub}/geotorch"); os.makedirs(f"{stub}/torchmetrics")
        open(f"{stub}/geotorch/__init__.py", "w").write(textwrap.dedent('''
            import torch
            def orthogonal(module, name="weight", triv="expm"):
                torch.nn.utils.parametrizations.orthogonal(module, name, orthogonal_map="matrix_exp")
            '''))
        open(f"{stub}/torchmetrics/__init__.py", "w").write("")
        open(f"{stub}/torchmetrics/functional.py", "w").write(textwrap.dedent('''
            import torch.nn.functional as F
            def pairwise_cosine_similarity(x, y):
                return F.normalize(x, dim=1) @ F.normalize(y, dim=1).T
            '''))
    out = tempfile.mkdtemp(prefix="pt_smoke_")
    script = textwrap.dedent(f'''
        import os, sys, numpy as np
        os.chdir({_ROOT!r}); sys.path.insert(0, {_ROOT!r})
        import multiprocessing as mp
        if __name__ == "__main__":
            mp.set_start_method("spawn", force=True)
            import pereira_targets as PT
            PT.RESULTS_DIRS["top20rel"] = {out!r}
            import PEREIRA_alignment_gpu as G
            G.plot_layerwise = lambda *a, **k: None
            from pereira_data import load_pereira
            D = load_pereira()
            sents = [s[1:-1] for s in D["net_sent_orders"]["PereiraE2_96pass"]]
            groups = G.load_passage_groups("PereiraE2_96pass", sents)
            reps, cache = G.get_all_layer_activations("gpt2", sents, device="cuda", cache_dir={out!r} + "/acts",
                                                      data_name="pereira_expt2", residual_only=True)
            emb = {{L: reps["block_outputs"][L] for L in (6, 12)}}
            bd = PT.build_targets(D["net_structs"]["PereiraE2_96pass"], "top20rel")
            df, _ = G.run_parallel_alignment(emb, bd, "gpt2", "gpt2", 2, 2, exp_name="pereira_expt2", n_epochs=2,
                                             return_layerwise=True, cache_path=cache, stream_key="block_outputs",
                                             raw_scores=True, passage_groups=groups, voxset="top20rel")
            print("ROWS", len(df), sorted(df.part.unique()))
        ''')
    open(f"{out}/run.py", "w").write(script)
    env = dict(os.environ, PYTHONPATH=stub + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, f"{out}/run.py"], env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-5:]
    check("   runner exits 0", r.returncode == 0, " | ".join(tail) if r.returncode else "")
    import pandas as pd
    csvs = glob.glob(f"{out}/pereira_expt2_gpt2.csv")
    check("   CSV written in the voxel-set directory", len(csvs) == 1)
    if csvs:
        df = pd.read_csv(csvs[0])
        check("   2 layers x 8 participants, no participant 5", len(df) == 16 and 5 not in set(df.part))
        ck = glob.glob(f"{out}/pereira_expt2_gpt2_{PT.CKPT_VERSION}_*_partial.pkl")
        check("   checkpoint present", len(ck) == 1)
        check("   no per-voxel files written", not glob.glob(f"{out}/voxelwise*"))
        cols = ["vox_n", "vox_frac_native_gt_random", "vox_frac_native_gt_random_concat",
                "vox_frac_native_beats_all_random_concat"]
        check("   voxel-count columns present", all(c in df.columns for c in cols))
        if all(c in df.columns for c in cols):
            check("   vox_n = 273 (top20rel)", (df.vox_n == 273).all(), str(df.vox_n.unique()))
            fr = df[cols[1:]].to_numpy()
            check("   fractions in [0, 1]", np.all((fr >= 0) & (fr <= 1)))
        # resume: a second run must find every task complete and recompute nothing
        r2 = subprocess.run([sys.executable, f"{out}/run.py"], env=env, capture_output=True, text=True)
        check("   resume recomputes nothing", "created 0 tasks" in r2.stdout, (r2.stdout + r2.stderr).strip().splitlines()[-1])

print(f"\n{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)
