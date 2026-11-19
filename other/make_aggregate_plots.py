"""
Regenerate every AGGREGATE (multi-model) figure in plots/summary/.

Per-model figures (plots/pereira/, plots/semantics/) and the manually
assembled paper panels (plots/panels/) are deliberately NOT touched.

Run locally:
  conda activate basis-alignment
  python other/make_aggregate_plots.py
"""

import os
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotting as P
from metric_config import METRIC_TAG

RES = "data/results"
JOINT = "data/results/joint"
OUT = "plots/summary"
OTHER = "plots/summary/other_plots"

os.makedirs(OUT, exist_ok=True)
os.makedirs(OTHER, exist_ok=True)

print(f"Active matching metric: {METRIC_TAG}\n")

# ---------------------------------------------------------------- brain ----
for exp in ("expt2", "expt3"):
    files = sorted(glob(f"{P.PEREIRA_RES}/pereira_{exp}_*.csv"))
    title = f"Pereira Expt. {exp[-1]}"
    print(f"[{exp}] {len(files)} models")
    P.plot_all(files, plot_title=title, to_replace=f"pereira_{exp}_",
               normalize=True, save_path=f"{OUT}/pereira_{exp}_rotation.png")
    P.plot_all(files, plot_title=title, to_replace=f"pereira_{exp}_",
               normalize=False, save_path=f"{OUT}/pereira_{exp}_rotation_nonorm.png")

# ------------------------------------------------------------- semantic ----
for task, label in (("affective", "Affective norms"), ("perceptual", "Perceptual norms")):
    files = sorted(glob(f"{RES}/{task}_*_semantic.csv"))
    print(f"[{task}] {len(files)} models")
    P.plot_all(files, plot_title=label, to_replace=f"{task}_",
               to_replace_2="_semantic", normalize=True,
               save_path=f"{OUT}/{task}_rotation.png")
    P.plot_all(files, plot_title=label, to_replace=f"{task}_",
               to_replace_2="_semantic", normalize=False,
               save_path=f"{OUT}/{task}_rotation_nonorm.png")

# ------------------------------------------------------- language tasks ----
# Syntax aggregates the 12 BLiMP paradigms per model.
for task, label in (("wug", "Morphology (Wug)"), ("hypernymy", "Lexical (Hypernymy)")):
    files = sorted(glob(f"{RES}/lang_{task}_*.csv"))
    print(f"[lang_{task}] {len(files)} models")
    P.plot_all(files, plot_title=label, to_replace=f"lang_{task}_", normalize=True,
               save_path=f"{OUT}/lang_{task}_rotation.png")
    P.plot_all(files, plot_title=label, to_replace=f"lang_{task}_", normalize=False,
               save_path=f"{OUT}/lang_{task}_rotation_nonorm.png")

P.plot_lang_syntax_combined(results_dir=RES, normalize=True,
                            save_path=f"{OUT}/lang_syntax_rotation.png")
P.plot_lang_syntax_combined(results_dir=RES, normalize=False,
                            save_path=f"{OUT}/lang_syntax_rotation_nonorm.png")

# ------------------------------------------------------------ model-model --
mm = sorted(glob(f"{RES}/model_model_EleutherAI_pythia-160m_vs_*.csv"))
print(f"[model-model] {len(mm)} pairs")
P.plot_model_model_alignment(mm, normalize=False,
                             save_path=f"{OUT}/model_model_rotation.png")

# ------------------------------------------------------------- checkpoints --
ck = sorted(glob(f"{RES}/model_model_checkpoints_EleutherAI_pythia-*-seed1_vs_*.csv"))
print(f"[checkpoints] {len(ck)} model sizes")
P.plot_checkpoint_alignment(ck, save_path=f"{OUT}/model_model_checkpoints.png")
P.plot_checkpoint_ratios(ck, save_dir=OUT)
rsa = sorted(glob(f"{RES}/model_model_checkpoints_rsa_*.csv"))
P.plot_checkpoint_rsa(rsa, save_path=f"{OUT}/model_model_checkpoints_rsa.png")

# --------------------------------------------------------------- sparsity --
# Metric-independent (no unit matching), regenerated for completeness.
P.plot_sparsity_kurtosis(sorted(glob(f"{RES}/sparsity_kurtosis_*.csv")),
                         save_path=f"{OUT}/sparsity_kurtosis.png")
P.plot_sparsity_readout(sorted(glob(f"{RES}/sparsity_readout_*.csv")),
                        save_path=f"{OUT}/sparsity_readout.png")
# Readout sparsity against its random-rotation null: native vs null per model (with per-layer dots), and the un-normalised alpha-curve with the
# null shaded. Both write .svg + .png.
_readout = sorted(glob(f"{RES}/sparsity_readout_*.csv"))
P.plot_readout_sparsity_ridges(_readout, save_path=f"{OUT}/readout_sparsity_ridges.png")
P.plot_readout_sparsity_raw(_readout, save_path=f"{OUT}/readout_sparsity_raw.png")
# Same curve in one panel, discrete + continuous features pooled with equal weight.
P.plot_readout_sparsity_raw_pooled(_readout, save_path=f"{OUT}/readout_sparsity_raw_pooled.png")

# ------------------------------------------------------ cross-task summary --
# include_random=True is the published shape of these two figures: three boxes
# per task (random / native / optimal). The function defaults to False, which
# silently drops the random baseline -- the whole point of the comparison.
P.plot_native_vs_optimal_three(results_dir=RES, include_random=True,
                               save_path=f"{OUT}/native_vs_optimal_three.png")
P.plot_native_vs_optimal_three_joint(results_dir=RES, joint_dir=JOINT,
                                     include_random=True,
                                     save_path=f"{OUT}/native_vs_optimal_three_joint.png")
P.plot_all_tasks_layerwise(results_dir=RES, normalize=True,
                           save_path=f"{OUT}/all_tasks_layer_depth.png")

for metric in ("difference", "ratio"):
    P.plot_inter_model_privilege(metric=metric, results_dir=RES,
                                 save_path=f"{OTHER}/inter_model_privilege_{metric}.png")

print("\nDone.")
