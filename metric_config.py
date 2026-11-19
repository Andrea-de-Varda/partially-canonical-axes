"""
Single source of truth for the unit-matching metric used by EVERY pipeline
(Pereira, LANG, SEM, model-model, Pythia checkpoints, optimal rotations, joint).

MATCH_ABS = True: each target (voxel / feature / unit of the other model) is
matched to the candidate unit with the largest |train correlation| and the
unit's sign is flipped when that correlation is negative -- axes are matched
up to orientation. The optimal-rotation objective and its scoring use |cos|
in the same way, so native, random and optimal sit on one scale.

Why: the signed argmax is asymmetric. A random rotation mixes native units
with random-signed weights, so any component that the native units share
with a CONSISTENT sign reaches the rotated basis with both signs but the
native basis with one. Where such a component dominates (gpt2-xl layers
36-46: effective rank 5-7, 100% same-sign PC1 loadings) a random rotation
beats the native basis for every target anticorrelated with it -- a metric
artefact that only ever penalises the native basis. |r| matching removes it
at the cost of some extra winner's curse for both bases (fairer, not more
powerful).

METRIC_TAG goes into every checkpoint filename so a run under one metric can
never resume a checkpoint written under the other.
"""
MATCH_ABS = True
METRIC_TAG = "absmatch" if MATCH_ABS else "signed"


def resolve_match_abs(value=None):
    """None -> project default; True/False -> explicit override (experiments only)."""
    return MATCH_ABS if value is None else bool(value)
