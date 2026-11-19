# privileged_axes

Pairwise (axis-by-axis) matching, fractional rotations $Q^\alpha$ and the optimal rotation on SO(n), adapted from the code of

> Khosla, M., Williams, A. H., McDermott, J., & Kanwisher, N. (2024). Privileged representational axes in biological and artificial neural networks. *bioRxiv*, doi:10.1101/2024.06.20.599957.

Please cite that paper when using this code.

| File | Role |
|---|---|
| `get_alignment_pairwise.py` | Pairwise matching score (match on train by \|r\|, sign fixed on train, score on test) at each α, for native and rotated bases |
| `rotation_code.py` | Random rotations $Q \sim$ Uniform(SO(n)) and fractional powers $Q^\alpha$ (Schur decomposition) |
| `get_optimal_pairwise.py` | Optimal rotation $Q^*$ (matrix-exponential parametrization via geotorch, Adam + one-cycle schedule, validation-selected epoch) |
| `get_optimal_joint.py` | One rotation fitted jointly to all feature tasks (SI) |

Modified for this project; the main methodological change is sign-invariant matching (`metric_config.MATCH_ABS`, see the paper's SI).
