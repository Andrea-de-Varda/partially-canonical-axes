"""
Illustrative animation for the README: what "rotating the basis" does to
axis-by-axis matching. Toy 2-D data (not results from the paper).

Stimuli are points in a 2-D representation space X. A target unit Y (e.g. a unit
of another model, a voxel, a feature) is a noisy linear readout of X along a
direction 25 deg away from the first native axis. For a given basis we
(1) match Y to the axis with the highest |r| on training stimuli and
(2) score that match on held-out stimuli (sign fixed on train).
Top row: the basis is rotated from native (alpha = 0) to a random rotation
(alpha = 1). Bottom row: from native to the optimal rotation Q*, which puts an
axis on Y.

    python docs/make_rotation_gif.py   ->  docs/rotation_matching.gif
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rotation_matching.gif")

NATIVE, RANDOM, OPTIMAL, TARGET, GREY = "#3b6fa8", "#8a8a8a", "#1d3557", "#c08a2b", "#d9d9d9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

rng = np.random.default_rng(3)
n = 160
X = rng.standard_normal((n, 2)) * np.array([1.0, 0.75])
phi_t = np.deg2rad(25)                                   # target direction
u_t = np.array([np.cos(phi_t), np.sin(phi_t)])
Y = X @ u_t + 0.45 * rng.standard_normal(n)
train = np.zeros(n, bool); train[rng.permutation(n)[: int(0.6 * n)]] = True
test = ~train

THETA_RANDOM = np.deg2rad(-20)   # angle of the "random" rotation Q (away from the target: in 2-D,
                                  # a rotation can at most put the target 45 deg from every axis)


def basis(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])                   # columns = axes


def match(angle):
    B = basis(angle)
    Z = X @ B
    r_tr = [np.corrcoef(Z[train, k], Y[train])[0, 1] for k in range(2)]
    k = int(np.argmax(np.abs(r_tr)))
    sgn = np.sign(r_tr[k])
    r_te = np.corrcoef(sgn * Z[test, k], Y[test])[0, 1]
    return B, Z, k, sgn, r_te


# Two rows animated in parallel, t = 0 -> 1:
#   top:    native basis -> random rotation  (angle t * THETA_RANDOM, i.e. Q^alpha with alpha = t)
#   bottom: native basis -> optimal rotation (angle t * phi_t, the rotation that puts an axis on Y)
ROWS = [dict(end=THETA_RANDOM, col=RANDOM, name="random rotation", xlabel="rotation $\\alpha$"),
        dict(end=phi_t, col=OPTIMAL, name="optimal rotation $Q^*$", xlabel="progress towards $Q^*$")]
T = [0.0] * 10 + list(np.linspace(0, 1, 40)) + [1.0] * 18
curve_t = np.linspace(0, 1, 41)
for row in ROWS:
    row["curve"] = np.array([match(t * row["end"])[4] for t in curve_t])
r_native = ROWS[0]["curve"][0]
YLIM = (min(r["curve"].min() for r in ROWS) - 0.06, max(r["curve"].max() for r in ROWS) + 0.06)


def blend(c0, c1, t):
    return tuple(np.array(matplotlib.colors.to_rgb(c0)) * (1 - t) + np.array(matplotlib.colors.to_rgb(c1)) * t)


fig = plt.figure(figsize=(9.6, 6.6), dpi=100)
gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1], wspace=0.38, hspace=0.45,
                      left=0.04, right=0.97, top=0.93, bottom=0.08)
AX = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]


def draw_row(axA, axB, axC, row, t, first):
    angle = t * row["end"]
    B, Z, k, sgn, r_te = match(angle)
    col = blend(NATIVE, row["col"], t)

    axA.clear()
    axA.scatter(X[:, 0], X[:, 1], c=Y, cmap="cividis", s=12, alpha=0.75, linewidths=0)
    for j in range(2):
        v = B[:, j] * 2.6
        axA.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6 if j == k else 1.4, mutation_scale=12))
        axA.annotate("", xy=-v * 0.75, xytext=(0, 0), arrowprops=dict(arrowstyle="-", color=col, lw=0.8, ls="--"))
        axA.text(*(v * 1.12), f"$x_{j + 1}$", color=col, ha="center", va="center", fontsize=11, weight="bold" if j == k else None)
    axA.plot(*np.c_[-u_t * 3.1, u_t * 3.1], color=TARGET, lw=1, ls=":", zorder=0)
    axA.text(*(u_t * 2.9 + np.array([0.35, -0.95])), "target\ndirection", color=TARGET, fontsize=8, ha="center")
    axA.set_xlim(-3.4, 3.4); axA.set_ylim(-3.4, 3.4); axA.set_aspect("equal"); axA.axis("off")
    if t == 0:
        title = "Native basis"
    elif t == 1:
        title = row["name"][0].upper() + row["name"][1:]
    else:
        title = f"Native $\\rightarrow$ {row['name']}"
    axA.set_title(title, color=col, fontsize=11, weight="bold")

    axB.clear()
    axB.scatter(sgn * Z[train, k], Y[train], s=9, color=GREY, linewidths=0, label="train (matching)")
    axB.scatter(sgn * Z[test, k], Y[test], s=14, color=col, linewidths=0, label="held-out (scoring)")
    axB.set_xlabel(f"best-matched axis $x_{k + 1}$", fontsize=10)
    axB.set_ylabel("target unit $Y$", fontsize=10)
    axB.text(0.04, 0.95, f"held-out $r$ = {r_te:.2f}", transform=axB.transAxes, va="top", fontsize=11, color=col,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1))
    axB.set_xlim(-3.6, 3.6); axB.set_ylim(-3.6, 3.6)
    axB.legend(loc="lower right", fontsize=7.5, frameon=False, handletextpad=0.2)

    axC.clear()
    m = curve_t <= t + 1e-9
    axC.plot(curve_t, row["curve"], color=row["col"], lw=0.6, alpha=0.2)
    axC.plot(curve_t[m], row["curve"][m], color=row["col"], lw=2)
    axC.axhline(r_native, color=NATIVE, lw=1, ls="--")
    axC.text(1.0, r_native, "native", color=NATIVE, fontsize=8, ha="right", va="bottom")
    axC.scatter([0], [r_native], color=NATIVE, s=30, zorder=4)
    axC.scatter([t], [r_te], color=col, s=42, zorder=5, edgecolors="black", linewidths=0.6)
    axC.set_xlim(-0.05, 1.05); axC.set_ylim(*YLIM)
    axC.set_xlabel(row["xlabel"], fontsize=10)
    axC.set_ylabel("alignment (held-out $r$)", fontsize=10)
    if first:
        axB.set_title("Match on train, score on held-out", fontsize=11)
        axC.set_title("Alignment along the path", fontsize=11)
    for ax in (axB, axC):
        ax.spines[["top", "right"]].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(1.3)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", ls="--", alpha=0.4)


def draw(i):
    for r, row in enumerate(ROWS):
        draw_row(*AX[r], row, T[i], first=(r == 0))
    return []


anim = FuncAnimation(fig, draw, frames=len(T), interval=90, blit=False)
anim.save(OUT, writer=PillowWriter(fps=11))
print("wrote", OUT)
