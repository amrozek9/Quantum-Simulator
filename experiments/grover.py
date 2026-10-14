"""Grover's success probability against the number of iterations.

Grover search is a rotation by 2 theta in the plane of the marked state and
the uniform superposition, with sin(theta) = 1/sqrt(N). Success therefore
follows sin^2((2k + 1) theta): it peaks after about (pi/4) sqrt(N) iterations
and then falls as the search over-rotates, rather than improving
monotonically. This script measures the curve by simulating each circuit
and checks every point against that formula.

Run from the repo root:  python -m experiments.grover
Writes experiments/results/grover.json and two figures.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from quantum_simulator.algorithms import grover, grover_success_theory, optimal_grover_iterations
from quantum_simulator.circuit import run

HERE = Path(__file__).parent


def measure(n: int, marked: int, max_iterations: int) -> dict:
    ks = list(range(max_iterations + 1))
    success = [float(run(grover(n, marked, k)).state.probabilities()[marked]) for k in ks]
    theory = [grover_success_theory(n, k) for k in ks]
    worst = max(abs(a - b) for a, b in zip(success, theory))
    assert worst < 1e-10, worst
    k_opt = optimal_grover_iterations(n)
    print(f"  n={n}, N={2**n}, marked={marked:0{n}b}: optimum k={k_opt}, success {success[k_opt]:.6f}; "
          f"max |simulated - theory| = {worst:.1e}")
    return {"n": n, "marked": marked, "k": ks, "success": success, "theory": theory,
            "k_opt": k_opt, "max_abs_error": worst}


# -- figure -----------------------------------------------------------------------

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "series": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#3a3a37", "series": "#3987e5"},
}


def plot(res: dict, path: Path, theme: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = THEMES[theme]
    plt.rcParams.update({
        "font.size": 10, "text.color": c["text"], "axes.labelcolor": c["muted"],
        "xtick.color": c["muted"], "ytick.color": c["muted"], "axes.edgecolor": c["grid"],
    })
    k = np.asarray(res["k"])
    fine = np.linspace(0, k[-1], 600)
    theta = np.arcsin(1 / np.sqrt(2 ** res["n"]))

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    ax.grid(True, color=c["grid"], lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.plot(fine, np.sin((2 * fine + 1) * theta) ** 2, color=c["muted"], lw=1, ls=(0, (3, 3)), zorder=1,
            label="theory: sin²((2k+1)θ), sin θ = 1/√N")
    ax.plot(k, res["success"], "o", color=c["series"], ms=6, mec=c["surface"], mew=1.5, zorder=2,
            label="simulated circuits")
    k_opt = res["k_opt"]
    ax.axvline(k_opt, color=c["muted"], lw=0.8, ls=":")
    arrow = {"arrowstyle": "-", "color": c["muted"], "lw": 0.8}
    # Labels sit in empty regions of the plot, with thin leaders to what they describe.
    ax.annotate(f"k = ⌊(π/4)√N⌋ = {k_opt}: success {res['success'][k_opt]:.4f}", (k_opt, res["success"][k_opt]),
                xytext=(16.5, 0.97), textcoords="data", va="center", color=c["text"], fontsize=9, arrowprops=arrow)
    trough = int(np.argmin(res["success"][k_opt:])) + k_opt
    ax.annotate("over-rotated:\nsuccess falls again", (trough, res["success"][trough]),
                xytext=(21.3, 0.47), textcoords="data", color=c["text"], fontsize=9, arrowprops=arrow)
    # Every corner of the plot holds data, so the legend goes above it.
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, labelcolor=c["text"],
              fontsize=9, borderaxespad=0.2)
    ax.set_xlabel("Grover iterations k")
    ax.set_ylabel("Probability of the marked item")
    ax.set_ylim(-0.03, 1.08)
    ax.set_title(f"Grover search, {res['n']} qubits (N = {2 ** res['n']})", loc="left", color=c["text"], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=c["surface"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--marked", type=int, default=0b10110010)
    ap.add_argument("--max-iterations", type=int, default=40)
    args = ap.parse_args()

    meta = {"numpy": np.__version__, "python": platform.python_version(), "platform": platform.platform()}
    print("  ".join(f"{k}={v}" for k, v in meta.items()))
    res = measure(args.n, args.marked, args.max_iterations)
    res["meta"] = meta

    results_dir, fig_dir = HERE / "results", HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    (results_dir / "grover.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for theme in THEMES:
        plot(res, fig_dir / f"grover-{theme}.png", theme)
    print(f"Wrote {results_dir / 'grover.json'} and figures in {fig_dir}")


if __name__ == "__main__":
    main()
