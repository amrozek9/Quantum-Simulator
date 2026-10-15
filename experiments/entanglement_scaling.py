"""Entanglement entropy against subsystem size: area law, critical log law, volume law.

For the periodic transverse-field Ising chain ``H = -sum Z_i Z_{i+1} - h sum X_i``
(exact ground states by diagonalization), and for Haar-random states, this
computes the von Neumann entropy S(l) of the block of qubits 0..l-1 for
every l.

What theory predicts:
  * Gapped ground states obey an area law: S(l) levels off once l exceeds
    the correlation length. Deep in the disordered phase (h > 1) it levels
    off near 0; in the ordered phase (h < 1) the finite-size ground state is
    a cat-like superposition of the two ordered states, worth about 1 bit.
  * At the critical point (h = 1), conformal field theory predicts
    S(l) = (c/3) log2[(n/pi) sin(pi l / n)] + const for a periodic chain,
    with central charge c = 1/2 for the Ising universality class.
  * A random state obeys a volume law: S(l) is close to min(l, n - l), as
    given exactly on average by Page's formula.

Run from the repo root:  python -m experiments.entanglement_scaling
Writes experiments/results/entanglement_scaling.json and two figures.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from experiments.style import THEMES, setup, style_axes

from quantum_simulator.entanglement import entropy_profile, page_entropy
from quantum_simulator.observables import tfim
from quantum_simulator.state import State

HERE = Path(__file__).parent
FIELDS = {"ordered": 0.5, "critical": 1.0, "disordered": 2.0}


def chord_coordinate(n: int, l: np.ndarray) -> np.ndarray:
    """x = (1/3) log2[(n/pi) sin(pi l/n)], so the critical prediction is S = c x + const."""
    return np.log2(n / np.pi * np.sin(np.pi * l / n)) / 3


def fit_central_charge(n: int, profile: np.ndarray) -> tuple[float, float]:
    l = np.arange(1, n)
    c, s0 = np.polyfit(chord_coordinate(n, l), profile[1:n], 1)
    return float(c), float(s0)


def compute(n: int, samples: int, seed: int) -> dict:
    out: dict = {"n": n, "l": list(range(n + 1)), "ground": {}}
    for name, h in FIELDS.items():
        t0 = time.perf_counter()
        energy, gs = tfim(n, J=1.0, h=h, periodic=True).ground_state(max_qubits=max(n, 12))
        profile = entropy_profile(gs)
        out["ground"][name] = {"h": h, "energy": energy, "entropy": profile.tolist()}
        print(f"  {name:10s} h={h}: S(n/2) = {profile[n // 2]:.4f} bits  ({time.perf_counter() - t0:.1f}s)")

    rng = np.random.default_rng(seed)
    profiles = np.array([
        entropy_profile(State(_haar_vector(n, rng))) for _ in range(samples)
    ])
    out["random"] = {
        "mean": profiles.mean(0).tolist(),
        "std": profiles.std(0).tolist(),
        "page": [page_entropy(l, n) for l in range(n + 1)],
    }
    print(f"  random     S(n/2) = {profiles.mean(0)[n // 2]:.4f} bits, Page {page_entropy(n // 2, n):.4f}")

    c, s0 = fit_central_charge(n, np.asarray(out["ground"]["critical"]["entropy"]))
    out["central_charge_fit"] = {"c": c, "intercept": s0, "theory": 0.5}
    print(f"  fitted central charge c = {c:.4f} (Ising CFT: 0.5)")
    return out


def _haar_vector(n: int, rng: np.random.Generator) -> np.ndarray:
    # Normalized complex Gaussian vector: uniformly (Haar) distributed on the unit sphere.
    v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return v / np.linalg.norm(v)


# -- figure -----------------------------------------------------------------------

LABELS = {"critical": "critical, h = 1", "ordered": "ordered, h = 0.5",
          "disordered": "disordered, h = 2", "random": "random states"}


def plot(res: dict, path: Path, theme: str) -> None:
    plt, c = setup(theme)
    colors = {"critical": c["slots"][0], "ordered": c["slots"][1], "disordered": c["slots"][2], "random": c["slots"][3]}
    n = res["n"]
    l = np.asarray(res["l"])
    fig, (left, mid, right) = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    fig.patch.set_facecolor(c["surface"])
    for ax in (left, mid, right):
        style_axes(ax, c)

    # Left: S(l) for every state, with Page's formula behind the random-state points.
    left.plot(l, res["random"]["page"], color=c["muted"], lw=1, ls=(0, (3, 3)), zorder=1)
    left.plot(l, res["random"]["mean"], "o", color=colors["random"], ms=6, mec=c["surface"], mew=1.5,
              label=LABELS["random"], zorder=3)
    for name in ("critical", "ordered", "disordered"):
        left.plot(l, res["ground"][name]["entropy"], "-o", color=colors[name], lw=2, ms=5,
                  mec=c["surface"], mew=1.5, label=LABELS[name], zorder=2)
    left.set_xlabel("Subsystem size l (qubits 0 .. l−1)")
    left.set_ylabel("Entanglement entropy S(l), bits")
    left.set_xticks(l[::2])
    left.set_title(f"Periodic TFIM ground states vs random states, n = {n}", loc="left",
                   color=c["text"], fontsize=10)
    left.annotate("Page's formula", (9, page_entropy(9, n)), xytext=(8, 2),
                  textcoords="offset points", color=c["muted"], fontsize=8)
    left.legend(frameon=False, loc="upper left", labelcolor=c["text"], fontsize=9)

    # Middle: the same ground states on their own scale, where the regimes separate.
    for name in ("critical", "ordered", "disordered"):
        mid.plot(l, res["ground"][name]["entropy"], "-o", color=colors[name], lw=2, ms=5,
                 mec=c["surface"], mew=1.5, zorder=2)
    for name, (x, dy) in {"ordered": (3, 9), "critical": (3, -16), "disordered": (3, 9)}.items():
        mid.annotate(LABELS[name], (x, res["ground"][name]["entropy"][x]), xytext=(4, dy),
                     textcoords="offset points", color=c["text"], fontsize=9)
    mid.set_xlabel("Subsystem size l")
    mid.set_ylabel("S(l), bits")
    mid.set_xticks(l[::2])
    mid.set_ylim(-0.05, 1.25)
    mid.set_title("Ground states only: area law vs log growth", loc="left", color=c["text"], fontsize=10)

    # Right: the critical profile against the CFT coordinate is a straight line of slope c.
    fit = res["central_charge_fit"]
    x = np.asarray(chord_coordinate(n, l[1:n]))
    s = np.asarray(res["ground"]["critical"]["entropy"])[1:n]
    xs = np.linspace(x.min(), x.max(), 50)
    right.plot(xs, fit["c"] * xs + fit["intercept"], color=c["muted"], lw=1, ls=(0, (3, 3)), zorder=1)
    right.plot(x, s, "o", color=colors["critical"], ms=7, mec=c["surface"], mew=1.5, zorder=2)
    right.set_xlabel("(1/3) log₂[(n/π) sin(πl/n)]")
    right.set_ylabel("S(l) at h = 1, bits")
    right.set_title("Critical point: slope is the central charge", loc="left", color=c["text"], fontsize=10)
    right.annotate(f"fitted c = {fit['c']:.3f}\nIsing theory c = 1/2", (0.04, 0.96), xycoords="axes fraction",
                   va="top", color=c["text"], fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=c["surface"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=12, help="chain length (even)")
    ap.add_argument("--samples", type=int, default=20, help="random states to average")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = {"seed": args.seed, "samples": args.samples, "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform()}
    print(f"seed={args.seed}  numpy={meta['numpy']}  python={meta['python']}  {meta['platform']}")
    res = compute(args.n, args.samples, args.seed)
    res["meta"] = meta

    results_dir, fig_dir = HERE / "results", HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    (results_dir / "entanglement_scaling.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for theme in THEMES:
        plot(res, fig_dir / f"entanglement-scaling-{theme}.png", theme)
    print(f"Wrote {results_dir / 'entanglement_scaling.json'} and figures in {fig_dir}")


if __name__ == "__main__":
    main()
