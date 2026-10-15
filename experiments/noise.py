"""Noise: density matrices against pure-state trajectories, on accuracy and on cost.

A GHZ state prepared with depolarizing noise (probability p on every qubit a
gate touches, after the gate). Its coherence <X X ... X> is 1 without noise
and decays with p, faster for more qubits.

  A. <X^n> against p for n = 2, 4, 6, 8 from density matrices, with
     trajectory estimates (2000 trajectories each) at n = 8 and their
     standard errors.
  B. Cost against n: one density-matrix run, whose memory grows as 4^n,
     against one pure-state trajectory, which grows as 2^n. A trajectory
     estimate needs many runs, with error falling as 1/sqrt(runs).

Run from the repo root:  python -m experiments.noise
Writes experiments/results/noise.json and two figures.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from experiments.style import THEMES, setup, style_axes

HERE = Path(__file__).parent
SIZES = [2, 4, 6, 8]


def compute(seed: int, trajectories: int) -> dict:
    from quantum_simulator import algorithms as alg
    from quantum_simulator.backends.density import NoiseModel, depolarizing, run_density, run_trajectories
    from quantum_simulator.observables import PauliSum

    ps = [round(x, 3) for x in np.linspace(0, 0.1, 11)]
    out: dict = {"p": ps, "density": {}, "trajectories": {}, "cost": {}}
    for n in SIZES:
        xs = PauliSum([("X" * n, 1.0)])
        out["density"][str(n)] = [run_density(alg.ghz(n), noise=NoiseModel(depolarizing(p))).expectation(xs) for p in ps]
        print(f"  n={n}: <X^n> at p = 0.05: {out['density'][str(n)][5]:.4f}", flush=True)

    n = SIZES[-1]
    xs = PauliSum([("X" * n, 1.0)])
    rows = []
    for k, p in enumerate((0.02, 0.05, 0.1)):
        mean, err = run_trajectories(alg.ghz(n), xs, trajectories, rng=seed + k, noise=NoiseModel(depolarizing(p)))
        exact = run_density(alg.ghz(n), noise=NoiseModel(depolarizing(p))).expectation(xs)
        rows.append({"p": p, "mean": mean, "stderr": err, "density": exact})
        print(f"  trajectories n={n}, p={p}: {mean:.4f} +- {err:.4f}, density {exact:.4f} "
              f"({abs(mean - exact) / err:.2f} sigma)", flush=True)
    out["trajectories"] = {"n": n, "count": trajectories, "rows": rows}

    cost_n = list(range(2, 12))
    dens_t, traj_t = [], []
    for m in cost_n:
        noise = NoiseModel(depolarizing(0.05))
        obs = PauliSum([("X" * m, 1.0)])
        t0 = time.perf_counter()
        run_density(alg.ghz(m), noise=noise)
        dens_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        run_trajectories(alg.ghz(m), obs, 20, rng=seed, noise=noise)
        traj_t.append((time.perf_counter() - t0) / 20)
        print(f"  cost n={m:2d}: density {dens_t[-1] * 1e3:9.2f} ms, one trajectory {traj_t[-1] * 1e3:7.2f} ms", flush=True)
    out["cost"] = {"n": cost_n, "density_s": dens_t, "trajectory_s": traj_t}
    return out


def plot(res: dict, path: Path, theme: str) -> None:
    plt, c = setup(theme)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0))
    fig.patch.set_facecolor(c["surface"])

    style_axes(left, c, "GHZ coherence under depolarizing noise")
    p = np.asarray(res["p"])
    for color, n in zip(c["slots"], SIZES):
        y = res["density"][str(n)]
        left.plot(p, y, color=color, lw=2)
        left.annotate(f"n = {n}", (p[-1], y[-1]), xytext=(14, 0), textcoords="offset points", va="center",
                      color=c["text"], fontsize=9)
    tr = res["trajectories"]
    left.errorbar([r["p"] for r in tr["rows"]], [r["mean"] for r in tr["rows"]],
                  yerr=[2 * r["stderr"] for r in tr["rows"]], fmt="o", color=c["text"], ms=5, capsize=3, lw=1,
                  label=f"{tr['count']} trajectories, n = {tr['n']} (±2 s.e.)")
    left.set_xlim(-0.003, p[-1] + 0.024)
    left.set_xlabel("Depolarizing probability p per gate per qubit")
    left.set_ylabel("⟨X X … X⟩ (lines: density matrix)")
    left.legend(frameon=False, loc="lower left", labelcolor=c["text"], fontsize=8)

    style_axes(right, c, "Cost of one run")
    cn = np.asarray(res["cost"]["n"])
    right.plot(cn, np.asarray(res["cost"]["density_s"]) * 1e3, "-o", color=c["slots"][0], lw=2, ms=5,
               mec=c["surface"], mew=1, label="density matrix (memory 4ⁿ)")
    right.plot(cn, np.asarray(res["cost"]["trajectory_s"]) * 1e3, "-o", color=c["slots"][1], lw=2, ms=5,
               mec=c["surface"], mew=1, label="one trajectory (memory 2ⁿ)")
    right.set_yscale("log")
    right.set_xticks(cn)
    right.set_xlabel("Qubits n")
    right.set_ylabel("Milliseconds")
    right.legend(frameon=False, loc="upper left", labelcolor=c["text"], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=c["surface"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trajectories", type=int, default=2000)
    args = ap.parse_args()

    meta = {"seed": args.seed, "trajectories": args.trajectories, "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform()}
    print("  ".join(f"{k}={v}" for k, v in meta.items()))
    res = compute(args.seed, args.trajectories)
    res["meta"] = meta
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "results" / "noise.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for theme in THEMES:
        plot(res, HERE / "figures" / f"noise-{theme}.png", theme)
    print("Wrote experiments/results/noise.json and figures")


if __name__ == "__main__":
    main()
