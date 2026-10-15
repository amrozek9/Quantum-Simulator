"""Barren plateaus: gradient variance against qubit count, for two entanglers.

The same layered ansatz (RY on every qubit, then 2n layers of an entangling
ladder followed by RY), with the cost <Z_0 Z_1>. For random parameters, the
variance of the gradient, averaged over parameters, is measured for n = 2..12
with two entanglers that differ only in one gate:

  * CX ladder: the ansatz's dynamical Lie algebra (DLA) is all of so(2^n),
    whose dimension grows like 4^n. Theory predicts exponentially vanishing
    gradients, a barren plateau.
  * CZ ladder: the DLA has dimension n(n+1), polynomial in n. Theory predicts
    gradient variance falling only polynomially (Larocca et al. 2022; Ragone
    et al. 2024), at the price of expressiveness (see vqe_convergence.py).

The DLA dimensions are computed exactly for small n.

Run from the repo root:  python -m experiments.barren_plateaus
Writes experiments/results/barren_plateaus.json and two figures.
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
LABELS = {"cx": "RY + CX ladder", "cz": "RY + CZ ladder"}


def gradient_variance(n: int, entangler: str, samples: int, seed: int) -> float:
    import jax

    from quantum_simulator.observables import PauliSum, pauli_string
    from quantum_simulator.vqe import make_ansatz_expectations

    layers = 2 * n
    cost = PauliSum([(pauli_string(n, {0: "Z", 1: "Z"}), 1.0)])
    f = make_ansatz_expectations(n, layers, [cost], entangler=entangler)
    grad = jax.jit(jax.vmap(jax.grad(lambda p: f(p)[0])))
    params = np.random.default_rng(seed).uniform(0, 2 * np.pi, (samples, layers + 1, n))
    g = np.asarray(grad(params))
    return float(g.var(axis=0).mean())


def compute(ns: list[int], samples: int, seed: int, dla_max: dict[str, int]) -> dict:
    from quantum_simulator.vqe import dynamical_lie_algebra_dimension

    out: dict = {"n": ns, "samples": samples, "variance": {}, "dla": {}, "fit": {}}
    for ent in ("cx", "cz"):
        out["variance"][ent] = []
        for n in ns:
            t0 = time.perf_counter()
            v = gradient_variance(n, ent, samples, seed + n)
            out["variance"][ent].append(v)
            print(f"  {ent} n={n:2d}: Var[dE/dtheta] = {v:.3e}  ({time.perf_counter() - t0:.1f}s)", flush=True)
        dla_ns = list(range(2, dla_max[ent] + 1))
        out["dla"][ent] = {"n": dla_ns, "dim": [dynamical_lie_algebra_dimension(n, ent) for n in dla_ns]}
        print(f"  {ent} DLA dimensions for n={dla_ns}: {out['dla'][ent]['dim']}", flush=True)

    n_arr = np.asarray(ns, dtype=float)
    fit_mask = n_arr >= 4
    cx = np.log2(np.asarray(out["variance"]["cx"]))
    slope, intercept = np.polyfit(n_arr[fit_mask], cx[fit_mask], 1)
    out["fit"]["cx"] = {"form": "Var = 2^(a n + b)", "a": float(slope), "b": float(intercept)}
    cz = np.log(np.asarray(out["variance"]["cz"]))
    power, logc = np.polyfit(np.log(n_arr[fit_mask]), cz[fit_mask], 1)
    out["fit"]["cz"] = {"form": "Var = C n^p", "p": float(power), "C": float(np.exp(logc))}
    print(f"  fits: CX variance ~ 2^({slope:.3f} n), CZ variance ~ n^({power:.3f})")
    return out


def plot(res: dict, path: Path, theme: str) -> None:
    plt, c = setup(theme)
    colors = {"cx": c["slots"][0], "cz": c["slots"][1]}
    n = np.asarray(res["n"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.patch.set_facecolor(c["surface"])

    style_axes(left, c, "Gradient variance at random parameters")
    nn = np.linspace(4, n[-1], 50)
    fx, fz = res["fit"]["cx"], res["fit"]["cz"]
    left.plot(nn, 2 ** (fx["a"] * nn + fx["b"]), color=c["muted"], lw=1, ls=(0, (3, 3)), zorder=1)
    left.plot(nn, fz["C"] * nn ** fz["p"], color=c["muted"], lw=1, ls=(0, (3, 3)), zorder=1)
    for ent in ("cx", "cz"):
        left.plot(n, res["variance"][ent], "-o", color=colors[ent], lw=2, ms=6, mec=c["surface"], mew=1.5,
                  label=LABELS[ent], zorder=2)
    # Fit labels in the empty space beside each curve: below the falling CX line, above the CZ line.
    left.text(5.3, 1.3e-3, f"CX: ∝ 2^({fx['a']:.2f} n)\nbarren plateau", color=c["text"], fontsize=9)
    left.text(7.6, 7e-2, f"CZ: ∝ n^({fz['p']:.2f})\npolynomial", color=c["text"], fontsize=9)
    left.set_yscale("log")
    left.set_xticks(n)
    left.set_xlabel("Qubits n (depth 2n layers)")
    left.set_ylabel("Var[∂⟨Z₀Z₁⟩/∂θ], mean over parameters")
    left.legend(frameon=False, loc="lower left", labelcolor=c["text"], fontsize=9)

    style_axes(right, c, "Size of the dynamical Lie algebra")
    nd = np.arange(2, 9)
    right.plot(nd, 2 ** (nd - 1) * (2**nd - 1), color=c["muted"], lw=1, ls=(0, (3, 3)), zorder=1)
    right.annotate("so(2ⁿ): 2ⁿ⁻¹(2ⁿ−1)", (6, 2**5 * (2**6 - 1)), xytext=(-10, 6),
                   textcoords="offset points", ha="right", color=c["muted"], fontsize=8)
    right.plot(nd, nd * (nd + 1), color=c["muted"], lw=1, ls=(0, (1, 2)), zorder=1)
    right.annotate("n(n+1)", (nd[-1], nd[-1] * (nd[-1] + 1)), xytext=(-4, -14), textcoords="offset points",
                   ha="right", color=c["muted"], fontsize=8)
    for ent in ("cx", "cz"):
        d = res["dla"][ent]
        right.plot(d["n"], d["dim"], "o", color=colors[ent], ms=7, mec=c["surface"], mew=1.5, zorder=2)
    right.set_yscale("log")
    right.set_xticks(nd)
    right.set_xlabel("Qubits n")
    right.set_ylabel("DLA dimension (computed points)")

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=c["surface"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-n", type=int, default=12)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax

    meta = {"seed": args.seed, "numpy": np.__version__, "jax": jax.__version__,
            "python": platform.python_version(), "platform": platform.platform()}
    print("  ".join(f"{k}={v}" for k, v in meta.items()))
    res = compute(list(range(2, args.max_n + 1)), args.samples, args.seed, {"cx": 5, "cz": 7})
    res["meta"] = meta

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "results" / "barren_plateaus.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for theme in THEMES:
        plot(res, HERE / "figures" / f"barren-plateaus-{theme}.png", theme)
    print("Wrote experiments/results/barren_plateaus.json and figures")


if __name__ == "__main__":
    main()
