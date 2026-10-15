"""The transverse-field Ising transition: entanglement and correlations against h/J.

Open chain of n = 10 qubits, H = -sum Z_i Z_{i+1} - h sum X_i, swept over
h = 0 .. 2. For each h this records the half-chain entanglement entropy and
the correlation <Z_a Z_b> between sites a = 1 and b = n - 2 (away from the
edges), for three states:

  * the exact ground state;
  * the exact ground state with a small symmetry-breaking field
    -eps sum Z_i (eps = 0.01). Below h = 1 the finite chain's true ground
    state is a cat-like superposition of the two ordered states, which adds
    one bit of entropy everywhere in the ordered phase; the field picks one
    of them, as a real magnet would;
  * VQE states (RY + CX ansatz) at a coarser set of h, minimizing the plain
    Hamiltonian, all sharing one compilation.

It also records <Z_a Z_b> against the separation b - a in each phase
(h = 0.5, 1, 1.5), for pairs centered on the chain: flat in the ordered
phase (long-range order), slowly decaying at the critical point, and
quickly decaying in the disordered phase.

Run from the repo root:  python -m experiments.phase_transition
Writes experiments/results/phase_transition.json and four figures
(phase-transition and correlations, light and dark).
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


def compute(n: int, eps: float, h_grid: list[float], h_vqe: list[float], layers: int, steps: int,
            restarts: int, seed: int) -> dict:
    from quantum_simulator.entanglement import entanglement_entropy
    from quantum_simulator.observables import PauliSum, pauli_string, tfim, z_correlation
    from quantum_simulator.vqe import ansatz_state, make_ansatz_expectations, vqe

    a, b = 1, n - 2
    half = range(n // 2)
    field = PauliSum([(pauli_string(n, {i: "Z"}), 1.0) for i in range(n)])
    out: dict = {"n": n, "eps": eps, "sites": [a, b], "h": h_grid, "exact": {}, "broken": {}, "vqe": {}}
    for key, extra in (("exact", 0.0), ("broken", eps)):
        S, C, E = [], [], []
        t0 = time.perf_counter()
        for h in h_grid:
            H = tfim(n, 1.0, h) - extra * field if extra else tfim(n, 1.0, h)
            energy, gs = H.ground_state()
            S.append(entanglement_entropy(gs, half))
            C.append(z_correlation(gs, a, b))
            E.append(energy)
        out[key] = {"entropy": S, "zz": C, "energy": E}
        peak = h_grid[int(np.argmax(S))]
        print(f"  {key:6s}: entropy peak at h = {peak:.2f} ({time.perf_counter() - t0:.1f}s)", flush=True)

    zz = PauliSum([(pauli_string(n, {i: "Z", i + 1: "Z"}), 1.0) for i in range(n - 1)])
    xs = PauliSum([(pauli_string(n, {i: "X"}), 1.0) for i in range(n)])
    expectations = make_ansatz_expectations(n, layers, [zz, xs])  # one compilation for every h
    rows = []
    for h in h_vqe:
        t0 = time.perf_counter()
        r = vqe(n, layers, [zz, xs], [-1.0, -h], steps=steps, restarts=restarts, seed=seed,
                expectations=expectations)
        s = ansatz_state(r.params)
        e0 = tfim(n, 1.0, h).ground_state()[0]
        rows.append({"h": h, "energy": r.energy, "error": r.energy - e0,
                     "entropy": entanglement_entropy(s, half), "zz": z_correlation(s, a, b)})
        print(f"  VQE h={h:.2f}: error {r.energy - e0:.2e}, S = {rows[-1]['entropy']:.3f}, "
              f"<Z{a}Z{b}> = {rows[-1]['zz']:.3f}  ({time.perf_counter() - t0:.1f}s)", flush=True)
    out["vqe"] = {"layers": layers, "rows": rows}

    # Correlation against separation, one curve per phase: pairs centered on the chain.
    out["correlation_vs_distance"] = {}
    for h in (0.5, 1.0, 1.5):
        _, gs = tfim(n, 1.0, h).ground_state()
        r_values = list(range(1, n))
        zz_r = [z_correlation(gs, (n - 1 - r) // 2, (n - 1 - r) // 2 + r) for r in r_values]
        out["correlation_vs_distance"][str(h)] = {"r": r_values, "zz": zz_r}
        print(f"  <Z Z> vs separation at h={h}: " + ", ".join(f"{v:.3g}" for v in zz_r), flush=True)
    return out


def plot(res: dict, path: Path, theme: str) -> None:
    plt, c = setup(theme)
    h = np.asarray(res["h"])
    a, b = res["sites"]
    series = {"exact": ("exact ground state (cat-like below h = 1)", c["slots"][0]),
              "broken": (f"with symmetry-breaking field ε = {res['eps']}", c["slots"][1])}
    vqe_rows = res["vqe"]["rows"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0))
    fig.patch.set_facecolor(c["surface"])
    panels = [(left, "entropy", "Half-chain entanglement entropy, bits", "Entanglement across the middle"),
              (right, "zz", f"⟨Z{a} Z{b}⟩", f"Correlation between sites {a} and {b}")]
    for ax, key, ylabel, title in panels:
        style_axes(ax, c, title)
        ax.axvline(1.0, color=c["muted"], lw=0.8, ls=":")
        for name, (label, color) in series.items():
            ax.plot(h, res[name][key], color=color, lw=2, label=label)
        ax.plot([r["h"] for r in vqe_rows], [r[key] for r in vqe_rows], "o", color=c["slots"][2], ms=7,
                mec=c["surface"], mew=1.5, label=f"VQE, RY + CX, {res['vqe']['layers']} layers")
        ax.set_xlabel("Transverse field h / J")
        ax.set_ylabel(ylabel)
    left.annotate("h = 1", (1.0, 0.02), xytext=(4, 0), textcoords="offset points", color=c["muted"], fontsize=8)
    left.legend(frameon=False, loc="upper right", labelcolor=c["text"], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=c["surface"])
    plt.close(fig)


def plot_correlations(res: dict, path: Path, theme: str) -> None:
    """<Z_a Z_b> against separation in each phase: flat, slowly decaying, fast decaying."""
    plt, c = setup(theme)
    names = {"0.5": "ordered, h = 0.5", "1.0": "critical, h = 1", "1.5": "disordered, h = 1.5"}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor(c["surface"])
    style_axes(ax, c, f"Correlation against separation, {res['n']}-qubit chain (pairs centered)")
    for color, (key, label) in zip(c["slots"], names.items()):
        d = res["correlation_vs_distance"][key]
        ax.plot(d["r"], d["zz"], "-o", color=color, lw=2, ms=6, mec=c["surface"], mew=1.5)
        ax.annotate(label, (d["r"][-1], d["zz"][-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=c["text"], fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(res["correlation_vs_distance"]["1.0"]["r"])
    ax.set_xlim(0.5, res["n"] + 2.2)  # room for the direct labels
    ax.set_xlabel("Separation r = b − a")
    ax.set_ylabel("⟨Z_a Z_b⟩, exact ground state")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=c["surface"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax

    meta = {"seed": args.seed, "steps": args.steps, "restarts": args.restarts, "numpy": np.__version__,
            "jax": jax.__version__, "python": platform.python_version(), "platform": platform.platform()}
    print("  ".join(f"{k}={v}" for k, v in meta.items()))
    h_grid = [round(x, 2) for x in np.arange(0.0, 2.0001, 0.05)]
    h_vqe = [round(x, 2) for x in np.arange(0.25, 2.0001, 0.25)]
    res = compute(args.n, args.eps, h_grid, h_vqe, args.layers, args.steps, args.restarts, args.seed)
    res["meta"] = meta

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "results" / "phase_transition.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for theme in THEMES:
        plot(res, HERE / "figures" / f"phase-transition-{theme}.png", theme)
        plot_correlations(res, HERE / "figures" / f"correlations-{theme}.png", theme)
    print("Wrote experiments/results/phase_transition.json and figures")


if __name__ == "__main__":
    main()
