"""Matrix product states against the dense simulator: exact, then truncated as entanglement grows.

A 16-qubit brickwork circuit: each layer applies random RY and RZ rotations
to every qubit, then CX on alternating neighbour pairs. An MPS with bond
dimension chi is exact exactly while the Schmidt rank (the number of nonzero
Schmidt coefficients) at every cut is at most chi. The entanglement entropy
is only a lower bound on log2 of that rank, and here it grows much more
slowly than the rank: each CX across a cut can double the rank while adding
far less than a bit of entropy. So truncation starts by discarding tiny
Schmidt coefficients, and the error grows smoothly from there.

For chi = 2 .. 256 the MPS and the dense simulator run the same circuit
side by side; after every layer this records the true fidelity, the MPS's
own estimate (the product of 1 - discarded weight), the dense state's
half-chain entropy, and log2 of its largest Schmidt rank over all cuts.

Run from the repo root:  python -m experiments.mps_truncation
Writes experiments/results/mps_truncation.json and two figures.
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
CHIS = [2, 4, 8, 16, 32, 64, 128, 256]

# Ordinal ramp (one hue, light to dark for larger chi) from the reference blue scale,
# within the documented bounds for discrete ordered marks on each surface.
RAMPS = {
    "light": ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"],
    "dark": ["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#86b6ef", "#cde2fb"],
}


def brickwork_layers(n: int, depth: int, seed: int):
    """Per layer: a list of (gate, qubits) using this package's gate matrices."""
    from quantum_simulator import gates

    rng = np.random.default_rng(seed)
    layers = []
    for d in range(depth):
        ops = []
        for q in range(n):
            ops.append((gates.ry(rng.uniform(0, 2 * np.pi)), [q]))
            ops.append((gates.rz(rng.uniform(0, 2 * np.pi)), [q]))
        for q in range(d % 2, n - 1, 2):
            ops.append((gates.CX, [q, q + 1]))
        layers.append(ops)
    return layers


def compute(n: int, depth: int, seed: int) -> dict:
    from quantum_simulator.backends.mps import MPS
    from quantum_simulator.entanglement import entanglement_entropy, schmidt_rank
    from quantum_simulator.state import State

    layers = brickwork_layers(n, depth, seed)
    dense = State.zero(n)
    dense_states, entropy, rank_bits = [], [], []
    for ops in layers:
        for gate, qubits in ops:
            dense.apply(gate, qubits)
        dense_states.append(dense.vector.copy())
        entropy.append(entanglement_entropy(dense, range(n // 2)))
        rank_bits.append(float(np.log2(max(schmidt_rank(dense, range(cut), tol=1e-12) for cut in range(1, n)))))

    out: dict = {"n": n, "depth": depth, "chis": CHIS, "entropy": entropy, "rank_bits": rank_bits, "runs": {}}
    for chi in CHIS:
        t0 = time.perf_counter()
        m = MPS(n, max_bond=chi)
        fid, est, bonds = [], [], []
        for d, ops in enumerate(layers):
            for gate, qubits in ops:
                m.apply(gate, qubits)
            fid.append(float(abs(np.vdot(dense_states[d], m.to_vector())) ** 2))
            est.append(m.fidelity_estimate())
            bonds.append(max(m.bond_dimensions))
        out["runs"][str(chi)] = {"fidelity": fid, "estimate": est, "max_bond": bonds}
        print(f"  chi={chi:3d}: final fidelity {fid[-1]:.6f} (estimate {est[-1]:.6f}), "
              f"first layer with infidelity > 1e-10: "
              f"{next((d + 1 for d, f in enumerate(fid) if 1 - f > 1e-10), None)}  ({time.perf_counter() - t0:.1f}s)",
              flush=True)
    return out


def plot(res: dict, path: Path, theme: str) -> None:
    plt, c = setup(theme)
    ramp = RAMPS[theme]
    depth = np.arange(1, res["depth"] + 1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor(c["surface"])

    style_axes(left, c, f"MPS infidelity against the dense state, {res['n']} qubits")
    floor = 1e-16
    saturated = []  # curves that end near infidelity 1 share one label
    for color, chi in zip(ramp, res["chis"]):
        inf = np.maximum(1 - np.asarray(res["runs"][str(chi)]["fidelity"]), floor)
        left.plot(depth, inf, "-o", color=color, lw=2, ms=4, mec=c["surface"], mew=1)
        if inf[-1] > 0.3:
            saturated.append(chi)
        else:
            left.annotate(f"χ = {chi}", (depth[-1], inf[-1]), xytext=(5, 0), textcoords="offset points",
                          va="center", color=c["text"], fontsize=8)
    if saturated:
        left.annotate(f"χ = {', '.join(map(str, saturated))}", (depth[-1], 1.0), xytext=(5, 6),
                      textcoords="offset points", va="center", color=c["text"], fontsize=8)
    left.set_yscale("log")
    left.set_ylim(floor / 3, 3)
    left.set_xlim(0.5, res["depth"] + 3.2)
    left.set_xticks(depth[::2])
    left.set_xlabel("Brickwork layers")
    left.set_ylabel("1 − fidelity (floored at 1e-16)")

    style_axes(right, c, "Schmidt rank sets exactness; entropy lags far behind")
    for chi in res["chis"]:
        right.axhline(np.log2(chi), color=c["grid"], lw=0.8)
        right.annotate(f"χ = {chi}", (res["depth"] + 0.6, np.log2(chi)), va="center", color=c["muted"], fontsize=7)
    right.step(depth, res["rank_bits"], where="mid", color=c["slots"][0], lw=2,
               label="log₂ largest Schmidt rank, any cut")
    right.plot(depth, res["entropy"], "-o", color=c["slots"][1], lw=2, ms=5, mec=c["surface"], mew=1,
               label="half-chain entropy")
    right.set_xlim(0.5, res["depth"] + 2.5)
    right.set_ylim(-0.3, np.log2(res["chis"][-1]) + 0.6)
    right.set_xticks(depth[::2])
    right.set_xlabel("Brickwork layers")
    right.set_ylabel("Bits")
    right.legend(frameon=False, loc="upper left", labelcolor=c["text"], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=c["surface"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = {"seed": args.seed, "numpy": np.__version__, "python": platform.python_version(),
            "platform": platform.platform()}
    print("  ".join(f"{k}={v}" for k, v in meta.items()))
    res = compute(args.n, args.depth, args.seed)
    res["meta"] = meta
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "results" / "mps_truncation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for theme in THEMES:
        plot(res, HERE / "figures" / f"mps-truncation-{theme}.png", theme)
    print("Wrote experiments/results/mps_truncation.json and figures")


if __name__ == "__main__":
    main()
