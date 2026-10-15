"""VQE energy against ansatz depth, checked against exact diagonalization.

The open transverse-field Ising chain at its critical point (J = h = 1) on 8
qubits, whose exact ground energy comes from diagonalization. For each depth,
VQE (Adam, several random restarts optimized together) minimizes the energy
over the layered RY ansatz with either entangler, and the error against the
exact energy is recorded, with the best restart's fidelity to the exact
ground state.

The threshold marked is 1.6e-3 in units of J: the size of chemical accuracy
(1.6 mHa) carried over to this model's energy scale, as a yardstick for
"accurate enough".

Run from the repo root:  python -m experiments.vqe_convergence
Writes experiments/results/vqe_convergence.json and two figures.
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
THRESHOLD = 1.6e-3
LABELS = {"cx": "RY + CX ladder", "cz": "RY + CZ ladder"}


def compute(n: int, h: float, depths: list[int], steps: int, restarts: int, lr: float, seed: int) -> dict:
    from quantum_simulator.observables import PauliSum, pauli_string, tfim
    from quantum_simulator.vqe import ansatz_state, vqe

    e0, gs = tfim(n, 1.0, h).ground_state()
    zz = PauliSum([(pauli_string(n, {i: "Z", i + 1: "Z"}), 1.0) for i in range(n - 1)])
    xs = PauliSum([(pauli_string(n, {i: "X"}), 1.0) for i in range(n)])
    out: dict = {"n": n, "h": h, "E0": e0, "depths": depths, "threshold": THRESHOLD, "runs": {}}
    for ent in ("cx", "cz"):
        rows = []
        for layers in depths:
            t0 = time.perf_counter()
            r = vqe(n, layers, [zz, xs], [-1.0, -h], steps=steps, restarts=restarts,
                    learning_rate=lr, seed=seed, entangler=ent)
            errors = r.final_energies - e0
            fid = ansatz_state(r.params, entangler=ent).fidelity(gs)
            rows.append({"layers": layers, "n_params": n * (layers + 1), "best_error": float(errors.min()),
                         "median_error": float(np.median(errors)), "fidelity": float(fid)})
            print(f"  {ent} L={layers:2d}: best error {errors.min():.2e}, median {np.median(errors):.2e}, "
                  f"fidelity {fid:.5f}  ({time.perf_counter() - t0:.1f}s)", flush=True)
        out["runs"][ent] = rows
        below = [r["layers"] for r in rows if r["best_error"] < THRESHOLD]
        out.setdefault("first_depth_below_threshold", {})[ent] = below[0] if below else None
    return out


def plot(res: dict, path: Path, theme: str) -> None:
    plt, c = setup(theme)
    colors = {"cx": c["slots"][0], "cz": c["slots"][1]}
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    fig.patch.set_facecolor(c["surface"])
    style_axes(ax, c, f"VQE on the critical Ising chain, {res['n']} qubits: error against exact diagonalization")
    ax.axhline(res["threshold"], color=c["muted"], lw=1, ls=(0, (3, 3)))
    ax.annotate("1.6×10⁻³: chemical-accuracy scale", (res["depths"][0], res["threshold"]), xytext=(2, -12),
                textcoords="offset points", color=c["muted"], fontsize=8)
    for ent in ("cx", "cz"):
        rows = res["runs"][ent]
        layers = [r["layers"] for r in rows]
        ax.plot(layers, [r["best_error"] for r in rows], "-o", color=colors[ent], lw=2, ms=6,
                mec=c["surface"], mew=1.5, label=f"{LABELS[ent]}, best of restarts")
        ax.plot(layers, [r["median_error"] for r in rows], "o", color=colors[ent], ms=4, alpha=0.45,
                label=f"{LABELS[ent]}, median restart")
    first = res["first_depth_below_threshold"]["cx"]
    if first is not None:
        row = next(r for r in res["runs"]["cx"] if r["layers"] == first)
        ax.annotate(f"below threshold at {first} layers", (first, row["best_error"]), xytext=(0, -18),
                    textcoords="offset points", ha="center", color=c["text"], fontsize=9)
    cz_last = res["runs"]["cz"][-1]
    ax.annotate(f"CZ ceiling: fidelity {cz_last['fidelity']:.3f}", (res["depths"][-1], cz_last["best_error"]),
                xytext=(-4, 10), textcoords="offset points", ha="right", color=c["text"], fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(res["depths"])
    ax.set_xlabel("Ansatz depth (entangling layers)")
    ax.set_ylabel("E_VQE − E₀ (units of J)")
    # The band between the CZ plateau and the CX medians is empty; the legend goes there.
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.72), labelcolor=c["text"], fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=c["surface"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax

    meta = {"seed": args.seed, "steps": args.steps, "restarts": args.restarts, "lr": args.lr,
            "numpy": np.__version__, "jax": jax.__version__, "python": platform.python_version(),
            "platform": platform.platform()}
    print("  ".join(f"{k}={v}" for k, v in meta.items()))
    res = compute(args.n, args.h, list(range(1, args.max_depth + 1)), args.steps, args.restarts, args.lr, args.seed)
    res["meta"] = meta
    print(f"  first depth below {THRESHOLD}: {res['first_depth_below_threshold']}")

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "results" / "vqe_convergence.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for theme in THEMES:
        plot(res, HERE / "figures" / f"vqe-convergence-{theme}.png", theme)
    print("Wrote experiments/results/vqe_convergence.json and figures")


if __name__ == "__main__":
    main()
