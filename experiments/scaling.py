"""Where Python actually costs time: per-call overhead against memory-bound work.

Three measurements, on one CPU:

  A. Time to apply one single-qubit gate with the NumPy simulator, for
     n = 1 .. 24. Small states are dominated by a fixed per-call overhead
     (Python and NumPy dispatch); large ones by memory traffic, doubling with
     each qubit. The crossover is where the time reaches twice the
     small-n floor.
  B. Time per gate for a whole circuit (4 layers of RX and RZ on every qubit
     and a CX ladder), run three ways: the NumPy simulator gate by gate, the
     same after fusing single-qubit runs (fuse_single_qubit_gates), and the
     JAX backend compiled into one program. Compile time is reported apart.
  C. 64 parameter sets of the same circuit: a Python loop of compiled calls
     against one call batched with jax.vmap.

Run from the repo root:  python -m experiments.scaling
Writes experiments/results/scaling.json and two figures.
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


def best_time(f, budget: float = 0.4, max_repeats: int = 200) -> float:
    """Minimum time of f() over as many repeats as fit in ``budget`` seconds."""
    f()  # warm up
    times = []
    start = time.perf_counter()
    while len(times) < max_repeats and (len(times) < 3 or time.perf_counter() - start < budget):
        t0 = time.perf_counter()
        f()
        times.append(time.perf_counter() - t0)
    return min(times)


def per_gate(ns: list[int]) -> dict:
    from quantum_simulator import gates
    from quantum_simulator.state import State, apply_gate

    times = []
    for n in ns:
        psi = State.from_label("+" * n).psi
        q = n // 2
        times.append(best_time(lambda: apply_gate(psi, gates.H, [q])))
        print(f"  A n={n:2d}: {times[-1] * 1e6:10.1f} us per gate", flush=True)
    floor = float(np.median(times[:4]))
    crossover = next((n for n, t in zip(ns, times) if t > 2 * floor), None)
    big = [(n, t) for n, t in zip(ns, times) if n >= 18]
    slope = float(np.polyfit([n for n, _ in big], np.log2([t for _, t in big]), 1)[0]) if len(big) >= 3 else None
    return {"n": ns, "seconds": times, "overhead_floor": floor, "crossover_n": crossover,
            "log2_slope_large_n": slope}


def layered_circuit(n: int, layers: int):
    from quantum_simulator.circuit import Circuit

    c = Circuit(n)
    for _ in range(layers):
        for q in range(n):
            c.rx(q).rz(q)
        for q in range(n - 1):
            c.cx(q, q + 1)
    return c


def circuits(ns: list[int], layers: int, seed: int) -> dict:
    import jax

    from quantum_simulator.backends import jax_backend as jb
    from quantum_simulator.circuit import fuse_single_qubit_gates, run

    rows = []
    for n in ns:
        c = layered_circuit(n, layers)
        params = np.random.default_rng(seed).uniform(0, 2 * np.pi, c.n_params)
        fused = fuse_single_qubit_gates(c)
        state = jax.jit(jb.make_state(c))
        t0 = time.perf_counter()
        jax.block_until_ready(state(params))
        compile_s = time.perf_counter() - t0
        np.testing.assert_allclose(np.asarray(state(params)).reshape(-1), run(c, params).state.vector, atol=1e-10)
        row = {
            "n": n, "ops": len(c.ops), "fused_ops": len(fused.ops), "compile_s": compile_s,
            "numpy_s": best_time(lambda: run(c, params)),
            "fused_s": best_time(lambda: run(fused, params)),
            "jax_s": best_time(lambda: jax.block_until_ready(state(params))),
        }
        rows.append(row)
        print(f"  B n={n:2d}: per gate  numpy {row['numpy_s'] / row['ops'] * 1e6:8.1f} us | fused "
              f"{row['fused_s'] / row['ops'] * 1e6:8.1f} us | jax {row['jax_s'] / row['ops'] * 1e6:8.1f} us "
              f"| compile {compile_s:5.1f} s", flush=True)
    return {"layers": layers, "rows": rows}


def batching(n: int, layers: int, batch: int, seed: int) -> dict:
    import jax

    from quantum_simulator.backends import jax_backend as jb
    from quantum_simulator.observables import tfim

    c = layered_circuit(n, layers)
    energy = jb.make_energy(c, tfim(n), jit=False)
    single = jax.jit(energy)
    batched = jax.jit(jax.vmap(energy))
    params = np.random.default_rng(seed).uniform(0, 2 * np.pi, (batch, c.n_params))
    jax.block_until_ready(single(params[0]))
    jax.block_until_ready(batched(params))
    loop_s = best_time(lambda: [float(single(p)) for p in params])
    vmap_s = best_time(lambda: jax.block_until_ready(batched(params)))
    np.testing.assert_allclose(np.asarray(batched(params)), [float(single(p)) for p in params], atol=1e-12)
    print(f"  C n={n}, {batch} parameter sets: loop {loop_s * 1e3:.2f} ms, vmap {vmap_s * 1e3:.2f} ms "
          f"({loop_s / vmap_s:.1f}x)", flush=True)
    return {"n": n, "layers": layers, "batch": batch, "loop_s": loop_s, "vmap_s": vmap_s}


def plot(res: dict, path: Path, theme: str) -> None:
    plt, c = setup(theme)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    fig.patch.set_facecolor(c["surface"])

    a = res["per_gate"]
    n = np.asarray(a["n"])
    t = np.asarray(a["seconds"]) * 1e6
    style_axes(left, c, "One gate, NumPy simulator")
    left.axhline(a["overhead_floor"] * 1e6, color=c["muted"], lw=1, ls=(0, (3, 3)))
    left.annotate(f"per-call overhead ≈ {a['overhead_floor'] * 1e6:.0f} µs", (n[0], a["overhead_floor"] * 1e6),
                  xytext=(0, 6), textcoords="offset points", color=c["muted"], fontsize=8)
    left.plot(n, t, "-o", color=c["slots"][0], lw=2, ms=5, mec=c["surface"], mew=1)
    if a["crossover_n"]:
        left.axvline(a["crossover_n"], color=c["muted"], lw=0.8, ls=":")
        left.annotate(f"overhead stops dominating\nnear n = {a['crossover_n']}", (a["crossover_n"], t.max() / 30),
                      xytext=(-6, 0), textcoords="offset points", ha="right", color=c["text"], fontsize=9)
    if a["log2_slope_large_n"]:
        left.annotate(f"memory-bound: ×{2 ** a['log2_slope_large_n']:.2f} per qubit", (n[-1], t[-1]),
                      xytext=(-6, 4), textcoords="offset points", ha="right", va="bottom", color=c["text"], fontsize=9)
    left.set_yscale("log")
    left.set_xticks(n[::2])
    left.set_xlabel("Qubits n")
    left.set_ylabel("Time per gate, µs")

    rows = res["circuits"]["rows"]
    nb = np.asarray([r["n"] for r in rows])
    style_axes(right, c, f"Whole circuit ({res['circuits']['layers']} layers), time per gate")
    series = [("numpy_s", "NumPy, gate by gate", c["slots"][0]),
              ("fused_s", "NumPy, single-qubit runs fused", c["slots"][1]),
              ("jax_s", "JAX, compiled once (compile not counted)", c["slots"][2])]
    for key, label, color in series:
        right.plot(nb, [r[key] / r["ops"] * 1e6 for r in rows], "-o", color=color, lw=2, ms=6,
                   mec=c["surface"], mew=1.5, label=label)
    right.set_yscale("log")
    right.set_xticks(nb)
    right.set_xlabel("Qubits n")
    right.set_ylabel("Time per original gate, µs")
    right.legend(frameon=False, loc="upper left", labelcolor=c["text"], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=c["surface"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax

    meta = {"seed": args.seed, "numpy": np.__version__, "jax": jax.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "devices": [str(d) for d in jax.devices()]}
    print("  ".join(f"{k}={v}" for k, v in meta.items()))
    res = {
        "per_gate": per_gate(list(range(1, args.max_n + 1))),
        "circuits": circuits([2, 4, 8, 12, 16, 20], 4, args.seed),
        "batching": batching(8, 4, 64, args.seed),
        "meta": meta,
    }
    print(f"  crossover near n = {res['per_gate']['crossover_n']}, large-n growth "
          f"x{2 ** res['per_gate']['log2_slope_large_n']:.2f} per qubit")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "results" / "scaling.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for theme in THEMES:
        plot(res, HERE / "figures" / f"scaling-{theme}.png", theme)
    print("Wrote experiments/results/scaling.json and figures")


if __name__ == "__main__":
    main()
