"""What a gradient costs: reverse-mode autodiff against the parameter-shift rule.

Ansatz: L layers of RY and RZ on every qubit followed by a CX ladder, on 8
qubits (2 n L parameters), with energy <H> for the open transverse-field
Ising chain at h = 1. For each depth, the same gradient is computed three
ways and checked to agree:

  * JAX autodiff: one compiled value-and-gradient call.
  * Parameter shift on the compiled JAX energy: 2 evaluations per parameter.
  * Parameter shift on the NumPy simulator: the same rule, uncompiled.

The claim being tested is that autodiff costs a small
constant number of energy evaluations however many parameters there are,
while parameter shift costs two per parameter.

Run from the repo root:  python -m experiments.gradient_cost
Writes experiments/results/gradient_cost.json and two figures.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from experiments.style import THEMES, setup, style_axes

from quantum_simulator.circuit import Circuit
from quantum_simulator.gradients import parameter_shift_gradient
from quantum_simulator.observables import tfim

HERE = Path(__file__).parent


def ansatz(n: int, layers: int) -> Circuit:
    c = Circuit(n)
    for _ in range(layers):
        for q in range(n):
            c.ry(q).rz(q)
        for q in range(n - 1):
            c.cx(q, q + 1)
    return c


def best_time(f, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        times.append(time.perf_counter() - t0)
    return min(times)


def measure(n: int, layers_list: list[int], seed: int, repeats: int) -> list[dict]:
    import jax

    from quantum_simulator.backends import jax_backend as jb

    h = tfim(n, J=1.0, h=1.0)
    rng = np.random.default_rng(seed)
    rows = []
    for layers in layers_list:
        c = ansatz(n, layers)
        params = rng.uniform(-np.pi, np.pi, c.n_params)

        energy = jb.make_energy(c, h)
        value_and_grad = jb.make_value_and_grad(c, h)
        t0 = time.perf_counter()
        _, ad = value_and_grad(params)
        jax.block_until_ready(ad)
        compile_s = time.perf_counter() - t0
        jax.block_until_ready(energy(params))  # compile the energy too, outside the timing

        def jax_shift():
            g = np.empty(c.n_params)
            for k in range(c.n_params):
                step = np.zeros(c.n_params)
                step[k] = np.pi / 2
                g[k] = (float(energy(params + step)) - float(energy(params - step))) / 2
            return g

        t_energy = best_time(lambda: jax.block_until_ready(energy(params)), repeats)
        t_autodiff = best_time(lambda: jax.block_until_ready(value_and_grad(params)[1]), repeats)
        t_jax_shift = best_time(jax_shift, max(1, repeats // 3))
        t_numpy_shift = best_time(lambda: parameter_shift_gradient(c, h, params), 1)

        # The three gradients must agree before their timings mean anything.
        ad = np.asarray(ad)
        np.testing.assert_allclose(jax_shift(), ad, atol=1e-10)
        np.testing.assert_allclose(parameter_shift_gradient(c, h, params), ad, atol=1e-10)

        row = {
            "layers": layers, "n_params": c.n_params, "n_gates": len(c.ops), "compile_s": compile_s,
            "energy_s": t_energy, "autodiff_s": t_autodiff,
            "jax_shift_s": t_jax_shift, "numpy_shift_s": t_numpy_shift,
        }
        rows.append(row)
        print(f"  P={c.n_params:4d}: energy {t_energy * 1e3:7.2f} ms | autodiff {t_autodiff * 1e3:7.2f} ms "
              f"({t_autodiff / t_energy:4.1f} energies) | shift: JAX {t_jax_shift:6.3f} s, NumPy {t_numpy_shift:6.3f} s "
              f"| compile {compile_s:5.1f} s")
    return rows


# -- figure -----------------------------------------------------------------------

LABELS = {"autodiff": "JAX autodiff", "jax_shift": "parameter shift, JAX energy",
          "numpy_shift": "parameter shift, NumPy simulator"}


def plot(res: dict, path: Path, theme: str) -> None:
    plt, c = setup(theme)
    colors = {"autodiff": c["slots"][0], "jax_shift": c["slots"][1], "numpy_shift": c["slots"][2]}
    rows = res["rows"]
    p = np.array([r["n_params"] for r in rows])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0))
    fig.patch.set_facecolor(c["surface"])
    for ax in (left, right):
        style_axes(ax, c)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(p)
        ax.set_xticklabels([str(v) for v in p])
        ax.minorticks_off()
        ax.set_xlabel("Number of parameters")

    # Left: wall-clock time per gradient.
    for key in ("numpy_shift", "jax_shift", "autodiff"):
        t = np.array([r[f"{key}_s"] for r in rows])
        left.plot(p, t, "-o", color=colors[key], lw=2, ms=6, mec=c["surface"], mew=1.5, label=LABELS[key])
    left.set_ylabel("Seconds per gradient")
    left.set_title(f"Time per gradient, {res['n']} qubits", loc="left", color=c["text"], fontsize=10)
    left.legend(frameon=False, loc="upper left", labelcolor=c["text"], fontsize=9)
    ratio = rows[-1]["jax_shift_s"] / rows[-1]["autodiff_s"]
    left.annotate(f"{ratio:.0f}× faster than JAX\nparameter shift at {p[-1]}", (p[-1], rows[-1]["autodiff_s"]),
                  xytext=(-6, 14), textcoords="offset points", ha="right", color=c["text"], fontsize=9)

    # Right: cost in units of one energy evaluation (same compiled JAX energy for both).
    ad_units = np.array([r["autodiff_s"] / r["energy_s"] for r in rows])
    shift_units = np.array([r["jax_shift_s"] / r["energy_s"] for r in rows])
    right.plot(p, 2 * p, color=c["muted"], lw=1, ls=(0, (3, 3)))
    right.annotate("2 × parameters", (p[2], 2 * p[2]), xytext=(4, -14), textcoords="offset points",
                   color=c["muted"], fontsize=8)
    right.plot(p, shift_units, "-o", color=colors["jax_shift"], lw=2, ms=6, mec=c["surface"], mew=1.5,
               label=LABELS["jax_shift"])
    right.plot(p, ad_units, "-o", color=colors["autodiff"], lw=2, ms=6, mec=c["surface"], mew=1.5,
               label=LABELS["autodiff"])
    right.legend(frameon=False, loc="center right", labelcolor=c["text"], fontsize=9)
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    right.set_yticks(ticks)
    right.set_yticklabels([str(t) for t in ticks])
    right.set_ylim(1, 600)
    right.set_ylabel("Cost in energy evaluations")
    right.set_title("Autodiff stays flat; shifts grow with parameters", loc="left", color=c["text"], fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=c["surface"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=9)
    args = ap.parse_args()

    import jax

    meta = {"seed": args.seed, "numpy": np.__version__, "jax": jax.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "devices": [str(d) for d in jax.devices()]}
    print("  ".join(f"{k}={v}" for k, v in meta.items()))
    res = {"n": args.n, "rows": measure(args.n, args.layers, args.seed, args.repeats), "meta": meta}

    results_dir, fig_dir = HERE / "results", HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    (results_dir / "gradient_cost.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for theme in THEMES:
        plot(res, fig_dir / f"gradient-cost-{theme}.png", theme)
    print(f"Wrote {results_dir / 'gradient_cost.json'} and figures in {fig_dir}")


if __name__ == "__main__":
    main()
