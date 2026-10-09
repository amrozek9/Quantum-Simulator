"""Energy error against floating-point precision, on a problem with a known answer.

Problem: the periodic transverse-field Ising chain at its critical point,
``H = -sum_i Z_i Z_{i+1} - g sum_i X_i`` with ``g = 1``. For even ``n`` its
ground energy is known in closed form (Jordan-Wigner / free fermions)::

    E0 = -sum_{m=0}^{n-1} sqrt(1 + g^2 - 2 g cos(pi (2m + 1) / n))

Protocol (an "echo"): start in the exact ground state, apply ``depth`` layers
of a hardware-efficient ansatz (RY, RZ on every qubit, then a CX ring), then
the exact inverse of those layers. In exact arithmetic the state returns to
the ground state, so the final energy is ``E0`` and everything left over is
rounding error, accumulated the way it is in a long variational loop. The
whole pipeline (state preparation, gates, energy evaluation) runs in the
precision under test.

Run from the repo root:  py -m experiments.precision
Writes experiments/results/precision.json and two figures (light, dark).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from quantum_simulator import gates
from quantum_simulator.observables import tfim, tfim_periodic_ground_energy
from quantum_simulator.state import State

HERE = Path(__file__).parent
DTYPES = {"complex64": np.complex64, "complex128": np.complex128}
EPS = {"complex64": np.finfo(np.float32).eps, "complex128": np.finfo(np.float64).eps}


# -- the echo -------------------------------------------------------------------

def forward(state: State, params: np.ndarray) -> None:
    n = state.n
    for layer in params:
        for q, (a, b) in enumerate(layer):
            state.apply(gates.ry(a), [q]).apply(gates.rz(b), [q])
        for q in range(n):
            state.apply(gates.CX, [q, (q + 1) % n])


def backward(state: State, params: np.ndarray) -> None:
    n = state.n
    for layer in params[::-1]:
        for q in reversed(range(n)):
            state.apply(gates.CX, [q, (q + 1) % n])
        for q, (a, b) in reversed(list(enumerate(layer))):
            state.apply(gates.rz(-b), [q]).apply(gates.ry(-a), [q])


def run_echo(n: int, depths: list[int], seeds: int, g: float) -> dict:
    h = tfim(n, J=1.0, h=g, periodic=True)
    e_exact = tfim_periodic_ground_energy(n, J=1.0, h=g)
    e_ed, gs = h.ground_state()  # complex128 exact diagonalization
    assert abs(e_ed - e_exact) < 1e-9 * abs(e_exact), (e_ed, e_exact)

    out = {"n": n, "E0": e_exact, "depths": depths, "rel_error": {}, "rel_error_renormalized": {}, "norm_error": {},
           "seconds": {}}
    for name, dt in DTYPES.items():
        rel = np.zeros((seeds, len(depths)))
        ren = np.zeros_like(rel)
        nrm = np.zeros_like(rel)
        t0 = time.perf_counter()
        for s in range(seeds):
            params = np.random.default_rng(s).uniform(-np.pi, np.pi, size=(max(depths), n, 2))
            for j, d in enumerate(depths):
                state = gs.astype(dt)
                forward(state, params[:d])
                backward(state, params[:d])
                energy, norm = h.expectation(state), state.norm()
                rel[s, j] = abs(energy - e_exact) / abs(e_exact)
                # Rayleigh quotient <psi|H|psi>/<psi|psi>: removes the error due to norm drift.
                ren[s, j] = abs(energy / norm**2 - e_exact) / abs(e_exact)
                nrm[s, j] = abs(norm - 1)
        out["seconds"][name] = time.perf_counter() - t0
        out["rel_error"][name] = rel.tolist()
        out["rel_error_renormalized"][name] = ren.tolist()
        out["norm_error"][name] = nrm.tolist()
        print(f"  n={n:2d} {name:10s} median rel. error at depth {depths[-1]}: "
              f"{np.median(rel[:, -1]):.2e}, renormalized {np.median(ren[:, -1]):.2e}  "
              f"({out['seconds'][name]:.1f}s)")
    return out


# -- memory and speed -------------------------------------------------------------

def gate_timing(ns: list[int], repeats: int) -> list[dict]:
    rows = []
    for n in ns:
        row = {"n": n}
        for name, dt in DTYPES.items():
            state = State.from_label("+" * n, dt)
            state.apply(gates.H, [n // 2])  # warm-up
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                state.apply(gates.ry(0.3), [n // 2])
                times.append(time.perf_counter() - t0)
            row[name] = {"seconds_per_gate": float(np.median(times)), "bytes": state.nbytes}
        rows.append(row)
        print(f"  n={n}: complex64 {row['complex64']['seconds_per_gate'] * 1e3:.1f} ms/gate, "
              f"complex128 {row['complex128']['seconds_per_gate'] * 1e3:.1f} ms/gate")
    return rows


def fit_slope(depths: list[int], rel: list[list[float]], min_depth: int) -> float:
    d = np.asarray(depths)
    med = np.median(np.asarray(rel), axis=0)
    mask = (d >= min_depth) & (med > 0)
    if mask.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log10(d[mask]), np.log10(med[mask]), 1)[0])


# -- figure -----------------------------------------------------------------------

# Categorical slots 1 and 2 of the reference palette, stepped per mode.
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "complex128": "#2a78d6", "complex64": "#eb6834"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#3a3a37",
             "complex128": "#3987e5", "complex64": "#d95926"},
}


def plot(results: list[dict], path: Path, theme: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = THEMES[theme]
    plt.rcParams.update({
        "font.size": 10, "text.color": c["text"], "axes.labelcolor": c["muted"],
        "xtick.color": c["muted"], "ytick.color": c["muted"], "axes.edgecolor": c["grid"],
    })
    fig, axes = plt.subplots(1, len(results), figsize=(3.4 * len(results), 3.6), sharey=True)
    fig.patch.set_facecolor(c["surface"])
    for ax, res in zip(np.atleast_1d(axes), results):
        ax.set_facecolor(c["surface"])
        d = np.asarray(res["depths"])
        keep = d > 0
        for name in ["complex128", "complex64"]:
            rel = np.asarray(res["rel_error"][name])[:, keep]
            rel = np.where(rel > 0, rel, np.nan)  # an exact zero has no place on a log axis
            ax.fill_between(d[keep], np.nanmin(rel, 0), np.nanmax(rel, 0), color=c[name], alpha=0.18, lw=0)
            ax.plot(d[keep], np.nanmedian(rel, 0), color=c[name], lw=2, label=name, solid_capstyle="round")
        for name, eps in EPS.items():
            ax.axhline(eps, color=c["muted"], lw=0.8, ls=(0, (2, 3)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"n = {res['n']} qubits", color=c["text"], fontsize=10, loc="left")
        ax.grid(True, which="major", color=c["grid"], lw=0.6)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.set_xlabel("Echo depth (ansatz layers)")
    first, last = np.atleast_1d(axes)[0], np.atleast_1d(axes)[-1]
    first.set_ylabel("Relative energy error  |E − E₀| / |E₀|")
    first.set_ylim(3e-17, 1e-4)
    for name, eps in EPS.items():
        # Below each reference line at the panel's right edge, clear of the data.
        first.annotate(f"{'float32' if name == 'complex64' else 'float64'} machine ε", (1, eps),
                       xycoords=("axes fraction", "data"), xytext=(-2, -4), textcoords="offset points",
                       ha="right", va="top", color=c["muted"], fontsize=8)
    # Direct labels at the line ends, plus the legend for identity.
    res = results[-1]
    for name in ["complex128", "complex64"]:
        med = np.median(np.asarray(res["rel_error"][name])[:, -1])
        last.annotate(name, (res["depths"][-1], med), xytext=(4, 0), textcoords="offset points",
                      va="center", color=c["text"], fontsize=9)
    first.legend(frameon=False, loc="center left", labelcolor=c["text"])  # empty band between precisions
    fig.suptitle("Critical periodic TFIM: energy error after a depth-L echo, by precision",
                 color=c["text"], fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=c["surface"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--timing-sizes", type=int, nargs="+", default=[20, 22, 24])
    args = ap.parse_args()

    print("Echo experiment")
    echo = [run_echo(n, args.depths, args.seeds, args.g) for n in args.sizes]
    print("Single-qubit gate timing")
    timing = gate_timing(args.timing_sizes, repeats=15)

    slopes = {name: [fit_slope(r["depths"], r["rel_error"][name], 10) for r in echo] for name in DTYPES}
    print("log-log slope of median error vs depth (depth >= 10):", slopes)

    results_dir, fig_dir = HERE / "results", HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    meta = {"g": args.g, "seeds": args.seeds, "numpy": np.__version__}
    (results_dir / "precision.json").write_text(
        json.dumps({"meta": meta, "echo": echo, "timing": timing, "slopes": slopes}, indent=1)
    )
    for theme in THEMES:
        plot(echo, fig_dir / f"precision-{theme}.png", theme)
    print(f"Wrote {results_dir / 'precision.json'} and figures in {fig_dir}")


if __name__ == "__main__":
    main()
