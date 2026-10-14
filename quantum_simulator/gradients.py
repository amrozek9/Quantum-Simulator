"""Exact circuit gradients by the parameter-shift rule, using the NumPy simulator.

This is how gradients are measured on hardware, which cannot backpropagate:
each derivative is a weighted difference of energies at shifted angles. It
is an independent derivation from :mod:`quantum_simulator.jax_backend`'s
autodiff, so the two checking each other is a strong test.

Which rule applies depends on the gate's generator. Writing the gate as
``exp(-i theta G)``:

* ``rx, ry, rz, rxx, ryy, rzz``: ``G = P/2`` with ``P^2 = I``, eigenvalues
  ``+-1/2``. The energy is ``a + b cos(theta) + c sin(theta)``, and the
  two-term rule ``[f(theta + pi/2) - f(theta - pi/2)] / 2`` is exact.
* ``p, cp``: ``G`` is a projector, eigenvalues 0 and 1, so again a single
  frequency (up to a global phase) and the same two-term rule.
* ``crx, cry, crz``: ``G = |1><1| (x) P/2``, eigenvalues ``0, +-1/2``. The
  energy then has frequencies 1/2 and 1, and two terms are not enough. The
  four-term rule (Anselmetti et al., New J. Phys. 23, 113010 (2021)) is
  ``d1 [f(+pi/2) - f(-pi/2)] - d2 [f(+3pi/2) - f(-3pi/2)]`` with
  ``d1, d2 = (sqrt(2) +- 1) / (4 sqrt(2))``.

A parameter used by several gates gets one shift per gate, summed (the
product rule); shifting the shared parameter once would be wrong.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np

from .circuit import Circuit, Op, run
from .observables import PauliSum

TWO_TERM = frozenset({"rx", "ry", "rz", "rxx", "ryy", "rzz", "p", "cp"})
FOUR_TERM = frozenset({"crx", "cry", "crz"})
_D1 = (np.sqrt(2) + 1) / (4 * np.sqrt(2))
_D2 = (np.sqrt(2) - 1) / (4 * np.sqrt(2))


def parameter_shift_gradient(
    circuit: Circuit, hamiltonian: PauliSum, params: Sequence[float], dtype=None
) -> np.ndarray:
    """Exact gradient of ``<H>`` with respect to ``params`` by parameter shifts.

    Costs 2 energy evaluations per two-term gate occurrence and 4 per
    controlled rotation, so it grows with the number of parameters, unlike
    reverse-mode autodiff.
    """
    params = np.asarray(params, dtype=np.float64)
    flat = _unfuse(circuit)
    grad = np.zeros(circuit.n_params)
    for i, op in enumerate(flat.ops):
        if op.param is None:
            continue

        def f(shift: float, i=i, op=op) -> float:
            shifted = dataclasses.replace(op, param=None, value=params[op.param] + shift)
            return _energy(_replace_op(flat, i, shifted), hamiltonian, params, dtype)

        if op.name in TWO_TERM:
            grad[op.param] += (f(np.pi / 2) - f(-np.pi / 2)) / 2
        elif op.name in FOUR_TERM:
            grad[op.param] += _D1 * (f(np.pi / 2) - f(-np.pi / 2)) - _D2 * (f(1.5 * np.pi) - f(-1.5 * np.pi))
        else:
            raise ValueError(f"no shift rule for {op.name}")
    return grad


def two_term_shift_gradient(
    circuit: Circuit, hamiltonian: PauliSum, params: Sequence[float], dtype=None
) -> np.ndarray:
    """The two-term rule applied to every gate, including controlled rotations.

    Wrong for ``crx, cry, crz``; it exists so the tests can show that, which
    is the reason :func:`parameter_shift_gradient` uses four terms for them.
    """
    params = np.asarray(params, dtype=np.float64)
    flat = _unfuse(circuit)
    grad = np.zeros(circuit.n_params)
    for i, op in enumerate(flat.ops):
        if op.param is None:
            continue
        plus = dataclasses.replace(op, param=None, value=params[op.param] + np.pi / 2)
        minus = dataclasses.replace(op, param=None, value=params[op.param] - np.pi / 2)
        grad[op.param] += (
            _energy(_replace_op(flat, i, plus), hamiltonian, params, dtype)
            - _energy(_replace_op(flat, i, minus), hamiltonian, params, dtype)
        ) / 2
    return grad


def finite_difference_gradient(f: Callable[[np.ndarray], float], params: Sequence[float], eps: float = 1e-6) -> np.ndarray:
    """Central differences: approximate, with error O(eps^2) plus rounding O(1/eps)."""
    params = np.asarray(params, dtype=np.float64)
    grad = np.zeros_like(params)
    for k in range(params.size):
        step = np.zeros_like(params)
        step[k] = eps
        grad[k] = (f(params + step) - f(params - step)) / (2 * eps)
    return grad


def _energy(circuit: Circuit, hamiltonian: PauliSum, params: np.ndarray, dtype) -> float:
    p = params if circuit.n_params else None
    return hamiltonian.expectation(run(circuit, p, dtype=dtype).state)


def _unfuse(circuit: Circuit) -> Circuit:
    out = Circuit(circuit.n, circuit.n_clbits, n_params=circuit.n_params)
    for op in circuit.ops:
        out.ops.extend(op.ops if op.name == "fused" else (op,))
    return out


def _replace_op(circuit: Circuit, i: int, op: Op) -> Circuit:
    out = Circuit(circuit.n, circuit.n_clbits, ops=list(circuit.ops), n_params=circuit.n_params)
    out.ops[i] = op
    return out
