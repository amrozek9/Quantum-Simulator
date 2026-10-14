"""Differentiable simulation with JAX.

The same tensor layout, qubit order and gate conventions as the NumPy
simulator, written as pure functions in ``jax.numpy`` so that ``jax.grad``
can differentiate an energy with respect to every circuit parameter at
once, and ``jax.jit`` can compile the whole circuit into one program::

    energy = make_energy(circuit, hamiltonian)          # params -> <H>
    value, grad = make_value_and_grad(circuit, hamiltonian)(params)

A :class:`~quantum_simulator.circuit.Circuit` already stores parameters as
indices into a vector, so its structure is fixed at trace time and only the
parameter array flows through the compiled function.

Importing this module enables JAX's 64-bit mode (``jax_enable_x64``);
without it JAX silently computes complex128 requests in complex64, which
would make the precision parameter meaningless. Only unitary circuits can be
differentiated: measurements and classically conditioned gates are rejected.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import reduce

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (after the config update)
import numpy as np  # noqa: E402

from .circuit import FIXED_GATES, MULTI_CONTROLLED, PARAMETRIC_GATES, Circuit, Op, multi_controlled_matrix  # noqa: E402
from .observables import PAULIS, PauliSum  # noqa: E402
from .state import resolve_dtype  # noqa: E402

# -- gates as differentiable functions of their angle -----------------------------

_I2 = jnp.eye(2)
_X, _Y, _Z = (jnp.asarray(PAULIS[c]) for c in "XYZ")
_P0 = jnp.array([[1, 0], [0, 0]])
_P1 = jnp.array([[0, 0], [0, 1]])


def rx(theta):
    c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
    return jnp.array([[c, -1j * s], [-1j * s, c]])


def ry(theta):
    c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.complex128)


def rz(theta):
    return jnp.diag(jnp.array([jnp.exp(-0.5j * theta), jnp.exp(0.5j * theta)]))


def phase(lam):
    return jnp.diag(jnp.array([1.0 + 0j, jnp.exp(1j * lam)]))


def controlled(u):
    """Control on the gate's least significant qubit, as in :func:`gates.controlled`."""
    return jnp.kron(u, _P1) + jnp.kron(jnp.eye(u.shape[0]), _P0)


def _pp_rotation(p, theta):
    return jnp.cos(theta / 2) * jnp.eye(4) - 1j * jnp.sin(theta / 2) * jnp.kron(p, p)


JAX_PARAMETRIC: dict[str, Callable] = {
    "rx": rx, "ry": ry, "rz": rz, "p": phase,
    "crx": lambda t: controlled(rx(t)), "cry": lambda t: controlled(ry(t)),
    "crz": lambda t: controlled(rz(t)), "cp": lambda t: controlled(phase(t)),
    "rxx": lambda t: _pp_rotation(_X, t), "ryy": lambda t: _pp_rotation(_Y, t),
    "rzz": lambda t: _pp_rotation(_Z, t),
}
assert JAX_PARAMETRIC.keys() == PARAMETRIC_GATES.keys()


# -- the state ----------------------------------------------------------------------


def apply_gate(psi, gate, qubits: Sequence[int]):
    """Pure-function ``U psi``: the same contraction as ``state.apply_gate``, returning a new array."""
    n, k = psi.ndim, len(qubits)
    axes = [n - 1 - q for q in reversed(qubits)]  # qubit t lives on axis n-1-t
    out = jnp.tensordot(gate.reshape((2,) * (2 * k)), psi, axes=(list(range(k, 2 * k)), axes))
    return jnp.moveaxis(out, list(range(k)), axes)


def make_state(circuit: Circuit, dtype=None) -> Callable:
    """``params -> psi``: the circuit applied to ``|0...0>``, as a ``(2,)*n`` JAX array."""
    _check_unitary(circuit)
    dt = resolve_dtype(dtype)
    ops, n = tuple(circuit.ops), circuit.n
    fixed = {name: jnp.asarray(m, dtype=dt) for name, m in FIXED_GATES.items()}

    def matrix(op: Op, params):
        if op.name in fixed:
            return fixed[op.name]
        if op.name in MULTI_CONTROLLED:
            return jnp.asarray(multi_controlled_matrix(op.name, len(op.qubits)), dtype=dt)
        if op.name == "fused":
            # Time order A, B, ... is the product ... B @ A.
            return reduce(lambda acc, m: m @ acc, (matrix(o, params) for o in op.ops))
        theta = op.value if op.value is not None else params[op.param]
        return JAX_PARAMETRIC[op.name](theta).astype(dt)

    def state(params):
        psi = jnp.zeros((2,) * n, dtype=dt).at[(0,) * n].set(1.0)
        for op in ops:
            psi = apply_gate(psi, matrix(op, params), op.qubits)
        return psi

    return state


# -- energies and gradients ----------------------------------------------------------


def make_energy(circuit: Circuit, hamiltonian: PauliSum, dtype=None, jit: bool = True) -> Callable:
    """``params -> <psi(params)|H|psi(params)>``, real and differentiable.

    Z/I-only terms are combined into one diagonal vector ahead of time, so
    they cost a single weighted sum of probabilities. Other terms apply their
    Paulis. ``H`` must be Hermitian; the NumPy path's check on the imaginary
    part cannot raise inside a traced function, so here the real part is
    taken and parity with the NumPy result is what the tests check.
    """
    if not hamiltonian.is_hermitian():
        raise ValueError("make_energy needs a Hermitian observable")
    if hamiltonian.n != circuit.n:
        raise ValueError(f"observable acts on {hamiltonian.n} qubits, circuit has {circuit.n}")
    dt = resolve_dtype(dtype)
    real_dt = jnp.float32 if dt == np.complex64 else jnp.float64
    state = make_state(circuit, dt)
    n = circuit.n

    idx = np.arange(2**n)
    diagonal = np.zeros(2**n)
    others = []
    for label, coeff in hamiltonian.terms:
        if set(label) <= {"I", "Z"}:
            mask = sum(1 << (n - 1 - pos) for pos, c in enumerate(label) if c == "Z")
            diagonal += coeff.real * (1 - 2 * (np.bitwise_count(idx & mask) & 1).astype(np.int64))
        else:
            others.append((label, coeff.real))
    diagonal = jnp.asarray(diagonal.reshape((2,) * n), dtype=real_dt)
    paulis = {c: jnp.asarray(PAULIS[c], dtype=dt) for c in "XYZ"}

    def energy(params):
        psi = state(params)
        value = jnp.sum(diagonal * jnp.abs(psi) ** 2)
        for label, coeff in others:
            phi = psi
            for pos, c in enumerate(label):
                if c != "I":
                    phi = apply_gate(phi, paulis[c], [n - 1 - pos])
            value = value + coeff * jnp.real(jnp.vdot(psi, phi))
        return value

    if not jit:
        return energy
    compiled = jax.jit(energy)
    return lambda params: compiled(jnp.asarray(params, dtype=real_dt))


def make_value_and_grad(circuit: Circuit, hamiltonian: PauliSum, dtype=None) -> Callable:
    """Compiled ``params -> (energy, gradient)`` by reverse-mode autodiff.

    ``params`` may be any sequence; it is converted to one array first, so
    the gradient is always an array of shape ``(n_params,)`` (``jax.grad``
    would otherwise mirror a Python list's structure).
    """
    real_dt = jnp.float32 if resolve_dtype(dtype) == np.complex64 else jnp.float64
    compiled = jax.jit(jax.value_and_grad(make_energy(circuit, hamiltonian, dtype, jit=False)))
    return lambda params: compiled(jnp.asarray(params, dtype=real_dt))


def _check_unitary(circuit: Circuit) -> None:
    for op in circuit.ops:
        if op.name == "measure" or op.condition is not None:
            raise ValueError(f"differentiable execution needs a unitary circuit; got {op}")
