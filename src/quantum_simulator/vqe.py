"""Variational ground states: a layered ansatz compiled with ``lax.scan``, and Adam.

The ansatz is the hardware-efficient layered circuit: a layer of RY on every
qubit, then ``layers`` repetitions of an entangling ladder (CX by default, or
CZ) followed by another RY layer, ``n * (layers + 1)`` parameters in all. It
produces real amplitudes, which suits stoquastic Hamiltonians such as the
transverse-field Ising model, whose ground states are real and non-negative.

The entangler matters more than it looks. With CX the ansatz's dynamical
Lie algebra (what its parameters can generate) is all of so(2^n); with CZ it
is only polynomial in size (:func:`dynamical_lie_algebra_dimension`), so RY+CZ
cannot reach generic states at any depth, but it also avoids the barren
plateaus that the full algebra brings.

:func:`make_ansatz_expectations` evaluates it with ``jax.lax.scan`` over
layers, so compiling costs the same at any depth (the generic
:mod:`~quantum_simulator.backends.jax_backend` path unrolls every gate, and its
compile time grows with the circuit). It returns several expectation values
at once, so a family of Hamiltonians that are linear combinations of the
same terms, such as the TFIM at every field ``h``, shares one compilation.

:func:`vqe` minimizes the energy with Adam from several random starts at
once (batched with ``vmap``) and keeps the best.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .backends import jax_backend  # noqa: F401  (imported to enable 64-bit JAX)
from .circuit import Circuit
from .observables import PauliSum

import jax  # noqa: E402  (after jax_backend has configured it)
import jax.numpy as jnp  # noqa: E402


ENTANGLERS = ("cx", "cz")


def hardware_efficient_ansatz(n: int, layers: int, entangler: str = "cx") -> Circuit:
    """RY on every qubit, then ``layers`` x (entangling ladder, RY on every qubit).

    The ladder is ``entangler`` on (0, 1), (1, 2), ..., (n-2, n-1), in that
    order. Parameter ``l * n + q`` is the RY angle on qubit ``q`` in RY layer
    ``l``, matching the ``(layers + 1, n)`` array taken by
    :func:`make_ansatz_expectations`.
    """
    if entangler not in ENTANGLERS:
        raise ValueError(f"entangler must be one of {ENTANGLERS}")
    c = Circuit(n)
    for q in range(n):
        c.ry(q)
    for _ in range(layers):
        for q in range(n - 1):
            getattr(c, entangler)(q, q + 1)
        for q in range(n):
            c.ry(q)
    return c


def make_ansatz_expectations(
    n: int, layers: int, observables: Sequence[PauliSum], dtype=None, entangler: str = "cx"
):
    """Compiled ``params -> [<O_1>, <O_2>, ...]`` for the hardware-efficient ansatz.

    ``params`` has shape ``(layers + 1, n)``. Each observable must be
    Hermitian. Differentiable with ``jax.grad`` and batchable with ``jax.vmap``.
    """
    from .state import resolve_dtype

    # RY, CX and CZ are real, so the state is a real vector: float32 for complex64, float64 otherwise.
    real_dt = jnp.float32 if resolve_dtype(dtype) == np.complex64 else jnp.float64
    if entangler not in ENTANGLERS:
        raise ValueError(f"entangler must be one of {ENTANGLERS}")
    for o in observables:
        if o.n != n or not o.is_hermitian():
            raise ValueError(f"observables must be Hermitian and act on {n} qubits")

    idx = np.arange(2**n)

    def parity_sign(mask: int) -> np.ndarray:
        return 1 - 2 * (np.bitwise_count(idx & mask) & 1).astype(np.int64)

    if entangler == "cz":
        # A CZ ladder is diagonal: (-1)^(number of adjacent 1-1 pairs).
        phases = np.ones(2**n)
        for q in range(n - 1):
            phases *= 1 - 2 * ((idx >> q) & (idx >> (q + 1)) & 1)
        phases = jnp.asarray(phases, dtype=real_dt)
        entangle = lambda psi: psi * phases  # noqa: E731
    else:
        # A CX ladder permutes basis states: |k> -> |f(k)>, applying CX(q, q+1) for q = 0, 1, ...
        f = idx.copy()
        for q in range(n - 1):
            f ^= ((f >> q) & 1) << (q + 1)
        source = jnp.asarray(np.argsort(f))  # the new amplitude at f(k) is the old one at k
        entangle = lambda psi: psi[source]  # noqa: E731

    def ry_layer(psi, angles):
        # RY on qubit q rotates each pair of amplitudes that differ only in bit q:
        # view the vector as (high bits, bit q, low bits) and mix the middle axis.
        for q in range(n):
            v = psi.reshape(2 ** (n - 1 - q), 2, 2**q)
            c, s = jnp.cos(angles[q] / 2), jnp.sin(angles[q] / 2)
            v = jnp.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], axis=1)
            psi = v.reshape(-1)
        return psi

    def layer(psi, angles):
        return ry_layer(entangle(psi), angles), None

    # Each Pauli string P maps |k> to i^(#Y) s(k) |k ^ x>, with x covering X and Y and
    # s(k) the sign from Z and Y. On a real state, <P> = Re(i^(#Y)) sum_k psi[k ^ x] s(k) psi[k]:
    # zero for an odd number of Y's. Diagonal strings (x = 0) of an observable are summed into one vector.
    plans = []
    for o in observables:
        diag = np.zeros(2**n)
        flips = []
        for label, coeff in o.terms:
            bit = {pos: 1 << (n - 1 - pos) for pos in range(n)}
            xmask = sum(bit[p] for p, c in enumerate(label) if c in "XY")
            signmask = sum(bit[p] for p, c in enumerate(label) if c in "YZ")
            n_y = label.count("Y")
            if n_y % 2:
                # The real operator left after factoring out i^(#Y) is antisymmetric, so its
                # expectation in a real state is exactly zero; skipping it only saves work.
                continue
            weight = coeff.real * (-1) ** (n_y // 2) * parity_sign(signmask)
            if xmask == 0:
                diag += weight
            else:
                flips.append((jnp.asarray(idx ^ xmask), jnp.asarray(weight, dtype=real_dt)))
        plans.append((jnp.asarray(diag, dtype=real_dt), flips))

    def expectations(params):
        params = jnp.asarray(params, dtype=real_dt)
        psi = jnp.zeros(2**n, dtype=real_dt).at[0].set(1.0)
        psi = ry_layer(psi, params[0])
        psi, _ = jax.lax.scan(layer, psi, params[1:])
        values = []
        for diag, flips in plans:
            value = jnp.sum(diag * psi**2)
            for perm, weight in flips:
                value = value + jnp.sum(psi[perm] * weight * psi)
            values.append(value)
        return jnp.stack(values)

    return jax.jit(expectations)


def ansatz_state(params: np.ndarray, dtype=None, entangler: str = "cx"):
    """The ansatz state for a ``(layers + 1, n)`` parameter array, from the NumPy simulator.

    For analysing optimized states (entropies, correlations); it is
    independent of the scan-based JAX path, so the tests compare the two.
    """
    from .circuit import run

    params = np.asarray(params)
    circuit = hardware_efficient_ansatz(params.shape[1], params.shape[0] - 1, entangler)
    return run(circuit, params.ravel(), dtype=dtype).state


# -- optimization -------------------------------------------------------------------


@dataclass(frozen=True)
class VQEResult:
    """Best parameters and energy over all restarts, plus every restart's energy history."""

    params: np.ndarray          # (layers + 1, n), the best restart's final parameters
    energy: float               # its final energy
    final_energies: np.ndarray  # (restarts,)
    history: np.ndarray         # (steps, restarts), energy before each step


def adam(learning_rate: float = 0.02, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
    """Adam (Kingma and Ba, 2015) as a pair of pure functions ``(init, update)``."""

    def init(params):
        return jnp.zeros_like(params), jnp.zeros_like(params), 0

    def update(params, grads, state):
        m, v, t = state
        t = t + 1
        m = b1 * m + (1 - b1) * grads
        v = b2 * v + (1 - b2) * grads**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        return params - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps), (m, v, t)

    return init, update


def vqe(
    n: int,
    layers: int,
    terms: Sequence[PauliSum],
    weights: Sequence[float],
    *,
    steps: int = 1000,
    restarts: int = 4,
    learning_rate: float = 0.02,
    init_scale: float = 2 * np.pi,
    seed: int = 0,
    entangler: str = "cx",
    expectations=None,
) -> VQEResult:
    """Minimize ``sum_k weights[k] <terms[k]>`` over the hardware-efficient ansatz.

    Each restart starts from angles uniform in ``[0, init_scale)``; all of
    them are optimized together in one batched, compiled step. Pass a
    precompiled ``expectations`` (from :func:`make_ansatz_expectations` with
    the same ``terms``) to reuse a compilation across calls, e.g. a field sweep.
    """
    if expectations is None:
        expectations = make_ansatz_expectations(n, layers, terms, entangler=entangler)
    w = jnp.asarray(weights, dtype=jnp.float64)
    energy = lambda p: jnp.dot(w, expectations(p))  # noqa: E731
    value_and_grad = jax.vmap(jax.value_and_grad(energy))  # one call evaluates every restart
    init, update = adam(learning_rate)  # elementwise, so it updates the whole batch at once

    @jax.jit
    def step(params, opt_state):
        values, grads = value_and_grad(params)
        params, opt_state = update(params, grads, opt_state)
        return params, opt_state, values

    rng = np.random.default_rng(seed)
    params = jnp.asarray(rng.uniform(0, init_scale, size=(restarts, layers + 1, n)))
    m, v, _ = init(params)
    opt_state = (m, v, jnp.asarray(0))
    history = np.empty((steps, restarts))
    for s in range(steps):
        params, opt_state, values = step(params, opt_state)
        history[s] = np.asarray(values)
    final = np.asarray(jax.vmap(energy)(params))
    best = int(np.argmin(final))
    return VQEResult(np.asarray(params[best]), float(final[best]), final, history)


# -- expressiveness -----------------------------------------------------------------


def dynamical_lie_algebra_dimension(n: int, entangler: str = "cx", max_dimension: int | None = None) -> int:
    """Dimension of the Lie algebra generated by the ansatz's parameters (its "DLA").

    The RY generators, conjugated by every power of the entangling ladder,
    are closed under commutators until nothing new appears. The full real
    algebra so(2^n) has dimension 2^(n-1) (2^n - 1). A smaller DLA bounds
    which states the ansatz can reach at any depth, and the size of the DLA
    governs gradient variance (Larocca et al., Quantum 6, 824 (2022); Ragone
    et al., Nat. Commun. 15, 7172 (2024)). Exact but expensive: practical up
    to about 5 qubits for CX.
    """
    from .circuit import run as run_circuit
    from .state import State

    if entangler not in ENTANGLERS:
        raise ValueError(f"entangler must be one of {ENTANGLERS}")
    dim = 2**n
    full = dim * (dim - 1) // 2
    max_dimension = full if max_dimension is None else max_dimension

    ladder = Circuit(n)
    for q in range(n - 1):
        getattr(ladder, entangler)(q, q + 1)
    C = np.column_stack([run_circuit(ladder, initial=State.basis(j, n)).state.vector.real for j in range(dim)])

    def ry_generator(q: int) -> np.ndarray:  # -iY on qubit q: real antisymmetric
        g = np.zeros((dim, dim))
        for k in range(dim):
            if not (k >> q) & 1:
                g[k | (1 << q), k] = 1
                g[k, k | (1 << q)] = -1
        return g

    # The RY generators conjugated by every distinct power of the ladder.
    generators, Ck, seen = [], np.eye(dim), set()
    while Ck.tobytes() not in seen:
        seen.add(Ck.tobytes())
        generators += [Ck @ ry_generator(q) @ Ck.T for q in range(n)]
        Ck = C @ Ck

    basis: list[np.ndarray] = []

    def add(m: np.ndarray) -> bool:
        v = m.ravel().copy()
        for b in basis:
            v -= (b @ v) * b
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            basis.append(v / norm)
            return True
        return False

    frontier = [g for g in generators if add(g)]
    while frontier and len(basis) < max_dimension:
        mats = [b.reshape(dim, dim) for b in basis]
        new = []
        for a in frontier:
            for b in mats:
                if add(a @ b - b @ a):
                    new.append(basis[-1].reshape(dim, dim))
        frontier = new
    return len(basis)
