"""Density matrices and noise: mixed states, Kraus channels, and quantum trajectories.

A density matrix rho on n qubits is stored as a ``(2,)*2n`` tensor: the first
n axes are the ket ("row") index and the last n the bra ("column") index,
each in the same little-endian layout as a state vector (qubit t on ket axis
n-1-t and bra axis 2n-1-t). A gate evolves it as ``U rho U^dagger``, which is
U on the ket axes and conj(U) on the bra axes; a channel with Kraus
operators ``K_k`` (``sum_k K_k^dagger K_k = I``) maps rho to
``sum_k K_k rho K_k^dagger``. Memory is 4^n, against 2^n for a pure state.

The usual way around that cost is :func:`run_trajectories`: simulate the pure
state, and after each noisy step pick one Kraus operator at random with
probability ``||K_k psi||^2`` and renormalize. The average of an observable
over many trajectories converges to its value in the density matrix, with
statistical error falling as 1/sqrt(trajectories).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .. import gates
from ..circuit import Circuit, op_matrix
from ..observables import PauliSum
from ..state import State, _as_rng, apply_gate

_DTYPE = np.complex128


# -- channels ---------------------------------------------------------------------


def depolarizing(p: float) -> list[np.ndarray]:
    """With probability p, replace the qubit by the maximally mixed state.

    Kraus form: ``sqrt(1 - 3p/4) I`` and ``sqrt(p/4)`` times each of X, Y, Z.
    """
    _check_probability(p)
    return [np.sqrt(1 - 3 * p / 4) * gates.I] + [np.sqrt(p / 4) * P for P in (gates.X, gates.Y, gates.Z)]


def amplitude_damping(gamma: float) -> list[np.ndarray]:
    """Energy loss: |1> decays to |0> with probability gamma (T1 processes)."""
    _check_probability(gamma)
    return [np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=_DTYPE),
            np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=_DTYPE)]


def phase_damping(lam: float) -> list[np.ndarray]:
    """Loss of coherence without energy loss (T2 processes): off-diagonals shrink by sqrt(1 - lam)."""
    _check_probability(lam)
    return [np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=_DTYPE),
            np.array([[0, 0], [0, np.sqrt(lam)]], dtype=_DTYPE)]


def bit_flip(p: float) -> list[np.ndarray]:
    """X with probability p."""
    _check_probability(p)
    return [np.sqrt(1 - p) * gates.I, np.sqrt(p) * gates.X]


def is_trace_preserving(kraus: Sequence[np.ndarray], atol: float = 1e-12) -> bool:
    """Whether ``sum_k K_k^dagger K_k = I``."""
    total = sum(k.conj().T @ k for k in kraus)
    return bool(np.allclose(total, np.eye(total.shape[0]), atol=atol))


def _check_probability(p: float) -> None:
    if not 0 <= p <= 1:
        raise ValueError(f"probability must be in [0, 1], got {p}")


@dataclass(frozen=True)
class NoiseModel:
    """A one-qubit channel applied to every qubit a gate touches, right after the gate."""

    kraus: list[np.ndarray] = field(default_factory=lambda: depolarizing(0.0))

    def __post_init__(self):
        if not is_trace_preserving(self.kraus):
            raise ValueError("Kraus operators must satisfy sum K^dagger K = I")


# -- the density matrix --------------------------------------------------------------


class DensityMatrix:
    """An n-qubit density matrix as a ``(2,)*2n`` tensor (ket axes, then bra axes)."""

    def __init__(self, rho: np.ndarray):
        rho = np.asarray(rho, dtype=_DTYPE)
        if rho.ndim == 2:
            n = rho.shape[0].bit_length() - 1
            if rho.shape != (2**n, 2**n):
                raise ValueError("a density matrix must be 2^n x 2^n")
            rho = rho.reshape((2,) * (2 * n))
        if rho.shape != (2,) * rho.ndim or rho.ndim % 2:
            raise ValueError(f"expected shape (2,)*2n, got {rho.shape}")
        self.rho = rho

    @classmethod
    def zero(cls, n: int) -> DensityMatrix:
        return cls.from_state(State.zero(n))

    @classmethod
    def from_state(cls, state: State) -> DensityMatrix:
        """``|psi><psi|``."""
        return cls(np.multiply.outer(state.psi, state.psi.conj()))

    @property
    def n(self) -> int:
        return self.rho.ndim // 2

    def matrix(self) -> np.ndarray:
        """The 2^n x 2^n matrix, little-endian rows and columns."""
        return self.rho.reshape(2**self.n, 2**self.n)

    def trace(self) -> float:
        return float(np.trace(self.matrix()).real)

    def purity(self) -> float:
        """``Tr(rho^2)``: 1 for a pure state, ``2^-n`` for the maximally mixed one."""
        m = self.matrix()
        return float(np.real(np.vdot(m.conj().T, m)))

    def von_neumann_entropy(self) -> float:
        """``-Tr(rho log2 rho)`` in bits."""
        w = np.linalg.eigvalsh(self.matrix())
        w = w[w > 1e-15]
        return float(-np.sum(w * np.log2(w)))

    def probabilities(self) -> np.ndarray:
        """Outcome probabilities: the diagonal, little-endian."""
        return np.real(np.diag(self.matrix())).copy()

    def fidelity_with(self, state: State) -> float:
        """``<psi|rho|psi>`` for a pure ``state``."""
        v = state.vector
        return float(np.real(v.conj() @ self.matrix() @ v))

    def expectation(self, observable: PauliSum) -> float:
        """``Tr(rho H)``, by applying each Pauli string to the ket side and tracing."""
        if observable.n != self.n:
            raise ValueError(f"observable acts on {observable.n} qubits, state has {self.n}")
        total = 0j
        for label, coeff in observable.terms:
            m = self.rho
            for pos, c in enumerate(label):
                if c != "I":  # label[pos] acts on qubit n-1-pos
                    m = _apply_to_axes(m, gates_for(c), [self.n - 1 - pos], self.n, side="ket")
            total += coeff * np.trace(m.reshape(2**self.n, 2**self.n))
        return float(total.real)

    def apply(self, gate: np.ndarray, qubits: Sequence[int]) -> DensityMatrix:
        """``U rho U^dagger``, in place; returns ``self``."""
        self.rho = _apply_to_axes(self.rho, gate, qubits, self.n, side="ket")
        self.rho = _apply_to_axes(self.rho, np.conj(gate), qubits, self.n, side="bra")
        return self

    def apply_channel(self, kraus: Sequence[np.ndarray], qubits: Sequence[int]) -> DensityMatrix:
        """``sum_k K_k rho K_k^dagger`` for a channel on ``qubits``, in place."""
        out = np.zeros_like(self.rho)
        for k in kraus:
            term = _apply_to_axes(self.rho, k, qubits, self.n, side="ket")
            out += _apply_to_axes(term, np.conj(k), qubits, self.n, side="bra")
        self.rho = out
        return self


def gates_for(letter: str) -> np.ndarray:
    return {"X": gates.X, "Y": gates.Y, "Z": gates.Z}[letter]


def _apply_to_axes(rho: np.ndarray, gate: np.ndarray, qubits: Sequence[int], n: int, side: str) -> np.ndarray:
    """Apply ``gate`` to the ket or bra indices of ``qubits`` (the same contraction as a state vector)."""
    k = len(qubits)
    offset = 0 if side == "ket" else n
    axes = [offset + n - 1 - q for q in reversed(qubits)]
    out = np.tensordot(np.asarray(gate).reshape((2,) * (2 * k)), rho, axes=(range(k, 2 * k), axes))
    return np.moveaxis(out, range(k), axes)


# -- running circuits -------------------------------------------------------------------


def run_density(circuit: Circuit, params=None, noise: NoiseModel | None = None) -> DensityMatrix:
    """Run a unitary circuit on a density matrix, with ``noise`` after every gate."""
    rho = DensityMatrix.zero(circuit.n)
    for op in circuit.ops:
        if op.name == "measure" or op.condition is not None:
            raise ValueError("run_density takes unitary circuits; noise models measurement error instead")
        rho.apply(op_matrix(op, params), op.qubits)
        if noise is not None:
            for q in op.qubits:
                rho.apply_channel(noise.kraus, [q])
    return rho


def run_trajectories(
    circuit: Circuit,
    observable: PauliSum,
    trajectories: int,
    rng: np.random.Generator | int,
    params=None,
    noise: NoiseModel | None = None,
) -> tuple[float, float]:
    """Monte Carlo estimate of ``Tr(rho H)`` from pure-state trajectories.

    After each gate, on each qubit it touched, one Kraus operator is drawn
    with probability ``||K psi||^2`` and applied, then the state is
    renormalized. Returns the mean of ``<H>`` over trajectories and its
    standard error. Memory stays 2^n.
    """
    rng = _as_rng(rng)
    kraus = noise.kraus if noise is not None else [gates.I]
    values = np.empty(trajectories)
    for t in range(trajectories):
        psi = State.zero(circuit.n).psi
        for op in circuit.ops:
            if op.name == "measure" or op.condition is not None:
                raise ValueError("run_trajectories takes unitary circuits")
            psi = apply_gate(psi, op_matrix(op, params), op.qubits)
            if noise is None:
                continue
            for q in op.qubits:
                branches = [apply_gate(psi, k, [q]) for k in kraus]
                weights = np.array([np.vdot(b, b).real for b in branches])
                pick = rng.choice(len(kraus), p=weights / weights.sum())
                psi = branches[pick] / np.sqrt(weights[pick])
        values[t] = observable.expectation(State(psi, check_norm=False))
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(trajectories)) if trajectories > 1 else 0.0
