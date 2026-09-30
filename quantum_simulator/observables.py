"""Pauli-string observables and exact diagonalization.

An observable is a real-weighted sum of Pauli strings, ``H = sum_j c_j P_j``.
Labels follow Qiskit: ``label[0]`` acts on the *highest* qubit, so ``"ZI"``
is Z on qubit 1 and identity on qubit 0.

Expectation values and ``H|psi>`` are computed matrix-free, one single-qubit
Pauli at a time on the state tensor. :meth:`PauliSum.to_matrix` builds the
dense ``2^n x 2^n`` matrix only for exact diagonalization with
``np.linalg.eigh``, which is limited to small ``n``.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Number

import numpy as np

from . import gates
from .linalg import ATOL, kron
from .state import DTYPE, State, apply_gate, as_tensor, num_qubits

PAULIS = {"I": gates.I, "X": gates.X, "Y": gates.Y, "Z": gates.Z}

# Dense matrices are 16 * 4^n bytes: 12 qubits is 268 MB.
DEFAULT_MAX_DENSE_QUBITS = 12


class PauliSum:
    """``sum_j c_j P_j`` over Pauli strings of equal length.

    Build from ``[(label, coeff), ...]`` like Qiskit's
    ``SparsePauliOp.from_list``. Duplicate labels are merged and zero terms
    dropped. Supports ``+``, ``-`` and scalar ``*``.
    """

    __slots__ = ("terms", "n")

    def __init__(self, terms: Iterable[tuple[str, complex]], atol: float = 0.0):
        merged: dict[str, complex] = {}
        n = None
        for label, coeff in terms:
            label = label.upper()
            if not label or set(label) - PAULIS.keys():
                raise ValueError(f"invalid Pauli label {label!r}")
            if n is None:
                n = len(label)
            elif len(label) != n:
                raise ValueError(f"label {label!r} has length {len(label)}, expected {n}")
            merged[label] = merged.get(label, 0) + complex(coeff)
        if n is None:
            raise ValueError("PauliSum needs at least one term")
        self.n = n
        self.terms = tuple((lab, c) for lab, c in merged.items() if abs(c) > atol) or (("I" * n, 0j),)

    # -- algebra ------------------------------------------------------------

    def __add__(self, other: PauliSum) -> PauliSum:
        if not isinstance(other, PauliSum):
            return NotImplemented
        return PauliSum(self.terms + other.terms)

    def __neg__(self) -> PauliSum:
        return -1 * self

    def __sub__(self, other: PauliSum) -> PauliSum:
        return self + (-other)

    def __mul__(self, scalar: Number) -> PauliSum:
        if not isinstance(scalar, Number):
            return NotImplemented
        return PauliSum((lab, scalar * c) for lab, c in self.terms)

    __rmul__ = __mul__

    def is_hermitian(self, atol: float = ATOL) -> bool:
        """Pauli strings are Hermitian, so the sum is iff every weight is real."""
        return all(abs(c.imag) <= atol for _, c in self.terms)

    # -- action on states ----------------------------------------------------

    def apply(self, state: State | np.ndarray) -> np.ndarray:
        """``H|psi>`` as a ``(2,)*n`` tensor (not normalized)."""
        psi = self._check(state)
        out = np.zeros_like(psi, dtype=DTYPE)
        for label, coeff in self.terms:
            out += coeff * _apply_pauli_string(psi, label)
        return out

    def expectation(self, state: State | np.ndarray) -> float | complex:
        """``<psi|H|psi>``; a float when ``H`` is Hermitian."""
        psi = self._check(state)
        val = complex(np.vdot(psi, self.apply(psi)))
        return val.real if self.is_hermitian() else val

    def variance(self, state: State | np.ndarray) -> float:
        """``<H^2> - <H>^2 = ||H psi||^2 - <H>^2``; zero exactly for eigenstates."""
        if not self.is_hermitian():
            raise ValueError("variance is defined here for Hermitian observables only")
        psi = self._check(state)
        h_psi = self.apply(psi)
        mean = np.vdot(psi, h_psi).real
        return float(max(np.vdot(h_psi, h_psi).real - mean**2, 0.0))

    # -- dense / exact ---------------------------------------------------------

    def to_matrix(self, max_qubits: int = DEFAULT_MAX_DENSE_QUBITS) -> np.ndarray:
        """Dense ``2^n x 2^n`` matrix, little-endian (matches ``SparsePauliOp.to_matrix``)."""
        if self.n > max_qubits:
            raise ValueError(
                f"{self.n}-qubit dense matrix needs {16 * 4**self.n / 1e9:.1f} GB; "
                f"raise max_qubits to force it"
            )
        mat = np.zeros((2**self.n, 2**self.n), dtype=DTYPE)
        for label, coeff in self.terms:
            mat += coeff * kron(*(PAULIS[c] for c in label))
        return mat

    def eigh(self, max_qubits: int = DEFAULT_MAX_DENSE_QUBITS) -> tuple[np.ndarray, np.ndarray]:
        """All eigenvalues (ascending) and eigenvectors (columns, flat little-endian)."""
        if not self.is_hermitian():
            raise ValueError("eigh requires a Hermitian observable")
        return np.linalg.eigh(self.to_matrix(max_qubits))

    def ground_state(self, max_qubits: int = DEFAULT_MAX_DENSE_QUBITS) -> tuple[float, State]:
        """Lowest eigenvalue and one eigenvector for it, by exact diagonalization.

        If the ground level is degenerate the returned vector is an arbitrary
        member of that eigenspace; check ``eigh()[0]`` for the gap.
        """
        energies, vecs = self.eigh(max_qubits)
        return float(energies[0]), State(vecs[:, 0])

    # -- helpers ------------------------------------------------------------

    def _check(self, state: State | np.ndarray) -> np.ndarray:
        psi = as_tensor(state)
        if num_qubits(psi) != self.n:
            raise ValueError(f"observable acts on {self.n} qubits, state has {psi.ndim}")
        return psi

    def __repr__(self) -> str:
        body = ", ".join(f"({lab!r}, {c:.6g})" for lab, c in self.terms)
        return f"PauliSum([{body}])"


def pauli(label: str, coeff: complex = 1.0) -> PauliSum:
    """Single weighted Pauli string, e.g. ``pauli("ZZ", -1.0)``."""
    return PauliSum([(label, coeff)])


def _apply_pauli_string(psi: np.ndarray, label: str) -> np.ndarray:
    n = psi.ndim
    for pos, c in enumerate(label):
        if c != "I":
            psi = apply_gate(psi, PAULIS[c], [n - 1 - pos])  # label[pos] acts on qubit n-1-pos
    return psi
