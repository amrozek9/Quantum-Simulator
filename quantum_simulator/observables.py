"""Pauli-string observables and exact diagonalization.

An observable is a real-weighted sum of Pauli strings, ``H = sum_j c_j P_j``.
Labels follow Qiskit: ``label[0]`` acts on the *highest* qubit, so ``"ZI"``
is Z on qubit 1 and identity on qubit 0.

Expectation values and ``H|psi>`` are computed matrix-free, one single-qubit
Pauli at a time on the state tensor. Terms made only of Z and I are diagonal,
so their expectation needs no gates at all: it is a signed sum of
probabilities (:func:`z_expectation`). :meth:`PauliSum.to_matrix` builds the
dense ``2^n x 2^n`` matrix only for exact diagonalization with
``np.linalg.eigh``, which is limited to small ``n``.

:func:`pauli_string` builds labels from ``{qubit: letter}`` so nobody has to
count positions from the right, and :func:`tfim` builds the transverse-field
Ising model used as the reference problem throughout.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from numbers import Number

import numpy as np

from . import gates
from .linalg import ATOL, kron
from .state import State, apply_gate, as_tensor, default_atol, num_qubits, resolve_dtype

PAULIS = {"I": gates.I, "X": gates.X, "Y": gates.Y, "Z": gates.Z}

# Dense complex128 matrices are 16 * 4^n bytes: 12 qubits is 268 MB.
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
        """``H|psi>`` as a ``(2,)*n`` tensor (not normalized), in ``psi``'s precision."""
        return _apply_terms(self._check(state), self.terms)

    def expectation(self, state: State | np.ndarray) -> float | complex:
        """``<psi|H|psi>``; a float when ``H`` is Hermitian.

        Z/I-only terms are evaluated from probabilities, without gates; the
        rest by applying their Paulis. For a Hermitian ``H`` the result must
        be real, so an imaginary part beyond rounding (scaled by the
        precision and by ``sum |c_j|``) raises ``RuntimeError`` instead of
        being discarded: it means something upstream is wrong.
        """
        psi = self._check(state)
        diagonal = [(lab, c) for lab, c in self.terms if set(lab) <= {"I", "Z"}]
        others = [(lab, c) for lab, c in self.terms if not set(lab) <= {"I", "Z"}]
        val = 0j
        if diagonal:
            p = (np.abs(psi) ** 2).reshape(-1).astype(np.float64)
            for label, coeff in diagonal:
                val += coeff * _signed_sum(p, _z_mask(label))
        if others:
            val += complex(np.vdot(psi, _apply_terms(psi, others)))
        if not self.is_hermitian():
            return val
        tol = default_atol(psi.dtype) * max(1.0, sum(abs(c) for _, c in self.terms))
        if abs(val.imag) > tol:
            raise RuntimeError(
                f"Hermitian observable gave expectation {val} with imaginary part above {tol:.1e}; "
                "this indicates a bug upstream (a wrong gate or Pauli matrix), not rounding"
            )
        return val.real

    def variance(self, state: State | np.ndarray) -> float:
        """``<H^2> - <H>^2 = ||H psi||^2 - <H>^2``; zero exactly for eigenstates."""
        if not self.is_hermitian():
            raise ValueError("variance is defined here for Hermitian observables only")
        psi = self._check(state)
        h_psi = self.apply(psi)
        mean = np.vdot(psi, h_psi).real
        return float(max(np.vdot(h_psi, h_psi).real - mean**2, 0.0))

    # -- dense / exact ---------------------------------------------------------

    def to_matrix(
        self,
        max_qubits: int = DEFAULT_MAX_DENSE_QUBITS,
        dtype: np.typing.DTypeLike | None = None,
    ) -> np.ndarray:
        """Dense ``2^n x 2^n`` matrix, little-endian (matches ``SparsePauliOp.to_matrix``).

        Accumulated in complex128 and rounded once to ``dtype``.
        """
        dt = resolve_dtype(dtype)
        if self.n > max_qubits:
            raise ValueError(
                f"{self.n}-qubit dense matrix needs {dt.itemsize * 4**self.n / 1e9:.1f} GB; "
                f"raise max_qubits to force it"
            )
        mat = np.zeros((2**self.n, 2**self.n), dtype=np.complex128)
        for label, coeff in self.terms:
            mat += coeff * kron(*(PAULIS[c] for c in label))
        return mat.astype(dt, copy=False)

    def eigh(
        self,
        max_qubits: int = DEFAULT_MAX_DENSE_QUBITS,
        dtype: np.typing.DTypeLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """All eigenvalues (ascending) and eigenvectors (columns, flat little-endian).

        ``dtype=np.complex64`` diagonalizes in single precision (LAPACK ``cheevd``).
        """
        if not self.is_hermitian():
            raise ValueError("eigh requires a Hermitian observable")
        return np.linalg.eigh(self.to_matrix(max_qubits, dtype))

    def ground_state(
        self,
        max_qubits: int = DEFAULT_MAX_DENSE_QUBITS,
        dtype: np.typing.DTypeLike | None = None,
    ) -> tuple[float, State]:
        """Lowest eigenvalue and one eigenvector for it, by exact diagonalization.

        If the ground level is degenerate the returned vector is an arbitrary
        member of that eigenspace; check ``eigh()[0]`` for the gap. The state
        is returned in ``dtype``.
        """
        energies, vecs = self.eigh(max_qubits, dtype)
        return float(energies[0]), State(vecs[:, 0], dtype=resolve_dtype(dtype))

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


def pauli_string(n: int, ops: Mapping[int, str]) -> str:
    """Label for an ``n``-qubit Pauli string from ``{qubit: letter}``.

    ``pauli_string(3, {0: "Z", 2: "X"})`` is ``"XIZ"``: qubit 0 is the
    rightmost character, as in Qiskit.
    """
    label = ["I"] * n
    for q, letter in ops.items():
        if not 0 <= q < n:
            raise ValueError(f"qubit {q} out of range for {n} qubits")
        if letter.upper() not in PAULIS:
            raise ValueError(f"invalid Pauli letter {letter!r}")
        label[n - 1 - q] = letter.upper()
    return "".join(label)


def z_expectation(state: State | np.ndarray, qubits: Sequence[int]) -> float:
    """``<Z_a Z_b ...>`` from the probabilities alone, without applying any gate.

    The value is ``sum_k (-1)^parity(k) p_k``, where the parity counts the
    bits of ``k`` under the listed qubits. A qubit listed twice cancels
    (``Z^2 = I``), and an empty list gives ``<psi|psi>``.
    """
    psi = as_tensor(state)
    n = num_qubits(psi)
    mask = 0
    for q in qubits:
        if not 0 <= q < n:
            raise ValueError(f"qubit {q} out of range for {n} qubits")
        mask ^= 1 << q
    return _signed_sum((np.abs(psi) ** 2).reshape(-1).astype(np.float64), mask)


def z_correlation(state: State | np.ndarray, i: int, j: int, connected: bool = False) -> float:
    """``<Z_i Z_j>``, or with ``connected=True`` the connected ``<Z_i Z_j> - <Z_i><Z_j>``.

    Plotted against ``|i - j|`` this is the correlation function that shows
    order, and the transverse-field Ising phase transition.
    """
    zz = z_expectation(state, [i, j])
    if not connected:
        return zz
    return zz - z_expectation(state, [i]) * z_expectation(state, [j])


def tfim(n: int, J: float = 1.0, h: float = 1.0, periodic: bool = False) -> PauliSum:
    """Transverse-field Ising chain ``H = -J sum_i Z_i Z_{i+1} - h sum_i X_i``.

    Open boundaries by default; ``periodic=True`` adds the bond between
    qubits ``n-1`` and ``0`` (for ``n = 2`` that bond duplicates the other
    one, and the two merge into ``-2J Z_0 Z_1``).
    """
    bonds = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 1:
        bonds.append((n - 1, 0))
    terms = [(pauli_string(n, {a: "Z", b: "Z"}), -J) for a, b in bonds]
    terms += [(pauli_string(n, {i: "X"}), -h) for i in range(n)]
    return PauliSum(terms)


def tfim_periodic_ground_energy(n: int, J: float = 1.0, h: float = 1.0) -> float:
    """Exact ground energy of the periodic :func:`tfim` for even ``n`` and ``J, h >= 0``.

    Free-fermion (Jordan-Wigner) solution: ``E0 = -sum_k sqrt(J^2 + h^2 -
    2 J h cos k)`` with ``k = pi (2m + 1) / n``, ``m = 0 .. n-1``. Known
    answers like this are what the precision study and the VQE are checked
    against.
    """
    if n % 2 or n < 2:
        raise ValueError("this closed form holds for even n >= 2")
    if J < 0 or h < 0:
        raise ValueError("this closed form assumes J, h >= 0")
    k = np.pi * (2 * np.arange(n) + 1) / n
    return float(-np.sum(np.sqrt(J**2 + h**2 - 2 * J * h * np.cos(k))))


def _z_mask(label: str) -> int:
    n = len(label)
    return sum(1 << (n - 1 - pos) for pos, c in enumerate(label) if c == "Z")


def _signed_sum(p: np.ndarray, mask: int) -> float:
    parity = np.bitwise_count(np.arange(p.size) & mask) & 1
    return float(np.sum(np.where(parity, -p, p)))


def _apply_terms(psi: np.ndarray, terms) -> np.ndarray:
    out = np.zeros_like(psi)
    for label, coeff in terms:
        out += coeff * _apply_pauli_string(psi, label)
    return out


def _apply_pauli_string(psi: np.ndarray, label: str) -> np.ndarray:
    n = psi.ndim
    for pos, c in enumerate(label):
        if c != "I":
            psi = apply_gate(psi, PAULIS[c], [n - 1 - pos])  # label[pos] acts on qubit n-1-pos
    return psi
