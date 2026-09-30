"""Bipartite entanglement of pure states via the Schmidt decomposition.

For a bipartition of the qubits into ``A`` (the ``qubits`` argument) and its
complement ``B``, the state tensor is regrouped into a ``2^|A| x 2^|B|``
matrix ``M`` and factored with ``np.linalg.svd``::

    M = U diag(s) V^H      =>     |psi> = sum_k s_k |u_k>_A |v_k>_B

The Schmidt coefficients ``s_k`` are the singular values; ``s_k^2`` is the
spectrum of the reduced density matrix, from which every entropy follows.

Within each subsystem, basis vectors are flat and little-endian over that
subsystem's qubits in ascending order: the smallest qubit of ``A`` is bit 0
of a row index of ``M``. This matches Qiskit's ``partial_trace``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .state import State, as_tensor, num_qubits, qubit_axis


def bipartition_matrix(state: State | np.ndarray, qubits: Sequence[int]) -> np.ndarray:
    """Reshape the state into the ``2^|A| x 2^|B|`` matrix for ``A = qubits``."""
    psi = as_tensor(state)
    n = num_qubits(psi)
    a = sorted(set(qubits))
    if len(a) != len(qubits):
        raise ValueError(f"repeated qubit in {list(qubits)}")
    for q in a:
        qubit_axis(q, n)  # range check
    b = [q for q in range(n) if q not in a]
    # Highest qubit first within each block keeps each block's flat index little-endian.
    order = [qubit_axis(q, n) for q in reversed(a)] + [qubit_axis(q, n) for q in reversed(b)]
    return np.transpose(psi, order).reshape(2 ** len(a), 2 ** len(b))


def schmidt_decomposition(
    state: State | np.ndarray, qubits: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(s, U, Vh)`` with ``psi = sum_k s[k] U[:, k] ⊗ Vh[k, :]``.

    ``s`` is non-negative and descending; ``U[:, k]`` lives on ``qubits`` and
    ``Vh[k, :]`` on the complement.
    """
    return _svd(bipartition_matrix(state, qubits))


def schmidt_coefficients(state: State | np.ndarray, qubits: Sequence[int]) -> np.ndarray:
    """Schmidt coefficients only (skips computing the singular vectors)."""
    return np.linalg.svd(bipartition_matrix(state, qubits), compute_uv=False)


def schmidt_rank(state: State | np.ndarray, qubits: Sequence[int], tol: float = 1e-10) -> int:
    """Number of Schmidt coefficients above ``tol``; 1 iff the cut is unentangled."""
    return int(np.count_nonzero(schmidt_coefficients(state, qubits) > tol))


def reduced_density_matrix(state: State | np.ndarray, qubits: Sequence[int]) -> np.ndarray:
    """``rho_A = Tr_B |psi><psi|`` keeping ``qubits``, as ``M M^H``."""
    m = bipartition_matrix(state, qubits)
    return m @ m.conj().T


def entanglement_entropy(
    state: State | np.ndarray,
    qubits: Sequence[int],
    alpha: float = 1.0,
    base: float = 2.0,
    tol: float = 1e-15,
) -> float:
    """Entropy of ``rho_A``: von Neumann for ``alpha = 1``, Rényi otherwise.

    Defaults to bits (``base=2``), so a Bell pair gives 1. Eigenvalues of
    ``rho_A`` below ``tol`` are dropped as numerical zeros.
    """
    p = schmidt_coefficients(state, qubits) ** 2
    p = p[p > tol]
    if alpha == 1:
        return float(-np.sum(p * np.log(p)) / np.log(base))
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if np.isinf(alpha):
        return float(-np.log(p.max()) / np.log(base))
    return float(np.log(np.sum(p**alpha)) / ((1 - alpha) * np.log(base)))


def purity(state: State | np.ndarray, qubits: Sequence[int]) -> float:
    """``Tr(rho_A^2) = sum_k s_k^4``; 1 for a product cut, ``2^-|A|`` at most entangled."""
    return float(np.sum(schmidt_coefficients(state, qubits) ** 4))


def _svd(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vh = np.linalg.svd(m, full_matrices=False)
    return s, u, vh
