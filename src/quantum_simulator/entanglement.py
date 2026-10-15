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


def entropy_profile(state: State | np.ndarray, alpha: float = 1.0, base: float = 2.0) -> np.ndarray:
    """Entropy of the block of qubits ``0 .. l-1``, for every ``l`` from 0 to ``n``.

    ``profile[l]`` is the entanglement across the cut after qubit ``l - 1``,
    so the ends are 0, and since ``S(A) = S(complement)`` for a pure state it
    also equals the entropy of qubits ``l .. n-1``. The profile is symmetric
    under ``l -> n - l`` only for states that look the same from both ends,
    such as translation-invariant chains. Plotted against ``l``, it distinguishes an
    area law (levels off), logarithmic growth at a critical point, and a
    volume law (grows with ``l``, as for random states).
    """
    n = num_qubits(as_tensor(state))
    return np.array([entanglement_entropy(state, range(l), alpha, base) for l in range(n + 1)])


def page_entropy(n_a: int, n: int, base: float = 2.0) -> float:
    """Average entanglement entropy of ``n_a`` qubits in a Haar-random ``n``-qubit state.

    Page's formula (Page 1993), with ``m = 2^min(n_a, n - n_a)`` and
    ``d = 2^max(...)``: ``S = sum_{k=d+1}^{m d} 1/k - (m - 1)/(2 d)`` nats.
    For large ``n`` it approaches ``n_a - 1/(2 ln 2) * 2^(2 n_a - n)`` bits:
    nearly maximal, which is the volume law.
    """
    if not 0 <= n_a <= n:
        raise ValueError(f"need 0 <= n_a <= n, got n_a={n_a}, n={n}")
    m, d = 2 ** min(n_a, n - n_a), 2 ** max(n_a, n - n_a)
    if m * d <= 2**26:
        harmonic = float(np.sum(1.0 / np.arange(d + 1, m * d + 1)))
    else:  # H_{md} - H_d to O(1/d^2), avoiding a huge array
        harmonic = np.log(m) - (m - 1) / (2 * m * d)
    return (harmonic - (m - 1) / (2 * d)) / np.log(base)


def _svd(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vh = np.linalg.svd(m, full_matrices=False)
    return s, u, vh
