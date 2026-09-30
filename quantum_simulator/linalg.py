"""Small dense linear-algebra primitives shared by the rest of the package.

Everything here works on plain NumPy arrays and has no knowledge of qubit
ordering; the conventions live in :mod:`quantum_simulator.state`.
"""

from __future__ import annotations

from functools import reduce

import numpy as np

ATOL = 1e-10


def dagger(a: np.ndarray) -> np.ndarray:
    """Conjugate transpose ``A†`` of a matrix."""
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {a.shape}")
    return a.conj().T


def inner(phi: np.ndarray, psi: np.ndarray) -> complex:
    """Inner product ``<phi|psi>``; conjugates the *first* argument.

    Accepts flat vectors or ``(2,)*n`` tensors (``np.vdot`` flattens both).
    """
    phi, psi = np.asarray(phi), np.asarray(psi)
    if phi.shape != psi.shape:
        raise ValueError(f"shape mismatch: {phi.shape} vs {psi.shape}")
    return complex(np.vdot(phi, psi))


def fidelity(phi: np.ndarray, psi: np.ndarray) -> float:
    """Pure-state fidelity ``|<phi|psi>|^2``, in [0, 1] for normalized states."""
    return abs(inner(phi, psi)) ** 2


def is_unitary(u: np.ndarray, atol: float = ATOL) -> bool:
    """Whether ``U†U = I``."""
    u = np.asarray(u)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        return False
    return bool(np.allclose(u.conj().T @ u, np.eye(u.shape[0]), atol=atol))


def is_hermitian(a: np.ndarray, atol: float = ATOL) -> bool:
    """Whether ``A = A†``."""
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return False
    return bool(np.allclose(a, a.conj().T, atol=atol))


def kron(*ops: np.ndarray) -> np.ndarray:
    """Kronecker product ``ops[0] ⊗ ops[1] ⊗ ...``.

    In the package's little-endian convention the *first* factor acts on the
    *highest* qubit, matching Qiskit's ``A.tensor(B)`` and bitstring labels.
    Only for small operators: states are never combined this way.
    """
    if not ops:
        raise ValueError("kron needs at least one operand")
    return reduce(np.kron, ops)


def equal_up_to_global_phase(a: np.ndarray, b: np.ndarray, atol: float = ATOL) -> bool:
    """Whether ``b = e^{i phi} a`` for some real ``phi`` (vectors or matrices)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not np.isclose(na, nb, atol=atol):
        return False
    return bool(np.isclose(abs(np.vdot(a, b)), na * nb, atol=atol))
