"""Deliberately naive reference implementations used to check the tensor code.

These loop over bit patterns and build full 2^n x 2^n matrices, i.e. exactly
what the package avoids, so they share no logic with it.
"""

import numpy as np


def dense_operator(gate: np.ndarray, qubits: list[int], n: int) -> np.ndarray:
    """Full operator for ``gate`` on ``qubits`` (qubits[0] = gate LSB), little-endian."""
    dim = 2**n
    full = np.zeros((dim, dim), dtype=complex)
    mask = sum(1 << q for q in qubits)
    for col in range(dim):
        c_sub = sum(((col >> q) & 1) << j for j, q in enumerate(qubits))
        for r_sub in range(2 ** len(qubits)):
            row = col & ~mask
            for j, q in enumerate(qubits):
                row |= ((r_sub >> j) & 1) << q
            full[row, col] = gate[r_sub, c_sub]
    return full


def random_state(n: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return vec / np.linalg.norm(vec)


def random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))
