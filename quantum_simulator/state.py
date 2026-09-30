"""Pure-state representation.

An n-qubit pure state is a unit vector in C^(2^n). We store it as a complex
NumPy array of shape ``(2,) * n``: one axis per qubit, each of length 2. The
flat vector and the tensor are the same memory; ``reshape`` moves between them
without copying, and the tensor view turns every gate into one call to
``tensordot``/``einsum`` along an axis instead of a loop over bit patterns.

Index convention
----------------
Qiskit is little-endian: qubit ``t`` is bit ``t`` of the flat index, so
qubit 0 is the least significant bit::

    flat index k = sum_t  b_t * 2**t          (b_t = value of qubit t)

NumPy's ``reshape`` uses C order, so axis 0 of the tensor is the *most*
significant bit of the flat index. Putting the two together:

    **qubit t lives on tensor axis n - 1 - t.**

So for n = 3, ``psi[b2, b1, b0]`` is the amplitude of the basis state
``|b2 b1 b0>`` (Qiskit's printed bitstring order), and
``psi.reshape(-1)[k]`` matches ``Statevector.data[k]`` exactly. Always go
through :func:`qubit_axis` rather than doing this arithmetic inline.
"""

from __future__ import annotations

import numpy as np

DTYPE = np.complex128


def qubit_axis(qubit: int, n: int) -> int:
    """Tensor axis holding ``qubit`` in an ``n``-qubit state: ``n - 1 - qubit``."""
    if not 0 <= qubit < n:
        raise ValueError(f"qubit {qubit} out of range for {n} qubits")
    return n - 1 - qubit


def num_qubits(psi: np.ndarray) -> int:
    """Number of qubits in a state tensor of shape ``(2,) * n``."""
    if psi.shape != (2,) * psi.ndim:
        raise ValueError(f"expected shape (2,)*n, got {psi.shape}")
    return psi.ndim


def zero_state(n: int) -> np.ndarray:
    """The state ``|0...0>`` on ``n`` qubits."""
    return basis_state(0, n)


def basis_state(index: int, n: int) -> np.ndarray:
    """Computational basis state with flat index ``index`` (Qiskit ordering).

    Bit ``t`` of ``index`` is the value of qubit ``t``, so
    ``basis_state(0b001, 3)`` has qubit 0 set to 1.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not 0 <= index < 2**n:
        raise ValueError(f"index {index} out of range for {n} qubits")
    vec = np.zeros(2**n, dtype=DTYPE)
    vec[index] = 1.0
    return vec.reshape((2,) * n)


def from_vector(vec: np.ndarray, *, check_norm: bool = True, atol: float = 1e-10) -> np.ndarray:
    """Reshape a flat length-``2**n`` amplitude vector into a state tensor.

    ``vec`` is interpreted in Qiskit's little-endian ordering, so the result
    can be compared directly with ``qiskit.quantum_info.Statevector.data``.
    """
    vec = np.asarray(vec, dtype=DTYPE)
    if vec.ndim != 1:
        raise ValueError(f"expected a 1-D vector, got shape {vec.shape}")
    size = vec.size
    n = size.bit_length() - 1
    if size == 0 or size != 1 << n:
        raise ValueError(f"length {size} is not a power of 2")
    if check_norm and not is_normalized(vec, atol=atol):
        raise ValueError(f"state has norm {np.linalg.norm(vec)}, expected 1")
    return vec.reshape((2,) * n)


def to_vector(psi: np.ndarray) -> np.ndarray:
    """Flatten a state tensor to its length-``2**n`` amplitude vector.

    This is a view when ``psi`` is C-contiguous (it always is unless an
    operation left its axes permuted), and a copy otherwise.
    """
    num_qubits(psi)
    return psi.reshape(-1)


def is_normalized(psi: np.ndarray, atol: float = 1e-10) -> bool:
    """Whether ``psi`` (tensor or flat) has unit 2-norm."""
    return bool(abs(np.vdot(psi, psi).real - 1.0) <= atol)
