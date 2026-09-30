"""Pure-state representation.

An n-qubit pure state is a unit vector in C^(2^n). We store it as a complex
NumPy array of shape ``(2,) * n``: one axis per qubit, each of length 2. The
flat vector and the tensor are the same memory; ``reshape`` moves between them
without copying, and the tensor view turns every gate into one call to
``tensordot`` along an axis instead of a loop over bit patterns.

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

Gate matrices
-------------
A k-qubit gate applied to ``qubits = [q0, q1, ...]`` follows Qiskit's matrix
convention: ``qubits[0]`` is the least significant bit of the gate's row and
column index. So CX applied to ``[control, target]`` is::

    [[1, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0],
     [0, 1, 0, 0]]
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce

import numpy as np

from . import linalg

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


def apply_gate(psi: np.ndarray, gate: np.ndarray, qubits: Sequence[int]) -> np.ndarray:
    """Return ``U psi`` for a ``2^k x 2^k`` gate acting on ``qubits``.

    The gate is reshaped to a ``(2,)*2k`` tensor, contracted against the
    target axes with ``tensordot``, and the new axes are moved back into
    place with ``moveaxis``. Cost is O(2^n * 2^k); the full 2^n x 2^n
    operator is never formed. ``qubits[0]`` is the gate's least significant
    bit (see the module docstring).
    """
    n = num_qubits(psi)
    k = len(qubits)
    gate = np.asarray(gate)
    if gate.shape != (2**k, 2**k):
        raise ValueError(f"gate shape {gate.shape} does not act on {k} qubit(s)")
    if len(set(qubits)) != k:
        raise ValueError(f"repeated qubit in {list(qubits)}")
    # The gate tensor's axes run from its most significant bit down, i.e. qubits reversed.
    axes = [qubit_axis(q, n) for q in reversed(qubits)]
    out = np.tensordot(gate.reshape((2,) * (2 * k)), psi, axes=(range(k, 2 * k), axes))
    return np.moveaxis(out, range(k), axes)


def as_tensor(state: State | np.ndarray) -> np.ndarray:
    """The ``(2,)*n`` tensor of a :class:`State` or a raw array (flat or tensor)."""
    if isinstance(state, State):
        return state.psi
    arr = np.asarray(state)
    if arr.ndim == 1:
        return from_vector(arr, check_norm=False)
    num_qubits(arr)
    return arr


_LABEL_STATES = {
    "0": np.array([1, 0], dtype=DTYPE),
    "1": np.array([0, 1], dtype=DTYPE),
    "+": np.array([1, 1], dtype=DTYPE) / np.sqrt(2),
    "-": np.array([1, -1], dtype=DTYPE) / np.sqrt(2),
    "r": np.array([1, 1j], dtype=DTYPE) / np.sqrt(2),
    "l": np.array([1, -1j], dtype=DTYPE) / np.sqrt(2),
}


class State:
    """A pure n-qubit state.

    ``psi`` is the ``(2,)*n`` complex128 amplitude tensor, laid out as in the
    module docstring. Gate application mutates ``psi`` in place (and returns
    ``self`` so calls chain); use :meth:`copy` to branch.
    """

    __slots__ = ("psi",)

    def __init__(self, psi: np.ndarray, *, check_norm: bool = True, atol: float = 1e-10):
        psi = np.asarray(psi, dtype=DTYPE)
        if psi.ndim == 1:
            psi = from_vector(psi, check_norm=check_norm, atol=atol)
        else:
            num_qubits(psi)
            if check_norm and not is_normalized(psi, atol=atol):
                raise ValueError(f"state has norm {np.linalg.norm(psi)}, expected 1")
        # A strided input (e.g. an eigenvector column) would otherwise make .vector strided too.
        self.psi = np.ascontiguousarray(psi)

    # -- construction -------------------------------------------------------

    @classmethod
    def zero(cls, n: int) -> State:
        """``|0...0>``."""
        return cls(zero_state(n))

    @classmethod
    def basis(cls, index: int, n: int) -> State:
        """Basis state with little-endian flat ``index``."""
        return cls(basis_state(index, n))

    @classmethod
    def from_label(cls, label: str) -> State:
        """Product state from a Qiskit-style label over ``0 1 + - r l``.

        ``label[0]`` is the highest qubit, so ``"01"`` has qubit 0 in ``|1>``.
        """
        try:
            factors = [_LABEL_STATES[c] for c in label]
        except KeyError as e:
            raise ValueError(f"invalid label character {e.args[0]!r} in {label!r}") from None
        if not factors:
            raise ValueError("label must be non-empty")
        # multiply.outer stacks axes left to right: label[0] lands on axis 0 = qubit n-1.
        return cls(reduce(np.multiply.outer, factors))

    # -- properties ---------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of qubits."""
        return self.psi.ndim

    @property
    def vector(self) -> np.ndarray:
        """Flat length-``2**n`` amplitudes, identical to Qiskit's ``Statevector.data``."""
        return to_vector(self.psi)

    def norm(self) -> float:
        """2-norm of the amplitudes (1 up to rounding for a valid state)."""
        return float(np.linalg.norm(self.psi))

    def probabilities(self, qubits: Sequence[int] | None = None) -> np.ndarray:
        """Born-rule outcome probabilities, flat and little-endian.

        With ``qubits``, returns the marginal distribution over those qubits,
        with ``qubits[0]`` as the least significant bit of the outcome index
        (Qiskit's ``Statevector.probabilities(qargs)``).
        """
        p = np.abs(self.psi) ** 2
        if qubits is None:
            return p.reshape(-1)
        qubits = list(qubits)
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"repeated qubit in {qubits}")
        keep = [qubit_axis(q, self.n) for q in qubits]
        p = p.sum(axis=tuple(ax for ax in range(self.n) if ax not in keep))
        # Surviving axes are in ascending-axis order; reorder so qubits[-1] comes first.
        remaining = sorted(keep)
        p = p.transpose([remaining.index(ax) for ax in reversed(keep)])
        return p.reshape(-1)

    # -- linear algebra -----------------------------------------------------

    def inner(self, other: State | np.ndarray) -> complex:
        """``<self|other>``."""
        return linalg.inner(self.psi, as_tensor(other))

    def fidelity(self, other: State | np.ndarray) -> float:
        """``|<self|other>|^2``."""
        return linalg.fidelity(self.psi, as_tensor(other))

    def apply(self, gate: np.ndarray, qubits: Sequence[int]) -> State:
        """Apply ``gate`` to ``qubits`` in place; returns ``self``."""
        self.psi = apply_gate(self.psi, gate, qubits)
        return self

    def tensor(self, other: State) -> State:
        """``self ⊗ other``; ``other`` occupies the low qubits (Qiskit's ``tensor``)."""
        return State(np.multiply.outer(self.psi, as_tensor(other)))

    # -- measurement --------------------------------------------------------

    def sample_counts(self, shots: int, rng: np.random.Generator | int | None = None) -> dict[str, int]:
        """Sample ``shots`` computational-basis measurements.

        Keys are bitstrings with qubit 0 rightmost, as in Qiskit's counts.
        """
        rng = np.random.default_rng(rng)
        p = self.probabilities()
        counts = rng.multinomial(shots, p / p.sum())
        return {format(k, f"0{self.n}b"): int(c) for k, c in enumerate(counts) if c}

    # -- misc ---------------------------------------------------------------

    def copy(self) -> State:
        return State(self.psi.copy(), check_norm=False)

    def __repr__(self) -> str:
        return f"State(n={self.n}, vector={np.array2string(self.vector, precision=4)})"
