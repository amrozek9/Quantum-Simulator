"""Gate matrices, gate fusion, and controlled-gate construction.

All matrices are ``complex128`` and follow Qiskit's convention: for a gate
applied to ``qubits = [q0, q1]``, ``q0`` is the least significant bit of the
row/column index (see :mod:`quantum_simulator.state`). Constant gates are
read-only so a shared matrix cannot be corrupted by accident.
"""

from __future__ import annotations

from functools import reduce

import numpy as np

from .linalg import dagger, is_unitary, kron
from .state import DTYPE, apply_gate

__all__ = [
    "I", "X", "Y", "Z", "H", "S", "SDG", "T", "TDG", "SX",
    "CX", "CNOT", "CY", "CZ", "CH", "SWAP", "ISWAP",
    "rx", "ry", "rz", "phase", "u",
    "crx", "cry", "crz", "cphase", "rxx", "ryy", "rzz",
    "controlled", "fuse", "apply_gate", "dagger", "is_unitary",
]


def _const(rows) -> np.ndarray:
    m = np.array(rows, dtype=DTYPE)
    m.flags.writeable = False
    return m


_s = 1 / np.sqrt(2)

# -- single-qubit constants ---------------------------------------------------

I = _const([[1, 0], [0, 1]])
X = _const([[0, 1], [1, 0]])
Y = _const([[0, -1j], [1j, 0]])
Z = _const([[1, 0], [0, -1]])
H = _const([[_s, _s], [_s, -_s]])
S = _const([[1, 0], [0, 1j]])
SDG = _const([[1, 0], [0, -1j]])
T = _const([[1, 0], [0, np.exp(1j * np.pi / 4)]])
TDG = _const([[1, 0], [0, np.exp(-1j * np.pi / 4)]])
SX = _const(np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2)

_P0 = np.array([[1, 0], [0, 0]], dtype=DTYPE)  # |0><0|
_P1 = np.array([[0, 0], [0, 1]], dtype=DTYPE)  # |1><1|


# -- single-qubit rotations ---------------------------------------------------

def rx(theta: float) -> np.ndarray:
    """``exp(-i theta X / 2)``."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=DTYPE)


def ry(theta: float) -> np.ndarray:
    """``exp(-i theta Y / 2)``."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=DTYPE)


def rz(theta: float) -> np.ndarray:
    """``exp(-i theta Z / 2)``."""
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]).astype(DTYPE)


def phase(lam: float) -> np.ndarray:
    """``diag(1, e^{i lam})``; equals ``rz(lam)`` up to global phase."""
    return np.diag([1, np.exp(1j * lam)]).astype(DTYPE)


def u(theta: float, phi: float, lam: float) -> np.ndarray:
    """Generic single-qubit gate, Qiskit's ``UGate(theta, phi, lam)``."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [[c, -np.exp(1j * lam) * s],
         [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
        dtype=DTYPE,
    )


# -- multi-qubit construction ---------------------------------------------------

def controlled(gate: np.ndarray) -> np.ndarray:
    """Controlled version of ``gate``, applied to ``[control, *targets]``.

    The control is the least significant bit, so the result is
    ``gate ⊗ |1><1| + I ⊗ |0><0|``.
    """
    gate = np.asarray(gate, dtype=DTYPE)
    d = gate.shape[0]
    if gate.shape != (d, d) or d & (d - 1):
        raise ValueError(f"expected a 2^k x 2^k gate, got shape {gate.shape}")
    return kron(gate, _P1) + kron(np.eye(d, dtype=DTYPE), _P0)


def _two_qubit_rotation(pauli: np.ndarray, theta: float) -> np.ndarray:
    # exp(-i theta P⊗P / 2) = cos(theta/2) I - i sin(theta/2) P⊗P, since (P⊗P)^2 = I.
    return np.cos(theta / 2) * np.eye(4, dtype=DTYPE) - 1j * np.sin(theta / 2) * kron(pauli, pauli)


def rxx(theta: float) -> np.ndarray:
    """``exp(-i theta X⊗X / 2)``."""
    return _two_qubit_rotation(X, theta)


def ryy(theta: float) -> np.ndarray:
    """``exp(-i theta Y⊗Y / 2)``."""
    return _two_qubit_rotation(Y, theta)


def rzz(theta: float) -> np.ndarray:
    """``exp(-i theta Z⊗Z / 2)``."""
    return _two_qubit_rotation(Z, theta)


def crx(theta: float) -> np.ndarray:
    return controlled(rx(theta))


def cry(theta: float) -> np.ndarray:
    return controlled(ry(theta))


def crz(theta: float) -> np.ndarray:
    return controlled(rz(theta))


def cphase(lam: float) -> np.ndarray:
    return controlled(phase(lam))


def fuse(*gates: np.ndarray) -> np.ndarray:
    """One matrix equivalent to applying ``gates`` in the order given.

    ``gates`` is in time order (first applied first), so the result is
    ``G_k ... G_2 G_1``: the matrix product runs in the *reverse* order.
    All gates must act on the same qubits in the same order.
    """
    if not gates:
        raise ValueError("fuse needs at least one gate")
    shape = np.shape(gates[0])
    if any(np.shape(g) != shape for g in gates):
        raise ValueError(f"all gates must have the same shape, got {[np.shape(g) for g in gates]}")
    return reduce(lambda acc, g: g @ acc, gates)


# -- two-qubit constants ------------------------------------------------------

CX = CNOT = _const(controlled(X))
CY = _const(controlled(Y))
CZ = _const(controlled(Z))
CH = _const(controlled(H))
SWAP = _const([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
ISWAP = _const([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
