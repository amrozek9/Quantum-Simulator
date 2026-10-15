"""Matrix product states: memory O(n chi^2), exact while every cut's Schmidt rank is at most chi.

The Schmidt rank (number of nonzero Schmidt coefficients) is what decides
exactness. The entanglement entropy is only a lower bound on log2 of it and
can be much smaller, so an MPS can start truncating while the entropy is
still well below log2(chi); it then drops small coefficients first, which is
why the error usually grows smoothly (see ``experiments/mps_truncation.py``).

Qubit ``i`` is site ``i`` of a chain of tensors ``A[i]`` with shape
``(left bond, 2, right bond)``. The state is kept in mixed canonical form:
every site left of the orthogonality center is a left isometry and every
site right of it a right isometry. With that, the singular values at a bond
are exactly the Schmidt coefficients of the cut there, the same numbers
:mod:`quantum_simulator.entanglement` gets by SVD of the dense state, so
truncating them is the optimal approximation at that bond.

Gates:

* one-qubit gates contract into their site (a unitary on the physical index
  keeps a site's isometry property);
* two-qubit gates on neighbouring sites merge the two sites, apply the gate,
  and split them again by SVD, keeping at most ``max_bond`` singular values
  and dropping any whose squared weight falls below ``cutoff``;
* two-qubit gates on distant qubits are routed with SWAPs.

The discarded weight of every split is recorded: the product of
``(1 - discarded)`` over all splits estimates the fidelity with the exact
state, and ``experiments/mps_truncation.py`` checks it against the dense
simulator.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..circuit import Circuit, op_matrix
from ..gates import SWAP

_DTYPE = np.complex128


class MPS:
    """An n-qubit matrix product state, starting in ``|0...0>``."""

    def __init__(self, n: int, max_bond: int | None = None, cutoff: float = 1e-14):
        if n < 1:
            raise ValueError("an MPS needs at least one qubit")
        self.n = n
        self.max_bond = max_bond
        self.cutoff = cutoff
        zero = np.zeros((1, 2, 1), dtype=_DTYPE)
        zero[0, 0, 0] = 1.0
        self.tensors = [zero.copy() for _ in range(n)]
        self.center = 0
        self.discarded: list[float] = []  # discarded weight of every truncating split

    # -- construction and readout ---------------------------------------------------

    @classmethod
    def from_vector(cls, vec: np.ndarray, max_bond: int | None = None, cutoff: float = 1e-14) -> MPS:
        """Factor a little-endian state vector into an MPS by successive SVDs."""
        vec = np.asarray(vec, dtype=_DTYPE)
        n = vec.size.bit_length() - 1
        if vec.size != 2**n:
            raise ValueError("vector length must be a power of 2")
        m = cls(n, max_bond, cutoff)
        # Axis t of the reshaped tensor is qubit n-1-t; reverse so axis i is qubit (site) i.
        rest = np.transpose(vec.reshape((2,) * n)).reshape(1, -1)
        for i in range(n - 1):
            left = rest.shape[0]
            u, s, vh = np.linalg.svd(rest.reshape(left * 2, -1), full_matrices=False)
            keep = m._keep(s)
            m.tensors[i] = u[:, :keep].reshape(left, 2, keep)
            rest = s[:keep, None] * vh[:keep]
        m.tensors[-1] = rest.reshape(rest.shape[0], 2, 1)
        m.center = n - 1
        return m

    def to_vector(self) -> np.ndarray:
        """Contract the chain into a little-endian vector (2^n memory: for checking only)."""
        out = self.tensors[0]  # (1, 2, b)
        for a in self.tensors[1:]:
            out = np.tensordot(out, a, axes=([-1], [0]))
        out = out.reshape((2,) * self.n)  # axis i is qubit i
        return np.transpose(out).reshape(-1)  # qubit n-1 first: little-endian

    @property
    def bond_dimensions(self) -> list[int]:
        """Dimension of each of the n - 1 bonds, left to right."""
        return [a.shape[2] for a in self.tensors[:-1]]

    def norm(self) -> float:
        return float(np.linalg.norm(self.tensors[self.center]))

    def fidelity_estimate(self) -> float:
        """Product of (1 - discarded weight) over every truncation so far."""
        return float(np.prod([1 - d for d in self.discarded])) if self.discarded else 1.0

    def schmidt_values(self, cut: int) -> np.ndarray:
        """Schmidt coefficients across the cut between qubits ``cut - 1`` and ``cut``."""
        if not 0 < cut < self.n:
            raise ValueError(f"cut must be between 1 and {self.n - 1}")
        self._move_center(cut - 1)
        a = self.tensors[cut - 1]
        return np.linalg.svd(a.reshape(a.shape[0] * 2, -1), compute_uv=False)

    def entanglement_entropy(self, cut: int) -> float:
        """Von Neumann entropy in bits across the cut (qubits 0..cut-1 against the rest)."""
        p = self.schmidt_values(cut) ** 2
        p = p[p > 1e-15]
        return float(-np.sum(p * np.log2(p)))

    # -- gates ------------------------------------------------------------------

    def apply(self, gate: np.ndarray, qubits: Sequence[int]) -> MPS:
        """Apply a one- or two-qubit gate; ``qubits[0]`` is the gate's least significant bit."""
        gate = np.asarray(gate, dtype=_DTYPE)
        qubits = list(qubits)
        for q in qubits:
            if not 0 <= q < self.n:
                raise ValueError(f"qubit {q} out of range for {self.n} qubits")
        if len(qubits) == 1:
            q = qubits[0]
            self.tensors[q] = np.einsum("ij,ajb->aib", gate, self.tensors[q])
        elif len(qubits) == 2:
            a, b = qubits
            if a == b:
                raise ValueError("repeated qubit")
            # Route b next to a with SWAPs, apply, and route it back.
            path = list(range(b, a + 1, 1)) if b < a else list(range(b, a - 1, -1))
            for x, y in zip(path[:-2], path[1:-1]):
                self._apply_adjacent(SWAP, x, y)
            neighbour = path[-2]
            self._apply_adjacent(gate, a, neighbour)
            for x, y in reversed(list(zip(path[:-2], path[1:-1]))):
                self._apply_adjacent(SWAP, x, y)
        else:
            raise ValueError("the MPS backend applies one- and two-qubit gates")
        return self

    def _apply_adjacent(self, gate: np.ndarray, q0: int, q1: int) -> None:
        """Gate on neighbouring qubits; ``q0`` is the gate's least significant bit."""
        i = min(q0, q1)
        self._move_center(i)
        theta = np.tensordot(self.tensors[i], self.tensors[i + 1], axes=([2], [0]))  # (a, s_i, s_i+1, b)
        u4 = gate.reshape(2, 2, 2, 2)  # (out_hi, out_lo, in_hi, in_lo)
        if q0 == i:  # site i carries the low bit
            theta = np.einsum("pqrs,asrb->aqpb", u4, theta)
        else:  # site i carries the high bit
            theta = np.einsum("pqrs,arsb->apqb", u4, theta)
        left, right = theta.shape[0], theta.shape[3]
        u, s, vh = np.linalg.svd(theta.reshape(left * 2, 2 * right), full_matrices=False)
        keep = self._keep(s)
        total = float(np.sum(s**2))
        dropped = float(np.sum(s[keep:] ** 2)) / total if total > 0 else 0.0
        if keep < len(s):
            self.discarded.append(dropped)
        s = s[:keep] / np.sqrt(np.sum(s[:keep] ** 2) / total)  # renormalize what is kept
        self.tensors[i] = u[:, :keep].reshape(left, 2, keep)
        self.tensors[i + 1] = (s[:, None] * vh[:keep]).reshape(keep, 2, right)
        self.center = i + 1

    def _keep(self, s: np.ndarray) -> int:
        total = np.sum(s**2)
        keep = int(np.count_nonzero(s**2 > self.cutoff * total)) or 1
        return min(keep, self.max_bond) if self.max_bond else keep

    def _move_center(self, k: int) -> None:
        """Shift the orthogonality center to site ``k`` with QR decompositions."""
        while self.center < k:
            c = self.center
            a = self.tensors[c]
            q, r = np.linalg.qr(a.reshape(a.shape[0] * 2, a.shape[2]))
            self.tensors[c] = q.reshape(a.shape[0], 2, q.shape[1])
            self.tensors[c + 1] = np.tensordot(r, self.tensors[c + 1], axes=([1], [0]))
            self.center += 1
        while self.center > k:
            c = self.center
            a = self.tensors[c]
            q, r = np.linalg.qr(a.reshape(a.shape[0], 2 * a.shape[2]).T)  # LQ via QR of the transpose
            self.tensors[c] = q.T.reshape(q.shape[1], 2, a.shape[2])
            self.tensors[c - 1] = np.tensordot(self.tensors[c - 1], r.T, axes=([2], [0]))
            self.center -= 1


def run_mps(
    circuit: Circuit,
    params: Sequence[float] | np.ndarray | None = None,
    max_bond: int | None = None,
    cutoff: float = 1e-14,
) -> MPS:
    """Run a unitary circuit of one- and two-qubit gates on an MPS."""
    m = MPS(circuit.n, max_bond, cutoff)
    for op in circuit.ops:
        if op.name == "measure" or op.condition is not None:
            raise ValueError("run_mps takes unitary circuits only")
        m.apply(op_matrix(op, params), op.qubits)
    return m
