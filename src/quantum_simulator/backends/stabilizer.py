"""Stabilizer states: the Aaronson-Gottesman tableau, O(n^2) memory, Clifford gates only.

A stabilizer state on n qubits is the unique state fixed (with eigenvalue
+1) by n commuting Pauli operators, its stabilizers. The tableau stores
those n stabilizers and n "destabilizers" as rows of bits: ``x[i, q]`` and
``z[i, q]`` say whether row i has X, Z or Y (both) on qubit q, and ``r[i]``
is its sign. Clifford gates map Pauli operators to Pauli operators, so each
gate is a few bitwise updates of two columns: O(n) per gate, and
measurement O(n^2), against O(2^n) for a state vector. This is the
Gottesman-Knill theorem made concrete (Aaronson and Gottesman, Phys. Rev. A
70, 052328 (2004)).

Supported gates: I, X, Y, Z, H, S, S-dagger, CX, CZ, SWAP, measurement and
classically conditioned gates. Anything else (T, rotations) leaves the
stabilizer formalism and is rejected.
"""

from __future__ import annotations

import numpy as np

from ..circuit import Circuit
from ..observables import PauliSum
from ..state import State, _as_rng

CLIFFORD_GATES = frozenset({"i", "x", "y", "z", "h", "s", "sdg", "cx", "cz", "swap"})


class StabilizerState:
    """n qubits in ``|0...0>``: destabilizers X_q, stabilizers Z_q."""

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("need at least one qubit")
        self.n = n
        # Rows 0..n-1 are destabilizers, n..2n-1 stabilizers, row 2n is scratch space.
        self.x = np.zeros((2 * n + 1, n), dtype=bool)
        self.z = np.zeros((2 * n + 1, n), dtype=bool)
        self.r = np.zeros(2 * n + 1, dtype=bool)
        self.x[np.arange(n), np.arange(n)] = True
        self.z[n + np.arange(n), np.arange(n)] = True

    # -- gates (each is a conjugation rule for Pauli operators) --------------------------

    def h(self, q: int) -> StabilizerState:
        self.r ^= self.x[:, q] & self.z[:, q]
        self.x[:, q], self.z[:, q] = self.z[:, q].copy(), self.x[:, q].copy()
        return self

    def s(self, q: int) -> StabilizerState:
        self.r ^= self.x[:, q] & self.z[:, q]
        self.z[:, q] ^= self.x[:, q]
        return self

    def sdg(self, q: int) -> StabilizerState:
        return self.s(q).s(q).s(q)

    def x_gate(self, q: int) -> StabilizerState:  # X flips the sign of Z and Y components
        self.r ^= self.z[:, q]
        return self

    def z_gate(self, q: int) -> StabilizerState:  # Z flips the sign of X and Y components
        self.r ^= self.x[:, q]
        return self

    def y_gate(self, q: int) -> StabilizerState:  # Y flips the sign of X and Z, not of Y
        self.r ^= self.x[:, q] ^ self.z[:, q]
        return self

    def cx(self, control: int, target: int) -> StabilizerState:
        a, b = control, target
        self.r ^= self.x[:, a] & self.z[:, b] & ~(self.x[:, b] ^ self.z[:, a])
        self.x[:, b] ^= self.x[:, a]
        self.z[:, a] ^= self.z[:, b]
        return self

    def cz(self, a: int, b: int) -> StabilizerState:
        return self.h(b).cx(a, b).h(b)

    def swap(self, a: int, b: int) -> StabilizerState:
        return self.cx(a, b).cx(b, a).cx(a, b)

    # -- measurement -----------------------------------------------------------------

    def measure(self, q: int, rng: np.random.Generator | int) -> int:
        """Measure qubit q in the Z basis, collapsing the state."""
        n = self.n
        anticommuting = np.flatnonzero(self.x[n:2 * n, q])
        if anticommuting.size:  # some stabilizer anticommutes with Z_q: the outcome is random
            p = n + int(anticommuting[0])
            for i in np.flatnonzero(self.x[: 2 * n, q]):
                if i != p:
                    self._rowsum(i, p)
            self.x[p - n], self.z[p - n], self.r[p - n] = self.x[p], self.z[p], self.r[p]
            self.x[p], self.z[p] = False, False
            self.z[p, q] = True
            outcome = int(_as_rng(rng).random() < 0.5)
            self.r[p] = bool(outcome)
            return outcome
        # Z_q commutes with every stabilizer: the outcome is fixed; build it in the scratch row.
        s = 2 * n
        self.x[s], self.z[s], self.r[s] = False, False, False
        for i in np.flatnonzero(self.x[:n, q]):
            self._rowsum(s, i + n)
        return int(self.r[s])

    def _rowsum(self, h: int, i: int) -> None:
        """Replace row h by the product of rows i and h, tracking the sign exactly (mod 4 phases)."""
        x1, z1, x2, z2 = self.x[i].astype(int), self.z[i].astype(int), self.x[h].astype(int), self.z[h].astype(int)
        # The power of i picked up by multiplying single-qubit Paulis (x1,z1)*(x2,z2).
        g = np.where(x1 & z1, z2 - x2, np.where(x1 & ~z1 & 1, z2 * (2 * x2 - 1), np.where(~x1 & z1 & 1, x2 * (1 - 2 * z2), 0)))
        total = (2 * int(self.r[h]) + 2 * int(self.r[i]) + int(g.sum())) % 4
        self.r[h] = total == 2
        self.x[h] ^= self.x[i]
        self.z[h] ^= self.z[i]

    # -- readout ---------------------------------------------------------------------

    def stabilizers(self) -> list[tuple[int, str]]:
        """The n stabilizer generators as ``(sign, label)``, labels with qubit 0 rightmost."""
        out = []
        for i in range(self.n, 2 * self.n):
            letters = ["I"] * self.n
            for q in range(self.n):
                letters[self.n - 1 - q] = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}[
                    (int(self.x[i, q]), int(self.z[i, q]))]
            out.append((-1 if self.r[i] else 1, "".join(letters)))
        return out

    def to_state(self) -> State:
        """The state vector, up to global phase, by projecting onto every stabilizer (2^n: small n only).

        ``prod_i (I + S_i) / 2`` is the projector onto the stabilizer state;
        applying it to a vector with nonzero overlap gives the state.
        """
        if self.n > 14:
            raise ValueError("to_state builds a 2^n vector; use it only for small n")
        rng = np.random.default_rng(0)
        vec = rng.normal(size=2**self.n) + 1j * rng.normal(size=2**self.n)
        for sign, label in self.stabilizers():
            g = PauliSum([(label, float(sign))])
            vec = (vec + g.apply(vec).reshape(-1)) / 2
        return State(vec / np.linalg.norm(vec))


def run_stabilizer(circuit: Circuit, rng: np.random.Generator | int | None = None) -> tuple[StabilizerState, tuple[int, ...]]:
    """Run a Clifford circuit (with measurements and conditions) on a stabilizer tableau."""
    rng = _as_rng(rng) if any(op.name == "measure" for op in circuit.ops) else None
    st = StabilizerState(circuit.n)
    clbits = [0] * circuit.n_clbits
    names = {"x": "x_gate", "y": "y_gate", "z": "z_gate"}
    for op in circuit.ops:
        if op.condition is not None and clbits[op.condition[0]] != op.condition[1]:
            continue
        if op.name == "measure":
            clbits[op.clbit] = st.measure(op.qubits[0], rng)
        elif op.name == "i":
            continue
        elif op.name in CLIFFORD_GATES:
            getattr(st, names.get(op.name, op.name))(*op.qubits)
        else:
            raise ValueError(f"{op.name} is not a Clifford gate; the stabilizer backend cannot simulate it")
    return st, tuple(clbits)
