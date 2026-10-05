"""Behavior of the standard gates: Pauli X/Y/Z, H, S, T, RX/RY/RZ, CNOT, CZ.

``test_gates.py`` checks matrices (unitarity, a few identities, Qiskit's CX
form) and ``test_qiskit_parity.py`` checks them entry by entry against Qiskit.
This file checks what each gate *does*: its action on basis states and
eigenstates, the algebra it satisfies, and its effect on the Bloch sphere.
Tests that act on states run in both precisions.
"""

import functools
import unittest

import numpy as np

from quantum_simulator import gates
from quantum_simulator.entanglement import entanglement_entropy
from quantum_simulator.linalg import dagger, kron
from quantum_simulator.observables import pauli
from quantum_simulator.state import State
from tests.reference import random_state

I, X, Y, Z, H, S, T = gates.I, gates.X, gates.Y, gates.Z, gates.H, gates.S, gates.T
SQ2 = 1 / np.sqrt(2)
PRECISIONS = {np.complex64: 1e-6, np.complex128: 1e-12}  # dtype -> atol
ANGLES = [-2.7, -np.pi / 3, 0.0, 0.4, np.pi / 2, np.pi, 5.0]


def both_precisions(test):
    """Run ``test(self, dtype, atol)`` once per precision, each in its own subTest."""

    @functools.wraps(test)
    def wrapper(self):
        for dtype, atol in PRECISIONS.items():
            with self.subTest(dtype=np.dtype(dtype).name):
                test(self, dtype, atol)

    return wrapper


def act(gate, qubits, start: State) -> np.ndarray:
    """Flat amplitudes of ``gate`` applied to a copy of ``start``."""
    return start.copy().apply(gate, qubits).vector


class GateTestCase(unittest.TestCase):
    def assertState(self, got, expected, atol):
        np.testing.assert_allclose(got, np.asarray(expected, dtype=complex), atol=atol)

    def assertMatrix(self, got, expected):
        np.testing.assert_allclose(got, expected, atol=1e-14)



class TestPauliGates(GateTestCase):
    @both_precisions
    def test_action_on_basis_states(self, dtype, atol):
        cases = [
            (X, [0, 1], [1, 0]),           # X|0> = |1>, X|1> = |0>
            (Y, [0, 1j], [-1j, 0]),        # Y|0> = i|1>, Y|1> = -i|0>
            (Z, [1, 0], [0, -1]),          # Z|0> = |0>, Z|1> = -|1>
        ]
        for gate, from0, from1 in cases:
            self.assertState(act(gate, [0], State.from_label("0", dtype)), from0, atol)
            self.assertState(act(gate, [0], State.from_label("1", dtype)), from1, atol)

    @both_precisions
    def test_eigenstates(self, dtype, atol):
        # Each Pauli has eigenvalues +1 and -1 with the expected eigenvectors.
        cases = [(X, "+", "-"), (Y, "r", "l"), (Z, "0", "1")]
        for gate, plus, minus in cases:
            up, down = State.from_label(plus, dtype), State.from_label(minus, dtype)
            self.assertState(act(gate, [0], up), up.vector, atol)
            self.assertState(act(gate, [0], down), -down.vector, atol)

    def test_involutory_hermitian_traceless(self):
        for p in (X, Y, Z):
            self.assertMatrix(p @ p, I)
            self.assertMatrix(p, dagger(p))
            self.assertAlmostEqual(np.trace(p), 0)

    def test_algebra(self):
        # XY = iZ and cyclic; distinct Paulis anticommute.
        self.assertMatrix(X @ Y, 1j * Z)
        self.assertMatrix(Y @ Z, 1j * X)
        self.assertMatrix(Z @ X, 1j * Y)
        for a, b in [(X, Y), (Y, Z), (Z, X)]:
            self.assertMatrix(a @ b, -(b @ a))
        self.assertMatrix(X @ Y @ Z, 1j * I)

    @both_precisions
    def test_acts_only_on_target_qubit(self, dtype, atol):
        # X on qubit 1 of |000> gives |010> (flat index 2), nothing else changes.
        for q in range(3):
            expected = np.zeros(8)
            expected[1 << q] = 1
            self.assertState(act(X, [q], State.zero(3, dtype)), expected, atol)


class TestHadamard(GateTestCase):
    @both_precisions
    def test_action(self, dtype, atol):
        self.assertState(act(H, [0], State.from_label("0", dtype)), [SQ2, SQ2], atol)
        self.assertState(act(H, [0], State.from_label("1", dtype)), [SQ2, -SQ2], atol)
        self.assertState(act(H, [0], State.from_label("+", dtype)), [1, 0], atol)
        self.assertState(act(H, [0], State.from_label("-", dtype)), [0, 1], atol)

    def test_self_inverse_and_hermitian(self):
        self.assertMatrix(H @ H, I)
        self.assertMatrix(H, dagger(H))

    def test_swaps_x_and_z(self):
        self.assertMatrix(H @ X @ H, Z)
        self.assertMatrix(H @ Z @ H, X)
        self.assertMatrix(H @ Y @ H, -Y)

    @both_precisions
    def test_uniform_superposition(self, dtype, atol):
        n = 5
        s = State.zero(n, dtype)
        for q in range(n):
            s.apply(H, [q])
        self.assertState(s.vector, np.full(2**n, 2 ** (-n / 2)), atol)


class TestPhaseGates(GateTestCase):
    @both_precisions
    def test_action(self, dtype, atol):
        self.assertState(act(S, [0], State.from_label("0", dtype)), [1, 0], atol)
        self.assertState(act(S, [0], State.from_label("1", dtype)), [0, 1j], atol)
        self.assertState(act(T, [0], State.from_label("1", dtype)), [0, np.exp(1j * np.pi / 4)], atol)
        # S rotates |+> a quarter turn about Z, onto |r> = (|0> + i|1>)/sqrt(2).
        self.assertState(act(S, [0], State.from_label("+", dtype)), State.from_label("r").vector, atol)

    def test_powers(self):
        mp = np.linalg.matrix_power
        self.assertMatrix(mp(S, 2), Z)
        self.assertMatrix(mp(S, 4), I)
        self.assertMatrix(mp(T, 2), S)
        self.assertMatrix(mp(T, 4), Z)
        self.assertMatrix(mp(T, 8), I)

    def test_inverses(self):
        self.assertMatrix(dagger(S), gates.SDG)
        self.assertMatrix(dagger(T), gates.TDG)
        self.assertMatrix(S @ gates.SDG, I)
        self.assertMatrix(T @ gates.TDG, I)

    def test_diagonal_so_commute_with_z(self):
        for g in (S, T):
            self.assertMatrix(g @ Z, Z @ g)
        self.assertFalse(np.allclose(S @ X, X @ S))

    def test_not_hermitian(self):
        # Unlike the Paulis and H, S and T are not their own inverses.
        self.assertFalse(np.allclose(S, dagger(S)))
        self.assertFalse(np.allclose(T, dagger(T)))


class TestRotations(GateTestCase):
    AXES = [(gates.rx, X), (gates.ry, Y), (gates.rz, Z)]

    def test_matches_exponential_of_generator(self):
        # R_P(theta) = exp(-i theta P / 2) = cos(theta/2) I - i sin(theta/2) P.
        for rot, p in self.AXES:
            for theta in ANGLES:
                with self.subTest(gate=rot.__name__, theta=theta):
                    w, v = np.linalg.eigh(p)
                    expected = v @ np.diag(np.exp(-0.5j * theta * w)) @ v.conj().T
                    self.assertMatrix(rot(theta), expected)

    def test_identity_spinor_sign_and_period(self):
        for rot, _ in self.AXES:
            with self.subTest(gate=rot.__name__):
                self.assertMatrix(rot(0), I)
                self.assertMatrix(rot(2 * np.pi), -I)  # a full turn is -1 on spinors
                self.assertMatrix(rot(4 * np.pi), I)
                self.assertMatrix(rot(np.pi), -1j * _pauli_of(rot))

    def test_angles_add_and_negate(self):
        for rot, _ in self.AXES:
            for a, b in [(0.3, 1.1), (-2.0, 0.7), (np.pi, np.pi / 2)]:
                with self.subTest(gate=rot.__name__, a=a, b=b):
                    self.assertMatrix(rot(a) @ rot(b), rot(a + b))
                    self.assertMatrix(rot(-a), dagger(rot(a)))

    def test_rz_equals_phase_gates_up_to_global_phase(self):
        for theta in ANGLES:
            self.assertMatrix(gates.rz(theta), np.exp(-0.5j * theta) * gates.phase(theta))
        self.assertMatrix(gates.rz(np.pi / 2), np.exp(-0.25j * np.pi) * S)
        self.assertMatrix(gates.rz(np.pi / 4), np.exp(-0.125j * np.pi) * T)
        self.assertMatrix(gates.rz(np.pi), -1j * Z)

    @both_precisions
    def test_quarter_turns_on_zero(self, dtype, atol):
        zero = State.from_label("0", dtype)
        self.assertState(act(gates.ry(np.pi / 2), [0], zero), State.from_label("+").vector, atol)
        self.assertState(act(gates.rx(np.pi / 2), [0], zero), State.from_label("l").vector, atol)
        self.assertState(act(gates.rx(np.pi), [0], zero), [0, -1j], atol)

    @both_precisions
    def test_bloch_vector_rotates_by_theta(self, dtype, atol):
        # Starting points and the Bloch vector (<X>, <Y>, <Z>) expected after rotating by theta.
        cases = [
            (gates.rx, "0", lambda t: (0, -np.sin(t), np.cos(t))),
            (gates.ry, "0", lambda t: (np.sin(t), 0, np.cos(t))),
            (gates.rz, "+", lambda t: (np.cos(t), np.sin(t), 0)),
        ]
        observables = [pauli("X"), pauli("Y"), pauli("Z")]
        for rot, label, bloch in cases:
            for theta in ANGLES:
                s = State.from_label(label, dtype).apply(rot(theta), [0])
                got = [o.expectation(s) for o in observables]
                np.testing.assert_allclose(got, bloch(theta), atol=10 * atol)

    @both_precisions
    def test_measurement_probability(self, dtype, atol):
        # RX and RY both take |0> to P(1) = sin^2(theta/2); RZ never changes populations.
        for theta in ANGLES:
            for rot in (gates.rx, gates.ry):
                p = State.zero(1, dtype).apply(rot(theta), [0]).probabilities()
                np.testing.assert_allclose(p, [np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2], atol=10 * atol)
            p = State.from_label("+", dtype).apply(gates.rz(theta), [0]).probabilities()
            np.testing.assert_allclose(p, [0.5, 0.5], atol=10 * atol)


def _pauli_of(rot):
    return {gates.rx: X, gates.ry: Y, gates.rz: Z}[rot]


class TestCNOT(GateTestCase):
    def test_alias(self):
        self.assertIs(gates.CNOT, gates.CX)

    @both_precisions
    def test_truth_table(self, dtype, atol):
        # Flat index = q0 + 2*q1. With [control, target] = [0, 1], flip q1 iff q0 = 1.
        table_01 = {0b00: 0b00, 0b01: 0b11, 0b10: 0b10, 0b11: 0b01}
        # With [control, target] = [1, 0], flip q0 iff q1 = 1.
        table_10 = {0b00: 0b00, 0b01: 0b01, 0b10: 0b11, 0b11: 0b10}
        for qubits, table in [([0, 1], table_01), ([1, 0], table_10)]:
            for k_in, k_out in table.items():
                expected = np.zeros(4)
                expected[k_out] = 1
                self.assertState(act(gates.CNOT, qubits, State.basis(k_in, 2, dtype)), expected, atol)

    @both_precisions
    def test_truth_table_in_larger_register(self, dtype, atol):
        # Control qubit 3, target qubit 0, idle qubits 1 and 2, all 16 basis states.
        for k in range(16):
            expected = np.zeros(16)
            expected[k ^ 1 if k & 0b1000 else k] = 1
            self.assertState(act(gates.CNOT, [3, 0], State.basis(k, 4, dtype)), expected, atol)

    def test_self_inverse_and_hermitian(self):
        self.assertMatrix(gates.CNOT @ gates.CNOT, np.eye(4))
        self.assertMatrix(gates.CNOT, dagger(gates.CNOT))

    @both_precisions
    def test_makes_all_four_bell_states(self, dtype, atol):
        # H on qubit 0 then CNOT(0 -> 1), from |q1 q0> = |00>, |01>, |10>, |11>.
        expected = {
            0b00: [SQ2, 0, 0, SQ2],    # |Phi+>
            0b01: [SQ2, 0, 0, -SQ2],   # |Phi->
            0b10: [0, SQ2, SQ2, 0],    # |Psi+>
            0b11: [0, -SQ2, SQ2, 0],   # |Psi->
        }
        for k, amps in expected.items():
            s = State.basis(k, 2, dtype).apply(H, [0]).apply(gates.CNOT, [0, 1])
            self.assertState(s.vector, amps, atol)
            self.assertAlmostEqual(entanglement_entropy(s, [0]), 1.0, places=5)

    @both_precisions
    def test_phase_kickback(self, dtype, atol):
        # Target in |->: CNOT applies Z to the control, so |+>_control becomes |->_control.
        # Labels put qubit 1 (the target) first.
        s = State.from_label("-+", dtype).apply(gates.CNOT, [0, 1])
        self.assertState(s.vector, State.from_label("--").vector, atol)

    def test_conjugating_by_hadamards_reverses_direction(self):
        # (H⊗H) CNOT(0->1) (H⊗H) = CNOT(1->0).
        hh = kron(H, H)
        reversed_cnot = np.eye(4)[:, [0, 1, 3, 2]]  # flips q0 iff q1 = 1
        self.assertMatrix(hh @ gates.CNOT @ hh, reversed_cnot)


class TestCZ(GateTestCase):
    @both_precisions
    def test_truth_table(self, dtype, atol):
        # Phase -1 on |11> only; populations never change.
        for k in range(4):
            expected = np.zeros(4)
            expected[k] = -1 if k == 0b11 else 1
            self.assertState(act(gates.CZ, [0, 1], State.basis(k, 2, dtype)), expected, atol)

    @both_precisions
    def test_symmetric_in_its_qubits(self, dtype, atol):
        rng = np.random.default_rng(0)
        s = State(random_state(4, rng), dtype=dtype)
        self.assertState(act(gates.CZ, [0, 2], s), act(gates.CZ, [2, 0], s), atol)

    def test_self_inverse_hermitian_diagonal(self):
        self.assertMatrix(gates.CZ @ gates.CZ, np.eye(4))
        self.assertMatrix(gates.CZ, dagger(gates.CZ))
        self.assertMatrix(gates.CZ, np.diag([1, 1, 1, -1]))

    @both_precisions
    def test_equals_cnot_between_hadamards_on_target(self, dtype, atol):
        rng = np.random.default_rng(1)
        s = State(random_state(3, rng), dtype=dtype)
        via_cnot = s.copy().apply(H, [2]).apply(gates.CNOT, [0, 2]).apply(H, [2]).vector
        self.assertState(act(gates.CZ, [0, 2], s), via_cnot, 10 * atol)

    @both_precisions
    def test_graph_state(self, dtype, atol):
        # CZ on |++> gives the two-qubit graph state (|00> + |01> + |10> - |11>)/2: maximally entangled.
        s = State.from_label("++", dtype).apply(gates.CZ, [0, 1])
        self.assertState(s.vector, [0.5, 0.5, 0.5, -0.5], atol)
        self.assertAlmostEqual(entanglement_entropy(s, [0]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
