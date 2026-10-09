"""Observables and expectation values (Part 4): labels, the diagonal fast path,
the imaginary-part check, Z correlations, and the transverse-field Ising model.
"""

import unittest
from unittest import mock

import numpy as np

from quantum_simulator import gates, observables
from quantum_simulator.circuit import Circuit, run
from quantum_simulator.observables import (
    PauliSum,
    pauli,
    pauli_string,
    tfim,
    tfim_periodic_ground_energy,
    z_correlation,
    z_expectation,
)
from quantum_simulator.state import State
from tests.reference import random_state


def gate_path(psi, label: str) -> complex:
    """<psi|P|psi> by applying the Paulis, i.e. without the diagonal fast path."""
    psi = np.asarray(psi).reshape((2,) * len(label))
    return complex(np.vdot(psi, observables._apply_pauli_string(psi, label)))


def product_state(angles) -> State:
    """RY(angles[q]) on each qubit q: a product state with <Z_q> = cos(angles[q])."""
    c = Circuit(len(angles))
    for q, a in enumerate(angles):
        c.ry(q, a)
    return run(c).state


def ghz(n: int) -> State:
    c = Circuit(n).h(0)
    for q in range(1, n):
        c.cx(0, q)
    return run(c).state


class TestPauliString(unittest.TestCase):
    def test_qubit_zero_is_rightmost(self):
        self.assertEqual(pauli_string(3, {0: "Z"}), "IIZ")
        self.assertEqual(pauli_string(3, {0: "Y", 2: "X"}), "XIY")
        self.assertEqual(pauli_string(4, {1: "z", 2: "z"}), "IZZI")
        self.assertEqual(pauli_string(2, {}), "II")

    def test_label_acts_on_named_qubit(self):
        # X on qubit 2 of |000> flips qubit 2 only.
        s = State.zero(3)
        self.assertAlmostEqual(pauli(pauli_string(3, {2: "Z"})).expectation(s), 1.0)
        flipped = run(Circuit(3).x(2)).state
        self.assertAlmostEqual(pauli(pauli_string(3, {2: "Z"})).expectation(flipped), -1.0)
        self.assertAlmostEqual(pauli(pauli_string(3, {0: "Z"})).expectation(flipped), 1.0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            pauli_string(2, {2: "Z"})
        with self.assertRaises(ValueError):
            pauli_string(2, {0: "Q"})


class TestDiagonalFastPath(unittest.TestCase):
    def test_matches_gate_path(self):
        rng = np.random.default_rng(0)
        labels = ["Z", "IZ", "ZI", "ZZ", "ZIZ", "IZZI", "ZZZZ", "IIII", "ZIIZZ"]
        for label in labels:
            vec = random_state(len(label), rng)
            with self.subTest(label=label):
                self.assertAlmostEqual(pauli(label).expectation(vec), gate_path(vec, label).real, places=12)

    def test_both_precisions(self):
        rng = np.random.default_rng(1)
        h = PauliSum([("ZZII", -1.0), ("IZZI", -0.5), ("IIZZ", 2.0), ("ZIIZ", 0.3)])
        vec = random_state(4, rng)
        exact = np.vdot(vec, h.to_matrix() @ vec).real
        for dtype, places in [(np.complex128, 12), (np.complex64, 5)]:
            with self.subTest(dtype=np.dtype(dtype).name):
                self.assertAlmostEqual(h.expectation(State(vec, dtype=dtype)), exact, places=places)

    def test_uses_no_gates(self):
        h = PauliSum([("ZZI", 1.0), ("IZZ", -0.4), ("III", 2.0)])
        vec = random_state(3, np.random.default_rng(2))
        with mock.patch.object(observables, "apply_gate", side_effect=AssertionError("gate applied")):
            value = h.expectation(vec)
        self.assertAlmostEqual(value, np.vdot(vec, h.to_matrix() @ vec).real, places=12)

    def test_mixed_sum_uses_both_paths(self):
        rng = np.random.default_rng(3)
        h = PauliSum([("ZZIZ", -1.0), ("XIYI", 0.7), ("IZZI", 0.2), ("YXZI", -0.3), ("IIII", 1.5)])
        for _ in range(5):
            vec = random_state(4, rng)
            self.assertAlmostEqual(h.expectation(vec), np.vdot(vec, h.to_matrix() @ vec).real, places=12)

    def test_z_expectation(self):
        rng = np.random.default_rng(4)
        vec = random_state(5, rng)
        for qubits in [[0], [4], [1, 3], [0, 2, 4], []]:
            with self.subTest(qubits=qubits):
                label = pauli_string(5, {q: "Z" for q in qubits})
                self.assertAlmostEqual(z_expectation(vec, qubits), gate_path(vec, label).real, places=12)
        # A repeated qubit cancels, since Z^2 = I.
        self.assertAlmostEqual(z_expectation(vec, [1, 3, 1]), z_expectation(vec, [3]), places=14)
        with self.assertRaises(ValueError):
            z_expectation(vec, [5])

    def test_signs_on_basis_states(self):
        # <Z_q> is +1 for a 0 bit and -1 for a 1 bit; products multiply.
        s = State.basis(0b0110, 4)  # q1 = q2 = 1
        self.assertEqual([z_expectation(s, [q]) for q in range(4)], [1.0, -1.0, -1.0, 1.0])
        self.assertEqual(z_expectation(s, [1, 2]), 1.0)
        self.assertEqual(z_expectation(s, [0, 1]), -1.0)


class TestImaginaryPartCheck(unittest.TestCase):
    def test_bug_upstream_raises(self):
        # Simulate a bug: the "X" matrix replaced by S, which is not Hermitian.
        # <+|S|+> = (1 + i)/2, so the "Hermitian" expectation comes out complex.
        with mock.patch.dict(observables.PAULIS, {"X": gates.S}):
            with self.assertRaises(RuntimeError):
                pauli("X").expectation(State.from_label("+"))

    def test_not_raised_for_valid_states(self):
        rng = np.random.default_rng(5)
        h = tfim(6, h=0.8) + pauli("XYZXYZ", 0.5)
        for dtype in (np.complex128, np.complex64):
            for _ in range(5):
                value = h.expectation(State(random_state(6, rng), dtype=dtype))
                self.assertIsInstance(value, float)

    def test_non_hermitian_returns_complex(self):
        value = pauli("Y", 1j).expectation(State.from_label("r"))  # i * <r|Y|r> = i
        self.assertAlmostEqual(value, 1j)


class TestZCorrelation(unittest.TestCase):
    def test_product_state_factorizes(self):
        angles = [0.3, 1.2, 2.0, 2.9]
        s = product_state(angles)
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                with self.subTest(i=i, j=j):
                    self.assertAlmostEqual(z_correlation(s, i, j), np.cos(angles[i]) * np.cos(angles[j]), places=12)
                    self.assertAlmostEqual(z_correlation(s, i, j, connected=True), 0.0, places=12)

    def test_ghz_is_correlated_at_every_distance(self):
        s = ghz(5)
        for j in range(1, 5):
            self.assertAlmostEqual(z_correlation(s, 0, j), 1.0, places=12)
            self.assertAlmostEqual(z_correlation(s, 0, j, connected=True), 1.0, places=12)
            self.assertAlmostEqual(z_expectation(s, [j]), 0.0, places=12)

    def test_same_site_is_one(self):
        s = State(random_state(3, np.random.default_rng(6)))
        self.assertAlmostEqual(z_correlation(s, 1, 1), 1.0, places=12)

    def test_ordered_vs_disordered_ground_states(self):
        # Deep in the ordered phase (h << J) correlations stay near 1 across the chain;
        # deep in the paramagnetic phase (h >> J) they decay to almost nothing.
        n = 8
        _, ordered = tfim(n, J=1.0, h=0.1).ground_state()
        _, disordered = tfim(n, J=1.0, h=5.0).ground_state()
        far = n - 1
        self.assertGreater(z_correlation(ordered, 0, far), 0.9)
        self.assertLess(abs(z_correlation(disordered, 0, far)), 1e-3)


class TestTfim(unittest.TestCase):
    def test_terms(self):
        open_chain = tfim(4, J=2.0, h=0.5)
        self.assertEqual(
            dict(open_chain.terms),
            {"IIZZ": -2.0, "IZZI": -2.0, "ZZII": -2.0, "IIIX": -0.5, "IIXI": -0.5, "IXII": -0.5, "XIII": -0.5},
        )
        ring = dict(tfim(4, J=2.0, h=0.5, periodic=True).terms)
        self.assertEqual(ring["ZIIZ"], -2.0)
        self.assertEqual(len(ring), 8)
        self.assertEqual(dict(tfim(2, periodic=True).terms)["ZZ"], -2.0)  # duplicate bond merges

    def test_closed_form_matches_exact_diagonalization(self):
        for n in [2, 4, 6, 8]:
            for J, h in [(1.0, 0.0), (1.0, 0.4), (1.0, 1.0), (0.5, 1.7), (2.0, 3.0)]:
                with self.subTest(n=n, J=J, h=h):
                    ed = tfim(n, J, h, periodic=True).eigh()[0][0]
                    self.assertAlmostEqual(tfim_periodic_ground_energy(n, J, h), ed, places=10)

    def test_closed_form_validation(self):
        for bad in [(3, 1.0, 1.0), (4, -1.0, 1.0), (4, 1.0, -1.0)]:
            with self.assertRaises(ValueError):
                tfim_periodic_ground_energy(*bad)

    def test_limits(self):
        # h = 0: classical ferromagnet, E0 = -J * bonds. J = 0: free spins in a field, E0 = -h n.
        self.assertAlmostEqual(tfim(5, J=1.3, h=0.0).ground_state()[0], -1.3 * 4)
        self.assertAlmostEqual(tfim(5, J=0.0, h=0.7).ground_state()[0], -0.7 * 5)


if __name__ == "__main__":
    unittest.main()
