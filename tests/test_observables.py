import unittest

import numpy as np

from quantum_simulator import gates
from quantum_simulator.entanglement import entanglement_entropy
from quantum_simulator.linalg import is_hermitian, kron
from quantum_simulator.observables import PauliSum, pauli
from quantum_simulator.state import State
from tests.reference import random_state


def tfim(n: int, g: float) -> PauliSum:
    """Open-chain transverse-field Ising model: -sum Z_i Z_{i+1} - g sum X_i."""
    terms = []
    for i in range(n - 1):
        label = ["I"] * n
        label[i] = label[i + 1] = "Z"
        terms.append(("".join(label), -1.0))
    for i in range(n):
        label = ["I"] * n
        label[i] = "X"
        terms.append(("".join(label), -g))
    return PauliSum(terms)


class TestPauliSum(unittest.TestCase):
    def test_label_order(self):
        s = State.from_label("01")  # qubit 1 = 0, qubit 0 = 1
        self.assertAlmostEqual(pauli("ZI").expectation(s), 1.0)
        self.assertAlmostEqual(pauli("IZ").expectation(s), -1.0)

    def test_merge_and_arithmetic(self):
        h = PauliSum([("ZZ", 1.0), ("XI", 0.5), ("ZZ", 2.0)])
        self.assertEqual(dict(h.terms), {"ZZ": 3.0, "XI": 0.5})
        diff = h - pauli("ZZ", 3.0)
        self.assertEqual(dict(diff.terms), {"XI": 0.5})
        np.testing.assert_allclose((2 * h).to_matrix(), 2 * h.to_matrix())
        with self.assertRaises(ValueError):
            PauliSum([("ZZ", 1.0), ("Z", 1.0)])
        with self.assertRaises(ValueError):
            pauli("ZA")

    def test_to_matrix_matches_kron(self):
        h = PauliSum([("XZ", 0.3), ("YI", -1.2)])
        expected = 0.3 * kron(gates.X, gates.Z) - 1.2 * kron(gates.Y, gates.I)
        np.testing.assert_allclose(h.to_matrix(), expected)
        self.assertTrue(is_hermitian(h.to_matrix()))

    def test_matrix_free_apply_matches_dense(self):
        rng = np.random.default_rng(0)
        h = tfim(5, 0.7) + pauli("YIXZI", 0.4)
        vec = random_state(5, rng)
        np.testing.assert_allclose(h.apply(vec).reshape(-1), h.to_matrix() @ vec, atol=1e-13)
        self.assertAlmostEqual(h.expectation(vec), np.vdot(vec, h.to_matrix() @ vec).real)

    def test_expectation_real_for_hermitian(self):
        vec = random_state(3, np.random.default_rng(1))
        self.assertIsInstance(tfim(3, 1.0).expectation(vec), float)
        self.assertIsInstance(pauli("XYZ", 1j).expectation(vec), complex)

    def test_dense_size_guard(self):
        with self.assertRaises(ValueError):
            pauli("Z" * 13).to_matrix()


class TestExactDiagonalization(unittest.TestCase):
    def test_single_qubit(self):
        energy, gs = pauli("Z").ground_state()
        self.assertAlmostEqual(energy, -1.0)
        self.assertAlmostEqual(gs.fidelity(State.from_label("1")), 1.0)

    def test_heisenberg_singlet(self):
        # XX + YY + ZZ: singlet at -3, triplet at +1; the singlet is maximally entangled.
        h = PauliSum([("XX", 1), ("YY", 1), ("ZZ", 1)])
        energies, _ = h.eigh()
        np.testing.assert_allclose(energies, [-3, 1, 1, 1], atol=1e-12)
        _, gs = h.ground_state()
        singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
        self.assertAlmostEqual(gs.fidelity(singlet), 1.0)
        self.assertAlmostEqual(entanglement_entropy(gs, [0]), 1.0)

    def test_tfim_two_sites(self):
        # Two-site open TFIM ground energy is -sqrt(1 + 4 g^2).
        for g in [0.0, 0.5, 1.0, 2.0]:
            with self.subTest(g=g):
                energy, gs = tfim(2, g).ground_state()
                self.assertAlmostEqual(energy, -np.sqrt(1 + 4 * g**2))
                self.assertAlmostEqual(tfim(2, g).expectation(gs), energy)
                self.assertAlmostEqual(tfim(2, g).variance(gs), 0.0, places=10)

    def test_eigenvectors_satisfy_eigen_equation(self):
        h = tfim(4, 0.9)
        energies, vecs = h.eigh()
        for k in [0, 5, 15]:
            np.testing.assert_allclose(h.apply(vecs[:, k]).reshape(-1), energies[k] * vecs[:, k], atol=1e-12)

    def test_variance_of_non_eigenstate(self):
        # |0> under X: <X> = 0, <X^2> = 1.
        self.assertAlmostEqual(pauli("X").variance(State.zero(1)), 1.0)

    def test_non_hermitian_rejected(self):
        with self.assertRaises(ValueError):
            pauli("X", 1j).ground_state()


if __name__ == "__main__":
    unittest.main()
