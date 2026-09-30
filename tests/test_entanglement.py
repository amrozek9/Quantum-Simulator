import unittest

import numpy as np

from quantum_simulator import gates
from quantum_simulator.entanglement import (
    bipartition_matrix,
    entanglement_entropy,
    purity,
    reduced_density_matrix,
    schmidt_decomposition,
    schmidt_rank,
)
from quantum_simulator.state import State
from tests.reference import random_state


def ghz(n: int) -> State:
    s = State.zero(n).apply(gates.H, [0])
    for q in range(1, n):
        s.apply(gates.CX, [0, q])
    return s


def brute_force_rdm(vec: np.ndarray, keep: list[int], n: int) -> np.ndarray:
    """Partial trace by explicit sums over bit patterns (keep sorted ascending)."""
    k = len(keep)
    rho = np.zeros((2**k, 2**k), dtype=complex)
    sub = lambda idx: sum(((idx >> q) & 1) << j for j, q in enumerate(keep))
    rest = lambda idx: idx & ~sum(1 << q for q in keep)
    for i in range(2**n):
        for j in range(2**n):
            if rest(i) == rest(j):
                rho[sub(i), sub(j)] += vec[i] * np.conj(vec[j])
    return rho


class TestSchmidt(unittest.TestCase):
    def test_reconstruction(self):
        rng = np.random.default_rng(0)
        vec = random_state(5, rng)
        for qubits in [[0], [4], [1, 3], [0, 2, 4]]:
            with self.subTest(qubits=qubits):
                s, u, vh = schmidt_decomposition(vec, qubits)
                m = bipartition_matrix(vec, qubits)
                np.testing.assert_allclose((u * s) @ vh, m, atol=1e-13)
                self.assertTrue(np.all(np.diff(s) <= 1e-15))
                self.assertAlmostEqual(np.sum(s**2), 1.0)

    def test_product_state_rank_one(self):
        s = State.from_label("+0-r")
        for qubits in [[0], [1, 2], [0, 3]]:
            self.assertEqual(schmidt_rank(s, qubits), 1)
            self.assertAlmostEqual(entanglement_entropy(s, qubits), 0.0)
            self.assertAlmostEqual(purity(s, qubits), 1.0)

    def test_bell_and_ghz(self):
        bell = ghz(2)
        self.assertEqual(schmidt_rank(bell, [0]), 2)
        self.assertAlmostEqual(entanglement_entropy(bell, [0]), 1.0)
        np.testing.assert_allclose(reduced_density_matrix(bell, [1]), np.eye(2) / 2, atol=1e-15)
        g = ghz(4)
        for qubits in [[0], [3], [1, 2], [0, 1, 3]]:
            self.assertAlmostEqual(entanglement_entropy(g, qubits), 1.0)

    def test_entropy_symmetric_across_cut(self):
        vec = random_state(5, np.random.default_rng(1))
        self.assertAlmostEqual(entanglement_entropy(vec, [0, 3]), entanglement_entropy(vec, [1, 2, 4]))

    def test_renyi(self):
        bell = ghz(2)
        for alpha in [0.5, 2, 3, np.inf]:
            self.assertAlmostEqual(entanglement_entropy(bell, [0], alpha=alpha), 1.0)
        vec = random_state(4, np.random.default_rng(2))
        s2 = entanglement_entropy(vec, [0, 1], alpha=2)
        self.assertAlmostEqual(s2, -np.log2(purity(vec, [0, 1])))
        self.assertLessEqual(s2, entanglement_entropy(vec, [0, 1]) + 1e-12)
        self.assertAlmostEqual(entanglement_entropy(bell, [0], base=np.e), np.log(2))

    def test_rdm_matches_brute_force(self):
        n = 4
        vec = random_state(n, np.random.default_rng(3))
        for keep in [[0], [2], [0, 3], [1, 2, 3]]:
            with self.subTest(keep=keep):
                np.testing.assert_allclose(
                    reduced_density_matrix(vec, keep), brute_force_rdm(vec, keep, n), atol=1e-14
                )

    def test_validation(self):
        with self.assertRaises(ValueError):
            schmidt_rank(State.zero(2), [2])
        with self.assertRaises(ValueError):
            schmidt_rank(State.zero(2), [0, 0])


if __name__ == "__main__":
    unittest.main()
