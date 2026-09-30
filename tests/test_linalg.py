import unittest

import numpy as np

from quantum_simulator.linalg import (
    dagger,
    equal_up_to_global_phase,
    fidelity,
    inner,
    is_hermitian,
    is_unitary,
    kron,
)
from tests.reference import random_state, random_unitary


class TestLinalg(unittest.TestCase):
    def test_inner_conjugates_first_argument(self):
        zero = np.array([1, 0], dtype=complex)
        self.assertEqual(inner(1j * zero, zero), -1j)
        self.assertEqual(inner(zero, 1j * zero), 1j)

    def test_inner_accepts_tensors(self):
        rng = np.random.default_rng(0)
        a, b = random_state(3, rng), random_state(3, rng)
        self.assertAlmostEqual(inner(a.reshape(2, 2, 2), b.reshape(2, 2, 2)), np.conj(a) @ b)
        with self.assertRaises(ValueError):
            inner(a, b[:4])

    def test_fidelity(self):
        rng = np.random.default_rng(1)
        a, b = random_state(2, rng), random_state(2, rng)
        self.assertAlmostEqual(fidelity(a, b), fidelity(b, a))
        self.assertAlmostEqual(fidelity(a, np.exp(0.7j) * a), 1.0)
        self.assertLessEqual(fidelity(a, b), 1.0)

    def test_unitary_and_hermitian(self):
        u = random_unitary(4, np.random.default_rng(2))
        self.assertTrue(is_unitary(u))
        self.assertFalse(is_unitary(2 * u))
        self.assertFalse(is_unitary(np.ones((2, 3))))
        h = u + dagger(u)
        self.assertTrue(is_hermitian(h))
        self.assertFalse(is_hermitian(u))
        np.testing.assert_allclose(dagger(u) @ u, np.eye(4), atol=1e-12)

    def test_kron_order(self):
        # First factor is the most significant (highest) qubit.
        x = np.array([[0, 1], [1, 0]])
        np.testing.assert_array_equal(kron(x, np.eye(2)) @ [1, 0, 0, 0], [0, 0, 1, 0])
        np.testing.assert_array_equal(kron(x, np.eye(2), np.eye(2)).shape, (8, 8))

    def test_equal_up_to_global_phase(self):
        a = random_state(2, np.random.default_rng(3))
        self.assertTrue(equal_up_to_global_phase(a, np.exp(1.3j) * a))
        self.assertFalse(equal_up_to_global_phase(a, a[::-1]))
        self.assertFalse(equal_up_to_global_phase(a, 2 * a))


if __name__ == "__main__":
    unittest.main()
