import unittest

import numpy as np

from quantum_simulator.state import (
    basis_state,
    from_vector,
    is_normalized,
    num_qubits,
    qubit_axis,
    to_vector,
    zero_state,
)


class TestIndexConvention(unittest.TestCase):
    def test_qubit_axis(self):
        self.assertEqual([qubit_axis(t, 3) for t in range(3)], [2, 1, 0])
        with self.assertRaises(ValueError):
            qubit_axis(3, 3)

    def test_qubit_t_is_bit_t_of_flat_index(self):
        # For every basis state, the bit read off qubit t's axis must equal
        # bit t of the flat index (Qiskit little-endian).
        n = 4
        for k in range(2**n):
            psi = basis_state(k, n)
            (pos,) = np.argwhere(psi)
            for t in range(n):
                self.assertEqual(pos[qubit_axis(t, n)], (k >> t) & 1)

    def test_tensor_index_reads_as_bitstring(self):
        # psi[b2, b1, b0] is |b2 b1 b0>; |011> has qubits 0 and 1 set -> index 3.
        psi = basis_state(0b011, 3)
        self.assertEqual(psi[0, 1, 1], 1)

    def test_single_qubit_x_flips_that_bit(self):
        # Apply X on qubit t by flipping its axis; flat index must change by 2**t.
        n = 3
        for t in range(n):
            psi = np.flip(zero_state(n), axis=qubit_axis(t, n))
            self.assertEqual(np.argmax(np.abs(to_vector(psi))), 1 << t)


class TestConstruction(unittest.TestCase):
    def test_zero_state(self):
        psi = zero_state(3)
        self.assertEqual(psi.shape, (2, 2, 2))
        self.assertEqual(psi.dtype, np.complex128)
        self.assertEqual(psi[0, 0, 0], 1)
        self.assertTrue(is_normalized(psi))

    def test_zero_qubits(self):
        psi = zero_state(0)
        self.assertEqual(psi.shape, ())
        self.assertEqual(num_qubits(psi), 0)

    def test_basis_state_range(self):
        with self.assertRaises(ValueError):
            basis_state(8, 3)

    def test_roundtrip_is_a_view(self):
        rng = np.random.default_rng(0)
        vec = rng.normal(size=16) + 1j * rng.normal(size=16)
        vec /= np.linalg.norm(vec)
        psi = from_vector(vec)
        self.assertEqual(psi.shape, (2,) * 4)
        flat = to_vector(psi)
        np.testing.assert_array_equal(flat, vec)
        self.assertTrue(np.shares_memory(flat, psi))

    def test_from_vector_validation(self):
        with self.assertRaises(ValueError):
            from_vector(np.ones(3) / np.sqrt(3))
        with self.assertRaises(ValueError):
            from_vector(np.ones(4))
        self.assertEqual(from_vector(np.ones(4), check_norm=False).shape, (2, 2))

    def test_num_qubits_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            num_qubits(np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
