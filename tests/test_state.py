import unittest

import numpy as np

from quantum_simulator import gates
from quantum_simulator.state import (
    State,
    apply_gate,
    basis_state,
    from_vector,
    is_normalized,
    num_qubits,
    qubit_axis,
    to_vector,
    zero_state,
)
from tests.reference import dense_operator, random_state, random_unitary


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


class TestApplyGate(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_matches_dense_reference(self):
        # Covers adjacent, non-adjacent and reversed qubit orders, 1-3 qubit gates.
        n = 5
        cases = [[0], [3], [4], [0, 1], [1, 0], [4, 1], [0, 4], [2, 0, 3], [4, 3, 2]]
        for qubits in cases:
            with self.subTest(qubits=qubits):
                u = random_unitary(2 ** len(qubits), self.rng)
                vec = random_state(n, self.rng)
                got = to_vector(apply_gate(from_vector(vec), u, qubits))
                np.testing.assert_allclose(got, dense_operator(u, qubits, n) @ vec, atol=1e-12)

    def test_preserves_norm(self):
        psi = from_vector(random_state(4, self.rng))
        out = apply_gate(psi, random_unitary(4, self.rng), [2, 0])
        self.assertAlmostEqual(np.linalg.norm(out), 1.0, places=12)

    def test_validation(self):
        psi = zero_state(2)
        with self.assertRaises(ValueError):
            apply_gate(psi, gates.CX, [0])
        with self.assertRaises(ValueError):
            apply_gate(psi, gates.CX, [1, 1])
        with self.assertRaises(ValueError):
            apply_gate(psi, gates.X, [2])


class TestStateClass(unittest.TestCase):
    def test_psi_layout(self):
        s = State.zero(3)
        self.assertEqual(s.psi.shape, (2, 2, 2))
        self.assertEqual(s.psi.dtype, np.complex128)
        self.assertEqual(s.n, 3)

    def test_norm_and_probabilities(self):
        s = State(random_state(4, np.random.default_rng(2)))
        self.assertAlmostEqual(s.norm(), 1.0, places=12)
        self.assertAlmostEqual(np.linalg.norm(s.psi), 1.0, places=12)
        self.assertAlmostEqual(s.probabilities().sum(), 1.0, places=12)

    def test_rejects_unnormalized(self):
        with self.assertRaises(ValueError):
            State(np.ones(4))
        State(np.ones(4), check_norm=False)

    def test_flat_and_tensor_inputs_agree(self):
        vec = random_state(3, np.random.default_rng(3))
        np.testing.assert_array_equal(State(vec).psi, State(vec.reshape(2, 2, 2)).psi)

    def test_strided_input_gives_contiguous_vector(self):
        vecs = np.linalg.qr(np.random.default_rng(7).normal(size=(8, 8)))[0]
        s = State(vecs[:, 0])
        self.assertTrue(s.vector.flags.c_contiguous)
        np.testing.assert_allclose(s.vector, vecs[:, 0])

    def test_from_label(self):
        # label[0] is the highest qubit: "01" -> qubit 0 is |1> -> flat index 1.
        np.testing.assert_array_equal(State.from_label("01").vector, [0, 1, 0, 0])
        plus = State.from_label("+")
        np.testing.assert_allclose(plus.vector, [1 / np.sqrt(2)] * 2)
        self.assertAlmostEqual(State.from_label("r").inner(State.from_label("l")), 0)
        with self.assertRaises(ValueError):
            State.from_label("0x")

    def test_marginal_probabilities_order(self):
        s = State.basis(0b110, 3)  # q2=1, q1=1, q0=0
        np.testing.assert_array_equal(s.probabilities([0, 1]), [0, 0, 1, 0])  # b0 + 2 b1 = 2
        np.testing.assert_array_equal(s.probabilities([1, 0]), [0, 1, 0, 0])  # b1 + 2 b0 = 1
        np.testing.assert_array_equal(s.probabilities([2]), [0, 1])

    def test_marginal_matches_brute_force(self):
        s = State(random_state(4, np.random.default_rng(4)))
        qubits = [3, 0]
        expected = np.zeros(4)
        for k, p in enumerate(s.probabilities()):
            expected[((k >> 3) & 1) + 2 * (k & 1)] += p
        np.testing.assert_allclose(s.probabilities(qubits), expected, atol=1e-14)

    def test_apply_chains_and_bell(self):
        bell = State.zero(2).apply(gates.H, [0]).apply(gates.CX, [0, 1])
        np.testing.assert_allclose(bell.vector, np.array([1, 0, 0, 1]) / np.sqrt(2), atol=1e-15)

    def test_inner_and_fidelity(self):
        zero, plus = State.from_label("0"), State.from_label("+")
        self.assertAlmostEqual(zero.inner(plus), 1 / np.sqrt(2))
        self.assertAlmostEqual(zero.fidelity(plus), 0.5)
        self.assertAlmostEqual(zero.fidelity(zero), 1.0)

    def test_tensor_product(self):
        combined = State.from_label("1").tensor(State.from_label("0"))
        np.testing.assert_array_equal(combined.vector, State.from_label("10").vector)
        a, b = State(random_state(2, np.random.default_rng(5))), State(random_state(1, np.random.default_rng(6)))
        np.testing.assert_allclose(a.tensor(b).vector, np.kron(a.vector, b.vector))

    def test_copy_is_independent(self):
        a = State.zero(1)
        b = a.copy().apply(gates.X, [0])
        self.assertEqual(a.vector[0], 1)
        self.assertEqual(b.vector[1], 1)

    def test_sample_counts(self):
        bell = State.zero(2).apply(gates.H, [0]).apply(gates.CX, [0, 1])
        counts = bell.sample_counts(2000, rng=0)
        self.assertEqual(set(counts), {"00", "11"})
        self.assertEqual(sum(counts.values()), 2000)
        # Bitstring order: qubit 0 rightmost.
        self.assertEqual(State.from_label("01").sample_counts(5, rng=0), {"01": 5})


if __name__ == "__main__":
    unittest.main()
