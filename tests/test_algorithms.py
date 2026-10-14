"""Algorithms as tests: each asserts the property that defines the algorithm."""

import unittest

import numpy as np

from quantum_simulator import algorithms as alg
from quantum_simulator import gates
from quantum_simulator.circuit import Circuit, Op, multi_controlled_matrix, run
from quantum_simulator.entanglement import entanglement_entropy, reduced_density_matrix
from quantum_simulator.state import State


def unitary_of(c: Circuit) -> np.ndarray:
    """Full matrix of a unitary circuit, column by column from basis states."""
    return np.column_stack([run(c, initial=State.basis(j, c.n)).state.vector for j in range(2**c.n)])


class TestBellAndGhz(unittest.TestCase):
    def test_bell_pair_is_one_bit_on_either_qubit(self):
        s = run(alg.bell_pair()).state
        np.testing.assert_allclose(s.vector, [2**-0.5, 0, 0, 2**-0.5], atol=1e-15)
        self.assertAlmostEqual(entanglement_entropy(s, [0]), 1.0)
        self.assertAlmostEqual(entanglement_entropy(s, [1]), 1.0)

    def test_ghz_has_exactly_two_amplitudes(self):
        for n in range(1, 11):
            vec = run(alg.ghz(n)).state.vector
            nonzero = np.flatnonzero(np.abs(vec) > 1e-12)
            with self.subTest(n=n):
                self.assertEqual(nonzero.tolist(), [0, 2**n - 1] if n > 1 else [0, 1])
                np.testing.assert_allclose(vec[nonzero], 2**-0.5, atol=1e-14)


class TestTeleportation(unittest.TestCase):
    def test_every_branch_delivers_the_input(self):
        rng = np.random.default_rng(0)
        for theta, phi in [(0.7, 1.9), (2.4, -0.6), (np.pi / 2, 0.0), (3.0, 2.2)]:
            psi = State.zero(1).apply(gates.ry(theta), [0]).apply(gates.rz(phi), [0]).vector
            target = np.outer(psi, psi.conj())
            branches = set()
            for _ in range(40):
                r = run(alg.teleportation(theta, phi), rng=rng)
                branches.add(r.clbits)
                np.testing.assert_allclose(reduced_density_matrix(r.state, [2]), target, atol=1e-12)
            self.assertEqual(branches, {(0, 0), (0, 1), (1, 0), (1, 1)})


class TestBernsteinVazirani(unittest.TestCase):
    def test_recovers_every_secret_with_certainty(self):
        n = 5
        for secret in range(2**n):
            probs = run(alg.bernstein_vazirani(secret, n)).state.probabilities(range(n))
            with self.subTest(secret=secret):
                self.assertAlmostEqual(probs[secret], 1.0, places=12)

    def test_one_oracle_query(self):
        # The oracle is the block of CX gates onto the ancilla: one per 1-bit of the secret,
        # applied once. A classical algorithm would need n queries.
        c = alg.bernstein_vazirani(0b10110, 5)
        self.assertEqual(sum(op.name == "cx" for op in c.ops), 3)

    def test_validation(self):
        with self.assertRaises(ValueError):
            alg.bernstein_vazirani(32, 5)


class TestGrover(unittest.TestCase):
    n, marked = 8, 0b10110010

    def success(self, k: int) -> float:
        return run(alg.grover(self.n, self.marked, k)).state.probabilities()[self.marked]

    def test_optimal_iterations_succeed(self):
        k_opt = alg.optimal_grover_iterations(self.n)
        self.assertEqual(k_opt, 12)  # floor((pi/4) * sqrt(256))
        self.assertGreater(self.success(k_opt), 0.99)

    def test_over_rotation_falls(self):
        # Grover is a rotation by 2 theta: past the optimum it rotates away again.
        k_opt = alg.optimal_grover_iterations(self.n)
        peak = self.success(k_opt)
        self.assertLess(self.success(k_opt + 4), peak - 0.2)
        self.assertLess(self.success(2 * k_opt), 0.05)

    def test_matches_rotation_theory(self):
        for k in [0, 1, 3, 7, 12, 18, 25, 37]:
            self.assertAlmostEqual(self.success(k), alg.grover_success_theory(self.n, k), places=10)

    def test_every_marked_item_on_small_registers(self):
        for n in (2, 3, 4):
            for marked in range(2**n):
                p = run(alg.grover(n, marked)).state.probabilities()[marked]
                with self.subTest(n=n, marked=marked):
                    self.assertAlmostEqual(p, alg.grover_success_theory(n, alg.optimal_grover_iterations(n)), places=10)

    def test_validation(self):
        with self.assertRaises(ValueError):
            alg.grover(1, 0)
        with self.assertRaises(ValueError):
            alg.grover(3, 8)


class TestQft(unittest.TestCase):
    def test_amplitudes_are_roots_of_unity(self):
        for n in range(1, 6):
            N = 2**n
            for j in range(N):
                out = run(alg.qft(n), initial=State.basis(j, n)).state.vector
                with self.subTest(n=n, j=j):
                    np.testing.assert_allclose(out, np.exp(2j * np.pi * j * np.arange(N) / N) / np.sqrt(N), atol=1e-13)

    def test_is_the_unitary_dft_matrix(self):
        n, N = 4, 16
        dft = np.exp(2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N) / np.sqrt(N)
        np.testing.assert_allclose(unitary_of(alg.qft(n)), dft, atol=1e-13)

    def test_inverse_qft_undoes_it(self):
        np.testing.assert_allclose(unitary_of(alg.qft(4).inverse()) @ unitary_of(alg.qft(4)), np.eye(16), atol=1e-13)

    def test_without_swaps_the_output_is_bit_reversed(self):
        n = 3
        rev = [int(format(k, f"0{n}b")[::-1], 2) for k in range(2**n)]
        np.testing.assert_allclose(unitary_of(alg.qft(n, swaps=False))[rev], unitary_of(alg.qft(n)), atol=1e-13)


class TestPhaseEstimation(unittest.TestCase):
    def test_exact_phases_are_recovered_with_certainty(self):
        t = 4
        for m in range(2**t):
            probs = run(alg.phase_estimation(m / 2**t, t)).state.probabilities(range(t))
            with self.subTest(m=m):
                self.assertAlmostEqual(probs[m], 1.0, places=10)

    def test_inexact_phases_land_on_the_nearest_integer(self):
        # The nearest t-bit estimate has probability at least 4/pi^2 (about 0.405).
        t = 6
        for phase in [0.3, 0.123, 0.777, 0.501]:
            probs = run(alg.phase_estimation(phase, t)).state.probabilities(range(t))
            best = int(np.argmax(probs))
            with self.subTest(phase=phase):
                self.assertEqual(best, round(phase * 2**t) % 2**t)
                self.assertGreaterEqual(probs[best], 4 / np.pi**2)

    def test_more_counting_qubits_more_precision(self):
        phase = 1 / 3  # never exact in binary
        errors = []
        for t in (3, 5, 7, 9):
            probs = run(alg.phase_estimation(phase, t)).state.probabilities(range(t))
            errors.append(abs(np.argmax(probs) / 2**t - phase))
        self.assertTrue(all(e <= 2 ** -(t + 1) + 1e-12 for e, t in zip(errors, (3, 5, 7, 9))))
        self.assertLess(errors[-1], errors[0])


class TestNewCircuitPieces(unittest.TestCase):
    """mcx, mcz, inverse and compose, which the algorithms above are built from."""

    def test_multi_controlled_matrices(self):
        self.assertTrue(np.array_equal(multi_controlled_matrix("mcx", 2), gates.CX))
        self.assertTrue(np.array_equal(multi_controlled_matrix("mcx", 3), gates.controlled(gates.controlled(gates.X))))
        self.assertTrue(np.array_equal(multi_controlled_matrix("mcz", 2), gates.CZ))
        with self.assertRaises(ValueError):
            multi_controlled_matrix("mcx", 1)

    def test_mcx_truth_table(self):
        n = 5
        for k in range(2**n):
            out = run(Circuit(n).mcx([0, 2, 3, 4], 1), initial=State.basis(k, n)).state.vector
            expected = k ^ 0b00010 if (k & 0b11101) == 0b11101 else k
            self.assertEqual(int(np.argmax(np.abs(out))), expected)

    def test_mcz_is_symmetric_and_flips_only_all_ones(self):
        rng = np.random.default_rng(1)
        vec = rng.normal(size=16) + 1j * rng.normal(size=16)
        s = State(vec / np.linalg.norm(vec))
        a = run(Circuit(4).mcz(0, 1, 2, 3), initial=s).state.vector
        b = run(Circuit(4).mcz(3, 1, 0, 2), initial=s).state.vector
        np.testing.assert_allclose(a, b, atol=1e-15)
        expected = s.vector.copy()
        expected[15] *= -1
        np.testing.assert_allclose(a, expected, atol=1e-15)

    def test_inverse_rules(self):
        c = Circuit(2).s(0).t(1).rx(0, 0.4).cp(0, 1, 1.2).h(1)
        self.assertEqual([op.name for op in c.inverse().ops], ["h", "cp", "rx", "tdg", "sdg"])
        self.assertEqual([op.value for op in c.inverse().ops if op.value is not None], [-1.2, -0.4])
        for bad in [Circuit(1).rx(0), Circuit(1).sx(0), Circuit(2).iswap(0, 1),
                    Circuit(1, 1).measure(0, 0), Circuit(1, 1).x(0).c_if(0)]:
            with self.assertRaises(ValueError):
                bad.inverse()

    def test_compose_maps_qubits(self):
        block = Circuit(2).h(0).cx(0, 1)
        c = Circuit(4).compose(block, qubits=[3, 1])
        self.assertEqual(c.ops, [Op("h", (3,)), Op("cx", (3, 1))])
        with self.assertRaises(ValueError):
            Circuit(4).compose(block, qubits=[0])
        with self.assertRaises(ValueError):
            Circuit(4).compose(Circuit(1).rx(0))
        with self.assertRaises(ValueError):
            Circuit(4).compose(Circuit(1, 1).measure(0, 0))


if __name__ == "__main__":
    unittest.main()
