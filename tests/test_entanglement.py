import unittest

import numpy as np

from quantum_simulator import algorithms as alg
from quantum_simulator import gates
from quantum_simulator.circuit import run
from quantum_simulator.entanglement import (
    bipartition_matrix,
    entanglement_entropy,
    entropy_profile,
    page_entropy,
    purity,
    reduced_density_matrix,
    schmidt_decomposition,
    schmidt_rank,
)
from quantum_simulator.observables import tfim
from quantum_simulator.state import State
from tests.reference import random_state


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
        bell = run(alg.ghz(2)).state
        self.assertEqual(schmidt_rank(bell, [0]), 2)
        self.assertAlmostEqual(entanglement_entropy(bell, [0]), 1.0)
        np.testing.assert_allclose(reduced_density_matrix(bell, [1]), np.eye(2) / 2, atol=1e-15)
        g = run(alg.ghz(4)).state
        for qubits in [[0], [3], [1, 2], [0, 1, 3]]:
            self.assertAlmostEqual(entanglement_entropy(g, qubits), 1.0)

    def test_entropy_symmetric_across_cut(self):
        vec = random_state(5, np.random.default_rng(1))
        self.assertAlmostEqual(entanglement_entropy(vec, [0, 3]), entanglement_entropy(vec, [1, 2, 4]))

    def test_renyi(self):
        bell = run(alg.ghz(2)).state
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


class TestEntropyProfile(unittest.TestCase):
    def test_matches_entropy_per_cut_and_its_complement(self):
        n = 7
        vec = random_state(n, np.random.default_rng(10))
        profile = entropy_profile(vec)
        self.assertEqual(profile.shape, (n + 1,))
        self.assertAlmostEqual(profile[0], 0.0)
        self.assertAlmostEqual(profile[n], 0.0)
        for l in range(n + 1):
            self.assertAlmostEqual(profile[l], entanglement_entropy(vec, range(l)), places=12)
            # S(A) = S(complement): the complement of qubits 0..l-1 is l..n-1.
            self.assertAlmostEqual(profile[l], entanglement_entropy(vec, range(l, n)), places=10)

    def test_symmetric_only_for_translation_invariant_states(self):
        # A periodic chain's ground state looks the same from both ends; a random state does not.
        ring = entropy_profile(tfim(8, h=0.7, periodic=True).ground_state()[1])
        np.testing.assert_allclose(ring, ring[::-1], atol=1e-10)
        rand = entropy_profile(random_state(7, np.random.default_rng(10)))
        self.assertFalse(np.allclose(rand, rand[::-1], atol=1e-3))

    def test_product_and_ghz(self):
        np.testing.assert_allclose(entropy_profile(State.from_label("+0-r1")), 0, atol=1e-12)
        np.testing.assert_allclose(entropy_profile(run(alg.ghz(5)).state), [0, 1, 1, 1, 1, 0], atol=1e-12)

    def test_renyi_profile(self):
        vec = random_state(5, np.random.default_rng(11))
        np.testing.assert_allclose(
            entropy_profile(vec, alpha=2), [entanglement_entropy(vec, range(l), alpha=2) for l in range(6)]
        )


class TestEntropyInequalities(unittest.TestCase):
    """Facts every pure state must satisfy, checked on random states."""

    def setUp(self):
        self.rng = np.random.default_rng(12)

    def test_bounded_by_smaller_side(self):
        n = 6
        for _ in range(10):
            vec = random_state(n, self.rng)
            for k in range(n + 1):
                qubits = list(self.rng.choice(n, k, replace=False))
                self.assertLessEqual(entanglement_entropy(vec, qubits), min(k, n - k) + 1e-12)

    def test_subadditivity_and_araki_lieb(self):
        # |S(A) - S(B)| <= S(AB) <= S(A) + S(B) for disjoint A and B.
        S = entanglement_entropy
        for _ in range(20):
            vec = random_state(6, self.rng)
            a, b = [0, 3], [1, 5]
            sa, sb, sab = S(vec, a), S(vec, b), S(vec, a + b)
            self.assertLessEqual(sab, sa + sb + 1e-12)
            self.assertLessEqual(abs(sa - sb), sab + 1e-12)

    def test_strong_subadditivity(self):
        # S(AB) + S(BC) >= S(B) + S(ABC): the deepest inequality of quantum entropy.
        S = entanglement_entropy
        for _ in range(20):
            vec = random_state(6, self.rng)
            a, b, c = [0], [2, 4], [5]
            self.assertGreaterEqual(S(vec, a + b) + S(vec, b + c), S(vec, b) + S(vec, a + b + c) - 1e-12)


class TestPageEntropy(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(page_entropy(1, 2), 1 / (3 * np.log(2)))  # Page: 1/3 nat for two qubits
        self.assertEqual(page_entropy(0, 5), 0.0)
        self.assertAlmostEqual(page_entropy(2, 6), page_entropy(4, 6))
        self.assertAlmostEqual(page_entropy(1, 2, base=np.e), 1 / 3)

    def test_large_n_branch_is_continuous(self):
        # Below and above the exact-sum cutoff, both close to n_a - 1/(2 ln 2) at half size.
        for n in (26, 28, 30):
            self.assertAlmostEqual(page_entropy(n // 2, n), n // 2 - 1 / (2 * np.log(2)), places=4)

    def test_matches_sampled_random_states(self):
        rng = np.random.default_rng(13)
        for n, n_a in [(4, 2), (6, 3), (7, 2)]:
            samples = [entanglement_entropy(random_state(n, rng), range(n_a)) for _ in range(400)]
            stderr = np.std(samples) / np.sqrt(len(samples))
            with self.subTest(n=n, n_a=n_a):
                self.assertLess(abs(np.mean(samples) - page_entropy(n_a, n)), 4 * stderr)

    def test_validation(self):
        with self.assertRaises(ValueError):
            page_entropy(5, 4)


class TestAreaLaw(unittest.TestCase):
    """Ground states of the periodic TFIM on 10 qubits, the regimes in experiments/entanglement_scaling.py."""

    @classmethod
    def setUpClass(cls):
        cls.profiles = {
            h: entropy_profile(tfim(10, h=h, periodic=True).ground_state()[1]) for h in (0.3, 1.0, 3.0)
        }

    def test_gapped_phases_saturate(self):
        disordered, ordered = self.profiles[3.0], self.profiles[0.3]
        self.assertLess(abs(disordered[5] - disordered[2]), 0.01)
        self.assertLess(disordered[5], 0.3)
        # Ordered: a cat-like superposition of the two ordered states, worth one bit at every cut.
        np.testing.assert_allclose(ordered[2:9], 1.0, atol=0.02)

    def test_critical_point_keeps_growing(self):
        critical = self.profiles[1.0]
        self.assertGreater(critical[5] - critical[2], 0.1)
        self.assertTrue(np.all(np.diff(critical[:6]) > 0))

    def test_random_state_obeys_volume_law(self):
        vec = random_state(10, np.random.default_rng(14))
        profile = entropy_profile(vec)
        self.assertGreater(profile[5], 4.0)
        self.assertGreater(profile[5], 10 * self.profiles[3.0][5])


class TestNoCloning(unittest.TestCase):
    def test_cnot_copies_basis_states_but_not_superpositions(self):
        # CNOT maps |b>|0> to |b>|b> for b in {0, 1}, so it "copies" basis states.
        for b in (0, 1):
            out = State.basis(b, 2).apply(gates.CX, [0, 1])
            self.assertAlmostEqual(out.fidelity(State.basis(0b11 * b, 2)), 1.0)
        # Linearity then forces |+>|0> to a Bell state, not |+>|+>: fidelity 1/2.
        out = State.from_label("0+").apply(gates.CX, [0, 1])
        self.assertAlmostEqual(out.fidelity(State.from_label("++")), 0.5)
        self.assertAlmostEqual(entanglement_entropy(out, [0]), 1.0)

    def test_copy_is_a_simulator_privilege(self):
        s = State(random_state(3, np.random.default_rng(15)))
        clone = s.copy()
        self.assertAlmostEqual(clone.fidelity(s), 1.0)
        self.assertIsNot(clone.psi, s.psi)


if __name__ == "__main__":
    unittest.main()
