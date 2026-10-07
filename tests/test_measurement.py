"""Measurement (collapse) and sampling, Part 2 of the guide.

The collapse checks compare against an independent projection written over
flat little-endian indices, so they test the qubit-axis convention too.
"""

import unittest
from collections import Counter

import numpy as np

from quantum_simulator import gates
from quantum_simulator.state import ZERO_BRANCH_TOL, State
from tests.reference import random_state

SQ2 = 1 / np.sqrt(2)


class FixedRng(np.random.Generator):
    """A Generator whose ``random()`` always returns ``value``, to force an outcome."""

    def __init__(self, value: float):
        super().__init__(np.random.PCG64(0))
        self.value = value

    def random(self, *args, **kwargs):
        return self.value


def projected(vec: np.ndarray, qubit: int, outcome: int) -> np.ndarray:
    """Reference collapse: keep amplitudes whose index has bit ``qubit`` == outcome, renormalize."""
    keep = ((np.arange(vec.size) >> qubit) & 1) == outcome
    out = np.where(keep, vec, 0)
    return out / np.linalg.norm(out)


def ghz(n: int) -> State:
    s = State.zero(n).apply(gates.H, [0])
    for q in range(1, n):
        s.apply(gates.CX, [0, q])
    return s


class TestProbOne(unittest.TestCase):
    def test_basis_state(self):
        s = State.basis(0b101, 3)  # q2 = 1, q1 = 0, q0 = 1
        self.assertEqual([s.prob_one(q) for q in range(3)], [1.0, 0.0, 1.0])

    def test_matches_marginal_probabilities(self):
        s = State(random_state(4, np.random.default_rng(0)))
        for q in range(4):
            self.assertAlmostEqual(s.prob_one(q), s.probabilities([q])[1], places=12)
        self.assertAlmostEqual(State.from_label("+").prob_one(0), 0.5)


class TestMeasure(unittest.TestCase):
    def test_basis_state_is_deterministic_and_unchanged(self):
        s = State.basis(0b101, 3)
        before = s.vector.copy()
        self.assertEqual([s.measure(q, rng=q) for q in range(3)], [1, 0, 1])
        np.testing.assert_array_equal(s.vector, before)

    def test_returns_plain_int(self):
        self.assertIs(type(State.zero(1).measure(0, 0)), int)

    def test_collapse_matches_reference_projection(self):
        rng = np.random.default_rng(1)
        vec = random_state(4, rng)
        for q in range(4):
            for _ in range(5):
                with self.subTest(qubit=q):
                    s = State(vec)
                    outcome = s.measure(q, rng)
                    np.testing.assert_allclose(s.vector, projected(vec, q, outcome), atol=1e-12)

    def test_collapse_leaves_other_qubits_alone(self):
        # q1 in |->, q0 in |+>: measuring q0 leaves |-> on q1 untouched.
        rng = np.random.default_rng(2)
        for _ in range(10):
            s = State.from_label("-+")
            outcome = s.measure(0, rng)
            np.testing.assert_allclose(s.vector, State.from_label("-" + "01"[outcome]).vector, atol=1e-15)

    def test_bell_pair_outcomes_agree(self):
        rng = np.random.default_rng(3)
        bell = State.zero(2).apply(gates.H, [0]).apply(gates.CX, [0, 1])
        seen = set()
        for _ in range(200):
            s = bell.copy()
            a = s.measure(0, rng)
            # After the first measurement the pair is |aa>, so the second is certain.
            np.testing.assert_allclose(s.vector, State.basis(0b11 * a, 2).vector, atol=1e-15)
            self.assertEqual(s.measure(1, rng), a)
            seen.add(a)
        self.assertEqual(seen, {0, 1})

    def test_ghz_one_measurement_fixes_all(self):
        rng = np.random.default_rng(4)
        for q in range(4):
            s = ghz(4)
            a = s.measure(q, rng)
            self.assertEqual([s.measure(other, rng) for other in range(4)], [a] * 4)

    def test_repeated_measurement_is_stable(self):
        rng = np.random.default_rng(5)
        s = State(random_state(3, rng))
        first = s.measure(1, rng)
        after = s.vector.copy()
        for _ in range(20):
            self.assertEqual(s.measure(1, rng), first)
        np.testing.assert_allclose(s.vector, after, atol=1e-15)

    def test_born_rule_frequencies(self):
        # RY(theta)|0> with P(1) = 0.3, and the middle qubit of an entangled state.
        rng = np.random.default_rng(6)
        theta = 2 * np.arcsin(np.sqrt(0.3))
        cases = [
            (State.zero(1).apply(gates.ry(theta), [0]), 0),
            (State(random_state(3, np.random.default_rng(7))), 1),
        ]
        shots = 20_000
        for start, q in cases:
            p = start.prob_one(q)
            ones = sum(start.copy().measure(q, rng) for _ in range(shots))
            sigma = np.sqrt(p * (1 - p) / shots)
            self.assertLess(abs(ones / shots - p), 5 * sigma)

    def test_post_state_normalized_and_precision_kept(self):
        rng = np.random.default_rng(8)
        for dtype, places in [(np.complex64, 6), (np.complex128, 12)]:
            s = State(random_state(5, rng), dtype=dtype)
            s.measure(2, rng)
            self.assertEqual(s.dtype, np.dtype(dtype))
            self.assertAlmostEqual(s.norm(), 1.0, places=places)

    def test_renormalizes_norm_drift(self):
        # Norm drifted to 1.1, so the raw |amplitude|^2 for 1 is 0.968 but the relative P(1) is 0.8.
        # random() = 0.9 separates the two: relative gives outcome 0, raw would give 1.
        vec = np.array([np.sqrt(0.2), np.sqrt(0.8)]) * 1.1
        s = State(vec, check_norm=False)
        self.assertEqual(s.measure(0, FixedRng(0.9)), 0)
        np.testing.assert_allclose(s.vector, [1, 0], atol=1e-15)

    def test_does_not_modify_caller_array(self):
        vec = np.array([SQ2, SQ2], dtype=complex)
        s = State(vec)
        s.measure(0, 0)
        np.testing.assert_array_equal(vec, [SQ2, SQ2])

    def test_rounding_noise_branch_is_never_selected(self):
        # P(1) = 1e-20 is rounding noise. random() = 0.0 would select it without the cutoff.
        s = State(np.array([1, 1e-10]) / np.linalg.norm([1, 1e-10]))
        self.assertEqual(s.measure(0, FixedRng(0.0)), 0)
        np.testing.assert_array_equal(s.vector, [1, 0])
        # And symmetrically when P(0) is the negligible one.
        s = State(np.array([1e-10, 1]) / np.linalg.norm([1e-10, 1]))
        self.assertEqual(s.measure(0, FixedRng(1 - 1e-12)), 1)

    def test_cutoff_depends_on_precision(self):
        # P(1) = 1e-7 is below complex64's rounding level but well above complex128's.
        p1 = 1e-7
        self.assertLess(p1, ZERO_BRANCH_TOL[np.dtype(np.complex64)])
        self.assertGreater(p1, ZERO_BRANCH_TOL[np.dtype(np.complex128)])
        vec = np.array([np.sqrt(1 - p1), np.sqrt(p1)])
        self.assertEqual(State(vec, dtype=np.complex64).measure(0, FixedRng(0.0)), 0)
        self.assertEqual(State(vec, dtype=np.complex128).measure(0, FixedRng(0.0)), 1)

    def test_zero_vector_rejected(self):
        with self.assertRaises(ValueError):
            State(np.zeros(2), check_norm=False).measure(0, 0)

    def test_reproducible_from_seed(self):
        def run(seed):
            rng = np.random.default_rng(seed)
            s = State(random_state(4, np.random.default_rng(9)))
            return [s.copy().measure(q, rng) for q in range(4) for _ in range(10)]

        self.assertEqual(run(42), run(42))
        self.assertNotEqual(run(42), run(43))


class TestRngIsRequired(unittest.TestCase):
    def test_rejects_missing_or_invalid_rng(self):
        s = State.from_label("+")
        for bad in [None, True, "0", 1.5]:
            with self.subTest(rng=bad):
                with self.assertRaises(TypeError):
                    s.copy().measure(0, bad)
                with self.assertRaises(TypeError):
                    s.sample(10, bad)
                with self.assertRaises(TypeError):
                    s.sample_counts(10, bad)
        with self.assertRaises(TypeError):
            s.sample(10)  # no default

    def test_accepts_numpy_integer_seed(self):
        self.assertEqual(len(State.zero(1).sample(3, np.int64(7))), 3)


class TestSample(unittest.TestCase):
    def test_shape_dtype_range(self):
        out = State(random_state(3, np.random.default_rng(10))).sample(500, 0)
        self.assertEqual(out.shape, (500,))
        self.assertEqual(out.dtype, np.int64)
        self.assertTrue(((out >= 0) & (out < 8)).all())

    def test_basis_state_always_same_index(self):
        np.testing.assert_array_equal(State.basis(0b110, 3).sample(50, 0), np.full(50, 6))

    def test_does_not_collapse(self):
        s = State(random_state(3, np.random.default_rng(11)))
        before = s.vector.copy()
        s.sample(1000, 0)
        np.testing.assert_array_equal(s.vector, before)

    def test_bell_pair_only_correlated_outcomes(self):
        bell = State.zero(2).apply(gates.H, [0]).apply(gates.CX, [0, 1])
        self.assertEqual(set(bell.sample(1000, 0).tolist()), {0b00, 0b11})

    def test_frequencies_match_probabilities(self):
        s = State(random_state(3, np.random.default_rng(12)))
        p = s.probabilities()
        shots = 200_000
        freq = np.bincount(s.sample(shots, 13), minlength=8) / shots
        sigma = np.sqrt(p * (1 - p) / shots)
        self.assertTrue((np.abs(freq - p) < 5 * sigma + 1e-12).all(), (freq, p))

    def test_reproducible_from_seed(self):
        s = State(random_state(3, np.random.default_rng(14)))
        np.testing.assert_array_equal(s.sample(100, 1), s.sample(100, 1))
        self.assertFalse(np.array_equal(s.sample(100, 1), s.sample(100, 2)))

    def test_consistent_with_sample_counts(self):
        s = State.from_label("+0+")
        labels = Counter(format(k, "03b") for k in s.sample(4000, 15))
        counts = s.sample_counts(4000, 15)
        self.assertEqual(set(labels), set(counts))
        self.assertEqual(set(labels), {"000", "001", "100", "101"})  # q1 is always 0

    def test_zero_vector_rejected(self):
        with self.assertRaises(ValueError):
            State(np.zeros(4), check_norm=False).sample(5, 0)


if __name__ == "__main__":
    unittest.main()
