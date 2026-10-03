"""Precision is a parameter: complex64 must stay complex64 through every operation."""

import unittest

import numpy as np

from quantum_simulator import gates
from quantum_simulator.entanglement import entanglement_entropy, schmidt_decomposition
from quantum_simulator.observables import PauliSum
from quantum_simulator.state import (
    State,
    apply_gate,
    basis_state,
    default_atol,
    from_vector,
    is_normalized,
    resolve_dtype,
    zero_state,
)
from tests.reference import random_state

C64, C128 = np.dtype(np.complex64), np.dtype(np.complex128)
BOTH = [C64, C128]


def random_circuit(state: State, depth: int, rng: np.random.Generator) -> State:
    n = state.n
    for _ in range(depth):
        for q in range(n):
            state.apply(gates.u(*rng.uniform(-np.pi, np.pi, 3)), [q])
        for q in range(n - 1):
            state.apply(gates.CX, [q, q + 1])
    return state


class TestDtypeResolution(unittest.TestCase):
    def test_default_is_complex128(self):
        self.assertEqual(resolve_dtype(None), C128)
        self.assertEqual(State.zero(2).dtype, C128)

    def test_accepts_aliases(self):
        self.assertEqual(resolve_dtype("complex64"), C64)
        self.assertEqual(resolve_dtype(np.complex64), C64)

    def test_rejects_other_dtypes(self):
        for bad in [np.float64, np.complex256 if hasattr(np, "complex256") else np.int32, "float32"]:
            with self.assertRaises(ValueError):
                resolve_dtype(bad)

    def test_inference(self):
        self.assertEqual(from_vector(np.array([1, 0], dtype=np.float32)).dtype, C64)
        self.assertEqual(from_vector(np.array([1, 0], dtype=np.complex64)).dtype, C64)
        self.assertEqual(from_vector([1, 0]).dtype, C128)
        self.assertEqual(State(np.array([0, 1], dtype=np.complex64)).dtype, C64)
        # An explicit dtype wins over inference.
        self.assertEqual(State(np.array([0, 1], dtype=np.complex64), dtype=C128).dtype, C128)

    def test_tolerance_scales_with_precision(self):
        self.assertGreater(default_atol(C64), default_atol(C128))
        # A norm error at single-precision scale passes for complex64 but not complex128.
        vec = random_state(10, np.random.default_rng(0)) * (1 + 3e-7)
        self.assertTrue(is_normalized(vec.astype(C64)))
        self.assertFalse(is_normalized(vec.astype(C128)))
        State(vec.astype(C64))
        with self.assertRaises(ValueError):
            State(vec.astype(C128))


class TestPrecisionIsPreserved(unittest.TestCase):
    def test_constructors(self):
        for dt in BOTH:
            with self.subTest(dtype=dt):
                self.assertEqual(zero_state(3, dt).dtype, dt)
                self.assertEqual(basis_state(5, 3, dt).dtype, dt)
                self.assertEqual(State.zero(3, dt).dtype, dt)
                self.assertEqual(State.basis(2, 3, dt).dtype, dt)
                self.assertEqual(State.from_label("+r1", dt).dtype, dt)
                self.assertEqual(State.zero(4, dt).nbytes, 16 * dt.itemsize)

    def test_gates_do_not_promote(self):
        # The complex128 gate constants must not drag a complex64 state up to complex128.
        s = State.zero(4, C64)
        random_circuit(s, 3, np.random.default_rng(1))
        s.apply(gates.rzz(0.3), [3, 0]).apply(gates.controlled(gates.controlled(gates.X)), [0, 2, 1])
        self.assertEqual(s.dtype, C64)
        self.assertEqual(apply_gate(zero_state(2, C64), gates.H, [0]).dtype, C64)

    def test_derived_quantities(self):
        s = random_circuit(State.zero(4, C64), 2, np.random.default_rng(2))
        self.assertEqual(s.probabilities().dtype, np.float32)
        self.assertEqual(s.copy().dtype, C64)
        self.assertEqual(s.tensor(State.zero(1, C64)).dtype, C64)
        self.assertEqual(s.tensor(State.zero(1, C128)).dtype, C128)  # mixed -> promote
        self.assertEqual(sum(s.sample_counts(100, rng=0).values()), 100)
        self.assertEqual(schmidt_decomposition(s, [0, 1])[0].dtype, np.float32)

    def test_astype(self):
        s = State.from_label("+-", C128)
        lo = s.astype(C64)
        self.assertEqual((lo.dtype, s.dtype), (C64, C128))
        np.testing.assert_allclose(lo.vector, s.vector, atol=1e-7)

    def test_observables(self):
        h = PauliSum([("ZZI", -1.0), ("IZZ", -1.0), ("XII", -0.5), ("IXI", -0.5), ("IIX", -0.5)])
        s = random_circuit(State.zero(3, C64), 2, np.random.default_rng(3))
        self.assertEqual(h.apply(s).dtype, C64)
        self.assertEqual(h.to_matrix(dtype=C64).dtype, C64)
        energies, vecs = h.eigh(dtype=C64)
        self.assertEqual((energies.dtype, vecs.dtype), (np.float32, C64))
        energy, gs = h.ground_state(dtype=C64)
        self.assertEqual(gs.dtype, C64)
        self.assertAlmostEqual(energy, h.ground_state()[0], places=5)


class TestComplex64Accuracy(unittest.TestCase):
    """complex64 results agree with complex128 to single-precision accuracy."""

    def test_circuit(self):
        rng_a, rng_b = np.random.default_rng(4), np.random.default_rng(4)
        lo = random_circuit(State.zero(6, C64), 10, rng_a)
        hi = random_circuit(State.zero(6, C128), 10, rng_b)
        np.testing.assert_allclose(lo.vector, hi.vector, atol=1e-5)
        self.assertAlmostEqual(lo.norm(), 1.0, places=5)

    def test_energy_and_entropy(self):
        h = PauliSum([("ZZII", -1), ("IZZI", -1), ("IIZZ", -1), ("XIII", -1), ("IXII", -1), ("IIXI", -1), ("IIIX", -1)])
        hi = random_circuit(State.zero(4), 5, np.random.default_rng(5))
        lo = hi.astype(C64)
        self.assertAlmostEqual(h.expectation(lo), h.expectation(hi), places=5)
        self.assertAlmostEqual(entanglement_entropy(lo, [0, 1]), entanglement_entropy(hi, [0, 1]), places=5)


if __name__ == "__main__":
    unittest.main()
