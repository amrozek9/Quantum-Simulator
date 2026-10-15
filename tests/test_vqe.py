"""Variational ground states: the scan-compiled ansatz, Adam, VQE, and the DLA dimension."""

import unittest

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    from quantum_simulator.backends import jax_backend as jb
    from quantum_simulator.vqe import (
        adam,
        ansatz_state,
        dynamical_lie_algebra_dimension,
        hardware_efficient_ansatz,
        make_ansatz_expectations,
        vqe,
    )
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("jax not installed") from None

from quantum_simulator.gradients import parameter_shift_gradient
from quantum_simulator.observables import PauliSum, pauli_string, tfim


def tfim_terms(n: int) -> tuple[PauliSum, PauliSum]:
    zz = PauliSum([(pauli_string(n, {i: "Z", i + 1: "Z"}), 1.0) for i in range(n - 1)])
    xs = PauliSum([(pauli_string(n, {i: "X"}), 1.0) for i in range(n)])
    return zz, xs


class TestAnsatz(unittest.TestCase):
    def test_structure(self):
        for ent in ("cx", "cz"):
            c = hardware_efficient_ansatz(4, 3, ent)
            self.assertEqual(c.n_params, 4 * 4)
            self.assertEqual(sum(op.name == ent for op in c.ops), 3 * 3)
        with self.assertRaises(ValueError):
            hardware_efficient_ansatz(3, 1, "swap")

    def test_scan_path_matches_numpy_simulator(self):
        # The fast path (real vector, ladder as one permutation or phase vector, Paulis
        # as gathers) against the generic NumPy simulator running the same circuit.
        rng = np.random.default_rng(0)
        n = 5
        zz, xs = tfim_terms(n)
        mixed = PauliSum([("XYYXI", 0.4), ("YIIIY", -0.3), ("ZXZII", 0.7), ("IIYZX", 0.2), ("ZIIIZ", -0.5)])
        for ent in ("cx", "cz"):
            for layers in (0, 1, 4):
                f = make_ansatz_expectations(n, layers, [zz, xs, mixed], entangler=ent)
                for _ in range(3):
                    p = rng.uniform(0, 2 * np.pi, (layers + 1, n))
                    s = ansatz_state(p, entangler=ent)
                    expected = [o.expectation(s) for o in (zz, xs, mixed)]
                    with self.subTest(entangler=ent, layers=layers):
                        np.testing.assert_allclose(np.asarray(f(p)), expected, atol=1e-12)

    def test_matches_generic_jax_backend(self):
        rng = np.random.default_rng(1)
        n, layers = 4, 3
        h = tfim(n, 1.0, 0.7)
        for ent in ("cx", "cz"):
            generic = jb.make_energy(hardware_efficient_ansatz(n, layers, ent), h)
            zz, xs = tfim_terms(n)
            fast = make_ansatz_expectations(n, layers, [zz, xs], entangler=ent)
            p = rng.uniform(0, 2 * np.pi, (layers + 1, n))
            self.assertAlmostEqual(float(jnp.dot(jnp.array([-1.0, -0.7]), fast(p))), float(generic(p.ravel())), places=12)

    def test_gradient_matches_parameter_shift(self):
        rng = np.random.default_rng(2)
        n, layers = 3, 2
        zz, xs = tfim_terms(n)
        h = -1.0 * zz + (-0.6) * xs
        for ent in ("cx", "cz"):
            f = make_ansatz_expectations(n, layers, [zz, xs], entangler=ent)
            grad = jax.grad(lambda p: jnp.dot(jnp.array([-1.0, -0.6]), f(p)))
            p = rng.uniform(0, 2 * np.pi, (layers + 1, n))
            shifts = parameter_shift_gradient(hardware_efficient_ansatz(n, layers, ent), h, p.ravel())
            np.testing.assert_allclose(np.asarray(grad(p)).ravel(), shifts, atol=1e-11)

    def test_single_precision(self):
        rng = np.random.default_rng(3)
        n, layers = 4, 2
        zz, xs = tfim_terms(n)
        p = rng.uniform(0, 2 * np.pi, (layers + 1, n))
        lo = make_ansatz_expectations(n, layers, [zz, xs], dtype=np.complex64)(p)
        hi = make_ansatz_expectations(n, layers, [zz, xs])(p)
        self.assertEqual(lo.dtype, jnp.float32)
        np.testing.assert_allclose(np.asarray(lo), np.asarray(hi), atol=1e-5)

    def test_deep_circuits_compile(self):
        # lax.scan keeps compilation independent of depth; 64 layers must just work.
        zz, xs = tfim_terms(4)
        f = make_ansatz_expectations(4, 64, [zz, xs])
        self.assertEqual(f(np.zeros((65, 4))).shape, (2,))

    def test_validation(self):
        with self.assertRaises(ValueError):
            make_ansatz_expectations(3, 1, [PauliSum([("XX", 1.0)])])
        with self.assertRaises(ValueError):
            make_ansatz_expectations(2, 1, [PauliSum([("XY", 1j)])])
        with self.assertRaises(ValueError):
            make_ansatz_expectations(2, 1, [PauliSum([("XX", 1.0)])], entangler="iswap")


class TestAdam(unittest.TestCase):
    def test_first_step_matches_the_formula(self):
        # With bias correction, the first step is lr * g / (|g| + eps): a signed step of size lr.
        init, update = adam(learning_rate=0.1)
        p = jnp.array([1.0, -2.0, 0.5])
        g = jnp.array([3.0, -0.2, 0.0])
        new, _ = update(p, g, (*init(p)[:2], jnp.asarray(0)))
        np.testing.assert_allclose(np.asarray(new), [0.9, -1.9, 0.5], atol=1e-6)

    def test_minimizes_a_quadratic(self):
        init, update = adam(learning_rate=0.05)
        target = jnp.array([1.5, -0.7, 3.0])
        grad = jax.grad(lambda p: jnp.sum((p - target) ** 2))
        step = jax.jit(lambda p, state: update(p, grad(p), state))  # compiled, as vqe() does
        p = jnp.zeros(3)
        state = (*init(p)[:2], jnp.asarray(0))
        for _ in range(2000):
            p, state = step(p, state)
        np.testing.assert_allclose(np.asarray(p), np.asarray(target), atol=1e-3)


class TestVQE(unittest.TestCase):
    def test_reaches_the_exact_ground_energy(self):
        for n, h in [(2, 0.7), (3, 1.0), (4, 1.3)]:
            zz, xs = tfim_terms(n)
            e0 = tfim(n, 1.0, h).ground_state()[0]
            r = vqe(n, n, [zz, xs], [-1.0, -h], steps=1500, restarts=4, seed=0)
            with self.subTest(n=n, h=h):
                self.assertLess(r.energy - e0, 1e-4)
                self.assertGreaterEqual(r.energy, e0 - 1e-10)  # variational: never below the true minimum

    def test_cz_ansatz_has_a_depth_independent_ceiling(self):
        # The RY+CZ dynamical Lie algebra is small, so more layers do not help; RY+CX gets there.
        n, h = 3, 1.0
        zz, xs = tfim_terms(n)
        _, gs = tfim(n, 1.0, h).ground_state()
        fidelity = {}
        for ent, layers in [("cz", 3), ("cz", 6), ("cx", 3)]:
            r = vqe(n, layers, [zz, xs], [-1.0, -h], steps=2000, restarts=6, seed=1, entangler=ent)
            fidelity[(ent, layers)] = ansatz_state(r.params, entangler=ent).fidelity(gs)
        self.assertLess(fidelity[("cz", 3)], 0.999)
        self.assertAlmostEqual(fidelity[("cz", 3)], fidelity[("cz", 6)], places=4)
        self.assertGreater(fidelity[("cx", 3)], 0.9999)

    def test_result_shapes_and_best_restart(self):
        zz, xs = tfim_terms(3)
        r = vqe(3, 2, [zz, xs], [-1.0, -1.0], steps=50, restarts=5, seed=3)
        self.assertEqual(r.params.shape, (3, 3))
        self.assertEqual(r.history.shape, (50, 5))
        self.assertEqual(r.energy, r.final_energies.min())

    def test_shared_compilation_across_fields(self):
        n = 3
        zz, xs = tfim_terms(n)
        f = make_ansatz_expectations(n, 3, [zz, xs])
        for h in (0.5, 1.5):
            r = vqe(n, 3, [zz, xs], [-1.0, -h], steps=1500, restarts=4, expectations=f)
            self.assertLess(r.energy - tfim(n, 1.0, h).ground_state()[0], 1e-4)


class TestDynamicalLieAlgebra(unittest.TestCase):
    def test_dimensions(self):
        # CZ: n(n+1), polynomial. CX: all of so(2^n), dimension 2^(n-1) (2^n - 1).
        for n in (2, 3, 4):
            self.assertEqual(dynamical_lie_algebra_dimension(n, "cz"), n * (n + 1))
            self.assertEqual(dynamical_lie_algebra_dimension(n, "cx"), 2 ** (n - 1) * (2**n - 1))

    def test_validation(self):
        with self.assertRaises(ValueError):
            dynamical_lie_algebra_dimension(3, "cy")


if __name__ == "__main__":
    unittest.main()
