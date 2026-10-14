"""Differentiable simulation: JAX autodiff against the parameter-shift rule.

The central test compares two independent derivations of the same gradient,
computed by two independent simulators: reverse-mode autodiff through the
JAX backend, and parameter shifts evaluated with the NumPy simulator.
Parameter-shift tests that do not need JAX run without it.
"""

import unittest

import numpy as np

from quantum_simulator import gates
from quantum_simulator.circuit import PARAMETRIC_GATES, Circuit, fuse_single_qubit_gates, run
from quantum_simulator.gradients import (
    FOUR_TERM,
    TWO_TERM,
    finite_difference_gradient,
    parameter_shift_gradient,
    two_term_shift_gradient,
)
from quantum_simulator.observables import PauliSum, pauli, tfim
from tests.test_circuit import random_circuit

try:
    import jax.numpy as jnp

    from quantum_simulator import jax_backend as jb
except ImportError:  # pragma: no cover
    jb = None

needs_jax = unittest.skipIf(jb is None, "jax not installed")


def energy_numpy(circuit, h, params):
    return h.expectation(run(circuit, params if circuit.n_params else None).state)


def mixed_hamiltonian(n: int) -> PauliSum:
    """TFIM plus terms with X and Y, so both the diagonal and the gate paths are exercised.

    The single Z on qubit 0 breaks the mirror symmetry of the other diagonal
    terms, so reversing the qubit order would change the energy.
    """
    extra = PauliSum([
        ("Y" + "X" * (n - 1), 0.3), ("Z" * n, -0.2), ("X" + "I" * (n - 2) + "Y", 0.25),
        ("I" * (n - 1) + "Z", 0.45),
    ])
    return tfim(n, J=1.0, h=0.6) + extra


def every_parametric_gate_circuit() -> Circuit:
    """Each parameterized gate at least once, plus a parameter shared by two gates."""
    return (
        Circuit(3).ry(0).rx(1).rz(2).h(0).crx(0, 1).cry(1, 2).crz(2, 0).cp(0, 1).p(2)
        .rxx(0, 1).ryy(1, 2).rzz(0, 2).ry(1, param=0).cx(2, 0)
    )


class TestShiftRules(unittest.TestCase):
    """NumPy only: the shift rules against finite differences and closed forms."""

    def test_rule_tables_cover_every_parametric_gate(self):
        self.assertEqual(TWO_TERM | FOUR_TERM, set(PARAMETRIC_GATES))
        self.assertFalse(TWO_TERM & FOUR_TERM)

    def test_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        h = mixed_hamiltonian(3)
        c = every_parametric_gate_circuit()
        for _ in range(3):
            params = rng.uniform(-np.pi, np.pi, c.n_params)
            fd = finite_difference_gradient(lambda p: energy_numpy(c, h, p), params)
            np.testing.assert_allclose(parameter_shift_gradient(c, h, params), fd, atol=1e-7)

    def test_controlled_rotation_closed_forms(self):
        # H on the control (qubit 0), then CRX(theta) onto the target (qubit 1).
        # The state is (|0>|0> + |1> RX(theta)|0>) / sqrt(2), which gives two closed forms:
        #   <Z_target>  = (1 + cos theta) / 2: frequency 1 only, so two terms happen to suffice;
        #   <X_control> = cos(theta / 2): frequency 1/2, from the coherence between the
        #                 control's branches, which only the four-term rule handles.
        c = Circuit(2).h(0).crx(0, 1)
        z_target, x_control = pauli("ZI"), pauli("IX")
        for theta in [0.4, 1.3, 2.9, -2.0]:
            with self.subTest(theta=theta):
                self.assertAlmostEqual(energy_numpy(c, z_target, [theta]), (1 + np.cos(theta)) / 2, places=12)
                self.assertAlmostEqual(energy_numpy(c, x_control, [theta]), np.cos(theta / 2), places=12)

                d_z, d_x = -np.sin(theta) / 2, -np.sin(theta / 2) / 2
                self.assertAlmostEqual(parameter_shift_gradient(c, z_target, [theta])[0], d_z, places=12)
                self.assertAlmostEqual(parameter_shift_gradient(c, x_control, [theta])[0], d_x, places=12)
                self.assertAlmostEqual(two_term_shift_gradient(c, z_target, [theta])[0], d_z, places=12)
                # Two terms give -sin(theta/2)/sqrt(2) here instead of -sin(theta/2)/2.
                self.assertAlmostEqual(two_term_shift_gradient(c, x_control, [theta])[0],
                                       -np.sin(theta / 2) / np.sqrt(2), places=12)

    def test_two_term_rule_fails_only_where_expected(self):
        # One gate at a time, each preceded by Hadamards so the gradient is nonzero.
        rng = np.random.default_rng(1)
        h = mixed_hamiltonian(2)
        for name in sorted(PARAMETRIC_GATES):
            c = Circuit(2).h(0).h(1).ry(0, 0.7).ry(1, -0.4)
            qubits = (0,) if name in ("rx", "ry", "rz", "p") else (0, 1)
            getattr(c, name)(*qubits)
            c.rx(0, 0.3).ry(1, 1.1)
            params = rng.uniform(-np.pi, np.pi, 1)
            fd = finite_difference_gradient(lambda p: energy_numpy(c, h, p), params)
            two = two_term_shift_gradient(c, h, params)
            with self.subTest(gate=name):
                self.assertGreater(abs(fd[0]), 1e-3)  # a meaningful test needs a nonzero gradient
                np.testing.assert_allclose(parameter_shift_gradient(c, h, params), fd, atol=1e-7)
                if name in TWO_TERM:
                    np.testing.assert_allclose(two, fd, atol=1e-7)
                else:
                    self.assertGreater(abs(two[0] - fd[0]), 1e-3)

    def test_shared_parameter_sums_over_gates(self):
        # ry(theta) twice on one qubit is ry(2 theta): <Z> = cos(2 theta), derivative -2 sin(2 theta).
        c = Circuit(1).ry(0).ry(0, param=0)
        for theta in [0.3, 1.0]:
            self.assertAlmostEqual(parameter_shift_gradient(c, pauli("Z"), [theta])[0], -2 * np.sin(2 * theta), places=12)

    def test_fixed_angles_have_no_gradient_and_fusion_is_transparent(self):
        rng = np.random.default_rng(2)
        c = Circuit(2).ry(0).rz(0, 0.5).rx(0).cx(0, 1).ry(1).rz(1, -0.2)
        h = mixed_hamiltonian(2)
        params = rng.uniform(-np.pi, np.pi, c.n_params)
        np.testing.assert_allclose(
            parameter_shift_gradient(fuse_single_qubit_gates(c), h, params),
            parameter_shift_gradient(c, h, params),
            atol=1e-12,
        )


@needs_jax
class TestJaxMatchesNumpy(unittest.TestCase):
    def test_gate_matrices(self):
        for name, f in PARAMETRIC_GATES.items():
            for theta in [-2.7, 0.0, 0.9, np.pi]:
                with self.subTest(gate=name, theta=theta):
                    np.testing.assert_allclose(np.asarray(jb.JAX_PARAMETRIC[name](theta)), f(theta), atol=1e-14)

    def test_states(self):
        rng = np.random.default_rng(3)
        for trial in range(10):
            c = random_circuit(int(rng.integers(1, 6)), 30, rng)
            params = rng.uniform(-np.pi, np.pi, c.n_params)
            expected = run(c, params).state.psi
            with self.subTest(trial=trial):
                np.testing.assert_allclose(np.asarray(jb.make_state(c)(params)), expected, atol=1e-12)
                fused = fuse_single_qubit_gates(c)
                np.testing.assert_allclose(np.asarray(jb.make_state(fused)(params)), expected, atol=1e-12)

    def test_multi_controlled_gates_and_algorithm_circuits(self):
        from quantum_simulator import algorithms as alg

        for c in [alg.grover(4, 0b1011), alg.phase_estimation(0.3, 4), Circuit(4).h(0).h(1).mcx([0, 1, 3], 2)]:
            with self.subTest(n=c.n, ops=len(c.ops)):
                np.testing.assert_allclose(np.asarray(jb.make_state(c)([])), run(c).state.psi, atol=1e-12)

    def test_energies(self):
        rng = np.random.default_rng(4)
        for n in (2, 3, 4):
            c = random_circuit(n, 25, rng)
            h = mixed_hamiltonian(n)
            energy = jb.make_energy(c, h)
            for _ in range(3):
                params = rng.uniform(-np.pi, np.pi, c.n_params)
                self.assertAlmostEqual(float(energy(params)), energy_numpy(c, h, params), places=12)

    def test_complex64(self):
        rng = np.random.default_rng(5)
        c = every_parametric_gate_circuit()
        h = mixed_hamiltonian(3)
        params = rng.uniform(-np.pi, np.pi, c.n_params)
        self.assertEqual(jb.make_state(c, np.complex64)(params).dtype, jnp.complex64)
        value, grad = jb.make_value_and_grad(c, h, np.complex64)(params.astype(np.float32))
        self.assertEqual(value.dtype, jnp.float32)
        np.testing.assert_allclose(np.asarray(grad), parameter_shift_gradient(c, h, params), atol=1e-4)


@needs_jax
class TestAutodiffMatchesParameterShift(unittest.TestCase):
    """Two independent derivations of the same gradient, on two independent simulators."""

    def test_every_parametric_gate_and_a_shared_parameter(self):
        rng = np.random.default_rng(6)
        c = every_parametric_gate_circuit()
        h = mixed_hamiltonian(3)
        value_and_grad = jb.make_value_and_grad(c, h)
        for _ in range(5):
            params = rng.uniform(-np.pi, np.pi, c.n_params)
            value, grad = value_and_grad(params)
            self.assertAlmostEqual(float(value), energy_numpy(c, h, params), places=12)
            np.testing.assert_allclose(np.asarray(grad), parameter_shift_gradient(c, h, params), atol=1e-11)

    def test_random_circuits(self):
        rng = np.random.default_rng(7)
        for trial in range(8):
            n = int(rng.integers(2, 5))
            c = random_circuit(n, 20, rng)
            if c.n_params == 0:
                continue
            h = mixed_hamiltonian(n)
            params = rng.uniform(-np.pi, np.pi, c.n_params)
            _, grad = jb.make_value_and_grad(c, h)(params)
            with self.subTest(trial=trial, n=n, n_params=c.n_params):
                np.testing.assert_allclose(np.asarray(grad), parameter_shift_gradient(c, h, params), atol=1e-11)

    def test_gradient_points_downhill(self):
        # A small step against the gradient lowers the energy.
        rng = np.random.default_rng(8)
        c = Circuit(4)
        for layer in range(3):
            for q in range(4):
                c.ry(q)
            for q in range(3):
                c.cx(q, q + 1)
        h = tfim(4, h=1.0)
        value_and_grad = jb.make_value_and_grad(c, h)
        params = rng.uniform(-np.pi, np.pi, c.n_params)
        e0, g = value_and_grad(params)
        e1, _ = value_and_grad(params - 1e-3 * np.asarray(g))
        self.assertLess(float(e1), float(e0))


@needs_jax
class TestCompilation(unittest.TestCase):
    def test_traced_once_for_many_parameter_sets(self):
        # The circuit is fixed structure; only the parameter array changes, so one trace serves all calls.
        import jax

        c = Circuit(3).ry(0).ry(1).ry(2).cx(0, 1).cx(1, 2).rz(2)
        h = tfim(3)
        inner = jb.make_energy(c, h, jit=False)
        traces = []

        def counted(p):
            traces.append(1)
            return inner(p)

        energy = jax.jit(counted)
        rng = np.random.default_rng(9)
        for _ in range(5):
            energy(rng.uniform(-np.pi, np.pi, c.n_params))
        self.assertEqual(len(traces), 1)

    def test_rejects_non_unitary_circuits_and_bad_observables(self):
        with self.assertRaises(ValueError):
            jb.make_state(Circuit(1, 1).h(0).measure(0, 0))
        with self.assertRaises(ValueError):
            jb.make_state(Circuit(2, 1).x(1).c_if(0))
        with self.assertRaises(ValueError):
            jb.make_energy(Circuit(1).rx(0), pauli("X", 1j))
        with self.assertRaises(ValueError):
            jb.make_energy(Circuit(2).rx(0), pauli("Z"))

    def test_list_parameters_give_an_array_gradient(self):
        c = Circuit(2).ry(0).ry(1).cx(0, 1)
        h = tfim(2)
        value, grad = jb.make_value_and_grad(c, h)([0.3, 1.2])
        self.assertEqual(grad.shape, (2,))
        np.testing.assert_allclose(np.asarray(grad), parameter_shift_gradient(c, h, [0.3, 1.2]), atol=1e-12)
        self.assertAlmostEqual(float(jb.make_energy(c, h)([0.3, 1.2])), float(value), places=14)

    def test_apply_gate_is_pure(self):
        psi = jnp.zeros((2, 2), dtype=jnp.complex128).at[0, 0].set(1.0)
        out = jb.apply_gate(psi, jnp.asarray(gates.H), [0])
        self.assertEqual(float(psi[0, 0].real), 1.0)  # input untouched
        np.testing.assert_allclose(np.asarray(out).reshape(-1), [2**-0.5, 2**-0.5, 0, 0], atol=1e-15)


if __name__ == "__main__":
    unittest.main()
