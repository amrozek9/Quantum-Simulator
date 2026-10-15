"""The alternative backends against the dense simulator: MPS, stabilizer tableau, density matrix."""

import unittest

import numpy as np

from quantum_simulator import algorithms as alg
from quantum_simulator import gates
from quantum_simulator.backends.density import (
    DensityMatrix,
    NoiseModel,
    amplitude_damping,
    bit_flip,
    depolarizing,
    is_trace_preserving,
    phase_damping,
    run_density,
    run_trajectories,
)
from quantum_simulator.backends.mps import MPS, run_mps
from quantum_simulator.backends.stabilizer import StabilizerState, run_stabilizer
from quantum_simulator.circuit import Circuit, run
from quantum_simulator.entanglement import entanglement_entropy
from quantum_simulator.observables import PauliSum, tfim
from quantum_simulator.state import State
from tests.reference import random_state
from tests.test_circuit import random_circuit


def two_qubit_circuit(rng, n, n_ops):
    """A random circuit restricted to one- and two-qubit gates (what the MPS takes)."""
    c = random_circuit(n, n_ops, rng, fixed_fraction=1.0)
    c.ops = [op for op in c.ops if len(op.qubits) <= 2]
    return c


def random_clifford(rng, n, n_ops, measure=False):
    c = Circuit(n, n if measure else 0)
    for _ in range(n_ops):
        if n > 1 and rng.random() < 0.4:
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            getattr(c, str(rng.choice(["cx", "cz", "swap"])))(a, b)
        else:
            getattr(c, str(rng.choice(["h", "s", "sdg", "x", "y", "z", "i"])))(int(rng.integers(n)))
    if measure:
        for q in range(n):
            c.measure(q, q)
    return c


class TestMPS(unittest.TestCase):
    def test_exact_without_truncation(self):
        rng = np.random.default_rng(0)
        for trial in range(25):
            c = two_qubit_circuit(rng, int(rng.integers(1, 7)), 40)
            with self.subTest(trial=trial, n=c.n):
                m = run_mps(c)
                np.testing.assert_allclose(m.to_vector(), run(c).state.vector, atol=1e-12)
                self.assertAlmostEqual(m.fidelity_estimate(), 1.0)

    def test_distant_two_qubit_gates_route_through_swaps(self):
        # CX with control 0 and target 5, both directions, on a superposition.
        for qubits in ([0, 5], [5, 0], [4, 1]):
            c = Circuit(6).h(0).h(4).h(5).cx(*qubits).ry(2, 0.3)
            np.testing.assert_allclose(run_mps(c).to_vector(), run(c).state.vector, atol=1e-13)

    def test_from_vector_and_schmidt_values(self):
        rng = np.random.default_rng(1)
        vec = random_state(7, rng)
        m = MPS.from_vector(vec)
        np.testing.assert_allclose(m.to_vector(), vec, atol=1e-13)
        self.assertEqual(m.bond_dimensions, [2, 4, 8, 8, 4, 2])
        for cut in range(1, 7):
            self.assertAlmostEqual(m.entanglement_entropy(cut), entanglement_entropy(vec, range(cut)), places=10)

    def test_product_and_ghz_bonds(self):
        self.assertEqual(run_mps(Circuit(8).h(0).h(3).ry(7, 0.4)).bond_dimensions, [1] * 7)
        ghz_chain = Circuit(8).h(0)
        for q in range(7):
            ghz_chain.cx(q, q + 1)
        m = run_mps(ghz_chain)
        self.assertEqual(m.bond_dimensions, [2] * 7)  # one bit of entanglement across every cut
        self.assertAlmostEqual(m.entanglement_entropy(4), 1.0)

    def test_truncation_estimate_and_error(self):
        # A deep random circuit needs bond 8 at the middle of 6 qubits; cap it at 2.
        rng = np.random.default_rng(2)
        c = two_qubit_circuit(rng, 6, 80)
        exact = run(c).state.vector
        m = run_mps(c, max_bond=2)
        self.assertLessEqual(max(m.bond_dimensions), 2)
        fidelity = abs(np.vdot(exact, m.to_vector())) ** 2
        self.assertLess(fidelity, 0.99)
        self.assertGreater(len(m.discarded), 0)
        self.assertAlmostEqual(np.linalg.norm(m.to_vector()), 1.0, places=10)
        # More bond dimension, better fidelity; enough of it, exact.
        fids = [abs(np.vdot(exact, run_mps(c, max_bond=chi).to_vector())) ** 2 for chi in (2, 4, 8)]
        self.assertLess(fids[0], fids[1])
        self.assertAlmostEqual(fids[2], 1.0, places=10)

    def test_validation(self):
        with self.assertRaises(ValueError):
            run_mps(Circuit(3).ccx(0, 1, 2))
        with self.assertRaises(ValueError):
            run_mps(Circuit(1, 1).measure(0, 0))
        with self.assertRaises(ValueError):
            MPS(3).apply(gates.CX, [0, 0])


class TestStabilizer(unittest.TestCase):
    def test_matches_dense_on_random_clifford_circuits(self):
        rng = np.random.default_rng(3)
        for trial in range(100):
            c = random_clifford(rng, int(rng.integers(1, 6)), 30)
            st, _ = run_stabilizer(c)
            with self.subTest(trial=trial):
                self.assertAlmostEqual(st.to_state().fidelity(run(c).state), 1.0, places=10)

    def test_deterministic_outcomes_match_dense(self):
        rng = np.random.default_rng(4)
        for trial in range(60):
            c = random_clifford(rng, 4, 15)
            probs = [run(c).state.prob_one(q) for q in range(4)]
            for p in probs:  # a stabilizer state's single-qubit marginals are 0, 1/2 or 1
                self.assertTrue(min(abs(p), abs(p - 0.5), abs(p - 1)) < 1e-9)
            c.n_clbits = 4
            for q in range(4):
                c.measure(q, q)
            _, bits = run_stabilizer(c, rng=int(rng.integers(1 << 30)))
            for q, p in enumerate(probs):
                if abs(p - 0.5) > 0.25:
                    self.assertEqual(bits[q], round(p))

    def test_bell_and_ghz_correlations(self):
        rng = np.random.default_rng(5)
        outcomes = set()
        for _ in range(40):
            c = alg.ghz(5)
            c.n_clbits = 5
            for q in range(5):
                c.measure(q, q)
            _, bits = run_stabilizer(c, rng=rng)
            self.assertEqual(len(set(bits)), 1)
            outcomes.add(bits[0])
        self.assertEqual(outcomes, {0, 1})

    def test_thousand_qubits(self):
        c = Circuit(1000, 1000).compose(alg.ghz(1000))
        for q in range(1000):
            c.measure(q, q)
        _, bits = run_stabilizer(c, rng=0)
        self.assertEqual(len(set(bits)), 1)

    def test_teleportation_of_stabilizer_states(self):
        # Teleport |0>, |1>, |+>, |->, |+i>: Clifford preparations, Clifford protocol.
        preps = {"0": [], "1": ["x"], "+": ["h"], "-": ["x", "h"], "r": ["h", "s"]}
        rng = np.random.default_rng(6)
        for label, prep in preps.items():
            for _ in range(8):
                c = Circuit(3, 2)
                for g in prep:
                    getattr(c, g)(0)
                c.h(1).cx(1, 2).cx(0, 1).h(0).measure(0, 0).measure(1, 1)
                c.x(2).c_if(1).z(2).c_if(0)
                st, _ = run_stabilizer(c, rng=rng)
                # Qubit 2 alone must carry the input: its stabilizer is the input's.
                target = State.from_label(label)
                state = st.to_state()
                rho2 = np.einsum("ijk,ljk->il", state.psi.reshape(2, 2, 2), state.psi.reshape(2, 2, 2).conj())
                self.assertAlmostEqual(np.real(target.vector.conj() @ rho2 @ target.vector), 1.0, places=10)

    def test_stabilizers_of_bell_pair(self):
        st, _ = run_stabilizer(alg.bell_pair())
        self.assertEqual(sorted(st.stabilizers()), [(1, "XX"), (1, "ZZ")])

    def test_rejects_non_clifford(self):
        for c in (Circuit(1).t(0), Circuit(1).rx(0, 0.3), Circuit(3).ccx(0, 1, 2)):
            with self.assertRaises(ValueError):
                run_stabilizer(c)


class TestDensityMatrix(unittest.TestCase):
    def test_noiseless_matches_pure_state(self):
        rng = np.random.default_rng(7)
        for _ in range(15):
            c = random_circuit(int(rng.integers(1, 5)), 25, rng, fixed_fraction=1.0)
            v = run(c).state.vector
            rho = run_density(c)
            np.testing.assert_allclose(rho.matrix(), np.outer(v, v.conj()), atol=1e-12)
            self.assertAlmostEqual(rho.purity(), 1.0)

    def test_expectation_matches_pure_state(self):
        rng = np.random.default_rng(8)
        h = tfim(3, 1.0, 0.7) + PauliSum([("XYZ", 0.3), ("IZY", -0.4)])
        c = random_circuit(3, 20, rng, fixed_fraction=1.0)
        self.assertAlmostEqual(run_density(c).expectation(h), h.expectation(run(c).state), places=12)

    def test_channels_are_trace_preserving_and_correct(self):
        for kraus in (depolarizing(0.3), amplitude_damping(0.4), phase_damping(0.5), bit_flip(0.2)):
            self.assertTrue(is_trace_preserving(kraus))
        # Closed forms on one qubit.
        plus = DensityMatrix.from_state(State.from_label("+"))
        x = PauliSum([("X", 1.0)])
        self.assertAlmostEqual(DensityMatrix(plus.rho.copy()).apply_channel(depolarizing(0.3), [0]).expectation(x), 0.7)
        self.assertAlmostEqual(DensityMatrix(plus.rho.copy()).apply_channel(phase_damping(0.36), [0]).expectation(x), 0.8)
        one = DensityMatrix.from_state(State.from_label("1"))
        self.assertAlmostEqual(one.apply_channel(amplitude_damping(0.25), [0]).probabilities()[0], 0.25)
        mixed = DensityMatrix.from_state(State.from_label("0")).apply_channel(depolarizing(1.0), [0])
        np.testing.assert_allclose(mixed.matrix(), np.eye(2) / 2, atol=1e-15)
        self.assertAlmostEqual(mixed.von_neumann_entropy(), 1.0)

    def test_noise_mixes_the_state(self):
        g = alg.ghz(4)
        rho = run_density(g, noise=NoiseModel(depolarizing(0.05)))
        self.assertAlmostEqual(rho.trace(), 1.0, places=12)
        self.assertLess(rho.purity(), 0.9)
        self.assertLess(rho.fidelity_with(run(g).state), 0.9)
        self.assertGreater(rho.von_neumann_entropy(), 0.1)

    def test_trajectories_converge_to_the_density_matrix(self):
        # Agreement within 4 standard errors, for a mixed-unitary channel and a non-unitary one.
        g = alg.ghz(3)
        noise = NoiseModel(depolarizing(0.08))
        rho = run_density(g, noise=noise)
        for label in ("ZZI", "XXX"):
            obs = PauliSum([(label, 1.0)])
            mean, err = run_trajectories(g, obs, 1500, rng=11, noise=noise)
            with self.subTest(observable=label):
                self.assertLess(abs(mean - rho.expectation(obs)), 4 * err)
        amp = NoiseModel(amplitude_damping(0.15))
        obs = PauliSum([("ZZZ", 1.0)])
        mean, err = run_trajectories(g, obs, 1500, rng=12, noise=amp)
        self.assertLess(abs(mean - run_density(g, noise=amp).expectation(obs)), 4 * err)

    def test_validation(self):
        with self.assertRaises(ValueError):
            depolarizing(1.5)
        with self.assertRaises(ValueError):
            NoiseModel([np.eye(2), np.eye(2)])
        with self.assertRaises(ValueError):
            DensityMatrix(np.eye(3))


if __name__ == "__main__":
    unittest.main()
