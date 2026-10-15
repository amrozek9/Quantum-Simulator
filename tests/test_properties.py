"""Property-based tests: random circuits against two independent oracles.

Hypothesis generates circuits with every gate kind on up to 6 qubits and up
to 30 ops. Each is checked against:

* Qiskit, the external reference whose conventions this package matches;
* deliberately naive simulators that build the full ``2^n x 2^n`` matrix of
  every op, one by looping over bit patterns and one with ``np.kron``
  (``tests/reference.py``). They share no code with the tensor simulator,
  and catch the case where this package and Qiskit share a misunderstanding.

When a property fails, Hypothesis shrinks the circuit to a minimal failing
one; the last test here plants a bug and checks that the shrunk circuit is
tiny. Run more examples with ``HYPOTHESIS_PROFILE=thorough``.
"""

import unittest

import numpy as np

try:
    from hypothesis import find, given
    from hypothesis import settings as hsettings

    from tests.strategies import circuits
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("hypothesis not installed") from None

from quantum_simulator.circuit import Circuit, fuse_single_qubit_gates, op_matrix, run
from quantum_simulator.state import State, num_qubits, qubit_axis
from tests.reference import dense_operator, kron_operator

try:
    from qiskit.quantum_info import Statevector

    from quantum_simulator.interop import to_qiskit
except ImportError:  # pragma: no cover
    Statevector = None


def naive_run(c: Circuit) -> np.ndarray:
    """Multiply full 2^n x 2^n matrices, one per op, built by looping over bit patterns."""
    vec = np.zeros(2**c.n, dtype=complex)
    vec[0] = 1
    for op in c.ops:
        vec = dense_operator(op_matrix(op), list(op.qubits), c.n) @ vec
    return vec


def kron_run(c: Circuit) -> np.ndarray:
    """Multiply full matrices built with np.kron (Kronecker chains around each gate)."""
    vec = np.zeros(2**c.n, dtype=complex)
    vec[0] = 1
    for op in c.ops:
        vec = kron_operator(op_matrix(op), list(op.qubits), c.n) @ vec
    return vec


class TestAgainstNaiveSimulator(unittest.TestCase):
    @given(circuits(max_qubits=5, max_depth=25))
    def test_matches_full_matrix_simulation(self, c):
        np.testing.assert_allclose(run(c).state.vector, naive_run(c), atol=1e-10)

    @given(circuits(max_qubits=5, max_depth=25))
    def test_matches_kronecker_product_simulation(self, c):
        np.testing.assert_allclose(run(c).state.vector, kron_run(c), atol=1e-10)


@unittest.skipIf(Statevector is None, "qiskit not installed")
class TestAgainstQiskit(unittest.TestCase):
    @given(circuits(max_qubits=6, max_depth=30))
    def test_matches_qiskit(self, c):
        ours = run(c).state.vector
        theirs = Statevector(to_qiskit(c)).data
        # The physical check: global phase is unobservable, so compare fidelity...
        self.assertAlmostEqual(abs(np.vdot(ours, theirs)), 1.0, places=10)
        # ...and the stricter one this package can afford, because its gate
        # definitions match Qiskit's exactly, including global phases.
        np.testing.assert_allclose(ours, theirs, atol=1e-10)


class TestInvariants(unittest.TestCase):
    @given(circuits())
    def test_norm_is_preserved(self, c):
        self.assertAlmostEqual(run(c).state.norm(), 1.0, places=12)

    @given(circuits())
    def test_fusion_preserves_the_state(self, c):
        np.testing.assert_allclose(run(fuse_single_qubit_gates(c)).state.vector, run(c).state.vector, atol=1e-10)

    @given(circuits(invertible=True))
    def test_inverse_undoes_the_circuit(self, c):
        back = run(c.inverse(), initial=run(c).state).state
        self.assertAlmostEqual(back.fidelity(State.zero(c.n)), 1.0, places=10)

    @given(circuits())
    def test_depth_is_at_most_the_op_count(self, c):
        self.assertLessEqual(c.depth(), len(c.ops))
        self.assertEqual(sum(len(layer) for layer in c.layers()), len(c.ops))


class TestShrinking(unittest.TestCase):
    def test_planted_bug_shrinks_to_a_minimal_circuit(self):
        # The most common bug in a tensor simulator: contracting a gate into an axis
        # and forgetting to move the new axis back (tensordot leaves it at axis 0).
        # The state stays normalized, so only a comparison catches it.
        def apply_without_moveaxis(psi, gate, qubits):
            n, k = num_qubits(psi), len(qubits)
            axes = [qubit_axis(q, n) for q in reversed(qubits)]
            return np.tensordot(np.asarray(gate).reshape((2,) * (2 * k)), psi, axes=(range(k, 2 * k), axes))

        def broken_run(c: Circuit) -> np.ndarray:
            psi = State.zero(c.n).psi
            for op in c.ops:
                psi = apply_without_moveaxis(psi, op_matrix(op), op.qubits)
            return psi.reshape(-1)

        minimal = find(
            circuits(),
            lambda c: not np.allclose(broken_run(c), run(c).state.vector, atol=1e-10),
            settings=hsettings(database=None, derandomize=True, max_examples=500),
        )
        # Hypothesis reduces whatever large failing circuit it found first to the
        # essence of the bug: one single-qubit gate on a two-qubit register,
        # acting on any qubit other than the highest.
        self.assertEqual(minimal.n, 2)
        self.assertEqual(len(minimal.ops), 1)
        self.assertEqual(minimal.ops[0].qubits, (0,))


if __name__ == "__main__":
    unittest.main()
