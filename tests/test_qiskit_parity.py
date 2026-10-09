"""Cross-checks against Qiskit, the reference for every convention in the package.

Comparisons are exact (up to float rounding), never up to global phase, so a
flipped qubit order or a differently phased gate definition fails loudly.
Skipped when Qiskit is not installed (``pip install qiskit``).
"""

import unittest

import numpy as np

from quantum_simulator import gates
from quantum_simulator.entanglement import entanglement_entropy, reduced_density_matrix
from quantum_simulator.observables import PauliSum
from quantum_simulator.state import State

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import library as lib
    from qiskit.quantum_info import (
        Operator,
        SparsePauliOp,
        Statevector,
        entropy,
        partial_trace,
        state_fidelity,
    )
except ImportError:  # pragma: no cover
    QuantumCircuit = None

ATOL = 1e-12


def _random_statevector(n, rng):
    vec = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return Statevector(vec / np.linalg.norm(vec))


@unittest.skipIf(QuantumCircuit is None, "qiskit not installed")
class TestGateMatricesMatchQiskit(unittest.TestCase):
    def test_constant_gates(self):
        pairs = {
            "I": lib.IGate(), "X": lib.XGate(), "Y": lib.YGate(), "Z": lib.ZGate(),
            "H": lib.HGate(), "S": lib.SGate(), "SDG": lib.SdgGate(),
            "T": lib.TGate(), "TDG": lib.TdgGate(), "SX": lib.SXGate(),
            "CX": lib.CXGate(), "CY": lib.CYGate(), "CZ": lib.CZGate(), "CH": lib.CHGate(),
            "SWAP": lib.SwapGate(), "ISWAP": lib.iSwapGate(),
        }
        for name, ref in pairs.items():
            with self.subTest(gate=name):
                np.testing.assert_allclose(getattr(gates, name), Operator(ref).data, atol=ATOL)

    def test_parametric_gates(self):
        pairs = [
            (gates.rx, lib.RXGate), (gates.ry, lib.RYGate), (gates.rz, lib.RZGate),
            (gates.phase, lib.PhaseGate), (gates.crx, lib.CRXGate), (gates.cry, lib.CRYGate),
            (gates.crz, lib.CRZGate), (gates.cphase, lib.CPhaseGate),
            (gates.rxx, lib.RXXGate), (gates.ryy, lib.RYYGate), (gates.rzz, lib.RZZGate),
        ]
        for theta in [-2.9, -0.4, 0.0, 0.7, np.pi, 5.1]:
            for ours, ref in pairs:
                with self.subTest(gate=ref.__name__, theta=theta):
                    np.testing.assert_allclose(ours(theta), Operator(ref(theta)).data, atol=ATOL)
            with self.subTest(gate="UGate", theta=theta):
                np.testing.assert_allclose(
                    gates.u(theta, 0.3, -1.2), Operator(lib.UGate(theta, 0.3, -1.2)).data, atol=ATOL
                )

    def test_controlled(self):
        np.testing.assert_allclose(gates.controlled(gates.ry(0.8)), Operator(lib.RYGate(0.8).control(1)).data, atol=ATOL)
        np.testing.assert_allclose(gates.controlled(gates.controlled(gates.X)), Operator(lib.CCXGate()).data, atol=ATOL)
        np.testing.assert_allclose(gates.controlled(gates.SWAP), Operator(lib.CSwapGate()).data, atol=ATOL)

    def test_fuse_matches_circuit_operator(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rzz(0.3, 0, 1)
        qc.s(1)
        fused = gates.fuse(
            gates.kron(gates.I, gates.H), gates.CX, gates.rzz(0.3), gates.kron(gates.S, gates.I)
        )
        np.testing.assert_allclose(fused, Operator(qc).data, atol=ATOL)


@unittest.skipIf(QuantumCircuit is None, "qiskit not installed")
class TestStatesMatchQiskit(unittest.TestCase):
    ONE_QUBIT = [
        ("h", gates.H), ("x", gates.X), ("y", gates.Y), ("z", gates.Z), ("s", gates.S),
        ("sdg", gates.SDG), ("t", gates.T), ("tdg", gates.TDG), ("sx", gates.SX),
    ]
    TWO_QUBIT = [
        ("cx", gates.CX), ("cy", gates.CY), ("cz", gates.CZ), ("ch", gates.CH),
        ("swap", gates.SWAP), ("iswap", gates.ISWAP),
    ]
    ROT_1Q = [("rx", gates.rx), ("ry", gates.ry), ("rz", gates.rz), ("p", gates.phase)]
    ROT_2Q = [("crx", gates.crx), ("cry", gates.cry), ("crz", gates.crz), ("cp", gates.cphase),
              ("rxx", gates.rxx), ("ryy", gates.ryy), ("rzz", gates.rzz)]

    def test_random_circuits(self):
        rng = np.random.default_rng(0)
        for trial in range(20):
            n = int(rng.integers(1, 7))
            qc = QuantumCircuit(n)
            ours = State.zero(n)
            for _ in range(40):
                kind = int(rng.integers(4 if n > 1 else 2))
                if kind == 0:
                    name, g = self.ONE_QUBIT[rng.integers(len(self.ONE_QUBIT))]
                    q = [int(rng.integers(n))]
                    getattr(qc, name)(*q)
                elif kind == 1:
                    name, f = self.ROT_1Q[rng.integers(len(self.ROT_1Q))]
                    theta, q = float(rng.uniform(-np.pi, np.pi)), [int(rng.integers(n))]
                    getattr(qc, name)(theta, *q)
                    g = f(theta)
                elif kind == 2:
                    name, g = self.TWO_QUBIT[rng.integers(len(self.TWO_QUBIT))]
                    q = [int(x) for x in rng.choice(n, 2, replace=False)]
                    getattr(qc, name)(*q)
                else:
                    name, f = self.ROT_2Q[rng.integers(len(self.ROT_2Q))]
                    theta = float(rng.uniform(-np.pi, np.pi))
                    q = [int(x) for x in rng.choice(n, 2, replace=False)]
                    getattr(qc, name)(theta, *q)
                    g = f(theta)
                ours.apply(g, q)
            with self.subTest(trial=trial, n=n):
                np.testing.assert_allclose(ours.vector, Statevector(qc).data, atol=1e-10)

    def test_three_qubit_gate(self):
        qc = QuantumCircuit(4)
        qc.h([0, 1, 2, 3])
        qc.ccx(3, 0, 2)
        ours = State.from_label("++++").apply(gates.controlled(gates.controlled(gates.X)), [3, 0, 2])
        np.testing.assert_allclose(ours.vector, Statevector(qc).data, atol=ATOL)

    def test_from_label(self):
        for label in ["0", "1", "01", "10", "+-", "rl", "0+1-rl", "110"]:
            with self.subTest(label=label):
                np.testing.assert_allclose(
                    State.from_label(label).vector, Statevector.from_label(label).data, atol=ATOL
                )

    def test_basis_index(self):
        for k in range(8):
            np.testing.assert_array_equal(State.basis(k, 3).vector, Statevector.from_int(k, 8).data)

    def test_probabilities(self):
        rng = np.random.default_rng(1)
        sv = _random_statevector(5, rng)
        ours = State(sv.data)
        np.testing.assert_allclose(ours.probabilities(), sv.probabilities(), atol=ATOL)
        for qargs in [[0], [4], [1, 3], [3, 1], [4, 0, 2], [2, 3, 0, 1]]:
            with self.subTest(qargs=qargs):
                np.testing.assert_allclose(ours.probabilities(qargs), sv.probabilities(qargs), atol=ATOL)

    def test_tensor(self):
        rng = np.random.default_rng(2)
        a, b = _random_statevector(2, rng), _random_statevector(3, rng)
        np.testing.assert_allclose(State(a.data).tensor(State(b.data)).vector, a.tensor(b).data, atol=ATOL)

    def test_inner_and_fidelity(self):
        rng = np.random.default_rng(3)
        a, b = _random_statevector(3, rng), _random_statevector(3, rng)
        self.assertAlmostEqual(State(a.data).inner(b.data), a.inner(b))
        self.assertAlmostEqual(State(a.data).fidelity(b.data), state_fidelity(a, b))

    def test_sample_count_keys(self):
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.h(2)
        sv = Statevector(qc)
        ours = State(sv.data).sample_counts(500, rng=0)
        self.assertEqual(set(ours), {k for k, p in sv.probabilities_dict().items() if p > 0})


@unittest.skipIf(QuantumCircuit is None, "qiskit not installed")
class TestObservablesMatchQiskit(unittest.TestCase):
    TERMS = [("ZZIII", -1.0), ("IZZII", -1.0), ("XIIII", -0.7), ("IIYXZ", 0.4), ("IIIIX", 0.25), ("YYIIZ", -0.1)]

    def test_matrix(self):
        np.testing.assert_allclose(
            PauliSum(self.TERMS).to_matrix(), SparsePauliOp.from_list(self.TERMS).to_matrix(), atol=ATOL
        )

    def test_expectation(self):
        rng = np.random.default_rng(4)
        op = SparsePauliOp.from_list(self.TERMS)
        for _ in range(5):
            sv = _random_statevector(5, rng)
            self.assertAlmostEqual(PauliSum(self.TERMS).expectation(sv.data), sv.expectation_value(op).real)

    def test_ground_state(self):
        op = SparsePauliOp.from_list(self.TERMS)
        energy, gs = PauliSum(self.TERMS).ground_state()
        self.assertAlmostEqual(energy, np.linalg.eigvalsh(op.to_matrix())[0])
        self.assertAlmostEqual(Statevector(gs.vector).expectation_value(op).real, energy)


@unittest.skipIf(QuantumCircuit is None, "qiskit not installed")
class TestPauliHelpersMatchQiskit(unittest.TestCase):
    def test_pauli_string_matches_sparse_list(self):
        from quantum_simulator.observables import pauli_string

        # Qiskit's from_sparse_list places each letter on an explicit qubit index.
        cases = [(4, {0: "Z"}), (4, {3: "X", 1: "Y"}), (5, {0: "Z", 4: "Z"}), (3, {2: "X", 1: "Z", 0: "Y"})]
        for n, ops in cases:
            with self.subTest(n=n, ops=ops):
                qubits = sorted(ops)
                ref = SparsePauliOp.from_sparse_list([("".join(ops[q] for q in qubits), qubits, 1.0)], n)
                ours = PauliSum([(pauli_string(n, ops), 1.0)])
                np.testing.assert_allclose(ours.to_matrix(), ref.to_matrix(), atol=ATOL)

    def test_tfim_matches_qiskit_construction(self):
        from quantum_simulator.observables import tfim

        n, J, h = 5, 0.8, 1.3
        bonds = [(i, (i + 1) % n) for i in range(n)]
        ref = SparsePauliOp.from_sparse_list(
            [("ZZ", list(b), -J) for b in bonds] + [("X", [i], -h) for i in range(n)], n
        )
        np.testing.assert_allclose(tfim(n, J, h, periodic=True).to_matrix(), ref.to_matrix(), atol=ATOL)

    def test_expectations_including_fast_path(self):
        from quantum_simulator.observables import tfim, z_expectation

        rng = np.random.default_rng(6)
        h = tfim(6, 1.0, 0.7, periodic=True)
        op = SparsePauliOp.from_list(list(h.terms))
        for _ in range(5):
            sv = _random_statevector(6, rng)
            self.assertAlmostEqual(h.expectation(sv.data), sv.expectation_value(op).real, places=12)
            zz = SparsePauliOp.from_sparse_list([("ZZZ", [0, 2, 5], 1.0)], 6)
            self.assertAlmostEqual(z_expectation(sv.data, [0, 2, 5]), sv.expectation_value(zz).real, places=12)


@unittest.skipIf(QuantumCircuit is None, "qiskit not installed")
class TestEntanglementMatchesQiskit(unittest.TestCase):
    def test_reduced_density_matrix_and_entropy(self):
        rng = np.random.default_rng(5)
        n = 5
        sv = _random_statevector(n, rng)
        for keep in [[0], [4], [0, 1], [1, 3], [0, 2, 4], [1, 2, 3, 4]]:
            traced = [q for q in range(n) if q not in keep]
            rho = partial_trace(sv, traced)
            with self.subTest(keep=keep):
                np.testing.assert_allclose(reduced_density_matrix(sv.data, keep), rho.data, atol=ATOL)
                self.assertAlmostEqual(entanglement_entropy(sv.data, keep), entropy(rho, base=2))

    def test_ghz(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        for q in range(1, 4):
            qc.cx(0, q)
        sv = Statevector(qc)
        self.assertAlmostEqual(entanglement_entropy(sv.data, [2]), entropy(partial_trace(sv, [0, 1, 3]), base=2))


@unittest.skipIf(QuantumCircuit is None, "qiskit not installed")
class TestCircuitsMatchQiskit(unittest.TestCase):
    """Circuit objects exported with to_qiskit run to the same state, and have the same depth."""

    def test_random_circuits(self):
        from quantum_simulator.circuit import fuse_single_qubit_gates, run
        from quantum_simulator.interop import to_qiskit
        from tests.test_circuit import random_circuit

        rng = np.random.default_rng(20)
        for trial in range(25):
            c = random_circuit(int(rng.integers(1, 7)), 40, rng)
            params = rng.uniform(-np.pi, np.pi, c.n_params)
            ours = run(c, params).state.vector
            with self.subTest(trial=trial, n=c.n):
                np.testing.assert_allclose(ours, Statevector(to_qiskit(c, params)).data, atol=1e-10)
                fused = fuse_single_qubit_gates(c)
                np.testing.assert_allclose(run(fused, params).state.vector, ours, atol=1e-10)
                np.testing.assert_allclose(Statevector(to_qiskit(fused, params)).data, ours, atol=1e-10)

    def test_symbolic_parameters(self):
        from quantum_simulator.circuit import Circuit, run
        from quantum_simulator.interop import to_qiskit

        c = Circuit(2).ry(0).rzz(0, 1).rx(1).ry(1, param=0).rz(0, 0.4)
        qc = to_qiskit(c)
        self.assertEqual(qc.num_parameters, c.n_params)
        params = [0.3, -1.1, 2.0]
        theta = sorted(qc.parameters, key=lambda p: p.index)
        bound = qc.assign_parameters(dict(zip(theta, params)))
        np.testing.assert_allclose(Statevector(bound).data, run(c, params).state.vector, atol=1e-12)

    def test_depth_matches_qiskit(self):
        from quantum_simulator.interop import to_qiskit
        from tests.test_circuit import random_circuit

        rng = np.random.default_rng(21)
        for trial in range(25):
            c = random_circuit(int(rng.integers(1, 7)), int(rng.integers(0, 40)), rng, fixed_fraction=1.0)
            if trial % 2:  # half of them also measure every qubit
                c.n_clbits = c.n
                for q in range(c.n):
                    c.measure(q, q)
            with self.subTest(trial=trial):
                self.assertEqual(c.depth(), to_qiskit(c).depth())

    def test_conditions_export_as_if_test(self):
        from quantum_simulator.circuit import Circuit
        from quantum_simulator.interop import to_qiskit

        c = Circuit(3, 2).h(1).cx(1, 2).cx(0, 1).h(0).measure(0, 0).measure(1, 1)
        c.x(2).c_if(1).z(2).c_if(0)
        ops = to_qiskit(c).count_ops()
        self.assertEqual(ops.get("if_else"), 2)
        self.assertEqual(ops.get("measure"), 2)


if __name__ == "__main__":
    unittest.main()
