"""Interop: OpenQASM 3 export and import, and conversion from Qiskit."""

import dataclasses
import unittest

import numpy as np

from quantum_simulator import algorithms as alg
from quantum_simulator.circuit import Circuit, run
from quantum_simulator.interop import from_qasm3, from_qiskit, to_qasm3, to_qiskit
from tests.test_circuit import random_circuit

try:
    from qiskit import QuantumCircuit, qasm3
    from qiskit.quantum_info import Statevector
except ImportError:  # pragma: no cover
    QuantumCircuit = None

try:
    import qiskit_qasm3_import  # noqa: F401  (Qiskit's OpenQASM 3 reader needs it)
except ImportError:  # pragma: no cover
    qiskit_qasm3_import = None


def with_multi_controlled(c: Circuit) -> Circuit:
    if c.n >= 3:
        c.mcx(list(range(c.n - 1)), c.n - 1).mcz(*range(c.n))
    return c


class TestQasm3RoundTrip(unittest.TestCase):
    def test_random_circuits(self):
        rng = np.random.default_rng(0)
        for trial in range(25):
            c = with_multi_controlled(random_circuit(int(rng.integers(1, 6)), 30, rng, fixed_fraction=1.0))
            back = from_qasm3(to_qasm3(c))
            with self.subTest(trial=trial):
                self.assertEqual(back.ops, c.ops)  # exact, angles included (repr round-trips floats)
                np.testing.assert_array_equal(run(back).state.vector, run(c).state.vector)

    def test_symbolic_parameters_become_inputs(self):
        c = Circuit(2).ry(0).rzz(0, 1).rx(1, param=0)
        text = to_qasm3(c)
        self.assertIn("input float[64] theta_0;", text)
        self.assertIn("input float[64] theta_1;", text)
        back = from_qasm3(text)
        self.assertEqual(back.ops, c.ops)
        self.assertEqual(back.n_params, 2)
        np.testing.assert_allclose(run(back, [0.3, -1.1]).state.vector, run(c, [0.3, -1.1]).state.vector)
        bound = to_qasm3(c, [0.3, -1.1])
        self.assertNotIn("input", bound)
        np.testing.assert_allclose(run(from_qasm3(bound)).state.vector, run(c, [0.3, -1.1]).state.vector)

    def test_measurement_and_classical_control(self):
        tele = alg.teleportation(0.7, 1.9)
        text = to_qasm3(tele)
        self.assertIn("c[0] = measure q[0];", text)
        self.assertIn("if (c[1] == true) { x q[2]; }", text)
        self.assertEqual(from_qasm3(text).ops, tele.ops)

    def test_fused_ops_are_expanded(self):
        from quantum_simulator.circuit import fuse_single_qubit_gates

        c = Circuit(1).h(0).s(0).rx(0, 0.4)
        np.testing.assert_allclose(run(from_qasm3(to_qasm3(fuse_single_qubit_gates(c)))).state.vector,
                                   run(c).state.vector, atol=1e-15)

    def test_hand_written_program(self):
        text = """
        OPENQASM 3.0;
        include "stdgates.inc";   // standard gates
        qubit[3] q;
        bit[1] c;
        h q[0];
        rz(-pi/4) q[0];
        cp(pi / 2) q[0], q[1];
        ctrl(2) @ x q[0], q[1], q[2];
        c[0] = measure q[2];
        if (c[0]) { id q[1]; }
        """
        c = from_qasm3(text)
        self.assertEqual([op.name for op in c.ops], ["h", "rz", "cp", "mcx", "measure", "i"])
        self.assertAlmostEqual(c.ops[1].value, -np.pi / 4)
        self.assertEqual(c.ops[-1].condition, (0, 1))

    def test_rejects_what_it_cannot_read(self):
        for bad in ["qubit[2] q;\nfoo q[0];", "qubit[2] q;\nctrl(1) @ y q[0], q[1];",
                    "qubit[1] q;\nrx(__import__('os')) q[0];", "h q[0];"]:
            with self.assertRaises(ValueError):
                from_qasm3(bad)


@unittest.skipIf(QuantumCircuit is None, "qiskit not installed")
class TestAgainstQiskit(unittest.TestCase):
    @unittest.skipIf(qiskit_qasm3_import is None, "qiskit-qasm3-import not installed")
    def test_qiskit_reads_our_qasm(self):
        # Our gate definitions (rzz, rxx, ryy, iswap) are exact, so amplitudes match exactly.
        rng = np.random.default_rng(1)
        for trial in range(20):
            c = with_multi_controlled(random_circuit(int(rng.integers(1, 6)), 30, rng, fixed_fraction=1.0))
            with self.subTest(trial=trial):
                ours = run(c).state.vector
                np.testing.assert_allclose(Statevector(qasm3.loads(to_qasm3(c))).data, ours, atol=1e-12)

    def test_from_qiskit_round_trip(self):
        # An mcx with two controls is the Toffoli gate, which Qiskit names ccx; compare modulo that.
        def canonical(ops):
            return [dataclasses.replace(op, name="ccx") if op.name == "mcx" and len(op.qubits) == 3 else op
                    for op in ops]

        rng = np.random.default_rng(2)
        for trial in range(20):
            c = with_multi_controlled(random_circuit(int(rng.integers(1, 6)), 30, rng, fixed_fraction=1.0))
            back = from_qiskit(to_qiskit(c))
            with self.subTest(trial=trial):
                self.assertEqual(canonical(back.ops), canonical(c.ops))
                np.testing.assert_array_equal(run(back).state.vector, run(c).state.vector)
        tele = alg.teleportation(0.7, 1.9)
        self.assertEqual(from_qiskit(to_qiskit(tele)).ops, tele.ops)

    def test_from_a_qiskit_built_circuit(self):
        qc = QuantumCircuit(4, 1)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.mcx([0, 1, 2], 3)
        qc.rzz(0.4, 1, 3)
        qc.id(2)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], 1)):
            qc.x(2)
        c = from_qiskit(qc)
        self.assertEqual([op.name for op in c.ops], ["h", "cx", "mcx", "rzz", "i", "measure", "x"])
        self.assertEqual(c.ops[-1].condition, (0, 1))
        no_measure = QuantumCircuit(3)
        no_measure.h(0)
        no_measure.cp(0.3, 0, 2)
        no_measure.mcp(np.pi, [0, 1], 2)
        np.testing.assert_allclose(run(from_qiskit(no_measure)).state.vector, Statevector(no_measure).data, atol=1e-12)

    def test_unbound_parameters_rejected(self):
        from qiskit.circuit import Parameter

        qc = QuantumCircuit(1)
        qc.rx(Parameter("a"), 0)
        with self.assertRaises(ValueError):
            from_qiskit(qc)


if __name__ == "__main__":
    unittest.main()
