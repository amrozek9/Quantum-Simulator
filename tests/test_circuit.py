"""Circuits as data (Part 3): building, validation, execution, scheduling, fusion."""

import unittest

import numpy as np

from quantum_simulator import gates
from quantum_simulator.circuit import (
    FIXED_GATES,
    PARAMETRIC_GATES,
    Circuit,
    Op,
    fuse_single_qubit_gates,
    op_matrix,
    run,
)
from quantum_simulator.entanglement import reduced_density_matrix
from quantum_simulator.state import State

ONE_QUBIT_FIXED = ["i", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx"]
TWO_QUBIT_FIXED = ["cx", "cy", "cz", "ch", "swap", "iswap"]
ONE_QUBIT_PARAM = ["rx", "ry", "rz", "p"]
TWO_QUBIT_PARAM = ["crx", "cry", "crz", "cp", "rxx", "ryy", "rzz"]


def random_circuit(n: int, n_ops: int, rng: np.random.Generator, fixed_fraction: float = 0.3) -> Circuit:
    """Random unitary circuit using every gate kind; some angles fixed, the rest parameters."""
    c = Circuit(n)
    for _ in range(n_ops):
        two = n > 1 and rng.random() < 0.4
        qubits = [int(q) for q in rng.choice(n, 2 if two else 1, replace=False)]
        if rng.random() < 0.5:
            name = str(rng.choice(TWO_QUBIT_FIXED if two else ONE_QUBIT_FIXED))
            getattr(c, name)(*qubits)
        else:
            name = str(rng.choice(TWO_QUBIT_PARAM if two else ONE_QUBIT_PARAM))
            theta = float(rng.uniform(-np.pi, np.pi)) if rng.random() < fixed_fraction else None
            getattr(c, name)(*qubits, theta)
    if n >= 3:
        c.ccx(*[int(q) for q in rng.choice(n, 3, replace=False)])
    return c


def apply_by_hand(c: Circuit, params) -> State:
    """Independent executor: look up each gate directly and apply it with State.apply."""
    s = State.zero(c.n)
    for op in c.ops:
        if op.name in FIXED_GATES:
            m = FIXED_GATES[op.name]
        else:
            m = PARAMETRIC_GATES[op.name](op.value if op.value is not None else params[op.param])
        s.apply(m, op.qubits)
    return s


class TestBuilding(unittest.TestCase):
    def test_chaining_records_ops(self):
        c = Circuit(2).h(0).cx(0, 1)
        self.assertEqual(c.ops, [Op("h", (0,)), Op("cx", (0, 1))])
        self.assertIs(Circuit(1).x(0).__class__, Circuit)

    def test_parameters_are_indices(self):
        c = Circuit(2).rx(0).ry(1).rz(0, 0.5).rzz(0, 1).rz(1, param=0)
        self.assertEqual([op.param for op in c.ops], [0, 1, None, 2, 0])
        self.assertEqual(c.ops[2].value, 0.5)
        self.assertEqual(c.n_params, 3)

    def test_cnot_alias(self):
        self.assertEqual(Circuit(2).cnot(0, 1).ops, Circuit(2).cx(0, 1).ops)

    def test_ops_are_immutable_and_hashable(self):
        op = Op("h", (0,))
        with self.assertRaises(AttributeError):
            op.name = "x"
        self.assertEqual(len({op, Op("h", (0,)), Op("h", (1,))}), 2)

    def test_validation(self):
        bad = [
            lambda: Circuit(2).h(2),                       # qubit out of range
            lambda: Circuit(2).cx(1, 1),                   # repeated qubit
            lambda: Circuit(2).append(Op("cx", (0,))),     # wrong arity
            lambda: Circuit(2).append(Op("foo", (0,))),    # unknown gate
            lambda: Circuit(2).append(Op("rx", (0,))),     # neither param nor value
            lambda: Circuit(2).append(Op("rx", (0,), param=0, value=1.0)),
            lambda: Circuit(2).append(Op("h", (0,), value=1.0)),
            lambda: Circuit(2).rx(0, 0.1, param=0),        # both angle and param
            lambda: Circuit(2).rx(0, param=0),             # param 0 doesn't exist yet
            lambda: Circuit(1, 1).measure(0, 1),           # clbit out of range
            lambda: Circuit(1).measure(0, 0),              # no classical bits
            lambda: Circuit(1, 1).x(0).c_if(1),            # condition on missing clbit
            lambda: Circuit(1, 1).x(0).c_if(0, 2),         # condition value not a bit
            lambda: Circuit(1, 1).c_if(0),                 # nothing to condition
            lambda: Circuit(2).append(Op("fused", (0,), ops=(Op("h", (1,)),))),
            lambda: Circuit(2).append(Op("fused", (0,), ops=(Op("cx", (0, 1)),))),
        ]
        for i, make in enumerate(bad):
            with self.subTest(case=i):
                with self.assertRaises(ValueError):
                    make()

    def test_failed_c_if_leaves_circuit_unchanged(self):
        c = Circuit(1, 1).x(0)
        with self.assertRaises(ValueError):
            c.c_if(5)
        self.assertEqual(c.ops, [Op("x", (0,))])


class TestRun(unittest.TestCase):
    def test_matches_applying_gates_by_hand(self):
        rng = np.random.default_rng(0)
        for trial in range(15):
            n = int(rng.integers(1, 6))
            c = random_circuit(n, 30, rng)
            params = rng.uniform(-np.pi, np.pi, c.n_params)
            with self.subTest(trial=trial, n=n):
                np.testing.assert_allclose(
                    run(c, params).state.vector, apply_by_hand(c, params).vector, atol=1e-12
                )

    def test_bell_state(self):
        r = run(Circuit(2).h(0).cx(0, 1))
        np.testing.assert_allclose(r.state.vector, [2**-0.5, 0, 0, 2**-0.5], atol=1e-15)
        self.assertEqual(r.clbits, ())

    def test_same_circuit_many_parameter_sets(self):
        c = Circuit(1).ry(0)
        for theta in [0.0, 0.4, np.pi / 2, 2.5]:
            p1 = run(c, [theta]).state.prob_one(0)
            self.assertAlmostEqual(p1, np.sin(theta / 2) ** 2)

    def test_shared_parameter(self):
        # rx(param 0) twice is rx(2 * theta).
        c = Circuit(1).rx(0).rx(0, param=0)
        np.testing.assert_allclose(run(c, [0.3]).state.vector, run(Circuit(1).rx(0, 0.6)).state.vector, atol=1e-15)

    def test_params_validated(self):
        c = Circuit(1).rx(0).ry(0)
        with self.assertRaises(ValueError):
            run(c)
        with self.assertRaises(ValueError):
            run(c, [0.1])
        with self.assertRaises(ValueError):
            run(Circuit(1).h(0), [0.1])
        run(Circuit(1).h(0), [])  # an empty vector is fine for a parameter-free circuit

    def test_dtype_and_initial_state(self):
        c = Circuit(2).h(0).cx(0, 1)
        self.assertEqual(run(c, dtype=np.complex64).state.dtype, np.complex64)
        start = State.from_label("10")
        r = run(Circuit(2).x(1), initial=start)
        np.testing.assert_array_equal(r.state.vector, State.zero(2).vector)
        np.testing.assert_array_equal(start.vector, State.from_label("10").vector)  # not modified
        with self.assertRaises(ValueError):
            run(Circuit(3).h(0), initial=start)

    def test_op_matrix(self):
        self.assertIs(op_matrix(Op("h", (0,))), gates.H)
        np.testing.assert_allclose(op_matrix(Op("rz", (0,), value=0.3)), gates.rz(0.3))
        np.testing.assert_allclose(op_matrix(Op("rz", (0,), param=1), [0.0, 0.3]), gates.rz(0.3))
        fused = Op("fused", (0,), ops=(Op("h", (0,)), Op("s", (0,))))
        np.testing.assert_allclose(op_matrix(fused), gates.S @ gates.H)  # time order H then S
        with self.assertRaises(ValueError):
            op_matrix(Op("rz", (0,), param=0))
        with self.assertRaises(ValueError):
            op_matrix(Op("measure", (0,), clbit=0))


class TestMeasurementAndClassicalControl(unittest.TestCase):
    def test_clbits_recorded(self):
        r = run(Circuit(3, 3).x(0).x(2).measure(0, 0).measure(1, 1).measure(2, 2), rng=0)
        self.assertEqual(r.clbits, (1, 0, 1))

    def test_rng_required_only_when_measuring(self):
        with self.assertRaises(TypeError):
            run(Circuit(1, 1).h(0).measure(0, 0))
        run(Circuit(1).h(0))  # no measurement, no rng needed

    def test_bell_measurements_agree_and_vary(self):
        c = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
        rng = np.random.default_rng(1)
        outcomes = {run(c, rng=rng).clbits for _ in range(100)}
        self.assertEqual(outcomes, {(0, 0), (1, 1)})

    def test_reproducible_from_seed(self):
        c = Circuit(3, 3).h(0).h(1).h(2).measure(0, 0).measure(1, 1).measure(2, 2)
        self.assertEqual([run(c, rng=s).clbits for s in range(20)], [run(c, rng=s).clbits for s in range(20)])
        self.assertGreater(len({run(c, rng=s).clbits for s in range(20)}), 1)

    def test_conditioned_gate(self):
        for prepare_one in (False, True):
            c = Circuit(2, 1)
            if prepare_one:
                c.x(0)
            c.measure(0, 0).x(1).c_if(0, 1)
            r = run(c, rng=0)
            self.assertEqual(r.clbits, (int(prepare_one),))
            self.assertEqual(r.state.prob_one(1), float(prepare_one))
        # Condition on value 0.
        r = run(Circuit(2, 1).measure(0, 0).x(1).c_if(0, 0), rng=0)
        self.assertEqual(r.state.prob_one(1), 1.0)

    def test_teleportation_on_all_four_branches(self):
        # q0 holds |psi> = RZ(phi) RY(theta)|0>; q1, q2 share a Bell pair; q2 ends in |psi>.
        rng = np.random.default_rng(2)
        for theta, phi in [(0.7, 1.9), (2.4, -0.6), (np.pi / 2, 0.0)]:
            c = Circuit(3, 2).ry(0, theta).rz(0, phi)
            c.h(1).cx(1, 2)
            c.cx(0, 1).h(0)
            c.measure(0, 0).measure(1, 1)
            c.x(2).c_if(1).z(2).c_if(0)

            psi = (State.zero(1).apply(gates.ry(theta), [0]).apply(gates.rz(phi), [0])).vector
            target = np.outer(psi, psi.conj())
            branches = set()
            for _ in range(40):
                r = run(c, rng=rng)
                branches.add(r.clbits)
                np.testing.assert_allclose(reduced_density_matrix(r.state, [2]), target, atol=1e-12)
            self.assertEqual(branches, {(0, 0), (0, 1), (1, 0), (1, 1)})


class TestStructure(unittest.TestCase):
    def test_depth_by_hand(self):
        cases = [
            (Circuit(3), 0),
            (Circuit(3).h(0).h(1).h(2), 1),
            (Circuit(2).h(0).cx(0, 1), 2),
            (Circuit(3).h(0).cx(0, 1).cx(1, 2), 3),          # GHZ, chain
            (Circuit(3).h(0).cx(0, 1).cx(0, 2), 3),          # GHZ, fan-out
            (Circuit(4).cx(0, 1).cx(2, 3).cx(1, 2), 2),      # first two run in parallel
            (Circuit(2).h(0).h(0).h(0).x(1), 3),
        ]
        for c, depth in cases:
            with self.subTest(ops=[(o.name, o.qubits) for o in c.ops]):
                self.assertEqual(c.depth(), depth)

    def test_classical_wire_creates_dependency(self):
        # Disjoint qubits, but the conditioned X must wait for the measurement.
        c = Circuit(2, 1).measure(0, 0).x(1).c_if(0)
        self.assertEqual(c.dag(), [(), (0,)])
        self.assertEqual(c.depth(), 2)
        self.assertEqual(Circuit(2, 1).measure(0, 0).x(1).depth(), 1)

    def test_dag_predecessors(self):
        c = Circuit(3).h(0).h(1).cx(0, 1).x(2).cx(1, 2)
        self.assertEqual(c.dag(), [(), (), (0, 1), (), (2, 3)])

    def test_layers_are_disjoint_and_runnable(self):
        rng = np.random.default_rng(3)
        c = random_circuit(5, 40, rng, fixed_fraction=1.0)
        layers = c.layers()
        self.assertEqual(sum(len(layer) for layer in layers), len(c.ops))
        for layer in layers:
            touched = [q for op in layer for q in op.qubits]
            self.assertEqual(len(touched), len(set(touched)))
        # Executing layer by layer gives the same state as the original order.
        relaid = Circuit(c.n)
        for layer in layers:
            for op in layer:
                relaid.append(op)
        np.testing.assert_allclose(run(relaid).state.vector, run(c).state.vector, atol=1e-12)


class TestFusion(unittest.TestCase):
    def test_merges_a_run_into_one_op(self):
        f = fuse_single_qubit_gates(Circuit(1).h(0).s(0).t(0))
        self.assertEqual(len(f.ops), 1)
        self.assertEqual(f.ops[0].name, "fused")
        self.assertEqual([o.name for o in f.ops[0].ops], ["h", "s", "t"])  # time order kept

    def test_order_is_b_times_a(self):
        # H then S is S @ H; the reverse order would give a different, still normalized, state.
        f = fuse_single_qubit_gates(Circuit(1).h(0).s(0))
        np.testing.assert_allclose(run(f).state.vector, (gates.S @ gates.H)[:, 0], atol=1e-15)
        self.assertFalse(np.allclose(run(f).state.vector, (gates.H @ gates.S)[:, 0]))

    def test_same_state_as_unfused(self):
        rng = np.random.default_rng(4)
        for trial in range(15):
            c = random_circuit(int(rng.integers(1, 6)), 40, rng)
            params = rng.uniform(-np.pi, np.pi, c.n_params)
            f = fuse_single_qubit_gates(c)
            self.assertLessEqual(len(f.ops), len(c.ops))
            self.assertEqual(f.n_params, c.n_params)
            for dtype, atol in [(np.complex128, 1e-12), (np.complex64, 1e-5)]:
                with self.subTest(trial=trial, dtype=np.dtype(dtype).name):
                    np.testing.assert_allclose(
                        run(f, params, dtype=dtype).state.vector,
                        run(c, params, dtype=dtype).state.vector,
                        atol=atol,
                    )

    def test_actually_reduces_op_count(self):
        c = Circuit(3)
        for _ in range(5):
            for q in range(3):
                c.rx(q).rz(q)
            c.cx(0, 1).cx(1, 2)
        f = fuse_single_qubit_gates(c)
        self.assertEqual(len(c.ops), 40)
        self.assertEqual(len(f.ops), 25)  # 5 x (3 fused + 2 cx)

    def test_merges_across_other_wires(self):
        f = fuse_single_qubit_gates(Circuit(2).h(0).x(1).s(0))
        names = sorted((o.name, o.qubits) for o in f.ops)
        self.assertEqual(names, [("fused", (0,)), ("x", (1,))])

    def test_does_not_cross_measurement_or_condition(self):
        f = fuse_single_qubit_gates(Circuit(1, 1).h(0).measure(0, 0).h(0))
        self.assertEqual([o.name for o in f.ops], ["h", "measure", "h"])
        f = fuse_single_qubit_gates(Circuit(1, 1).h(0).x(0).c_if(0).h(0))
        self.assertEqual([o.name for o in f.ops], ["h", "x", "h"])
        self.assertEqual(f.ops[1].condition, (0, 1))

    def test_does_not_cross_two_qubit_gate(self):
        f = fuse_single_qubit_gates(Circuit(2).h(0).cx(0, 1).h(0))
        self.assertEqual([o.name for o in f.ops], ["h", "cx", "h"])

    def test_same_measurement_results(self):
        rng = np.random.default_rng(5)
        c = random_circuit(4, 30, rng, fixed_fraction=1.0)
        c.n_clbits = 4
        for q in range(4):
            c.measure(q, q)
        f = fuse_single_qubit_gates(c)
        for seed in range(10):
            a, b = run(c, rng=seed), run(f, rng=seed)
            self.assertEqual(a.clbits, b.clbits)
            np.testing.assert_allclose(a.state.vector, b.state.vector, atol=1e-12)

    def test_idempotent(self):
        c = random_circuit(3, 30, np.random.default_rng(6))
        once = fuse_single_qubit_gates(c)
        self.assertEqual(fuse_single_qubit_gates(once).ops, once.ops)


if __name__ == "__main__":
    unittest.main()
