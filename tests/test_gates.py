import unittest

import numpy as np

from quantum_simulator import gates
from quantum_simulator.linalg import equal_up_to_global_phase, is_unitary, kron
from quantum_simulator.state import State, apply_gate, from_vector, to_vector
from tests.reference import dense_operator, random_state

CONSTANTS = ["I", "X", "Y", "Z", "H", "S", "SDG", "T", "TDG", "SX",
             "CX", "CY", "CZ", "CH", "SWAP", "ISWAP"]
PARAMETRIC_1Q = [gates.rx, gates.ry, gates.rz, gates.phase]
PARAMETRIC_2Q = [gates.crx, gates.cry, gates.crz, gates.cphase, gates.rxx, gates.ryy, gates.rzz]


class TestGateMatrices(unittest.TestCase):
    def test_all_unitary(self):
        for name in CONSTANTS:
            with self.subTest(gate=name):
                self.assertTrue(is_unitary(getattr(gates, name)))
        for theta in np.linspace(-2 * np.pi, 2 * np.pi, 7):
            for f in PARAMETRIC_1Q + PARAMETRIC_2Q:
                with self.subTest(gate=f.__name__, theta=theta):
                    self.assertTrue(is_unitary(f(theta)))
            self.assertTrue(is_unitary(gates.u(theta, 0.3, -1.1)))

    def test_constants_read_only(self):
        with self.assertRaises(ValueError):
            gates.X[0, 0] = 5

    def test_known_identities(self):
        np.testing.assert_allclose(gates.rx(np.pi), -1j * gates.X, atol=1e-15)
        np.testing.assert_allclose(gates.ry(np.pi), -1j * gates.Y, atol=1e-15)
        np.testing.assert_allclose(gates.rz(np.pi), -1j * gates.Z, atol=1e-15)
        np.testing.assert_allclose(gates.SX @ gates.SX, gates.X, atol=1e-15)
        np.testing.assert_allclose(gates.T @ gates.T, gates.S, atol=1e-15)
        np.testing.assert_allclose(gates.H @ gates.Z @ gates.H, gates.X, atol=1e-15)
        self.assertTrue(equal_up_to_global_phase(gates.rz(0.4), gates.phase(0.4)))
        np.testing.assert_allclose(gates.u(np.pi / 2, 0, np.pi), gates.H, atol=1e-15)

    def test_two_qubit_rotations_are_exponentials(self):
        theta = 0.37
        for f, p in [(gates.rxx, gates.X), (gates.ryy, gates.Y), (gates.rzz, gates.Z)]:
            w, v = np.linalg.eigh(kron(p, p))
            expected = v @ np.diag(np.exp(-0.5j * theta * w)) @ v.conj().T
            np.testing.assert_allclose(f(theta), expected, atol=1e-14)


class TestConventions(unittest.TestCase):
    def test_cx_control_is_first_qubit(self):
        # qubit 0 = 1 (control), qubit 1 = 0 -> CX on [0, 1] flips qubit 1 -> |11>.
        out = State.from_label("01").apply(gates.CX, [0, 1])
        np.testing.assert_array_equal(out.vector, State.from_label("11").vector)
        # Control on qubit 1 (which is 0): nothing happens.
        out = State.from_label("01").apply(gates.CX, [1, 0])
        np.testing.assert_array_equal(out.vector, State.from_label("01").vector)

    def test_cx_matrix_is_qiskit_form(self):
        expected = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
        np.testing.assert_array_equal(gates.CX, expected)

    def test_controlled_builds_cx(self):
        np.testing.assert_array_equal(gates.controlled(gates.X), gates.CX)
        # Doubly-controlled X via nesting: Toffoli on [c0, c1, target].
        ccx = gates.controlled(gates.controlled(gates.X))
        for k in range(8):
            out = to_vector(apply_gate(State.basis(k, 3).psi, ccx, [0, 1, 2]))
            flip = 4 if (k & 0b011) == 0b011 else 0
            self.assertEqual(np.argmax(np.abs(out)), k ^ flip)

    def test_swap(self):
        rng = np.random.default_rng(0)
        a, b = State(random_state(1, rng)), State(random_state(1, rng))
        out = a.tensor(b).apply(gates.SWAP, [0, 1])
        np.testing.assert_allclose(out.vector, b.tensor(a).vector, atol=1e-15)


class TestFusion(unittest.TestCase):
    def test_order_reverses(self):
        np.testing.assert_allclose(gates.fuse(gates.H, gates.S), gates.S @ gates.H)
        self.assertFalse(np.allclose(gates.fuse(gates.H, gates.S), gates.fuse(gates.S, gates.H)))

    def test_fused_equals_sequential(self):
        rng = np.random.default_rng(1)
        seq = [gates.CX, gates.rzz(0.3), gates.CH, gates.ISWAP]
        vec = random_state(3, rng)
        psi = from_vector(vec)
        for g in seq:
            psi = apply_gate(psi, g, [2, 0])
        fused = apply_gate(from_vector(vec), gates.fuse(*seq), [2, 0])
        np.testing.assert_allclose(to_vector(psi), to_vector(fused), atol=1e-14)
        np.testing.assert_allclose(
            to_vector(fused), dense_operator(gates.fuse(*seq), [2, 0], 3) @ vec, atol=1e-14
        )

    def test_validation(self):
        with self.assertRaises(ValueError):
            gates.fuse()
        with self.assertRaises(ValueError):
            gates.fuse(gates.X, gates.CX)


if __name__ == "__main__":
    unittest.main()
