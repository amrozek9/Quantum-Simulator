"""Circuits as data: ops, parameters, classical bits, scheduling and fusion.

A :class:`Circuit` is a list of :class:`Op` values, built with chained calls::

    c = Circuit(2).h(0).cx(0, 1).rz(1)      # rz(1) gets parameter index 0
    result = run(c, params=[0.3])

A parameterized gate stores an *index* into a parameter vector, not an angle,
so the circuit is a fixed structure and the angles are an array supplied at
execution time. One circuit can then be run many times with different
parameters, which is what a variational loop does. Angles that never change
can be fixed instead: ``c.rz(1, 0.25)``.

Measurements write to classical bits, and any op can be conditioned on one
(``c.x(2).c_if(0)``), which is what teleportation and error correction need.

Ops on disjoint wires commute. :meth:`Circuit.dag` records, for each op, the
previous op on every wire it touches (qubits and classical bits), which gives
the depth, a parallel schedule (:meth:`Circuit.layers`), and the legality
test for :func:`fuse_single_qubit_gates`.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from . import gates
from .state import State, _as_rng

FIXED_GATES = {
    "i": gates.I, "x": gates.X, "y": gates.Y, "z": gates.Z, "h": gates.H,
    "s": gates.S, "sdg": gates.SDG, "t": gates.T, "tdg": gates.TDG, "sx": gates.SX,
    "cx": gates.CX, "cy": gates.CY, "cz": gates.CZ, "ch": gates.CH,
    "swap": gates.SWAP, "iswap": gates.ISWAP,
    "ccx": gates.controlled(gates.controlled(gates.X)),
}
PARAMETRIC_GATES = {
    "rx": gates.rx, "ry": gates.ry, "rz": gates.rz, "p": gates.phase,
    "crx": gates.crx, "cry": gates.cry, "crz": gates.crz, "cp": gates.cphase,
    "rxx": gates.rxx, "ryy": gates.ryy, "rzz": gates.rzz,
}
_ARITY = {name: m.shape[0].bit_length() - 1 for name, m in FIXED_GATES.items()}
_ARITY |= {name: f(0.0).shape[0].bit_length() - 1 for name, f in PARAMETRIC_GATES.items()}

# Gates on any number (>= 2) of qubits: the controls first, the target last.
MULTI_CONTROLLED = frozenset({"mcx", "mcz"})

_SELF_INVERSE = frozenset({"i", "x", "y", "z", "h", "cx", "cy", "cz", "ch", "swap", "ccx"}) | MULTI_CONTROLLED
_INVERSE_NAME = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}


@functools.lru_cache(maxsize=None)
def multi_controlled_matrix(name: str, n_qubits: int) -> np.ndarray:
    """``mcx`` or ``mcz`` on ``n_qubits`` qubits ``(controls..., target)``, in gate-matrix order.

    The controls are the low bits, so the all-controls-on block is the pair
    of indices ``2^k - 1`` (target 0) and ``2^(k+1) - 1`` (target 1).
    """
    if name not in MULTI_CONTROLLED or n_qubits < 2:
        raise ValueError(f"{name} needs at least 2 qubits")
    k = n_qubits - 1
    m = np.eye(2**n_qubits, dtype=np.complex128)
    lo, hi = 2**k - 1, 2 ** (k + 1) - 1
    if name == "mcx":
        m[[lo, hi]] = m[[hi, lo]]
    else:
        m[hi, hi] = -1
    m.flags.writeable = False
    return m


@dataclass(frozen=True, slots=True)
class Op:
    """One instruction.

    ``name`` is a key of :data:`FIXED_GATES` or :data:`PARAMETRIC_GATES`,
    ``"measure"``, or ``"fused"``. ``qubits`` is in gate-matrix order:
    ``qubits[0]`` is the matrix's least significant bit, so ``"cx"`` takes
    ``(control, target)``.

    A parameterized gate has exactly one of ``param`` (an index into the
    parameter vector) and ``value`` (a fixed angle). ``clbit`` is the
    classical bit a ``"measure"`` writes. ``condition = (clbit, value)``
    applies the op only if that classical bit holds ``value``. A ``"fused"``
    op stands for the single-qubit ops in ``ops``, in time order.
    """

    name: str
    qubits: tuple[int, ...]
    param: int | None = None
    value: float | None = None
    clbit: int | None = None
    condition: tuple[int, int] | None = None
    ops: tuple[Op, ...] = ()


@dataclass(slots=True)
class Circuit:
    """``n`` qubits, ``n_clbits`` classical bits, and a list of ops.

    Builder methods append one op and return the circuit, so calls chain.
    ``n_params`` is the length of the parameter vector :func:`run` expects.
    """

    n: int
    n_clbits: int = 0
    ops: list[Op] = field(default_factory=list)
    n_params: int = 0

    # -- fixed gates ------------------------------------------------------------

    def i(self, q: int) -> Circuit: return self._gate("i", q)
    def x(self, q: int) -> Circuit: return self._gate("x", q)
    def y(self, q: int) -> Circuit: return self._gate("y", q)
    def z(self, q: int) -> Circuit: return self._gate("z", q)
    def h(self, q: int) -> Circuit: return self._gate("h", q)
    def s(self, q: int) -> Circuit: return self._gate("s", q)
    def sdg(self, q: int) -> Circuit: return self._gate("sdg", q)
    def t(self, q: int) -> Circuit: return self._gate("t", q)
    def tdg(self, q: int) -> Circuit: return self._gate("tdg", q)
    def sx(self, q: int) -> Circuit: return self._gate("sx", q)
    def cx(self, control: int, target: int) -> Circuit: return self._gate("cx", control, target)
    def cy(self, control: int, target: int) -> Circuit: return self._gate("cy", control, target)
    def cz(self, control: int, target: int) -> Circuit: return self._gate("cz", control, target)
    def ch(self, control: int, target: int) -> Circuit: return self._gate("ch", control, target)
    def swap(self, a: int, b: int) -> Circuit: return self._gate("swap", a, b)
    def iswap(self, a: int, b: int) -> Circuit: return self._gate("iswap", a, b)
    def ccx(self, c0: int, c1: int, target: int) -> Circuit: return self._gate("ccx", c0, c1, target)

    cnot = cx

    def mcx(self, controls: Sequence[int], target: int) -> Circuit:
        """X on ``target`` if every control is 1 (Toffoli for two controls)."""
        return self._gate("mcx", *controls, target)

    def mcz(self, *qubits: int) -> Circuit:
        """Phase -1 on the all-ones state of ``qubits``; symmetric, so no qubit is special."""
        return self._gate("mcz", *qubits)

    # -- parameterized gates ------------------------------------------------------
    #
    # theta=None allocates the next parameter index; a number fixes the angle;
    # param=k reuses existing parameter k (e.g. one angle shared by a layer).

    def rx(self, q: int, theta: float | None = None, *, param: int | None = None) -> Circuit:
        return self._rotation("rx", (q,), theta, param)

    def ry(self, q: int, theta: float | None = None, *, param: int | None = None) -> Circuit:
        return self._rotation("ry", (q,), theta, param)

    def rz(self, q: int, theta: float | None = None, *, param: int | None = None) -> Circuit:
        return self._rotation("rz", (q,), theta, param)

    def p(self, q: int, theta: float | None = None, *, param: int | None = None) -> Circuit:
        return self._rotation("p", (q,), theta, param)

    def crx(self, c: int, t: int, theta: float | None = None, *, param: int | None = None) -> Circuit:
        return self._rotation("crx", (c, t), theta, param)

    def cry(self, c: int, t: int, theta: float | None = None, *, param: int | None = None) -> Circuit:
        return self._rotation("cry", (c, t), theta, param)

    def crz(self, c: int, t: int, theta: float | None = None, *, param: int | None = None) -> Circuit:
        return self._rotation("crz", (c, t), theta, param)

    def cp(self, c: int, t: int, theta: float | None = None, *, param: int | None = None) -> Circuit:
        return self._rotation("cp", (c, t), theta, param)

    def rxx(self, a: int, b: int, theta: float | None = None, *, param: int | None = None) -> Circuit:
        return self._rotation("rxx", (a, b), theta, param)

    def ryy(self, a: int, b: int, theta: float | None = None, *, param: int | None = None) -> Circuit:
        return self._rotation("ryy", (a, b), theta, param)

    def rzz(self, a: int, b: int, theta: float | None = None, *, param: int | None = None) -> Circuit:
        return self._rotation("rzz", (a, b), theta, param)

    # -- measurement and classical control ----------------------------------------

    def measure(self, qubit: int, clbit: int) -> Circuit:
        """Measure ``qubit`` and write the outcome to classical bit ``clbit``."""
        return self.append(Op("measure", (qubit,), clbit=clbit))

    def c_if(self, clbit: int, value: int = 1) -> Circuit:
        """Condition the most recently added op: apply it only if ``clbit`` holds ``value``."""
        if not self.ops:
            raise ValueError("c_if needs an op to condition")
        last = self.ops.pop()
        try:
            return self.append(dataclasses.replace(last, condition=(clbit, value)))
        except ValueError:
            self.ops.append(last)
            raise

    # -- general --------------------------------------------------------------

    def append(self, op: Op) -> Circuit:
        """Validate ``op`` and add it. Grows ``n_params`` to cover ``op.param``."""
        _validate(op, self.n, self.n_clbits)
        self.ops.append(op)
        for o in (op, *op.ops):
            if o.param is not None:
                self.n_params = max(self.n_params, o.param + 1)
        return self

    def _gate(self, name: str, *qubits: int) -> Circuit:
        return self.append(Op(name, tuple(qubits)))

    def _rotation(self, name: str, qubits: tuple[int, ...], theta: float | None, param: int | None) -> Circuit:
        if theta is not None and param is not None:
            raise ValueError("give a fixed angle or a parameter index, not both")
        if theta is not None:
            return self.append(Op(name, qubits, value=float(theta)))
        if param is None:
            param = self.n_params  # allocate the next index
        elif not 0 <= param < self.n_params:
            raise ValueError(
                f"param={param} does not exist yet (circuit has {self.n_params}); omit it to allocate one"
            )
        return self.append(Op(name, qubits, param=param))

    # -- structure ------------------------------------------------------------

    def dag(self) -> list[tuple[int, ...]]:
        """For each op, the indices of its direct predecessors.

        A predecessor is the previous op on any wire this op touches, where
        the wires are its qubits, the classical bit a measurement writes, and
        the classical bit a condition reads.
        """
        last: dict[tuple[str, int], int] = {}
        preds = []
        for i, op in enumerate(self.ops):
            wires = _wires(op)
            preds.append(tuple(sorted({last[w] for w in wires if w in last})))
            for w in wires:
                last[w] = i
        return preds

    def layers(self) -> list[list[Op]]:
        """As-soon-as-possible schedule: each layer's ops touch disjoint wires."""
        level: list[int] = []
        for preds in self.dag():
            level.append(1 + max((level[p] for p in preds), default=-1))
        out: list[list[Op]] = [[] for _ in range(max(level, default=-1) + 1)]
        for op, lv in zip(self.ops, level):
            out[lv].append(op)
        return out

    def depth(self) -> int:
        """Number of layers: the longest chain of ops that depend on each other."""
        return len(self.layers())

    def inverse(self) -> Circuit:
        """The circuit that undoes this one: ops in reverse order, each inverted.

        Fixed-angle rotations are negated, S and T become S-dagger and
        T-dagger, and self-inverse gates stay. Measurements and conditions
        are not reversible; parameterized gates (an index, not an angle) and
        gates whose inverse is not in the gate set (``sx``, ``iswap``) are
        rejected too.
        """
        out = Circuit(self.n, self.n_clbits)
        for op in reversed(self.ops):
            out.append(_inverse_op(op))
        return out

    def compose(self, other: Circuit, qubits: Sequence[int] | None = None) -> Circuit:
        """Append ``other``'s ops, with its qubit ``i`` placed on ``qubits[i]``; returns ``self``.

        ``other`` must be a fixed unitary block: no parameter indices and no
        classical bits, since those would need renumbering.
        """
        qubits = list(range(other.n)) if qubits is None else list(qubits)
        if len(qubits) != other.n:
            raise ValueError(f"other has {other.n} qubits, got a map onto {len(qubits)}")
        if other.n_params or other.n_clbits:
            raise ValueError("compose takes circuits without parameter indices or classical bits")

        def remap(op: Op) -> Op:
            return dataclasses.replace(
                op, qubits=tuple(qubits[q] for q in op.qubits), ops=tuple(remap(o) for o in op.ops)
            )

        for op in other.ops:
            self.append(remap(op))
        return self


def _inverse_op(op: Op) -> Op:
    if op.name == "measure" or op.condition is not None:
        raise ValueError(f"cannot invert a measurement or a conditioned op: {op}")
    if op.name == "fused":
        return dataclasses.replace(op, ops=tuple(_inverse_op(o) for o in reversed(op.ops)))
    if op.name in _SELF_INVERSE:
        return op
    if op.name in _INVERSE_NAME:
        return dataclasses.replace(op, name=_INVERSE_NAME[op.name])
    if op.name in PARAMETRIC_GATES:
        if op.param is not None:
            raise ValueError(f"cannot invert a parameterized gate (it holds an index, not an angle): {op}")
        return dataclasses.replace(op, value=-op.value)
    raise ValueError(f"no inverse for {op.name} in the gate set")


def _validate(op: Op, n: int, n_clbits: int) -> None:
    for q in op.qubits:
        if not isinstance(q, (int, np.integer)) or not 0 <= q < n:
            raise ValueError(f"qubit {q!r} out of range for {n} qubits in {op}")
    if len(set(op.qubits)) != len(op.qubits):
        raise ValueError(f"repeated qubit in {op}")
    if op.condition is not None:
        clbit, value = op.condition
        if not 0 <= clbit < n_clbits or value not in (0, 1):
            raise ValueError(f"condition {op.condition} invalid with {n_clbits} classical bits")

    if op.name == "measure":
        if len(op.qubits) != 1 or op.clbit is None or not 0 <= op.clbit < n_clbits:
            raise ValueError(f"measure needs one qubit and a classical bit < {n_clbits}: {op}")
        return
    if op.name == "fused":
        if len(op.qubits) != 1 or not op.ops:
            raise ValueError(f"a fused op acts on one qubit and needs ops: {op}")
        for o in op.ops:
            if o.qubits != op.qubits or o.condition is not None or o.name in ("measure", "fused"):
                raise ValueError(f"fused ops must be unconditioned gates on {op.qubits}: {o}")
            _validate(o, n, n_clbits)
        return
    if op.name in MULTI_CONTROLLED:
        if len(op.qubits) < 2 or op.param is not None or op.value is not None:
            raise ValueError(f"{op.name} needs at least 2 qubits and takes no parameter: {op}")
        return
    if op.name not in _ARITY:
        raise ValueError(f"unknown gate {op.name!r}")
    if len(op.qubits) != _ARITY[op.name]:
        raise ValueError(f"{op.name} acts on {_ARITY[op.name]} qubit(s), got {op.qubits}")
    if op.name in PARAMETRIC_GATES:
        if (op.param is None) == (op.value is None):
            raise ValueError(f"{op.name} needs exactly one of param (index) or value (angle): {op}")
        if op.param is not None and op.param < 0:
            raise ValueError(f"negative parameter index in {op}")
    elif op.param is not None or op.value is not None:
        raise ValueError(f"{op.name} takes no parameter: {op}")


def _wires(op: Op) -> list[tuple[str, int]]:
    wires = [("q", q) for q in op.qubits]
    if op.clbit is not None:
        wires.append(("c", op.clbit))
    if op.condition is not None:
        wires.append(("c", op.condition[0]))
    return wires


# -- execution ----------------------------------------------------------------------


@dataclass(frozen=True)
class Result:
    """The final state and the classical bits after :func:`run`."""

    state: State
    clbits: tuple[int, ...]


def op_matrix(op: Op, params: Sequence[float] | np.ndarray | None = None) -> np.ndarray:
    """The unitary for a gate op, with its angle taken from ``params`` if it has one."""
    if op.name in FIXED_GATES:
        return FIXED_GATES[op.name]
    if op.name in MULTI_CONTROLLED:
        return multi_controlled_matrix(op.name, len(op.qubits))
    if op.name in PARAMETRIC_GATES:
        if op.value is not None:
            return PARAMETRIC_GATES[op.name](op.value)
        if params is None:
            raise ValueError(f"{op} uses parameter {op.param}; pass params")
        return PARAMETRIC_GATES[op.name](params[op.param])
    if op.name == "fused":
        return gates.fuse(*(op_matrix(o, params) for o in op.ops))
    raise ValueError(f"{op.name!r} has no matrix")


def run(
    circuit: Circuit,
    params: Sequence[float] | np.ndarray | None = None,
    *,
    rng: np.random.Generator | int | None = None,
    dtype: np.typing.DTypeLike | None = None,
    initial: State | None = None,
) -> Result:
    """Execute ``circuit`` from ``|0...0>`` (or a copy of ``initial``).

    ``params`` must have length ``circuit.n_params``. ``rng`` is required if
    the circuit measures anything; an int seeds one generator for the run.
    """
    params = _check_params(circuit, params)
    rng = _as_rng(rng) if any(op.name == "measure" for op in circuit.ops) else None
    if initial is None:
        state = State.zero(circuit.n, dtype)
    else:
        if initial.n != circuit.n:
            raise ValueError(f"initial state has {initial.n} qubits, circuit has {circuit.n}")
        state = initial.copy() if dtype is None else initial.astype(dtype)

    clbits = [0] * circuit.n_clbits
    for op in circuit.ops:
        if op.condition is not None and clbits[op.condition[0]] != op.condition[1]:
            continue
        if op.name == "measure":
            clbits[op.clbit] = state.measure(op.qubits[0], rng)
        else:
            state.apply(op_matrix(op, params), op.qubits)
    return Result(state, tuple(clbits))


def _check_params(circuit: Circuit, params) -> np.ndarray | None:
    if params is None:
        if circuit.n_params:
            raise ValueError(f"circuit has {circuit.n_params} parameter(s); pass params")
        return None
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (circuit.n_params,):
        raise ValueError(f"expected {circuit.n_params} parameter(s), got shape {params.shape}")
    return params


# -- optimization -------------------------------------------------------------------


def fuse_single_qubit_gates(circuit: Circuit) -> Circuit:
    """Merge each run of single-qubit gates on a wire into one ``"fused"`` op.

    A run ends at anything else that touches its qubit: a multi-qubit gate,
    a measurement, or a conditioned op. Fewer ops means fewer passes over
    the state and fewer Python calls. Gates on other wires may be emitted in
    a different order, which is legal because ops on disjoint qubits
    commute. Parameters stay symbolic: the fused matrix is built at run time.
    """
    out = Circuit(circuit.n, circuit.n_clbits, n_params=circuit.n_params)
    pending: dict[int, list[Op]] = {}

    def flush(q: int) -> None:
        run_ = [c for o in pending.pop(q, []) for c in (o.ops if o.name == "fused" else (o,))]
        if len(run_) == 1:
            out.ops.append(run_[0])
        elif run_:
            out.ops.append(Op("fused", (q,), ops=tuple(run_)))

    for op in circuit.ops:
        if len(op.qubits) == 1 and op.name != "measure" and op.condition is None:
            pending.setdefault(op.qubits[0], []).append(op)
            continue
        for q in op.qubits:
            flush(q)
        out.ops.append(op)
    for q in sorted(pending):
        flush(q)
    return out
