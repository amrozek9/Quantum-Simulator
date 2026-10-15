"""Interop: circuits to and from Qiskit, and to and from OpenQASM 3.

Qiskit is imported lazily: the rest of the package does not need it. The
OpenQASM 3 writer and reader need nothing outside this package.

OpenQASM 3's standard library (``stdgates.inc``) lacks ``iswap``, ``rxx``,
``ryy``, ``rzz`` and multi-controlled gates. :func:`to_qasm3` defines the
first four in the program with exact decompositions (no global phase), and
writes ``mcx`` and ``mcz`` with the ``ctrl(k) @`` modifier, so the output
loads in any OpenQASM 3 tool, Qiskit's importer included.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import re
from collections.abc import Sequence

import numpy as np

from .circuit import FIXED_GATES, MULTI_CONTROLLED, PARAMETRIC_GATES, Circuit, Op

# -- Qiskit ---------------------------------------------------------------------------


def to_qiskit(circuit: Circuit, params: Sequence[float] | np.ndarray | None = None):
    """The equivalent ``qiskit.QuantumCircuit``.

    With ``params``, angles are bound to those numbers. Without them, a
    parameterized circuit gets a ``ParameterVector`` named ``"theta"`` of
    length ``circuit.n_params``, so ``theta[k]`` is parameter index ``k``.
    Fused ops are expanded back into their gates, and conditions become
    ``if_test`` blocks.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    qc = QuantumCircuit(circuit.n, circuit.n_clbits)
    if params is None and circuit.n_params:
        values = ParameterVector("theta", circuit.n_params)
    else:
        values = params
    for op in circuit.ops:
        _emit_qiskit(qc, op, values)
    return qc


def _emit_qiskit(qc, op: Op, values) -> None:
    if op.condition is not None:
        clbit, value = op.condition
        with qc.if_test((qc.clbits[clbit], value)):
            _emit_qiskit(qc, dataclasses.replace(op, condition=None), values)
    elif op.name == "measure":
        qc.measure(op.qubits[0], op.clbit)
    elif op.name == "fused":
        for o in op.ops:
            _emit_qiskit(qc, o, values)
    elif op.name in PARAMETRIC_GATES:
        angle = op.value if op.value is not None else values[op.param]
        getattr(qc, op.name)(angle, *op.qubits)  # method names match Qiskit's
    elif op.name == "i":
        qc.id(op.qubits[0])
    elif op.name == "mcx":
        qc.mcx(list(op.qubits[:-1]), op.qubits[-1])
    elif op.name == "mcz":
        qc.mcp(np.pi, list(op.qubits[:-1]), op.qubits[-1])  # a phase of pi on all-ones is Z
    else:
        getattr(qc, op.name)(*op.qubits)


def from_qiskit(qc) -> Circuit:
    """A ``Circuit`` from a ``qiskit.QuantumCircuit`` with bound parameters.

    Supports this package's gate set (``id`` becomes ``i``, ``mcx`` keeps its
    controls, ``mcphase`` with angle pi becomes ``mcz``), measurements,
    barriers (dropped) and ``if_test`` blocks on a single classical bit with
    no else branch. Anything else raises ``ValueError``.
    """
    circuit = Circuit(qc.num_qubits, qc.num_clbits)
    for instruction in qc.data:
        qubits = [qc.find_bit(q).index for q in instruction.qubits]
        clbits = [qc.find_bit(c).index for c in instruction.clbits]
        for op in _ops_from_qiskit(qc, instruction.operation, qubits, clbits):
            circuit.append(op)
    return circuit


def _ops_from_qiskit(qc, operation, qubits: list[int], clbits: list[int]) -> list[Op]:
    name = operation.name
    if name == "barrier":
        return []
    if name == "measure":
        return [Op("measure", (qubits[0],), clbit=clbits[0])]
    if name == "if_else":
        condition = operation.condition
        if not isinstance(condition, tuple) or len(operation.blocks) != 1:
            raise ValueError("only if_test on a single classical bit, without else, is supported")
        clbit = qc.find_bit(condition[0]).index
        body = operation.blocks[0]
        out = []
        for inner in body.data:
            inner_qubits = [qubits[body.find_bit(q).index] for q in inner.qubits]
            for op in _ops_from_qiskit(body, inner.operation, inner_qubits, []):
                out.append(dataclasses.replace(op, condition=(clbit, int(condition[1]))))
        return out
    params = []
    for p in operation.params:
        try:
            params.append(float(p))
        except TypeError:
            raise ValueError(f"{name} has an unbound parameter {p}; bind parameters first") from None
    if name == "id":
        return [Op("i", tuple(qubits))]
    if name in ("mcx", "c3x", "c4x"):  # Qiskit's names for multi-controlled X (controls first)
        return [Op("mcx", tuple(qubits))]
    if name == "mcphase":
        if not math.isclose(params[0], math.pi):
            raise ValueError("only mcphase(pi), which is mcz, is supported")
        return [Op("mcz", tuple(qubits))]
    if name in PARAMETRIC_GATES:
        return [Op(name, tuple(qubits), value=params[0])]
    if name in FIXED_GATES:
        return [Op(name, tuple(qubits))]
    raise ValueError(f"unsupported Qiskit instruction {name!r}")


# -- OpenQASM 3 -------------------------------------------------------------------------

_DEFINITIONS = {
    "rzz": "gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }",
    "rxx": "gate rxx(theta) a, b { h a; h b; cx a, b; rz(theta) b; cx a, b; h a; h b; }",
    "ryy": "gate ryy(theta) a, b { rx(pi/2) a; rx(pi/2) b; cx a, b; rz(theta) b; cx a, b; rx(-pi/2) a; rx(-pi/2) b; }",
    "iswap": "gate iswap a, b { rxx(-pi/2) a, b; ryy(-pi/2) a, b; }",
}
_NEEDS = {"iswap": ("rxx", "ryy", "iswap")}


def to_qasm3(circuit: Circuit, params: Sequence[float] | np.ndarray | None = None) -> str:
    """The circuit as an OpenQASM 3 program.

    With ``params``, angles are written as numbers. Without them, each
    parameter index ``k`` becomes ``input float[64] theta_k``, so the
    program stays parameterized. Angles are written with ``repr``, which
    round-trips floats exactly.
    """
    flat = [o for op in circuit.ops for o in (op.ops if op.name == "fused" else (op,))]
    used = {op.name for op in flat}
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";']
    defined: list[str] = []
    for name in ("rzz", "rxx", "ryy", "iswap"):
        if name in used:
            for need in _NEEDS.get(name, (name,)):
                if need not in defined:
                    defined.append(need)
    lines += [_DEFINITIONS[name] for name in ("rzz", "rxx", "ryy", "iswap") if name in defined]
    symbolic = params is None and circuit.n_params > 0
    if symbolic:
        lines += [f"input float[64] theta_{k};" for k in range(circuit.n_params)]
    lines.append(f"qubit[{circuit.n}] q;")
    if circuit.n_clbits:
        lines.append(f"bit[{circuit.n_clbits}] c;")
    for op in circuit.ops:
        for o in (op.ops if op.name == "fused" else (op,)):
            lines.append(_qasm_statement(dataclasses.replace(o, condition=op.condition), params, symbolic))
    return "\n".join(lines) + "\n"


def _qasm_statement(op: Op, params, symbolic: bool) -> str:
    targets = ", ".join(f"q[{q}]" for q in op.qubits)
    if op.name == "measure":
        stmt = f"c[{op.clbit}] = measure q[{op.qubits[0]}];"
    elif op.name in MULTI_CONTROLLED:
        stmt = f"ctrl({len(op.qubits) - 1}) @ {op.name[-1]} {targets};"
    elif op.name in PARAMETRIC_GATES:
        if op.value is not None:
            angle = repr(float(op.value))
        elif symbolic:
            angle = f"theta_{op.param}"
        else:
            angle = repr(float(params[op.param]))
        stmt = f"{op.name}({angle}) {targets};"
    else:
        stmt = f"{'id' if op.name == 'i' else op.name} {targets};"
    if op.condition is not None:
        clbit, value = op.condition
        stmt = f"if (c[{clbit}] == {'true' if value else 'false'}) {{ {stmt} }}"
    return stmt


_STATEMENT = re.compile(r"^(?:ctrl\((\d+)\)\s*@\s*)?([A-Za-z_]\w*)\s*(?:\((.*)\))?\s+(.+)$")


def from_qasm3(source: str) -> Circuit:
    """Parse an OpenQASM 3 program in the subset :func:`to_qasm3` writes.

    That is: one qubit register, at most one bit register, ``input float[64]``
    parameters, this package's gates (``id`` for ``i``), ``ctrl(k) @ x`` and
    ``ctrl(k) @ z``, measurement into single bits, and ``if (c[j] == true)``
    (or ``false``, ``1``, ``0``, or a bare ``c[j]``) around a single
    statement. Gate definitions are skipped, since every gate this package
    defines is native to it. Angles may be numbers, ``pi`` and arithmetic.
    """
    source = re.sub(r"//[^\n]*", "", source)
    inputs: dict[str, int] = {}
    n = n_clbits = None
    statements = _split_statements(source)
    ops: list[Op] = []
    for stmt in statements:
        if not stmt or stmt.startswith(("OPENQASM", "include", "gate ")):
            continue
        if m := re.fullmatch(r"input\s+float\[64\]\s+(\w+)", stmt):
            inputs[m.group(1)] = len(inputs)
        elif m := re.fullmatch(r"qubit\[(\d+)\]\s+q", stmt):
            n = int(m.group(1))
        elif m := re.fullmatch(r"bit\[(\d+)\]\s+c", stmt):
            n_clbits = int(m.group(1))
        else:
            ops.append(_parse_op(stmt, inputs))
    if n is None:
        raise ValueError("expected one qubit register named q")
    circuit = Circuit(n, n_clbits or 0)
    for op in ops:
        circuit.append(op)
    circuit.n_params = max(circuit.n_params, len(inputs))
    return circuit


def _split_statements(source: str) -> list[str]:
    out, buf, depth = [], [], 0
    for ch in source:
        buf.append(ch)
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append("".join(buf).strip())
                buf = []
        elif ch == ";" and depth == 0:
            out.append("".join(buf).strip()[:-1].strip())
            buf = []
    return [s for s in out if s]


def _parse_op(stmt: str, inputs: dict[str, int]) -> Op:
    if m := re.fullmatch(r"if\s*\(\s*c\[(\d+)\]\s*(?:==\s*(true|false|1|0)\s*)?\)\s*\{\s*(.*?);?\s*\}", stmt, re.S):
        value = 1 if m.group(2) in (None, "true", "1") else 0
        return dataclasses.replace(_parse_op(m.group(3).strip(), inputs), condition=(int(m.group(1)), value))
    if m := re.fullmatch(r"c\[(\d+)\]\s*=\s*measure\s+q\[(\d+)\]", stmt):
        return Op("measure", (int(m.group(2)),), clbit=int(m.group(1)))
    m = _STATEMENT.fullmatch(stmt)
    if not m:
        raise ValueError(f"cannot parse statement: {stmt!r}")
    controls, name, arg, operands = m.groups()
    qubits = tuple(int(x) for x in re.findall(r"q\[(\d+)\]", operands))
    if controls is not None:
        if name not in ("x", "z"):
            raise ValueError(f"only ctrl @ x and ctrl @ z are supported: {stmt!r}")
        if len(qubits) != int(controls) + 1:
            raise ValueError(f"ctrl({controls}) needs {int(controls) + 1} qubits: {stmt!r}")
        return Op("mc" + name, qubits)
    name = "i" if name == "id" else name
    if name in PARAMETRIC_GATES:
        arg = (arg or "").strip()
        if arg in inputs:
            return Op(name, qubits, param=inputs[arg])
        return Op(name, qubits, value=_eval_angle(arg))
    if name in FIXED_GATES:
        return Op(name, qubits)
    raise ValueError(f"unknown gate {name!r}")


def _eval_angle(expr: str) -> float:
    """Evaluate a numeric angle: numbers, pi, unary minus, + - * / and parentheses only."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in ("pi", "π"):
            return math.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -ev(node.operand) if isinstance(node.op, ast.USub) else ev(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = ev(node.left), ev(node.right)
            return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b, ast.Div: a / b}[type(node.op)]
        raise ValueError(f"unsupported angle expression: {expr!r}")

    return ev(ast.parse(expr.replace("π", "pi"), mode="eval"))
