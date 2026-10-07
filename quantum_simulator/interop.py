"""Conversion to Qiskit, so circuits can be checked against Qiskit (Part 7 of the guide).

Qiskit is imported lazily: the rest of the package does not need it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from .circuit import PARAMETRIC_GATES, Circuit, Op


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
        _emit(qc, op, values)
    return qc


def _emit(qc, op: Op, values) -> None:
    if op.condition is not None:
        clbit, value = op.condition
        with qc.if_test((qc.clbits[clbit], value)):
            _emit(qc, dataclasses.replace(op, condition=None), values)
    elif op.name == "measure":
        qc.measure(op.qubits[0], op.clbit)
    elif op.name == "fused":
        for o in op.ops:
            _emit(qc, o, values)
    elif op.name in PARAMETRIC_GATES:
        angle = op.value if op.value is not None else values[op.param]
        getattr(qc, op.name)(angle, *op.qubits)  # method names match Qiskit's
    elif op.name == "i":
        qc.id(op.qubits[0])
    else:
        getattr(qc, op.name)(*op.qubits)
