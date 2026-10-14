"""Hypothesis strategies for random circuits, and the test-suite's Hypothesis profiles.

Profiles (choose with the HYPOTHESIS_PROFILE environment variable):

* ``default``: 100 examples per property, derandomized, so every run of the
  suite tests the same circuits and a failure always reproduces.
* ``thorough``: 2000 random examples per property, for hunting bugs.
"""

from __future__ import annotations

import os

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

from quantum_simulator.circuit import Circuit

settings.register_profile(
    "default", max_examples=100, deadline=None, derandomize=True, database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "thorough", max_examples=2000, deadline=None, database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

ONE_QUBIT_FIXED = ["i", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx"]
TWO_QUBIT_FIXED = ["cx", "cy", "cz", "ch", "swap", "iswap"]
ONE_QUBIT_ROTATIONS = ["rx", "ry", "rz", "p"]
TWO_QUBIT_ROTATIONS = ["crx", "cry", "crz", "cp", "rxx", "ryy", "rzz"]
NOT_INVERTIBLE = {"sx", "iswap"}  # their inverses are not in the gate set

angles = st.floats(min_value=-2 * np.pi, max_value=2 * np.pi, allow_nan=False, allow_infinity=False)


@st.composite
def circuits(draw, max_qubits: int = 6, max_depth: int = 30, invertible: bool = False) -> Circuit:
    """A random unitary circuit using every gate kind, with fixed angles.

    Ops are drawn one at a time, so Hypothesis can shrink a failure by
    removing ops, lowering the qubit count and simplifying angles.
    """
    n = draw(st.integers(min_value=1, max_value=max_qubits))
    kinds = ["1q", "1q-rotation"]
    if n >= 2:
        kinds += ["2q", "2q-rotation", "mcz"]
    if n >= 3:
        kinds += ["ccx", "mcx"]
    c = Circuit(n)
    for _ in range(draw(st.integers(min_value=0, max_value=max_depth))):
        kind = draw(st.sampled_from(kinds))
        if kind in ("1q", "1q-rotation"):
            q = draw(st.integers(0, n - 1))
            if kind == "1q":
                names = [g for g in ONE_QUBIT_FIXED if not (invertible and g in NOT_INVERTIBLE)]
                getattr(c, draw(st.sampled_from(names)))(q)
            else:
                getattr(c, draw(st.sampled_from(ONE_QUBIT_ROTATIONS)))(q, draw(angles))
        elif kind in ("2q", "2q-rotation"):
            a, b = draw(st.permutations(range(n)))[:2]
            if kind == "2q":
                names = [g for g in TWO_QUBIT_FIXED if not (invertible and g in NOT_INVERTIBLE)]
                getattr(c, draw(st.sampled_from(names)))(a, b)
            else:
                getattr(c, draw(st.sampled_from(TWO_QUBIT_ROTATIONS)))(a, b, draw(angles))
        elif kind == "ccx":
            c.ccx(*draw(st.permutations(range(n)))[:3])
        else:
            k = draw(st.integers(min_value=2, max_value=n))
            qubits = draw(st.permutations(range(n)))[:k]
            if kind == "mcx":
                c.mcx(qubits[:-1], qubits[-1])
            else:
                c.mcz(*qubits)
    return c
