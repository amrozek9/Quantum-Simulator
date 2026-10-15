"""Textbook algorithms as circuits, used as end-to-end tests of the simulator.

Each builder returns a :class:`~quantum_simulator.circuit.Circuit`. Each
algorithm has a property that only holds if every convention is right
(qubit order, gate matrices, controlled phases, measurement), which
``tests/test_algorithms.py`` asserts:

* Bell pair and GHZ: one bit of entanglement; exactly two nonzero amplitudes.
* Teleportation: the target ends in the input state on every branch.
* Bernstein-Vazirani: one oracle query reveals the hidden string.
* Grover: success near 1 after about (pi/4) sqrt(N) iterations, then falling.
* QFT: amplitudes ``exp(2 pi i j k / N) / sqrt(N)``.
* Phase estimation: recovers a known eigenphase to register precision.

Integers are little-endian throughout: bit ``i`` of a register value is
qubit ``i``.
"""

from __future__ import annotations

import numpy as np

from .circuit import Circuit


def bell_pair() -> Circuit:
    """``(|00> + |11>) / sqrt(2)``."""
    return Circuit(2).h(0).cx(0, 1)


def ghz(n: int) -> Circuit:
    """``(|0...0> + |1...1>) / sqrt(2)`` on ``n`` qubits, by fanning out from qubit 0."""
    c = Circuit(n).h(0)
    for q in range(1, n):
        c.cx(0, q)
    return c


def teleportation(theta: float, phi: float) -> Circuit:
    """Teleport ``RZ(phi) RY(theta)|0>`` from qubit 0 to qubit 2.

    Qubits 1 and 2 share a Bell pair; qubits 0 and 1 are measured into
    classical bits 0 and 1, which then control X and Z corrections on qubit 2.
    """
    c = Circuit(3, 2).ry(0, theta).rz(0, phi)
    c.h(1).cx(1, 2)
    c.cx(0, 1).h(0)
    c.measure(0, 0).measure(1, 1)
    return c.x(2).c_if(1).z(2).c_if(0)


def bernstein_vazirani(secret: int, n: int) -> Circuit:
    """Recover ``secret`` from one query to the oracle ``f(x) = secret . x mod 2``.

    Data qubits ``0 .. n-1`` end in ``|secret>``; qubit ``n`` is an ancilla in
    ``|->``, so each CX in the oracle kicks a phase back onto its control.
    """
    if not 0 <= secret < 2**n:
        raise ValueError(f"secret {secret} does not fit in {n} bits")
    c = Circuit(n + 1).x(n).h(n)
    for q in range(n):
        c.h(q)
    for q in range(n):  # the oracle: one query
        if (secret >> q) & 1:
            c.cx(q, n)
    for q in range(n):
        c.h(q)
    return c


def grover(n: int, marked: int, iterations: int | None = None) -> Circuit:
    """Grover search for ``marked`` among ``N = 2^n`` items.

    ``iterations`` defaults to ``floor((pi/4) sqrt(N))``, the optimum. Each
    iteration is the oracle (a phase flip on ``|marked>``) followed by the
    diffusion operator (a reflection about the uniform superposition).
    """
    if n < 2:
        raise ValueError("grover needs at least 2 qubits")
    if not 0 <= marked < 2**n:
        raise ValueError(f"marked {marked} does not fit in {n} bits")
    if iterations is None:
        iterations = optimal_grover_iterations(n)
    qubits = range(n)
    zeros = [q for q in qubits if not (marked >> q) & 1]

    c = Circuit(n)
    for q in qubits:
        c.h(q)
    for _ in range(iterations):
        for q in zeros:  # oracle: map |marked> to |1...1>, flip its sign, map back
            c.x(q)
        c.mcz(*qubits)
        for q in zeros:
            c.x(q)
        for q in qubits:  # diffusion: 2|s><s| - I, up to a global phase
            c.h(q).x(q)
        c.mcz(*qubits)
        for q in qubits:
            c.x(q).h(q)
    return c


def optimal_grover_iterations(n: int) -> int:
    return int(np.floor(np.pi / 4 * np.sqrt(2**n)))


def grover_success_theory(n: int, iterations: int) -> float:
    """``sin^2((2k + 1) theta)`` with ``sin(theta) = 1/sqrt(N)``: Grover is a rotation by ``2 theta``."""
    theta = np.arcsin(1 / np.sqrt(2**n))
    return float(np.sin((2 * iterations + 1) * theta) ** 2)


def qft(n: int, swaps: bool = True) -> Circuit:
    """Quantum Fourier transform: ``|j> -> sum_k exp(2 pi i j k / N) |k> / sqrt(N)``.

    Hadamards and controlled phases build the transform with the bit order
    reversed; the final swaps restore it (``swaps=False`` leaves it reversed).
    This matches Qiskit's ``QFTGate``.
    """
    c = Circuit(n)
    for j in reversed(range(n)):
        c.h(j)
        for k in reversed(range(j)):
            c.cp(j, k, np.pi / 2 ** (j - k))
    if swaps:
        for i in range(n // 2):
            c.swap(i, n - 1 - i)
    return c


def phase_estimation(phase: float, t: int) -> Circuit:
    """Estimate ``phase`` in ``U = P(2 pi phase)``, whose eigenstate is ``|1>``.

    Counting qubits ``0 .. t-1``, target qubit ``t``. Counting qubit ``k``
    controls ``U^(2^k)``, which kicks the phase ``2 pi phase 2^k`` back onto
    it; the inverse QFT then turns those phases into the integer nearest
    ``phase * 2^t``. If ``phase * 2^t`` is an integer the outcome is
    certain; otherwise the nearest integer has probability at least 4/pi^2.
    """
    c = Circuit(t + 1).x(t)
    for k in range(t):
        c.h(k)
    for k in range(t):
        c.cp(k, t, 2 * np.pi * phase * 2**k)
    return c.compose(qft(t).inverse(), qubits=range(t))
