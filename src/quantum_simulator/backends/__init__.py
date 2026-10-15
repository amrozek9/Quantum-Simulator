"""Alternative simulation backends alongside the dense NumPy state vector.

Each one buys memory or speed by assuming structure, and each is checked
against the dense simulator:

* ``jax_backend``: the dense simulator in JAX, compiled and differentiable.
* ``mps``: matrix product states, O(n chi^2) memory, exact while the
  Schmidt rank at every cut is at most chi.
* ``stabilizer``: an Aaronson-Gottesman tableau, O(n^2), Clifford gates only.
* ``density``: density matrices, O(4^n), for mixed states and noise
  channels, with pure-state trajectory averaging as the cheaper alternative.

The submodules are imported on demand; ``jax_backend`` needs JAX.
"""
