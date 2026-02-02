Parity with Fortran REGCOIL
===========================

The primary parity target is the **Fortran netCDF output** schema + values for a curated set of inputs.

What is checked in CI
---------------------

The pytest suite runs ``regcoil_jax`` on a set of example inputs and compares the resulting
``regcoil_out.*.nc`` file to a committed reference file produced by the Fortran implementation:

- Reference files live in ``tests/fortran_outputs/`` (these are produced locally, then committed so CI can run without Fortran).
- Tests compare:
  - exact equality of dimension names + sizes
  - exact equality of variable *sets* (no missing or extra variables)
  - integer variables exactly
  - float variables with tight tolerances (with a few explicitly documented exceptions below)

Tolerances
----------

Most floating-point variables match extremely tightly (often ~1e-9 relative). The tests use a default tolerance of
``rtol=5e-8, atol=1e-8`` to allow tiny differences from dense linear algebra ordering (BLAS/XLA) while still being strict.

Some arrays are inherently more sensitive (e.g. involve cancellation, second derivatives, or near-zero coefficients).
For these, the test suite uses slightly looser but still small tolerances:

- ``Laplace_Beltrami2``: looser tolerance due to second-derivative sensitivity
- ``RHS_regularization`` and current-potential fields: larger absolute tolerances because many entries are near zero

Regenerating the Fortran reference outputs
------------------------------------------

If you have the Fortran executable available locally (typically ``../regcoil/regcoil`` from the repo root), run:

.. code-block:: bash

   python scripts/generate_fortran_reference_outputs.py

See ``docs/porting_notes.rst`` for additional notes and troubleshooting.
