Contributing
============

This project aims to be:

- **Parity-first** with the reference Fortran REGCOIL outputs for the supported feature set.
- **JAX-native** for forward solves and autodiff-based workflows.
- **Pedagogic** (examples + documentation that a new user can follow end-to-end).

Development install
-------------------

From the repo root::

  pip install -e '.[dev]'

Run tests
---------

Fast test suite::

  pytest -q

Optional slow examples (kept out of CI by default)::

  REGCOIL_JAX_RUN_SLOW=1 pytest -q

Generate / update Fortran reference outputs
-------------------------------------------

The parity tests compare ``regcoil_jax`` outputs against *committed* Fortran netCDF files under
:tree:`tests/fortran_outputs/` so CI does not require Fortran.

To regenerate one or more reference outputs, you need a compiled Fortran ``regcoil`` executable.
In this workspace it lives at ``../regcoil/regcoil``.

Run the helper script from the repo root::

  python scripts/generate_fortran_reference_outputs.py

Or specify particular inputs::

  python scripts/generate_fortran_reference_outputs.py examples/1_simple/regcoil_in.compareToMatlab1

Notes:

- The script copies only the **needed** auxiliary files (VMEC ``wout``, EFIT gfile, BNORM, NESCOIL files)
  next to the input before invoking the Fortran executable, matching REGCOIL’s “paths relative to input” behavior.
- If you add a new parity case, commit the new ``regcoil_in.*`` file under :tree:`examples/` and the resulting
  reference file under :tree:`tests/fortran_outputs/` (named ``regcoil_out.*.nc``).
- If you add new example folders, prefer linking to them with :tree:`examples/` so Sphinx linkcheck does not report
  a ``blob/``→``tree/`` redirect for directories.

Build docs
----------

Sphinx docs live in ``docs/``::

  python -m sphinx -b html docs docs/_build/html -q

Verbosity / timing
------------------

The CLI prints Fortran-style progress blocks when ``--verbose`` is used.

By default, timing lines are **wall-clock** and may include JAX compilation time on first call.
To force synchronous timing (useful for profiling, slower), set::

  REGCOIL_JAX_SYNC_TIMING=1
