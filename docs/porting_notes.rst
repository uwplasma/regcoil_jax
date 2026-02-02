Porting notes
=============

This project is a parity-first JAX rewrite of the Fortran **REGCOIL** code.

Goals
-----

- Match Fortran REGCOIL netCDF outputs for the same inputs (within tight tolerances).
- Keep the forward pipeline compatible with JAX JIT + autodiff.
- Provide a clean, pedagogic examples suite showing both parity and “JAX-native” workflows.

Current parity scope (tested)
-----------------------------

The CI/test suite compares the full netCDF schema + values against committed Fortran reference outputs for a curated set
of inputs (see ``tests/test_examples.py`` and ``tests/fortran_outputs/``). Tests additionally check output self-consistency
by recomputing key scalar diagnostics from the written 2D fields.

Example folder structure
------------------------

Examples are organized by “difficulty” / required physics machinery:

- ``examples/1_simple/``: analytic torus geometry, small grids, easiest to understand.
- ``examples/2_intermediate/``: JAX-first scripts (JIT/autodiff) that demonstrate differentiable kernels and small optimizations.
- ``examples/3_advanced/``: VMEC-based cases (requires a ``wout_*.nc`` file), lambda search, and more expensive runs.

All ``regcoil_in.*`` files are meant to be runnable via::

  regcoil_jax --platform cpu --verbose path/to/regcoil_in.some_case

The CLI writes ``regcoil_out.some_case.nc`` and ``regcoil_out.some_case.log`` next to the input file.

Notes on key fixes
------------------

Historically, one common failure mode was unintended broadcasting growth in scalar surface evaluations for VMEC offset surfaces
(``geometry_option_coil=2``). The current implementation includes explicit scalar-shape hygiene and (in verbose/debug runs) asserts
the expected REGCOIL shape convention ``(3, ntheta, nzeta)`` early in the pipeline.

Plasma geometry options 4/5
---------------------------

- ``geometry_option_plasma=4`` (VMEC straight-field-line poloidal coordinate) is implemented in a radians-consistent form:
  solve ``theta_new = theta_old + lambda(theta_old, zeta)`` using VMEC ``lmns`` and Fourier-transform the resulting ``R,Z``
  on a high-resolution grid.

  The Fortran reference in this workspace can fail for typical VMEC files due to an angle/units inconsistency, so this option is
  smoke-tested rather than netCDF-parity-tested.

- ``geometry_option_plasma=5`` (EFIT) parses EFIT gfiles and computes the axisymmetric Fourier series used by REGCOIL. Parity tests cover
  ``efit_psiN=1.0`` (LCFS-only). For ``efit_psiN<1``, ``regcoil_jax`` uses a coarse-grid bilinear interpolation approximation.

Manual parity validation
------------------------

Run the JAX port (paths will depend on which tier you pick)::

  regcoil_jax --platform cpu --verbose examples/1_simple/regcoil_in.compareToMatlab1
  regcoil_jax --platform cpu --verbose examples/1_simple/regcoil_in.compareToMatlab1_option1
  regcoil_jax --platform cpu --verbose examples/3_advanced/regcoil_in.lambda_search_1

Run the Fortran reference in the same directory (rename outputs to avoid overwriting)::

  ../../regcoil/regcoil regcoil_in.compareToMatlab1
  ../../regcoil/regcoil regcoil_in.compareToMatlab1_option1
  ../../regcoil/regcoil regcoil_in.lambda_search_1

Compare netCDF files::

  python scripts/compare_nc.py fortran.nc jax.nc

