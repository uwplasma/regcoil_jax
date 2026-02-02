Feature matrix
==============

This page tracks **feature coverage** vs the reference Fortran REGCOIL implementation.
The project philosophy is:

1. Keep a **parity-first** baseline (netCDF schema + numerical values).
2. Add JAX-native improvements (JIT, autodiff, optimization) **without changing default parity behavior**.

Core execution modes (``general_option``)
-----------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Fortran option
     - Status
     - Notes / examples
   * - ``general_option=1``
     - OK
     - Lambda scan (most examples)
   * - ``general_option=2``
     - OK
     - NESCOIL ``nescout`` diagnostics (``examples/2_intermediate/regcoil_in.torus_nescout_diagnostics``)
   * - ``general_option=3``
     - OK
     - Truncated SVD scan (``examples/2_intermediate/regcoil_in.torus_svd_scan``)
   * - ``general_option=4``
     - OK
     - Lambda search (no feasibility checks) (``examples/2_intermediate/regcoil_in.lambda_search_option4_torus``)
   * - ``general_option=5``
     - OK
     - Lambda search (with feasibility checks) (``examples/3_advanced/regcoil_in.lambda_search_*``)

Geometry options
----------------

Plasma surface (``geometry_option_plasma``):

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Option
     - Status
     - Notes
   * - 0 (parity convenience)
     - OK
     - VMEC boundary if ``wout_filename`` present, else analytic torus
   * - 1 (analytic torus)
     - OK
     -
   * - 2 (VMEC boundary, full mesh)
     - OK
     -
   * - 3 (VMEC boundary, half mesh)
     - OK
     - Implemented via outermost-half approximation (average last two full-grid surfaces)
   * - 4 (VMEC + extra handling)
     - TODO
     - Not implemented yet
   * - 5 (EFIT)
     - TODO
     - Not implemented yet
   * - 6 (Fourier table)
     - OK
     -
   * - 7 (FOCUS rdsurf)
     - OK
     - Includes embedded BNORM coefficients when present

Coil surface (``geometry_option_coil``):

.. list-table::
   :header-rows: 1
   :widths: 40 10

   * - Option
     - Status
   * - 0 (parity convenience)
     - OK
   * - 1 (analytic torus)
     - OK
   * - 2 (VMEC uniform offset surface)
     - OK
   * - 3 (NESCOIL ``nescin`` surface)
     - OK
   * - 4 (constant-arclength offset)
     - OK

Derivatives / optimization
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 10 50

   * - Capability
     - Status
     - Notes
   * - Adjoint sensitivities (Fortran)
     - TODO
     - Not ported; regcoil_jax targets autodiff instead
   * - Autodiff through solve pipeline
     - OK
     - JAX differentiates through geometry → matrices → solve → diagnostics
   * - Winding surface optimization
     - OK
     - ``separation(θ,ζ)`` Fourier-parameterized optimizer + before/after visualization demo

Outputs and tests
-----------------

- NetCDF writer targets **schema parity** with the Fortran output for the covered modes.
- Tests in ``tests/test_examples.py`` run curated inputs and compare the full netCDF contents against committed
  Fortran reference outputs.
