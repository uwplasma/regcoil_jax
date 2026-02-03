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
     - NESCOIL ``nescout`` diagnostics (:ex:`examples/2_intermediate/regcoil_in.torus_nescout_diagnostics`)
   * - ``general_option=3``
     - OK
     - Truncated SVD scan (:ex:`examples/2_intermediate/regcoil_in.torus_svd_scan`)
   * - ``general_option=4``
     - OK
     - Lambda search (no feasibility checks) (:ex:`examples/2_intermediate/regcoil_in.lambda_search_option4_torus`)
   * - ``general_option=5``
     - OK
     - Lambda search (with feasibility checks) (see :tree:`examples/3_advanced/` inputs named ``regcoil_in.lambda_search_*``)

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
     - OK (JAX)
     - VMEC straight-field-line poloidal coordinate. The reference Fortran in this workspace can fail for typical VMEC files (angle/units inconsistency), so this option is smoke-tested (not netCDF parity-tested).
   * - 5 (EFIT)
     - OK
     - EFIT gfile support. Full netCDF parity is regression-tested for ``efit_psiN=1.0`` (LCFS); ``efit_psiN<1`` uses a coarse-grid interpolation approximation.
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
     - N/A
     - regcoil_jax targets autodiff instead (no Fortran adjoint port)
   * - Sensitivity outputs (``sensitivity_option>1``)
     - OK
     - ``xm_sensitivity``, ``xn_sensitivity``, ``omega_coil`` and autodiff-based ``dchi2domega`` parity-tested in :ex:`examples/2_intermediate/regcoil_in.torus_sensitivity_option2_small`
   * - Autodiff through solve pipeline
     - OK
     - JAX differentiates through geometry → matrices → solve → diagnostics
   * - Winding surface optimization
     - OK
     - ``separation(θ,ζ)`` Fourier-parameterized optimizer + before/after visualization demo

Outputs and tests
-----------------

- NetCDF writer targets **schema parity** with the Fortran output for the covered modes.
- Tests in :src:`tests/test_examples.py` run curated inputs and compare the full netCDF contents against committed
  Fortran reference outputs.

Beyond-REGCOIL demos
--------------------

- Coil cutting + per-coil current optimization (autodiff through Biot–Savart) is implemented and demonstrated.
- JAX-native field line tracing + differentiable “soft Poincaré” utilities are provided in :src:`regcoil_jax/fieldlines_jax.py` and demonstrated in :ex:`examples/3_advanced/jax_poincare_grad_demo.py`.
- Hybrid “few loops + many dipoles” optimization is provided as a pedagogic example (not part of Fortran parity goals).
- Quadcoil-style differentiable diagnostics from :math:`\\nabla_s\\Phi` (coil spacing / total length estimates) are provided in :src:`regcoil_jax/quadcoil_objectives.py`.
- Permanent-magnet / coillet workflows using a dipole lattice (ridge least squares + fixed-magnitude relaxation) are provided in :src:`regcoil_jax/permanent_magnets.py`.
- Differentiable coil cutting **relaxation demo** (soft contour extraction) is provided in :src:`regcoil_jax/diff_coil_cutting.py` (research/pedagogic; not a robust replacement for contouring).
- Differentiable *topology-fixed* multi-coil extraction (“snakes” relaxation) is provided in :src:`regcoil_jax/diff_coil_cutting.py` and demonstrated in :ex:`examples/3_advanced/differentiable_coil_cutting_snakes_multicoil.py`.
