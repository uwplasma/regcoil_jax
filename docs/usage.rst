Usage
=====

Run the CLI on CPU::

  regcoil_jax --platform cpu --verbose examples/3_advanced/regcoil_in.lambda_search_1

Key inputs
----------

The Fortran REGCOIL namelist has many options; this parity-first port currently supports a subset. A few
commonly-used keys:

- ``regularization_term_option``: ``"chi2_K"`` (default), ``"K_xy"``, or ``"Laplace-Beltrami"``
- ``geometry_option_plasma``:

  - ``0``: parity convenience mode (VMEC boundary if ``wout_filename`` is present, else analytic torus)
  - ``1``: analytic circular torus (``R0_plasma``, ``a_plasma``, ``nfp_imposed``)
  - ``2``: VMEC boundary (outermost full radial grid point)
  - ``3``: VMEC boundary (outermost half radial grid point, approximated by averaging the last two full-grid surfaces)
  - ``4``: VMEC straight-field-line poloidal coordinate (requires ``wout_filename`` and uses ``mpol_transform_refinement`` / ``ntor_transform_refinement``)
  - ``5``: EFIT gfile (requires ``efit_filename``, uses ``efit_num_modes`` and ``efit_psiN``; ``efit_psiN<1`` uses a coarse-grid interpolation approximation)
  - ``6``: Fourier coefficient table (``shape_filename_plasma``)
  - ``7``: FOCUS ``rdsurf`` boundary file (``shape_filename_plasma``; may embed Bn coefficients)

- ``geometry_option_coil``:

  - ``0``: parity convenience mode (VMEC boundary if ``wout_filename`` is present, else analytic torus)
  - ``1``: analytic circular torus (``R0_coil``, ``a_coil``)
  - ``2``: uniform offset from VMEC plasma boundary (``separation``; Fourier-fit controlled by ``max_mpol_coil``/``max_ntor_coil`` and filters)
  - ``3``: winding surface read from a NESCOIL ``nescin`` file (``nescin_filename``)
  - ``4``: constant-arclength theta coordinate on an offset-from-VMEC winding surface (``separation`` and ``constant_arclength_tolerance``)
- ``general_option=5``: lambda search (see the ``examples/3_advanced`` inputs)
- ``general_option=2``: compute diagnostics for one or more NESCOIL current potentials stored in ``nescout_filename``
  (see ``examples/2_intermediate/regcoil_in.torus_nescout_diagnostics``)
- ``general_option=3``: emulate NESCOIL’s truncated SVD scan (see ``examples/2_intermediate/regcoil_in.torus_svd_scan``)

Lambda search targets
---------------------

For ``general_option=4`` or ``general_option=5``, REGCOIL searches for a ``lambda`` such that a chosen diagnostic hits
``target_value``. This port supports the same target options as the Fortran code:

- ``target_option = "max_K"`` (default)
- ``target_option = "rms_K"``
- ``target_option = "chi2_K"``
- ``target_option = "max_Bnormal"``
- ``target_option = "rms_Bnormal"``
- ``target_option = "chi2_B"``
- ``target_option = "max_K_lse"`` (uses ``target_option_p`` as the log-sum-exp sharpness)
- ``target_option = "lp_norm_K"`` (uses ``target_option_p`` as the p-norm exponent)

Examples
--------

See `examples/README.md` for the tiered examples layout (`1_simple/`, `2_intermediate/`, `3_advanced/`).

Note: for strict parity with the reference Fortran code, this port reproduces some edge-case behavior,
including the ``nlambda=2`` lambda-grid producing ``lambda(2)=NaN`` (as in ``regcoil_compute_lambda.f90``).

Autodiff / Optimization demos
-----------------------------

Two examples highlight “JAX-native” workflows:

- `examples/2_intermediate/jax_optimize_coil_minor_radius.py`: differentiable toy optimization (fast).
- `examples/3_advanced/jax_optimize_separation_vmec_offset.py`: optimizes the VMEC offset-surface ``separation`` parameter
  by differentiating through surface construction, matrix build, and the linear solve.
- `examples/3_advanced/winding_surface_autodiff_optimize_and_visualize.py`: optimizes a *spatially varying* ``separation(θ,ζ)``
  field, then runs and visualizes REGCOIL “before” and “after”.

Geometry-option examples
------------------------

The `examples/1_simple/` tier includes small inputs that exercise the non-VMEC plasma geometry options:

- ``regcoil_in.plasma_option_6_fourier_table`` (``geometry_option_plasma=6``)
- ``regcoil_in.plasma_option_7_focus_embedded_bnorm`` (``geometry_option_plasma=7`` with embedded Bn coefficients)

The `examples/2_intermediate/` tier includes:

- ``regcoil_in.plasma_option_4_vmec_straight_fieldline`` (``geometry_option_plasma=4``; VMEC straight-field-line coordinate)
- ``regcoil_in.plasma_option_5_efit_lcfs`` (``geometry_option_plasma=5``; EFIT LCFS, ``efit_psiN=1.0``)

Outputs
-------

For an input file named ``regcoil_in.<case>``, the CLI writes next to the input:

- ``regcoil_out.<case>.nc``
- ``regcoil_out.<case>.log``

Tests
-----

From the project root::

  pytest -q

To also run slow/high-resolution examples (not recommended for CI)::

  REGCOIL_JAX_RUN_SLOW=1 pytest -q
