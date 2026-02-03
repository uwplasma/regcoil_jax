Inputs and outputs reference
============================

This page documents the main **inputs** (namelist keys) and **outputs** (netCDF variables)
available in ``regcoil_jax``.

Namelist input file
-------------------

The CLI expects a file named ``regcoil_in.SOMETHING`` containing a Fortran-style namelist.
The namelist name is ignored; only the key-value pairs matter.

Common inputs
~~~~~~~~~~~~~

Solver control:

- ``general_option``:
  - ``1``: lambda scan (write all lambdas)
  - ``2``: read NESCOIL potentials from ``nescout_filename``
  - ``3``: SVD scan (NESCOIL-style truncated solutions)
  - ``4``/``5``: lambda search (Brent-style) to hit a target (see ``docs/theory.rst``)
- ``nlambda``, ``lambda_min``, ``lambda_max``: lambda scan / search settings
- ``target_option``, ``target_value`` (for ``general_option=4/5``): target a quantity such as ``max_K``
- ``target_option_p`` (for ``target_option="max_K_lse"`` and ``"lp_norm_K"``): the LSE sharpness / p-norm exponent

Discretization:

- ``ntheta_plasma``, ``nzeta_plasma``: plasma surface grid
- ``ntheta_coil``, ``nzeta_coil``: winding surface grid

Fourier potential basis:

- ``mpol_potential``, ``ntor_potential``
- ``symmetry_option``: 1=sin, 2=cos, 3=both (matches Fortran)

Geometry options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plasma surface:

- ``geometry_option_plasma``:
  - ``1``: analytic torus (``r0_plasma``, ``a_plasma``)
  - ``2``: VMEC boundary (requires ``wout_filename`` or VMEC ``input.*`` boundary file)
  - ``4``: VMEC straight-field-line theta (requires ``wout_filename``)
  - ``5``: EFIT (requires ``efit_filename``)
  - ``7``: FOCUS ``.rdsurf`` table (requires ``shape_filename_plasma``)

Coil / winding surface:

- ``geometry_option_coil``:
  - ``1``: analytic torus (``r0_coil``, ``a_coil``)
  - ``2``: uniform offset from VMEC plasma boundary (requires ``wout_filename``/``input.*`` + ``separation``)
  - ``3``: read winding surface from NESCOIL ``nescin_filename``
  - ``4``: offset + constant arclength theta (VMEC-only; uses ``separation`` and arc-length controls)

Special geometry inputs:

- ``wout_filename``: VMEC ``wout_*.nc`` file (recommended), or VMEC ``input.*`` file (boundary-only mode)
- ``efit_filename``: EFIT gfile for axisymmetric plasmas
- ``shape_filename_plasma``: FOCUS ``.rdsurf`` file for plasma surface tables

Magnetic boundary condition inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``load_bnorm`` (bool) and ``bnorm_filename``: load a BNORM Fourier-mode file and set ``Bnormal_from_plasma_current``.
  This matches the Fortran behavior and is used in several examples.

Regularization options
~~~~~~~~~~~~~~~~~~~~~~

- ``regularization_term_option``:
  - ``chi2_K`` (default)
  - ``K_xy``
  - ``K_zeta``
  - ``Laplace-Beltrami``
- ``gradphi2_weight`` (optional): adds a Quadcoil-style regularizer on
  :math:`\int |\nabla_s \Phi|^2\,dA` (see ``docs/quadcoil_objectives.rst``)

Net currents
~~~~~~~~~~~~

- ``net_poloidal_current_amperes``
- ``net_toroidal_current_amperes``

When a VMEC ``wout`` file contains the needed profile arrays, ``regcoil_jax`` mirrors the Fortran workflow
and overwrites ``net_poloidal_current_amperes`` and ``curpol`` from the VMEC profiles.

Output netCDF (``regcoil_out.*.nc``)
------------------------------------------------

The output file aims for **schema parity** with the Fortran code for the supported feature set.
Not all Fortran variables are written, but the set is large enough for:

- parity regression tests,
- plotting / postprocessing,
- coil cutting workflows,
- self-consistency checks.

Key scalar outputs
~~~~~~~~~~~~~~~~~~

- ``nfp``, ``nlambda``, ``exit_code``, ``total_time``
- ``area_plasma``, ``area_coil`` (and often ``volume_plasma``, ``volume_coil``)
- ``net_poloidal_current_Amperes``, ``net_toroidal_current_Amperes``, ``curpol``

Key arrays (lambda-dependent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``lambda``: (nlambda,)
- ``solution`` and ``single_valued_current_potential_mn``: (nlambda, num_basis_functions)
- ``chi2_B``, ``chi2_K``: (nlambda,)
- ``max_Bnormal``, ``max_K``: (nlambda,)
- ``max_K_lse`` or ``lp_norm_K``: (nlambda,) when the corresponding ``target_option`` is selected in a lambda search
- ``Bnormal_total``: (nlambda, nzeta_plasma, ntheta_plasma)
- ``current_potential`` and ``single_valued_current_potential_thetazeta``: (nlambda, nzeta_coil, ntheta_coil)
- ``K2``: (nlambda, nzeta_coil, ntheta_coil)

Geometry arrays (full torus)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``r_plasma``: (nzetal_plasma, ntheta_plasma, 3)
- ``r_coil``: (nzetal_coil, ntheta_coil, 3)

Auxiliary B·n fields
~~~~~~~~~~~~~~~~~~~~

- ``Bnormal_from_plasma_current``: (nzeta_plasma, ntheta_plasma)
- ``Bnormal_from_net_coil_currents``: (nzeta_plasma, ntheta_plasma)

See also
--------

- ``docs/parity.rst`` for how output parity is validated.
- ``scripts/compare_nc.py`` for comparing two output netCDF files.
