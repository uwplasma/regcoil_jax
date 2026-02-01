Usage
=====

Run the CLI on CPU::

  regcoil_jax --platform cpu --verbose examples/3_advanced/regcoil_in.lambda_search_1

Key inputs
----------

The Fortran REGCOIL namelist has many options; this parity-first port currently supports a subset. A few
commonly-used keys:

- ``regularization_term_option``: ``"chi2_K"`` (default), ``"K_xy"``, or ``"Laplace-Beltrami"``
- ``geometry_option_plasma``: ``1`` (analytic torus) or ``2`` (VMEC boundary)
- ``geometry_option_coil``: ``1`` (analytic torus), ``2`` (VMEC offset surface), or ``3`` (NESCOIL ``nescin`` winding surface)
- ``general_option=5``: lambda search (see the ``examples/3_advanced`` inputs)

Lambda search targets
---------------------

For ``general_option=5``, REGCOIL searches for a ``lambda`` such that a chosen diagnostic hits ``target_value``.
This port supports the same target options as the Fortran code:

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

Outputs
-------

For an input file named ``regcoil_in.<case>``, the CLI writes next to the input:

- ``regcoil_out.<case>.nc``
- ``regcoil_out.<case>.log``

Tests
-----

From the project root::

  pytest -q
