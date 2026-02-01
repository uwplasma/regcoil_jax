Usage
=====

Run the CLI on CPU::

  regcoil_jax --platform cpu --verbose examples/3_advanced/regcoil_in.lambda_search_1

Key inputs
----------

The Fortran REGCOIL namelist has many options; this parity-first port currently supports a subset. A few
commonly-used keys:

- ``regularization_term_option``: ``"chi2_K"`` (default), ``"K_xy"``, or ``"Laplace-Beltrami"``
- ``geometry_option_plasma`` and ``geometry_option_coil``: ``1`` (analytic torus) or ``2`` (VMEC boundary / VMEC-offset)
- ``general_option=5``: lambda search (see the ``examples/3_advanced`` inputs)

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
