Design notes (JAX-native implementation)
========================================

This page documents the *program structure* of ``regcoil_jax`` and the main JAX-related design constraints.
It is intended for contributors and advanced users who want to extend the code.

High-level pipeline
-------------------

The forward pipeline mirrors the Fortran code:

1. Parse a Fortran-style namelist (``regcoil_in.*``).
2. Construct plasma and winding surfaces (analytic, VMEC, EFIT, tables).
3. Assemble dense least-squares matrices for the current potential coefficients.
4. Solve for one or more :math:`\\lambda` values.
5. Write a Fortran-style netCDF output (schema parity for the covered set).

Key modules:

- IO and CLI: ``regcoil_jax/cli.py``, ``regcoil_jax/run.py``
- Surface construction: ``regcoil_jax/surfaces.py`` and geometry readers in ``regcoil_jax/io_*``
- Matrix assembly: ``regcoil_jax/build_matrices_jax.py``
- Linear solves + diagnostics: ``regcoil_jax/solve_jax.py``
- NetCDF output: ``regcoil_jax/io_output.py``

JAX constraints and choices
---------------------------

``float64`` and numerical parity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For parity-first correctness the default configuration enables ``jax_enable_x64=True`` and sets matmul precision
to ``highest``. Tests assume this.

Static shapes and JIT
~~~~~~~~~~~~~~~~~~~~~

JAX compilation works best when array shapes and loop lengths are static.
In practice:

- grid sizes (``ntheta_*``, ``nzeta_*``) and basis sizes (``mpol_potential``, ``ntor_potential``) are treated as
  *static* parameters for the compiled solve path,
- the lambda-scan solve uses a JIT-compiled ``vmap`` over lambdas.

Avoiding Python loops in hot paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following are performance-critical and are implemented in JAX without Python loops:

- batch solve for many :math:`\\lambda` values (``solve_for_lambdas``),
- field/current diagnostics for many solutions (vectorized),
- Adam-style optimization loops in examples (``jax.lax.scan``).

Some components remain intentionally Python-level because they are:

- inherently discrete and non-differentiable (exact coil cutting via contouring), or
- pure IO (netCDF and text files).

Differentiability boundaries
----------------------------

The core REGCOIL solve pipeline is differentiable end-to-end:

geometry → matrices → solve → diagnostics.

Two important postprocessing steps are *not* differentiable in their classic form:

1. exact contour-based coil cutting (marching squares), and
2. Poincaré section extraction (discrete crossing detection).

This repo therefore provides:

- robust workflows that **do not differentiate through cutting**, and
- research/pedagogic *relaxations* that keep filament geometry differentiable
  (see ``docs/differentiable_coil_cutting.rst``).

Adding new geometry options
---------------------------

New geometry readers should return a ``FourierSurface`` (or compatible evaluated surface arrays) with consistent
shapes. The most common integration point is:

- add a reader in ``regcoil_jax/io_*.py``,
- add an option case in ``regcoil_jax/surfaces.py``.

Then update:

- ``docs/feature_matrix.rst`` (coverage),
- examples and parity tests if applicable.
