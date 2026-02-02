Optimization and autodiff
=========================

REGCOIL’s Fortran implementation includes adjoint-based sensitivity workflows for coil-surface parameterizations.
In ``regcoil_jax``, the preferred approach is to use **automatic differentiation (autodiff)**:

- differentiate through geometry construction (e.g. VMEC offset surface),
- matrix assembly,
- dense linear solve,
- diagnostics.

This section documents the main autodiff/optimization entry points and the examples that exercise them.

Winding surface optimization (autodiff)
---------------------------------------

The most direct analogue of “winding surface optimization” is optimizing a winding surface parameterization.

This repo includes a spatially varying separation-field optimizer (Fourier-parameterized separation ``sep(θ,ζ)``)
in ``regcoil_jax/winding_surface_optimization.py`` and an end-to-end demo:

``examples/3_advanced/winding_surface_autodiff_optimize_and_visualize.py``

That script:

1. runs REGCOIL “before”
2. optimizes ``sep(θ,ζ)`` using autodiff
3. runs REGCOIL “after”
4. writes publication-style figures and ParaView files (winding surface + coils + field lines)

Per-coil current optimization after cutting
-------------------------------------------

Cutting coils from a current potential is a **geometric postprocessing** step (contouring). It is not smooth
and is not intended to be differentiated through. However, once coils are cut, many downstream problems become
smooth in parameters like:

- a current scale factor for each coil,
- small geometric perturbations to coil points (advanced).

The example:

``examples/2_intermediate/jax_optimize_cut_coil_currents_and_visualize.py``

demonstrates:

- run REGCOIL_JAX to obtain a current potential
- cut coils (filaments) from current-potential contours
- optimize an independent current for each coil using autodiff through a Biot–Savart filament model
- write VTK files (coils + 3D field lines + Poincaré points) and figures

The implementation lives in:

- ``regcoil_jax/biot_savart_jax.py`` (JAX Biot–Savart on fixed segment geometry)
- ``regcoil_jax/coil_current_optimization.py`` (autodiff objective + Adam)
- ``regcoil_jax/optimize.py`` (small Adam implementation; avoids extra deps)

Poincaré plots and 3D visualization
-----------------------------------

The postprocessing pipeline supports ParaView outputs for:

- winding surface and plasma surface (`.vtp` + `.vts`)
- cut coils (`coils.vtp`)
- coil-only field lines (`fieldlines.vtp`)
- Poincaré section points (`poincare_points.vtp`)

Enable Poincaré output in the shared postprocess script with ``--poincare``.

