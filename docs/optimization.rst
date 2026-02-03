Optimization and autodiff
=========================

REGCOIL’s Fortran implementation includes adjoint-based sensitivity workflows for coil-surface parameterizations.
In ``regcoil_jax``, the preferred approach is to use **automatic differentiation (autodiff)**:

- differentiate through geometry construction (e.g. VMEC offset surface),
- matrix assembly,
- dense linear solve,
- diagnostics.

This section documents the main autodiff/optimization entry points and the examples that exercise them.

Sensitivity outputs (omega / ``dchi2domega``)
---------------------------------------------

The Fortran REGCOIL code can write *sensitivities of the objective* with respect to a Fourier
parameterization of the winding surface (often used by the bundled winding-surface optimization scripts).

In ``regcoil_jax``, these sensitivities are produced using **autodiff** through the full pipeline
(surface evaluation → matrix assembly → dense solve → diagnostics) when ``sensitivity_option > 1``.

Key inputs:

- ``sensitivity_option``: enable sensitivity output variables (``>1``)
- ``mmax_sensitivity``, ``nmax_sensitivity``: Fourier truncation for the sensitivity mode list (both must be ``>=1``)
- ``sensitivity_symmetry_option``: which Fourier coefficient families are included (matches Fortran):

  - ``1``: stellarator symmetric (``omega_coil`` in {1=rmnc, 2=zmns})
  - ``2``: even in ``theta`` and ``zeta`` (``omega_coil`` in {3=rmns, 4=zmnc})
  - ``3``: no symmetry (all families)

Example input (strict netCDF parity-tested against Fortran output):

:ex:`examples/2_intermediate/regcoil_in.torus_sensitivity_option2_small`

Winding surface optimization (autodiff)
---------------------------------------

The most direct analogue of “winding surface optimization” is optimizing a winding surface parameterization.

This repo includes a spatially varying separation-field optimizer (Fourier-parameterized separation ``sep(θ,ζ)``)
in :src:`regcoil_jax/winding_surface_optimization.py` and an end-to-end demo:

:ex:`examples/3_advanced/winding_surface_autodiff_optimize_and_visualize.py`

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

:ex:`examples/2_intermediate/jax_optimize_cut_coil_currents_and_visualize.py`

demonstrates:

- run REGCOIL_JAX to obtain a current potential
- cut coils (filaments) from current-potential contours
- optimize an independent current for each coil using autodiff through a Biot–Savart filament model
- write VTK files (coils + 3D field lines + Poincaré points) and figures

The implementation lives in:

- :src:`regcoil_jax/biot_savart_jax.py` (JAX Biot–Savart on fixed segment geometry)
- :src:`regcoil_jax/coil_current_optimization.py` (autodiff objective + Adam)
- :src:`regcoil_jax/optimize.py` (small Adam implementation; avoids extra deps)

Robust current optimization (misalignments)
-------------------------------------------

Because the filament Biot–Savart model is fully differentiable, it is straightforward to optimize
**robust** objectives that average performance over an ensemble of perturbations.

This is a natural fit for JAX because the ensemble dimension can be handled with ``jax.vmap``.

The example:

:ex:`examples/3_advanced/jax_robust_optimize_cut_coil_currents_misalignment.py`

optimizes per-coil currents to reduce the mean (and a small variance penalty) of the filament-model
normal-field error under random per-coil rigid misalignments.

Rigid alignment optimization (geometry via autodiff)
----------------------------------------------------

To show how geometry parameters can participate in autodiff optimization (without any Fortran adjoints),
the example:

:ex:`examples/3_advanced/jax_optimize_coilset_rigid_alignment.py`

optimizes a small rigid-transform parameterization (rotation about :math:`z` plus translation) applied to a
cut coil set, minimizing a normal-field error objective evaluated via a JAX Biot–Savart filament model.

This is a simplified analogue of “finite-build / coil placement” sensitivity ideas explored in the FOCUS/FOCUSADD literature.

Hybrid optimizations (filaments + dipoles)
------------------------------------------

The following “beyond REGCOIL” demo introduces **point dipoles** as a differentiable proxy for
small local coils / windowpane coils / permanent magnets:

:ex:`examples/3_advanced/hybrid_few_loops_many_dipoles_optimize_and_poincare.py`

It optimizes:

- a small set of filament-loop currents, and
- many dipole moments,

to minimize :math:`B\cdot n` on a target surface, then generates ParaView outputs and Poincaré plots.

See also :doc:`hybrid_design` for the dipole equations and objective definition.

Quadcoil-style objectives (coil spacing / length from ∇Φ)
---------------------------------------------------------------

Some coil-quality metrics can be computed directly from the winding-surface current potential :math:`\Phi(\theta,\zeta)`
and its surface gradient, without explicitly cutting coils. This makes them convenient for:

- diagnostics during lambda scans, and
- differentiable regularizers on the current potential.

See:

- :doc:`quadcoil_objectives` (derivations and normalizations)
- :src:`regcoil_jax/quadcoil_objectives.py` (implementations)
- :ex:`examples/3_advanced/quadcoil_style_spacing_length_scan.py` (end-to-end demo + plots)

Permanent magnets (dipole lattices)
-----------------------------------

Permanent magnets (or small coillets) can be modeled as a lattice of point dipoles placed on / near
the winding surface, with their moments optimized to cancel :math:`B_{\mathrm{plasma}}\cdot n` or other targets.

See:

- :doc:`permanent_magnets`
- :src:`regcoil_jax/permanent_magnets.py`
- :ex:`examples/3_advanced/permanent_magnets_cancel_bplasma.py`

Differentiable coil cutting (research demo)
------------------------------------------------

Coil cutting via exact contour extraction is non-differentiable. This repo includes a **pedagogic**
relaxation based on soft contour extraction (single-valued :math:`\theta(\zeta)`), intended for
research and experimentation:

- :doc:`differentiable_coil_cutting`
- :ex:`examples/3_advanced/differentiable_coil_cutting_softcontour.py`

For a more robust (topology-fixed) differentiable coil *set* extraction, see:

- :ex:`examples/3_advanced/differentiable_coil_cutting_snakes_multicoil.py`

Poincaré plots and 3D visualization
-----------------------------------

The postprocessing pipeline supports ParaView outputs for:

- winding surface and plasma surface (`.vtp` + `.vts`)
- cut coils (`coils.vtp`)
- coil-only field lines (`fieldlines.vtp`)
- Poincaré section points (`poincare_points.vtp`)

Enable Poincaré output in the shared postprocess script with ``--poincare``.

JAX-native Poincaré (soft) for optimization
-------------------------------------------

The classic Poincaré plot pipeline is discrete: it detects crossings of a field line with a plane and interpolates
to find the crossing point. This is not differentiable.

For autodiff-based objectives, ``regcoil_jax`` includes a smooth alternative:

- trace field lines in JAX: ``regcoil_jax.fieldlines_jax.trace_fieldlines_rk4``
- assign smooth weights to points near a section plane: ``poincare_section_weights``
- compute differentiable soft section statistics: ``poincare_weighted_RZ``

The demo :ex:`examples/3_advanced/jax_poincare_grad_demo.py` computes a toy differentiable objective and prints
the gradient norm with respect to per-coil currents.

For a fuller write-up and a practical end-to-end example, see :doc:`differentiable_poincare`.
