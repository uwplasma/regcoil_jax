# 2_intermediate

JAX-first examples that highlight differentiability, gradients, and JIT.

These scripts are not necessarily “full REGCOIL solves”; some focus on individual kernels
to keep the objective functions simple and fast to run.

CLI (intermediate torus case)
-----------------------------

This tier also includes a slightly larger torus run that is still VMEC-free::

  regcoil_jax --platform cpu --verbose examples/2_intermediate/regcoil_in.torus_cli_intermediate

Optional figures/VTK (coil cutting, ParaView files) via the shared postprocess script::

  python examples/3_advanced/postprocess_make_figures_and_vtk.py --run --input examples/2_intermediate/regcoil_in.torus_cli_intermediate

Or use the wrapper (runs CLI, and optionally postprocesses)::

  python examples/2_intermediate/run_cli_and_postprocess.py
  python examples/2_intermediate/run_cli_and_postprocess.py --postprocess --no_vtk

Additional CLI inputs in this tier:
- `examples/2_intermediate/regcoil_in.torus_svd_scan`: REGCOIL `general_option=3` (NESCOIL-style truncated SVD scan).
- `examples/2_intermediate/regcoil_in.torus_nescout_diagnostics`: REGCOIL `general_option=2` (diagnostics for a NESCOIL `nescout` current potential).
- `examples/2_intermediate/regcoil_in.lambda_search_option4_torus`: REGCOIL `general_option=4` (lambda search without feasibility checks).
- `examples/2_intermediate/regcoil_in.plasma_option_4_vmec_straight_fieldline`: plasma `geometry_option_plasma=4` (VMEC straight-field-line poloidal coordinate).
- `examples/2_intermediate/regcoil_in.plasma_option_5_efit_lcfs`: plasma `geometry_option_plasma=5` (EFIT gfile, LCFS via `efit_psiN=1.0`).

Included scripts:
- `jax_optimize_coil_minor_radius.py`: optimize `a_coil` for the `Bnormal_from_net_coil_currents` kernel (fastest).
- `jax_optimize_coil_radius_full_regcoil.py`: optimize `a_coil` while differentiating through matrix assembly + linear solve.
- `jax_optimize_then_cut_coils_torus.py`: end-to-end demo (autodiff optimization → CLI solve → coil cutting → figures/VTK), all from one script.
- `jax_optimize_cut_coil_currents_and_visualize.py`: run → cut coils → optimize a different current for each coil (autodiff through Biot–Savart) → VTK/Poincaré.
- `differentiable_contouring_relaxations_demo.py`: toy torus + toy scalar field; compares relaxations for contour extraction.
- `differentiable_isosurface_marching_cubes_demo.py`: 3D analogue: soft marching-cubes candidates + lightweight mesh output.
