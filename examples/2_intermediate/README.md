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

Included scripts:
- `jax_optimize_coil_minor_radius.py`: optimize `a_coil` for the `Bnormal_from_net_coil_currents` kernel (fastest).
- `jax_optimize_coil_radius_full_regcoil.py`: optimize `a_coil` while differentiating through matrix assembly + linear solve.
- `jax_optimize_then_cut_coils_torus.py`: end-to-end demo (autodiff optimization → CLI solve → coil cutting → figures/VTK), all from one script.
