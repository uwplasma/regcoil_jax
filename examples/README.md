# Examples

This folder is organized into 3 tiers:

- `1_simple/`: analytic torus geometry; easiest to run and understand.
- `2_intermediate/`: JAX-first scripts (JIT/autodiff) demonstrating differentiable kernels and simple optimization loops.
- `3_advanced/`: VMEC-based cases (requires a `wout_*.nc` file) and lambda-search runs.

All `regcoil_in.*` inputs are intended to be run via:

```bash
regcoil_jax --platform cpu --verbose path/to/regcoil_in.some_case
```

The CLI writes `regcoil_out.some_case.nc` and `regcoil_out.some_case.log` next to the input file.

Tier wrappers (Python)
----------------------

Each tier includes a small wrapper that runs the CLI and can optionally postprocess:

- `examples/1_simple/run_cli_and_postprocess.py`
- `examples/2_intermediate/run_cli_and_postprocess.py`
- `examples/3_advanced/run_cli_and_postprocess.py`

Figures + ParaView VTK (optional)
--------------------------------

The shared postprocessing script can optionally:
- generate publication-ready PNG figures
- cut filamentary coils (contours of `current_potential`)
- write ParaView-readable VTK files for surfaces/coils/field lines
- (optionally) extract a Poincaré section and write `poincare_points.vtp`

Example::

```bash
python examples/3_advanced/postprocess_make_figures_and_vtk.py --run \
  --input examples/3_advanced/regcoil_in.lambda_search_1
```

Enable Poincaré section output:

```bash
python examples/3_advanced/postprocess_make_figures_and_vtk.py --run --poincare \
  --input examples/3_advanced/regcoil_in.lambda_search_1
```

For a one-command “full workflow” wrapper (solve → cut coils → field lines → Poincaré), see:

```bash
python examples/3_advanced/full_solve_cut_coils_and_poincare.py
```

New “beyond REGCOIL” demos
--------------------------

These scripts showcase workflows that are practical in a JAX rewrite (autodiff + optimization), beyond the
original Fortran REGCOIL distribution:

- Compare coils **with and without** autodiff winding-surface optimization, then cut coils, optimize per-coil currents,
  and generate Poincaré plots overlaid with the target surface:

  ```bash
  python examples/3_advanced/compare_winding_surface_optimization_cut_coils_currents_poincare.py
  ```

- Optimize a **hybrid source set**: a few simple filament loops + many point dipoles (proxy for windowpane coils / magnets),
  then generate ParaView outputs and Poincaré plots:

  ```bash
  python examples/3_advanced/hybrid_few_loops_many_dipoles_optimize_and_poincare.py
  ```

- Quadcoil-style diagnostics from the winding-surface potential (estimate coil spacing and total coil length from ∇Φ),
  with optional coil cutting and VTK output:

  ```bash
  python examples/3_advanced/quadcoil_style_spacing_length_scan.py --cut_coils
  ```

- Permanent magnets / dipole lattice workflow: place dipoles near the winding surface and solve for their moments to cancel
  ``Bnormal_from_plasma_current`` on the plasma surface:

  ```bash
  python examples/3_advanced/permanent_magnets_cancel_bplasma.py
  ```

- Real-geometry stellarator demos (VMEC wout or input.*): cut coils + optimize per-coil currents, and dipole fits:

  ```bash
  python examples/3_advanced/stellarators/LP2021_QA/run_qa_coil_design.py
  python examples/3_advanced/stellarators/LP2021_QA/run_qa_dipole_fit.py
  ```

- Differentiable coil cutting (approximate): soft contour relaxation and gradient-based tuning:

  ```bash
  python examples/3_advanced/differentiable_coil_cutting_softcontour.py
  ```

- Differentiable multi-coil cutting (topology-fixed): optimize smooth θ_k(ζ) curves to satisfy Φ-level constraints,
  then tune per-coil currents (end-to-end differentiable filament geometry):

  ```bash
  python examples/3_advanced/differentiable_coil_cutting_snakes_multicoil.py
  ```

- JAX-native fieldline tracing + soft Poincaré gradients (toy demo):

  ```bash
  python examples/3_advanced/jax_poincare_grad_demo.py
  ```

- Optimize per-coil currents with a differentiable (soft) Poincaré penalty:

  ```bash
  python examples/3_advanced/jax_optimize_currents_with_differentiable_poincare.py
  ```

You can independently disable figures or VTK outputs:

```bash
python examples/3_advanced/postprocess_make_figures_and_vtk.py --run --no_figures \
  --input examples/3_advanced/regcoil_in.lambda_search_1
python examples/3_advanced/postprocess_make_figures_and_vtk.py --run --no_vtk \
  --input examples/3_advanced/regcoil_in.lambda_search_1
```
