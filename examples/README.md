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

Figures + ParaView VTK (optional)
--------------------------------

The shared postprocessing script can optionally:
- generate publication-ready PNG figures
- cut filamentary coils (contours of `current_potential`)
- write ParaView-readable VTK files for surfaces/coils/field lines

Example::

```bash
python examples/3_advanced/postprocess_make_figures_and_vtk.py --run \
  --input examples/3_advanced/regcoil_in.lambda_search_1
```

You can independently disable figures or VTK outputs:

```bash
python examples/3_advanced/postprocess_make_figures_and_vtk.py --run --no_figures \
  --input examples/3_advanced/regcoil_in.lambda_search_1
python examples/3_advanced/postprocess_make_figures_and_vtk.py --run --no_vtk \
  --input examples/3_advanced/regcoil_in.lambda_search_1
```
