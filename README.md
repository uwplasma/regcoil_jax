# regcoil_jax

JAX port of the Fortran **REGCOIL** codebase, with an initial focus on **parity-first correctness** while keeping the
forward pipeline compatible with JAX JIT + autodiff.

## Quickstart

From this folder:

```bash
pip install -e '.[dev]'
regcoil_jax --platform cpu --verbose examples/1_simple/regcoil_in.compareToMatlab1
regcoil_jax --platform cpu --verbose examples/1_simple/regcoil_in.compareToMatlab1_option1
regcoil_jax --platform cpu --verbose examples/3_advanced/regcoil_in.lambda_search_1
regcoil_jax --platform cpu --verbose examples/3_advanced/regcoil_in.lambda_search_5_with_bnorm
regcoil_jax --platform cpu --verbose examples/3_advanced/regcoil_in.compareToMatlab2_geometry_option_coil_3
regcoil_jax --platform cpu --verbose examples/2_intermediate/regcoil_in.torus_svd_scan
regcoil_jax --platform cpu --verbose examples/2_intermediate/regcoil_in.torus_nescout_diagnostics
pytest -q
```

Optional: also smoke-run the slow/high-resolution examples (will take much longer):

```bash
REGCOIL_JAX_RUN_SLOW=1 pytest -q
```

Postprocess into figures + ParaView files (coil cutting + field lines):

```bash
python examples/3_advanced/postprocess_make_figures_and_vtk.py --run --input examples/3_advanced/regcoil_in.lambda_search_1
```

Disable figures or ParaView outputs:

```bash
python examples/3_advanced/postprocess_make_figures_and_vtk.py --run --no_figures --input examples/3_advanced/regcoil_in.lambda_search_1
python examples/3_advanced/postprocess_make_figures_and_vtk.py --run --no_vtk --input examples/3_advanced/regcoil_in.lambda_search_1
```

Optional: include a 3D VTU point cloud of the coil-filament field magnitude:

```bash
python examples/3_advanced/postprocess_make_figures_and_vtk.py --run --write_vtu_point_cloud --input examples/3_advanced/regcoil_in.lambda_search_1
```

JAX-specific demo (autodiff + JIT on a physics kernel):

```bash
python examples/2_intermediate/jax_optimize_coil_minor_radius.py
```

JAX-specific demo (autodiff through matrix build + linear solve):

```bash
python examples/2_intermediate/jax_optimize_coil_radius_full_regcoil.py
```

JAX-specific demo (autodiff through VMEC offset-surface geometry + matrix build + solve):

```bash
python examples/3_advanced/jax_optimize_separation_vmec_offset.py
```

## Status
- ✅ CLI runs the 3 example inputs above and writes `regcoil_out.*.nc` + `regcoil_out.*.log` next to the input file.
- ✅ NetCDF output schema + values are regression-tested against committed Fortran reference `.nc` files (`tests/fortran_outputs/`) for a curated set of inputs.
- ✅ Supports `load_bnorm=.true.` (BNORM file) to include a nonzero `Bnormal_from_plasma_current`.
- ✅ Supports `regularization_term_option` in `{ "chi2_K", "K_xy", "Laplace-Beltrami" }` (see `docs/theory.rst`).
- ✅ Supports `general_option` in `{1,2,3,4,5}` (lambda scan, NESCOIL `nescout` diagnostics, truncated SVD scan, and lambda search).
- ✅ Additional geometry options: plasma `geometry_option_plasma` in `{0,1,2,3,6,7}` and coil `geometry_option_coil` in `{0,1,2,3,4}`.
- ✅ Includes pytest regression tests (`tests/`) that run `regcoil_jax` and compare against stored Fortran outputs.
- ✅ Includes a coil-cutting + VTK postprocessing example (`docs/visualization.rst`).

See `PORTING_NOTES.md` for details on what was fixed and how to validate parity.
