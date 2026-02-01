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
pytest -q
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

## Status
- ✅ CLI runs the 3 example inputs above and writes `regcoil_out.*.nc` + `regcoil_out.*.log` next to the input file.
- ✅ Key scalar arrays (`lambda`, `chi2_B`, `chi2_K`, `max_Bnormal`, `max_K`) match the reference Fortran REGCOIL outputs for these cases.
- ✅ Supports `load_bnorm=.true.` (BNORM file) to include a nonzero `Bnormal_from_plasma_current`.
- ✅ Supports `regularization_term_option` in `{ "chi2_K", "K_xy", "Laplace-Beltrami" }` (see `docs/theory.rst`).
- ✅ Additional geometry options: plasma `geometry_option_plasma` in `{0,1,2,3,6,7}` and coil `geometry_option_coil` in `{0,1,2,3,4}`.
- ✅ Includes pytest regression tests (`tests/`) with stored Fortran baselines.
- ✅ Includes a coil-cutting + VTK postprocessing example (`docs/visualization.rst`).

See `PORTING_NOTES.md` for details on what was fixed and how to validate parity.
