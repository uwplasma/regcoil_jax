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
pytest -q
```

Postprocess into figures + ParaView files (coil cutting + field lines):

```bash
python examples/3_advanced/postprocess_make_figures_and_vtk.py --run --input examples/3_advanced/regcoil_in.lambda_search_1
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
- ✅ Includes pytest regression tests (`tests/`) with stored Fortran baselines.
- ✅ Includes a coil-cutting + VTK postprocessing example (`docs/visualization.rst`).

See `PORTING_NOTES.md` for details on what was fixed and how to validate parity.
