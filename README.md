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
pytest -q
```

JAX-specific demo (autodiff + JIT on a physics kernel):

```bash
python examples/2_intermediate/jax_optimize_coil_minor_radius.py
```

## Status
- ✅ CLI runs the 3 example inputs above and writes `regcoil_out.*.nc` + `regcoil_out.*.log` next to the input file.
- ✅ Key scalar arrays (`lambda`, `chi2_B`, `chi2_K`, `max_Bnormal`, `max_K`) match the reference Fortran REGCOIL outputs for these cases.
- ✅ Includes pytest regression tests (`tests/`) with stored Fortran baselines.

See `PORTING_NOTES.md` for details on what was fixed and how to validate parity.
