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
