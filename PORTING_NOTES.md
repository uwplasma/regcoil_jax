# regcoil_jax porting notes

This repository is a parity-first JAX rewrite of the Fortran **REGCOIL** code.

The goals are:
- Match Fortran REGCOIL outputs for the same inputs (within tight tolerances).
- Keep the forward pipeline compatible with JAX JIT + autodiff.
- Provide a clean, pedagogic examples suite showing both parity and “JAX-native” workflows.

## Current parity scope (tested)

For the example inputs in `examples/` (see folder structure below), the CI/test suite checks that these scalar
diagnostic arrays match the reference Fortran REGCOIL output files:

- `lambda`
- `chi2_B`
- `chi2_K`
- `max_Bnormal`
- `max_K`

The tests also verify output-file self-consistency by recomputing these quantities from the written 2D fields
(`Bnormal_total`, `K2`, grid spacings, and surface Jacobians) and checking agreement.

This parity set now includes a VMEC + BNORM example (`load_bnorm=.true.`), so nonzero `Bnormal_from_plasma_current`
is covered by regression tests.

Additional parity-tested control-flow paths:
- `general_option=2`: diagnostics for one or more NESCOIL `nescout` current potentials (example: `examples/2_intermediate/regcoil_in.torus_nescout_diagnostics`).
- `general_option=3`: NESCOIL-style truncated SVD scan (example: `examples/2_intermediate/regcoil_in.torus_svd_scan`).
- `general_option=4`: lambda search without feasibility checks (example: `examples/2_intermediate/regcoil_in.lambda_search_option4_torus`).

## Example folder structure

Examples are organized by “difficulty” / required physics machinery:

- `examples/1_simple/`: analytic torus geometry, small grids, easiest to understand.
- `examples/2_intermediate/`: JAX-first scripts (JIT/autodiff) that demonstrate differentiable kernels and small optimizations.
- `examples/3_advanced/`: VMEC-based cases (requires a `wout_*.nc` file), lambda search, and more expensive runs.

All `regcoil_in.*` files are meant to be runnable via:

```bash
regcoil_jax --platform cpu --verbose path/to/regcoil_in.some_case
```

The CLI writes `regcoil_out.some_case.nc` and `regcoil_out.some_case.log` *next to the input file*.

## What was wrong (lambda_search_1 crash)

`examples/3_advanced/regcoil_in.lambda_search_1` previously failed with a JAX broadcasting error in
`regcoil_jax/build_basis.py` when forming:

`f3 = rze * dPhi_dtheta - rth * dPhi_dzeta`.

Root cause:
- The VMEC-offset coil surface path (`geometry_option_coil=2`) called `offset_surface_point()` with scalar
  `(theta, zeta)` inputs.
- `eval_surface_xyz_and_derivs()` returned a degenerate leading dimension for scalar inputs (e.g. `(1, 3)`), so the Newton iterations in
  `solve_zeta_plasma_for_target_toroidal_angle()` accidentally *grew array rank* each iteration via broadcasting (`() -> (1,) -> (1,1) -> ...`).
- That “singleton-dimension explosion” contaminated coil surface arrays (`r`, `rth`, `rze`, `nunit`, `normN`) with many hidden `1` axes, which later
  triggered the broadcast failure during basis assembly.

## Key fixes

- **Offset-surface scalar hygiene:** ensure scalar `(theta, zeta)` evaluations return shape `(3,)` and keep Newton iterates as scalar arrays.
- **Debuggability:** `--verbose` prints plasma/coil surface shapes and (in `--debug_dir`) writes `surface_shapes.json`.
  In verbose/debug runs we assert the expected REGCOIL shape convention `(3, ntheta, nzeta)` early.
- **VMEC net-poloidal-current override:** when the plasma surface uses a VMEC `wout` file (`geometry_option_plasma=2/3/4`),
  override `net_poloidal_current_Amperes` with the VMEC-derived value (matching `regcoil_init_plasma_mod.f90`).
- **Correct secular term `d` vector:** fix the swapped `(rth, rze)` usage so `d = (G * dr/dtheta - I * dr/dzeta) / (2π)` matches
  `regcoil_build_matrices.f90`.
- **VMEC offset coil surface parity:** for `geometry_option_coil=2`, fit the offset surface to a truncated Fourier representation
  (`mpol_coil`, `ntor_coil`, filtering), then evaluate `r`, `dr/dtheta`, `dr/dzeta` from that Fourier surface (matching
  `regcoil_init_coil_surface.f90`).
- **Kernel sign convention:** fix `dr` direction in the `h` kernel to match REGCOIL (`dr = r_plasma - r_coil`), which was flipping the sign of
  `Bnormal_from_net_coil_currents` and breaking `chi2_B` diagnostics for the `lambda=∞` solve.
- **Lambda handling parity:**
  - `general_option=1`: `lambda = [0, logspace(lambda_min, lambda_max, nlambda-1)]` (matches `regcoil_compute_lambda.f90`)
  - `general_option=5`: ported Brent-style lambda search (matches `regcoil_auto_regularization_solve.f90`)
  - Note: for strict netCDF parity, the port reproduces the Fortran edge-case `nlambda=2 => lambda(2)=NaN` from `regcoil_compute_lambda.f90`.

## Validate parity vs original REGCOIL (manual)

Run the JAX port (paths will depend on which tier you pick):

```bash
regcoil_jax --platform cpu --verbose examples/1_simple/regcoil_in.compareToMatlab1
regcoil_jax --platform cpu --verbose examples/1_simple/regcoil_in.compareToMatlab1_option1
regcoil_jax --platform cpu --verbose examples/3_advanced/regcoil_in.lambda_search_1
```

Run the Fortran reference in the same directory (rename one side’s output `.nc` files to avoid overwriting):

```bash
../../regcoil/regcoil regcoil_in.compareToMatlab1
../../regcoil/regcoil regcoil_in.compareToMatlab1_option1
../../regcoil/regcoil regcoil_in.lambda_search_1
```

Compare key scalar arrays:

```bash
python tools/compare_nc.py fortran.nc jax.nc
```

Tolerances used in CI (`tests/test_examples.py`):
- `rtol=1e-9`, `atol=1e-11` for `lambda`, `chi2_B`, `chi2_K`, `max_Bnormal`, `max_K`.

## Notes on repository state

The previous development folder `regcoil_jax/` (outside this git repo) is now redundant: the current `regcoil_jax_git/`
tree contains the same tracked source/config/docs/tests content (excluding build artifacts and output files).

## Next steps (roadmap)

- **Input parity:** additional geometry options (e.g. NESCOIL / `nescin`) and remaining regularization options
  (`Laplace–Beltrami`, `K_xy`, `K_zeta`).
- **Output parity:** write a larger subset of `regcoil_write_output.f90` variables/dimensions (beyond the current “scalar parity set”
  + a few field arrays for self-consistency tests).
- **JAX-native performance:** reduce Python loops in solve paths (`lax.scan`/`lax.map` where appropriate), avoid rebuilding matrices across lambdas,
  and add simple timing hooks for matrix assembly/solve.
- **Docs:** expand `docs/theory.rst` with a full derivation and map each equation to code symbols/functions.
