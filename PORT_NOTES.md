# regcoil_jax parity notes

## What was wrong (lambda_search_1 crash)

`examples/regcoil_in.lambda_search_1` previously failed with a JAX broadcasting error in `regcoil_jax/build_basis.py` when forming
`f3 = rze * dPhi_dtheta - rth * dPhi_dzeta`.

Root cause:
- The VMEC-offset coil surface path (`geometry_option_coil=2`) called `offset_surface_point()` with scalar `(theta, zeta)` inputs.
- `eval_surface_xyz_and_derivs()` returned a degenerate leading dimension for scalar inputs (e.g. `(1, 3)`), so the Newton iterations in
  `solve_zeta_plasma_for_target_toroidal_angle()` accidentally *grew array rank* each iteration via broadcasting (`() -> (1,) -> (1,1) -> ...`).
- That “singleton-dimension explosion” contaminated coil surface arrays (`r`, `rth`, `rze`, `nunit`, `normN`) with many hidden `1` axes, which later
  triggered the broadcast failure during basis assembly.

## Key fixes

- **Offset-surface scalar hygiene:** ensure scalar `(theta, zeta)` evaluations return shape `(3,)` and keep Newton iterates as scalar arrays.
- **Debuggability:** `--verbose` now prints plasma/coil surface shapes and (in `--debug_dir`) writes `surface_shapes.json`. In verbose/debug runs we
  assert the expected REGCOIL shape convention `(3, ntheta, nzeta)` early.
- **VMEC net-poloidal-current override:** when the plasma surface uses a VMEC `wout` file (`geometry_option_plasma=2/3/4`), override
  `net_poloidal_current_Amperes` with the VMEC-derived value, matching `regcoil_init_plasma_mod.f90`.
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

## How to validate parity vs original REGCOIL

Run the JAX port:
- `regcoil_jax --platform cpu --verbose examples/regcoil_in.compareToMatlab1`
- `regcoil_jax --platform cpu --verbose examples/regcoil_in.compareToMatlab1_option1`
- `regcoil_jax --platform cpu --verbose examples/regcoil_in.lambda_search_1`

Run the Fortran reference in the same `examples/` directory (rename one side’s output `.nc` files to avoid overwriting):
- `../../regcoil/regcoil regcoil_in.compareToMatlab1`
- `../../regcoil/regcoil regcoil_in.compareToMatlab1_option1`
- `../../regcoil/regcoil regcoil_in.lambda_search_1`

Compare key scalar arrays:
- `python tools/compare_nc.py fortran.nc jax.nc`

Tolerances used in CI (`regcoil_jax/tests/test_examples.py`):
- `rtol=1e-9`, `atol=1e-11` for `lambda`, `chi2_B`, `chi2_K`, `max_Bnormal`, `max_K`.

## Next steps (roadmap)

- **Output parity:** expand `regcoil_jax/io_output.py` to write the same variables/dimensions as `regcoil_write_output.f90` (not just the scalar arrays).
- **Input feature parity:** implement `load_bnorm` (`regcoil_read_bnorm.f90`), additional geometry options, and remaining regularization options
  (`Laplace-Beltrami`, `K_xy`, `K_zeta`).
- **JAX performance:** consider `jit`/`vmap` for the lambda scan solves; avoid large intermediate trig tensors where possible (e.g. FFT-based transforms),
  and add simple timing hooks for matrix assembly/solve.
- **Packaging:** improve `README.md`, add a GitHub Actions workflow running `pytest`, and add docs scaffolding (`docs/`, `.readthedocs.yaml`) once the API
  stabilizes.
