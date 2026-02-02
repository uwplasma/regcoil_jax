# 3_advanced

VMEC-based examples (require a `wout_*.nc` file) and more expensive runs.

These inputs demonstrate:
- plasma/coil surfaces derived from VMEC Fourier coefficients
- coil surface built as an offset surface from the plasma boundary
- coil surface read from a NESCOIL `nescin` file (`geometry_option_coil=3`)
- automatic lambda search (`general_option=5`)

Some cases also load a BNORM file (`load_bnorm=.true.`) to include a nonzero
`Bnormal_from_plasma_current` target field.

Larger showcase case (not CI-tested):
- `regcoil_in.regcoilPaper_figure10d_originalAngle_loRes`: W7-X-like configuration with a NESCOIL winding surface + BNORM.
- `regcoil_in.regcoilPaper_figure10d_geometry_option_coil_4_loRes`: same equilibrium, but with `geometry_option_coil=4` (constant-arclength theta).

Postprocessing:
- `postprocess_make_figures_and_vtk.py`: generates publication-ready figures and ParaView `.vtp` files
  (winding surface, plasma surface, cut coils, and filament field lines), plus optional `.vts`/`.vtu` outputs.
- `run_cli_and_postprocess.py`: wrapper that runs the CLI and (optionally) calls the postprocess script.
- `compare_fortran_and_jax.py`: runs the local Fortran `regcoil` binary and `regcoil_jax` on the same input and compares outputs.
- `jax_optimize_separation_vmec_offset.py`: autodiff demo that optimizes the offset-surface separation parameter.

Postprocess toggles:
- Skip figures: `--no_figures`
- Skip ParaView outputs: `--no_vtk`
- Skip coil cutting: `--no_coils`
- Skip field line tracing: `--no_fieldlines`
