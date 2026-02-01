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

Postprocessing:
- `postprocess_make_figures_and_vtk.py`: generates publication-ready figures and ParaView `.vtp` files
  (winding surface, plasma surface, cut coils, and filament field lines), plus optional `.vts`/`.vtu` outputs.
