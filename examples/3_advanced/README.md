# 3_advanced

VMEC-based examples (require a `wout_*.nc` file) and more expensive runs.

These inputs demonstrate:
- plasma/coil surfaces derived from VMEC Fourier coefficients
- coil surface built as an offset surface from the plasma boundary
- automatic lambda search (`general_option=5`)

Some cases also load a BNORM file (`load_bnorm=.true.`) to include a nonzero
`Bnormal_from_plasma_current` target field.

Postprocessing:
- `postprocess_make_figures_and_vtk.py`: generates publication-ready figures and ParaView `.vtp` files
  (winding surface, plasma surface, cut coils, and filament field lines).
