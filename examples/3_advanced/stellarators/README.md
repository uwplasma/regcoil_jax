# Stellarator examples (VMEC boundaries)

These examples demonstrate end-to-end coil design workflows for real VMEC geometries:

1) run `regcoil_jax` on a VMEC plasma boundary and an offset winding surface
2) cut discrete coils from the current potential
3) optimize per-coil currents (autodiff through Biot–Savart)
4) write figures + ParaView outputs

Run one of the case scripts:

```bash
python examples/3_advanced/stellarators/LP2021_QA/run_qa_coil_design.py
python examples/3_advanced/stellarators/LP2021_QH/run_qh_coil_design.py
python examples/3_advanced/stellarators/n3are_lowres/run_n3are_coil_design.py
```

Or run the generic driver directly:

```bash
python examples/3_advanced/stellarators/coil_design_cut_optimize.py --vmec path/to/wout_or_input
```

Notes:
- The driver accepts either a `wout_*.nc` file or a VMEC `input.*` file (boundary-only mode).
- Output folders are created next to the driver script and include:
  - `regcoil_out.*.nc`, `regcoil_out.*.log`
  - `coils_initial.*`, `coils_optimized.*` (MAKECOIL-style filament files)
  - `figures/` and `vtk/` subfolders

