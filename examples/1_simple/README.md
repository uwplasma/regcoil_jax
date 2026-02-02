# 1_simple

Small, analytic geometry examples intended as the first place to start.

These inputs use simple (non-VMEC) torus geometry options and run quickly.

CLI (no figures/VTK)
--------------------

Run any input with::

  regcoil_jax --platform cpu --verbose examples/1_simple/regcoil_in.compareToMatlab1

This writes `regcoil_out.*.{nc,log}` next to the input file.

Python wrapper (optional figures/VTK)
------------------------------------

Run the CLI only (default)::

  python examples/1_simple/run_cli_and_postprocess.py

Run and also postprocess into figures + ParaView files::

  python examples/1_simple/run_cli_and_postprocess.py --postprocess

You can disable outputs individually::

  python examples/1_simple/run_cli_and_postprocess.py --postprocess --no_figures
  python examples/1_simple/run_cli_and_postprocess.py --postprocess --no_vtk

Suggested starting points:
- `regcoil_in.compareToMatlab1`: basic parity case.
- `regcoil_in.axisymmetrySanityTest_chi2K_regularization`: sanity check with `chi2_K` regularization.
- `regcoil_in.axisymmetrySanityTest_Laplace_Beltrami_regularization`: same geometry but using `Laplace-Beltrami` regularization.
- `regcoil_in.torus_K_zeta_regularization`: same analytic torus, but regularizing only the toroidal (zeta) component of the current density (`K_zeta`).
- `regcoil_in.plasma_option_6_fourier_table`: plasma `geometry_option_plasma=6` (ASCII Fourier table).
- `regcoil_in.plasma_option_7_focus_embedded_bnorm`: plasma `geometry_option_plasma=7` (FOCUS boundary) with embedded Bn coefficients.
