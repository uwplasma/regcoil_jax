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
