Visualization and Coil Cutting
==============================

Publication-ready figures
------------------------

The repository includes a pedagogic postprocessing script that generates figures from a run:

::

  python examples/3_advanced/postprocess_make_figures_and_vtk.py --run \\
    --input examples/3_advanced/regcoil_in.lambda_search_1

It writes:

- ``figures_<case>/``: PNG figures (lambda scan, 2D maps of :math:`B_n`, :math:`|K|`, and :math:`\\Phi`)
- ``vtk_<case>/``: ParaView-readable ``.vtp`` PolyData files

VTK / ParaView
--------------

The postprocessing script writes these VTK files:

- ``coil_surface.vtp``: winding surface mesh with point-data (e.g. ``Phi``, ``Kmag``)
- ``plasma_surface.vtp``: plasma surface mesh with point-data (e.g. ``Bnormal``)
- ``coils.vtp``: filamentary coils obtained by cutting contours of the current potential
- ``fieldlines.vtp``: field line traces of the *filament-coil* approximation

Open them in ParaView and use the normal visualization pipeline (surface coloring, clipping, etc).

Cutting coils (filament extraction)
-----------------------------------

REGCOIL represents the coil currents as a surface current density
:math:`\\mathbf{K} = \\mathbf{n} \\times \\nabla \\Phi` on a winding surface.

To obtain discrete filament coils, a standard approach is to take contours of the total current potential
:math:`\\Phi(\\theta,\\zeta)`.

This repository includes a small implementation in ``regcoil_jax/coil_cutting.py`` and exposes it through
the postprocessing script above, which also writes a MAKECOIL-style ``coils.<case>`` file.

Field line tracing
------------------

The field line example uses a simple Biot–Savart midpoint rule on straight segments (coil filaments only),
implemented in ``regcoil_jax/fieldlines.py``. This is intended for visualization and sanity checks.
