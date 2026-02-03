Differentiable Poincaré sections
================================

Classic Poincaré sections are a powerful visualization and analysis tool: trace a field line, then record its
intersections with a plane :math:`\\phi=\\phi_0` (mod :math:`2\\pi/N_{fp}`), plotting the resulting point set in :math:`(R,Z)`.

However, **exact** Poincaré extraction is *not differentiable* because it involves:

- discrete event detection (“did we cross the plane between step i and i+1?”),
- branch choices for periodic angles (unwrapping / modulo),
- a variable number of intersections.

For autodiff-based optimization, we therefore use a *smooth surrogate*.

JAX-native field line tracing
-----------------------------

We trace coil-only field lines with fixed-step RK4 in JAX:

.. math::

   \\frac{d\\mathbf{x}}{ds} = \\pm\\frac{\\mathbf{B}(\\mathbf{x})}{\\lVert\\mathbf{B}(\\mathbf{x})\\rVert},

where :math:`\\mathbf{B}` is computed from filament segments via Biot–Savart.

Implementation:

- ``regcoil_jax.fieldlines_jax.trace_fieldlines_rk4``

Soft Poincaré candidates (weighted point cloud)
-----------------------------------------------

Let :math:`\\phi_i = \\mathrm{atan2}(y_i,x_i)` for a traced sample :math:`\\mathbf{x}_i`. Define the phase

.. math::

   u_i = N_{fp}(\\phi_i-\\phi_0)

and an “event function”

.. math::

   r_i = \\sin(u_i).

In a classic extractor, a crossing occurs when :math:`r` changes sign between adjacent samples.
To avoid discrete sign tests, we form **candidate crossing points** for every segment and assign a smooth weight.

For a segment :math:`(\\mathbf{x}_0,\\mathbf{x}_1)` with :math:`(r_0,r_1)`:

.. math::

   t = \\frac{r_0}{r_0-r_1+\\varepsilon},\\qquad
   \\mathbf{p}= (1-t)\\mathbf{x}_0 + t\\mathbf{x}_1.

We define a smooth “is this a crossing?” factor:

.. math::

   w_{\\mathrm{cross}} = \\sigma(-\\alpha r_0 r_1),

which is close to 1 if the product is negative (opposite signs) and close to 0 otherwise.

We also require proximity to the plane:

.. math::

   w_{\\mathrm{near}} = \\exp\\big(-\\beta(r_0^2+r_1^2)\\big),

and select the desired plane :math:`\\phi=\\phi_0` rather than :math:`\\phi=\\phi_0+\\pi/N_{fp}` by favoring
:math:`\\cos(u_{mid})>0`:

.. math::

   w_{\\mathrm{plane}} = \\sigma(\\gamma\\cos(u_{mid})).

The final weight is:

.. math::

   w = w_{\\mathrm{cross}}\\,w_{\\mathrm{near}}\\,w_{\\mathrm{plane}}.

The output is a **weighted point cloud** of candidates. In ParaView you can threshold by ``weight`` to recover a plot
that looks like a standard Poincaré section, while preserving differentiability for optimization.

Implementation:

- ``regcoil_jax.fieldlines_jax.soft_poincare_candidates``

Practical example: optimize per-coil currents with a soft Poincaré penalty
--------------------------------------------------------------------------

The example:

.. code-block:: bash

   python examples/3_advanced/jax_optimize_currents_with_differentiable_poincare.py --platform cpu

shows a clear use-case:

1. run REGCOIL and cut coils,
2. optimize per-coil currents using autodiff,
3. add a penalty that pulls the soft Poincaré point cloud toward the plasma cross-section curve at :math:`\\phi=0`,
4. write publication-ready figures and ParaView point clouds (before/after).

This example also includes a stabilizing :math:`B\\cdot n` mismatch term that targets the REGCOIL surface-current field
:math:`B_{sv}` on the plasma surface (when it is available in the netCDF output).

For speed and robustness, the example optimizes a **topology-fixed current parameterization** (2 scalars):

- an overall current scale factor, and
- an even/odd coil current split.

Limitations
-----------

- This is a surrogate for optimization. For publication-quality Poincaré plots, use the discrete extractor in
  ``regcoil_jax.fieldlines.py`` (visualization-first).
- The surrogate assumes sufficient step resolution so candidate segments meaningfully bracket crossings.
- The coil-only field line model is intended for qualitative constraints in this repo (filament approximation).
