Quadcoil-style objectives (∇Φ diagnostics)
====================================================

REGCOIL represents coils as **contours of the winding-surface current potential** :math:`\Phi(\theta,\zeta)`.
This has an important practical consequence:

*Many coil-quality metrics can be estimated directly from* :math:`\nabla_s \Phi` *on the winding surface*,
**without** explicitly cutting coil filaments.

This page documents the differentiable diagnostics implemented in
:src:`regcoil_jax/quadcoil_objectives.py` and the example
:ex:`examples/3_advanced/quadcoil_style_spacing_length_scan.py`.

Background and motivation
-------------------------

The Quadratic-constraint coil optimization literature (e.g. the ``quadcoil`` formulation)
uses the fact that many constraints/objectives can be written as **quadratic** functions
of the current-potential coefficients (or as quadratic pointwise constraints), enabling
efficient optimization workflows.

In ``regcoil_jax`` we keep the standard REGCOIL linear solve for :math:`\Phi` and add:

* differentiable *diagnostics* from :math:`\nabla_s \Phi`, and
* an optional quadratic regularizer on :math:`\int |\nabla_s \Phi|^2\,dA` (``gradphi2_weight``)
  that can be included in the REGCOIL matrix solve.

Surface gradient magnitude
--------------------------

Let the winding surface be parameterized by :math:`(\theta,\zeta)`. Let the surface metric tensor be

.. math::

  g = \begin{pmatrix}
    g_{\theta\theta} & g_{\theta\zeta} \\
    g_{\theta\zeta} & g_{\zeta\zeta}
  \end{pmatrix}

where

.. math::

  g_{\theta\theta} = \partial_\theta \mathbf{r}\cdot \partial_\theta \mathbf{r},\qquad
  g_{\zeta\zeta} = \partial_\zeta \mathbf{r}\cdot \partial_\zeta \mathbf{r},\qquad
  g_{\theta\zeta} = \partial_\theta \mathbf{r}\cdot \partial_\zeta \mathbf{r}.

The surface gradient magnitude is computed using the inverse metric:

.. math::

  |\nabla_s \Phi|^2
  = g^{\theta\theta}\,\Phi_\theta^2 + 2 g^{\theta\zeta}\,\Phi_\theta\,\Phi_\zeta + g^{\zeta\zeta}\,\Phi_\zeta^2.

Implementation: :src:`regcoil_jax/quadcoil_objectives.py` (``gradphi_squared``).

Coil-to-coil spacing estimate
-----------------------------

Coils cut from REGCOIL are approximately *equally spaced contours of* :math:`\Phi`.
If successive contours differ by :math:`\Delta \Phi`, the **local spacing** between contours is
approximately

.. math::

  \Delta s \approx \frac{\Delta \Phi}{|\nabla_s \Phi|}.

In REGCOIL-style cutting with ``coils_per_half_period``:

* number of coils on the full torus: :math:`N_{\mathrm{coils}} = 2\,N_{\mathrm{half}}\,N_{\mathrm{fp}}`,
* and the contour spacing in :math:`\Phi` is approximately
  :math:`\Delta\Phi \approx I_{\mathrm{net}}/N_{\mathrm{coils}}`, where :math:`I_{\mathrm{net}}` is the
  net poloidal current (``net_poloidal_current_Amperes``).

Implementation: :src:`regcoil_jax/quadcoil_objectives.py` (``quadcoil_metrics_from_*``) returns:

* ``coil_spacing_min``: :math:`\min(\Delta s)`,
* ``coil_spacing_rms``: area-weighted RMS of :math:`\Delta s`.

Total contour length estimate (coarea-inspired)
----------------------------------------------------------

There is a useful identity (coarea formula) connecting contour lengths to :math:`|\nabla_s \Phi|`:

.. math::

  \int_{S} |\nabla_s \Phi|\,dA
  = \int L(\Phi_0)\,d\Phi_0,

where :math:`L(\Phi_0)` is the length of the level set :math:`\{\Phi=\Phi_0\}`.

If contours are cut with uniform spacing :math:`\Delta\Phi`, then a practical estimate of the
**total contour length** is:

.. math::

  L_{\mathrm{total}} \approx \frac{1}{\Delta\Phi}\int_S |\nabla_s \Phi|\,dA.

Implementation: :src:`regcoil_jax/quadcoil_objectives.py` (``total_contour_length_est``).

Quadratic regularizer on ∫|∇Φ|² dA
------------------------------------------------

Since :math:`\Phi_\theta` and :math:`\Phi_\zeta` are *linear* in the Fourier coefficients, the
integral

.. math::

  \int_S |\nabla_s \Phi|^2\,dA

is a **quadratic form** in the coefficients and can be included as an extra regularization matrix
in the REGCOIL solve. In ``regcoil_jax`` this is enabled by setting:

* ``gradphi2_weight`` (float): adds ``gradphi2_weight * Q`` to the regularization matrix.

Implementation:

* matrix builder: :src:`regcoil_jax/quadcoil_objectives.py` (``build_gradphi2_matrix``)
* integrated in: :src:`regcoil_jax/build_matrices_jax.py` (``build_matrices``).

Example
-------

Run:

.. code-block:: bash

  regcoil_jax --platform cpu --verbose examples/3_advanced/regcoil_in.lambda_search_5_with_bnorm
  python examples/3_advanced/quadcoil_style_spacing_length_scan.py --platform cpu --cut_coils

The script produces:

* a tradeoff curve (:math:`\chi^2_B` vs :math:`\chi^2_K`),
* coil spacing estimate vs :math:`\lambda`,
* total coil length estimate vs :math:`\lambda`,
* optional cut coils + VTK outputs for ParaView.
