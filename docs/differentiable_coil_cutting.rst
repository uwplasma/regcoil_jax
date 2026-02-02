Differentiable coil cutting (and why it is hard)
================================================

In the original REGCOIL workflow, **coils are obtained by cutting contours** of the winding-surface
current potential :math:`\Phi(\theta,\zeta)`:

.. math::

   \text{coil} = \{(\theta,\zeta)\;|\;\Phi(\theta,\zeta)=\Phi_0\}.

This contouring step is geometric and contains **discrete choices** (topology changes, branch selection,
case distinctions in marching-squares logic). As a result, it is **not differentiable** in the usual sense.

Nevertheless, a JAX port can still leverage autodiff in two ways:

1. avoid differentiating through coil cutting (recommended for robust workflows), and
2. use *relaxations* that approximate contour extraction with differentiable surrogates (useful for research).

Recommended approach: do not differentiate through cutting
--------------------------------------------------------------------------------

The most robust approach is:

1. solve REGCOIL for :math:`\Phi`,
2. cut coils using a standard contouring algorithm,
3. optimize downstream smooth parameters (e.g. per-coil currents) with autodiff.

This repository includes such workflows in:

- ``examples/2_intermediate/jax_optimize_cut_coil_currents_and_visualize.py``
- ``examples/3_advanced/compare_winding_surface_optimization_cut_coils_currents_poincare.py``

Relaxation demo: soft contour extraction (single-valued θ(ζ))
--------------------------------------------------------------------------------

For pedagogic purposes, we also include a differentiable relaxation that can be used when the contour
is approximately *single-valued* in :math:`\theta(\zeta)`:

.. math::

   \theta(\zeta) \approx
   \operatorname{cmean}_\theta
   \operatorname{softmax}_i\Big(-\beta(\Phi(\zeta,\theta_i)-\Phi_0)^2\Big)

where :math:`\operatorname{cmean}` is a circular mean and :math:`\beta` is a sharpness parameter.

This produces a smooth polyline on the winding surface (one point per :math:`\zeta` sample) which is
then mapped to XYZ via differentiable bilinear interpolation, so you can differentiate through:

- the soft contour extraction,
- the surface interpolation,
- a Biot–Savart evaluation on the resulting polyline.

Example:

.. code-block:: bash

   python examples/3_advanced/differentiable_coil_cutting_softcontour.py --platform cpu

Implementation:

- ``regcoil_jax.diff_coil_cutting`` (soft contour + interpolator + Biot–Savart on a polyline)

Limitations
-----------

This relaxation is not a general replacement for contouring:

- It returns a **single** :math:`\theta` value for each :math:`\zeta`. If the true contour has multiple
  branches for a given :math:`\zeta`, the result becomes a weighted average.
- Choosing :math:`\beta` is problem-dependent. Too small → blurry; too large → unstable gradients.
- It does not resolve topological events (splits/merges) in a physically meaningful way.

Related research directions
---------------------------

If you need differentiable geometry extraction beyond this toy relaxation, a few families of methods are relevant:

- Differentiable isosurface extraction / marching cubes variants (for implicit surfaces).
- Implicit differentiation through a root-finding problem that defines the contour (still requires careful
  branch handling).
- Re-parameterizing coils directly (Fourier curves / splines) and **skipping contouring entirely**.

References
----------

- Landreman, M. (2017). *An improved current potential method for fast computation of stellarator coil shapes* (REGCOIL). Journal of Plasma Physics.
- Hergt et al. (2024). *Global Stellarator Coil Optimization with Quadratic Constraints and Objectives* (“quadcoil”). arXiv:2408.08267.
- Helander et al. (2020). *Stellarator construction with permanent magnets*. Physical Review Letters 125, 135002.
- Remelli et al. (2020). *MeshSDF: Differentiable Iso-Surface Extraction*. NeurIPS 2020.
