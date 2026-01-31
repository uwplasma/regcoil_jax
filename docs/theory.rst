Theory (implemented subset)
===========================

This page summarizes the main equations used by the current parity-first implementation.
The goal is to match the Fortran REGCOIL implementation and the notation in the REGCOIL paper/manual.

Surface parameterization
------------------------

We represent a toroidal surface using Fourier series in poloidal angle :math:`\theta` and toroidal angle :math:`\zeta`.
Let :math:`m` be the poloidal mode number and :math:`n` the (field-period scaled) toroidal mode number.

For a VMEC surface (or an offset surface represented in the same basis), the cylindrical major radius :math:`R` and vertical coordinate :math:`Z`
are expanded as

.. math::

   R(\theta,\zeta) &= \sum_{m,n} \Big( R_{m,n}^{c}\cos(m\theta-n\zeta) + R_{m,n}^{s}\sin(m\theta-n\zeta) \Big) \\
   Z(\theta,\zeta) &= \sum_{m,n} \Big( Z_{m,n}^{s}\sin(m\theta-n\zeta) + Z_{m,n}^{c}\cos(m\theta-n\zeta) \Big)

The Cartesian position is

.. math::

   \phi &= \zeta \\
   \mathbf{r}(\theta,\zeta) &= (R\cos\phi,\; R\sin\phi,\; Z)

Surface tangents and normals
----------------------------

We compute tangents

.. math::

   \mathbf{r}_\theta = \frac{\partial \mathbf{r}}{\partial \theta},\qquad
   \mathbf{r}_\zeta = \frac{\partial \mathbf{r}}{\partial \zeta}

and the (non-unit) normal vector using the REGCOIL convention

.. math::

   \mathbf{N} = \mathbf{r}_\zeta \times \mathbf{r}_\theta,\qquad
   \|\mathbf{N}\| = \sqrt{\mathbf{N}\cdot\mathbf{N}},\qquad
   \hat{\mathbf{n}} = \frac{\mathbf{N}}{\|\mathbf{N}\|}

Current potential basis
-----------------------

The single-valued part of the current potential is expanded in trigonometric basis functions

.. math::

   \Phi(\theta,\zeta) = \sum_{j} \Phi_j \, \varphi_j(\theta,\zeta)

with

.. math::

   \varphi_j(\theta,\zeta) \in \{ \sin(m\theta-n\zeta),\; \cos(m\theta-n\zeta)\}

depending on the symmetry option.

The derivatives used throughout are

.. math::

   \frac{\partial}{\partial\theta}\sin(m\theta-n\zeta) &= m\cos(m\theta-n\zeta) \\
   \frac{\partial}{\partial\zeta}\sin(m\theta-n\zeta) &= -n\cos(m\theta-n\zeta) \\
   \frac{\partial}{\partial\theta}\cos(m\theta-n\zeta) &= -m\sin(m\theta-n\zeta) \\
   \frac{\partial}{\partial\zeta}\cos(m\theta-n\zeta) &= n\sin(m\theta-n\zeta)

REGCOIL matrices (chi2_K regularization)
----------------------------------------

REGCOIL constructs matrices so that the least-squares problem can be written (schematically) as

.. math::

   \min_{\Phi} \;\; \chi_B^2(\Phi) + \lambda \, \chi_K^2(\Phi)

For the implemented subset (regularization term ``chi2_K``), the diagnostics are

.. math::

   \chi_B^2 &= n_\text{fp}\,\Delta\theta_p\,\Delta\zeta_p \sum_{(\theta,\zeta)\in S_p}
      B_n(\theta,\zeta)^2 \, \|\mathbf{N}_p(\theta,\zeta)\| \\
   \chi_K^2 &= n_\text{fp}\,\Delta\theta_c\,\Delta\zeta_c \sum_{(\theta,\zeta)\in S_c}
      \frac{\|\Delta\mathbf{K}(\theta,\zeta)\|^2}{\|\mathbf{N}_c(\theta,\zeta)\|}

where :math:`S_p` is the plasma surface grid and :math:`S_c` is the coil (winding) surface grid.

For any quantity :math:`Q(\theta,\zeta)`, the max-norm diagnostics are

.. math::

   \max|B_n| = \max_{S_p} |B_n(\theta,\zeta)|,\qquad
   \max K = \max_{S_c} \|\Delta\mathbf{K}(\theta,\zeta)\|

Lambda scaling in the linear solve
----------------------------------

REGCOIL solves a symmetric linear system for each :math:`\lambda`. For numerical stability, the Fortran code scales the combination as

.. math::

   \mathbf{A}(\lambda) &= \frac{1}{1+\lambda}\mathbf{A}_B + \frac{\lambda}{1+\lambda}\mathbf{A}_K \\
   \mathbf{b}(\lambda) &= \frac{1}{1+\lambda}\mathbf{b}_B + \frac{\lambda}{1+\lambda}\mathbf{b}_K

This scaling leaves the solution unchanged but keeps :math:`\mathbf{A}` and :math:`\mathbf{b}` :math:`\mathcal{O}(1)` for very large :math:`\lambda`.

