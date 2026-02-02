Theory
======

This page documents the core equations used by ``regcoil_jax`` and maps each equation
to the corresponding code symbols.

Notation follows the REGCOIL paper/manual where possible.

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

Code mapping
~~~~~~~~~~~~

- Fourier surfaces and analytic derivatives: ``regcoil_jax/geometry_fourier.py`` (``FourierSurface``, ``eval_surface_xyz_and_derivs2``)
- Metric tensors and normals: ``regcoil_jax/surface_metrics.py`` (``metrics_and_normals``)

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

Code mapping
~~~~~~~~~~~~

- Fourier mode ordering: ``regcoil_jax/modes.py`` (``init_fourier_modes``)
- Basis + derivative operators: ``regcoil_jax/build_basis.py`` (``build_basis_and_f``)

Least-squares formulation
-------------------------

REGCOIL constructs matrices so that the least-squares problem can be written (schematically) as

.. math::

   \min_{\Phi} \;\; \chi_B^2(\Phi) + \lambda \, \chi_K^2(\Phi)

For each :math:`\lambda`, REGCOIL solves for the single-valued Fourier coefficients :math:`\Phi_j`.
The multi-valued (secular) part that enforces net currents is represented as a known vector field
:math:`\mathbf{d}(\theta,\zeta)` on the winding surface, and the single-valued part contributes through
basis vector fields :math:`\mathbf{f}_j(\theta,\zeta)`.

Define

.. math::

   \Delta \mathbf{K}(\theta,\zeta) = \mathbf{d}(\theta,\zeta) - \sum_j \Phi_j \, \mathbf{f}_j(\theta,\zeta),
   \qquad
   \mathbf{K}(\theta,\zeta) = \frac{\Delta \mathbf{K}(\theta,\zeta)}{\|\mathbf{N}_c(\theta,\zeta)\|}.

Code mapping (core arrays)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symbol
     - Code / variable name
   * - :math:`\mathbf{f}_j`
     - ``mats["fx"]``, ``mats["fy"]``, ``mats["fz"]`` (from ``regcoil_jax/build_basis.py``)
   * - :math:`\mathbf{d}`
     - ``mats["dx"]``, ``mats["dy"]``, ``mats["dz"]`` (from ``regcoil_jax/build_matrices_jax.py``)
   * - :math:`\|\mathbf{N}_c\|`
     - ``mats["normNc"]`` (flattened coil ``normN``)
   * - :math:`\|\mathbf{N}_p\|`
     - ``mats["normNp"]`` (flattened plasma ``normN``)

Diagnostics (field and current)
-------------------------------

The field error is

.. math::

   \chi_B^2 &= n_\text{fp}\,\Delta\theta_p\,\Delta\zeta_p \sum_{(\theta,\zeta)\in S_p}
      B_n(\theta,\zeta)^2 \, \|\mathbf{N}_p(\theta,\zeta)\|

The current density diagnostic reported as ``chi2_K`` is

.. math::

   \chi_K^2 = n_\text{fp}\,\Delta\theta_c\,\Delta\zeta_c \sum_{(\theta,\zeta)\in S_c}
      \frac{\|\Delta\mathbf{K}(\theta,\zeta)\|^2}{\|\mathbf{N}_c(\theta,\zeta)\|}.

For any quantity :math:`Q(\theta,\zeta)`, the max-norm diagnostics are

.. math::

   \max|B_n| = \max_{S_p} |B_n(\theta,\zeta)|,\qquad
   \max K = \max_{S_c} \|\Delta\mathbf{K}(\theta,\zeta)\|

Code mapping
~~~~~~~~~~~~

- Diagnostics: ``regcoil_jax/solve_jax.py`` (``diagnostics``)
- NetCDF outputs: ``regcoil_jax/io_output.py`` (``chi2_B``, ``chi2_K``, ``max_Bnormal``, ``max_K``)

Lambda scaling in the linear solve
----------------------------------

REGCOIL solves a symmetric linear system for each :math:`\lambda`. For numerical stability, the Fortran code scales the combination as

.. math::

   \mathbf{A}(\lambda) &= \frac{1}{1+\lambda}\mathbf{A}_B + \frac{\lambda}{1+\lambda}\mathbf{A}_K \\
   \mathbf{b}(\lambda) &= \frac{1}{1+\lambda}\mathbf{b}_B + \frac{\lambda}{1+\lambda}\mathbf{b}_K

This scaling leaves the solution unchanged but keeps :math:`\mathbf{A}` and :math:`\mathbf{b}` :math:`\mathcal{O}(1)` for very large :math:`\lambda`.

Code mapping
~~~~~~~~~~~~

- ``regcoil_jax/solve_jax.py`` (``solve_for_lambdas`` / ``solve_one_lambda``)

Regularization options
----------------------

``regularization_term_option`` selects the regularization term:

- ``"chi2_K"``: regularize all Cartesian components of :math:`\Delta\mathbf{K}`.
- ``"K_xy"``: regularize only :math:`(\Delta K_x,\Delta K_y)`.
- ``"K_zeta"``: regularize only the toroidal (zeta) tangent component of :math:`\Delta\mathbf{K}`.
- ``"Laplace-Beltrami"``: regularize the Laplace–Beltrami operator applied to :math:`\Phi` on the winding surface.

K-regularization family
~~~~~~~~~~~~~~~~~~~~~~~

All K-style regularizations use the same :math:`1/\|\mathbf{N}_c\|` weighting:

.. math::

   \langle a, b \rangle = \Delta\theta_c\,\Delta\zeta_c \sum_{S_c} a(\theta,\zeta)\,b(\theta,\zeta)\,\frac{1}{\|\mathbf{N}_c(\theta,\zeta)\|}.

Then:

.. math::

   \chi_{K}^2(\Phi) &= \langle \Delta K_x, \Delta K_x\rangle + \langle \Delta K_y, \Delta K_y\rangle + \langle \Delta K_z, \Delta K_z\rangle, \\
   \chi_{K_{xy}}^2(\Phi) &= \langle \Delta K_x, \Delta K_x\rangle + \langle \Delta K_y, \Delta K_y\rangle.

For ``K_zeta``, define the unit tangent along the winding-surface coordinate :math:`\zeta`:

.. math::

   \hat{\mathbf{t}}_\zeta(\theta,\zeta) = \frac{\mathbf{r}_{c,\zeta}(\theta,\zeta)}{\|\mathbf{r}_{c,\zeta}(\theta,\zeta)\|},
   \qquad
   \Delta K_\zeta(\theta,\zeta) = \Delta\mathbf{K}(\theta,\zeta)\cdot \hat{\mathbf{t}}_\zeta(\theta,\zeta).

Then:

.. math::

   \chi_{K_\zeta}^2(\Phi) = \langle \Delta K_\zeta, \Delta K_\zeta\rangle.

Code mapping:

- ``regcoil_jax/build_matrices_jax.py`` computes ``matrix_reg`` and ``RHS_reg`` from ``fx/fy/fz`` and ``dx/dy/dz``.
- ``K_zeta`` additionally projects onto ``rze`` (``dr/dzeta``) to form :math:`\hat{\mathbf{t}}_\zeta`.

Laplace–Beltrami regularization
-------------------------------

REGCOIL’s Laplace–Beltrami option regularizes a surface-Laplacian-like operator applied to the current potential
:math:`\Phi` on the winding surface.

For a parametric surface with coordinates :math:`(\theta,\zeta)`, define the 2D metric tensor

.. math::

   g_{ij} =
   \begin{pmatrix}
     g_{\theta\theta} & g_{\theta\zeta} \\
     g_{\theta\zeta} & g_{\zeta\zeta}
   \end{pmatrix},
   \qquad
   \sqrt{g} = \sqrt{\det(g)}.

The Laplace–Beltrami operator on a scalar field :math:`\Phi(\theta,\zeta)` is

.. math::

   \Delta_s \Phi
   =
   \frac{1}{\sqrt{g}}
   \partial_i\left(\sqrt{g}\, g^{ij}\, \partial_j \Phi\right),

where :math:`g^{ij}` is the inverse metric.

In this codebase, the operator is implemented in a way that matches ``regcoil_build_matrices.f90`` by assembling
an additional basis matrix ``flb`` representing :math:`\Delta_s \varphi_j` evaluated on the winding-surface grid.
This matrix is used both for:

- the ``regularization_term_option="Laplace-Beltrami"`` solve, and
- diagnostics fields/metrics like ``Laplace_Beltrami2`` and ``chi2_Laplace_Beltrami``.

Code mapping
~~~~~~~~~~~~

- Metric tensor and its derivatives (used to form coefficients): ``regcoil_jax/build_matrices_jax.py``
- Laplace–Beltrami basis construction: ``regcoil_jax/build_basis.py`` (``flb``)
- Output fields: ``regcoil_jax/io_output.py`` (``Laplace_Beltrami2``, ``chi2_Laplace_Beltrami``)

Autodiff postprocessing: coil currents after cutting
----------------------------------------------------

REGCOIL’s output current potential :math:`\Phi(\theta,\zeta)` represents a continuous surface current distribution
on the winding surface. A common postprocessing step is to **cut filamentary coils** by taking contours of
:math:`\Phi(\theta,\zeta)`. This contouring operation is not differentiable, so ``regcoil_jax`` treats it as a
geometric postprocess.

However, *after* coil cutting, many important tuning problems are smooth and can be optimized with autodiff,
notably **per-coil currents**.

Let :math:`\{\mathcal{C}_k\}` be a set of cut filament loops, each with current :math:`I_k`. Define the Biot–Savart field

.. math::

   \mathbf{B}(\mathbf{x}) = \sum_k I_k \, \mathbf{B}_k(\mathbf{x}),

and evaluate the normal component on target surface points :math:`\mathbf{x}_i` with unit normals :math:`\hat{\mathbf{n}}_i`:

.. math::

   b_i(\mathbf{I}) = \mathbf{B}(\mathbf{x}_i)\cdot \hat{\mathbf{n}}_i.

The demo optimizes a mean-square loss

.. math::

   \mathcal{L}(\mathbf{I}) = \frac{1}{N}\sum_i (b_i(\mathbf{I}) - b_i^\text{target})^2 + \alpha \|\mathbf{I}-\mathbf{I}_0\|_2^2.

Code mapping:

- coil cutting: ``regcoil_jax/coil_cutting.py``
- filament Biot–Savart: ``regcoil_jax/biot_savart_jax.py`` (JAX) and ``regcoil_jax/fieldlines.py`` (numpy visualization)
- per-coil current optimization: ``regcoil_jax/coil_current_optimization.py``

Dipole sources (hybrid demos)
-----------------------------

For “beyond REGCOIL” demos, we also include **point dipoles** as a differentiable proxy for small local coils
or permanent magnets. The dipole field formula and hybrid optimization objective are documented in
``docs/hybrid_design.rst`` and implemented in:

- ``regcoil_jax/dipoles.py``
- ``regcoil_jax/dipole_optimization.py``

REGCOIL also supports a Laplace–Beltrami regularization of the current potential on the winding surface.
In the Fortran code this is enabled by setting ``regularization_term_option = "Laplace-Beltrami"``.

The implementation constructs a basis matrix :math:`f_{\mathrm{LB}}` such that

.. math::

   \Delta_{\mathrm{LB}}\Phi(\theta,\zeta) \approx \mathbf{f}_{\mathrm{LB}}(\theta,\zeta)\cdot \boldsymbol{\Phi},

and defines a corresponding secular (net-current) contribution vector :math:`d_{\mathrm{LB}}(\theta,\zeta)`.
Diagnostics are then computed analogously to :math:`\chi_K^2`:

.. math::

   \chi^2_{\mathrm{LB}} = n_\mathrm{fp}\,\Delta\theta_c\,\Delta\zeta_c
   \sum_{S_c} \frac{\left(d_{\mathrm{LB}} - f_{\mathrm{LB}}\boldsymbol{\Phi}\right)^2}{\|\mathbf{N}_c\|}.

In this port, :math:`f_{\mathrm{LB}}` and :math:`d_{\mathrm{LB}}` follow the same formulas as
``regcoil_build_matrices.f90`` and are included in the output file as ``chi2_Laplace_Beltrami`` and ``Laplace_Beltrami2``.
