Hybrid coil/magnet demos
========================

This page documents “beyond REGCOIL” example workflows enabled by a JAX rewrite:

- differentiable Biot–Savart evaluation for filamentary coils,
- autodiff-based optimization of per-coil currents,
- differentiable point-dipole sources (proxy for small local coils / windowpane coils / permanent magnets),
- Poincaré plots and ParaView outputs for the resulting fields.

The emphasis is pedagogic: each example script is meant to be read and modified.

Point dipole field
------------------

We model a dipole at location :math:`\mathbf{x}_0` with dipole moment :math:`\mathbf{m}` (units A·m\ :sup:`2`).
Let

.. math::

   \mathbf{r} = \mathbf{x} - \mathbf{x}_0, \qquad r = \|\mathbf{r}\|.

The magnetic field is

.. math::

   \mathbf{B}(\mathbf{x})
   = \frac{\mu_0}{4\pi}\left( \frac{3\mathbf{r}(\mathbf{m}\cdot \mathbf{r})}{r^5} - \frac{\mathbf{m}}{r^3} \right).

In code, a small softening :math:`\epsilon` is added as :math:`r^2 \leftarrow r^2 + \epsilon^2` to avoid numerical
blowups at very small distances.

Code mapping
~~~~~~~~~~~~

- Dipole field: :src:`regcoil_jax/dipoles.py` (``dipole_bfield``, ``dipole_bnormal``)

Hybrid normal-field objective
-----------------------------

Given a target surface with unit normal :math:`\hat{\mathbf{n}}(\mathbf{x})` and evaluation points :math:`\mathbf{x}_i`,
we define the normal-field residual

.. math::

   b_i(\theta) = \left(\mathbf{B}_\text{filaments}(\mathbf{x}_i;\theta) + \mathbf{B}_\text{dipoles}(\mathbf{x}_i;\theta)\right)\cdot \hat{\mathbf{n}}(\mathbf{x}_i)
               - b_i^\text{target}.

The simplest objective (used in the demos) is mean-square error with regularization:

.. math::

   \mathcal{L}(\theta)
   = \frac{1}{N}\sum_{i=1}^N b_i(\theta)^2
   + \alpha_I \|\mathbf{I}-\mathbf{I}_0\|_2^2
   + \alpha_m \|\mathbf{m}-\mathbf{m}_0\|_2^2.

Here, :math:`\theta` stacks the per-filament currents :math:`\mathbf{I}` and the dipole moments :math:`\mathbf{m}`.

Because ``regcoil_jax`` uses JAX, gradients are obtained via autodiff:

- :math:`\partial \mathcal{L}/\partial \mathbf{I}` is computed by differentiating through Biot–Savart,
- :math:`\partial \mathcal{L}/\partial \mathbf{m}` is computed by differentiating through the dipole formula.

Code mapping
~~~~~~~~~~~~

- Joint current+dipole optimization: :src:`regcoil_jax/dipole_optimization.py` (``optimize_filaments_and_dipoles_to_match_bnormal``)

Examples
--------

Compare coil cutting with and without winding-surface optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ex:`examples/3_advanced/compare_winding_surface_optimization_cut_coils_currents_poincare.py`

This script shows:

1. baseline REGCOIL solve on a VMEC plasma boundary with a baseline winding surface
2. autodiff winding-surface optimization (a separation field :math:`\mathrm{sep}(\theta,\zeta)`)
3. REGCOIL solve on the optimized winding surface
4. cut coils from the current potential
5. optimize per-coil currents so the *discrete coil set* preserves small :math:`B\cdot n` on the target surface
6. Poincaré plots overlaid with the target surface slice

Few loops + many dipoles (windowpane/magnet proxy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ex:`examples/3_advanced/hybrid_few_loops_many_dipoles_optimize_and_poincare.py`

This script demonstrates a hybrid optimization in which:

- a small number of simple axisymmetric filament loops provide “bulk” field,
- many dipoles correct the remaining normal-field error on the target 3D surface.

Inputs and outputs (for both scripts)
-------------------------------------

Common inputs (CLI args)
~~~~~~~~~~~~~~~~~~~~~~~~

- ``--wout``: VMEC boundary file used as the target surface
- ``--platform``: JAX platform (``cpu`` / ``gpu``)

Common outputs
~~~~~~~~~~~~~~

Each script writes to a timestamped ``outputs_*`` folder containing:

- ``figures/*.png``: publication-style plots (loss histories, Poincaré overlays, etc.)
- ``vtk/*.vts``: target surface as a structured grid (ParaView)
- ``vtk/*.vtp``: coils (polylines), field lines (polylines), Poincaré points (point clouds)
- ``vtk/dipoles.vtp`` (hybrid dipole demo): dipole locations with moment vectors as point data (use ParaView Glyphs)
