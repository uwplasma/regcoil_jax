Permanent magnets (dipole lattices)
===================================

This page documents the "REGCOIL-PM-like" capability in ``regcoil_jax``:

*representing permanent magnets (or small local coils) as a lattice of point dipoles and solving
for their moments to cancel the normal field on the plasma surface.*

The key implementation modules are:

* ``regcoil_jax.dipoles``: point-dipole magnetic field and B·n evaluation (batched, JAX).
* ``regcoil_jax.permanent_magnets``: least-squares and fixed-magnitude optimization utilities.

Dipole model
------------

A point dipole at position :math:`\mathbf{x}_0` with moment :math:`\mathbf{m}` produces:

.. math::

  \mathbf{B}(\mathbf{x})
  = \frac{\mu_0}{4\pi}
    \left(
      \frac{3\,\mathbf{r}(\mathbf{m}\cdot\mathbf{r})}{|\mathbf{r}|^5}
      - \frac{\mathbf{m}}{|\mathbf{r}|^3}
    \right),
  \qquad \mathbf{r} = \mathbf{x}-\mathbf{x}_0.

The normal component at a surface point with unit normal :math:`\hat{\mathbf{n}}` is
:math:`B_n = \hat{\mathbf{n}}\cdot\mathbf{B}`.

Implementation: ``regcoil_jax.dipoles.dipole_bfield`` and
``regcoil_jax.dipoles.dipole_bnormal``.

Ridge-regularized least squares (linear solve)
----------------------------------------------

If dipole positions are fixed, :math:`B_n` is a **linear function** of the moments. If we stack all
moments into a vector :math:`m`, we can write:

.. math::

  B_n = A\,m,

for a suitable linear operator :math:`A`.

To cancel a target normal field :math:`b` (e.g. :math:`b = -B_{\mathrm{plasma}}\cdot n`), we solve:

.. math::

  \min_m \|A m - b\|_2^2 + \alpha\|m\|_2^2,

which leads to the normal equations

.. math::

  (A^\top A + \alpha I)\,m = A^\top b.

Implementation: ``regcoil_jax.permanent_magnets.solve_dipole_moments_ridge_cg``.

Notes:

* We use conjugate gradients (CG) on the normal equations.
* We apply :math:`A^\top` using JAX reverse-mode autodiff (VJP), so we do not have to explicitly
  assemble a dense matrix :math:`A` for large problems.

Fixed-magnitude magnets (smooth relaxation)
------------------------------------------

Often magnets are constrained to a fixed magnitude :math:`\|\mathbf{m}_i\| = m_0`.
This constraint is non-linear. We provide a smooth, differentiable relaxation by parameterizing

.. math::

  \mathbf{m}_i = m_0\,\frac{\mathbf{v}_i}{\|\mathbf{v}_i\|}.

and optimizing the unconstrained :math:`\mathbf{v}_i` with gradient descent / Adam.

Implementation: ``regcoil_jax.permanent_magnets.optimize_dipole_orientations_fixed_magnitude``.

Example
-------

Run the pedagogic end-to-end demo:

.. code-block:: bash

  python examples/3_advanced/permanent_magnets_cancel_bplasma.py --platform cpu

This script:

1. runs ``regcoil_jax`` on an input with ``load_bnorm=.true.`` to obtain ``Bnormal_from_plasma_current``,
2. places dipoles on an offset winding surface,
3. solves for dipole moments to cancel :math:`B_{\mathrm{plasma}}\cdot n`,
4. writes figures and ParaView VTK outputs.

