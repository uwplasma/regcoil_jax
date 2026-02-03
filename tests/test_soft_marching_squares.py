from __future__ import annotations

import jax
import jax.numpy as jnp

from regcoil_jax.diff_isosurface import soft_marching_squares_candidates


def test_soft_marching_squares_shapes_and_weights():
    nzeta = 10
    ntheta = 12
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi, nzeta, endpoint=False)

    # A simple parametric surface for xyz interpolation (not physically important here).
    R0 = 5.0
    a = 1.0
    R = R0 + a * jnp.cos(theta)[None, :]
    Z = a * jnp.sin(theta)[None, :]
    phi = zeta[:, None]
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    r_zt3 = jnp.stack([x, y, jnp.broadcast_to(Z, x.shape)], axis=2)

    # Field with both positive and negative values so crossings exist.
    phi_zt = jnp.sin(theta)[None, :] + 0.3 * jnp.cos(zeta)[:, None]

    pts, w = soft_marching_squares_candidates(r_zt3=r_zt3, phi_zt=phi_zt, level=0.0, alpha=150.0)
    assert pts.shape == (2 * nzeta * ntheta, 3)
    assert w.shape == (2 * nzeta * ntheta,)
    assert jnp.all(jnp.isfinite(pts))
    assert jnp.all(jnp.isfinite(w))
    assert float(jnp.min(w)) >= 0.0
    assert float(jnp.max(w)) <= 1.0

    # Gradient should exist w.r.t. the scalar field (relaxation is differentiable).
    def loss(phi_in):
        _, ww = soft_marching_squares_candidates(r_zt3=r_zt3, phi_zt=phi_in, level=0.0, alpha=150.0)
        return jnp.sum(ww)

    g = jax.grad(loss)(phi_zt)
    assert g.shape == phi_zt.shape
    assert jnp.all(jnp.isfinite(g))

