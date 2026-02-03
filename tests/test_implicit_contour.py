from __future__ import annotations

import jax
import jax.numpy as jnp

from regcoil_jax.diff_coil_cutting import (
    _interp_theta_periodic,
    implicit_contour_theta_of_zeta,
    implicit_coil_polyline_xyz,
)


def test_implicit_contour_is_root_and_has_reasonable_gradients():
    nfp = 3
    nzeta = 18
    ntheta = 64
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, nzeta, endpoint=False)

    # A smooth field with a well-defined (but non-unique) contour Φ(ζ,θ)=0 for each ζ.
    # The implicit solver chooses a branch based on a smooth initial guess.
    phi_zt = jnp.sin(theta)[None, :] + 0.25 * jnp.cos(2.0 * zeta)[:, None]
    level = jnp.array(0.0, dtype=jnp.float64)

    theta_z = implicit_contour_theta_of_zeta(phi_zt, theta, level)
    assert theta_z.shape == (nzeta,)

    # Check that Φ(ζ, θ(ζ)) ≈ level.
    phi_at = _interp_theta_periodic(f_zt=phi_zt, theta=theta, theta_query=theta_z)
    resid = phi_at - level
    assert float(jnp.max(jnp.abs(resid))) < 5e-6

    # Gradient sanity: compare a single entry vs finite difference.
    def loss(phi):
        thz = implicit_contour_theta_of_zeta(phi, theta, level)
        return jnp.mean(thz * thz)

    g = jax.grad(loss)(phi_zt)
    assert g.shape == phi_zt.shape

    eps = 1e-6
    idx = (0, 3)
    lp = loss(phi_zt.at[idx].add(eps))
    lm = loss(phi_zt.at[idx].add(-eps))
    fd = (lp - lm) / (2.0 * eps)
    assert jnp.isfinite(g[idx])
    assert jnp.isfinite(fd)

    # The piecewise-linear interpolation introduces small kinks; accept modest tolerance.
    assert float(jnp.abs(g[idx] - fd)) < 5e-3


def test_implicit_coil_polyline_xyz_shapes():
    nfp = 2
    nzeta = 12
    ntheta = 32
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, nzeta, endpoint=False)

    # Dummy surface: a simple torus-like embedding in xyz on the same (zeta,theta) grid.
    R0 = 10.0
    a = 1.0
    R = R0 + a * jnp.cos(theta)[None, :]
    Z = a * jnp.sin(theta)[None, :]
    phi = zeta[:, None]
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    r_zt3 = jnp.stack([x, y, jnp.broadcast_to(Z, x.shape)], axis=2)

    phi_zt = jnp.sin(theta)[None, :] + 0.1 * jnp.cos(zeta)[:, None]
    pts = implicit_coil_polyline_xyz(phi_zt=phi_zt, theta=theta, zeta=zeta, r_coil_zt3=r_zt3, level=0.0, nfp=nfp)
    assert pts.shape == (nzeta, 3)
