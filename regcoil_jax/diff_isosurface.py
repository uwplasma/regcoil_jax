from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def soft_marching_squares_candidates(
    *,
    r_zt3: Any,
    phi_zt: Any,
    level: float,
    alpha: float = 200.0,
    eps: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Differentiable marching-squares-style *candidate* contour points on a (zeta,theta) grid.

    This is a smooth relaxation of classic marching squares:

    - It does **not** return a discrete polyline (topology/branch selection is still discrete).
    - Instead it returns a weighted point cloud of *edge intersection candidates*.

    For each grid edge, we:
      1) compute a linear interpolation location `t` for `phi(level)=0`,
      2) compute the corresponding XYZ point via linear interpolation of `r_zt3`,
      3) assign a smooth "crossing weight" based on whether the edge endpoints straddle the level.

    This is useful for optimization objectives that want a differentiable proxy for a contour,
    similarly in spirit to the repo's differentiable "soft Poincaré" utilities.

    Args:
      r_zt3: (nzeta, ntheta, 3) surface points (one period or full torus; periodicity handled by caller)
      phi_zt: (nzeta, ntheta) scalar field sampled on the same grid
      level: isovalue
      alpha: sharpness for the smooth straddle detector (larger -> closer to hard sign test)
      eps: stabilizer for divisions
    Returns:
      points: (N,3) candidate points, with N = 2 * nzeta * ntheta (theta-edges + zeta-edges)
      weights: (N,) in (0,1), representing likelihood that the edge crosses the contour
    """
    r = jnp.asarray(r_zt3, dtype=jnp.float64)
    phi = jnp.asarray(phi_zt, dtype=jnp.float64)
    if r.ndim != 3 or int(r.shape[2]) != 3:
        raise ValueError("r_zt3 must be (nzeta, ntheta, 3)")
    if phi.ndim != 2:
        raise ValueError("phi_zt must be (nzeta, ntheta)")
    if int(phi.shape[0]) != int(r.shape[0]) or int(phi.shape[1]) != int(r.shape[1]):
        raise ValueError("phi_zt shape must match r_zt3 first two dims")

    nzeta, ntheta = int(phi.shape[0]), int(phi.shape[1])
    lvl = jnp.asarray(level, dtype=jnp.float64)
    alpha = float(alpha)
    eps = float(eps)

    # Theta-directed edges: (iz,it) -> (iz,it1)
    it1 = (jnp.arange(ntheta, dtype=jnp.int32) + 1) % ntheta
    iz = jnp.arange(nzeta, dtype=jnp.int32)[:, None]
    it = jnp.arange(ntheta, dtype=jnp.int32)[None, :]
    it1g = it1[None, :]

    phi0_t = phi[iz, it]
    phi1_t = phi[iz, it1g]
    r0_t = r[iz, it]
    r1_t = r[iz, it1g]

    dphi_t = phi1_t - phi0_t
    t_t = (lvl - phi0_t) / (dphi_t + eps)
    t_tc = jnp.clip(t_t, 0.0, 1.0)
    p_t = (1.0 - t_tc)[..., None] * r0_t + t_tc[..., None] * r1_t

    prod_t = (phi0_t - lvl) * (phi1_t - lvl)
    w_cross_t = jax.nn.sigmoid(-alpha * prod_t)
    w_in_t = jax.nn.sigmoid(alpha * t_t) * jax.nn.sigmoid(alpha * (1.0 - t_t))
    w_t = w_cross_t * w_in_t

    # Zeta-directed edges: (iz,it) -> (iz1,it)
    iz1 = (jnp.arange(nzeta, dtype=jnp.int32) + 1) % nzeta
    iz1g = iz1[:, None]

    phi0_z = phi[iz, it]
    phi1_z = phi[iz1g, it]
    r0_z = r[iz, it]
    r1_z = r[iz1g, it]

    dphi_z = phi1_z - phi0_z
    t_z = (lvl - phi0_z) / (dphi_z + eps)
    t_zc = jnp.clip(t_z, 0.0, 1.0)
    p_z = (1.0 - t_zc)[..., None] * r0_z + t_zc[..., None] * r1_z

    prod_z = (phi0_z - lvl) * (phi1_z - lvl)
    w_cross_z = jax.nn.sigmoid(-alpha * prod_z)
    w_in_z = jax.nn.sigmoid(alpha * t_z) * jax.nn.sigmoid(alpha * (1.0 - t_z))
    w_z = w_cross_z * w_in_z

    pts = jnp.concatenate([p_t.reshape((-1, 3)), p_z.reshape((-1, 3))], axis=0)
    w = jnp.concatenate([w_t.reshape((-1,)), w_z.reshape((-1,))], axis=0)
    return pts, w

