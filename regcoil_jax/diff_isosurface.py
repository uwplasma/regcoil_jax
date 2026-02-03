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


def soft_marching_cubes_candidates(
    *,
    xyz_ijk3: Any,
    phi_ijk: Any,
    level: float,
    alpha: float = 200.0,
    eps: float = 1e-12,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Differentiable marching-cubes-style *candidate* isosurface points on a 3D grid.

    Like :func:`soft_marching_squares_candidates`, this routine is a **smooth relaxation**
    of a discrete extractor:

    - It does **not** return a connected triangle mesh (topology/branch selection is discrete).
    - Instead it returns a weighted point cloud of *edge intersection candidates* on the
      3D grid edges (x-, y-, and z-directed).

    For each grid edge, we:
      1) compute a linear interpolation location `t` for `phi(level)=0`,
      2) compute the corresponding 3D point via linear interpolation of `xyz_ijk3`,
      3) assign a smooth "crossing weight" based on whether the edge endpoints straddle the level.

    Args:
      xyz_ijk3: (nx, ny, nz, 3) grid vertex coordinates.
      phi_ijk:  (nx, ny, nz) scalar field sampled at the same vertices.
      level: isovalue.
      alpha: sharpness for the smooth straddle detector (larger -> closer to hard sign test).
      eps: stabilizer for divisions.
      periodic: periodicity flags for (x, y, z). When True, includes edges between the last and first index.
    Returns:
      points: (N,3) candidate points, where N is the total number of edges considered.
      weights: (N,) in (0,1), indicating likelihood that the edge crosses the isosurface.

    This function is intended for differentiable objectives that need an isosurface proxy,
    not for producing publication-quality meshes.
    """
    xyz = jnp.asarray(xyz_ijk3, dtype=jnp.float64)
    phi = jnp.asarray(phi_ijk, dtype=jnp.float64)
    if xyz.ndim != 4 or int(xyz.shape[3]) != 3:
        raise ValueError("xyz_ijk3 must be (nx, ny, nz, 3)")
    if phi.ndim != 3:
        raise ValueError("phi_ijk must be (nx, ny, nz)")
    if tuple(phi.shape) != tuple(xyz.shape[:3]):
        raise ValueError("phi_ijk shape must match xyz_ijk3 first three dims")

    alpha = float(alpha)
    eps = float(eps)
    lvl = jnp.asarray(level, dtype=jnp.float64)
    per_x, per_y, per_z = (bool(periodic[0]), bool(periodic[1]), bool(periodic[2]))

    def _edge_candidates(x0, x1, f0, f1):
        dphi = f1 - f0
        t = (lvl - f0) / (dphi + eps)
        tc = jnp.clip(t, 0.0, 1.0)
        p = (1.0 - tc)[..., None] * x0 + tc[..., None] * x1
        prod = (f0 - lvl) * (f1 - lvl)
        w_cross = jax.nn.sigmoid(-alpha * prod)
        w_in = jax.nn.sigmoid(alpha * t) * jax.nn.sigmoid(alpha * (1.0 - t))
        return p, (w_cross * w_in)

    # X edges
    if per_x:
        x0 = xyz
        x1 = jnp.roll(xyz, shift=-1, axis=0)
        f0 = phi
        f1 = jnp.roll(phi, shift=-1, axis=0)
    else:
        x0 = xyz[:-1, :, :, :]
        x1 = xyz[1:, :, :, :]
        f0 = phi[:-1, :, :]
        f1 = phi[1:, :, :]
    p_x, w_x = _edge_candidates(x0, x1, f0, f1)

    # Y edges
    if per_y:
        y0 = xyz
        y1 = jnp.roll(xyz, shift=-1, axis=1)
        g0 = phi
        g1 = jnp.roll(phi, shift=-1, axis=1)
    else:
        y0 = xyz[:, :-1, :, :]
        y1 = xyz[:, 1:, :, :]
        g0 = phi[:, :-1, :]
        g1 = phi[:, 1:, :]
    p_y, w_y = _edge_candidates(y0, y1, g0, g1)

    # Z edges
    if per_z:
        z0 = xyz
        z1 = jnp.roll(xyz, shift=-1, axis=2)
        h0 = phi
        h1 = jnp.roll(phi, shift=-1, axis=2)
    else:
        z0 = xyz[:, :, :-1, :]
        z1 = xyz[:, :, 1:, :]
        h0 = phi[:, :, :-1]
        h1 = phi[:, :, 1:]
    p_z, w_z = _edge_candidates(z0, z1, h0, h1)

    pts = jnp.concatenate([p_x.reshape((-1, 3)), p_y.reshape((-1, 3)), p_z.reshape((-1, 3))], axis=0)
    w = jnp.concatenate([w_x.reshape((-1,)), w_y.reshape((-1,)), w_z.reshape((-1,))], axis=0)
    return pts, w
