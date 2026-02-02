from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .constants import mu0, pi


def soft_contour_theta_of_zeta(
    *,
    phi_zt: Any,
    theta: Any,
    level: float,
    beta: float = 2e4,
    eps: float = 1e-30,
) -> jnp.ndarray:
    """Differentiable approximation of a contour θ(ζ) for Φ(ζ,θ)=level.

    This is a *soft* relaxation of contour extraction:

    .. math::

       \\theta(\\zeta) \\approx \\operatorname{circular\\_mean}_{\\theta}\\;\\operatorname{softmax}_i\\left(-\\beta\\,\\left(\\Phi(\\zeta,\\theta_i)-\\mathrm{level}\\right)^2\\right)

    It produces one (single-valued) θ per ζ. This is useful for pedagogic demos and for
    cases where the chosen contour is approximately single-valued θ(ζ).

    Limitations:

    - if the true contour has multiple branches for a given ζ, this returns a weighted average
      and may "jump" between branches depending on β and initialization.
    """
    phi = jnp.asarray(phi_zt, dtype=jnp.float64)
    th = jnp.asarray(theta, dtype=jnp.float64).reshape((-1,))
    if phi.ndim != 2:
        raise ValueError("phi_zt must be (nzeta, ntheta)")
    if th.ndim != 1 or int(th.shape[0]) != int(phi.shape[1]):
        raise ValueError("theta must be (ntheta,) matching phi_zt.shape[1]")

    # weights over theta axis
    d2 = (phi - float(level)) ** 2
    w = jax.nn.softmax((-float(beta)) * d2, axis=1)

    # circular mean: arg(sum w exp(i theta))
    c = jnp.sum(w * jnp.cos(th)[None, :], axis=1)
    s = jnp.sum(w * jnp.sin(th)[None, :], axis=1)
    th_est = jnp.arctan2(s, c + float(eps))
    # map to [0, 2pi)
    th_est = jnp.mod(th_est, 2.0 * jnp.pi)
    return th_est


def bilinear_interp_periodic_zt3(
    *,
    r_zt3: Any,
    theta: Any,
    zeta: Any,
    query_theta: Any,
    query_zeta: Any,
    nfp: int,
) -> jnp.ndarray:
    """Bilinear interpolation on a periodic uniform (zeta,theta) grid for a vector field.

    Args:
      r_zt3: (nzeta, ntheta, 3) for *one field period* (zeta in [0, 2π/nfp))
      theta: (ntheta,) uniform on [0, 2π)
      zeta:  (nzeta,) uniform on [0, 2π/nfp)
      query_*: (N,)
    Returns:
      points: (N,3)
    """
    r = jnp.asarray(r_zt3, dtype=jnp.float64)
    th = jnp.asarray(theta, dtype=jnp.float64).reshape((-1,))
    ze = jnp.asarray(zeta, dtype=jnp.float64).reshape((-1,))
    tq = jnp.asarray(query_theta, dtype=jnp.float64).reshape((-1,))
    zq = jnp.asarray(query_zeta, dtype=jnp.float64).reshape((-1,))

    if r.ndim != 3 or int(r.shape[2]) != 3:
        raise ValueError("r_zt3 must be (nzeta, ntheta, 3)")
    nzeta, ntheta = int(r.shape[0]), int(r.shape[1])
    if int(th.size) != ntheta:
        raise ValueError("theta must match r_zt3 second dimension")
    if int(ze.size) != nzeta:
        raise ValueError("zeta must match r_zt3 first dimension")

    dt = 2.0 * jnp.pi / float(ntheta)
    dz = (2.0 * jnp.pi / float(int(nfp))) / float(nzeta)
    t0 = th[0]
    z0 = ze[0]

    t = jnp.mod(tq - t0, 2.0 * jnp.pi)
    z = jnp.mod(zq - z0, 2.0 * jnp.pi / float(int(nfp)))

    it = jnp.floor(t / dt).astype(jnp.int32) % ntheta
    iz = jnp.floor(z / dz).astype(jnp.int32) % nzeta
    it1 = (it + 1) % ntheta
    iz1 = (iz + 1) % nzeta

    wt = (t / dt) - jnp.floor(t / dt)
    wz = (z / dz) - jnp.floor(z / dz)

    p00 = r[iz, it]
    p01 = r[iz, it1]
    p10 = r[iz1, it]
    p11 = r[iz1, it1]

    out = (
        (1.0 - wt)[:, None] * (1.0 - wz)[:, None] * p00
        + wt[:, None] * (1.0 - wz)[:, None] * p01
        + (1.0 - wt)[:, None] * wz[:, None] * p10
        + wt[:, None] * wz[:, None] * p11
    )
    return out


def soft_coil_polyline_xyz(
    *,
    phi_zt: Any,
    theta: Any,
    zeta: Any,
    r_coil_zt3: Any,
    level: float,
    beta: float = 2e4,
    nfp: int,
) -> jnp.ndarray:
    """Return a differentiable polyline (nzeta,3) approximating a Φ-contour on the winding surface."""
    phi = jnp.asarray(phi_zt, dtype=jnp.float64)
    th_grid = jnp.asarray(theta, dtype=jnp.float64)
    ze_grid = jnp.asarray(zeta, dtype=jnp.float64)
    r = jnp.asarray(r_coil_zt3, dtype=jnp.float64)
    if phi.ndim != 2:
        raise ValueError("phi_zt must be (nzeta, ntheta)")
    if r.shape[0] != phi.shape[0] or r.shape[1] != phi.shape[1]:
        raise ValueError("r_coil_zt3 must have shape (nzeta,ntheta,3) matching phi_zt")
    th_est = soft_contour_theta_of_zeta(phi_zt=phi, theta=th_grid, level=float(level), beta=float(beta))
    # Use the grid zeta positions (one point per zeta index).
    zq = ze_grid
    pts = bilinear_interp_periodic_zt3(r_zt3=r, theta=th_grid, zeta=ze_grid, query_theta=th_est, query_zeta=zq, nfp=int(nfp))
    return pts


def bnormal_from_polyline(
    *,
    coil_points: Any,
    coil_current: float,
    eval_points: Any,
    eval_normals_unit: Any,
    eps: float = 1e-9,
    seg_batch: int = 2048,
) -> jnp.ndarray:
    """B·n_hat from a single closed polyline, differentiable in the point locations."""
    pts = jnp.asarray(coil_points, dtype=jnp.float64)
    x = jnp.asarray(eval_points, dtype=jnp.float64)
    nhat = jnp.asarray(eval_normals_unit, dtype=jnp.float64)
    if pts.ndim != 2 or int(pts.shape[1]) != 3:
        raise ValueError("coil_points must be (N,3)")
    if x.ndim != 2 or int(x.shape[1]) != 3:
        raise ValueError("eval_points must be (P,3)")
    if nhat.shape != x.shape:
        raise ValueError("eval_normals_unit must match eval_points shape")

    r0 = pts
    r1 = jnp.roll(pts, shift=-1, axis=0)
    dl = r1 - r0
    mid = 0.5 * (r0 + r1)

    M = int(mid.shape[0])
    seg_batch = int(seg_batch)
    seg_batch = min(seg_batch, M) if M > 0 else seg_batch
    n_batches = (M + seg_batch - 1) // seg_batch if M > 0 else 0

    # Pad to avoid out-of-bounds slicing.
    Mp = int(n_batches * seg_batch) if M > 0 else 0
    if Mp != M:
        pad = Mp - M
        mid = jnp.pad(mid, ((0, pad), (0, 0)))
        dl = jnp.pad(dl, ((0, pad), (0, 0)))
        M = Mp

    I = jnp.asarray(coil_current, dtype=jnp.float64)
    coeff0 = (mu0 / (4.0 * pi)) * I
    eps2 = float(eps) * float(eps)

    def b_at_point(xi, ni):
        acc = jnp.zeros((3,), dtype=jnp.float64)

        def body(i, a):
            lo = i * seg_batch
            mid_b = jax.lax.dynamic_slice(mid, (lo, 0), (seg_batch, 3))
            dl_b = jax.lax.dynamic_slice(dl, (lo, 0), (seg_batch, 3))
            R = xi[None, :] - mid_b
            r2 = jnp.sum(R * R, axis=-1) + eps2
            inv_r3 = 1.0 / (r2 * jnp.sqrt(r2))
            cr = jnp.cross(dl_b, R)
            contrib = jnp.sum(cr * (coeff0 * inv_r3)[:, None], axis=0)
            return a + contrib

        acc = jax.lax.fori_loop(0, n_batches, body, acc)
        return jnp.dot(acc, ni)

    return jax.vmap(b_at_point)(x, nhat)


def polyline_length(points: Any, eps: float = 1e-30) -> jnp.ndarray:
    pts = jnp.asarray(points, dtype=jnp.float64)
    d = jnp.roll(pts, shift=-1, axis=0) - pts
    return jnp.sum(jnp.sqrt(jnp.sum(d * d, axis=1) + float(eps)))
