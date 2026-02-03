from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .biot_savart_jax import FilamentSegments, bfield_from_segments


@dataclass(frozen=True)
class FieldlineTraceResult:
    """Result of JAX field line tracing.

    Attributes:
      points: (nlines, nsteps+1, 3) traced XYZ points.
      active: (nlines, nsteps+1) boolean mask indicating whether the line was still active.
    """

    points: jnp.ndarray
    active: jnp.ndarray


def _unit(v: jnp.ndarray, eps: float) -> jnp.ndarray:
    n = jnp.linalg.norm(v, axis=-1, keepdims=True)
    n = jnp.where(n == 0.0, 1.0, n)
    return v / (n + jnp.asarray(eps, dtype=v.dtype))


def trace_fieldlines_rk4(
    segs: FilamentSegments,
    *,
    starts: Any,
    filament_currents: Any,
    ds: float,
    n_steps: int,
    direction: float = 1.0,
    normalize: bool = True,
    stop_radius: float | None = None,
    eps: float = 1e-12,
    bs_eps: float = 1e-9,
    seg_batch: int = 2048,
) -> FieldlineTraceResult:
    """Trace coil-only field lines using fixed-step RK4 in JAX.

    The ODE is:

      d x / d s = ± B(x) / ||B(x)||     (default)

    where the normalization makes the parameter `s` approximately arclength-like.

    This function is designed to be compatible with autodiff:
    - fixed loop length using `jax.lax.scan`
    - optional "stop" implemented via a boolean mask (shape stays static)
    """
    x0 = jnp.asarray(starts, dtype=jnp.float64)
    if x0.ndim == 1:
        x0 = x0[None, :]
    if x0.ndim != 2 or int(x0.shape[1]) != 3:
        raise ValueError("starts must be (3,) or (nlines,3)")

    I = jnp.asarray(filament_currents, dtype=jnp.float64).reshape((-1,))
    if int(I.shape[0]) != int(segs.n_filaments):
        raise ValueError("filament_currents must have length equal to segs.n_filaments")

    ds = jnp.asarray(ds, dtype=jnp.float64)
    direction = jnp.asarray(direction, dtype=jnp.float64)
    eps = float(eps)
    bs_eps = float(bs_eps)
    seg_batch = int(seg_batch)
    stop_radius_val = None if stop_radius is None else float(stop_radius)

    def f(x: jnp.ndarray) -> jnp.ndarray:
        B = bfield_from_segments(segs, points=x, filament_currents=I, eps=bs_eps, seg_batch=seg_batch)
        if normalize:
            B = _unit(B, eps=eps)
        return direction * B

    def step(carry, _):
        x, active = carry  # x: (nlines,3), active: (nlines,)
        k1 = f(x)
        k2 = f(x + 0.5 * ds * k1)
        k3 = f(x + 0.5 * ds * k2)
        k4 = f(x + ds * k3)
        x_new = x + (ds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if stop_radius_val is not None:
            R = jnp.linalg.norm(x_new[:, :2], axis=1)
            active_new = active & (R <= jnp.asarray(stop_radius_val, dtype=jnp.float64))
        else:
            active_new = active

        # Freeze once inactive to keep the trajectory well-defined.
        x_out = jnp.where(active_new[:, None], x_new, x)
        return (x_out, active_new), (x_out, active_new)

    active0 = jnp.ones((int(x0.shape[0]),), dtype=bool)
    (xT, aT), (xs, actives) = jax.lax.scan(step, (x0, active0), xs=None, length=int(n_steps))
    # scan returns (n_steps, nlines, ...); transpose to (nlines, n_steps, ...)
    xs = jnp.swapaxes(xs, 0, 1)
    actives = jnp.swapaxes(actives, 0, 1)
    points = jnp.concatenate([x0[:, None, :], xs], axis=1)
    active_hist = jnp.concatenate([active0[:, None], actives], axis=1)
    return FieldlineTraceResult(points=points, active=active_hist)


def poincare_section_weights(
    points: Any,
    *,
    nfp: int,
    phi0: float = 0.0,
    sigma: float = 0.05,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Return smooth weights for points near a Poincaré section.

    A classic Poincaré section takes the *crossings* of a field line with the plane
    φ = φ0 (mod 2π/nfp). Exact crossing detection is discrete.

    For differentiable optimization, we instead assign each sample point a smooth
    weight based on how close it is to the section:

      w = exp( - (sin(nfp*(φ-φ0))/sigma)^2 )

    Returns:
      weights: (nlines, npts) in [0,1]
    """
    x = jnp.asarray(points, dtype=jnp.float64)
    if x.ndim != 3 or int(x.shape[2]) != 3:
        raise ValueError("points must be (nlines, npts, 3)")
    nfp = int(nfp)
    if nfp <= 0:
        raise ValueError("nfp must be positive")

    phi = jnp.arctan2(x[:, :, 1], x[:, :, 0])
    arg = float(nfp) * (phi - jnp.asarray(phi0, dtype=jnp.float64))
    s = jnp.sin(arg)
    sig = jnp.asarray(sigma, dtype=jnp.float64)
    sig = jnp.where(sig == 0.0, jnp.asarray(1.0, dtype=jnp.float64), sig)
    w = jnp.exp(-((s / sig) ** 2))
    # Small floor to keep objectives well-conditioned if a line never gets close to the plane.
    return jnp.asarray(eps, dtype=jnp.float64) + (1.0 - float(eps)) * w


def poincare_weighted_RZ(
    points: Any,
    weights: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute weighted (R,Z) means per line for a soft Poincaré section."""
    x = jnp.asarray(points, dtype=jnp.float64)
    w = jnp.asarray(weights, dtype=jnp.float64)
    if x.ndim != 3 or int(x.shape[2]) != 3:
        raise ValueError("points must be (nlines, npts, 3)")
    if w.shape != x.shape[:2]:
        raise ValueError("weights must be (nlines, npts)")
    R = jnp.sqrt(x[:, :, 0] * x[:, :, 0] + x[:, :, 1] * x[:, :, 1])
    Z = x[:, :, 2]
    wn = w / jnp.sum(w, axis=1, keepdims=True)
    Rm = jnp.sum(wn * R, axis=1)
    Zm = jnp.sum(wn * Z, axis=1)
    return Rm, Zm


def soft_poincare_candidates(
    points: Any,
    *,
    nfp: int,
    phi0: float = 0.0,
    alpha: float = 50.0,
    beta: float = 50.0,
    gamma: float = 10.0,
    eps: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Differentiable candidate crossing points for a Poincaré section.

    This function produces a **weighted point cloud** of candidate section points.
    It is designed for autodiff-based optimization, where discrete crossing detection
    is problematic.

    Given sampled fieldline points :math:`x_i`, define:

    - toroidal angle :math:`\\phi_i = \\mathrm{atan2}(y_i, x_i)`
    - phase :math:`u_i = n_{fp}(\\phi_i - \\phi_0)`
    - event :math:`r_i = \\sin(u_i)`

    For each segment :math:`(x_0,x_1)`, we form a candidate point by linear interpolation
    at the (unconstrained) root of :math:`r`:

    - :math:`t = r_0 / (r_0 - r_1 + \\varepsilon)`
    - :math:`p = (1-t)\\,x_0 + t\\,x_1`

    We then assign a smooth weight:

    - :math:`w_{cross} = \\sigma(-\\alpha\\,r_0 r_1)` (large when a sign change is likely)
    - :math:`w_{near} = \\exp(-\\beta\\,(r_0^2 + r_1^2))` (large when endpoints are near the plane)
    - :math:`w_{plane} = \\sigma(\\gamma\\,\\cos(u_{mid}))` (selects :math:`\\phi=\\phi_0` vs :math:`\\phi_0+\\pi/n_{fp}`)

    The final weight is :math:`w = w_{cross} w_{near} w_{plane}`.

    The returned candidates are (nlines, nseg, 3) with corresponding weights (nlines, nseg).

    Notes:

    - Candidates are meaningful only when the discretization is fine enough that crossings
      are bracketed by adjacent samples.
    - This is a differentiable surrogate for optimization, not a plotting-quality extractor.
    """
    x = jnp.asarray(points, dtype=jnp.float64)
    if x.ndim != 3 or int(x.shape[2]) != 3:
        raise ValueError("points must be (nlines, npts, 3)")
    if int(x.shape[1]) < 2:
        raise ValueError("points must have at least 2 samples along the line")

    nfp = int(nfp)
    if nfp <= 0:
        raise ValueError("nfp must be positive")

    phi = jnp.arctan2(x[:, :, 1], x[:, :, 0])
    u = float(nfp) * (phi - jnp.asarray(phi0, dtype=jnp.float64))
    r = jnp.sin(u)

    x0 = x[:, :-1, :]
    x1 = x[:, 1:, :]
    r0 = r[:, :-1]
    r1 = r[:, 1:]
    u0 = u[:, :-1]
    u1 = u[:, 1:]
    um = 0.5 * (u0 + u1)

    denom = (r0 - r1)
    t = r0 / (denom + jnp.asarray(eps, dtype=jnp.float64))
    p = x0 + t[:, :, None] * (x1 - x0)

    w_cross = jax.nn.sigmoid((-float(alpha)) * (r0 * r1))
    w_near = jnp.exp((-float(beta)) * (r0 * r0 + r1 * r1))
    w_plane = jax.nn.sigmoid(float(gamma) * jnp.cos(um))
    w = w_cross * w_near * w_plane
    return p, w


def squared_distance_point_to_segment_2d(
    p_xy: jnp.ndarray,
    a_xy: jnp.ndarray,
    b_xy: jnp.ndarray,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Squared distance from 2D points p to segments a->b (broadcastable).

    p_xy: (..., 2)
    a_xy: (M, 2) or broadcastable with p
    b_xy: (M, 2) or broadcastable with p
    """
    p = jnp.asarray(p_xy, dtype=jnp.float64)
    a = jnp.asarray(a_xy, dtype=jnp.float64)
    b = jnp.asarray(b_xy, dtype=jnp.float64)
    ab = b - a
    ap = p[..., None, :] - a[None, :, :]  # (..., M, 2)
    ab2 = jnp.sum(ab * ab, axis=-1) + jnp.asarray(eps, dtype=jnp.float64)  # (M,)
    t = jnp.sum(ap * ab[None, :, :], axis=-1) / ab2[None, :]  # (..., M)
    t = jnp.clip(t, 0.0, 1.0)
    proj = a[None, :, :] + t[..., None] * ab[None, :, :]
    d = (p[..., None, :] - proj)
    return jnp.sum(d * d, axis=-1)  # (..., M)


def softmin_squared_distance_to_polyline_2d(
    points_xy: Any,
    *,
    polyline_xy: Any,
    beta: float = 200.0,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Smooth approximation to min squared distance from points to a 2D polyline.

    Returns:
      d2: (N,) approximate min squared distance
    """
    pts = jnp.asarray(points_xy, dtype=jnp.float64)
    poly = jnp.asarray(polyline_xy, dtype=jnp.float64)
    if pts.ndim != 2 or int(pts.shape[1]) != 2:
        raise ValueError("points_xy must be (N,2)")
    if poly.ndim != 2 or int(poly.shape[1]) != 2 or int(poly.shape[0]) < 2:
        raise ValueError("polyline_xy must be (M,2) with M>=2")

    a = poly[:-1]
    b = poly[1:]
    d2 = squared_distance_point_to_segment_2d(pts, a, b, eps=float(eps))  # (N, M-1)

    bta = jnp.asarray(beta, dtype=jnp.float64)
    bta = jnp.where(bta == 0.0, jnp.asarray(1.0, dtype=jnp.float64), bta)
    # softmin(d2) = -1/beta * logsumexp(-beta*d2)
    z = (-bta) * d2
    zmax = jnp.max(z, axis=1, keepdims=True)
    lse = zmax + jnp.log(jnp.sum(jnp.exp(z - zmax), axis=1, keepdims=True) + jnp.asarray(eps, dtype=jnp.float64))
    return (-1.0 / bta) * lse[:, 0]
