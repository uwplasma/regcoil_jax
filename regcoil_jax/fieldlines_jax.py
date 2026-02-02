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
