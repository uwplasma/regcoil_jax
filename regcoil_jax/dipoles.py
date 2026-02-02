from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .constants import mu0, pi


@dataclass(frozen=True)
class DipoleArray:
    """Magnetic dipole array (for hybrid coil/magnet demos).

    A magnetic dipole is the far-field limit of a small current loop. This is a convenient,
    differentiable proxy for "windowpane"/local coils or permanent magnets.

    Attributes:
      positions: (M,3) dipole locations in meters
      moments:   (M,3) dipole moments in A·m^2
    """

    positions: jnp.ndarray
    moments: jnp.ndarray


def dipole_bfield(
    *,
    points: Any,
    positions: Any,
    moments: Any,
    eps: float = 1e-9,
    batch: int = 2048,
) -> jnp.ndarray:
    """Magnetic field from point dipoles.

    Uses the standard dipole formula:

      B(r) = μ0/(4π) * ( 3 r (m·r)/|r|^5 - m/|r|^3 )

    where r = x - x0, m is the dipole moment, and eps is a softening length to
    avoid numerical blowups near dipole locations.

    Args:
      points:    (N,3)
      positions: (M,3)
      moments:   (M,3)
      batch: batch over dipoles to cap memory
    Returns:
      B: (N,3)
    """
    x = jnp.asarray(points, dtype=jnp.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"points must be (3,) or (N,3), got {x.shape}")

    pos = jnp.asarray(positions, dtype=jnp.float64)
    mom = jnp.asarray(moments, dtype=jnp.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must be (M,3)")
    if mom.shape != pos.shape:
        raise ValueError("moments must have the same shape as positions")

    M = int(pos.shape[0])
    if M == 0:
        return jnp.zeros((x.shape[0], 3), dtype=jnp.float64)

    batch = int(batch)
    if batch <= 0:
        raise ValueError("batch must be positive")
    batch = min(batch, M)
    n_batches = (M + batch - 1) // batch

    coeff = mu0 / (4.0 * pi)
    eps2 = float(eps) * float(eps)

    def b_at_point(xi: jnp.ndarray) -> jnp.ndarray:
        acc = jnp.zeros((3,), dtype=jnp.float64)

        def body(i, a):
            lo = i * batch
            hi = jnp.minimum(lo + batch, M)
            p = jax.lax.dynamic_slice(pos, (lo, 0), (batch, 3))
            m = jax.lax.dynamic_slice(mom, (lo, 0), (batch, 3))
            mask = (jnp.arange(batch) + lo) < hi

            r = xi[None, :] - p  # (B,3)
            r2 = jnp.sum(r * r, axis=-1) + eps2
            inv_r = 1.0 / jnp.sqrt(r2)
            inv_r3 = inv_r / r2
            inv_r5 = inv_r3 / r2
            mdotr = jnp.sum(m * r, axis=-1)
            term = (3.0 * r * (mdotr * inv_r5)[:, None]) - (m * inv_r3[:, None])
            contrib = jnp.sum(term * mask[:, None], axis=0)
            return a + (coeff * contrib)

        return jax.lax.fori_loop(0, n_batches, body, acc)

    return jax.vmap(b_at_point)(x)


def dipole_bnormal(
    *,
    points: Any,
    normals_unit: Any,
    positions: Any,
    moments: Any,
    eps: float = 1e-9,
    batch: int = 2048,
) -> jnp.ndarray:
    """Compute B·n_hat from point dipoles."""
    pts = jnp.asarray(points, dtype=jnp.float64)
    nhat = jnp.asarray(normals_unit, dtype=jnp.float64)
    if pts.shape != nhat.shape:
        raise ValueError("normals_unit must have the same shape as points")
    B = dipole_bfield(points=pts, positions=positions, moments=moments, eps=eps, batch=batch)
    return jnp.sum(B * nhat, axis=-1)


def dipole_array_from_surface_offset(
    *,
    surface_points: np.ndarray,
    surface_normals_unit: np.ndarray,
    offset: float,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Create dipole positions on an offset surface by shifting along normals.

    This helper is intentionally numpy-based: it is used for example setup.
    """
    pts = np.asarray(surface_points, dtype=float)
    nhat = np.asarray(surface_normals_unit, dtype=float)
    if pts.shape != nhat.shape:
        raise ValueError("surface_points and surface_normals_unit must have the same shape")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("surface_points must be (N,3)")
    stride = int(stride)
    if stride <= 0:
        raise ValueError("stride must be positive")
    sel = np.arange(0, pts.shape[0], stride, dtype=int)
    pos = pts[sel] + float(offset) * nhat[sel]
    # Start with zero moments; optimization scripts fill these.
    mom = np.zeros_like(pos)
    return pos, mom

