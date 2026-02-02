from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .constants import mu0, pi


@dataclass(frozen=True)
class FilamentSegments:
    """Fixed coil geometry for Biot–Savart evaluation (JAX-friendly).

    Attributes:
      seg_midpoints: (M,3)
      seg_dls:       (M,3)
      seg_filament:  (M,) int32, mapping each segment to a filament index in [0, n_filaments)
      n_filaments:   number of filaments (coils) in the set
    """

    seg_midpoints: jnp.ndarray
    seg_dls: jnp.ndarray
    seg_filament: jnp.ndarray
    n_filaments: int


def segments_from_filaments(*, filaments_xyz: list[Any]) -> FilamentSegments:
    """Convert a list of closed polylines into segment arrays.

    Filaments are represented as point loops. Each consecutive pair defines a segment;
    the last point connects back to the first.
    """
    mids: list[np.ndarray] = []
    dls: list[np.ndarray] = []
    fids: list[np.ndarray] = []

    for j, pts_any in enumerate(filaments_xyz):
        pts = np.asarray(pts_any, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"filament points must be (N,3), got {pts.shape}")
        r0 = pts
        r1 = np.roll(pts, shift=-1, axis=0)
        dl = r1 - r0
        mid = 0.5 * (r0 + r1)
        mids.append(mid)
        dls.append(dl)
        fids.append(np.full((mid.shape[0],), j, dtype=np.int32))

    seg_mid = np.concatenate(mids, axis=0) if mids else np.zeros((0, 3), dtype=float)
    seg_dl = np.concatenate(dls, axis=0) if dls else np.zeros((0, 3), dtype=float)
    seg_fid = np.concatenate(fids, axis=0) if fids else np.zeros((0,), dtype=np.int32)

    return FilamentSegments(
        seg_midpoints=jnp.asarray(seg_mid, dtype=jnp.float64),
        seg_dls=jnp.asarray(seg_dl, dtype=jnp.float64),
        seg_filament=jnp.asarray(seg_fid, dtype=jnp.int32),
        n_filaments=int(len(filaments_xyz)),
    )


def bfield_from_segments(
    segs: FilamentSegments,
    *,
    points: jnp.ndarray,
    filament_currents: jnp.ndarray,
    eps: float = 1e-9,
    seg_batch: int = 2048,
) -> jnp.ndarray:
    """Magnetic field B(points) from piecewise-linear filaments via midpoint Biot–Savart.

    Computes:

    .. math::

       d\\mathbf{B} = \\frac{\\mu_0 I}{4\\pi}\\,\\frac{d\\mathbf{l}\\times\\mathbf{R}}{\\lVert\\mathbf{R}\\rVert^3}

    Notes on performance:
      - This implementation batches over segments to cap memory.
      - It is JIT-friendly when `seg_batch` is static.
    """
    points = jnp.asarray(points, dtype=jnp.float64)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (3,) or (N,3), got {points.shape}")

    filament_currents = jnp.asarray(filament_currents, dtype=jnp.float64)
    if filament_currents.ndim != 1 or int(filament_currents.shape[0]) != int(segs.n_filaments):
        raise ValueError("filament_currents must be (n_filaments,)")

    M = int(segs.seg_midpoints.shape[0])
    if M == 0:
        return jnp.zeros((points.shape[0], 3), dtype=jnp.float64)

    seg_batch = int(seg_batch)
    if seg_batch <= 0:
        raise ValueError("seg_batch must be positive")
    seg_batch = min(seg_batch, M)
    n_batches = (M + seg_batch - 1) // seg_batch

    def b_at_point(x: jnp.ndarray) -> jnp.ndarray:
        acc = jnp.zeros((3,), dtype=jnp.float64)

        def body(i, a):
            lo = i * seg_batch
            hi = jnp.minimum(lo + seg_batch, M)
            mid = jax.lax.dynamic_slice(segs.seg_midpoints, (lo, 0), (seg_batch, 3))
            dl = jax.lax.dynamic_slice(segs.seg_dls, (lo, 0), (seg_batch, 3))
            fid = jax.lax.dynamic_slice(segs.seg_filament, (lo,), (seg_batch,))

            # Mask out the padded tail for the last batch.
            mask = (jnp.arange(seg_batch) + lo) < hi
            I = filament_currents[jnp.clip(fid, 0, segs.n_filaments - 1)]

            R = x[None, :] - mid  # (B,3)
            r2 = jnp.sum(R * R, axis=-1) + (eps * eps)
            inv_r3 = 1.0 / (r2 * jnp.sqrt(r2))
            cr = jnp.cross(dl, R)  # (B,3)

            coeff = (mu0 / (4.0 * pi)) * I * inv_r3  # (B,)
            contrib = jnp.sum(cr * coeff[:, None] * mask[:, None], axis=0)  # (3,)
            return a + contrib

        acc = jax.lax.fori_loop(0, n_batches, body, acc)
        return acc

    return jax.vmap(b_at_point)(points)


def bnormal_from_segments(
    segs: FilamentSegments,
    *,
    points: jnp.ndarray,
    normals_unit: jnp.ndarray,
    filament_currents: jnp.ndarray,
    eps: float = 1e-9,
    seg_batch: int = 2048,
) -> jnp.ndarray:
    """Compute B·n_hat at points for a filament set."""
    points = jnp.asarray(points, dtype=jnp.float64)
    normals_unit = jnp.asarray(normals_unit, dtype=jnp.float64)
    if normals_unit.shape != points.shape:
        raise ValueError("normals_unit must have the same shape as points")
    B = bfield_from_segments(segs, points=points, filament_currents=filament_currents, eps=eps, seg_batch=seg_batch)
    return jnp.sum(B * normals_unit, axis=-1)
