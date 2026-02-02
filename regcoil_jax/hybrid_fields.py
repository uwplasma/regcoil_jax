from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from .biot_savart_jax import FilamentSegments, bfield_from_segments, bnormal_from_segments
from .dipoles import dipole_bfield, dipole_bnormal
from .fieldlines import FilamentField, bfield_from_filaments


def bfield_from_segments_and_dipoles(
    segs: FilamentSegments,
    *,
    points: Any,
    filament_currents: Any,
    dipole_positions: Any,
    dipole_moments: Any,
    seg_batch: int = 2048,
    dipole_batch: int = 2048,
) -> jnp.ndarray:
    """B(points) from Biot–Savart filaments + point dipoles (JAX)."""
    Bc = bfield_from_segments(
        segs,
        points=jnp.asarray(points, dtype=jnp.float64),
        filament_currents=jnp.asarray(filament_currents, dtype=jnp.float64),
        seg_batch=int(seg_batch),
    )
    Bd = dipole_bfield(
        points=points,
        positions=dipole_positions,
        moments=dipole_moments,
        batch=int(dipole_batch),
    )
    return Bc + Bd


def bnormal_from_segments_and_dipoles(
    segs: FilamentSegments,
    *,
    points: Any,
    normals_unit: Any,
    filament_currents: Any,
    dipole_positions: Any,
    dipole_moments: Any,
    seg_batch: int = 2048,
    dipole_batch: int = 2048,
) -> jnp.ndarray:
    """B·n_hat from Biot–Savart filaments + point dipoles (JAX)."""
    bn_c = bnormal_from_segments(
        segs,
        points=points,
        normals_unit=normals_unit,
        filament_currents=filament_currents,
        seg_batch=int(seg_batch),
    )
    bn_d = dipole_bnormal(
        points=points,
        normals_unit=normals_unit,
        positions=dipole_positions,
        moments=dipole_moments,
        batch=int(dipole_batch),
    )
    return bn_c + bn_d


@dataclass(frozen=True)
class HybridFieldNumpy:
    """Numpy-side field model for visualization (filaments + dipoles)."""

    filaments: FilamentField
    dipole_positions: np.ndarray  # (M,3)
    dipole_moments: np.ndarray  # (M,3)


def bfield_from_hybrid_numpy(
    field: HybridFieldNumpy,
    x: np.ndarray,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """B(x) from filaments + dipoles (numpy; used for field line tracing)."""
    x = np.asarray(x, dtype=float)
    Bc = bfield_from_filaments(field.filaments, x, eps=eps)

    pos = np.asarray(field.dipole_positions, dtype=float)
    mom = np.asarray(field.dipole_moments, dtype=float)
    if pos.size == 0:
        return Bc

    single = x.ndim == 1
    if single:
        x2 = x[None, :]
    else:
        x2 = x

    r = x2[:, None, :] - pos[None, :, :]
    r2 = np.sum(r * r, axis=-1) + eps * eps
    inv_r = 1.0 / np.sqrt(r2)
    inv_r3 = inv_r / r2
    inv_r5 = inv_r3 / r2
    mdotr = np.sum(mom[None, :, :] * r, axis=-1)
    term = 3.0 * r * (mdotr * inv_r5)[:, :, None] - mom[None, :, :] * inv_r3[:, :, None]
    Bd = (1.0e-7) * np.sum(term, axis=1)  # mu0/(4pi) = 1e-7
    B = Bd + (Bc[None, :] if single else Bc)
    return B[0] if single else B

