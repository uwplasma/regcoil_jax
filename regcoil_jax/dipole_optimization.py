from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .biot_savart_jax import FilamentSegments
from .hybrid_fields import bnormal_from_segments_and_dipoles
from .optimize import minimize_adam, OptimizeResult


@dataclass(frozen=True)
class HybridCurrentDipoleOptResult:
    filament_currents: np.ndarray  # (n_filaments,)
    dipole_moments: np.ndarray  # (M,3)
    loss_history: np.ndarray  # (steps,)


def optimize_filaments_and_dipoles_to_match_bnormal(
    segs: FilamentSegments,
    *,
    dipole_positions: Any,
    points: Any,
    normals_unit: Any,
    target_bnormal: Any,
    filament_currents0: Any,
    dipole_moments0: Any,
    steps: int = 400,
    lr: float = 2e-2,
    l2_current: float = 1e-6,
    l2_moment: float = 1e-8,
    seg_batch: int = 2048,
    dipole_batch: int = 2048,
) -> HybridCurrentDipoleOptResult:
    """Jointly optimize filament currents and dipole moments to match target B·n.

    This is a pedagogic building block for hybrid designs:
      - a small set of "large" coils (filaments)
      - many local coillets / magnets (dipoles)

    Args:
      dipole_positions: (M,3) fixed locations
      dipole_moments0:  (M,3) initial moments
    """
    pts = jnp.asarray(points, dtype=jnp.float64)
    nhat = jnp.asarray(normals_unit, dtype=jnp.float64)
    targ = jnp.asarray(target_bnormal, dtype=jnp.float64).reshape(-1)
    if pts.shape[0] != targ.shape[0]:
        raise ValueError("target_bnormal must have length equal to number of points")

    dip_pos = jnp.asarray(dipole_positions, dtype=jnp.float64)
    m0 = jnp.asarray(dipole_moments0, dtype=jnp.float64)
    if dip_pos.ndim != 2 or dip_pos.shape[1] != 3:
        raise ValueError("dipole_positions must be (M,3)")
    if m0.shape != dip_pos.shape:
        raise ValueError("dipole_moments0 must have shape (M,3)")

    I0 = jnp.asarray(filament_currents0, dtype=jnp.float64).reshape(-1)
    if int(I0.shape[0]) != int(segs.n_filaments):
        raise ValueError("filament_currents0 must have length equal to segs.n_filaments")

    x0 = jnp.concatenate([I0, m0.reshape(-1)], axis=0)

    nI = int(I0.shape[0])
    nM = int(m0.shape[0])

    def unpack(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        I = x[:nI]
        m = x[nI:].reshape((nM, 3))
        return I, m

    def loss_fn(x: jnp.ndarray) -> jnp.ndarray:
        I, m = unpack(x)
        bn = bnormal_from_segments_and_dipoles(
            segs,
            points=pts,
            normals_unit=nhat,
            filament_currents=I,
            dipole_positions=dip_pos,
            dipole_moments=m,
            seg_batch=int(seg_batch),
            dipole_batch=int(dipole_batch),
        )
        err = bn - targ
        loss = jnp.mean(err * err)
        loss = loss + float(l2_current) * jnp.mean((I - I0) * (I - I0))
        loss = loss + float(l2_moment) * jnp.mean((m - m0) * (m - m0))
        return loss

    res: OptimizeResult = minimize_adam(loss_fn, x0, steps=int(steps), lr=float(lr), jit=True)
    I_opt, m_opt = unpack(res.x)
    return HybridCurrentDipoleOptResult(
        filament_currents=np.asarray(I_opt, dtype=float),
        dipole_moments=np.asarray(m_opt, dtype=float),
        loss_history=np.asarray(res.loss_history, dtype=float),
    )

