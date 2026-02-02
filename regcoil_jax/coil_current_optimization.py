from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .biot_savart_jax import FilamentSegments, bnormal_from_segments
from .optimize import minimize_adam, OptimizeResult


@dataclass(frozen=True)
class CurrentOptimizationResult:
    coil_currents: np.ndarray  # (ncoils,)
    loss_history: np.ndarray   # (steps,)


def optimize_coil_currents_to_match_bnormal(
    segs: FilamentSegments,
    *,
    points: Any,
    normals_unit: Any,
    target_bnormal: Any,
    coil_currents0: Any,
    steps: int = 200,
    lr: float = 2e-2,
    l2_reg: float = 1e-6,
    net_current_target: float | None = None,
    net_current_weight: float = 1e-2,
    seg_batch: int = 2048,
) -> CurrentOptimizationResult:
    """Optimize per-coil currents to match a target B·n on the plasma surface.

    This is a JAX-autodiff replacement for Fortran-style adjoint sensitivity workflows:
    the gradient d(loss)/dI is obtained automatically by differentiating through the
    Biot–Savart evaluation.

    Args:
      segs: fixed coil geometry (segments)
      points: (N,3) evaluation points
      normals_unit: (N,3) unit normals at points
      target_bnormal: (N,) desired B·n (Tesla)
      coil_currents0: (ncoils,) initial currents [A]
      net_current_target: if set, add penalty on sum(I)-net_current_target
    """
    pts = jnp.asarray(points, dtype=jnp.float64)
    nhat = jnp.asarray(normals_unit, dtype=jnp.float64)
    targ = jnp.asarray(target_bnormal, dtype=jnp.float64).reshape(-1)
    I0 = jnp.asarray(coil_currents0, dtype=jnp.float64).reshape(-1)
    if int(I0.shape[0]) != int(segs.n_filaments):
        raise ValueError("coil_currents0 must have length equal to segs.n_filaments")

    if int(pts.shape[0]) != int(targ.shape[0]):
        raise ValueError("target_bnormal must have length equal to number of points")

    net_target = None if net_current_target is None else float(net_current_target)

    def loss_fn(I: jnp.ndarray) -> jnp.ndarray:
        bn = bnormal_from_segments(
            segs,
            points=pts,
            normals_unit=nhat,
            filament_currents=I,
            seg_batch=int(seg_batch),
        )
        # Mean-square mismatch + mild L2 regularization.
        err = bn - targ
        loss = jnp.mean(err * err) + float(l2_reg) * jnp.mean((I - I0) * (I - I0))
        if net_target is not None:
            loss = loss + float(net_current_weight) * (jnp.sum(I) - net_target) ** 2
        return loss

    res: OptimizeResult = minimize_adam(loss_fn, I0, steps=int(steps), lr=float(lr), jit=True)
    return CurrentOptimizationResult(
        coil_currents=np.asarray(res.x, dtype=float),
        loss_history=np.asarray(res.loss_history, dtype=float),
    )

