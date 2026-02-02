from __future__ import annotations

from typing import Any

import numpy as np


def normals_from_r_zt3(*, r_zt3: Any) -> np.ndarray:
    """Compute non-unit surface normals from a periodic (zeta,theta) grid of points.

    Uses centered finite differences on uniform grids and matches REGCOIL convention:
      normal = dr/dzeta × dr/dtheta
    """
    r = np.asarray(r_zt3, dtype=float)
    if r.ndim != 3 or r.shape[2] != 3:
        raise ValueError(f"Expected r_zt3 shape (nzetal,ntheta,3), got {r.shape}")
    nzetal, ntheta, _ = r.shape
    dtheta = (2.0 * np.pi) / ntheta
    dzeta = (2.0 * np.pi) / nzetal
    dr_dtheta = (np.roll(r, -1, axis=1) - np.roll(r, 1, axis=1)) / (2.0 * dtheta)
    dr_dzeta = (np.roll(r, -1, axis=0) - np.roll(r, 1, axis=0)) / (2.0 * dzeta)
    return np.cross(dr_dzeta, dr_dtheta)


def unit_normals_from_r_zt3(*, r_zt3: Any) -> np.ndarray:
    n = normals_from_r_zt3(r_zt3=r_zt3)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / (norm + 1e-300)


def reshape_first_period(*, r_full: Any, nfp: int) -> np.ndarray:
    """Return the first field period (nzeta, ntheta, 3) from a full-torus (nzetal, ntheta, 3) array."""
    r = np.asarray(r_full, dtype=float)
    if r.ndim != 3 or r.shape[2] != 3:
        raise ValueError("r_full must be (nzetal,ntheta,3)")
    nfp = int(nfp)
    if nfp <= 0:
        raise ValueError("nfp must be positive")
    nzeta = r.shape[0] // nfp
    return r[:nzeta]

