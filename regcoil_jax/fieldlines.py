from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FilamentField:
    """Preprocessed filament set for Biot–Savart evaluation."""

    seg_midpoints: np.ndarray  # (M,3)
    seg_dls: np.ndarray  # (M,3)
    seg_currents: np.ndarray  # (M,)


def build_filament_field(*, filaments_xyz: list[Any], coil_current: float) -> FilamentField:
    mids = []
    dls = []
    currents = []
    for pts_any in filaments_xyz:
        pts = np.asarray(pts_any, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"filament points must be (N,3), got {pts.shape}")
        r0 = pts
        r1 = np.roll(pts, shift=-1, axis=0)
        dl = r1 - r0
        mid = 0.5 * (r0 + r1)
        mids.append(mid)
        dls.append(dl)
        currents.append(np.full((mid.shape[0],), float(coil_current), dtype=float))

    seg_mid = np.concatenate(mids, axis=0)
    seg_dl = np.concatenate(dls, axis=0)
    seg_I = np.concatenate(currents, axis=0)
    return FilamentField(seg_midpoints=seg_mid, seg_dls=seg_dl, seg_currents=seg_I)


def bfield_from_filaments(field: FilamentField, x: np.ndarray, *, mu0: float = 4e-7 * np.pi, eps: float = 1e-9) -> np.ndarray:
    """Magnetic field from filament segments at point(s) `x`.

    Uses a midpoint rule for each segment:
      dB = μ0 I/(4π) * (dl × R) / |R|^3
    """
    x = np.asarray(x, dtype=float)
    single_point = x.ndim == 1
    if single_point:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"x must be (3,) or (N,3), got {x.shape}")

    R = x[:, None, :] - field.seg_midpoints[None, :, :]  # (N,M,3)
    r2 = np.sum(R * R, axis=-1) + eps * eps
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))

    cr = np.cross(field.seg_dls[None, :, :], R)  # (N,M,3)
    coeff = (mu0 / (4.0 * np.pi)) * field.seg_currents[None, :] * inv_r3  # (N,M)
    B = np.sum(cr * coeff[:, :, None], axis=1)  # (N,3)
    return B[0] if single_point else B


def trace_fieldline(
    field: FilamentField,
    *,
    x0: np.ndarray,
    ds: float,
    n_steps: int,
    direction: float = 1.0,
    stop_radius: float | None = None,
) -> np.ndarray:
    """Trace a single field line using fixed-step RK4 on the *coil-only* field."""
    x = np.asarray(x0, dtype=float).reshape(3)
    pts = [x.copy()]

    def f(x_):
        B = bfield_from_filaments(field, x_)
        n = np.linalg.norm(B)
        if n == 0.0:
            return np.zeros((3,))
        return direction * (B / n)

    for _ in range(int(n_steps)):
        k1 = f(x)
        k2 = f(x + 0.5 * ds * k1)
        k3 = f(x + 0.5 * ds * k2)
        k4 = f(x + ds * k3)
        x = x + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        pts.append(x.copy())
        if stop_radius is not None:
            R = float(np.linalg.norm(x[:2]))
            if R > float(stop_radius):
                break
    return np.asarray(pts, dtype=float)


def trace_fieldlines(
    field: FilamentField,
    *,
    starts: np.ndarray,
    ds: float,
    n_steps: int,
    stop_radius: float | None = None,
) -> list[np.ndarray]:
    """Trace multiple field lines (both directions) from each start point."""
    starts = np.asarray(starts, dtype=float)
    if starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError("starts must be (N,3)")

    lines = []
    for x0 in starts:
        fwd = trace_fieldline(field, x0=x0, ds=ds, n_steps=n_steps, direction=+1.0, stop_radius=stop_radius)
        bwd = trace_fieldline(field, x0=x0, ds=ds, n_steps=n_steps, direction=-1.0, stop_radius=stop_radius)
        # Stitch (reverse bwd excluding first point, then fwd)
        line = np.concatenate([bwd[::-1][:-1], fwd], axis=0)
        lines.append(line)
    return lines

