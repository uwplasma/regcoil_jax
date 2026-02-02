from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CutCoilsResult:
    filaments_xyz: list[np.ndarray]  # list of (N,3) arrays
    coil_currents: np.ndarray  # (ncoils,) current per filament [A]
    coils_per_half_period: int
    nfp: int


def _roll_theta(*, theta: np.ndarray, data_zt: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same theta 'shift' convention as the Fortran repo's cutCoilsFromRegcoil script.

    Args:
      theta: (ntheta,)
      data_zt: (nzeta, ntheta)
    """
    if shift == 0:
        return theta, data_zt
    th = np.roll(theta, shift)
    # Rebuild a monotonic array with the shifted first value:
    th = th[0] + np.linspace(0.0, 2.0 * np.pi, len(th), endpoint=False)
    data = np.roll(data_zt, shift, axis=1)
    return th, data


def _bilinear_interp_periodic_zt3(
    *,
    r_zt3: np.ndarray,
    zeta: np.ndarray,
    theta: np.ndarray,
    query_zeta: np.ndarray,
    query_theta: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation on a periodic uniform (zeta,theta) grid for a vector field.

    r_zt3: (nzeta, ntheta, 3)
    zeta:  (nzeta,) assumed uniform over [0, 2π)
    theta: (ntheta,) assumed uniform over [0, 2π)
    query_*: (N,) in radians
    """
    r = np.asarray(r_zt3, dtype=float)
    if r.ndim != 3 or r.shape[2] != 3:
        raise ValueError(f"Expected r_zt3 (nzeta,ntheta,3), got {r.shape}")
    nzeta, ntheta, _ = r.shape

    z0 = float(zeta[0])
    t0 = float(theta[0])
    dz = float((2.0 * np.pi) / nzeta)
    dt = float((2.0 * np.pi) / ntheta)

    z = (np.asarray(query_zeta, dtype=float) - z0) % (2.0 * np.pi)
    t = (np.asarray(query_theta, dtype=float) - t0) % (2.0 * np.pi)

    j = np.floor(z / dz).astype(int) % nzeta
    i = np.floor(t / dt).astype(int) % ntheta
    jp = (j + 1) % nzeta
    ip = (i + 1) % ntheta

    wz = (z / dz) - np.floor(z / dz)
    wt = (t / dt) - np.floor(t / dt)

    p00 = r[j, i]
    p01 = r[j, ip]
    p10 = r[jp, i]
    p11 = r[jp, ip]

    out = (
        (1.0 - wt)[:, None] * (1.0 - wz)[:, None] * p00
        + wt[:, None] * (1.0 - wz)[:, None] * p01
        + (1.0 - wt)[:, None] * wz[:, None] * p10
        + wt[:, None] * wz[:, None] * p11
    )
    return out


def _contours_matplotlib(
    *,
    zeta_3: np.ndarray,
    theta: np.ndarray,
    data_3_zt: np.ndarray,
    levels: np.ndarray,
) -> list[np.ndarray]:
    """Return contour polylines as arrays of shape (N,2) with columns (zeta, theta)."""
    try:
        import os
        import tempfile
        import matplotlib

        # Avoid user-home cache issues in CI / sandboxes.
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for coil cutting (install regcoil_jax[viz]).") from e

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    cs = ax.contour(zeta_3, theta, data_3_zt.T, levels=levels)
    plt.close(fig)

    polylines = []
    for coll in cs.collections:
        paths = coll.get_paths()
        if not paths:
            continue
        # Pick the longest path (the most common case is a single closed contour).
        verts = max((p.vertices for p in paths), key=lambda v: v.shape[0])
        polylines.append(np.asarray(verts, dtype=float))
    return polylines


def cut_coils_from_current_potential(
    *,
    current_potential_zt: Any,
    theta: Any,
    zeta: Any,
    r_coil_zt3_full: Any,
    theta_shift: int,
    coils_per_half_period: int,
    nfp: int,
    net_poloidal_current_Amperes: float,
) -> CutCoilsResult:
    """Cut filamentary coils as current-potential contours (REGCOIL-style).

    This is a close port of the `regcoil/cutCoilsFromRegcoil` script:
    - Normalize `current_potential` by `net_poloidal_current_Amperes` and scale by `nfp`
      so the secular term spans ~[0,1) over one field period.
    - Extract `2*coils_per_half_period` contours in [0,1) for one field period.
    - Replicate each contour across `nfp` field periods.

    Args:
      current_potential_zt: (nzeta, ntheta) on one field period
      theta, zeta: 1D arrays for one field period
      r_coil_zt3_full: (nzetal, ntheta, 3) coil winding surface in XYZ for all field periods,
        where nzetal = nzeta*nfp and the zeta coordinate spans [0, 2π).
    """
    theta = np.asarray(theta, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    data = np.asarray(current_potential_zt, dtype=float)
    r_full = np.asarray(r_coil_zt3_full, dtype=float)

    if data.shape != (zeta.size, theta.size):
        raise ValueError(f"current_potential_zt shape {data.shape} != (nzeta,ntheta)=({zeta.size},{theta.size})")
    if r_full.ndim != 3 or r_full.shape[1] != theta.size or r_full.shape[2] != 3:
        raise ValueError(f"r_coil_zt3_full shape {r_full.shape} != (nzetal,ntheta,3)")
    if r_full.shape[0] != zeta.size * nfp:
        raise ValueError("r_coil_zt3_full must include all field periods: nzetal == nzeta*nfp")

    # Normalize potential as in the reference script.
    if abs(net_poloidal_current_Amperes) > np.finfo(float).eps:
        data = data / net_poloidal_current_Amperes * nfp
    else:
        data = data / np.max(np.abs(data))

    theta, data = _roll_theta(theta=theta, data_zt=data, shift=int(theta_shift))

    # Extend zeta to 3 copies to avoid contours breaking at the zeta=0 seam.
    d = 2.0 * np.pi / nfp
    zeta_3 = np.concatenate((zeta - d, zeta, zeta + d), axis=0)
    data_3 = np.concatenate((data - 1.0, data, data + 1.0), axis=0)  # (3*nzeta, ntheta)

    # Contour levels: 2*coils_per_half_period values in [0,1).
    nlevels = int(2 * coils_per_half_period)
    levels = np.linspace(0.0, 1.0, nlevels, endpoint=False)
    levels = levels + (levels[1] - levels[0]) / 2.0

    base_polylines = _contours_matplotlib(zeta_3=zeta_3, theta=theta, data_3_zt=data_3, levels=levels)
    if len(base_polylines) != nlevels:
        # Keep going; we still return whatever we found.
        pass

    filaments_xyz: list[np.ndarray] = []

    # Full-zeta grid for interpolation.
    zeta_full = np.linspace(0.0, 2.0 * np.pi, zeta.size * nfp, endpoint=False)

    for poly in base_polylines:
        z = poly[:, 0]
        t = poly[:, 1]
        # Ensure increasing theta like the reference script does (helps avoid zigzags on some cases).
        if t.size >= 2 and (t[1] < t[0]):
            z = z[::-1]
            t = t[::-1]

        for jfp in range(nfp):
            z_j = z + jfp * d
            # Interpolate on the *full* winding surface.
            pts = _bilinear_interp_periodic_zt3(
                r_zt3=r_full,
                zeta=zeta_full,
                theta=theta,
                query_zeta=z_j,
                query_theta=t,
            )
            filaments_xyz.append(pts)

    ncoils = len(filaments_xyz)
    if ncoils == 0:
        raise RuntimeError("No coils were found by contouring current_potential.")

    coil_current = float(net_poloidal_current_Amperes) / float(ncoils) if abs(net_poloidal_current_Amperes) > 0 else 0.0
    coil_currents = np.full((ncoils,), coil_current, dtype=float)

    return CutCoilsResult(
        filaments_xyz=filaments_xyz,
        coil_currents=coil_currents,
        coils_per_half_period=int(coils_per_half_period),
        nfp=int(nfp),
    )


def write_makecoil_filaments(
    path: str | Path,
    *,
    filaments_xyz: list[np.ndarray],
    coil_currents: np.ndarray | None = None,
    coil_current: float | None = None,
    nfp: int,
) -> None:
    """Write a `coils.*` filament file (MAKECOIL-style), matching regcoil's helper script."""
    path = Path(path)
    if coil_currents is None:
        if coil_current is None:
            raise ValueError("Must provide either coil_currents or coil_current")
        coil_currents = np.full((len(filaments_xyz),), float(coil_current), dtype=float)
    coil_currents = np.asarray(coil_currents, dtype=float).reshape(-1)
    if coil_currents.size != len(filaments_xyz):
        raise ValueError("coil_currents must have length equal to number of filaments")

    with path.open("w", encoding="utf-8") as f:
        f.write(f"periods {int(nfp)}\n")
        f.write("begin filament\n")
        f.write("mirror NIL\n")
        for pts, I in zip(filaments_xyz, coil_currents):
            pts = np.asarray(pts, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError(f"filament points must be (N,3), got {pts.shape}")
            for (x, y, z) in pts:
                f.write(f"{x:14.22e} {y:14.22e} {z:14.22e} {float(I):14.22e}\n")
            # Close the loop with MAKECOIL's sentinel.
            x0, y0, z0 = pts[0]
            f.write(f"{x0:14.22e} {y0:14.22e} {z0:14.22e} {0.0:14.22e} 1 Modular\n")
        f.write("end\n")
