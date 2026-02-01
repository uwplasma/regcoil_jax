from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FocusSurface:
    """FOCUS rdsurf plasma boundary (REGCOIL geometry_option_plasma=7)."""

    mnmax: int
    nfp: int
    nbf: int
    xn: np.ndarray  # (mnmax,), int (already includes nfp, i.e. multiplied in file parser)
    xm: np.ndarray  # (mnmax,), int
    rmnc: np.ndarray  # (mnmax,), float
    rmns: np.ndarray  # (mnmax,), float
    zmnc: np.ndarray  # (mnmax,), float
    zmns: np.ndarray  # (mnmax,), float
    # Optional Bn coefficients (if nbf>0): used when load_bnorm=.true.
    bfn: np.ndarray | None  # (nbf,), int (already includes nfp)
    bfm: np.ndarray | None  # (nbf,), int
    bfc: np.ndarray | None  # (nbf,), float (cos coeffs)
    bfs: np.ndarray | None  # (nbf,), float (sin coeffs)


def read_focus_surface(path: str | Path) -> FocusSurface:
    """Read a FOCUS rdsurf-format boundary file.

    Matches the parsing logic in `regcoil_init_plasma_mod.f90` (geometry_option_plasma=7).
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 6:
        raise ValueError(f"Invalid FOCUS surface file (too short): {path}")

    # Skip first line (comment), then read: mnmax_plasma, nfp, nbf
    parts = lines[1].split()
    if len(parts) < 3:
        raise ValueError(f"Invalid FOCUS header line: {lines[1]!r}")
    mnmax = int(parts[0])
    nfp = int(parts[1])
    nbf = int(parts[2])

    # Skip 2 lines, then mnmax rows with: xn, xm, rmnc, rmns, zmnc, zmns
    start = 4
    if len(lines) < start + mnmax:
        raise ValueError(f"Invalid FOCUS surface file (expected {mnmax} surface rows): {path}")

    xn = np.zeros((mnmax,), dtype=int)
    xm = np.zeros((mnmax,), dtype=int)
    rmnc = np.zeros((mnmax,), dtype=float)
    rmns = np.zeros((mnmax,), dtype=float)
    zmnc = np.zeros((mnmax,), dtype=float)
    zmns = np.zeros((mnmax,), dtype=float)

    for i in range(mnmax):
        p = lines[start + i].strip().replace("D", "E").split()
        if len(p) < 6:
            raise ValueError(f"Malformed FOCUS surface row: {lines[start+i]!r}")
        xn[i] = int(float(p[0])) * nfp  # include nfp (Fortran does this)
        xm[i] = int(float(p[1]))
        rmnc[i] = float(p[2])
        rmns[i] = float(p[3])
        zmnc[i] = float(p[4])
        zmns[i] = float(p[5])

    # Optional Bn coefficients section:
    bfn = bfm = bfc = bfs = None
    if nbf > 0:
        # Fortran skips 2 lines after the surface table, then reads nbf lines: bfn, bfm, bfc, bfs
        bn_start = start + mnmax + 2
        if len(lines) < bn_start + nbf:
            raise ValueError(f"Invalid FOCUS surface file (expected {nbf} Bn rows): {path}")
        bfn = np.zeros((nbf,), dtype=int)
        bfm = np.zeros((nbf,), dtype=int)
        bfc = np.zeros((nbf,), dtype=float)
        bfs = np.zeros((nbf,), dtype=float)
        for i in range(nbf):
            p = lines[bn_start + i].strip().replace("D", "E").split()
            if len(p) < 4:
                raise ValueError(f"Malformed FOCUS Bn row: {lines[bn_start+i]!r}")
            bfn[i] = int(float(p[0])) * nfp  # include nfp (Fortran does this)
            bfm[i] = int(float(p[1]))
            bfc[i] = float(p[2])
            bfs[i] = float(p[3])

    return FocusSurface(
        mnmax=mnmax,
        nfp=nfp,
        nbf=nbf,
        xn=xn,
        xm=xm,
        rmnc=rmnc,
        rmns=rmns,
        zmnc=zmnc,
        zmns=zmns,
        bfn=bfn,
        bfm=bfm,
        bfc=bfc,
        bfs=bfs,
    )

