from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FourierTableSurface:
    """Fourier surface read from a simple ASCII coefficient table (REGCOIL geometry_option_plasma=6)."""

    xm: np.ndarray  # (mnmax,), int
    xn: np.ndarray  # (mnmax,), int (already includes nfp if the file uses VMEC convention)
    rmnc: np.ndarray  # (mnmax,), float
    zmns: np.ndarray  # (mnmax,), float
    rmns: np.ndarray  # (mnmax,), float
    zmnc: np.ndarray  # (mnmax,), float


def read_surface_fourier_table(path: str | Path) -> FourierTableSurface:
    """Read a Fourier surface from REGCOIL's simple ASCII table format.

    This matches the Fortran branch:
      geometry_option_plasma = 6
    in `regcoil_init_plasma_mod.f90`.

    File format:
      - first line: comment (ignored)
      - second line: mnmax
      - third line: header (ignored)
      - then mnmax lines:
          m  n  rmnc  zmns  rmns  zmnc
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 4:
        raise ValueError(f"Invalid Fourier table surface file (too short): {path}")

    mnmax = int(lines[1].strip().split()[0])
    start = 3
    if len(lines) < start + mnmax:
        raise ValueError(f"Invalid Fourier table surface file (expected {mnmax} rows): {path}")

    xm = np.zeros((mnmax,), dtype=int)
    xn = np.zeros((mnmax,), dtype=int)
    rmnc = np.zeros((mnmax,), dtype=float)
    zmns = np.zeros((mnmax,), dtype=float)
    rmns = np.zeros((mnmax,), dtype=float)
    zmnc = np.zeros((mnmax,), dtype=float)

    for i in range(mnmax):
        parts = lines[start + i].strip().replace("D", "E").split()
        if len(parts) < 6:
            raise ValueError(f"Malformed coefficient row in {path}: {lines[start+i]!r}")
        xm[i] = int(float(parts[0]))
        xn[i] = int(float(parts[1]))
        rmnc[i] = float(parts[2])
        zmns[i] = float(parts[3])
        rmns[i] = float(parts[4])
        zmnc[i] = float(parts[5])

    return FourierTableSurface(xm=xm, xn=xn, rmnc=rmnc, zmns=zmns, rmns=rmns, zmnc=zmnc)

