from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NescinCurrentSurface:
    """Fourier description of the NESCOIL 'Current Surface' section in a nescin file."""

    xm: np.ndarray  # (mnmax,), int
    xn: np.ndarray  # (mnmax,), int (as in file, not multiplied by nfp)
    rmnc: np.ndarray  # (mnmax,), float
    zmns: np.ndarray  # (mnmax,), float
    rmns: np.ndarray  # (mnmax,), float
    zmnc: np.ndarray  # (mnmax,), float


def read_nescin_current_surface(path: str | Path) -> NescinCurrentSurface:
    """Parse the '------ Current Surface' section from a NESCOIL nescin file.

    Matches regcoil_read_nescin.f90 behavior (but does not apply the -nfp scaling/sign flip).
    """
    path = Path(path)
    match = "------ Current Surface"
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Find the start marker line.
    start = None
    for i, line in enumerate(lines):
        if line.startswith(match):
            start = i
            break
    if start is None:
        raise ValueError(f"Could not find nescin marker {match!r} in {path}")

    # Fortran: after marker line, read(iunit, *) (skip 1 line), read mnmax, then skip 2 lines, then table.
    i = start + 1
    if i >= len(lines):
        raise ValueError(f"Unexpected EOF after {match!r} in {path}")

    # Skip one line (usually: "Number of fourier modes in table")
    i += 1
    if i >= len(lines):
        raise ValueError(f"Unexpected EOF while reading mnmax in {path}")

    # Read mnmax (may be the only token on the line)
    mnmax = int(lines[i].strip().split()[0])
    i += 1

    # Skip 2 header lines (typically: "Table of fourier coefficients" and column header)
    i += 2
    if i >= len(lines):
        raise ValueError(f"Unexpected EOF while reading table in {path}")

    xm = np.zeros((mnmax,), dtype=int)
    xn = np.zeros((mnmax,), dtype=int)
    rmnc = np.zeros((mnmax,), dtype=float)
    zmns = np.zeros((mnmax,), dtype=float)
    rmns = np.zeros((mnmax,), dtype=float)
    zmnc = np.zeros((mnmax,), dtype=float)

    for k in range(mnmax):
        if i + k >= len(lines):
            raise ValueError(f"Unexpected EOF while reading mode {k+1}/{mnmax} in {path}")
        parts = lines[i + k].strip().replace("D", "E").split()
        if len(parts) < 6:
            raise ValueError(f"Malformed nescin table line: {lines[i+k]!r}")
        xm[k] = int(float(parts[0]))
        xn[k] = int(float(parts[1]))
        rmnc[k] = float(parts[2])
        zmns[k] = float(parts[3])
        rmns[k] = float(parts[4])
        zmnc[k] = float(parts[5])

    return NescinCurrentSurface(xm=xm, xn=xn, rmnc=rmnc, zmns=zmns, rmns=rmns, zmnc=zmnc)

