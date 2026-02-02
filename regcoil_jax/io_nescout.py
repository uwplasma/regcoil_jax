from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .constants import mu0


@dataclass(frozen=True)
class NescoutPotentials:
    curpol: float
    solutions: list[np.ndarray]  # each (nbasis,)


def _starts_with(line: str, prefix: str) -> bool:
    return line[: len(prefix)] == prefix


def _parse_three_int_float(line: str) -> tuple[int, int, float]:
    parts = line.strip().split()
    if len(parts) < 3:
        raise ValueError(f"Invalid nescout coefficient line (expected 'm n amp'): {line!r}")
    return int(parts[0]), int(parts[1]), float(parts[2])


def read_nescout_potentials(
    path: str,
    *,
    mpol_potential: int,
    ntor_potential: int,
    nfp: int,
    symmetry_option: int = 1,
) -> NescoutPotentials:
    """Read one or more current potentials from a NESCOIL ``nescout`` file.

    This matches the Fortran logic in ``regcoil_compute_diagnostics_for_nescout_potential.f90``:
      - find and read ``curpol`` from the header line
      - for each ``---- Phi(m,n) for`` block, read Phi(m,n) in NESCOIL convention
      - map to REGCOIL coefficient ordering & sign conventions
      - convert NESCOIL normalization to REGCOIL normalization via ``curpol/mu0``

    Notes:
      - REGCOIL's mapping fills only the *single-valued* Fourier coefficient vector
        (i.e. the first block of modes). This matches the Fortran implementation and
        is sufficient for NESCOIL-style runs where stellarator symmetry is used.
      - ``symmetry_option=3`` (no symmetry) is accepted, but the returned solutions
        will populate only the first (mnmax,) block, leaving the second block zeros
        as in Fortran.
    """
    mpol_potential = int(mpol_potential)
    ntor_potential = int(ntor_potential)
    nfp = int(nfp)
    symmetry_option = int(symmetry_option)

    if mpol_potential < 0 or ntor_potential < 0:
        raise ValueError("mpol_potential and ntor_potential must be non-negative")

    # Match regcoil_init_Fourier_modes_mod.f90:
    mnmax = mpol_potential * (2 * ntor_potential + 1) + ntor_potential
    nbasis = mnmax if symmetry_option in (1, 2) else 2 * mnmax

    match_curpol = "np, iota_edge, phip_edge, curpol"
    match_phi = "---- Phi(m,n) for"
    match_phi_end = "---- end Phi(m,n)"

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    curpol = None
    for i, line in enumerate(lines):
        if _starts_with(line, match_curpol):
            if i + 1 >= len(lines):
                raise ValueError("nescout file ended before curpol line values")
            vals = lines[i + 1].strip().split()
            if len(vals) < 4:
                raise ValueError(f"Could not parse curpol line: {lines[i + 1]!r}")
            curpol = float(vals[3])
            break
    if curpol is None:
        raise ValueError(f"Could not find curpol header line {match_curpol!r} in {path!r}")

    sols: list[np.ndarray] = []

    idx = 0
    while idx < len(lines):
        if not _starts_with(lines[idx], match_phi):
            idx += 1
            continue

        # Found a potential block.
        idx += 1
        sol = np.zeros((nbasis,), dtype=float)

        # Fortran reads m=0..mpol and n=-ntor..ntor (inclusive).
        for mm in range(0, mpol_potential + 1):
            for nn in range(-ntor_potential, ntor_potential + 1):
                if idx >= len(lines):
                    raise ValueError("nescout file ended unexpectedly while reading Phi(m,n) block")
                m, n, amplitude = _parse_three_int_float(lines[idx])
                idx += 1

                if (m != mm) or (n != nn):
                    raise ValueError(
                        "Unexpected mode ordering in nescout file. "
                        f"Expected (m,n)=({mm},{nn}) but got ({m},{n})."
                    )

                if mm == 0 and nn > 0:
                    # Need *2 here since NESCOIL prints n<0 values for m=0 (REGCOIL comment).
                    sol[nn - 1] = amplitude * (-2.0)
                elif mm > 0:
                    # NESCOIL convention is m*u + n*v; REGCOIL is m*u - n*v.
                    # So map n -> -n, and apply nfp scaling used in REGCOIL.
                    #
                    # Fortran indexing (1-based):
                    #   index = ntor + (m-1)*(2*ntor+1) + ntor - n + 1
                    # Convert to 0-based:
                    #   idx0  = ntor + (m-1)*(2*ntor+1) + ntor - n
                    idx0 = ntor_potential + (mm - 1) * (2 * ntor_potential + 1) + (ntor_potential - nn)
                    if not (0 <= idx0 < mnmax):
                        raise ValueError(f"Computed out-of-range coefficient index {idx0} for (m,n)=({mm},{nn})")
                    sol[idx0] = amplitude

        # Next line must be end marker.
        if idx >= len(lines) or (not _starts_with(lines[idx], match_phi_end)):
            got = lines[idx] if idx < len(lines) else "<eof>"
            raise ValueError(f"Expected {match_phi_end!r} after Phi(m,n) block, got: {got!r}")
        idx += 1

        # Convert from NESCOIL normalization to REGCOIL normalization.
        sol *= (curpol / mu0)

        sols.append(sol)

    return NescoutPotentials(curpol=float(curpol), solutions=sols)

