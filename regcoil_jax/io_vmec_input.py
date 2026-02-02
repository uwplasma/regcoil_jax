from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry_fourier import FourierSurface


@dataclass(frozen=True)
class VmecInputBoundary:
    """Boundary Fourier coefficients parsed from a VMEC ``input.*`` file.

    This reader is intentionally minimal: it targets the boundary-only information needed
    to run REGCOIL on a fixed plasma boundary without requiring a VMEC ``wout`` file.
    """

    nfp: int
    lasym: bool
    xm: np.ndarray  # (mnmax,)
    xn: np.ndarray  # (mnmax,) already multiplied by nfp
    rmnc: np.ndarray
    zmns: np.ndarray
    rmns: np.ndarray
    zmnc: np.ndarray


_RE_NFP = re.compile(r"^\s*NFP\s*=\s*([0-9]+)\s*$", re.IGNORECASE)
_RE_LASYM = re.compile(r"^\s*LASYM\s*=\s*([TF])\s*$", re.IGNORECASE)

# VMEC boundary lines typically look like:
#   RBC(   0,   0) = 1.0,      ZBS(   0,   0) = 0.0
# The file itself often includes the note "n comes before m!" for these arrays.
_RE_RBC = re.compile(r"RBC\(\s*([-0-9]+)\s*,\s*([-0-9]+)\s*\)\s*=\s*([+\-0-9.eEdD]+)")
_RE_ZBS = re.compile(r"ZBS\(\s*([-0-9]+)\s*,\s*([-0-9]+)\s*\)\s*=\s*([+\-0-9.eEdD]+)")
_RE_RBS = re.compile(r"RBS\(\s*([-0-9]+)\s*,\s*([-0-9]+)\s*\)\s*=\s*([+\-0-9.eEdD]+)")
_RE_ZBC = re.compile(r"ZBC\(\s*([-0-9]+)\s*,\s*([-0-9]+)\s*\)\s*=\s*([+\-0-9.eEdD]+)")


def _to_float(s: str) -> float:
    # VMEC input files sometimes use 'D' exponent.
    return float(s.replace("D", "E").replace("d", "e"))


def read_vmec_input_boundary(path: str | Path) -> VmecInputBoundary:
    """Parse a VMEC ``input.*`` file and return boundary Fourier coefficients.

    For symmetric inputs (``LASYM=F``), VMEC uses:
      R(θ,ζ) = Σ RBC(n,m) cos(m θ - n NFP ζ)
      Z(θ,ζ) = Σ ZBS(n,m) sin(m θ - n NFP ζ)

    For asymmetric inputs (``LASYM=T``), the additional families are:
      R += Σ RBS(n,m) sin(...)
      Z += Σ ZBC(n,m) cos(...)

    In REGCOIL / ``FourierSurface`` conventions, we store:
      xm = m
      xn = n*NFP
      angle = m*theta - xn*zeta
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    nfp = None
    lasym = None

    # maps (m, n) -> val, where n is the VMEC n index (not multiplied by nfp)
    rbc: dict[tuple[int, int], float] = {}
    zbs: dict[tuple[int, int], float] = {}
    rbs: dict[tuple[int, int], float] = {}
    zbc: dict[tuple[int, int], float] = {}

    for line in text:
        m = _RE_NFP.match(line)
        if m:
            nfp = int(m.group(1))
            continue
        m = _RE_LASYM.match(line)
        if m:
            lasym = (m.group(1).upper() == "T")
            continue

        for mm in _RE_RBC.finditer(line):
            n_idx = int(mm.group(1))
            m_idx = int(mm.group(2))
            rbc[(m_idx, n_idx)] = _to_float(mm.group(3))
        for mm in _RE_ZBS.finditer(line):
            n_idx = int(mm.group(1))
            m_idx = int(mm.group(2))
            zbs[(m_idx, n_idx)] = _to_float(mm.group(3))
        for mm in _RE_RBS.finditer(line):
            n_idx = int(mm.group(1))
            m_idx = int(mm.group(2))
            rbs[(m_idx, n_idx)] = _to_float(mm.group(3))
        for mm in _RE_ZBC.finditer(line):
            n_idx = int(mm.group(1))
            m_idx = int(mm.group(2))
            zbc[(m_idx, n_idx)] = _to_float(mm.group(3))

    if nfp is None:
        raise ValueError(f"Could not find NFP in VMEC input file: {path}")
    if lasym is None:
        lasym = False

    keys = set(rbc.keys()) | set(zbs.keys())
    if lasym:
        keys |= set(rbs.keys()) | set(zbc.keys())
    if not keys:
        raise ValueError(f"No boundary coefficients (RBC/ZBS) found in VMEC input file: {path}")

    # Sort primarily by m, then by n, matching typical wout ordering and stable outputs.
    pairs = sorted(keys, key=lambda t: (t[0], t[1]))
    xm = np.array([m for (m, n) in pairs], dtype=int)
    xn = np.array([n * nfp for (m, n) in pairs], dtype=int)

    rmnc = np.array([rbc.get((m, n), 0.0) for (m, n) in pairs], dtype=float)
    zmns = np.array([zbs.get((m, n), 0.0) for (m, n) in pairs], dtype=float)
    rmns = np.array([rbs.get((m, n), 0.0) for (m, n) in pairs], dtype=float)
    zmnc = np.array([zbc.get((m, n), 0.0) for (m, n) in pairs], dtype=float)

    return VmecInputBoundary(
        nfp=int(nfp),
        lasym=bool(lasym),
        xm=xm,
        xn=xn,
        rmnc=rmnc,
        zmns=zmns,
        rmns=rmns,
        zmnc=zmnc,
    )


def vmec_input_boundary_as_fourier_surface(bound: VmecInputBoundary) -> FourierSurface:
    """Convert :class:`VmecInputBoundary` to a JAX ``FourierSurface``."""
    import jax.numpy as jnp

    return FourierSurface(
        nfp=int(bound.nfp),
        lasym=bool(bound.lasym),
        xm=jnp.asarray(bound.xm, dtype=jnp.int32),
        xn=jnp.asarray(bound.xn, dtype=jnp.int32),
        rmnc=jnp.asarray(bound.rmnc, dtype=jnp.float64),
        zmns=jnp.asarray(bound.zmns, dtype=jnp.float64),
        rmns=jnp.asarray(bound.rmns, dtype=jnp.float64),
        zmnc=jnp.asarray(bound.zmnc, dtype=jnp.float64),
    )

