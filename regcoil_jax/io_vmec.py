from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

try:
    import netCDF4
except Exception as e:
    netCDF4 = None

@dataclass(frozen=True)
class VmecBoundary:
    nfp: int
    lasym: bool
    xm: np.ndarray
    xn: np.ndarray
    rmnc: np.ndarray
    zmns: np.ndarray
    rmns: np.ndarray
    zmnc: np.ndarray
    # For bnorm scaling / net poloidal current:
    bvco: np.ndarray | None
    bsubvmnc: np.ndarray | None
    ns: int | None
    Rmajor_p: float | None

@dataclass(frozen=True)
class VmecWoutMeta:
    """VMEC metadata needed for REGCOIL plasma geometry options beyond 2/3.

    This dataclass is intentionally numpy-only (no JAX) because it is used during
    one-time preprocessing. Downstream geometry evaluation is done via FourierSurface.
    """

    nfp: int
    lasym: bool
    ns: int
    mpol: int
    ntor: int
    xm: np.ndarray
    xn: np.ndarray
    lmns_last: np.ndarray | None

def read_wout_boundary(path: str, *, radial_mode: str = "full") -> VmecBoundary:
    """
    Read the outer boundary Fourier representation from a VMEC wout_*.nc.

    IMPORTANT: VMEC NetCDF files are not fully consistent across builds:
      - Some store rmnc as (mnmax, ns), others as (ns, mnmax).
      - Same for zmns, rmns, zmnc, and bsubvmnc.

    This routine detects the orientation using the length of xm/xn (mnmax) and
    always returns 1D boundary arrays of length mnmax.

    Args:
      radial_mode:
        - "full": use the outermost point in VMEC's FULL radial grid (REGCOIL geometry_option_plasma=2)
        - "half": average the outermost two FULL grid points to approximate the outermost HALF grid point
                  (REGCOIL geometry_option_plasma=3)
    """
    if netCDF4 is None:
        raise ImportError("netCDF4 is required to read VMEC wout files.")
    radial_mode = str(radial_mode).strip().lower()
    if radial_mode not in ("full", "half"):
        raise ValueError(f"radial_mode must be 'full' or 'half', got {radial_mode!r}")
    ds = netCDF4.Dataset(path, "r")
    nfp = int(ds.variables["nfp"][()])
    lasym = bool(int(ds.variables["lasym__logical__"][()])) if "lasym__logical__" in ds.variables else bool(ds.variables["lasym"][()])
    xm = np.array(ds.variables["xm"][()], dtype=int)
    xn = np.array(ds.variables["xn"][()], dtype=int)

    def _read2d(name: str) -> np.ndarray:
        if name not in ds.variables:
            return None
        return np.array(ds.variables[name][()])

    rmnc_all = _read2d("rmnc")
    zmns_all = _read2d("zmns")
    rmns_all = _read2d("rmns")
    zmnc_all = _read2d("zmnc")

    if rmnc_all is None or zmns_all is None:
        ds.close()
        raise ValueError("VMEC wout file is missing rmnc/zmns arrays.")

    mnmax = int(xm.size)

    # Determine orientation and ns:
    if rmnc_all.ndim != 2:
        ds.close()
        raise ValueError(f"Unexpected rmnc array rank {rmnc_all.ndim}, expected 2.")
    if radial_mode == "half" and min(rmnc_all.shape) < 2:
        ds.close()
        raise ValueError(f"radial_mode='half' requires at least 2 radial surfaces in rmnc, got shape={rmnc_all.shape}")
    if rmnc_all.shape[0] == mnmax:
        # (mnmax, ns)
        ns = int(rmnc_all.shape[1])
        if radial_mode == "half":
            rmnc = (0.5 * rmnc_all[:, -2] + 0.5 * rmnc_all[:, -1]).astype(float)
            zmns = (0.5 * zmns_all[:, -2] + 0.5 * zmns_all[:, -1]).astype(float)
            rmns = (0.5 * (rmns_all[:, -2] + rmns_all[:, -1]) if rmns_all is not None else np.zeros_like(rmnc_all[:, -1])).astype(float)
            zmnc = (0.5 * (zmnc_all[:, -2] + zmnc_all[:, -1]) if zmnc_all is not None else np.zeros_like(rmnc_all[:, -1])).astype(float)
        else:
            rmnc = rmnc_all[:, -1].astype(float)
            zmns = zmns_all[:, -1].astype(float)
            rmns = (rmns_all[:, -1] if rmns_all is not None else np.zeros_like(rmnc_all[:, -1])).astype(float)
            zmnc = (zmnc_all[:, -1] if zmnc_all is not None else np.zeros_like(rmnc_all[:, -1])).astype(float)
    elif rmnc_all.shape[1] == mnmax:
        # (ns, mnmax)
        ns = int(rmnc_all.shape[0])
        if radial_mode == "half":
            rmnc = (0.5 * rmnc_all[-2, :] + 0.5 * rmnc_all[-1, :]).astype(float)
            zmns = (0.5 * zmns_all[-2, :] + 0.5 * zmns_all[-1, :]).astype(float)
            rmns = (0.5 * (rmns_all[-2, :] + rmns_all[-1, :]) if rmns_all is not None else np.zeros_like(rmnc_all[-1, :])).astype(float)
            zmnc = (0.5 * (zmnc_all[-2, :] + zmnc_all[-1, :]) if zmnc_all is not None else np.zeros_like(rmnc_all[-1, :])).astype(float)
        else:
            rmnc = rmnc_all[-1, :].astype(float)
            zmns = zmns_all[-1, :].astype(float)
            rmns = (rmns_all[-1, :] if rmns_all is not None else np.zeros_like(rmnc_all[-1, :])).astype(float)
            zmnc = (zmnc_all[-1, :] if zmnc_all is not None else np.zeros_like(rmnc_all[-1, :])).astype(float)
    else:
        ds.close()
        raise ValueError(
            f"Could not infer rmnc orientation: rmnc.shape={rmnc_all.shape} but mnmax={mnmax}."
        )

    bvco = np.array(ds.variables["bvco"][()]) if "bvco" in ds.variables else None
    bsubvmnc = np.array(ds.variables["bsubvmnc"][()]) if "bsubvmnc" in ds.variables else None
    Rmajor_p = float(ds.variables["Rmajor_p"][()]) if "Rmajor_p" in ds.variables else None
    ds.close()

    return VmecBoundary(
        nfp=nfp,
        lasym=lasym,
        xm=xm,
        xn=xn,
        rmnc=rmnc,
        zmns=zmns,
        rmns=rmns,
        zmnc=zmnc,
        bvco=bvco,
        bsubvmnc=bsubvmnc,
        ns=ns,
        Rmajor_p=Rmajor_p,
    )

def read_wout_meta(path: str, *, need_lmns_last: bool = False) -> VmecWoutMeta:
    """Read VMEC metadata (and optionally ``lmns`` on the outermost surface).

    This helper exists primarily for REGCOIL plasma geometry_option_plasma=4, which
    requires the VMEC lambda spectrum ``lmns`` (straight-field-line coordinate transform).
    """
    if netCDF4 is None:
        raise ImportError("netCDF4 is required to read VMEC wout files.")
    ds = netCDF4.Dataset(path, "r")
    try:
        nfp = int(ds.variables["nfp"][()])
        lasym = bool(int(ds.variables["lasym__logical__"][()])) if "lasym__logical__" in ds.variables else bool(ds.variables["lasym"][()])
        ns = int(ds.variables["ns"][()]) if "ns" in ds.variables else None
        mpol = int(ds.variables["mpol"][()]) if "mpol" in ds.variables else None
        ntor = int(ds.variables["ntor"][()]) if "ntor" in ds.variables else None
        xm = np.array(ds.variables["xm"][()], dtype=int)
        xn = np.array(ds.variables["xn"][()], dtype=int)

        lmns_last = None
        if need_lmns_last:
            if "lmns" not in ds.variables:
                raise ValueError("VMEC wout file is missing lmns, required for geometry_option_plasma=4.")
            lmns_all = np.array(ds.variables["lmns"][()])
            mnmax = int(xm.size)
            if lmns_all.ndim != 2:
                raise ValueError(f"Unexpected lmns array rank {lmns_all.ndim}, expected 2.")
            if ns is None:
                # Infer ns from lmns array.
                ns = int(lmns_all.shape[0] if lmns_all.shape[1] == mnmax else lmns_all.shape[1])
            if lmns_all.shape[0] == ns and lmns_all.shape[1] == mnmax:
                lmns_last = lmns_all[-1, :].astype(float)
            elif lmns_all.shape[0] == mnmax and lmns_all.shape[1] == ns:
                lmns_last = lmns_all[:, -1].astype(float)
            else:
                raise ValueError(f"Could not infer lmns orientation: lmns.shape={lmns_all.shape} mnmax={mnmax} ns={ns}")
        if ns is None or mpol is None or ntor is None:
            raise ValueError("VMEC wout file missing required metadata ns/mpol/ntor.")
        return VmecWoutMeta(
            nfp=nfp,
            lasym=lasym,
            ns=int(ns),
            mpol=int(mpol),
            ntor=int(ntor),
            xm=xm,
            xn=xn,
            lmns_last=lmns_last,
        )
    finally:
        ds.close()


def compute_curpol_and_G(obj):
    """Compute VMEC ``curpol`` and ``G`` using the same half-mesh extrapolation used in
    REGCOIL/VMEC.

    This helper is intentionally *robust* because in regcoil_jax we may pass around either:
      - a VMEC wout filename (str),
      - a :class:`VmecBoundary` (from :func:`read_wout_boundary`), or
      - a Fourier-surface-like object (e.g. ``FourierSurface``) that lacks VMEC profile arrays.

    Returns:
      (curpol, G, nfp, lasym)

    If the VMEC profile arrays are unavailable, ``curpol`` and ``G`` are returned as ``None``.
    """
    # If user passed a filename, read boundary first.
    if isinstance(obj, str):
        boundary = read_wout_boundary(obj)
        return compute_curpol_and_G(boundary)

    # Duck-typed access to nfp/lasym is required for any surface object.
    if not hasattr(obj, "nfp") or not hasattr(obj, "lasym"):
        raise TypeError("compute_curpol_and_G expects a wout filename, VmecBoundary, or surface-like object with nfp/lasym")

    nfp = int(getattr(obj, "nfp"))
    lasym = bool(getattr(obj, "lasym"))

    # VMEC-specific arrays (may be missing for FourierSurface-only objects).
    bvco = getattr(obj, "bvco", None)
    bsubvmnc = getattr(obj, "bsubvmnc", None)
    ns = getattr(obj, "ns", None)
    xm = getattr(obj, "xm", None)

    if bvco is None or bsubvmnc is None or ns is None or xm is None:
        return None, None, nfp, lasym

    ns = int(ns)
    if ns < 2:
        return None, None, nfp, lasym

    mu0 = 4e-7 * np.pi

    # bvco is (ns,)
    G = 2 * np.pi / mu0 * (1.5 * bvco[ns - 1] - 0.5 * bvco[ns - 2])

    b = np.asarray(bsubvmnc)
    mnmax = int(np.asarray(xm).size)
    if b.ndim != 2:
        return None, None, nfp, lasym

    # bsubvmnc can be oriented either (mnmax, ns) or (ns, mnmax)
    if b.shape[0] == mnmax:
        b00 = (1.5 * b[0, ns - 1] - 0.5 * b[0, ns - 2])
    elif b.shape[1] == mnmax:
        b00 = (1.5 * b[ns - 1, 0] - 0.5 * b[ns - 2, 0])
    else:
        return None, None, nfp, lasym

    curpol = (2 * np.pi / nfp) * b00
    return float(curpol), float(G), nfp, lasym
