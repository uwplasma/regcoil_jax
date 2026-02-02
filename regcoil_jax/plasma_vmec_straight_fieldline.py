from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .modes import init_fourier_modes
from .io_vmec import read_wout_boundary, read_wout_meta


@dataclass(frozen=True)
class VmecStraightFieldlineSurface:
    """Fourier representation of the plasma surface in VMEC straight-field-line poloidal coordinate.

    This is the REGCOIL geometry_option_plasma=4 preprocessing step:
      1) Read VMEC outer surface and lambda spectrum.
      2) Invert u_new = u_old + lambda(u_old, zeta) for each grid point.
      3) Evaluate R,Z on a high-res grid in (u_new,zeta).
      4) Discrete Fourier transform to obtain a new Fourier surface.

    The result is suitable to be wrapped as a :class:`regcoil_jax.geometry_fourier.FourierSurface`.
    """

    nfp: int
    lasym: bool
    xm: np.ndarray
    xn: np.ndarray
    rmnc: np.ndarray
    zmns: np.ndarray
    rmns: np.ndarray
    zmnc: np.ndarray


def _bracketed_bisect(
    f,
    a: float,
    b: float,
    *,
    rtol: float,
    atol: float,
    maxiter: int,
) -> float:
    fa = float(f(a))
    fb = float(f(b))
    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)
    if fa * fb > 0.0:
        raise ValueError("No sign change in residual on bracket.")
    left = float(a)
    right = float(b)
    f_left = fa
    f_right = fb
    for _ in range(int(maxiter)):
        mid = 0.5 * (left + right)
        f_mid = float(f(mid))
        if f_mid == 0.0:
            return float(mid)
        # Termination: small interval in absolute or relative size.
        if abs(right - left) <= max(atol, rtol * max(abs(left), abs(right), 1.0)):
            return float(mid)
        if f_left * f_mid < 0.0:
            right = mid
            f_right = f_mid
        else:
            left = mid
            f_left = f_mid
    return float(0.5 * (left + right))


def build_vmec_straight_fieldline_plasma_surface(
    *,
    wout_filename: str,
    mpol_transform_refinement: float = 5.0,
    ntor_transform_refinement: float = 1.0,
    root_abserr: float = 1.0e-10,
    root_relerr: float = 1.0e-10,
    max_root_iter: int = 80,
) -> VmecStraightFieldlineSurface:
    """VMEC straight-field-line poloidal coordinate transform.

    Important:
      - The version of REGCOIL in this workspace contains a known inconsistency in
        its geometry_option_plasma=4 implementation (angles/units), which can cause
        failures for typical VMEC files.
      - This JAX port implements the *intended* algorithm in consistent radians:
          theta_new = theta_old + lambda(theta_old, zeta)
        where lambda is VMEC's Fourier series with coefficients ``lmns`` from the wout file.

    The returned surface is a reparameterization of the same physical boundary.
    """
    meta = read_wout_meta(wout_filename, need_lmns_last=True)
    if bool(meta.lasym):
        raise ValueError("geometry_option_plasma=4 is not implemented for lasym=true (matches REGCOIL).")

    boundary = read_wout_boundary(wout_filename, radial_mode="half")

    xm_vmec = np.asarray(meta.xm, dtype=np.int32)
    xn_vmec = np.asarray(meta.xn, dtype=np.int32)
    rmnc_last = np.asarray(boundary.rmnc, dtype=np.float64)
    zmns_last = np.asarray(boundary.zmns, dtype=np.float64)
    lmns_last = np.asarray(meta.lmns_last, dtype=np.float64)

    # Choose spectral resolution for the transformed representation.
    mpol_new = int(int(meta.mpol) * float(mpol_transform_refinement))
    ntor_new = int(int(meta.ntor) * float(ntor_transform_refinement))
    if mpol_new < 1 or ntor_new < 1:
        raise ValueError(f"Invalid transformed resolution: mpol={mpol_new} ntor={ntor_new}")

    ntheta_ct = int(2 * mpol_new)
    nzeta_ct = int(2 * ntor_new)

    twopi = 2.0 * np.pi
    # Use one field period in zeta (consistent with REGCOIL's surface evaluation conventions).
    theta_new = twopi * (np.arange(ntheta_ct, dtype=np.float64) / float(ntheta_ct))  # [0,2π)
    zeta = (twopi / float(meta.nfp)) * (np.arange(nzeta_ct, dtype=np.float64) / float(nzeta_ct))  # [0,2π/nfp)

    r_ct = np.zeros((ntheta_ct, nzeta_ct), dtype=np.float64)
    z_ct = np.zeros((ntheta_ct, nzeta_ct), dtype=np.float64)

    # Root solve for each (theta_new, zeta) to find theta_old, then evaluate R,Z.
    for iz in range(nzeta_ct):
        zeta_iz = float(zeta[iz])
        # Precompute the toroidal phase contribution for this zeta.
        xn_z = xn_vmec.astype(np.float64) * zeta_iz

        def residual(theta_old: float, target: float) -> float:
            ang = xm_vmec.astype(np.float64) * theta_old - xn_z
            lam = np.sum(lmns_last * np.sin(ang))
            return float(theta_old + lam - target)

        for it in range(ntheta_ct):
            target = float(theta_new[it])
            # REGCOIL uses a fixed ±0.3 bracket (in its internal units). In practice,
            # VMEC lambda can require a wider bracket, so we expand symmetrically until
            # a sign change is found.
            f = lambda th: residual(th, target)
            width = 0.3
            a = target - width
            b = target + width
            for _ in range(12):
                try:
                    theta_old = _bracketed_bisect(
                        f, a, b, rtol=root_relerr, atol=root_abserr, maxiter=max_root_iter
                    )
                    break
                except ValueError:
                    width = min(width * 2.0, np.pi)
                    a = target - width
                    b = target + width
            else:
                raise ValueError("Failed to bracket theta_old root for straight-field-line transform.")

            ang = xm_vmec.astype(np.float64) * theta_old - xn_z
            r_ct[it, iz] = float(np.sum(rmnc_last * np.cos(ang)))
            z_ct[it, iz] = float(np.sum(zmns_last * np.sin(ang)))

    # Build the new Fourier modes.
    xm_new, xn_new = init_fourier_modes(mpol_new, ntor_new, include_00=True)
    xn_new = (xn_new.astype(np.int32) * int(meta.nfp)).astype(np.int32)  # VMEC convention: n includes nfp.

    # Discrete Fourier transform from r_ct/z_ct on the transformed grid.
    rmnc_new = np.zeros((xm_new.size,), dtype=np.float64)
    zmns_new = np.zeros((xm_new.size,), dtype=np.float64)

    th_grid = theta_new[:, None]  # (T,1)
    ze_grid = zeta[None, :]  # (1,Z)

    norm0 = 1.0 / float(ntheta_ct * nzeta_ct)
    for k in range(xm_new.size):
        m = float(xm_new[k])
        n = float(xn_new[k])
        ang = m * th_grid - n * ze_grid
        dnorm = norm0 if (m == 0.0 and n == 0.0) else (2.0 * norm0)
        rmnc_new[k] = float(dnorm * np.sum(r_ct * np.cos(ang)))
        zmns_new[k] = float(dnorm * np.sum(z_ct * np.sin(ang)))

    return VmecStraightFieldlineSurface(
        nfp=int(meta.nfp),
        lasym=False,
        xm=xm_new.astype(np.int32),
        xn=xn_new.astype(np.int32),
        rmnc=rmnc_new,
        zmns=zmns_new,
        rmns=np.zeros_like(rmnc_new),
        zmnc=np.zeros_like(rmnc_new),
    )
