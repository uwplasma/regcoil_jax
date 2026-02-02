from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import jax
import jax.numpy as jnp

from .constants import twopi
from .geometry_fourier import FourierSurface, eval_surface_xyz_and_derivs2
from .surface_metrics import metrics_and_normals
from .build_matrices_jax import build_matrices
from .solve_jax import solve_for_lambdas, diagnostics
from .modes import init_fourier_modes


@dataclass(frozen=True)
class SensitivityModeLists:
    """Mode lists for REGCOIL's winding-surface sensitivity parameterization."""

    mnmax_sensitivity: int
    nomega_coil: int
    xm_sensitivity: np.ndarray  # (nomega,)
    xn_sensitivity: np.ndarray  # (nomega,) unscaled by nfp, matching Fortran output
    omega_coil: np.ndarray  # (nomega,) codes: 1=rmnc, 2=zmns, 3=rmns, 4=zmnc


def init_fourier_modes_sensitivity(*, mpol: int, ntor: int, sensitivity_symmetry_option: int) -> SensitivityModeLists:
    """Match regcoil_init_Fourier_modes_sensitivity() in regcoil_init_Fourier_modes_mod.f90."""
    mpol = int(mpol)
    ntor = int(ntor)
    opt = int(sensitivity_symmetry_option)

    # mnmax includes (0,0) mode unconditionally for sensitivity.
    mnmax = mpol * (2 * ntor + 1) + ntor + 1

    if opt == 1:
        nomega = mnmax * 2
        min_sym = 1
        max_sym = 2
    elif opt == 2:
        nomega = mnmax * 2
        min_sym = 3
        max_sym = 4
    elif opt == 3:
        nomega = mnmax * 4
        min_sym = 1
        max_sym = 4
    else:
        raise ValueError(f"Unsupported sensitivity_symmetry_option={opt}. Expected 1, 2, or 3.")

    xm = np.zeros((nomega,), dtype=np.int32)
    xn = np.zeros((nomega,), dtype=np.int32)
    omega = np.zeros((nomega,), dtype=np.int32)

    iomega = 0
    # m=0, n=0..ntor
    for jn in range(0, ntor + 1):
        for i in range(min_sym, max_sym + 1):
            xm[iomega] = 0
            xn[iomega] = jn
            omega[iomega] = i
            iomega += 1

    # m=1..mpol, n=-ntor..ntor
    for jm in range(1, mpol + 1):
        for jn in range(-ntor, ntor + 1):
            for i in range(min_sym, max_sym + 1):
                xm[iomega] = jm
                xn[iomega] = jn
                omega[iomega] = i
                iomega += 1

    if iomega != nomega:
        raise RuntimeError(f"init_fourier_modes_sensitivity internal error: iomega={iomega} nomega={nomega}")

    return SensitivityModeLists(
        mnmax_sensitivity=int(mnmax),
        nomega_coil=int(nomega),
        xm_sensitivity=xm,
        xn_sensitivity=xn,
        omega_coil=omega,
    )


def _to_3TZ(xyz_tz3: jnp.ndarray) -> jnp.ndarray:
    # (T,Z,3) -> (3,T,Z)
    return jnp.moveaxis(xyz_tz3, -1, 0)


def _build_coil_surface_from_coeffs(
    *,
    nfp: int,
    lasym: bool,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    xm: jnp.ndarray,
    xn: jnp.ndarray,
    rmnc: jnp.ndarray,
    zmns: jnp.ndarray,
    rmns: jnp.ndarray,
    zmnc: jnp.ndarray,
) -> dict:
    surf = FourierSurface(
        nfp=int(nfp),
        lasym=bool(lasym),
        xm=jnp.asarray(xm, dtype=jnp.int32),
        xn=jnp.asarray(xn, dtype=jnp.int32),
        rmnc=jnp.asarray(rmnc, dtype=jnp.float64),
        zmns=jnp.asarray(zmns, dtype=jnp.float64),
        rmns=jnp.asarray(rmns, dtype=jnp.float64),
        zmnc=jnp.asarray(zmnc, dtype=jnp.float64),
    )
    xyz, dth, dze, d2th2, d2thze, d2ze2 = eval_surface_xyz_and_derivs2(surf, theta, zeta)
    r = _to_3TZ(xyz)
    rth = _to_3TZ(dth)
    rze = _to_3TZ(dze)
    rtt = _to_3TZ(d2th2)
    rtz = _to_3TZ(d2thze)
    rzz = _to_3TZ(d2ze2)
    _, _, _, nunit, normN = metrics_and_normals(rth, rze)
    return dict(
        theta=theta,
        zeta=zeta,
        r=r,
        rth=rth,
        rze=rze,
        rtt=rtt,
        rtz=rtz,
        rzz=rzz,
        nunit=nunit,
        normN=normN,
        lasym=bool(lasym),
        xm_coil=surf.xm,
        xn_coil=surf.xn,
        rmnc_coil=surf.rmnc,
        zmns_coil=surf.zmns,
        rmns_coil=surf.rmns,
        zmnc_coil=surf.zmnc,
    )


def _volume_r2_dz(r3tz: jnp.ndarray, *, dze: float, nfp: int) -> jnp.ndarray:
    r3tz = jnp.asarray(r3tz)  # (3,ntheta,nzeta) for a single field period in zeta
    R2 = r3tz[0] * r3tz[0] + r3tz[1] * r3tz[1]  # (T,Z)
    Z = r3tz[2]  # (T,Z)
    R2_half = 0.5 * (R2[:-1, :] + R2[1:, :])
    dZ = Z[1:, :] - Z[:-1, :]
    acc = jnp.sum(R2_half * dZ)
    acc = acc + jnp.sum(0.5 * (R2[0, :] + R2[-1, :]) * (Z[0, :] - Z[-1, :]))
    return jnp.abs((nfp * acc) * jnp.asarray(dze, dtype=r3tz.dtype) / 2.0)


def _coil_plasma_distance_metrics(
    *,
    r_coil_3tz: jnp.ndarray,
    normN_coil_tz: jnp.ndarray,
    r_plasma_3tz: jnp.ndarray,
    normN_plasma_tz: jnp.ndarray,
    p_lse: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (dist_min, dist_max, dist_min_lse, dist_max_lse) for one field period.

    Matches regcoil_init_sensitivity.f90 weight convention:
      weights = |N_plasma| * |N_coil|
    and uses stop_gradient on the hard min/max shift for stable derivatives.
    """
    coil_xyz = jnp.moveaxis(r_coil_3tz, 0, -1).reshape((-1, 3))  # (Nc,3)
    plasma_xyz = jnp.moveaxis(r_plasma_3tz, 0, -1).reshape((-1, 3))  # (Np,3)
    w_coil = jnp.reshape(normN_coil_tz, (-1,))
    w_plasma = jnp.reshape(normN_plasma_tz, (-1,))
    w = w_coil[:, None] * w_plasma[None, :]

    diff = coil_xyz[:, None, :] - plasma_xyz[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-300)  # (Nc,Np)

    dist_min = jnp.min(dist)
    dist_max = jnp.max(dist)

    p = jnp.asarray(p_lse, dtype=dist.dtype)
    sum_w = jnp.sum(w) + 1e-300

    shift_min = jax.lax.stop_gradient(dist_min)
    sum_exp_min = jnp.sum(w * jnp.exp(-p * (dist - shift_min)))
    dist_min_lse = -(jnp.log(sum_exp_min / sum_w) / p) + shift_min

    shift_max = jax.lax.stop_gradient(dist_max)
    sum_exp_max = jnp.sum(w * jnp.exp(p * (dist - shift_max)))
    dist_max_lse = (jnp.log(sum_exp_max / sum_w) / p) + shift_max

    return dist_min, dist_max, dist_min_lse, dist_max_lse


def compute_sensitivity_outputs(
    *,
    inputs: dict,
    plasma: dict,
    coil: dict,
    lambdas: jnp.ndarray,
    exit_code: int,
) -> dict:
    """Compute sensitivity-related outputs (and optionally dchi2/domega) using autodiff.

    This function is intentionally used only when sensitivity_option>1, so adding the
    extra work and schema does not affect strict parity-tested runs with sensitivity_option=1.
    """
    sensitivity_option = int(inputs.get("sensitivity_option", 1))
    if sensitivity_option <= 1:
        return {}

    nfp = int(plasma.get("nfp", 1))
    mmax = int(inputs.get("mmax_sensitivity", 0))
    nmax = int(inputs.get("nmax_sensitivity", 0))
    symopt = int(inputs.get("sensitivity_symmetry_option", 1))
    fixed_norm = bool(inputs.get("fixed_norm_sensitivity_option", False))
    p_lse = float(inputs.get("coil_plasma_dist_lse_p", 1.0e4))

    mode_lists = init_fourier_modes_sensitivity(mpol=mmax, ntor=nmax, sensitivity_symmetry_option=symopt)

    if mmax < 1:
        raise ValueError("mmax_sensitivity must be >= 1.")
    if nmax < 1:
        raise ValueError("nmax_sensitivity must be >= 1.")

    # Build a "design" Fourier representation that includes:
    #  - all modes already present in the coil surface
    #  - plus any sensitivity modes up to (mmax_sensitivity, nmax_sensitivity), even if the base surface has 0 coeffs there
    #
    # This matches the Fortran sensitivity workflow, in which omega parameters can include
    # modes that are not present in the base surface coefficients.
    base_xm = np.asarray(coil["xm_coil"], dtype=np.int32)
    base_xn = np.asarray(coil["xn_coil"], dtype=np.int32)
    rmnc_base_np = np.asarray(coil["rmnc_coil"], dtype=float)
    zmns_base_np = np.asarray(coil["zmns_coil"], dtype=float)
    rmns_base_np = np.asarray(coil["rmns_coil"], dtype=float)
    zmnc_base_np = np.asarray(coil["zmnc_coil"], dtype=float)

    sens_xm_u, sens_xn_u = init_fourier_modes(int(mmax), int(nmax), include_00=True)
    sens_xm_u = sens_xm_u.astype(np.int32)
    sens_xn_u = (sens_xn_u.astype(np.int32) * int(nfp)).astype(np.int32)

    seen = {(int(m), int(n)) for m, n in zip(base_xm, base_xn)}
    union_modes = [(int(m), int(n)) for m, n in zip(base_xm, base_xn)]
    for m, n in zip(sens_xm_u, sens_xn_u):
        key = (int(m), int(n))
        if key not in seen:
            seen.add(key)
            union_modes.append(key)

    xm_u = np.asarray([m for (m, _) in union_modes], dtype=np.int32)
    xn_u = np.asarray([n for (_, n) in union_modes], dtype=np.int32)
    mode_to_u = {(int(m), int(n)): i for i, (m, n) in enumerate(union_modes)}

    nmode_u = int(len(union_modes))
    rmnc_u = np.zeros((nmode_u,), dtype=float)
    zmns_u = np.zeros((nmode_u,), dtype=float)
    rmns_u = np.zeros((nmode_u,), dtype=float)
    zmnc_u = np.zeros((nmode_u,), dtype=float)
    for j in range(base_xm.size):
        k = (int(base_xm[j]), int(base_xn[j]))
        iu = mode_to_u[k]
        rmnc_u[iu] = rmnc_base_np[j]
        zmns_u[iu] = zmns_base_np[j]
        rmns_u[iu] = rmns_base_np[j]
        zmnc_u[iu] = zmnc_base_np[j]

    # Map each (m,n) sensitivity mode to the index in the union ("design") representation.
    xn_scaled = (mode_lists.xn_sensitivity.astype(np.int32) * int(nfp)).astype(np.int32)
    idx_np = np.empty((mode_lists.nomega_coil,), dtype=np.int32)
    for i in range(mode_lists.nomega_coil):
        key = (int(mode_lists.xm_sensitivity[i]), int(xn_scaled[i]))
        idx_np[i] = mode_to_u[key]

    idx = jnp.asarray(idx_np, dtype=jnp.int32)
    omega_code = jnp.asarray(mode_lists.omega_coil, dtype=jnp.int32)

    rmnc0 = jnp.asarray(rmnc_u, dtype=jnp.float64)
    zmns0 = jnp.asarray(zmns_u, dtype=jnp.float64)
    rmns0 = jnp.asarray(rmns_u, dtype=jnp.float64)
    zmnc0 = jnp.asarray(zmnc_u, dtype=jnp.float64)

    # Baseline omega values: the selected coil Fourier coefficients.
    omega0 = jnp.where(
        omega_code == 1,
        rmnc0[idx],
        jnp.where(
            omega_code == 2,
            zmns0[idx],
            jnp.where(omega_code == 3, rmns0[idx], zmnc0[idx]),
        ),
    )

    # Precompute selection indices for each coefficient type (static, used inside autodiff).
    sel_rmnc = np.where(mode_lists.omega_coil == 1)[0].astype(np.int32)
    sel_zmns = np.where(mode_lists.omega_coil == 2)[0].astype(np.int32)
    sel_rmns = np.where(mode_lists.omega_coil == 3)[0].astype(np.int32)
    sel_zmnc = np.where(mode_lists.omega_coil == 4)[0].astype(np.int32)

    idx_rmnc = idx_np[sel_rmnc].astype(np.int32)
    idx_zmns = idx_np[sel_zmns].astype(np.int32)
    idx_rmns = idx_np[sel_rmns].astype(np.int32)
    idx_zmnc = idx_np[sel_zmnc].astype(np.int32)

    theta = jnp.asarray(coil["theta"])
    zeta = jnp.asarray(coil["zeta"])
    xm = jnp.asarray(xm_u, dtype=jnp.int32)
    xn = jnp.asarray(xn_u, dtype=jnp.int32)
    # Allow asymmetric Fourier coefficients if the sensitivity symmetry option includes them.
    lasym = bool(coil.get("lasym", False) or symopt in (2, 3))

    def _coeffs_from_omega(omega: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        omega = jnp.asarray(omega, dtype=jnp.float64)
        rmnc = rmnc0.at[jnp.asarray(idx_rmnc)].set(omega[jnp.asarray(sel_rmnc)]) if sel_rmnc.size else rmnc0
        zmns = zmns0.at[jnp.asarray(idx_zmns)].set(omega[jnp.asarray(sel_zmns)]) if sel_zmns.size else zmns0
        rmns = rmns0.at[jnp.asarray(idx_rmns)].set(omega[jnp.asarray(sel_rmns)]) if sel_rmns.size else rmns0
        zmnc = zmnc0.at[jnp.asarray(idx_zmnc)].set(omega[jnp.asarray(sel_zmnc)]) if sel_zmnc.size else zmnc0

        return rmnc, zmns, rmns, zmnc

    def _coil_from_omega(omega: jnp.ndarray) -> dict:
        rmnc, zmns, rmns, zmnc = _coeffs_from_omega(omega)
        return _build_coil_surface_from_coeffs(
            nfp=nfp,
            lasym=lasym,
            theta=theta,
            zeta=zeta,
            xm=xm,
            xn=xn,
            rmnc=rmnc,
            zmns=zmns,
            rmns=rmns,
            zmnc=zmnc,
        )

    # Coil-plasma distance diagnostics at the baseline geometry.
    dist_min, dist_max, dist_min_lse, dist_max_lse = _coil_plasma_distance_metrics(
        r_coil_3tz=jnp.asarray(coil["r"]),
        normN_coil_tz=jnp.asarray(coil["normN"]),
        r_plasma_3tz=jnp.asarray(plasma["r"]),
        normN_plasma_tz=jnp.asarray(plasma["normN"]),
        p_lse=p_lse,
    )
    # Match the current Fortran reference implementation in this workspace: it computes only
    # the minimum-distance metrics but still writes the "max" scalars, which remain 0.
    dist_max = jnp.asarray(0.0, dtype=dist_min.dtype)
    dist_max_lse = jnp.asarray(0.0, dtype=dist_min.dtype)

    # Derivatives of geometry-only diagnostics w.r.t. omega.
    ntheta_c = int(inputs.get("ntheta_coil", theta.shape[0]))
    nzeta_c = int(inputs.get("nzeta_coil", zeta.shape[0]))
    dth_c = float(twopi / ntheta_c)
    dze_c = float((twopi / nfp) / nzeta_c)

    def _area_from_omega(omega: jnp.ndarray) -> jnp.ndarray:
        c = _coil_from_omega(omega)
        return jnp.asarray(nfp * dth_c * dze_c, dtype=jnp.float64) * jnp.sum(c["normN"])

    def _volume_from_omega(omega: jnp.ndarray) -> jnp.ndarray:
        c = _coil_from_omega(omega)
        return _volume_r2_dz(c["r"], dze=dze_c, nfp=nfp)

    darea = jax.grad(_area_from_omega)(omega0)
    dvolume = jax.grad(_volume_from_omega)(omega0)

    # Match the Fortran reference workflow for dcoil_plasma_dist_mindomega.
    #
    # In regcoil_init_sensitivity.f90, the "min LSE" value is computed from:
    #   dmin_lse = -(1/p)*log(sum_exp_min / sum_w) + dist_min
    # with sum_w = sum(normN_plasma)*sum(normN_coil) and dist_min used only as a constant shift.
    #
    # However, the derivative that is actually written uses an additional normalization term
    # involving area_coil and sum(normN_plasma). We reproduce that exact expression for strict
    # parity with the reference netCDF output in this workspace.
    def _sum_exp_min_from_omega(omega: jnp.ndarray) -> jnp.ndarray:
        c = _coil_from_omega(omega)
        coil_xyz = jnp.moveaxis(c["r"], 0, -1).reshape((-1, 3))
        plasma_xyz = jnp.moveaxis(jnp.asarray(plasma["r"]), 0, -1).reshape((-1, 3))
        w_coil = jnp.reshape(c["normN"], (-1,))
        w_plasma = jnp.reshape(jnp.asarray(plasma["normN"]), (-1,))
        w = w_coil[:, None] * w_plasma[None, :]
        diff = coil_xyz[:, None, :] - plasma_xyz[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-300)
        dist_min_local = jnp.min(dist)
        shift = jax.lax.stop_gradient(dist_min_local)
        p = jnp.asarray(p_lse, dtype=dist.dtype)
        return jnp.sum(w * jnp.exp(-p * (dist - shift)))

    sum_exp_min0 = _sum_exp_min_from_omega(omega0)
    dsum_exp = jax.grad(_sum_exp_min_from_omega)(omega0)
    p = jnp.asarray(p_lse, dtype=jnp.float64)
    sum_norm_plasma = jnp.sum(jnp.asarray(plasma["normN"])) + 1e-300
    area_coil0 = _area_from_omega(omega0) + 1e-300
    ddist_min = -(dsum_exp / (sum_exp_min0 * p)) + (darea / (sum_norm_plasma * area_coil0 * p))

    out = dict(
        mnmax_sensitivity=int(mode_lists.mnmax_sensitivity),
        nomega_coil=int(mode_lists.nomega_coil),
        xm_sensitivity=jnp.asarray(mode_lists.xm_sensitivity, dtype=jnp.int32),
        xn_sensitivity=jnp.asarray(mode_lists.xn_sensitivity, dtype=jnp.int32),
        omega_coil=jnp.asarray(mode_lists.omega_coil, dtype=jnp.int32),
        sensitivity_symmetry_option=int(symopt),
        fixed_norm_sensitivity_option=int(1 if fixed_norm else 0),
        coil_plasma_dist_lse_p=float(p_lse),
        coil_plasma_dist_min=dist_min,
        coil_plasma_dist_max=dist_max,
        coil_plasma_dist_min_lse=dist_min_lse,
        coil_plasma_dist_max_lse=dist_max_lse,
        darea_coildomega=darea,
        dvolume_coildomega=dvolume,
        dcoil_plasma_dist_mindomega=ddist_min,
    )

    if int(exit_code) != 0:
        return out

    lambdas = jnp.asarray(lambdas, dtype=jnp.float64)

    def _chi2_BK_from_omega(omega: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        c = _coil_from_omega(omega)
        mats = build_matrices(inputs, plasma, c)
        sols = solve_for_lambdas(mats, lambdas)
        chi2_B, chi2_K, _, _ = diagnostics(mats, sols)
        area = mats["area_coil"]
        return chi2_B, chi2_K, area

    def _chi2_B_vec(omega: jnp.ndarray) -> jnp.ndarray:
        chi2_B, _, _ = _chi2_BK_from_omega(omega)
        return chi2_B

    def _chi2_K_vec(omega: jnp.ndarray) -> jnp.ndarray:
        _, chi2_K, _ = _chi2_BK_from_omega(omega)
        return chi2_K

    # Jacobians come back with shape (nlambda, nomega). REGCOIL writes (nomega, nlambda).
    dchi2B = jax.jacrev(_chi2_B_vec)(omega0).T
    dchi2K = jax.jacrev(_chi2_K_vec)(omega0).T
    dchi2 = dchi2B + (lambdas[None, :] * dchi2K)
    out["dchi2domega"] = dchi2

    if sensitivity_option > 2:
        out["dchi2Bdomega"] = dchi2B
        out["dchi2Kdomega"] = dchi2K

        def _RMSK_vec(omega: jnp.ndarray) -> jnp.ndarray:
            _, chi2_K, area = _chi2_BK_from_omega(omega)
            return jnp.sqrt(chi2_K / (area + 1e-300))

        out["dRMSKdomega"] = jax.jacrev(_RMSK_vec)(omega0).T

    return out
