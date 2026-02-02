from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .geometry_fourier import FourierSurface
from .io_vmec import read_wout_boundary
from .io_nescin import NescinCurrentSurface, write_nescin_current_surface
from .surfaces import plasma_surface_from_inputs, coil_surface_from_inputs
from .build_matrices_jax import build_matrices
from .solve_jax import solve_one_lambda, diagnostics
from .offset_surface import offset_surface_point
from .geometry_fourier import eval_surface_xyz_and_derivs2
from .surface_metrics import metrics_and_normals
from .modes import init_fourier_modes


@dataclass(frozen=True)
class SeparationOptResult:
    separation_history: jnp.ndarray  # (nsteps+1,)
    objective_history: jnp.ndarray   # (nsteps+1,)


def _as_fourier_surface(bound) -> FourierSurface:
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


def optimize_vmec_offset_separation(
    *,
    wout_filename: str,
    separation0: float,
    nsteps: int = 20,
    step_size: float = 0.05,
    # Small grids by default to keep this runnable on CPU in a reasonable time.
    ntheta: int = 16,
    nzeta: int = 16,
    mpol_potential: int = 6,
    ntor_potential: int = 6,
    lam: float = 1.0e-14,
    objective: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
) -> SeparationOptResult:
    """Autodiff-based optimization of the VMEC offset-surface separation parameter.

    This is a minimal “winding surface optimization” example that stays within JAX:
    it differentiates through surface construction, matrix build, and the linear solve.

    The objective defaults to chi2_B(lam) + 1e-20 * chi2_K(lam) to keep the problem
    well-scaled without dominating the physics target.
    """
    if objective is None:
        def objective(chi2_B, chi2_K, max_B, max_K):
            return chi2_B + (1.0e-20 * chi2_K)

    bound = read_wout_boundary(wout_filename, radial_mode="full")
    vmec = _as_fourier_surface(bound)

    base_inputs = dict(
        # Geometry / grids:
        geometry_option_plasma=2,
        geometry_option_coil=2,
        wout_filename=wout_filename,
        ntheta_plasma=int(ntheta),
        nzeta_plasma=int(nzeta),
        ntheta_coil=int(ntheta),
        nzeta_coil=int(nzeta),
        # Potential basis:
        mpol_potential=int(mpol_potential),
        ntor_potential=int(ntor_potential),
        symmetry_option=1,
        # Regularization:
        regularization_term_option="chi2_K",
        # No extra target field:
        load_bnorm=False,
        # Currents are taken from the input (not VMEC) here; this is just a demo.
        net_poloidal_current_amperes=1.0,
        net_toroidal_current_amperes=0.0,
        curpol=1.0,
        save_level=3,
    )

    plasma = plasma_surface_from_inputs(base_inputs, vmec)

    def run_one(separation: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        inputs = dict(base_inputs)
        inputs["separation"] = separation
        coil = coil_surface_from_inputs(inputs, plasma, vmec)
        mats = build_matrices(inputs, plasma, coil)
        sol = solve_one_lambda(mats, jnp.asarray(lam, dtype=jnp.float64))
        chi2_B, chi2_K, max_B, max_K = diagnostics(mats, sol[None, :])
        return chi2_B[0], chi2_K[0], max_B[0], max_K[0]

    def obj_from_separation(separation_unconstrained: jnp.ndarray) -> jnp.ndarray:
        separation = jax.nn.softplus(separation_unconstrained)
        chi2_B, chi2_K, max_B, max_K = run_one(separation)
        return objective(chi2_B, chi2_K, max_B, max_K)

    vng = jax.value_and_grad(obj_from_separation)

    sep0 = jnp.asarray(separation0, dtype=jnp.float64)
    # Softplus parameterization to enforce separation > 0. Interpret separation0 as
    # the physical initial value and map to an unconstrained raw parameter.
    s_raw = jnp.log(jnp.expm1(sep0) + 1e-300)

    sep_hist = [sep0]
    obj_hist = [obj_from_separation(s_raw)]

    for _ in range(int(nsteps)):
        _, grad = vng(s_raw)
        # Simple gradient descent; callers can wrap with more sophisticated optimizers if desired.
        s_raw = s_raw - (jnp.asarray(step_size, dtype=jnp.float64) * grad)
        sep_hist.append(jax.nn.softplus(s_raw))
        obj_hist.append(obj_from_separation(s_raw))

    return SeparationOptResult(
        separation_history=jnp.stack(sep_hist, axis=0),
        objective_history=jnp.stack(obj_hist, axis=0),
    )


@dataclass(frozen=True)
class SeparationFieldOptConfig:
    """Configuration for optimizing a spatially-varying offset separation field.

    The separation field is parameterized as:
      sep(θ,ζ) = softplus( b + Σ a_sin*sin(mθ - nζ) + a_cos*cos(mθ - nζ) )
    where (m,n) are the mode list defined by (mpol_sep, ntor_sep).
    """

    mpol_sep: int = 3
    ntor_sep: int = 3
    separation_min: float = 0.05
    smooth_weight: float = 1.0e-2
    mean_weight: float = 1.0e-1
    mean_target: float | None = None


@dataclass(frozen=True)
class SeparationFieldOptResult:
    base_history: jnp.ndarray  # (nsteps+1,)
    coeff_sin_history: jnp.ndarray  # (nsteps+1,nmodes)
    coeff_cos_history: jnp.ndarray  # (nsteps+1,nmodes)
    objective_history: jnp.ndarray  # (nsteps+1,)


def _init_mode_list(*, mpol: int, ntor: int, nfp: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    xm_np, xn_np = init_fourier_modes(int(mpol), int(ntor), include_00=False)
    xn_np = xn_np * int(nfp)
    return jnp.asarray(xm_np, dtype=jnp.int32), jnp.asarray(xn_np, dtype=jnp.int32)


def separation_field_from_modes(
    *,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    xm: jnp.ndarray,
    xn: jnp.ndarray,
    base: jnp.ndarray,
    coeff_sin: jnp.ndarray,
    coeff_cos: jnp.ndarray,
    separation_min: float,
) -> jnp.ndarray:
    """Evaluate sep(θ,ζ) on a (theta,zeta) grid, returning shape (ntheta,nzeta)."""
    th = theta[:, None, None]
    ze = zeta[None, :, None]
    angle = xm[None, None, :] * th - xn[None, None, :] * ze
    raw = base + jnp.sum(coeff_sin[None, None, :] * jnp.sin(angle) + coeff_cos[None, None, :] * jnp.cos(angle), axis=-1)
    # Enforce sep >= separation_min smoothly.
    return jnp.asarray(separation_min, dtype=raw.dtype) + jax.nn.softplus(raw)


def _finite_diff_smoothness(sep_tz: jnp.ndarray) -> jnp.ndarray:
    """Periodic smoothness penalty using centered finite differences."""
    dth = 0.5 * (jnp.roll(sep_tz, -1, axis=0) - jnp.roll(sep_tz, 1, axis=0))
    dze = 0.5 * (jnp.roll(sep_tz, -1, axis=1) - jnp.roll(sep_tz, 1, axis=1))
    return jnp.mean(dth * dth + dze * dze)


def coil_surface_from_vmec_offset_separation_field(
    *,
    vmec_surface: FourierSurface,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    separation_tz: jnp.ndarray,  # (ntheta,nzeta)
    max_mpol_coil: int = 24,
    max_ntor_coil: int = 24,
    mpol_coil_filter: int | None = None,
    ntor_coil_filter: int | None = None,
    offset_newton_iters: int = 12,
) -> tuple[FourierSurface, dict]:
    """Build a coil winding surface by offsetting a VMEC boundary with a spatially varying separation.

    Returns:
      - FourierSurface describing the fitted winding surface
      - coil surface dict compatible with build_matrices()
    """
    nfp = int(vmec_surface.nfp)
    ntheta = int(theta.shape[0])
    nzeta = int(zeta.shape[0])

    if separation_tz.shape != (ntheta, nzeta):
        raise ValueError(f"separation_tz must have shape (ntheta,nzeta)=({ntheta},{nzeta}), got {tuple(separation_tz.shape)}")

    # Offset points on a uniform grid.
    def one_point(th_i, ze_j, sep_ij):
        return offset_surface_point(vmec_surface, th_i, ze_j, sep_ij, iters=int(offset_newton_iters))

    v_zeta = jax.vmap(lambda ze_j, sep_ij, th_i: one_point(th_i, ze_j, sep_ij), in_axes=(0, 0, None))
    v_theta = jax.vmap(lambda th_i, sep_row: v_zeta(zeta, sep_row, th_i), in_axes=(0, 0))
    xyz_off_tz3 = v_theta(theta, separation_tz)  # (T,Z,3)

    major_R = jnp.sqrt(xyz_off_tz3[..., 0] * xyz_off_tz3[..., 0] + xyz_off_tz3[..., 1] * xyz_off_tz3[..., 1])
    z_coord = xyz_off_tz3[..., 2]

    mpol_coil = min(ntheta // 2, int(max_mpol_coil))
    ntor_coil = min(nzeta // 2, int(max_ntor_coil))
    xm_np, xn_np = init_fourier_modes(mpol_coil, ntor_coil, include_00=True)
    xn_np = xn_np * nfp
    xm = jnp.asarray(xm_np, dtype=jnp.int32)
    xn = jnp.asarray(xn_np, dtype=jnp.int32)

    # Discrete Fourier transform normalization (matches surfaces.py geometry_option_coil=2).
    factor = 2.0 / (ntheta * nzeta)
    factor2 = jnp.full((xm.shape[0],), factor, dtype=major_R.dtype)
    if (ntheta % 2) == 0:
        factor2 = jnp.where(xm == (ntheta // 2), factor2 / 2.0, factor2)
    if (nzeta % 2) == 0:
        factor2 = jnp.where(jnp.abs(xn) == (nfp * (nzeta // 2)), factor2 / 2.0, factor2)

    ang = theta[:, None, None] * xm[None, None, :] - zeta[None, :, None] * xn[None, None, :]
    cosang = jnp.cos(ang)
    sinang = jnp.sin(ang)

    rmnc = jnp.sum(major_R[:, :, None] * cosang, axis=(0, 1)) * factor2
    rmns = jnp.sum(major_R[:, :, None] * sinang, axis=(0, 1)) * factor2
    zmnc = jnp.sum(z_coord[:, :, None] * cosang, axis=(0, 1)) * factor2
    zmns = jnp.sum(z_coord[:, :, None] * sinang, axis=(0, 1)) * factor2

    # Special-case (m,n)=(0,0) to match REGCOIL's normalization.
    rmnc = rmnc.at[0].set(jnp.mean(major_R))
    zmnc = zmnc.at[0].set(jnp.mean(z_coord))

    lasym = bool(getattr(vmec_surface, "lasym", False))
    if not lasym:
        rmns = jnp.zeros_like(rmns)
        zmnc = jnp.zeros_like(zmnc)

    mpol_coil_filter = int(mpol_coil_filter) if mpol_coil_filter is not None else int(max_mpol_coil)
    ntor_coil_filter = int(ntor_coil_filter) if ntor_coil_filter is not None else int(max_ntor_coil)
    keep = (jnp.abs(xm) <= mpol_coil_filter) & (jnp.abs(xn) <= (ntor_coil_filter * nfp))
    rmnc = jnp.where(keep, rmnc, 0.0)
    rmns = jnp.where(keep, rmns, 0.0)
    zmnc = jnp.where(keep, zmnc, 0.0)
    zmns = jnp.where(keep, zmns, 0.0)

    coil_surf = FourierSurface(nfp=nfp, lasym=lasym, xm=xm, xn=xn, rmnc=rmnc, zmns=zmns, rmns=rmns, zmnc=zmnc)
    xyz, dth_xyz, dze_xyz, d2th2, d2thze, d2ze2 = eval_surface_xyz_and_derivs2(coil_surf, theta, zeta)
    r = jnp.moveaxis(xyz, -1, 0)
    rth = jnp.moveaxis(dth_xyz, -1, 0)
    rze = jnp.moveaxis(dze_xyz, -1, 0)
    rtt = jnp.moveaxis(d2th2, -1, 0)
    rtz = jnp.moveaxis(d2thze, -1, 0)
    rzz = jnp.moveaxis(d2ze2, -1, 0)
    _, _, _, nunit, normN = metrics_and_normals(rth, rze)

    coil = dict(
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
        xm_coil=xm,
        xn_coil=xn,
        rmnc_coil=rmnc,
        zmns_coil=zmns,
        rmns_coil=rmns,
        zmnc_coil=zmnc,
    )
    return coil_surf, coil


def nescin_current_surface_from_regcoil_fourier_surface(*, coil_surf: FourierSurface) -> NescinCurrentSurface:
    """Convert an internal FourierSurface (REGCOIL convention) to NESCOIL nescin convention.

    REGCOIL uses angle = m*theta - (n*nfp)*zeta.
    NESCOIL nescin 'Current Surface' uses xn without nfp and opposite sign, and REGCOIL reads via:
      xn_regcoil = -nfp * xn_nescin.
    """
    nfp = int(coil_surf.nfp)
    xm = jnp.asarray(coil_surf.xm).astype(int)
    xn = jnp.asarray(coil_surf.xn).astype(int)
    xn_np = np.asarray(xn)
    if np.any(xn_np % nfp != 0):
        raise ValueError("coil_surf.xn must be divisible by nfp to write a nescin file.")
    xn_file = (-xn_np // nfp).astype(int)
    return NescinCurrentSurface(
        xm=np.asarray(xm, dtype=int),
        xn=xn_file,
        rmnc=np.asarray(coil_surf.rmnc, dtype=float),
        zmns=np.asarray(coil_surf.zmns, dtype=float),
        rmns=np.asarray(coil_surf.rmns, dtype=float),
        zmnc=np.asarray(coil_surf.zmnc, dtype=float),
    )


def optimize_vmec_offset_separation_field(
    *,
    wout_filename: str,
    separation0: float = 0.5,
    nsteps: int = 30,
    step_size: float = 0.05,
    ntheta: int = 16,
    nzeta: int = 16,
    mpol_potential: int = 6,
    ntor_potential: int = 6,
    lam: float = 1.0e-14,
    chi2K_weight: float = 1.0e-20,
    config: SeparationFieldOptConfig | None = None,
) -> SeparationFieldOptResult:
    """Optimize a spatially-varying separation field for a VMEC offset winding surface.

    This function is intended for *autodiff-based winding-surface optimization*:
    the optimized parameters can be used to generate a coil surface (and optionally a nescin file)
    that improves the Bnormal/current tradeoff at a chosen lambda.
    """
    if config is None:
        config = SeparationFieldOptConfig()

    bound = read_wout_boundary(wout_filename, radial_mode="full")
    vmec = _as_fourier_surface(bound)

    base_inputs = dict(
        geometry_option_plasma=2,
        wout_filename=wout_filename,
        ntheta_plasma=int(ntheta),
        nzeta_plasma=int(nzeta),
        ntheta_coil=int(ntheta),
        nzeta_coil=int(nzeta),
        mpol_potential=int(mpol_potential),
        ntor_potential=int(ntor_potential),
        symmetry_option=1,
        regularization_term_option="chi2_K",
        load_bnorm=False,
        net_poloidal_current_amperes=1.0,
        net_toroidal_current_amperes=0.0,
        curpol=1.0,
        save_level=3,
    )

    plasma = plasma_surface_from_inputs(base_inputs, vmec)
    theta = plasma["theta"]
    zeta = plasma["zeta"]
    xm_sep, xn_sep = _init_mode_list(mpol=int(config.mpol_sep), ntor=int(config.ntor_sep), nfp=int(vmec.nfp))
    nmodes = int(xm_sep.shape[0])

    mean_target = float(config.mean_target) if config.mean_target is not None else float(separation0)

    def objective_from_params(params: jnp.ndarray) -> jnp.ndarray:
        base = params[0]
        coeff_sin = params[1 : 1 + nmodes]
        coeff_cos = params[1 + nmodes : 1 + 2 * nmodes]
        sep = separation_field_from_modes(
            theta=theta,
            zeta=zeta,
            xm=xm_sep,
            xn=xn_sep,
            base=base,
            coeff_sin=coeff_sin,
            coeff_cos=coeff_cos,
            separation_min=float(config.separation_min),
        )

        smooth = _finite_diff_smoothness(sep)
        mean_pen = (jnp.mean(sep) - mean_target) ** 2

        _, coil = coil_surface_from_vmec_offset_separation_field(
            vmec_surface=vmec,
            theta=theta,
            zeta=zeta,
            separation_tz=sep,
        )
        mats = build_matrices(
            dict(base_inputs, geometry_option_coil=3, nescin_filename="(generated)", separation=float(separation0)),
            plasma,
            coil,
        )
        sol = solve_one_lambda(mats, jnp.asarray(lam, dtype=jnp.float64))
        chi2_B, chi2_K, max_B, max_K = diagnostics(mats, sol[None, :])
        phys = chi2_B[0] + (jnp.asarray(chi2K_weight, dtype=chi2_B.dtype) * chi2_K[0])
        return phys + (jnp.asarray(config.smooth_weight, dtype=chi2_B.dtype) * smooth) + (
            jnp.asarray(config.mean_weight, dtype=chi2_B.dtype) * mean_pen
        )

    vng = jax.jit(jax.value_and_grad(objective_from_params))

    # Initialize parameters so that the mean separation starts near `separation0`.
    sep0_eff = jnp.maximum(jnp.asarray(separation0, dtype=jnp.float64) - jnp.asarray(config.separation_min, dtype=jnp.float64), 1e-6)
    base0 = jnp.log(jnp.expm1(sep0_eff) + 1e-300)
    params = jnp.concatenate(
        [base0[None], jnp.zeros((2 * nmodes,), dtype=jnp.float64)],
        axis=0,
    )

    # Simple Adam (implemented inline to avoid deps).
    b1 = 0.9
    b2 = 0.999
    eps = 1.0e-8
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)

    base_hist = [params[0]]
    cs_hist = [params[1 : 1 + nmodes]]
    cc_hist = [params[1 + nmodes : 1 + 2 * nmodes]]
    obj_hist = [vng(params)[0]]

    for t in range(1, int(nsteps) + 1):
        _, g = vng(params)
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g * g)
        mhat = m / (1.0 - (b1**t))
        vhat = v / (1.0 - (b2**t))
        params = params - (jnp.asarray(step_size, dtype=params.dtype) * mhat / (jnp.sqrt(vhat) + eps))

        base_hist.append(params[0])
        cs_hist.append(params[1 : 1 + nmodes])
        cc_hist.append(params[1 + nmodes : 1 + 2 * nmodes])
        obj_hist.append(vng(params)[0])

    return SeparationFieldOptResult(
        base_history=jnp.stack(base_hist, axis=0),
        coeff_sin_history=jnp.stack(cs_hist, axis=0),
        coeff_cos_history=jnp.stack(cc_hist, axis=0),
        objective_history=jnp.stack(obj_hist, axis=0),
    )


def write_optimized_winding_surface_nescin(
    *,
    path: str,
    wout_filename: str,
    base: float,
    coeff_sin: jnp.ndarray,
    coeff_cos: jnp.ndarray,
    config: SeparationFieldOptConfig,
    ntheta: int,
    nzeta: int,
    max_mpol_coil: int = 24,
    max_ntor_coil: int = 24,
) -> None:
    """Generate and write a nescin file for an optimized separation field."""
    bound = read_wout_boundary(wout_filename, radial_mode="full")
    vmec = _as_fourier_surface(bound)

    # Build theta/zeta for the requested resolution.
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, (2.0 * jnp.pi) / int(vmec.nfp), int(nzeta), endpoint=False, dtype=jnp.float64)
    xm_sep, xn_sep = _init_mode_list(mpol=int(config.mpol_sep), ntor=int(config.ntor_sep), nfp=int(vmec.nfp))

    sep = separation_field_from_modes(
        theta=theta,
        zeta=zeta,
        xm=xm_sep,
        xn=xn_sep,
        base=jnp.asarray(base, dtype=jnp.float64),
        coeff_sin=jnp.asarray(coeff_sin, dtype=jnp.float64),
        coeff_cos=jnp.asarray(coeff_cos, dtype=jnp.float64),
        separation_min=float(config.separation_min),
    )

    coil_surf, _ = coil_surface_from_vmec_offset_separation_field(
        vmec_surface=vmec,
        theta=theta,
        zeta=zeta,
        separation_tz=sep,
        max_mpol_coil=int(max_mpol_coil),
        max_ntor_coil=int(max_ntor_coil),
    )
    nescin = nescin_current_surface_from_regcoil_fourier_surface(coil_surf=coil_surf)
    write_nescin_current_surface(path, nescin)
