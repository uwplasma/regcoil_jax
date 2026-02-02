from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass(frozen=True)
class QuadcoilMetrics:
    delta_phi: float
    gradphi_rms: float
    gradphi_max: float
    coil_spacing_min: float
    coil_spacing_rms: float
    total_contour_length_est: float
    int_gradphi2: float


def surface_metric_inverse(*, gtt: jnp.ndarray, gtz: jnp.ndarray, gzz: jnp.ndarray):
    """Return inverse 2-metric components and determinant on a (theta,zeta) surface grid.

    The metric tensor is:
      g = [[gtt, gtz],
           [gtz, gzz]]

    Returns:
      ginv_tt, ginv_tz, ginv_zz, det
    """
    det = gtt * gzz - gtz * gtz
    det_safe = jnp.where(det == 0.0, 1.0, det)
    ginv_tt = gzz / det_safe
    ginv_tz = -gtz / det_safe
    ginv_zz = gtt / det_safe
    return ginv_tt, ginv_tz, ginv_zz, det_safe


def gradphi_squared(
    *,
    dphi_dtheta: jnp.ndarray,
    dphi_dzeta: jnp.ndarray,
    gtt: jnp.ndarray,
    gtz: jnp.ndarray,
    gzz: jnp.ndarray,
) -> jnp.ndarray:
    """Compute |∇_s Φ|^2 on a parametric surface.

    Uses:
      |∇_s Φ|^2 = g^{θθ} Φ_θ^2 + 2 g^{θζ} Φ_θ Φ_ζ + g^{ζζ} Φ_ζ^2
    where g^{ij} is the inverse of the surface metric tensor.
    """
    ginv_tt, ginv_tz, ginv_zz, _ = surface_metric_inverse(gtt=gtt, gtz=gtz, gzz=gzz)
    return ginv_tt * (dphi_dtheta * dphi_dtheta) + 2.0 * ginv_tz * (dphi_dtheta * dphi_dzeta) + ginv_zz * (
        dphi_dzeta * dphi_dzeta
    )


def build_gradphi2_matrix(
    *,
    dphi_dtheta_basis: jnp.ndarray,
    dphi_dzeta_basis: jnp.ndarray,
    normN_coil: jnp.ndarray,
    gtt: jnp.ndarray,
    gtz: jnp.ndarray,
    gzz: jnp.ndarray,
    dth: float,
    dze: float,
    nfp: int,
) -> jnp.ndarray:
    """Build the quadratic form matrix Q for ∫ |∇_s Φ|^2 dA.

    With coefficients c, define Φ_θ = Dθ c and Φ_ζ = Dζ c on the coil surface grid.
    Then:
      ∫ |∇_s Φ|^2 dA = c^T Q c

    Args:
      dphi_*_basis: (Ncoil, nbasis) flattened derivative basis matrices.
      normN_coil: (Ncoil,) = |r_ζ × r_θ| on the coil surface.
      gtt,gtz,gzz: (Ncoil,) metric tensor components on the coil surface.
    """
    Dth = jnp.asarray(dphi_dtheta_basis, dtype=jnp.float64)
    Dze = jnp.asarray(dphi_dzeta_basis, dtype=jnp.float64)
    normN = jnp.asarray(normN_coil, dtype=jnp.float64)
    gtt = jnp.asarray(gtt, dtype=jnp.float64)
    gtz = jnp.asarray(gtz, dtype=jnp.float64)
    gzz = jnp.asarray(gzz, dtype=jnp.float64)

    ginv_tt, ginv_tz, ginv_zz, _ = surface_metric_inverse(gtt=gtt, gtz=gtz, gzz=gzz)
    w = (float(nfp) * float(dth) * float(dze)) * normN  # dA = |N| dθ dζ, multiplied by nfp

    # Form Q = Dθᵀ W g^{θθ} Dθ + Dθᵀ W g^{θζ} Dζ + Dζᵀ W g^{θζ} Dθ + Dζᵀ W g^{ζζ} Dζ.
    w_tt = (w * ginv_tt)[:, None]
    w_tz = (w * ginv_tz)[:, None]
    w_zz = (w * ginv_zz)[:, None]
    Q = Dth.T @ (Dth * w_tt)
    Q = Q + Dth.T @ (Dze * w_tz)
    Q = Q + Dze.T @ (Dth * w_tz)
    Q = Q + Dze.T @ (Dze * w_zz)
    return Q


def quadcoil_metrics_from_mats_and_solution(
    mats: dict[str, Any],
    sol: jnp.ndarray,
    *,
    coils_per_half_period: int,
    eps: float = 1e-30,
) -> QuadcoilMetrics:
    """Compute coil-spacing/length metrics from a REGCOIL solution (Quadcoil-style diagnostics).

    These metrics are differentiable w.r.t the current-potential coefficients because they depend
    only on Φ and its surface gradient on the winding surface (no contour cutting).
    """
    nfp = int(mats["nfp"])
    ncoils = int(2 * int(coils_per_half_period) * nfp)
    net_pol = float(mats.get("net_poloidal_current_Amperes", 1.0))
    delta_phi = abs(net_pol) / float(ncoils) if ncoils > 0 else 0.0

    nb = int(sol.shape[0])
    Dth = jnp.asarray(mats["dphi_dtheta_basis"], dtype=jnp.float64)
    Dze = jnp.asarray(mats["dphi_dzeta_basis"], dtype=jnp.float64)
    if Dth.shape[1] != nb or Dze.shape[1] != nb:
        raise ValueError("mats derivative basis shapes do not match solution length")

    phi_theta = Dth @ sol
    phi_zeta = Dze @ sol
    normN = jnp.asarray(mats["normN_coil"], dtype=jnp.float64).reshape((-1,))
    gtt = jnp.asarray(mats["gtt_coil"], dtype=jnp.float64).reshape((-1,))
    gtz = jnp.asarray(mats["gtz_coil"], dtype=jnp.float64).reshape((-1,))
    gzz = jnp.asarray(mats["gzz_coil"], dtype=jnp.float64).reshape((-1,))
    dth = float(mats["dth_c"])
    dze = float(mats["dze_c"])

    grad2 = gradphi_squared(dphi_dtheta=phi_theta, dphi_dzeta=phi_zeta, gtt=gtt, gtz=gtz, gzz=gzz)
    grad2 = jnp.maximum(grad2, 0.0)
    grad = jnp.sqrt(grad2 + float(eps))

    # Coil-to-coil spacing estimate (local):
    #   Δs ≈ ΔΦ / |∇_s Φ|
    spacing = jnp.where(grad > 0.0, float(delta_phi) / grad, jnp.inf)

    # Coarea formula-inspired length estimate:
    #   ∫_surface |∇Φ| dA = ∫_levels L(level) dΦ
    # For equally spaced contours with spacing ΔΦ, total length ≈ (1/ΔΦ) ∫ |∇Φ| dA.
    dA = (float(nfp) * float(dth) * float(dze)) * normN
    total_length = jnp.sum(grad * dA) / float(delta_phi) if delta_phi > 0 else jnp.nan

    int_gradphi2 = jnp.sum(grad2 * dA)

    # "RMS" for spacing uses area weighting.
    area = jnp.sum(dA)
    w = jnp.where(area == 0.0, 0.0, dA / area)
    spacing_rms = jnp.sqrt(jnp.sum(w * spacing * spacing))

    grad_rms = jnp.sqrt(jnp.sum(w * grad2))

    return QuadcoilMetrics(
        delta_phi=float(delta_phi),
        gradphi_rms=float(grad_rms),
        gradphi_max=float(jnp.max(grad)),
        coil_spacing_min=float(jnp.min(spacing)),
        coil_spacing_rms=float(spacing_rms),
        total_contour_length_est=float(total_length),
        int_gradphi2=float(int_gradphi2),
    )


def quadcoil_metrics_from_phi_grid(
    *,
    phi_zt: jnp.ndarray,
    r_coil_zt3: jnp.ndarray,
    nfp: int,
    net_poloidal_current_Amperes: float,
    coils_per_half_period: int,
    eps: float = 1e-30,
) -> QuadcoilMetrics:
    """Compute Quadcoil-style metrics from the *grid* current potential Φ(ζ,θ) and coil surface geometry.

    This variant is convenient for postprocessing from netCDF outputs, and does not require
    access to the spectral basis / coefficient vector.
    """
    from .spectral_deriv import deriv_theta, deriv_zeta

    phi_zt = jnp.asarray(phi_zt, dtype=jnp.float64)
    r_zt3 = jnp.asarray(r_coil_zt3, dtype=jnp.float64)
    if phi_zt.ndim != 2:
        raise ValueError("phi_zt must be (nzeta, ntheta)")
    if r_zt3.ndim != 3 or r_zt3.shape[2] != 3:
        raise ValueError("r_coil_zt3 must be (nzeta, ntheta, 3)")
    nzeta, ntheta = int(phi_zt.shape[0]), int(phi_zt.shape[1])
    if r_zt3.shape[0] != nzeta or r_zt3.shape[1] != ntheta:
        raise ValueError("phi_zt and r_coil_zt3 must have matching (nzeta, ntheta)")

    # Convert to internal (3,T,Z) representation expected by spectral derivatives.
    r_3TZ = jnp.moveaxis(r_zt3, -1, 0).transpose(0, 2, 1)  # (3,ntheta,nzeta)
    phi_TZ = phi_zt.T  # (ntheta,nzeta)
    phi_1TZ = phi_TZ[None, :, :]

    rth = deriv_theta(r_3TZ)
    rze = deriv_zeta(r_3TZ, nfp=int(nfp))
    dphi_dtheta = deriv_theta(phi_1TZ)[0]
    dphi_dzeta = deriv_zeta(phi_1TZ, nfp=int(nfp))[0]

    gtt = jnp.sum(rth * rth, axis=0)
    gzz = jnp.sum(rze * rze, axis=0)
    gtz = jnp.sum(rth * rze, axis=0)
    normN = jnp.sqrt(jnp.maximum(gtt * gzz - gtz * gtz, 0.0))

    # Grid spacings.
    dth = float(2.0 * jnp.pi / float(ntheta))
    dze = float((2.0 * jnp.pi / float(nfp)) / float(nzeta))

    ncoils = int(2 * int(coils_per_half_period) * int(nfp))
    delta_phi = abs(float(net_poloidal_current_Amperes)) / float(ncoils) if ncoils > 0 else 0.0

    grad2 = gradphi_squared(dphi_dtheta=dphi_dtheta.reshape(-1), dphi_dzeta=dphi_dzeta.reshape(-1), gtt=gtt.reshape(-1), gtz=gtz.reshape(-1), gzz=gzz.reshape(-1))
    grad2 = jnp.maximum(grad2, 0.0)
    grad = jnp.sqrt(grad2 + float(eps))

    spacing = jnp.where(grad > 0.0, float(delta_phi) / grad, jnp.inf)
    dA = (float(nfp) * float(dth) * float(dze)) * normN.reshape(-1)
    total_length = jnp.sum(grad * dA) / float(delta_phi) if delta_phi > 0 else jnp.nan
    int_gradphi2 = jnp.sum(grad2 * dA)

    area = jnp.sum(dA)
    w = jnp.where(area == 0.0, 0.0, dA / area)
    spacing_rms = jnp.sqrt(jnp.sum(w * spacing * spacing))
    grad_rms = jnp.sqrt(jnp.sum(w * grad2))

    return QuadcoilMetrics(
        delta_phi=float(delta_phi),
        gradphi_rms=float(grad_rms),
        gradphi_max=float(jnp.max(grad)),
        coil_spacing_min=float(jnp.min(spacing)),
        coil_spacing_rms=float(spacing_rms),
        total_contour_length_est=float(total_length),
        int_gradphi2=float(int_gradphi2),
    )
