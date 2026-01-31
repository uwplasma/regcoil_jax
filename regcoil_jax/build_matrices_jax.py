from __future__ import annotations
import jax.numpy as jnp

from .constants import twopi, mu0, pi
from .build_basis import build_basis_and_f
from .kernel import inductance_and_h_sum

def _flatten_TZ_to_N3(r_3TZ):
    # (3,T,Z) -> (N,3)
    return jnp.reshape(jnp.moveaxis(r_3TZ, 0, -1), (-1,3))

def _flatten_TZ_to_N(r_TZ):
    return jnp.reshape(r_TZ, (-1,))

def build_matrices(inputs, plasma, coil):
    """Build REGCOIL matrices in JAX (subset).

    Currently supports:
      - regularization_term_option = chi2_K (default)
      - geometry_option_plasma in {1,2}; geometry_option_coil in {1,2}
      - load_bnorm not yet supported (Bnormal_from_plasma_current=0)
    """
    nfp = int(plasma["nfp"])
    ntheta_p = int(inputs["ntheta_plasma"]); nzeta_p = int(inputs["nzeta_plasma"])
    ntheta_c = int(inputs["ntheta_coil"]);   nzeta_c = int(inputs["nzeta_coil"])
    dth_p = float(twopi / ntheta_p)
    dze_p = float((twopi / nfp) / nzeta_p)
    dth_c = float(twopi / ntheta_c)
    dze_c = float((twopi / nfp) / nzeta_c)

    # Defaults
    symmetry_option = int(inputs.get("symmetry_option", 1))
    mpol_pot = int(inputs.get("mpol_potential", inputs.get("mpol_coil", 10)))
    ntor_pot = int(inputs.get("ntor_potential", inputs.get("ntor_coil", 10)))
    # Match regcoil_variables.f90 defaults:
    #   net_poloidal_current_Amperes = 1
    #   net_toroidal_current_Amperes = 0
    net_pol = float(inputs.get("net_poloidal_current_amperes", 1.0))
    net_tor = float(inputs.get("net_toroidal_current_amperes", 0.0))

    # coil metric (LB placeholders)
    gtt = jnp.sum(coil["rth"]*coil["rth"], axis=0)
    gzz = jnp.sum(coil["rze"]*coil["rze"], axis=0)
    gtz = jnp.sum(coil["rth"]*coil["rze"], axis=0)
    lb_th = jnp.zeros_like(gtt)
    lb_ze = jnp.zeros_like(gtt)

    xm, xn, basis, fx, fy, fz, flb = build_basis_and_f(
        coil["theta"], coil["zeta"], coil["rth"], coil["rze"],
        gtt, gtz, gzz, lb_th, lb_ze,
        mpol_pot, ntor_pot, nfp, symmetry_option
    )
    nb = basis.shape[1]

    # Inductance and h
    rP = _flatten_TZ_to_N3(plasma["r"])        # (Np,3)
    # For parity with regcoil_build_matrices.f90, the inductance kernel uses the
    # *full* (non-unit) surface normals, i.e. N = r_zeta × r_theta.
    nP_full = plasma["nunit"] * plasma["normN"][None, :, :]
    nC_full = coil["nunit"] * coil["normN"][None, :, :]
    nP = _flatten_TZ_to_N3(nP_full)            # (Np,3)
    rC = _flatten_TZ_to_N3(coil["r"])          # (Nc,3)
    nC = _flatten_TZ_to_N3(nC_full)            # (Nc,3)

    f_h_3TZ = net_pol * coil["rth"] - net_tor * coil["rze"]
    f_h = _flatten_TZ_to_N3(f_h_3TZ)

    ind_eff, h_sum = inductance_and_h_sum(rP, nP, rC, nC, f_h, nfp=nfp)

    g = (dth_c*dze_c) * (ind_eff @ basis)  # (Np,nb)
    # h scaling (Fortran): h *= dth_c*dze_c*mu0/(8*pi*pi)
    h = h_sum * (dth_c*dze_c*mu0/(8.0*pi*pi))  # (Np,)

    # h is computed using full plasma normals; convert to Bnormal by dividing by |N_plasma|.
    Bnet = jnp.reshape(h, (ntheta_p, nzeta_p)) / plasma["normN"]
    Bplasma = jnp.zeros_like(Bnet)
    Btarget = Bplasma + Bnet

    RHS_B = -(dth_p*dze_p) * (jnp.reshape(Btarget, (-1,)) @ g)  # (nb,)

    # Normals
    normNp = _flatten_TZ_to_N(plasma["normN"])
    normNc = _flatten_TZ_to_N(coil["normN"])
    g_over_Np = g / normNp[:,None]

    matrix_B = (dth_p*dze_p) * (g.T @ g_over_Np)  # (nb,nb)

    # Regularization chi2_K
    fx_over_Nc = fx / normNc[:,None]
    fy_over_Nc = fy / normNc[:,None]
    fz_over_Nc = fz / normNc[:,None]

    matrix_reg = (dth_c*dze_c) * (fx.T @ fx_over_Nc + fy.T @ fy_over_Nc + fz.T @ fz_over_Nc)

    # Match regcoil_build_matrices.f90:
    #   d = (net_poloidal_current * dr/dtheta - net_toroidal_current * dr/dzeta) / (2*pi)
    dvec = (net_pol * coil["rth"] - net_tor * coil["rze"]) / twopi  # (3,T,Z)
    dx = _flatten_TZ_to_N(dvec[0]); dy=_flatten_TZ_to_N(dvec[1]); dz=_flatten_TZ_to_N(dvec[2])
    RHS_reg = (dth_c*dze_c) * (dx @ fx_over_Nc + dy @ fy_over_Nc + dz @ fz_over_Nc)

    area_plasma = nfp * dth_p * dze_p * jnp.sum(plasma["normN"])
    area_coil = nfp * dth_c * dze_c * jnp.sum(coil["normN"])

    return dict(
        nfp=nfp,
        # Grids / geometry (for output + diagnostics)
        theta_plasma=plasma["theta"],
        zeta_plasma=plasma["zeta"],
        theta_coil=coil["theta"],
        zeta_coil=coil["zeta"],
        normN_plasma=plasma["normN"],
        normN_coil=coil["normN"],
        area_plasma=area_plasma,
        area_coil=area_coil,
        xm=xm, xn=xn,
        basis=basis,
        g=g, g_over_Np=g_over_Np,
        fx=fx, fy=fy, fz=fz,
        dx=dx, dy=dy, dz=dz,
        normNp=normNp, normNc=normNc,
        Bnet=Bnet, Bplasma=Bplasma,
        matrix_B=matrix_B, RHS_B=RHS_B,
        matrix_reg=matrix_reg, RHS_reg=RHS_reg,
        dth_p=dth_p, dze_p=dze_p, dth_c=dth_c, dze_c=dze_c,
    )
