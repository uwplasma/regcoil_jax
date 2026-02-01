from __future__ import annotations
import jax.numpy as jnp

from .constants import twopi, mu0, pi
from .build_basis import build_basis_and_f
from .kernel import inductance_and_h_sum
from .io_bnorm import read_bnorm_modes
from .spectral_deriv import deriv_theta, deriv_zeta

def _flatten_TZ_to_N3(r_3TZ):
    # (3,T,Z) -> (N,3)
    return jnp.reshape(jnp.moveaxis(r_3TZ, 0, -1), (-1,3))

def _flatten_TZ_to_N(r_TZ):
    return jnp.reshape(r_TZ, (-1,))

def build_matrices(inputs, plasma, coil):
    """Build REGCOIL matrices in JAX (subset).

    Currently supports:
      - regularization_term_option in {chi2_K, K_xy, Laplace-Beltrami}
      - geometry_option_plasma in {1,2}; geometry_option_coil in {1,2}
      - load_bnorm (bnorm file) for Bnormal_from_plasma_current
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

    # Coil metric tensors.
    gtt = jnp.sum(coil["rth"]*coil["rth"], axis=0)
    gzz = jnp.sum(coil["rze"]*coil["rze"], axis=0)
    gtz = jnp.sum(coil["rth"]*coil["rze"], axis=0)

    # Laplace–Beltrami coefficients (see regcoil_build_matrices.f90).
    # These coefficients are used both for:
    #   - Laplace–Beltrami regularization option
    #   - diagnostics chi2_Laplace_Beltrami
    #
    # We compute them unconditionally to match the Fortran structure.
    rth = coil["rth"]
    rze = coil["rze"]
    d2rdtheta2 = deriv_theta(rth)
    d2rdthetadzeta = deriv_zeta(rth, nfp=nfp)
    d2rdzeta2 = deriv_zeta(rze, nfp=nfp)

    d_g_theta_theta_d_theta = 2.0 * jnp.sum(d2rdtheta2 * rth, axis=0)
    d_g_theta_theta_d_zeta = 2.0 * jnp.sum(d2rdthetadzeta * rth, axis=0)

    d_g_theta_zeta_d_theta = jnp.sum(d2rdtheta2 * rze + d2rdthetadzeta * rth, axis=0)
    d_g_theta_zeta_d_zeta = jnp.sum(d2rdzeta2 * rth + d2rdthetadzeta * rze, axis=0)

    d_g_zeta_zeta_d_theta = 2.0 * jnp.sum(d2rdthetadzeta * rze, axis=0)
    d_g_zeta_zeta_d_zeta = 2.0 * jnp.sum(d2rdzeta2 * rze, axis=0)

    normN = coil["normN"]
    # Protect division at isolated bad points (should not happen for valid surfaces).
    normN_safe = jnp.where(normN == 0.0, 1.0, normN)

    d_N_d_theta = (d_g_zeta_zeta_d_theta * gtt + d_g_theta_theta_d_theta * gzz - 2.0 * d_g_theta_zeta_d_theta * gtz) / (
        2.0 * normN_safe
    )
    d_N_d_zeta = (d_g_zeta_zeta_d_zeta * gtt + d_g_theta_theta_d_zeta * gzz - 2.0 * d_g_theta_zeta_d_zeta * gtz) / (
        2.0 * normN_safe
    )

    LB_dPhi_dtheta_coeff = (
        d_g_zeta_zeta_d_theta
        - d_g_theta_zeta_d_zeta
        + (-gzz * d_N_d_theta + gtz * d_N_d_zeta) / normN_safe
    ) / normN_safe
    LB_dPhi_dzeta_coeff = (
        d_g_theta_theta_d_zeta
        - d_g_theta_zeta_d_theta
        + (gtz * d_N_d_theta - gtt * d_N_d_zeta) / normN_safe
    ) / normN_safe

    xm, xn, basis, fx, fy, fz, flb = build_basis_and_f(
        coil["theta"], coil["zeta"], coil["rth"], coil["rze"],
        gtt, gtz, gzz, LB_dPhi_dtheta_coeff, LB_dPhi_dzeta_coeff,
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

    # Bnormal_from_plasma_current:
    #   - default is 0 (parity with REGCOIL when load_bnorm=.false.)
    #   - when load_bnorm=.true., read Fourier modes from file and evaluate:
    #         sum bf * sin(m*theta + n*nfp*zeta)
    #     then undo BNORM scaling by multiplying by curpol.
    load_bnorm = bool(inputs.get("load_bnorm", False))
    if load_bnorm:
        bnorm_filename = inputs.get("bnorm_filename", None)
        if bnorm_filename is None:
            raise ValueError("load_bnorm=.true. requires bnorm_filename")
        if "curpol" not in inputs:
            raise ValueError("load_bnorm requires curpol (set by VMEC wout or specify curpol in the input)")
        curpol = float(inputs["curpol"])

        m_np, n_np, bf_np = read_bnorm_modes(str(bnorm_filename))
        if m_np.size == 0:
            raise ValueError(f"No modes found in bnorm file: {bnorm_filename}")
        m = jnp.asarray(m_np, dtype=jnp.int32)
        n = jnp.asarray(n_np, dtype=jnp.int32)
        bf = jnp.asarray(bf_np, dtype=jnp.float64)

        th = plasma["theta"]
        ze = plasma["zeta"]
        ang = m[:, None, None] * th[None, :, None] + (n[:, None, None] * nfp) * ze[None, None, :]
        Bplasma = jnp.sum(bf[:, None, None] * jnp.sin(ang), axis=0) * curpol
    else:
        Bplasma = jnp.zeros_like(Bnet)
    Btarget = Bplasma + Bnet

    RHS_B = -(dth_p*dze_p) * (jnp.reshape(Btarget, (-1,)) @ g)  # (nb,)

    # Normals
    normNp = _flatten_TZ_to_N(plasma["normN"])
    normNc = _flatten_TZ_to_N(coil["normN"])
    g_over_Np = g / normNp[:,None]

    matrix_B = (dth_p*dze_p) * (g.T @ g_over_Np)  # (nb,nb)

    # Match regcoil_build_matrices.f90:
    #   d = (net_poloidal_current * dr/dtheta - net_toroidal_current * dr/dzeta) / (2*pi)
    dvec = (net_pol * coil["rth"] - net_tor * coil["rze"]) / twopi  # (3,T,Z)
    dx = _flatten_TZ_to_N(dvec[0]); dy=_flatten_TZ_to_N(dvec[1]); dz=_flatten_TZ_to_N(dvec[2])

    # Laplace–Beltrami d-vector (scalar)
    d_LB_TZ = -(net_pol / twopi) * LB_dPhi_dzeta_coeff - (net_tor / twopi) * LB_dPhi_dtheta_coeff
    d_LB = _flatten_TZ_to_N(d_LB_TZ)

    # Regularization matrix and RHS
    reg_opt = str(inputs.get("regularization_term_option", "chi2_K")).strip().lower()

    fx_over_Nc = fx / normNc[:, None]
    fy_over_Nc = fy / normNc[:, None]
    fz_over_Nc = fz / normNc[:, None]
    flb_over_Nc = flb / normNc[:, None]

    if reg_opt in ("chi2_k", "chi2-k"):
        matrix_reg = (dth_c * dze_c) * (fx.T @ fx_over_Nc + fy.T @ fy_over_Nc + fz.T @ fz_over_Nc)
        RHS_reg = (dth_c * dze_c) * (dx @ fx_over_Nc + dy @ fy_over_Nc + dz @ fz_over_Nc)
    elif reg_opt in ("k_xy", "k-xy"):
        matrix_reg = (dth_c * dze_c) * (fx.T @ fx_over_Nc + fy.T @ fy_over_Nc)
        RHS_reg = (dth_c * dze_c) * (dx @ fx_over_Nc + dy @ fy_over_Nc)
    elif reg_opt in ("laplace-beltrami", "laplace_beltrami", "laplace beltrami"):
        matrix_reg = (dth_c * dze_c) * (flb.T @ flb_over_Nc)
        RHS_reg = (dth_c * dze_c) * (d_LB @ flb_over_Nc)
    else:
        raise ValueError(f"Unsupported regularization_term_option={reg_opt!r}")

    area_plasma = nfp * dth_p * dze_p * jnp.sum(plasma["normN"])
    area_coil = nfp * dth_c * dze_c * jnp.sum(coil["normN"])

    return dict(
        nfp=nfp,
        net_poloidal_current_Amperes=net_pol,
        net_toroidal_current_Amperes=net_tor,
        # Grids / geometry (for output + diagnostics)
        theta_plasma=plasma["theta"],
        zeta_plasma=plasma["zeta"],
        theta_coil=coil["theta"],
        zeta_coil=coil["zeta"],
        r_plasma=plasma["r"],
        r_coil=coil["r"],
        drdtheta_plasma=plasma["rth"],
        drdzeta_plasma=plasma["rze"],
        drdtheta_coil=coil["rth"],
        drdzeta_coil=coil["rze"],
        normal_plasma=nP_full,
        normal_coil=nC_full,
        normN_plasma=plasma["normN"],
        normN_coil=coil["normN"],
        area_plasma=area_plasma,
        area_coil=area_coil,
        xm=xm, xn=xn,
        basis=basis,
        g=g, g_over_Np=g_over_Np,
        fx=fx, fy=fy, fz=fz,
        flb=flb,
        dx=dx, dy=dy, dz=dz,
        d_Laplace_Beltrami=d_LB,
        normNp=normNp, normNc=normNc,
        Bnet=Bnet, Bplasma=Bplasma,
        matrix_B=matrix_B, RHS_B=RHS_B,
        matrix_reg=matrix_reg, RHS_reg=RHS_reg,
        dth_p=dth_p, dze_p=dze_p, dth_c=dth_c, dze_c=dze_c,
    )
