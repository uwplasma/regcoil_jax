from __future__ import annotations
import jax.numpy as jnp

from .constants import twopi, mu0, pi
from .build_basis import build_basis_and_f
from .kernel import inductance_and_h_sum
from .io_bnorm import read_bnorm_modes
from .spectral_deriv import deriv_theta, deriv_zeta
from .quadcoil_objectives import build_gradphi2_matrix

def _flatten_TZ_to_N3(r_3TZ):
    # (3,T,Z) -> (N,3)
    return jnp.reshape(jnp.moveaxis(r_3TZ, 0, -1), (-1,3))

def _flatten_TZ_to_N(r_TZ):
    return jnp.reshape(r_TZ, (-1,))

def build_matrices(inputs, plasma, coil):
    """Build REGCOIL matrices in JAX (subset).

    Currently supports:
      - regularization_term_option in {chi2_K, K_xy, K_zeta, Laplace-Beltrami}
      - geometry_option_plasma in {0,1,2,3,4,5,6,7} (see surfaces.py)
      - geometry_option_coil in {0,1,2,3,4} (see surfaces.py)
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
    save_level = int(inputs.get("save_level", 3))
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
    # We compute them unconditionally to match the Fortran structure. For parity, prefer
    # analytic 2nd derivatives if they are available (VMEC/Fourier surfaces); otherwise
    # fall back to spectral differentiation on the (theta,zeta) grid.
    rth = coil["rth"]
    rze = coil["rze"]
    if "rtt" in coil and "rtz" in coil and "rzz" in coil:
        d2rdtheta2 = coil["rtt"]
        d2rdthetadzeta = coil["rtz"]
        d2rdzeta2 = coil["rzz"]
    else:
        d2rdtheta2 = deriv_theta(rth)
        d2rdthetadzeta = deriv_zeta(rth, nfp=nfp)
        d2rdzeta2 = deriv_zeta(rze, nfp=nfp)

    # Use the same formulas as regcoil_build_matrices.f90 (variable naming follows Fortran).
    d_g_theta_theta_d_theta = 2.0 * jnp.sum(d2rdtheta2 * rth, axis=0)
    d_g_theta_theta_d_zeta = 2.0 * jnp.sum(d2rdthetadzeta * rth, axis=0)

    d_g_theta_zeta_d_theta = jnp.sum(d2rdtheta2 * rze + d2rdthetadzeta * rth, axis=0)
    d_g_theta_zeta_d_zeta = jnp.sum(d2rdthetadzeta * rze + d2rdzeta2 * rth, axis=0)

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

    xm, xn, basis, fx, fy, fz, flb, dphi_dtheta_basis, dphi_dzeta_basis = build_basis_and_f(
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
        # FOCUS-format boundary files (geometry_option_plasma=7) can embed Bn coefficients.
        # In Fortran this is handled in regcoil_read_bnorm.f90 by checking nbf>0.
        focus = plasma.get("focus_bnorm", None)
        if focus is not None:
            bfm = jnp.asarray(focus["bfm"], dtype=jnp.int32)
            bfn = jnp.asarray(focus["bfn"], dtype=jnp.int32)
            bfc = jnp.asarray(focus["bfc"], dtype=jnp.float64)
            bfs = jnp.asarray(focus["bfs"], dtype=jnp.float64)

            th = plasma["theta"]
            ze = plasma["zeta"]
            ang = bfm[:, None, None] * th[None, :, None] - bfn[:, None, None] * ze[None, None, :]
            Bplasma = jnp.sum(bfc[:, None, None] * jnp.cos(ang) + bfs[:, None, None] * jnp.sin(ang), axis=0)
        else:
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
    elif reg_opt in ("k_zeta", "k-zeta", "kzeta"):
        # Regularize only the *toroidal* (zeta) component of the surface current density K.
        #
        # Let t_zeta = dr/dzeta be the physical tangent vector along the zeta coordinate on the coil surface,
        # and t̂_zeta = t_zeta / |t_zeta|.
        #
        # REGCOIL's "KDifference" vectors satisfy:
        #   K(θ,ζ) = (d(θ,ζ) - f(θ,ζ)·Φ) / |N(θ,ζ)|
        # where N = r_zeta × r_theta is the (non-unit) normal vector.
        #
        # The zeta-component regularization is then:
        #   χ²_{Kζ} = ∫ (K · t̂_zeta)^2 dA
        #          = ∫ ( (d·t̂_zeta - (f·Φ)·t̂_zeta)^2 / |N| ) dθ dζ
        #
        # So we form a scalar basis matrix f_zeta = f · t̂_zeta and a scalar d_zeta = d · t̂_zeta,
        # and reuse the same 1/|N| weighting used throughout the Fortran implementation.
        t_zeta = _flatten_TZ_to_N3(coil["rze"])  # (Nc,3)
        t_norm = jnp.linalg.norm(t_zeta, axis=1)
        t_norm = jnp.where(t_norm == 0.0, 1.0, t_norm)
        t_hat = t_zeta / t_norm[:, None]
        tx = t_hat[:, 0]
        ty = t_hat[:, 1]
        tz = t_hat[:, 2]
        f_zeta = fx * tx[:, None] + fy * ty[:, None] + fz * tz[:, None]  # (Nc,nb)
        f_zeta_over_Nc = f_zeta / normNc[:, None]
        d_zeta = dx * tx + dy * ty + dz * tz  # (Nc,)
        matrix_reg = (dth_c * dze_c) * (f_zeta.T @ f_zeta_over_Nc)
        RHS_reg = (dth_c * dze_c) * (d_zeta @ f_zeta_over_Nc)
    elif reg_opt in ("laplace-beltrami", "laplace_beltrami", "laplace beltrami"):
        matrix_reg = (dth_c * dze_c) * (flb.T @ flb_over_Nc)
        RHS_reg = (dth_c * dze_c) * (d_LB @ flb_over_Nc)
    else:
        raise ValueError(f"Unsupported regularization_term_option={reg_opt!r}")

    # Quadcoil-style regularization on |∇_s Φ|^2 (purely on the winding-surface current potential).
    # This term is quadratic in the coefficients and can be included as an additional matrix.
    gradphi2_weight = float(inputs.get("gradphi2_weight", 0.0))
    if gradphi2_weight != 0.0:
        normNc = _flatten_TZ_to_N(coil["normN"])
        gtt_flat = _flatten_TZ_to_N(gtt)
        gtz_flat = _flatten_TZ_to_N(gtz)
        gzz_flat = _flatten_TZ_to_N(gzz)
        Q = build_gradphi2_matrix(
            dphi_dtheta_basis=dphi_dtheta_basis,
            dphi_dzeta_basis=dphi_dzeta_basis,
            normN_coil=normNc,
            gtt=gtt_flat,
            gtz=gtz_flat,
            gzz=gzz_flat,
            dth=dth_c,
            dze=dze_c,
            nfp=nfp,
        )
        matrix_reg = matrix_reg + gradphi2_weight * Q

    area_plasma = nfp * dth_p * dze_p * jnp.sum(plasma["normN"])
    area_coil = nfp * dth_c * dze_c * jnp.sum(coil["normN"])

    # Volumes (for output parity): match REGCOIL Fortran formula used in
    # regcoil_init_plasma_mod.f90 and regcoil_evaluate_coil_surface.f90:
    #   V = | ∫ (1/2) R^2 dZ dζ |
    # evaluated on the half-theta grid (R^2 interpolated from the full to half grid).
    def _volume_r2_dz(r3tz: jnp.ndarray, dze: float, nfp: int) -> jnp.ndarray:
        r3tz = jnp.asarray(r3tz)  # (3,ntheta,nzeta) for a single field period in zeta
        R2 = r3tz[0] * r3tz[0] + r3tz[1] * r3tz[1]  # (T,Z)
        Z = r3tz[2]  # (T,Z)
        # Interior theta segments:
        R2_half = 0.5 * (R2[:-1, :] + R2[1:, :])
        dZ = Z[1:, :] - Z[:-1, :]
        acc = jnp.sum(R2_half * dZ)
        # End segment (theta wraps around):
        acc = acc + jnp.sum(0.5 * (R2[0, :] + R2[-1, :]) * (Z[0, :] - Z[-1, :]))
        # r includes only one field period; multiply by nfp to match Fortran's nzetal accumulation.
        return jnp.abs((nfp * acc) * dze / 2.0)

    vol_plasma = _volume_r2_dz(plasma["r"], dze_p, nfp)
    vol_coil = _volume_r2_dz(coil["r"], dze_c, nfp)

    out = dict(
        nfp=nfp,
        lasym=bool(plasma.get("lasym", False) or coil.get("lasym", False)),
        # Metadata scalars (match regcoil_write_output.f90 naming where possible)
        mpol_potential=int(mpol_pot),
        ntor_potential=int(ntor_pot),
        mnmax_potential=int(xm.shape[0]),
        num_basis_functions=int(nb),
        symmetry_option=int(symmetry_option),
        save_level=int(save_level),
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
        volume_plasma=vol_plasma,
        volume_coil=vol_coil,
        xm=xm, xn=xn,
        basis=basis,
        g_over_Np=g_over_Np,
        fx=fx, fy=fy, fz=fz,
        flb=flb,
        dphi_dtheta_basis=dphi_dtheta_basis,
        dphi_dzeta_basis=dphi_dzeta_basis,
        dx=dx, dy=dy, dz=dz,
        d_Laplace_Beltrami=d_LB,
        normNp=normNp, normNc=normNc,
        gtt_coil=gtt,
        gtz_coil=gtz,
        gzz_coil=gzz,
        Bnet=Bnet, Bplasma=Bplasma,
        matrix_B=matrix_B, RHS_B=RHS_B,
        matrix_reg=matrix_reg, RHS_reg=RHS_reg,
        RHS_regularization=RHS_reg,
        h=h,
        dth_p=dth_p, dze_p=dze_p, dth_c=dth_c, dze_c=dze_c,
        # Surface Fourier coefficients (for output parity)
        xm_plasma=plasma.get("xm_plasma", None),
        xn_plasma=plasma.get("xn_plasma", None),
        rmnc_plasma=plasma.get("rmnc_plasma", None),
        zmns_plasma=plasma.get("zmns_plasma", None),
        rmns_plasma=plasma.get("rmns_plasma", None),
        zmnc_plasma=plasma.get("zmnc_plasma", None),
        xm_coil=coil.get("xm_coil", None),
        xn_coil=coil.get("xn_coil", None),
        rmnc_coil=coil.get("rmnc_coil", None),
        zmns_coil=coil.get("zmns_coil", None),
        rmns_coil=coil.get("rmns_coil", None),
        zmnc_coil=coil.get("zmnc_coil", None),
    )
    # Optional heavy outputs, matching REGCOIL save_level behavior:
    #   save_level < 1: save inductance
    #   save_level < 2: save g
    if save_level < 1:
        out["inductance"] = ind_eff
    if save_level < 2:
        out["g"] = g
    return out
