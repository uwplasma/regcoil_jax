from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import netCDF4
except Exception:
    netCDF4 = None

def _rotate_z_np(xyz: np.ndarray, phi: float) -> np.ndarray:
    """Rotate points/vectors about +z by angle `phi`.

    Accepts arrays shaped (..., 3).
    """
    c = np.cos(phi)
    s = np.sin(phi)
    out = np.array(xyz, copy=True)
    x = xyz[..., 0]
    y = xyz[..., 1]
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    out[..., 2] = xyz[..., 2]
    return out


def write_output_nc(
    path: str,
    inputs: dict,
    mats: dict,
    lambdas,
    sols,
    chi2_B,
    chi2_K,
    max_B,
    max_K,
    *,
    exit_code: int = 0,
    total_time: float | None = None,
):
    if netCDF4 is None:
        raise ImportError("netCDF4 is required to write REGCOIL output .nc files.")
    # -----------------------------
    # Fortran REGCOIL-compatible writer
    # -----------------------------
    ds = netCDF4.Dataset(path, "w")
    nlambda = int(len(lambdas))
    nb = int(sols.shape[1])
    nfp = int(mats.get("nfp", 1))
    lasym = bool(mats.get("lasym", False))
    save_level = int(inputs.get("save_level", 3))
    general_option = int(inputs.get("general_option", 1))
    target_option = str(inputs.get("target_option", "max_K")).strip()

    # Defaults from regcoil_variables.f90 (used when a value is not specified in the input file).
    defaults = dict(
        r0_plasma=10.0,
        a_plasma=0.5,
        r0_coil=10.0,
        a_coil=1.0,
        symmetry_option=1,
        mpol_potential=12,
        ntor_potential=12,
        net_poloidal_current_amperes=1.0,
        net_toroidal_current_amperes=0.0,
        curpol=1.0,
        sensitivity_option=1,
    )

    # Grid sizes and arrays (single field period for theta/zeta; full torus for r_* etc.)
    theta_p = np.asarray(mats.get("theta_plasma"))
    zeta_p = np.asarray(mats.get("zeta_plasma"))
    theta_c = np.asarray(mats.get("theta_coil"))
    zeta_c = np.asarray(mats.get("zeta_coil"))
    ntheta_plasma = int(theta_p.size)
    nzeta_plasma = int(zeta_p.size)
    ntheta_coil = int(theta_c.size)
    nzeta_coil = int(zeta_c.size)
    nzetal_plasma = int(nzeta_plasma * nfp)
    nzetal_coil = int(nzeta_coil * nfp)

    # Mode lists
    xm_potential = np.asarray(mats.get("xm"), dtype=np.int32)
    xn_potential = np.asarray(mats.get("xn"), dtype=np.int32)
    xm_plasma = np.asarray(mats.get("xm_plasma"), dtype=np.int32)
    xn_plasma = np.asarray(mats.get("xn_plasma"), dtype=np.int32)
    xm_coil = np.asarray(mats.get("xm_coil"), dtype=np.int32)
    xn_coil = np.asarray(mats.get("xn_coil"), dtype=np.int32)
    mnmax_potential = int(xm_potential.size)
    mnmax_plasma = int(xm_plasma.size)
    mnmax_coil = int(xm_coil.size)

    # Dimensions (create only those used by variables we define, matching Fortran behavior)
    ds.createDimension("nlambda", nlambda)
    ds.createDimension("xyz", 3)
    ds.createDimension("ntheta_plasma", ntheta_plasma)
    ds.createDimension("nzeta_plasma", nzeta_plasma)
    ds.createDimension("nzetal_plasma", nzetal_plasma)
    ds.createDimension("ntheta_coil", ntheta_coil)
    ds.createDimension("nzeta_coil", nzeta_coil)
    ds.createDimension("nzetal_coil", nzetal_coil)
    ds.createDimension("mnmax_potential", mnmax_potential)
    ds.createDimension("mnmax_plasma", mnmax_plasma)
    ds.createDimension("mnmax_coil", mnmax_coil)
    ds.createDimension("ntheta_nzeta_plasma", ntheta_plasma * nzeta_plasma)
    ds.createDimension("num_basis_functions", nb)

    # Scalars (names match regcoil_write_output.f90)
    ds.createVariable("nfp", "i4")[...] = int(nfp)
    ds.createVariable("geometry_option_plasma", "i4")[...] = int(inputs.get("geometry_option_plasma", 0))
    ds.createVariable("geometry_option_coil", "i4")[...] = int(inputs.get("geometry_option_coil", 0))
    ds.createVariable("ntheta_plasma", "i4")[...] = int(ntheta_plasma)
    ds.createVariable("nzeta_plasma", "i4")[...] = int(nzeta_plasma)
    ds.createVariable("nzetal_plasma", "i4")[...] = int(nzetal_plasma)
    ds.createVariable("ntheta_coil", "i4")[...] = int(ntheta_coil)
    ds.createVariable("nzeta_coil", "i4")[...] = int(nzeta_coil)
    ds.createVariable("nzetal_coil", "i4")[...] = int(nzetal_coil)

    # The Fortran code writes these torus scalars regardless of geometry option (VMEC cases keep defaults
    # unless the Fortran initializer overwrites them).
    ds.createVariable("a_plasma", "f8")[...] = float(inputs.get("a_plasma", defaults["a_plasma"]))
    ds.createVariable("a_coil", "f8")[...] = float(inputs.get("a_coil", defaults["a_coil"]))
    ds.createVariable("R0_plasma", "f8")[...] = float(mats.get("R0_plasma", inputs.get("r0_plasma", defaults["r0_plasma"])))
    ds.createVariable("R0_coil", "f8")[...] = float(inputs.get("r0_coil", defaults["r0_coil"]))

    ds.createVariable("mpol_potential", "i4")[...] = int(inputs.get("mpol_potential", mats.get("mpol_potential", defaults["mpol_potential"])))
    ds.createVariable("ntor_potential", "i4")[...] = int(inputs.get("ntor_potential", mats.get("ntor_potential", defaults["ntor_potential"])))
    ds.createVariable("mnmax_potential", "i4")[...] = int(mnmax_potential)
    ds.createVariable("mnmax_plasma", "i4")[...] = int(mnmax_plasma)
    ds.createVariable("mnmax_coil", "i4")[...] = int(mnmax_coil)
    ds.createVariable("num_basis_functions", "i4")[...] = int(nb)
    ds.createVariable("symmetry_option", "i4")[...] = int(inputs.get("symmetry_option", mats.get("symmetry_option", defaults["symmetry_option"])))

    if "area_plasma" in mats:
        ds.createVariable("area_plasma", "f8")[...] = float(np.asarray(mats["area_plasma"]))
    if "area_coil" in mats:
        ds.createVariable("area_coil", "f8")[...] = float(np.asarray(mats["area_coil"]))
    if "volume_plasma" in mats:
        ds.createVariable("volume_plasma", "f8")[...] = float(np.asarray(mats["volume_plasma"]))
    if "volume_coil" in mats:
        ds.createVariable("volume_coil", "f8")[...] = float(np.asarray(mats["volume_coil"]))

    ds.createVariable("net_poloidal_current_Amperes", "f8")[...] = float(
        mats.get("net_poloidal_current_Amperes", inputs.get("net_poloidal_current_amperes", defaults["net_poloidal_current_amperes"]))
    )
    ds.createVariable("net_toroidal_current_Amperes", "f8")[...] = float(
        mats.get("net_toroidal_current_Amperes", inputs.get("net_toroidal_current_amperes", defaults["net_toroidal_current_amperes"]))
    )
    ds.createVariable("curpol", "f8")[...] = float(inputs.get("curpol", defaults["curpol"]))
    ds.createVariable("nlambda", "i4")[...] = int(nlambda)
    ds.createVariable("total_time", "f8")[...] = float(total_time) if total_time is not None else np.nan
    ds.createVariable("exit_code", "i4")[...] = int(exit_code)
    ds.createVariable("sensitivity_option", "i4")[...] = int(inputs.get("sensitivity_option", defaults["sensitivity_option"]))

    if general_option in (4, 5):
        # Match regcoil_auto_regularization_solve.f90:
        #  - exit_code==0: chi2_B_target = chi2_B(Nlambda) (final/accepted lambda)
        #  - exit_code==-2/-3: chi2_B_target = chi2_B(1) ('worst' achieved chi2_B at infinite regularization)
        if nlambda <= 0:
            chi2_B_target = 0.0
        elif int(exit_code) in (-2, -3):
            chi2_B_target = float(np.asarray(chi2_B[0]))
        else:
            chi2_B_target = float(np.asarray(chi2_B[-1]))
        ds.createVariable("chi2_B_target", "f8")[...] = chi2_B_target

    # Arrays (dimension 1)
    ds.createVariable("theta_plasma", "f8", ("ntheta_plasma",))[:] = theta_p
    ds.createVariable("zeta_plasma", "f8", ("nzeta_plasma",))[:] = zeta_p
    dz = 2.0 * np.pi / nfp
    zetal = np.concatenate([zeta_p + j * dz for j in range(nfp)], axis=0)
    ds.createVariable("zetal_plasma", "f8", ("nzetal_plasma",))[:] = zetal

    ds.createVariable("theta_coil", "f8", ("ntheta_coil",))[:] = theta_c
    ds.createVariable("zeta_coil", "f8", ("nzeta_coil",))[:] = zeta_c
    zetal = np.concatenate([zeta_c + j * dz for j in range(nfp)], axis=0)
    ds.createVariable("zetal_coil", "f8", ("nzetal_coil",))[:] = zetal

    ds.createVariable("xm_potential", "i4", ("mnmax_potential",))[:] = xm_potential
    ds.createVariable("xn_potential", "i4", ("mnmax_potential",))[:] = xn_potential
    ds.createVariable("xm_plasma", "i4", ("mnmax_plasma",))[:] = xm_plasma
    ds.createVariable("xn_plasma", "i4", ("mnmax_plasma",))[:] = xn_plasma
    ds.createVariable("xm_coil", "i4", ("mnmax_coil",))[:] = xm_coil
    ds.createVariable("xn_coil", "i4", ("mnmax_coil",))[:] = xn_coil

    ds.createVariable("rmnc_plasma", "f8", ("mnmax_plasma",))[:] = np.asarray(mats.get("rmnc_plasma", np.zeros((mnmax_plasma,))), dtype=float)
    if lasym:
        ds.createVariable("rmns_plasma", "f8", ("mnmax_plasma",))[:] = np.asarray(mats.get("rmns_plasma", np.zeros((mnmax_plasma,))), dtype=float)
        ds.createVariable("zmnc_plasma", "f8", ("mnmax_plasma",))[:] = np.asarray(mats.get("zmnc_plasma", np.zeros((mnmax_plasma,))), dtype=float)
    ds.createVariable("zmns_plasma", "f8", ("mnmax_plasma",))[:] = np.asarray(mats.get("zmns_plasma", np.zeros((mnmax_plasma,))), dtype=float)

    ds.createVariable("rmnc_coil", "f8", ("mnmax_coil",))[:] = np.asarray(mats.get("rmnc_coil", np.zeros((mnmax_coil,))), dtype=float)
    if lasym:
        ds.createVariable("rmns_coil", "f8", ("mnmax_coil",))[:] = np.asarray(mats.get("rmns_coil", np.zeros((mnmax_coil,))), dtype=float)
        ds.createVariable("zmnc_coil", "f8", ("mnmax_coil",))[:] = np.asarray(mats.get("zmnc_coil", np.zeros((mnmax_coil,))), dtype=float)
    ds.createVariable("zmns_coil", "f8", ("mnmax_coil",))[:] = np.asarray(mats.get("zmns_coil", np.zeros((mnmax_coil,))), dtype=float)

    # h is a flattened plasma-grid vector with theta varying fastest (Fortran indexing).
    if "h" in mats:
        h = np.asarray(mats["h"], dtype=float).reshape(ntheta_plasma, nzeta_plasma).T.reshape(-1)
    else:
        # Fall back to Bnet*|N| if needed.
        Bnet = np.asarray(mats.get("Bnet", np.zeros((ntheta_plasma, nzeta_plasma))), dtype=float)
        normN_p = np.asarray(mats.get("normN_plasma", np.ones((ntheta_plasma, nzeta_plasma))), dtype=float)
        h = (Bnet * normN_p).T.reshape(-1)
    ds.createVariable("h", "f8", ("ntheta_nzeta_plasma",))[:] = h

    ds.createVariable("RHS_B", "f8", ("num_basis_functions",))[:] = np.asarray(mats.get("RHS_B", np.zeros((nb,))), dtype=float)
    ds.createVariable("RHS_regularization", "f8", ("num_basis_functions",))[:] = np.asarray(
        mats.get("RHS_regularization", mats.get("RHS_reg", np.zeros((nb,)))), dtype=float
    )

    ds.createVariable("lambda", "f8", ("nlambda",))[:] = np.asarray(lambdas, dtype=float)
    ds.createVariable("chi2_B", "f8", ("nlambda",))[:] = np.asarray(chi2_B, dtype=float)
    ds.createVariable("chi2_K", "f8", ("nlambda",))[:] = np.asarray(chi2_K, dtype=float)
    # This is always present in Fortran output.
    ds.createVariable("chi2_Laplace_Beltrami", "f8", ("nlambda",))[:] = np.asarray(mats.get("chi2_Laplace_Beltrami", np.full((nlambda,), np.nan)), dtype=float)
    ds.createVariable("max_Bnormal", "f8", ("nlambda",))[:] = np.asarray(max_B, dtype=float)
    ds.createVariable("max_K", "f8", ("nlambda",))[:] = np.asarray(max_K, dtype=float)

    if target_option == "max_K_lse":
        ds.createVariable("max_K_lse", "f8", ("nlambda",))[:] = np.asarray(mats.get("max_K_lse", np.full((nlambda,), np.nan)), dtype=float)
    if target_option == "lp_norm_K":
        ds.createVariable("lp_norm_K", "f8", ("nlambda",))[:] = np.asarray(mats.get("lp_norm_K", np.full((nlambda,), np.nan)), dtype=float)

    # Arrays (dimension 2)
    normN_p = np.asarray(mats.get("normN_plasma"), dtype=float)
    normN_c = np.asarray(mats.get("normN_coil"), dtype=float)
    ds.createVariable("norm_normal_plasma", "f8", ("nzeta_plasma", "ntheta_plasma"))[:] = normN_p.T
    ds.createVariable("norm_normal_coil", "f8", ("nzeta_coil", "ntheta_coil"))[:] = normN_c.T

    Bplasma = np.asarray(mats.get("Bplasma"), dtype=float)
    Bnet = np.asarray(mats.get("Bnet"), dtype=float)
    ds.createVariable("Bnormal_from_plasma_current", "f8", ("nzeta_plasma", "ntheta_plasma"))[:] = Bplasma.T
    ds.createVariable("Bnormal_from_net_coil_currents", "f8", ("nzeta_plasma", "ntheta_plasma"))[:] = Bnet.T

    if save_level < 1 and ("inductance" in mats):
        # Fortran dims are (ntheta_nzeta_plasma, ntheta_nzeta_coil), so the coil dimension is created only here.
        ds.createDimension("ntheta_nzeta_coil", ntheta_coil * nzeta_coil)
        ind = np.asarray(mats["inductance"], dtype=float)  # (Np,Nc) with theta-fastest ordering on each side
        ind_zt = ind.reshape(ntheta_plasma, nzeta_plasma, ntheta_coil * nzeta_coil).transpose(1, 0, 2).reshape(
            ntheta_plasma * nzeta_plasma, ntheta_coil, nzeta_coil
        ).transpose(0, 2, 1).reshape(ntheta_plasma * nzeta_plasma, ntheta_coil * nzeta_coil)
        ds.createVariable("inductance", "f8", ("ntheta_nzeta_plasma", "ntheta_nzeta_coil"))[:] = ind_zt

    if save_level < 2 and ("g" in mats):
        g = np.asarray(mats["g"], dtype=float)  # (Np,nb)
        g_zt = g.reshape(ntheta_plasma, nzeta_plasma, nb).transpose(1, 0, 2).reshape(ntheta_plasma * nzeta_plasma, nb)
        ds.createVariable("g", "f8", ("ntheta_nzeta_plasma", "num_basis_functions"))[:] = g_zt

    # single_valued_current_potential_mn: Fortran uses (num_basis_functions, nlambda) but appears as (nlambda, num_basis_functions).
    ds.createVariable("single_valued_current_potential_mn", "f8", ("nlambda", "num_basis_functions"))[:] = np.asarray(sols, dtype=float)

    # Optional fields for improved parity & testability.
    # Arrays (dimension 3)
    # r_plasma and r_coil always exist in Fortran output.
    r3tz = np.asarray(mats.get("r_plasma"), dtype=float)  # (3,T,Z)
    r_tz3 = np.moveaxis(r3tz, 0, -1)  # (T,Z,3)
    r_zt3 = r_tz3.transpose(1, 0, 2)  # (Z,T,3)
    r_full = np.concatenate([_rotate_z_np(r_zt3, (2.0 * np.pi / nfp) * j) for j in range(nfp)], axis=0)
    ds.createVariable("r_plasma", "f8", ("nzetal_plasma", "ntheta_plasma", "xyz"))[:] = r_full

    r3tz = np.asarray(mats.get("r_coil"), dtype=float)
    r_tz3 = np.moveaxis(r3tz, 0, -1)
    r_zt3 = r_tz3.transpose(1, 0, 2)
    r_full = np.concatenate([_rotate_z_np(r_zt3, (2.0 * np.pi / nfp) * j) for j in range(nfp)], axis=0)
    ds.createVariable("r_coil", "f8", ("nzetal_coil", "ntheta_coil", "xyz"))[:] = r_full

    if save_level < 3:
        for name, key, dims in [
            ("drdtheta_plasma", "drdtheta_plasma", ("nzetal_plasma", "ntheta_plasma", "xyz")),
            ("drdzeta_plasma", "drdzeta_plasma", ("nzetal_plasma", "ntheta_plasma", "xyz")),
            ("normal_plasma", "normal_plasma", ("nzetal_plasma", "ntheta_plasma", "xyz")),
            ("drdtheta_coil", "drdtheta_coil", ("nzetal_coil", "ntheta_coil", "xyz")),
            ("drdzeta_coil", "drdzeta_coil", ("nzetal_coil", "ntheta_coil", "xyz")),
            ("normal_coil", "normal_coil", ("nzetal_coil", "ntheta_coil", "xyz")),
        ]:
            arr3tz = np.asarray(mats.get(key))
            a_tz3 = np.moveaxis(arr3tz, 0, -1).transpose(1, 0, 2)  # (Z,T,3)
            a_full = np.concatenate([_rotate_z_np(a_tz3, (2.0 * np.pi / nfp) * j) for j in range(nfp)], axis=0)
            ds.createVariable(name, "f8", dims)[:] = a_full

    # Bnormal_total for each lambda: (nlambda, nzeta, ntheta)
    g_over_Np = mats["g_over_Np"]
    Btarget = (mats["Bplasma"] + mats["Bnet"]).reshape(-1)
    Bsv = np.asarray(sols @ g_over_Np.T)  # (nlambda, Np)
    Btot = Bsv + np.asarray(Btarget)[None, :]
    Btot = Btot.reshape(nlambda, ntheta_plasma, nzeta_plasma).transpose(0, 2, 1)
    ds.createVariable("Bnormal_total", "f8", ("nlambda", "nzeta_plasma", "ntheta_plasma"))[:] = Btot

    # Current potential (single-valued + secular terms)
    basis = np.asarray(mats["basis"])  # (Ncoil, nb)
    phi_sv_flat = np.asarray(sols @ basis.T)  # (nlambda, Ncoil)
    phi_sv = phi_sv_flat.reshape(nlambda, ntheta_coil, nzeta_coil).transpose(0, 2, 1)  # (nlambda,nzeta,ntheta)
    net_pol = float(mats.get("net_poloidal_current_Amperes", 0.0))
    net_tor = float(mats.get("net_toroidal_current_Amperes", 0.0))
    factor_zeta = net_pol / (2.0 * np.pi)
    factor_theta = net_tor / (2.0 * np.pi)
    phi_total = phi_sv + factor_zeta * zeta_c[None, :, None] + factor_theta * theta_c[None, None, :]
    ds.createVariable("single_valued_current_potential_thetazeta", "f8", ("nlambda", "nzeta_coil", "ntheta_coil"))[:] = phi_sv
    ds.createVariable("current_potential", "f8", ("nlambda", "nzeta_coil", "ntheta_coil"))[:] = phi_total

    # K2 field
    fx = mats["fx"]; fy = mats["fy"]; fz = mats["fz"]
    dxv = mats["dx"]; dyv = mats["dy"]; dzv = mats["dz"]
    normNc = mats["normNc"]
    Kdx = np.asarray(dxv)[None, :] - np.asarray(sols @ fx.T)
    Kdy = np.asarray(dyv)[None, :] - np.asarray(sols @ fy.T)
    Kdz = np.asarray(dzv)[None, :] - np.asarray(sols @ fz.T)
    K2 = (Kdx * Kdx + Kdy * Kdy + Kdz * Kdz) / (np.asarray(normNc)[None, :] ** 2)
    K2 = K2.reshape(nlambda, ntheta_coil, nzeta_coil).transpose(0, 2, 1)
    ds.createVariable("K2", "f8", ("nlambda", "nzeta_coil", "ntheta_coil"))[:] = K2

    # Laplace–Beltrami diagnostics field
    flb = mats["flb"]
    dLB = mats["d_Laplace_Beltrami"]
    dth_c = float(mats["dth_c"]); dze_c = float(mats["dze_c"])
    KLB = np.asarray(dLB)[None, :] - np.asarray(sols @ np.asarray(flb).T)  # (nlambda, Ncoil)
    LB2_times_N = (KLB * KLB) / np.asarray(normNc)[None, :]
    chi2_LB = nfp * dth_c * dze_c * np.sum(LB2_times_N, axis=1)
    ds.variables["chi2_Laplace_Beltrami"][:] = chi2_LB
    LB2 = (LB2_times_N / np.asarray(normNc)[None, :]).reshape(nlambda, ntheta_coil, nzeta_coil).transpose(0, 2, 1)
    ds.createVariable("Laplace_Beltrami2", "f8", ("nlambda", "nzeta_coil", "ntheta_coil"))[:] = LB2

    ds.setncattr("regcoil_jax_version", "0.3")
    ds.close()
