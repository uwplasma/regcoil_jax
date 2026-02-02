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
    chosen_idx: int | None = None,
    exit_code: int = 0,
):
    if netCDF4 is None:
        raise ImportError("netCDF4 is required to write REGCOIL output .nc files.")
    ds = netCDF4.Dataset(path, "w")
    nlambda = len(lambdas)
    nb = int(sols.shape[1])

    # Geometry / grid sizes (REGCOIL convention: theta, zeta grids for a single field period)
    theta_p = np.asarray(mats.get("theta_plasma", []))
    zeta_p = np.asarray(mats.get("zeta_plasma", []))
    theta_c = np.asarray(mats.get("theta_coil", []))
    zeta_c = np.asarray(mats.get("zeta_coil", []))
    ntheta_plasma = int(theta_p.size) if theta_p.size else None
    nzeta_plasma = int(zeta_p.size) if zeta_p.size else None
    ntheta_coil = int(theta_c.size) if theta_c.size else None
    nzeta_coil = int(zeta_c.size) if zeta_c.size else None
    nfp = int(mats.get("nfp", 1))

    def _def_dim(name: str, size: int | None):
        if size is None:
            return
        if name in ds.dimensions:
            return
        ds.createDimension(name, int(size))

    def _write_scalar(name: str, value, *, dtype="f8"):
        v = ds.createVariable(name, dtype)
        v[...] = value

    def _maybe_write_scalar(name: str, value, *, dtype="f8"):
        if value is None:
            return
        _write_scalar(name, value, dtype=dtype)

    def _maybe_write_1d(name: str, value, dim: str, *, dtype="f8"):
        if value is None:
            return
        arr = np.asarray(value)
        if arr.ndim != 1:
            return
        ds.createVariable(name, dtype, (dim,))[:] = arr

    # Predefine mode-count dimensions early so we can safely create scalar variables
    # with the same name later (matching the Fortran output structure).
    if "xm" in mats:
        try:
            _def_dim("mnmax_potential", int(np.asarray(mats["xm"]).size))
        except Exception:
            pass
    if mats.get("xm_plasma", None) is not None:
        try:
            _def_dim("mnmax_plasma", int(np.asarray(mats["xm_plasma"]).size))
        except Exception:
            pass
    if mats.get("xm_coil", None) is not None:
        try:
            _def_dim("mnmax_coil", int(np.asarray(mats["xm_coil"]).size))
        except Exception:
            pass

    ds.createDimension("nlambda", nlambda)
    ds.createDimension("xyz", 3)
    # Fortran REGCOIL naming:
    _def_dim("num_basis_functions", nb)
    # Back-compat: older regcoil_jax outputs used "nbasis".
    _def_dim("nbasis", nb)

    if ntheta_plasma is not None and nzeta_plasma is not None:
        _def_dim("ntheta_plasma", ntheta_plasma)
        _def_dim("nzeta_plasma", nzeta_plasma)
        _def_dim("nzetal_plasma", nzeta_plasma * nfp)
        # Flattened indexing dims used by Fortran output:
        _def_dim("ntheta_nzeta_plasma", ntheta_plasma * nzeta_plasma)
        _def_dim("ntheta_times_nzeta_plasma", ntheta_plasma * nzeta_plasma)
    if ntheta_coil is not None and nzeta_coil is not None:
        _def_dim("ntheta_coil", ntheta_coil)
        _def_dim("nzeta_coil", nzeta_coil)
        _def_dim("nzetal_coil", nzeta_coil * nfp)
        _def_dim("ntheta_nzeta_coil", ntheta_coil * nzeta_coil)
        _def_dim("ntheta_times_nzeta_coil", ntheta_coil * nzeta_coil)

    # Scalars
    _maybe_write_scalar("nfp", int(mats.get("nfp")) if "nfp" in mats else None, dtype="i4")
    _write_scalar("nlambda", int(nlambda), dtype="i4")
    _write_scalar("exit_code", int(exit_code), dtype="i4")
    if chosen_idx is not None:
        _write_scalar("chosen_idx", int(chosen_idx), dtype="i4")

    # Input geometry options (Fortran variables):
    if "geometry_option_plasma" in inputs:
        _write_scalar("geometry_option_plasma", int(inputs["geometry_option_plasma"]), dtype="i4")
    if "geometry_option_coil" in inputs:
        _write_scalar("geometry_option_coil", int(inputs["geometry_option_coil"]), dtype="i4")

    # Grid sizes
    if ntheta_plasma is not None:
        _write_scalar("ntheta_plasma", int(ntheta_plasma), dtype="i4")
    if nzeta_plasma is not None:
        _write_scalar("nzeta_plasma", int(nzeta_plasma), dtype="i4")
        _write_scalar("nzetal_plasma", int(nzeta_plasma * nfp), dtype="i4")
    if ntheta_coil is not None:
        _write_scalar("ntheta_coil", int(ntheta_coil), dtype="i4")
    if nzeta_coil is not None:
        _write_scalar("nzeta_coil", int(nzeta_coil), dtype="i4")
        _write_scalar("nzetal_coil", int(nzeta_coil * nfp), dtype="i4")

    # Torus parameters (when present)
    _maybe_write_scalar("R0_plasma", float(inputs["r0_plasma"]) if "r0_plasma" in inputs else None)
    _maybe_write_scalar("a_plasma", float(inputs["a_plasma"]) if "a_plasma" in inputs else None)
    _maybe_write_scalar("R0_coil", float(inputs["r0_coil"]) if "r0_coil" in inputs else None)
    _maybe_write_scalar("a_coil", float(inputs["a_coil"]) if "a_coil" in inputs else None)

    # Basis / mode sizes
    _maybe_write_scalar("mpol_potential", int(mats.get("mpol_potential")) if "mpol_potential" in mats else None, dtype="i4")
    _maybe_write_scalar("ntor_potential", int(mats.get("ntor_potential")) if "ntor_potential" in mats else None, dtype="i4")
    _maybe_write_scalar("mnmax_potential", int(mats.get("mnmax_potential")) if "mnmax_potential" in mats else None, dtype="i4")
    _maybe_write_scalar("num_basis_functions", int(mats.get("num_basis_functions")) if "num_basis_functions" in mats else None, dtype="i4")
    _maybe_write_scalar("symmetry_option", int(mats.get("symmetry_option")) if "symmetry_option" in mats else None, dtype="i4")
    _maybe_write_scalar("save_level", int(mats.get("save_level")) if "save_level" in mats else None, dtype="i4")

    # Physics scalars
    _maybe_write_scalar(
        "net_poloidal_current_Amperes",
        float(mats.get("net_poloidal_current_Amperes")) if "net_poloidal_current_Amperes" in mats else None,
    )
    _maybe_write_scalar(
        "net_toroidal_current_Amperes",
        float(mats.get("net_toroidal_current_Amperes")) if "net_toroidal_current_Amperes" in mats else None,
    )
    _maybe_write_scalar("curpol", float(inputs["curpol"]) if "curpol" in inputs else None)
    _maybe_write_scalar("area_plasma", float(np.asarray(mats["area_plasma"])) if "area_plasma" in mats else None)
    _maybe_write_scalar("area_coil", float(np.asarray(mats["area_coil"])) if "area_coil" in mats else None)
    _maybe_write_scalar("volume_plasma", float(np.asarray(mats["volume_plasma"])) if "volume_plasma" in mats else None)
    _maybe_write_scalar("volume_coil", float(np.asarray(mats["volume_coil"])) if "volume_coil" in mats else None)

    # Grids
    if ntheta_plasma is not None:
        ds.createVariable("theta_plasma", "f8", ("ntheta_plasma",))[:] = theta_p
    if nzeta_plasma is not None:
        ds.createVariable("zeta_plasma", "f8", ("nzeta_plasma",))[:] = zeta_p
        # Full toroidal range (all field periods), matching REGCOIL's `zetal_*` outputs.
        dz = 2.0 * np.pi / nfp
        zetal = np.concatenate([zeta_p + j * dz for j in range(nfp)], axis=0)
        ds.createVariable("zetal_plasma", "f8", ("nzetal_plasma",))[:] = zetal
    if ntheta_coil is not None:
        ds.createVariable("theta_coil", "f8", ("ntheta_coil",))[:] = theta_c
    if nzeta_coil is not None:
        ds.createVariable("zeta_coil", "f8", ("nzeta_coil",))[:] = zeta_c
        dz = 2.0 * np.pi / nfp
        zetal = np.concatenate([zeta_c + j * dz for j in range(nfp)], axis=0)
        ds.createVariable("zetal_coil", "f8", ("nzetal_coil",))[:] = zetal

    vlam = ds.createVariable("lambda", "f8", ("nlambda",))
    vlam[:] = np.asarray(lambdas)
    ds.createVariable("chi2_B","f8",("nlambda",))[:] = np.asarray(chi2_B)
    ds.createVariable("chi2_K","f8",("nlambda",))[:] = np.asarray(chi2_K)
    ds.createVariable("max_Bnormal","f8",("nlambda",))[:] = np.asarray(max_B)
    ds.createVariable("max_K","f8",("nlambda",))[:] = np.asarray(max_K)

    vsol = ds.createVariable("single_valued_current_potential_mn", "f8", ("nlambda","num_basis_functions"))
    vsol[:] = np.asarray(sols)

    # Optional fields for improved parity & testability.
    if ntheta_plasma is not None and nzeta_plasma is not None:
        normN_p = np.asarray(mats.get("normN_plasma", np.zeros((ntheta_plasma, nzeta_plasma))))
        ds.createVariable("norm_normal_plasma", "f8", ("nzeta_plasma", "ntheta_plasma"))[:] = normN_p.T

        Bplasma = np.asarray(mats.get("Bplasma", np.zeros((ntheta_plasma, nzeta_plasma))))
        ds.createVariable("Bnormal_from_plasma_current", "f8", ("nzeta_plasma", "ntheta_plasma"))[:] = Bplasma.T

        Bnet = np.asarray(mats.get("Bnet", np.zeros((ntheta_plasma, nzeta_plasma))))
        ds.createVariable("Bnormal_from_net_coil_currents", "f8", ("nzeta_plasma", "ntheta_plasma"))[:] = Bnet.T

        # Fortran also saves h (flattened) which satisfies:
        #   Bnormal_from_net_coil_currents = reshape(h,(ntheta,nzeta)) / norm_normal_plasma
        try:
            h_tz = Bnet * normN_p
            h_flat = h_tz.T.reshape(-1)  # (ntheta_nzeta_plasma,) with theta fastest
            ds.createVariable("h", "f8", ("ntheta_nzeta_plasma",))[:] = h_flat
        except Exception:
            pass

        _maybe_write_1d("RHS_B", mats.get("RHS_B", None), "num_basis_functions")
        _maybe_write_1d("RHS_regularization", mats.get("RHS_regularization", mats.get("RHS_reg", None)), "num_basis_functions")

        # Optional heavy matrices (match REGCOIL save_level behavior):
        # - inductance: (ntheta_nzeta_plasma, ntheta_nzeta_coil) (only if save_level<1 in Fortran)
        # - g:          (ntheta_nzeta_plasma, num_basis_functions) (only if save_level<2 in Fortran)
        save_level = int(mats.get("save_level", 3))
        if save_level < 1 and ("inductance" in mats) and (ntheta_coil is not None) and (nzeta_coil is not None):
            try:
                ind = np.asarray(mats["inductance"], dtype=float)  # (Np,Nc) in regcoil_jax flattening
                T = int(ntheta_plasma); Z = int(nzeta_plasma)
                Tc = int(ntheta_coil); Zc = int(nzeta_coil)
                # Reorder both plasma rows and coil columns to match Fortran's (izeta,itheta) indexing.
                ind_zt = ind.reshape(T, Z, Tc * Zc).transpose(1, 0, 2).reshape(T * Z, Tc, Zc).transpose(0, 2, 1).reshape(T * Z, Tc * Zc)
                ds.createVariable("inductance", "f8", ("ntheta_nzeta_plasma", "ntheta_nzeta_coil"))[:] = ind_zt
            except Exception:
                pass

        if save_level < 2 and ("g" in mats):
            try:
                g = np.asarray(mats["g"], dtype=float)  # (Np,nb) in regcoil_jax flattening
                T = int(ntheta_plasma); Z = int(nzeta_plasma)
                nb2 = int(g.shape[1])
                if nb2 == nb:
                    g_zt = g.reshape(T, Z, nb).transpose(1, 0, 2).reshape(T * Z, nb)
                    ds.createVariable("g", "f8", ("ntheta_nzeta_plasma", "num_basis_functions"))[:] = g_zt
            except Exception:
                pass

        # Plasma surface geometry on all nfp field periods.
        try:
            r3tz = np.asarray(mats["r_plasma"])  # (3,ntheta,nzeta)
            r_tz3 = np.moveaxis(r3tz, 0, -1)  # (ntheta,nzeta,3)
            r_zt3 = r_tz3.transpose(1, 0, 2)  # (nzeta,ntheta,3)

            r_full = np.concatenate([_rotate_z_np(r_zt3, (2.0 * np.pi / nfp) * j) for j in range(nfp)], axis=0)
            ds.createVariable("r_plasma", "f8", ("nzetal_plasma", "ntheta_plasma", "xyz"))[:] = r_full

            drdth = np.asarray(mats["drdtheta_plasma"])
            drdze = np.asarray(mats["drdzeta_plasma"])
            nvec = np.asarray(mats["normal_plasma"])
            for name, arr3tz in [
                ("drdtheta_plasma", drdth),
                ("drdzeta_plasma", drdze),
                ("normal_plasma", nvec),
            ]:
                a_tz3 = np.moveaxis(arr3tz, 0, -1).transpose(1, 0, 2)  # (nzeta,ntheta,3)
                a_full = np.concatenate(
                    [_rotate_z_np(a_tz3, (2.0 * np.pi / nfp) * j) for j in range(nfp)], axis=0
                )
                ds.createVariable(name, "f8", ("nzetal_plasma", "ntheta_plasma", "xyz"))[:] = a_full
        except Exception:
            pass

        # Bnormal_total for each lambda: (nlambda, nzeta, ntheta) to match REGCOIL output ordering.
        try:
            g_over_Np = mats["g_over_Np"]
            Btarget = (mats["Bplasma"] + mats["Bnet"]).reshape(-1)
            Bsv = np.asarray(sols @ g_over_Np.T)  # (nlambda, Np)
            Btot = Bsv + np.asarray(Btarget)[None, :]
            Btot = Btot.reshape(nlambda, ntheta_plasma, nzeta_plasma).transpose(0, 2, 1)
            ds.createVariable("Bnormal_total", "f8", ("nlambda", "nzeta_plasma", "ntheta_plasma"))[:] = Btot
        except Exception:
            pass

    if ntheta_coil is not None and nzeta_coil is not None:
        normN_c = np.asarray(mats.get("normN_coil", np.zeros((ntheta_coil, nzeta_coil))))
        ds.createVariable("norm_normal_coil", "f8", ("nzeta_coil", "ntheta_coil"))[:] = normN_c.T

        # Coil surface geometry on all nfp field periods.
        try:
            r3tz = np.asarray(mats["r_coil"])  # (3,ntheta,nzeta)
            r_tz3 = np.moveaxis(r3tz, 0, -1)  # (ntheta,nzeta,3)
            r_zt3 = r_tz3.transpose(1, 0, 2)  # (nzeta,ntheta,3)

            r_full = np.concatenate([_rotate_z_np(r_zt3, (2.0 * np.pi / nfp) * j) for j in range(nfp)], axis=0)
            ds.createVariable("r_coil", "f8", ("nzetal_coil", "ntheta_coil", "xyz"))[:] = r_full

            drdth = np.asarray(mats["drdtheta_coil"])
            drdze = np.asarray(mats["drdzeta_coil"])
            nvec = np.asarray(mats["normal_coil"])
            for name, arr3tz in [
                ("drdtheta_coil", drdth),
                ("drdzeta_coil", drdze),
                ("normal_coil", nvec),
            ]:
                a_tz3 = np.moveaxis(arr3tz, 0, -1).transpose(1, 0, 2)  # (nzeta,ntheta,3)
                a_full = np.concatenate(
                    [_rotate_z_np(a_tz3, (2.0 * np.pi / nfp) * j) for j in range(nfp)], axis=0
                )
                ds.createVariable(name, "f8", ("nzetal_coil", "ntheta_coil", "xyz"))[:] = a_full
        except Exception:
            pass

        # Current potential (single-valued + secular terms), matching regcoil_diagnostics.f90
        try:
            basis = np.asarray(mats["basis"])  # (Ncoil, nb)
            phi_sv_flat = np.asarray(sols @ basis.T)  # (nlambda, Ncoil)
            phi_sv = phi_sv_flat.reshape(nlambda, ntheta_coil, nzeta_coil).transpose(0, 2, 1)  # (nlambda,nzeta,ntheta)

            net_pol = float(mats.get("net_poloidal_current_Amperes", 0.0))
            net_tor = float(mats.get("net_toroidal_current_Amperes", 0.0))
            factor_zeta = net_pol / (2.0 * np.pi)
            factor_theta = net_tor / (2.0 * np.pi)
            phi_total = phi_sv + factor_zeta * zeta_c[None, :, None] + factor_theta * theta_c[None, None, :]

            ds.createVariable(
                "single_valued_current_potential_thetazeta",
                "f8",
                ("nlambda", "nzeta_coil", "ntheta_coil"),
            )[:] = phi_sv
            ds.createVariable("current_potential", "f8", ("nlambda", "nzeta_coil", "ntheta_coil"))[:] = phi_total
        except Exception:
            pass

        # K2 diagnostics field (so max_K can be validated from the file contents)
        try:
            fx = mats["fx"]; fy = mats["fy"]; fz = mats["fz"]
            dx = mats["dx"]; dy = mats["dy"]; dz = mats["dz"]
            normNc = mats["normNc"]
            Kdx = np.asarray(dx)[None, :] - np.asarray(sols @ fx.T)
            Kdy = np.asarray(dy)[None, :] - np.asarray(sols @ fy.T)
            Kdz = np.asarray(dz)[None, :] - np.asarray(sols @ fz.T)
            K2 = (Kdx * Kdx + Kdy * Kdy + Kdz * Kdz) / (np.asarray(normNc)[None, :] ** 2)
            K2 = K2.reshape(nlambda, ntheta_coil, nzeta_coil).transpose(0, 2, 1)
            ds.createVariable("K2", "f8", ("nlambda", "nzeta_coil", "ntheta_coil"))[:] = K2
        except Exception:
            pass

        # Laplace–Beltrami diagnostics field (to match REGCOIL outputs)
        try:
            flb = mats["flb"]
            dLB = mats["d_Laplace_Beltrami"]
            normNc = mats["normNc"]
            dth_c = float(mats["dth_c"])
            dze_c = float(mats["dze_c"])
            nfp = int(mats.get("nfp", 1))

            # KDifference_Laplace_Beltrami = d_LB - f_LB @ sol
            KLB = np.asarray(dLB)[None, :] - np.asarray(sols @ np.asarray(flb).T)  # (nlambda, Ncoil)
            LB2_times_N = (KLB * KLB) / np.asarray(normNc)[None, :]  # |KLB|^2 / |N|
            chi2_LB = nfp * dth_c * dze_c * np.sum(LB2_times_N, axis=1)
            ds.createVariable("chi2_Laplace_Beltrami", "f8", ("nlambda",))[:] = chi2_LB

            LB2 = LB2_times_N / np.asarray(normNc)[None, :]  # |KLB|^2
            LB2 = LB2.reshape(nlambda, ntheta_coil, nzeta_coil).transpose(0, 2, 1)
            ds.createVariable("Laplace_Beltrami2", "f8", ("nlambda", "nzeta_coil", "ntheta_coil"))[:] = LB2
        except Exception:
            pass

    # also store (xm,xn) for potential modes if available
    if "xm" in mats and "xn" in mats:
        xm = np.asarray(mats["xm"])
        xn = np.asarray(mats["xn"])
        _def_dim("mnmax_potential", int(xm.size))
        ds.createVariable("xm_potential","i4",("mnmax_potential",))[:] = xm
        ds.createVariable("xn_potential","i4",("mnmax_potential",))[:] = xn

    # Surface mode lists and coefficients (when available)
    for prefix, dim_name in [("plasma", "mnmax_plasma"), ("coil", "mnmax_coil")]:
        xm_key = f"xm_{prefix}"
        xn_key = f"xn_{prefix}"
        rmnc_key = f"rmnc_{prefix}"
        zmns_key = f"zmns_{prefix}"
        rmns_key = f"rmns_{prefix}"
        zmnc_key = f"zmnc_{prefix}"
        xm_val = mats.get(xm_key, None)
        xn_val = mats.get(xn_key, None)
        if xm_val is None or xn_val is None:
            continue
        xm_arr = np.asarray(xm_val)
        xn_arr = np.asarray(xn_val)
        _def_dim(dim_name, int(xm_arr.size))
        ds.createVariable(f"xm_{prefix}", "i4", (dim_name,))[:] = xm_arr.astype(np.int32)
        ds.createVariable(f"xn_{prefix}", "i4", (dim_name,))[:] = xn_arr.astype(np.int32)
        _maybe_write_1d(f"rmnc_{prefix}", mats.get(rmnc_key, None), dim_name)
        _maybe_write_1d(f"zmns_{prefix}", mats.get(zmns_key, None), dim_name)
        _maybe_write_1d(f"rmns_{prefix}", mats.get(rmns_key, None), dim_name)
        _maybe_write_1d(f"zmnc_{prefix}", mats.get(zmnc_key, None), dim_name)
        # Fortran exposes mnmax_* scalars too.
        _maybe_write_scalar(f"mnmax_{prefix}", int(xm_arr.size), dtype="i4")

    ds.setncattr("regcoil_jax_version", "0.2")
    ds.close()
