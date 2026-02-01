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
    nb = sols.shape[1]

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

    ds.createDimension("nlambda", nlambda)
    ds.createDimension("nbasis", nb)
    ds.createDimension("xyz", 3)

    if ntheta_plasma is not None and nzeta_plasma is not None:
        ds.createDimension("ntheta_plasma", ntheta_plasma)
        ds.createDimension("nzeta_plasma", nzeta_plasma)
        ds.createDimension("nzetal_plasma", nzeta_plasma * nfp)
    if ntheta_coil is not None and nzeta_coil is not None:
        ds.createDimension("ntheta_coil", ntheta_coil)
        ds.createDimension("nzeta_coil", nzeta_coil)
        ds.createDimension("nzetal_coil", nzeta_coil * nfp)

    # Scalars
    if "nfp" in mats:
        ds.createVariable("nfp", "i4")[...] = int(mats["nfp"])
    ds.createVariable("nlambda", "i4")[...] = int(nlambda)
    ds.createVariable("exit_code", "i4")[...] = int(exit_code)
    if chosen_idx is not None:
        ds.createVariable("chosen_idx", "i4")[...] = int(chosen_idx)
    if "net_poloidal_current_Amperes" in mats:
        ds.createVariable("net_poloidal_current_Amperes", "f8")[...] = float(mats["net_poloidal_current_Amperes"])
    if "net_toroidal_current_Amperes" in mats:
        ds.createVariable("net_toroidal_current_Amperes", "f8")[...] = float(mats["net_toroidal_current_Amperes"])
    if "curpol" in inputs:
        ds.createVariable("curpol", "f8")[...] = float(inputs["curpol"])
    if "area_plasma" in mats:
        ds.createVariable("area_plasma", "f8")[...] = float(np.asarray(mats["area_plasma"]))
    if "area_coil" in mats:
        ds.createVariable("area_coil", "f8")[...] = float(np.asarray(mats["area_coil"]))

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

    vsol = ds.createVariable("single_valued_current_potential_mn", "f8", ("nlambda","nbasis"))
    vsol[:] = np.asarray(sols)

    # Optional fields for improved parity & testability.
    if ntheta_plasma is not None and nzeta_plasma is not None:
        normN_p = np.asarray(mats.get("normN_plasma", np.zeros((ntheta_plasma, nzeta_plasma))))
        ds.createVariable("norm_normal_plasma", "f8", ("nzeta_plasma", "ntheta_plasma"))[:] = normN_p.T

        Bplasma = np.asarray(mats.get("Bplasma", np.zeros((ntheta_plasma, nzeta_plasma))))
        ds.createVariable("Bnormal_from_plasma_current", "f8", ("nzeta_plasma", "ntheta_plasma"))[:] = Bplasma.T

        Bnet = np.asarray(mats.get("Bnet", np.zeros((ntheta_plasma, nzeta_plasma))))
        ds.createVariable("Bnormal_from_net_coil_currents", "f8", ("nzeta_plasma", "ntheta_plasma"))[:] = Bnet.T

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
        ds.createDimension("mnmax", xm.size)
        ds.createVariable("xm_potential","i4",("mnmax",))[:] = xm
        ds.createVariable("xn_potential","i4",("mnmax",))[:] = xn

    ds.setncattr("regcoil_jax_version", "0.2")
    ds.close()
