from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import netCDF4
except Exception:
    netCDF4 = None

def write_output_nc(path: str, inputs: dict, mats: dict, lambdas, sols, chi2_B, chi2_K, max_B, max_K):
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

    ds.createDimension("nlambda", nlambda)
    ds.createDimension("nbasis", nb)

    if ntheta_plasma is not None and nzeta_plasma is not None:
        ds.createDimension("ntheta_plasma", ntheta_plasma)
        ds.createDimension("nzeta_plasma", nzeta_plasma)
    if ntheta_coil is not None and nzeta_coil is not None:
        ds.createDimension("ntheta_coil", ntheta_coil)
        ds.createDimension("nzeta_coil", nzeta_coil)

    # Scalars
    if "nfp" in mats:
        ds.createVariable("nfp", "i4")[...] = int(mats["nfp"])
    if "area_plasma" in mats:
        ds.createVariable("area_plasma", "f8")[...] = float(np.asarray(mats["area_plasma"]))
    if "area_coil" in mats:
        ds.createVariable("area_coil", "f8")[...] = float(np.asarray(mats["area_coil"]))

    # Grids
    if ntheta_plasma is not None:
        ds.createVariable("theta_plasma", "f8", ("ntheta_plasma",))[:] = theta_p
    if nzeta_plasma is not None:
        ds.createVariable("zeta_plasma", "f8", ("nzeta_plasma",))[:] = zeta_p
    if ntheta_coil is not None:
        ds.createVariable("theta_coil", "f8", ("ntheta_coil",))[:] = theta_c
    if nzeta_coil is not None:
        ds.createVariable("zeta_coil", "f8", ("nzeta_coil",))[:] = zeta_c

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

    # also store (xm,xn) for potential modes if available
    if "xm" in mats and "xn" in mats:
        xm = np.asarray(mats["xm"])
        xn = np.asarray(mats["xn"])
        ds.createDimension("mnmax", xm.size)
        ds.createVariable("xm_potential","i4",("mnmax",))[:] = xm
        ds.createVariable("xn_potential","i4",("mnmax",))[:] = xn

    ds.setncattr("regcoil_jax_version", "0.2")
    ds.close()
