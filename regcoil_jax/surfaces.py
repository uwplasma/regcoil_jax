from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

from .grids import theta_grid, zeta_grid
from .geometry_torus import torus_xyz_and_derivs
from .geometry_fourier import FourierSurface, eval_surface_xyz_and_derivs
from .offset_surface import offset_surface_point
from .modes import init_fourier_modes
from .surface_metrics import metrics_and_normals

def _to_3TZ(xyz_TZ3):
    # (T,Z,3) -> (3,T,Z)
    return jnp.moveaxis(xyz_TZ3, -1, 0)

def plasma_surface_from_inputs(inputs, vmec_boundary: FourierSurface | None):
    ntheta = int(inputs["ntheta_plasma"]); nzeta = int(inputs["nzeta_plasma"])
    nfp = int(inputs.get("nfp_imposed", vmec_boundary.nfp if vmec_boundary is not None else 1))
    th = theta_grid(ntheta)
    ze = zeta_grid(nzeta, nfp)
    gopt = int(inputs["geometry_option_plasma"])
    if gopt == 0:
        # In the original Fortran, option 0 is used in some simple comparison inputs.
        # For parity-first, treat it as:
        #   - VMEC boundary if provided (wout_filename present / vmec_boundary not None)
        #   - otherwise analytic torus (same as option 1)
        if vmec_boundary is not None and inputs.get("wout_filename", None) is not None:
            gopt_eff = 2
        else:
            gopt_eff = 1
    else:
        gopt_eff = gopt

    if gopt_eff == 1:
        R0 = float(inputs["r0_plasma"]); a = float(inputs["a_plasma"])
        r, rth, rze, nunit, normN = torus_xyz_and_derivs(th, ze, R0, a)
    elif gopt_eff == 2:
        if vmec_boundary is None:
            raise ValueError("geometry_option_plasma=2 requires wout_filename / VMEC boundary.")
        # vmec_boundary already in jnp and has xn multiplied by nfp? in our loader it's not yet, do here:
        s = vmec_boundary
        # eval returns (T,Z,3)
        xyz, dth, dze = eval_surface_xyz_and_derivs(s, th, ze)
        r = _to_3TZ(xyz)
        rth = _to_3TZ(dth)
        rze = _to_3TZ(dze)
        _,_,_, nunit, normN = metrics_and_normals(rth, rze)
    else:
        raise NotImplementedError(f"geometry_option_plasma={gopt} not implemented yet (only 0,1,2).")
    return dict(nfp=nfp, theta=th, zeta=ze, r=r, rth=rth, rze=rze, nunit=nunit, normN=normN)

def coil_surface_from_inputs(inputs, plasma, vmec_boundary: FourierSurface | None):
    ntheta = int(inputs["ntheta_coil"]); nzeta = int(inputs["nzeta_coil"])
    nfp = int(plasma["nfp"])
    th = theta_grid(ntheta)
    ze = zeta_grid(nzeta, nfp)
    gopt = int(inputs["geometry_option_coil"])
    if gopt == 0:
        # In the original Fortran, option 0 is used in some simple comparison inputs.
        # For parity-first, treat it as:
        #   - VMEC boundary if provided (wout_filename present / vmec_boundary not None)
        #   - otherwise analytic torus (same as option 1)
        if vmec_boundary is not None and inputs.get("wout_filename", None) is not None:
            gopt_eff = 2
        else:
            gopt_eff = 1
    else:
        gopt_eff = gopt

    if gopt_eff == 1:
        R0 = float(inputs["r0_coil"]); a = float(inputs["a_coil"])
        r, rth, rze, nunit, normN = torus_xyz_and_derivs(th, ze, R0, a)
    elif gopt_eff == 2:
        if vmec_boundary is None:
            raise ValueError("geometry_option_coil=2 requires plasma surface from VMEC.")
        separation = float(inputs["separation"])
        s = vmec_boundary

        # Compute offset points in (x,y,z), then represent the resulting surface in the
        # same Fourier form used by REGCOIL (mpol/max_mpol_coil, ntor/max_ntor_coil),
        # and finally evaluate r, dr/dtheta, dr/dzeta from that Fourier representation.
        #
        # This matches regcoil_init_coil_surface.f90 for geometry_option_coil=2.
        def one_point(theta, zeta):
            return offset_surface_point(s, theta, zeta, separation, iters=12)

        v_theta = jax.vmap(lambda th_i: jax.vmap(lambda ze_j: one_point(th_i, ze_j))(ze))
        xyz_off = v_theta(th)  # (T,Z,3)

        major_R = jnp.sqrt(xyz_off[..., 0] * xyz_off[..., 0] + xyz_off[..., 1] * xyz_off[..., 1])
        z_coord = xyz_off[..., 2]

        max_mpol_coil = int(inputs.get("max_mpol_coil", 24))
        max_ntor_coil = int(inputs.get("max_ntor_coil", 24))
        mpol_coil = min(ntheta // 2, max_mpol_coil)
        ntor_coil = min(nzeta // 2, max_ntor_coil)

        # Build Fourier mode lists (include (0,0) here, unlike the potential basis).
        xm_np, xn_np = init_fourier_modes(mpol_coil, ntor_coil, include_00=True)
        xn_np = xn_np * nfp
        xm = jnp.asarray(xm_np, dtype=jnp.int32)
        xn = jnp.asarray(xn_np, dtype=jnp.int32)

        # Fourier transform the offset surface on the uniform grid.
        factor = 2.0 / (ntheta * nzeta)
        factor2 = jnp.full((xm.shape[0],), factor, dtype=major_R.dtype)
        if (ntheta % 2) == 0:
            factor2 = jnp.where(xm == (ntheta // 2), factor2 / 2.0, factor2)
        if (nzeta % 2) == 0:
            factor2 = jnp.where(jnp.abs(xn) == (nfp * (nzeta // 2)), factor2 / 2.0, factor2)

        ang = th[:, None, None] * xm[None, None, :] - ze[None, :, None] * xn[None, None, :]
        cosang = jnp.cos(ang)
        sinang = jnp.sin(ang)

        rmnc = jnp.sum(major_R[:, :, None] * cosang, axis=(0, 1)) * factor2
        rmns = jnp.sum(major_R[:, :, None] * sinang, axis=(0, 1)) * factor2
        zmnc = jnp.sum(z_coord[:, :, None] * cosang, axis=(0, 1)) * factor2
        zmns = jnp.sum(z_coord[:, :, None] * sinang, axis=(0, 1)) * factor2

        # Special-case (m,n)=(0,0) to match REGCOIL's normalization.
        rmnc = rmnc.at[0].set(jnp.mean(major_R))
        zmnc = zmnc.at[0].set(jnp.mean(z_coord))

        lasym = bool(getattr(vmec_boundary, "lasym", False))
        if not lasym:
            rmns = jnp.zeros_like(rmns)
            zmnc = jnp.zeros_like(zmnc)

        # Optional filtering (defaults match regcoil_variables.f90).
        mpol_coil_filter = int(inputs.get("mpol_coil_filter", max_mpol_coil))
        ntor_coil_filter = int(inputs.get("ntor_coil_filter", max_ntor_coil))
        keep = (jnp.abs(xm) <= mpol_coil_filter) & (jnp.abs(xn) <= (ntor_coil_filter * nfp))
        rmnc = jnp.where(keep, rmnc, 0.0)
        rmns = jnp.where(keep, rmns, 0.0)
        zmnc = jnp.where(keep, zmnc, 0.0)
        zmns = jnp.where(keep, zmns, 0.0)

        coil_surf = FourierSurface(
            nfp=nfp,
            lasym=lasym,
            xm=xm,
            xn=xn,
            rmnc=rmnc,
            zmns=zmns,
            rmns=rmns,
            zmnc=zmnc,
        )

        xyz, dth_xyz, dze_xyz = eval_surface_xyz_and_derivs(coil_surf, th, ze)
        r = _to_3TZ(xyz)
        rth = _to_3TZ(dth_xyz)
        rze = _to_3TZ(dze_xyz)
        _, _, _, nunit, normN = metrics_and_normals(rth, rze)
    else:
        raise NotImplementedError(f"geometry_option_coil={gopt} not implemented yet (only 0,1,2).")
    return dict(theta=th, zeta=ze, r=r, rth=rth, rze=rze, nunit=nunit, normN=normN)
