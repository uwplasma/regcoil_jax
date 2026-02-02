from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

from .grids import theta_grid, zeta_grid
from .geometry_torus import torus_xyz_and_derivs2
from .geometry_fourier import FourierSurface, eval_surface_xyz_and_derivs2
from .offset_surface import offset_surface_point
from .modes import init_fourier_modes
from .surface_metrics import metrics_and_normals
from .io_nescin import read_nescin_current_surface
from .io_surface_fourier_table import read_surface_fourier_table
from .io_focus import read_focus_surface
from .plasma_vmec_straight_fieldline import build_vmec_straight_fieldline_plasma_surface
from .io_efit import read_efit_gfile, efit_boundary_fourier

def _to_3TZ(xyz_TZ3):
    # (T,Z,3) -> (3,T,Z)
    return jnp.moveaxis(xyz_TZ3, -1, 0)


def _invert_monotonic_piecewise_linear(*, xp: jnp.ndarray, fp: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Piecewise-linear inverse map for a strictly-increasing xp."""
    xp = jnp.asarray(xp)
    fp = jnp.asarray(fp)
    x = jnp.asarray(x)
    # Find i such that xp[i] <= x < xp[i+1]
    i = jnp.searchsorted(xp, x, side="right") - 1
    i = jnp.clip(i, 0, xp.shape[0] - 2)
    x0 = xp[i]
    x1 = xp[i + 1]
    y0 = fp[i]
    y1 = fp[i + 1]
    t = (x - x0) / (x1 - x0 + 1e-300)
    return y0 + t * (y1 - y0)


def _interp_uniform_periodic_theta(*, data_t: jnp.ndarray, query_theta: jnp.ndarray) -> jnp.ndarray:
    """Linear interpolation on a uniform periodic theta grid in [0,2π)."""
    data_t = jnp.asarray(data_t)
    query_theta = jnp.asarray(query_theta)
    n = int(data_t.shape[0])
    dt = (2.0 * jnp.pi) / n
    x = (query_theta % (2.0 * jnp.pi)) / dt
    i0 = jnp.floor(x).astype(jnp.int32) % n
    i1 = (i0 + 1) % n
    t = x - jnp.floor(x)
    return (1.0 - t)[..., None] * data_t[i0] + t[..., None] * data_t[i1]

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
        r, rth, rze, rtt, rtz, rzz, nunit, normN = torus_xyz_and_derivs2(th, ze, R0, a)
        xm_plasma = jnp.asarray([0, 1], dtype=jnp.int32)
        xn_plasma = jnp.asarray([0, 0], dtype=jnp.int32)
        rmnc_plasma = jnp.asarray([R0, a], dtype=jnp.float64)
        zmns_plasma = jnp.asarray([0.0, a], dtype=jnp.float64)
        rmns_plasma = jnp.zeros_like(rmnc_plasma)
        zmnc_plasma = jnp.zeros_like(rmnc_plasma)
        lasym = False
    elif gopt_eff in (2, 3):
        if vmec_boundary is None:
            raise ValueError("geometry_option_plasma in {2,3} requires wout_filename / VMEC boundary.")
        # vmec_boundary already in jnp and has xn multiplied by nfp? in our loader it's not yet, do here:
        s = vmec_boundary
        # eval returns (T,Z,3)
        xyz, dth, dze, d2th2, d2thze, d2ze2 = eval_surface_xyz_and_derivs2(s, th, ze)
        r = _to_3TZ(xyz)
        rth = _to_3TZ(dth)
        rze = _to_3TZ(dze)
        rtt = _to_3TZ(d2th2)
        rtz = _to_3TZ(d2thze)
        rzz = _to_3TZ(d2ze2)
        _,_,_, nunit, normN = metrics_and_normals(rth, rze)
        xm_plasma = s.xm
        xn_plasma = s.xn
        rmnc_plasma = s.rmnc
        zmns_plasma = s.zmns
        rmns_plasma = s.rmns
        zmnc_plasma = s.zmnc
        lasym = bool(s.lasym)
    elif gopt_eff == 4:
        # VMEC, straight-field-line poloidal coordinate (regcoil_init_plasma_mod.f90 case(4)).
        wout = inputs.get("wout_filename", None)
        if wout is None:
            raise ValueError("geometry_option_plasma=4 requires wout_filename")
        surf_np = build_vmec_straight_fieldline_plasma_surface(
            wout_filename=str(wout),
            mpol_transform_refinement=float(inputs.get("mpol_transform_refinement", 5.0)),
            ntor_transform_refinement=float(inputs.get("ntor_transform_refinement", 1.0)),
        )
        s = FourierSurface(
            nfp=int(surf_np.nfp),
            lasym=False,
            xm=jnp.asarray(surf_np.xm, dtype=jnp.int32),
            xn=jnp.asarray(surf_np.xn, dtype=jnp.int32),
            rmnc=jnp.asarray(surf_np.rmnc, dtype=jnp.float64),
            zmns=jnp.asarray(surf_np.zmns, dtype=jnp.float64),
            rmns=jnp.asarray(surf_np.rmns, dtype=jnp.float64),
            zmnc=jnp.asarray(surf_np.zmnc, dtype=jnp.float64),
        )
        nfp = int(s.nfp)
        ze = zeta_grid(nzeta, nfp)
        xyz, dth, dze, d2th2, d2thze, d2ze2 = eval_surface_xyz_and_derivs2(s, th, ze)
        r = _to_3TZ(xyz)
        rth = _to_3TZ(dth)
        rze = _to_3TZ(dze)
        rtt = _to_3TZ(d2th2)
        rtz = _to_3TZ(d2thze)
        rzz = _to_3TZ(d2ze2)
        _, _, _, nunit, normN = metrics_and_normals(rth, rze)
        xm_plasma = s.xm
        xn_plasma = s.xn
        rmnc_plasma = s.rmnc
        zmns_plasma = s.zmns
        rmns_plasma = s.rmns
        zmnc_plasma = s.zmnc
        lasym = False
    elif gopt_eff == 5:
        # EFIT (regcoil_init_plasma_mod.f90 case(5)).
        efit_filename = inputs.get("efit_filename", None)
        if efit_filename is None:
            raise ValueError("geometry_option_plasma=5 requires efit_filename")
        # parse_namelist lowercases keys, so prefer efit_psin but keep a fallback.
        psiN = float(inputs.get("efit_psin", inputs.get("efit_psiN", 0.98)))
        num_modes = int(inputs.get("efit_num_modes", 10))
        gfile = read_efit_gfile(str(efit_filename))
        rmnc, zmns, rmns, zmnc = efit_boundary_fourier(gfile=gfile, psiN_desired=psiN, num_modes=num_modes)
        nfp = int(inputs.get("nfp_imposed", 1))
        ze = zeta_grid(nzeta, nfp)
        xm = jnp.asarray(np.arange(num_modes, dtype=np.int32), dtype=jnp.int32)
        xn = jnp.asarray(np.zeros((num_modes,), dtype=np.int32), dtype=jnp.int32)
        s = FourierSurface(
            nfp=nfp,
            lasym=True,
            xm=xm,
            xn=xn,
            rmnc=jnp.asarray(rmnc, dtype=jnp.float64),
            zmns=jnp.asarray(zmns, dtype=jnp.float64),
            rmns=jnp.asarray(rmns, dtype=jnp.float64),
            zmnc=jnp.asarray(zmnc, dtype=jnp.float64),
        )
        xyz, dth, dze, d2th2, d2thze, d2ze2 = eval_surface_xyz_and_derivs2(s, th, ze)
        r = _to_3TZ(xyz)
        rth = _to_3TZ(dth)
        rze = _to_3TZ(dze)
        rtt = _to_3TZ(d2th2)
        rtz = _to_3TZ(d2thze)
        rzz = _to_3TZ(d2ze2)
        _, _, _, nunit, normN = metrics_and_normals(rth, rze)
        xm_plasma = s.xm
        xn_plasma = s.xn
        rmnc_plasma = s.rmnc
        zmns_plasma = s.zmns
        rmns_plasma = s.rmns
        zmnc_plasma = s.zmnc
        lasym = True
    elif gopt_eff == 6:
        # Read a simple Fourier table (see regcoil_init_plasma_mod.f90 case(6)).
        fname = inputs.get("shape_filename_plasma", None)
        if fname is None:
            raise ValueError("geometry_option_plasma=6 requires shape_filename_plasma")
        tab = read_surface_fourier_table(str(fname))
        nfp = int(inputs.get("nfp_imposed", 1))
        s = FourierSurface(
            nfp=nfp,
            lasym=True,
            xm=jnp.asarray(tab.xm, dtype=jnp.int32),
            xn=jnp.asarray(tab.xn, dtype=jnp.int32),
            rmnc=jnp.asarray(tab.rmnc, dtype=jnp.float64),
            zmns=jnp.asarray(tab.zmns, dtype=jnp.float64),
            rmns=jnp.asarray(tab.rmns, dtype=jnp.float64),
            zmnc=jnp.asarray(tab.zmnc, dtype=jnp.float64),
        )
        xyz, dth, dze, d2th2, d2thze, d2ze2 = eval_surface_xyz_and_derivs2(s, th, ze)
        r = _to_3TZ(xyz)
        rth = _to_3TZ(dth)
        rze = _to_3TZ(dze)
        rtt = _to_3TZ(d2th2)
        rtz = _to_3TZ(d2thze)
        rzz = _to_3TZ(d2ze2)
        _, _, _, nunit, normN = metrics_and_normals(rth, rze)
        xm_plasma = s.xm
        xn_plasma = s.xn
        rmnc_plasma = s.rmnc
        zmns_plasma = s.zmns
        rmns_plasma = s.rmns
        zmnc_plasma = s.zmnc
        lasym = True
    elif gopt_eff == 7:
        # Read a FOCUS boundary file (see regcoil_init_plasma_mod.f90 case(7)).
        fname = inputs.get("shape_filename_plasma", None)
        if fname is None:
            raise ValueError("geometry_option_plasma=7 requires shape_filename_plasma")
        foc = read_focus_surface(str(fname))
        nfp = int(foc.nfp)
        ze = zeta_grid(nzeta, nfp)
        s = FourierSurface(
            nfp=nfp,
            lasym=True,
            xm=jnp.asarray(foc.xm, dtype=jnp.int32),
            xn=jnp.asarray(foc.xn, dtype=jnp.int32),
            rmnc=jnp.asarray(foc.rmnc, dtype=jnp.float64),
            zmns=jnp.asarray(foc.zmns, dtype=jnp.float64),
            rmns=jnp.asarray(foc.rmns, dtype=jnp.float64),
            zmnc=jnp.asarray(foc.zmnc, dtype=jnp.float64),
        )
        xyz, dth, dze, d2th2, d2thze, d2ze2 = eval_surface_xyz_and_derivs2(s, th, ze)
        r = _to_3TZ(xyz)
        rth = _to_3TZ(dth)
        rze = _to_3TZ(dze)
        rtt = _to_3TZ(d2th2)
        rtz = _to_3TZ(d2thze)
        rzz = _to_3TZ(d2ze2)
        _, _, _, nunit, normN = metrics_and_normals(rth, rze)
        # Store optional Bn coefficient tables from FOCUS (used when load_bnorm=.true.).
        focus_bnorm = None
        if foc.nbf > 0 and foc.bfn is not None:
            focus_bnorm = dict(bfm=foc.bfm, bfn=foc.bfn, bfc=foc.bfc, bfs=foc.bfs)
        xm_plasma = s.xm
        xn_plasma = s.xn
        rmnc_plasma = s.rmnc
        zmns_plasma = s.zmns
        rmns_plasma = s.rmns
        zmnc_plasma = s.zmnc
        lasym = True
    else:
        raise NotImplementedError(f"geometry_option_plasma={gopt} not implemented yet (only 0,1,2,3,4,5,6,7).")
    out = dict(
        nfp=nfp,
        lasym=bool(lasym),
        theta=th,
        zeta=ze,
        r=r,
        rth=rth,
        rze=rze,
        rtt=rtt,
        rtz=rtz,
        rzz=rzz,
        nunit=nunit,
        normN=normN,
        xm_plasma=xm_plasma,
        xn_plasma=xn_plasma,
        rmnc_plasma=rmnc_plasma,
        zmns_plasma=zmns_plasma,
        rmns_plasma=rmns_plasma,
        zmnc_plasma=zmnc_plasma,
    )
    if gopt_eff == 7:
        out["focus_bnorm"] = focus_bnorm
    return out

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
        r, rth, rze, rtt, rtz, rzz, nunit, normN = torus_xyz_and_derivs2(th, ze, R0, a)
        xm_coil = jnp.asarray([0, 1], dtype=jnp.int32)
        xn_coil = jnp.asarray([0, 0], dtype=jnp.int32)
        rmnc_coil = jnp.asarray([R0, a], dtype=jnp.float64)
        zmns_coil = jnp.asarray([0.0, a], dtype=jnp.float64)
        rmns_coil = jnp.zeros_like(rmnc_coil)
        zmnc_coil = jnp.zeros_like(rmnc_coil)
        lasym = False
    elif gopt_eff == 2:
        if vmec_boundary is None:
            raise ValueError("geometry_option_coil=2 requires plasma surface from VMEC.")
        # Keep separation as a JAX value to allow autodiff-based winding-surface optimization.
        separation = jnp.asarray(inputs["separation"], dtype=jnp.float64)
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

        xyz, dth_xyz, dze_xyz, d2th2, d2thze, d2ze2 = eval_surface_xyz_and_derivs2(coil_surf, th, ze)
        r = _to_3TZ(xyz)
        rth = _to_3TZ(dth_xyz)
        rze = _to_3TZ(dze_xyz)
        rtt = _to_3TZ(d2th2)
        rtz = _to_3TZ(d2thze)
        rzz = _to_3TZ(d2ze2)
        _, _, _, nunit, normN = metrics_and_normals(rth, rze)
        xm_coil = coil_surf.xm
        xn_coil = coil_surf.xn
        rmnc_coil = coil_surf.rmnc
        zmns_coil = coil_surf.zmns
        rmns_coil = coil_surf.rmns
        zmnc_coil = coil_surf.zmnc
        lasym = bool(coil_surf.lasym)
    elif gopt_eff == 4:
        # Constant-arclength theta coordinate on an offset-from-plasma winding surface
        # (REGCOIL geometry_option_coil=4).
        if vmec_boundary is None:
            raise ValueError("geometry_option_coil=4 requires plasma surface from VMEC.")
        separation = jnp.asarray(inputs["separation"], dtype=jnp.float64)
        s = vmec_boundary

        theta_refinement = int(inputs.get("theta_arclength_refinement", 4))
        ntheta_ref = int(max(ntheta * theta_refinement, ntheta))
        th_ref = theta_grid(ntheta_ref)

        # Build a dense offset curve once, then do the constant-arclength reparameterization
        # using interpolation on that curve (much cheaper than re-solving the offset point
        # repeatedly, and matches the intent of the Fortran iteration in
        # regcoil_init_coil_surface.f90).
        def one_point(theta, zeta):
            return offset_surface_point(s, theta, zeta, separation, iters=12)

        v_ref = jax.vmap(lambda ze_j: jax.vmap(lambda th_i: one_point(th_i, ze_j))(th_ref))
        xyz_ref_zt3 = v_ref(ze)  # (Z,Tref,3)

        # Iteratively adjust theta_plasma(theta_coil) to achieve near-constant dl in (R,Z).
        tol = float(inputs.get("constant_arclength_tolerance", 1.0e-6))
        max_iter = int(inputs.get("constant_arclength_max_iterations", 50))

        theta_plasma_zt = jnp.tile(th[None, :], (nzeta, 1))  # (Z,T)
        for _ in range(max_iter):
            # Interpolate dense curve at current theta_plasma.
            xyz_slice = jax.vmap(lambda xyz_row, thq: _interp_uniform_periodic_theta(data_t=xyz_row, query_theta=thq))(
                xyz_ref_zt3, theta_plasma_zt
            )  # (Z,T,3)
            R_slice = jnp.sqrt(xyz_slice[..., 0] * xyz_slice[..., 0] + xyz_slice[..., 1] * xyz_slice[..., 1])
            Z_slice = xyz_slice[..., 2]
            R_next = jnp.roll(R_slice, shift=-1, axis=1)
            Z_next = jnp.roll(Z_slice, shift=-1, axis=1)
            dl = jnp.sqrt((R_next - R_slice) ** 2 + (Z_next - Z_slice) ** 2)  # (Z,T)

            rel_var = (jnp.max(dl, axis=1) - jnp.min(dl, axis=1)) / (jnp.mean(dl, axis=1) + 1e-300)  # (Z,)
            if bool(jnp.all(rel_var < tol)):
                break

            s_cum = jnp.concatenate([jnp.zeros((nzeta,), dtype=dl.dtype)[:, None], jnp.cumsum(dl[:, :-1], axis=1)], axis=1)  # (Z,T)
            s_end = jnp.sum(dl, axis=1)  # (Z,)
            s_norm = (2.0 * jnp.pi) * (s_cum / (s_end[:, None] + 1e-300))  # (Z,T)

            # Fortran builds a periodic spline of y(s) where y = theta_plasma - s_norm.
            # We use a piecewise-linear periodic interpolation, which is sufficient for parity
            # in lo-res examples and avoids extra dependencies.
            y = theta_plasma_zt - s_norm
            y_new = jax.vmap(lambda srow, yrow: _invert_monotonic_piecewise_linear(xp=srow, fp=yrow, x=th))(
                s_norm, y
            )  # (Z,T)
            theta_plasma_zt = (y_new + th[None, :]) % (2.0 * jnp.pi)

        xyz_off_zt3 = jax.vmap(lambda xyz_row, thq: _interp_uniform_periodic_theta(data_t=xyz_row, query_theta=thq))(
            xyz_ref_zt3, theta_plasma_zt
        )  # (Z,T,3)
        xyz_off = jnp.transpose(xyz_off_zt3, (1, 0, 2))  # (T,Z,3)

        major_R = jnp.sqrt(xyz_off[..., 0] * xyz_off[..., 0] + xyz_off[..., 1] * xyz_off[..., 1])
        z_coord = xyz_off[..., 2]

        max_mpol_coil = int(inputs.get("max_mpol_coil", 24))
        max_ntor_coil = int(inputs.get("max_ntor_coil", 24))
        mpol_coil = min(ntheta // 2, max_mpol_coil)
        ntor_coil = min(nzeta // 2, max_ntor_coil)

        xm_np, xn_np = init_fourier_modes(mpol_coil, ntor_coil, include_00=True)
        xn_np = xn_np * nfp
        xm = jnp.asarray(xm_np, dtype=jnp.int32)
        xn = jnp.asarray(xn_np, dtype=jnp.int32)

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

        rmnc = rmnc.at[0].set(jnp.mean(major_R))
        zmnc = zmnc.at[0].set(jnp.mean(z_coord))

        lasym = bool(getattr(vmec_boundary, "lasym", False))
        if not lasym:
            rmns = jnp.zeros_like(rmns)
            zmnc = jnp.zeros_like(zmnc)

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

        xyz, dth_xyz, dze_xyz, d2th2, d2thze, d2ze2 = eval_surface_xyz_and_derivs2(coil_surf, th, ze)
        r = _to_3TZ(xyz)
        rth = _to_3TZ(dth_xyz)
        rze = _to_3TZ(dze_xyz)
        rtt = _to_3TZ(d2th2)
        rtz = _to_3TZ(d2thze)
        rzz = _to_3TZ(d2ze2)
        _, _, _, nunit, normN = metrics_and_normals(rth, rze)
        xm_coil = coil_surf.xm
        xn_coil = coil_surf.xn
        rmnc_coil = coil_surf.rmnc
        zmns_coil = coil_surf.zmns
        rmns_coil = coil_surf.rmns
        zmnc_coil = coil_surf.zmnc
        lasym = bool(coil_surf.lasym)
    elif gopt_eff == 3:
        # Read the winding surface from a NESCOIL nescin file (REGCOIL geometry_option_coil=3).
        nescin = inputs.get("nescin_filename", None)
        if nescin is None:
            raise ValueError("geometry_option_coil=3 requires nescin_filename")

        surf = read_nescin_current_surface(str(nescin))
        xm = jnp.asarray(surf.xm, dtype=jnp.int32)
        # regcoil_read_nescin.f90: xn_coil = -nfp * xn_coil (convert NESCOIL convention to VMEC/REGCOIL convention)
        xn = jnp.asarray((-nfp) * surf.xn, dtype=jnp.int32)

        rmnc = jnp.asarray(surf.rmnc, dtype=jnp.float64)
        zmns = jnp.asarray(surf.zmns, dtype=jnp.float64)
        rmns = jnp.asarray(surf.rmns, dtype=jnp.float64)
        zmnc = jnp.asarray(surf.zmnc, dtype=jnp.float64)

        # Apply REGCOIL-style filtering if requested.
        max_mpol_coil = int(inputs.get("max_mpol_coil", 24))
        max_ntor_coil = int(inputs.get("max_ntor_coil", 24))
        mpol_coil_filter = int(inputs.get("mpol_coil_filter", max_mpol_coil))
        ntor_coil_filter = int(inputs.get("ntor_coil_filter", max_ntor_coil))
        keep = (jnp.abs(xm) <= mpol_coil_filter) & (jnp.abs(xn) <= (ntor_coil_filter * nfp))
        rmnc = jnp.where(keep, rmnc, 0.0)
        rmns = jnp.where(keep, rmns, 0.0)
        zmnc = jnp.where(keep, zmnc, 0.0)
        zmns = jnp.where(keep, zmns, 0.0)

        # NESCOIL nescin files may include asymmetric coefficients; treat nonzero arrays as lasym.
        lasym = bool(np.any(np.abs(surf.rmns) > 0.0) or np.any(np.abs(surf.zmnc) > 0.0))

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

        xyz, dth_xyz, dze_xyz, d2th2, d2thze, d2ze2 = eval_surface_xyz_and_derivs2(coil_surf, th, ze)
        r = _to_3TZ(xyz)
        rth = _to_3TZ(dth_xyz)
        rze = _to_3TZ(dze_xyz)
        rtt = _to_3TZ(d2th2)
        rtz = _to_3TZ(d2thze)
        rzz = _to_3TZ(d2ze2)
        _, _, _, nunit, normN = metrics_and_normals(rth, rze)
        xm_coil = coil_surf.xm
        xn_coil = coil_surf.xn
        rmnc_coil = coil_surf.rmnc
        zmns_coil = coil_surf.zmns
        rmns_coil = coil_surf.rmns
        zmnc_coil = coil_surf.zmnc
        lasym = bool(coil_surf.lasym)
    else:
        raise NotImplementedError(f"geometry_option_coil={gopt} not implemented yet (only 0,1,2,3,4).")
    return dict(
        theta=th,
        zeta=ze,
        r=r,
        rth=rth,
        rze=rze,
        rtt=rtt,
        rtz=rtz,
        rzz=rzz,
        nunit=nunit,
        normN=normN,
        lasym=bool(lasym),
        xm_coil=xm_coil,
        xn_coil=xn_coil,
        rmnc_coil=rmnc_coil,
        zmns_coil=zmns_coil,
        rmns_coil=rmns_coil,
        zmnc_coil=zmnc_coil,
    )
