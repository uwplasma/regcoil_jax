#!/usr/bin/env python3
"""Axisymmetric winding-surface filament optimization (end-to-end, autodiff).

This example demonstrates an *axisymmetric* winding surface (a circular torus) used to
optimize a set of **filamentary coils** for a generally **non-axisymmetric** plasma boundary
from a VMEC `wout_*.nc`.

The key point is that the coil *parameterization* is differentiable end-to-end:

- coil curves are represented by θ_k(ζ) Fourier series on the axisymmetric surface,
- Biot–Savart evaluation is in JAX,
- the objective uses JAX autodiff + Adam to update both coil geometry and per-coil currents,
- the resulting coil-only field lines and a soft Poincaré section are traced in JAX.

Outputs (written under `--out_dir`, defaulting to a gitignored `outputs_*` folder):
  - publication-ready figures (`figures/*.png`)
  - ParaView-ready VTK files (`vtk/*.vts` / `vtk/*.vtp`)

Notes:
  - This script intentionally avoids the discrete REGCOIL contour-cutting step. It instead
    optimizes filaments directly on a winding surface.
  - The axisymmetric winding surface used here is a circular torus. More general axisymmetric
    meridional shapes R(θ), Z(θ) could be added as a follow-up.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

try:
    import netCDF4  # noqa: F401
except Exception:  # pragma: no cover
    netCDF4 = None


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _estimate_plasma_r0_a(*, xyz_tz3: np.ndarray) -> tuple[float, float]:
    """Crude (robust) estimate of major radius R0 and minor radius a from sampled plasma XYZ."""
    x = xyz_tz3[..., 0].reshape(-1)
    y = xyz_tz3[..., 1].reshape(-1)
    z = xyz_tz3[..., 2].reshape(-1)
    R = np.sqrt(x * x + y * y)
    R0 = float(np.mean(R))
    # Use the max distance from the (R0, Z=0) center in the RZ plane.
    a = float(np.max(np.sqrt((R - R0) ** 2 + z**2)))
    return R0, a


def _write_vtp_coils(path: Path, coils_kz3: np.ndarray) -> None:
    from regcoil_jax.vtk_io import write_vtp_polydata

    coils = np.asarray(coils_kz3, dtype=float)
    ncoils, npts, _ = coils.shape
    pts = coils.reshape(ncoils * npts, 3)
    lines = [list(range(i * npts, (i + 1) * npts)) + [i * npts] for i in range(ncoils)]
    write_vtp_polydata(path, points=pts, lines=lines)


def main() -> None:
    here = Path(__file__).resolve()
    examples_dir = here.parent
    project_root = here.parents[2]

    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", type=str, default="cpu")
    ap.add_argument(
        "--wout",
        type=str,
        default=str(examples_dir / "simsopt_vmec_cases" / "wout_li383_low_res_reference.nc"),
        help="VMEC wout file defining the target plasma boundary.",
    )
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--fast", action="store_true", help="Use tiny settings for smoke tests / quick iteration.")

    # Plasma evaluation grid (one field period).
    ap.add_argument("--ntheta_plasma", type=int, default=48)
    ap.add_argument("--nzeta_plasma", type=int, default=48)

    # Coil set and parameterization.
    ap.add_argument("--ncoils_base", type=int, default=6, help="Number of coils in the base set (replicated across nfp).")
    ap.add_argument("--npts_coil", type=int, default=256)
    ap.add_argument("--theta_modes", type=int, default=6, help="Number of Fourier modes in θ(ζ) per coil.")

    # Axisymmetric winding surface (circular torus): choose from plasma if not given.
    ap.add_argument("--winding_R0", type=float, default=float("nan"))
    ap.add_argument("--winding_a", type=float, default=float("nan"))
    ap.add_argument("--winding_sep", type=float, default=0.25, help="Extra minor-radius padding added to the winding surface (meters).")

    # Optimization.
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=8e-3)
    ap.add_argument("--w_smooth", type=float, default=5e-3, help="Smoothness penalty weight for θ(ζ).")
    ap.add_argument("--w_current", type=float, default=1e-10, help="L2 penalty weight for coil currents (A).")
    ap.add_argument("--seed", type=int, default=0)

    # Field line / Poincaré visualization.
    ap.add_argument("--do_fieldlines", action="store_true", help="Trace coil-only field lines and write soft Poincaré VTK.")
    ap.add_argument("--phi0", type=float, default=0.0)
    ap.add_argument("--fieldline_ds", type=float, default=0.03)
    ap.add_argument("--fieldline_steps", type=int, default=900)
    ap.add_argument("--n_starts", type=int, default=10)

    args = ap.parse_args()

    if netCDF4 is None:  # pragma: no cover
        raise SystemExit("netCDF4 is required (install regcoil_jax[viz])")

    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("JAX_PLATFORM_NAME", str(args.platform))

    if args.fast:
        args.ntheta_plasma = 20
        args.nzeta_plasma = 20
        args.ncoils_base = 2
        args.theta_modes = 2
        args.npts_coil = 80
        args.steps = 40
        args.lr = 2e-2
        args.do_fieldlines = False

    wout = Path(args.wout).resolve()
    if not wout.exists():
        raise SystemExit(f"Missing wout file: {wout}")

    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = examples_dir / f"outputs_axisym_ws_filaments_{stamp}"
    else:
        out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "vtk").mkdir(exist_ok=True)

    # Copy wout next to outputs for reproducibility.
    wout_dst = out_dir / wout.name
    if wout_dst != wout:
        wout_dst.write_bytes(wout.read_bytes())

    import jax
    import jax.numpy as jnp

    from regcoil_jax.geometry_fourier import FourierSurface, eval_surface_xyz_and_derivs
    from regcoil_jax.grids import theta_grid, zeta_grid
    from regcoil_jax.diff_coil_cutting import bnormal_from_coil_curves, replicate_coils_across_nfp
    from regcoil_jax.io_vmec import read_wout_boundary
    from regcoil_jax.optimize import minimize_adam
    from regcoil_jax.vtk_io import write_vts_structured_grid

    # -----
    # Plasma geometry (one field period)
    # -----
    b = read_wout_boundary(str(wout_dst), radial_mode="full")
    plasma = FourierSurface(
        nfp=int(b.nfp),
        lasym=bool(b.lasym),
        xm=jnp.asarray(b.xm, dtype=jnp.int32),
        xn=jnp.asarray(b.xn, dtype=jnp.int32),
        rmnc=jnp.asarray(b.rmnc, dtype=jnp.float64),
        zmns=jnp.asarray(b.zmns, dtype=jnp.float64),
        rmns=jnp.asarray(b.rmns, dtype=jnp.float64),
        zmnc=jnp.asarray(b.zmnc, dtype=jnp.float64),
    )
    nfp = int(plasma.nfp)

    th_p = theta_grid(int(args.ntheta_plasma))
    ze_p = zeta_grid(int(args.nzeta_plasma), nfp=nfp)
    xyz_p_tz3, dth_p_tz3, dze_p_tz3 = eval_surface_xyz_and_derivs(plasma, th_p, ze_p)
    nvec_p_tz3 = jnp.cross(dze_p_tz3, dth_p_tz3)
    n_norm = jnp.linalg.norm(nvec_p_tz3, axis=-1, keepdims=True)
    n_norm = jnp.where(n_norm == 0.0, 1.0, n_norm)
    n_hat_p_tz3 = nvec_p_tz3 / n_norm

    plasma_xyz_np = np.asarray(xyz_p_tz3, dtype=float)
    R0_plasma, a_plasma = _estimate_plasma_r0_a(xyz_tz3=plasma_xyz_np)

    # -----
    # Axisymmetric winding surface (circular torus)
    # -----
    if not np.isfinite(float(args.winding_R0)):
        R0_ws = R0_plasma
    else:
        R0_ws = float(args.winding_R0)
    if not np.isfinite(float(args.winding_a)):
        a_ws = a_plasma + float(args.winding_sep)
    else:
        a_ws = float(args.winding_a)

    ze_full = jnp.linspace(0.0, 2.0 * jnp.pi, int(args.nzeta_plasma) * nfp, endpoint=False, dtype=jnp.float64)
    th_full = jnp.linspace(0.0, 2.0 * jnp.pi, int(args.ntheta_plasma), endpoint=False, dtype=jnp.float64)

    def torus_xyz(theta: jnp.ndarray, zeta: jnp.ndarray) -> jnp.ndarray:
        th = theta[None, :]
        ze = zeta[:, None]
        cth = jnp.cos(th)
        sth = jnp.sin(th)
        cze = jnp.cos(ze)
        sze = jnp.sin(ze)
        R = jnp.asarray(R0_ws, dtype=jnp.float64) + jnp.asarray(a_ws, dtype=jnp.float64) * cth
        x = R * cze
        y = R * sze
        z = jnp.asarray(a_ws, dtype=jnp.float64) * sth * jnp.ones_like(ze)
        return jnp.stack([x, y, z], axis=-1)  # (nzeta, ntheta, 3)

    ws_xyz_zt3 = torus_xyz(th_full, ze_full)
    write_vts_structured_grid(out_dir / "vtk" / "winding_surface_full.vts", points_zt3=np.asarray(ws_xyz_zt3, dtype=float))

    # -----
    # Coil parameterization: θ_k(ζ) Fourier series on the axisymmetric surface
    # -----
    ncoils = int(args.ncoils_base)
    nm = int(args.theta_modes)
    nz = int(args.npts_coil)
    z_coil = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False, dtype=jnp.float64)
    m = jnp.arange(1, nm + 1, dtype=jnp.float64)[:, None]  # (nm,1)
    cos_mz = jnp.cos(m * z_coil[None, :])  # (nm,nz)
    sin_mz = jnp.sin(m * z_coil[None, :])  # (nm,nz)

    theta0_init = (2.0 * jnp.pi / float(ncoils)) * jnp.arange(ncoils, dtype=jnp.float64)
    coeff0 = jnp.zeros((ncoils, 2 * nm), dtype=jnp.float64)
    I0 = jnp.ones((ncoils,), dtype=jnp.float64) * 8e4

    # Small random perturbation to break symmetry (reproducible).
    key = jax.random.PRNGKey(int(args.seed))
    coeff0 = coeff0 + 2e-2 * jax.random.normal(key, shape=coeff0.shape, dtype=jnp.float64)

    # Pack (theta0, coeff_cos, coeff_sin, I) per coil into a single vector.
    x0 = jnp.concatenate(
        [
            theta0_init[:, None],
            coeff0[:, :nm],
            coeff0[:, nm:],
            I0[:, None],
        ],
        axis=1,
    ).reshape((-1,))

    plasma_pts = xyz_p_tz3.reshape((-1, 3))
    plasma_nhat = n_hat_p_tz3.reshape((-1, 3))

    def coils_from_x(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = x.reshape((ncoils, 2 * nm + 2))
        theta0 = x[:, 0]  # (ncoils,)
        a_cos = x[:, 1 : 1 + nm]  # (ncoils,nm)
        a_sin = x[:, 1 + nm : 1 + 2 * nm]
        I = x[:, -1]  # (ncoils,)

        theta = theta0[:, None] + a_cos @ cos_mz + a_sin @ sin_mz  # (ncoils,nzeta_pts)
        R = jnp.asarray(R0_ws, dtype=jnp.float64) + jnp.asarray(a_ws, dtype=jnp.float64) * jnp.cos(theta)
        X = R * jnp.cos(z_coil[None, :])
        Y = R * jnp.sin(z_coil[None, :])
        Z = jnp.asarray(a_ws, dtype=jnp.float64) * jnp.sin(theta)
        coils_kz3 = jnp.stack([X, Y, Z], axis=-1)
        return coils_kz3, I, theta

    def objective(x: jnp.ndarray) -> jnp.ndarray:
        coils_base_kz3, I_base, theta = coils_from_x(x)
        coils_full = replicate_coils_across_nfp(coils_kz3=coils_base_kz3, nfp=nfp)
        I_full = jnp.tile(I_base, reps=(nfp,))

        Bn = bnormal_from_coil_curves(
            coils_kz3=coils_full,
            coil_currents=I_full,
            eval_points=plasma_pts,
            eval_normals_unit=plasma_nhat,
        )
        chi2 = jnp.mean(Bn * Bn)

        dtheta = theta - jnp.roll(theta, shift=-1, axis=1)
        smooth = jnp.mean(dtheta * dtheta)
        cur = jnp.mean(I_base * I_base)
        return chi2 + float(args.w_smooth) * smooth + float(args.w_current) * cur

    objective_jit = jax.jit(objective)

    def _metrics(x: jnp.ndarray) -> dict[str, float]:
        coils_base_kz3, I_base, _ = coils_from_x(x)
        coils_full = replicate_coils_across_nfp(coils_kz3=coils_base_kz3, nfp=nfp)
        I_full = jnp.tile(I_base, reps=(nfp,))
        Bn = bnormal_from_coil_curves(
            coils_kz3=coils_full,
            coil_currents=I_full,
            eval_points=plasma_pts,
            eval_normals_unit=plasma_nhat,
        )
        Bn = jnp.asarray(Bn)
        return dict(
            chi2_B=float(jnp.mean(Bn * Bn)),
            rms_B=float(jnp.sqrt(jnp.mean(Bn * Bn))),
            max_abs_B=float(jnp.max(jnp.abs(Bn))),
            max_abs_I=float(jnp.max(jnp.abs(I_base))),
            loss=float(objective_jit(x)),
        )

    m0 = _metrics(x0)
    print(f"[axisym] nfp={nfp}  R0_plasma≈{R0_plasma:.3f}  a_plasma≈{a_plasma:.3f}", flush=True)
    print(f"[axisym] winding surface: R0={R0_ws:.3f}, a={a_ws:.3f}", flush=True)
    print(f"[axisym] initial: loss={m0['loss']:.6e}  rms(Bn)={m0['rms_B']:.6e}  max|Bn|={m0['max_abs_B']:.6e}", flush=True)

    res = minimize_adam(objective, x0, steps=int(args.steps), lr=float(args.lr), jit=True)
    x_opt = res.x
    m1 = _metrics(x_opt)
    print(f"[axisym] final:   loss={m1['loss']:.6e}  rms(Bn)={m1['rms_B']:.6e}  max|Bn|={m1['max_abs_B']:.6e}", flush=True)

    # Write summary for tests / reproducibility.
    summary = dict(
        wout=wout_dst.name,
        nfp=nfp,
        ntheta_plasma=int(args.ntheta_plasma),
        nzeta_plasma=int(args.nzeta_plasma),
        ncoils_base=ncoils,
        theta_modes=nm,
        npts_coil=nz,
        winding_R0=R0_ws,
        winding_a=a_ws,
        seed=int(args.seed),
        opt_steps=int(args.steps),
        opt_lr=float(args.lr),
        metrics_before=m0,
        metrics_after=m1,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    # -----
    # VTK outputs
    # -----
    coils0_base, I0_base, _ = coils_from_x(x0)
    coils1_base, I1_base, _ = coils_from_x(x_opt)

    coils0_full = replicate_coils_across_nfp(coils_kz3=coils0_base, nfp=nfp)
    coils1_full = replicate_coils_across_nfp(coils_kz3=coils1_base, nfp=nfp)
    I0_full = jnp.tile(I0_base, reps=(nfp,))
    I1_full = jnp.tile(I1_base, reps=(nfp,))

    _write_vtp_coils(out_dir / "vtk" / "coils_before.vtp", np.asarray(coils0_full, dtype=float))
    _write_vtp_coils(out_dir / "vtk" / "coils_after.vtp", np.asarray(coils1_full, dtype=float))

    # Plasma surface (full torus) + Bn before/after on one period repeated for viz.
    Bn0 = np.asarray(
        bnormal_from_coil_curves(
            coils_kz3=coils0_full,
            coil_currents=I0_full,
            eval_points=plasma_pts,
            eval_normals_unit=plasma_nhat,
        ).reshape((int(args.ntheta_plasma), int(args.nzeta_plasma))),
        dtype=float,
    ).T  # (nzeta,ntheta)
    Bn1 = np.asarray(
        bnormal_from_coil_curves(
            coils_kz3=coils1_full,
            coil_currents=I1_full,
            eval_points=plasma_pts,
            eval_normals_unit=plasma_nhat,
        ).reshape((int(args.ntheta_plasma), int(args.nzeta_plasma))),
        dtype=float,
    ).T

    plasma_full_xyz = np.asarray(eval_surface_xyz_and_derivs(plasma, th_full, ze_full)[0], dtype=float)  # (ntheta, nzeta_full, 3)
    plasma_full_xyz = np.swapaxes(plasma_full_xyz, 0, 1)  # (nzeta_full, ntheta, 3)
    # Tile Bn maps across field periods for display.
    Bn0_full = np.tile(Bn0, reps=(nfp, 1))
    Bn1_full = np.tile(Bn1, reps=(nfp, 1))
    write_vts_structured_grid(
        out_dir / "vtk" / "plasma_full.vts",
        points_zt3=plasma_full_xyz,
        point_data={"Bn_before": Bn0_full, "Bn_after": Bn1_full},
    )

    # -----
    # Figures
    # -----
    plt = _setup_matplotlib()

    # Loss history (log-scale).
    fig = plt.figure(figsize=(6.0, 3.2))
    ax = fig.add_subplot(1, 1, 1)
    loss_hist = np.asarray(res.loss_history, dtype=float)
    ax.plot(loss_hist, lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("loss")
    ax.set_title("Axisymmetric winding surface filament optimization")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "loss_history.png")
    plt.close(fig)

    # Bn maps before/after (one field period).
    fig = plt.figure(figsize=(8.2, 3.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    vmax = float(max(np.max(np.abs(Bn0)), np.max(np.abs(Bn1))) + 1e-300)
    im1 = ax1.imshow(Bn0, origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax1.set_title("B·n (before)")
    ax1.set_xlabel(r"$\theta$ index")
    ax1.set_ylabel(r"$\zeta$ index")
    im2 = ax2.imshow(Bn1, origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax2.set_title("B·n (after)")
    ax2.set_xlabel(r"$\theta$ index")
    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.85, pad=0.02, label=r"$B\cdot n$ [T]")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "bn_before_after.png")
    plt.close(fig)

    # 3D coil view (after).
    fig = plt.figure(figsize=(6.0, 5.2))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    coils_np = np.asarray(coils1_full, dtype=float)
    for c in coils_np:
        ax.plot(c[:, 0], c[:, 1], c[:, 2], lw=1.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Optimized filament coils (axisymmetric winding surface)")
    ax.view_init(elev=20.0, azim=35.0)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "coils_3d.png")
    plt.close(fig)

    # -----
    # Optional: field lines + soft Poincaré section (JAX)
    # -----
    if args.do_fieldlines:
        from regcoil_jax.diff_coil_cutting import segments_from_coil_polylines
        from regcoil_jax.fieldlines_jax import trace_fieldlines_rk4, soft_poincare_candidates
        from regcoil_jax.vtk_io import write_vtp_polydata

        segs = segments_from_coil_polylines(coils_kz3=coils1_full)
        I_full_np = np.asarray(I1_full, dtype=float)

        # Start points along a simple radial line in the phi0 plane, centered at Z=0.
        phi0 = float(args.phi0)
        # Use plasma cross-section at the closest zeta slice in the *one-period* data.
        rpl = plasma_xyz_np.transpose(1, 0, 2)  # (nzeta,ntheta,3)
        phi = np.arctan2(rpl[:, :, 1], rpl[:, :, 0])
        phi_mean = np.unwrap(np.mean(phi, axis=1))
        idx = int(np.argmin(np.abs(((phi_mean - phi0 + np.pi) % (2.0 * np.pi)) - np.pi)))
        Rb = np.sqrt(rpl[idx, :, 0] ** 2 + rpl[idx, :, 1] ** 2)
        Zb = rpl[idx, :, 2]
        R_axis = float(np.mean(Rb))
        R_edge = float(np.max(Rb))
        starts_R = np.linspace(R_axis + 0.02, R_edge - 0.02, int(args.n_starts))
        starts = np.stack([starts_R * np.cos(phi0), starts_R * np.sin(phi0), np.zeros_like(starts_R)], axis=1)

        tr = trace_fieldlines_rk4(
            segs,
            starts=starts,
            filament_currents=I_full_np,
            ds=float(args.fieldline_ds),
            n_steps=int(args.fieldline_steps),
            direction=1.0,
            stop_radius=float(R0_ws + 3.0 * a_ws),
        )
        pts = np.asarray(tr.points, dtype=float)  # (nlines, npts, 3)

        # Write field lines as a single polydata with multiple polylines.
        nlines, npts, _ = pts.shape
        all_pts = pts.reshape(nlines * npts, 3)
        lines = [list(range(i * npts, (i + 1) * npts)) for i in range(nlines)]
        write_vtp_polydata(out_dir / "vtk" / "fieldlines_after.vtp", points=all_pts, lines=lines)

        cand, w = soft_poincare_candidates(jnp.asarray(tr.points), nfp=nfp, phi0=phi0)
        cand_np = np.asarray(cand, dtype=float).reshape(-1, 3)
        w_np = np.asarray(w, dtype=float).reshape(-1)
        line_id = np.repeat(np.arange(nlines, dtype=np.int32), repeats=(npts - 1))
        write_vtp_polydata(
            out_dir / "vtk" / "poincare_candidates_after.vtp",
            points=cand_np,
            verts=np.arange(cand_np.shape[0], dtype=np.int64),
            point_data={"weight": w_np, "line_id": line_id.astype(float)},
        )

    print(f"[axisym] wrote outputs to: {out_dir}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()

