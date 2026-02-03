#!/usr/bin/env python3
"""Robust coil-current optimization under coil misalignments (JAX + vmap + autodiff).

This example goes beyond the original Fortran REGCOIL workflow:

- REGCOIL computes a *continuous* surface current (via a current potential Φ) that minimizes B·n on the plasma surface.
- We then *cut filamentary coils* as Φ-contours (REGCOIL-style).
- Finally, we optimize the **discrete coil currents** to stay good **on average** under small coil misalignments,
  using a fully JAX-native, vectorized objective (ensemble via `jax.vmap`) and autodiff gradients.

Outputs:
  - publication-ready figures (loss curves, current values, B·n histograms)
  - ParaView files:
      - coils (`.vtp`)
      - fieldline traces (`.vtp`)
      - soft Poincaré candidate cloud (`.vtu`, with `weight` scalar)

All outputs are written under a gitignored directory: `outputs_robust_currents_misalignment_<timestamp>/`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    import netCDF4
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
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def main() -> None:
    if netCDF4 is None:  # pragma: no cover
        raise SystemExit("netCDF4 is required for this example")

    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    default_input = repo_root / "examples" / "2_intermediate" / "regcoil_in.torus_cli_intermediate"

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(default_input))
    ap.add_argument("--platform", type=str, default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--coils_per_half_period", type=int, default=6)
    ap.add_argument("--theta_shift", type=int, default=0)

    # Robust objective settings (misalignment ensemble).
    ap.add_argument("--samples", type=int, default=8, help="Number of misalignment scenarios in the robust ensemble.")
    ap.add_argument("--rot_std_deg", type=float, default=0.15, help="Std dev of per-coil toroidal rotation [deg].")
    ap.add_argument("--shift_std_m", type=float, default=0.003, help="Std dev of per-coil (x,y,z) shifts [m].")
    ap.add_argument("--subsample", type=int, default=768, help="Number of plasma points used in the objective (speed).")

    # Optimization settings.
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=2.0e-2)
    ap.add_argument("--l2_weight", type=float, default=1.0e-6)
    ap.add_argument("--net_current_weight", type=float, default=1.0e-2)
    ap.add_argument("--seg_batch", type=int, default=2048)

    # Visualization settings.
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--no_vtk", action="store_true")
    ap.add_argument("--fieldline_steps", type=int, default=1600)
    ap.add_argument("--fieldline_ds", type=float, default=0.03)
    ap.add_argument("--poincare_phi0", type=float, default=0.0)
    ap.add_argument("--poincare_sigma", type=float, default=0.06)

    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Missing input: {input_path}")
    if not input_path.name.startswith("regcoil_in."):
        raise SystemExit("Input must be named regcoil_in.*")

    out_dir = here.parent / f"outputs_robust_currents_misalignment_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy input into the output directory so outputs stay together.
    input_copy = out_dir / input_path.name
    input_copy.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")

    # ------------------------------------------------------------
    # 1) Run REGCOIL_JAX and extract the last-lambda fields.
    # ------------------------------------------------------------
    from regcoil_jax.run import run_regcoil

    # Ensure platform selection happens before importing JAX inside regcoil_jax.run.
    import os

    if args.platform:
        os.environ["JAX_PLATFORM_NAME"] = str(args.platform)
    os.environ.setdefault("JAX_ENABLE_X64", "True")

    res = run_regcoil(str(input_copy), verbose=True)
    out_nc = Path(res.output_nc)

    ds = netCDF4.Dataset(str(out_nc), "r")
    nfp = int(ds.variables["nfp"][()])
    net_pol = float(ds.variables["net_poloidal_current_Amperes"][()])
    ilam = int(len(ds.variables["lambda"][:]) - 1)

    theta_p = np.asarray(ds.variables["theta_plasma"][:], dtype=float)
    zeta_p = np.asarray(ds.variables["zeta_plasma"][:], dtype=float)
    theta_c = np.asarray(ds.variables["theta_coil"][:], dtype=float)
    zeta_c = np.asarray(ds.variables["zeta_coil"][:], dtype=float)

    Btot = np.asarray(ds.variables["Bnormal_total"][ilam], dtype=float)  # (nzeta, ntheta)
    Bplasma = np.asarray(ds.variables["Bnormal_from_plasma_current"][:], dtype=float)
    Bnet = np.asarray(ds.variables["Bnormal_from_net_coil_currents"][:], dtype=float)
    Bsv = Btot - (Bplasma + Bnet)  # target from surface current (the part coils should reproduce)

    Phi = np.asarray(ds.variables["current_potential"][ilam], dtype=float)  # (nzeta_coil, ntheta_coil)
    r_coil_full = np.asarray(ds.variables["r_coil"][:], dtype=float)  # (nzetal, ntheta, 3)
    r_plasma_full = np.asarray(ds.variables["r_plasma"][:], dtype=float)  # (nzetal, ntheta, 3)
    ds.close()

    # Use first field period geometry so shapes match Bsv which is per-period.
    nzeta = int(zeta_p.size)
    ntheta = int(theta_p.size)
    r_plasma = r_plasma_full[:nzeta]  # (nzeta, ntheta, 3)

    from regcoil_jax.surface_utils import unit_normals_from_r_zt3

    nunit = unit_normals_from_r_zt3(r_zt3=r_plasma)
    points_full = r_plasma.reshape(-1, 3)
    normals_full = nunit.reshape(-1, 3)
    target_full = Bsv.reshape(-1)

    # Subsample plasma points for a fast robust objective.
    rng = np.random.default_rng(0)
    n_total = int(points_full.shape[0])
    n_sub = int(min(max(int(args.subsample), 16), n_total))
    idx = rng.choice(n_total, size=n_sub, replace=False)
    points = points_full[idx]
    normals = normals_full[idx]
    target = target_full[idx]

    # ------------------------------------------------------------
    # 2) Cut coils from Phi and build filament segments.
    # ------------------------------------------------------------
    from regcoil_jax.coil_cutting import cut_coils_from_current_potential, write_makecoil_filaments
    from regcoil_jax.biot_savart_jax import FilamentSegments, segments_from_filaments, bnormal_from_segments

    coils = cut_coils_from_current_potential(
        current_potential_zt=Phi,
        theta=theta_c,
        zeta=zeta_c,
        r_coil_zt3_full=r_coil_full,
        theta_shift=int(args.theta_shift),
        coils_per_half_period=int(args.coils_per_half_period),
        nfp=int(nfp),
        net_poloidal_current_Amperes=float(net_pol),
    )

    segs0 = segments_from_filaments(filaments_xyz=coils.filaments_xyz)
    I0 = np.asarray(coils.coil_currents, dtype=float)
    net_target = float(np.sum(I0))

    # Save the initial coils for ParaView.
    write_makecoil_filaments(out_dir / "coils_initial", filaments_xyz=coils.filaments_xyz, coil_currents=I0, nfp=nfp)

    # ------------------------------------------------------------
    # 3) Robust objective: misalignment ensemble + autodiff optimization.
    # ------------------------------------------------------------
    import jax
    import jax.numpy as jnp

    from regcoil_jax.optimize import minimize_adam

    seg_mid0 = jnp.asarray(segs0.seg_midpoints, dtype=jnp.float64)  # (M,3)
    seg_dl0 = jnp.asarray(segs0.seg_dls, dtype=jnp.float64)  # (M,3)
    seg_fid = jnp.asarray(segs0.seg_filament, dtype=jnp.int32)  # (M,)
    n_fil = int(segs0.n_filaments)

    points_j = jnp.asarray(points, dtype=jnp.float64)
    normals_j = jnp.asarray(normals, dtype=jnp.float64)
    target_j = jnp.asarray(target, dtype=jnp.float64)

    # Sample per-coil rigid misalignments for each scenario.
    key = jax.random.PRNGKey(0)
    nsamp = int(max(int(args.samples), 1))

    rot_std = float(np.deg2rad(float(args.rot_std_deg)))
    shift_std = float(args.shift_std_m)

    key, k1, k2 = jax.random.split(key, 3)
    phi_f = rot_std * jax.random.normal(k1, shape=(nsamp, n_fil), dtype=jnp.float64)
    shift_f = shift_std * jax.random.normal(k2, shape=(nsamp, n_fil, 3), dtype=jnp.float64)

    # Gather per-segment misalignment parameters: (nsamp, M) and (nsamp, M, 3)
    phi_seg = phi_f[:, seg_fid]
    shift_seg = shift_f[:, seg_fid, :]

    def _apply_z_rotation(x: jnp.ndarray, c: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
        # x: (...,3), c/s: (...,) broadcastable to x[...,0]
        x0, x1, x2 = x[..., 0], x[..., 1], x[..., 2]
        xr0 = c * x0 - s * x1
        xr1 = s * x0 + c * x1
        return jnp.stack([xr0, xr1, x2], axis=-1)

    def mse_bnormal_for_scenario(I: jnp.ndarray, phi_s: jnp.ndarray, shift_s: jnp.ndarray) -> jnp.ndarray:
        c = jnp.cos(phi_s)
        s = jnp.sin(phi_s)
        mid = _apply_z_rotation(seg_mid0, c, s) + shift_s
        dl = _apply_z_rotation(seg_dl0, c, s)
        segs = FilamentSegments(seg_midpoints=mid, seg_dls=dl, seg_filament=seg_fid, n_filaments=n_fil)
        b = bnormal_from_segments(
            segs,
            points=points_j,
            normals_unit=normals_j,
            filament_currents=I,
            seg_batch=int(args.seg_batch),
        )
        return jnp.mean((b - target_j) ** 2)

    v_mse = jax.vmap(lambda phi_s, shift_s, I: mse_bnormal_for_scenario(I, phi_s, shift_s), in_axes=(0, 0, None))

    def project_sum(I_raw: jnp.ndarray) -> jnp.ndarray:
        # Enforce sum(I) = net_target with a differentiable shift.
        s = jnp.sum(I_raw)
        return I_raw + (jnp.asarray(net_target, dtype=jnp.float64) - s) / float(n_fil)

    l2_weight = float(args.l2_weight)
    net_w = float(args.net_current_weight)

    def robust_loss(I_raw: jnp.ndarray) -> jnp.ndarray:
        I = project_sum(I_raw)
        mses = v_mse(phi_seg, shift_seg, I)  # (nsamp,)
        mean = jnp.mean(mses)
        std = jnp.sqrt(jnp.maximum(jnp.mean((mses - mean) ** 2), 0.0))
        reg_l2 = l2_weight * jnp.mean((I / (jnp.asarray(net_target, dtype=jnp.float64) + 1e-30)) ** 2)
        sum_pen = net_w * (jnp.sum(I) - jnp.asarray(net_target, dtype=jnp.float64)) ** 2
        # Minimize average + a small robustness term.
        return mean + 0.25 * std + reg_l2 + sum_pen

    print("[robust] optimizing currents ...", flush=True)
    opt = minimize_adam(robust_loss, jnp.asarray(I0, dtype=jnp.float64), steps=int(args.steps), lr=float(args.lr), jit=True)
    I_opt = np.asarray(project_sum(opt.x))

    write_makecoil_filaments(out_dir / "coils_robust", filaments_xyz=coils.filaments_xyz, coil_currents=I_opt, nfp=nfp)

    # Evaluate nominal MSE on the full plasma grid for reporting/figures.
    segs_nom = segs0
    mse_nom0 = float(np.mean((np.asarray(bnormal_from_segments(segs_nom, points=points_full, normals_unit=normals_full, filament_currents=I0)) - target_full) ** 2))
    mse_nom1 = float(np.mean((np.asarray(bnormal_from_segments(segs_nom, points=points_full, normals_unit=normals_full, filament_currents=I_opt)) - target_full) ** 2))
    print(f"[report] nominal MSE(Bn): initial={mse_nom0:.6e}  robust={mse_nom1:.6e}", flush=True)

    # ------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------
    if not args.no_figures:
        plt = _setup_matplotlib()

        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.plot(np.asarray(opt.loss_history), "-k", lw=1.6)
        ax.set_xlabel("Adam step")
        ax.set_ylabel("robust loss")
        ax.set_title("Robust per-coil current optimization")
        fig.tight_layout()
        fig.savefig(out_dir / "loss_history.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.plot(I0, "o-", label="initial")
        ax.plot(I_opt, "o-", label="robust")
        ax.set_xlabel("coil index")
        ax.set_ylabel("current [A]")
        ax.legend(loc="best")
        ax.set_title("Per-coil currents")
        fig.tight_layout()
        fig.savefig(out_dir / "coil_currents.png")
        plt.close(fig)

        # Histogram of nominal residuals on full plasma grid.
        b0 = np.asarray(bnormal_from_segments(segs_nom, points=points_full, normals_unit=normals_full, filament_currents=I0)) - target_full
        b1 = np.asarray(bnormal_from_segments(segs_nom, points=points_full, normals_unit=normals_full, filament_currents=I_opt)) - target_full
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.hist(b0, bins=60, alpha=0.55, label="initial", density=True)
        ax.hist(b1, bins=60, alpha=0.55, label="robust", density=True)
        ax.set_xlabel(r"$(B_n^{filaments} - B_n^{target})$ [T]")
        ax.set_ylabel("density")
        ax.legend(loc="best")
        ax.set_title("Nominal residual distribution on plasma surface")
        fig.tight_layout()
        fig.savefig(out_dir / "bn_residual_hist.png")
        plt.close(fig)

    # ------------------------------------------------------------
    # ParaView: coils + fieldlines + soft Poincaré candidates
    # ------------------------------------------------------------
    if not args.no_vtk:
        from regcoil_jax.vtk_io import write_vtp_polydata, write_vtu_point_cloud
        from regcoil_jax.fieldlines_jax import trace_fieldlines_rk4, poincare_section_weights, soft_poincare_candidates

        vtk_dir = out_dir / "vtk"
        vtk_dir.mkdir(exist_ok=True)

        def write_coils_vtp(name: str, coil_currents: np.ndarray):
            pts_all = []
            lines = []
            offset = 0
            for filament in coils.filaments_xyz:
                filament = np.asarray(filament, dtype=float)
                n = filament.shape[0]
                pts_all.append(filament)
                lines.append(list(range(offset, offset + n)) + [offset])
                offset += n
            pts_all = np.concatenate(pts_all, axis=0)
            write_vtp_polydata(
                vtk_dir / name,
                points=pts_all,
                lines=lines,
                cell_data={"coil_current": np.asarray(coil_currents, dtype=float)},
            )

        write_coils_vtp("coils_initial.vtp", I0)
        write_coils_vtp("coils_robust.vtp", I_opt)

        # Fieldline starts: a small ring around the zeta=0 slice.
        starts = []
        iz = 0
        nunit_grid = normals_full.reshape(nzeta, ntheta, 3)
        for k in range(10):
            it = int(round(k * (ntheta / 10))) % ntheta
            x = r_plasma[iz, it]
            nh = nunit_grid[iz, it]
            starts.append(x + 0.18 * nh)
        starts = np.asarray(starts, dtype=float)

        trace = trace_fieldlines_rk4(
            segs0,
            starts=starts,
            filament_currents=I_opt,
            ds=float(args.fieldline_ds),
            n_steps=int(args.fieldline_steps),
            stop_radius=12.0,
            seg_batch=int(args.seg_batch),
        )

        pts = np.asarray(trace.points)
        active = np.asarray(trace.active).astype(np.uint8)
        # Write as polyline-by-polyline `write_vtp_polydata`:
        pts_all = []
        lines = []
        offset = 0
        for j in range(pts.shape[0]):
            line = pts[j]
            # keep all points; inactive segments are frozen in trace_fieldlines_rk4
            pts_all.append(line)
            lines.append(list(range(offset, offset + line.shape[0])))
            offset += line.shape[0]
        pts_all = np.concatenate(pts_all, axis=0)
        write_vtp_polydata(vtk_dir / "fieldlines_robust.vtp", points=pts_all, lines=lines)

        # Soft Poincaré: either weights on sample points or candidate crossings.
        weights = np.asarray(
            poincare_section_weights(pts, nfp=int(nfp), phi0=float(args.poincare_phi0), sigma=float(args.poincare_sigma))
        )
        cand, w = soft_poincare_candidates(pts, nfp=int(nfp), phi0=float(args.poincare_phi0))
        cand = np.asarray(cand).reshape(-1, 3)
        w = np.asarray(w).reshape(-1)
        write_vtu_point_cloud(vtk_dir / "poincare_candidates_soft.vtu", points=cand, point_data={"weight": w})

        # Also write all fieldline sample points as a point cloud with weights, for convenient thresholding in ParaView.
        pts_flat = pts.reshape(-1, 3)
        w_flat = weights.reshape(-1)
        active_flat = active.reshape(-1)
        write_vtu_point_cloud(
            vtk_dir / "fieldline_points_with_poincare_weight.vtu",
            points=pts_flat,
            point_data={"poincare_weight": w_flat, "active": active_flat},
        )

    print(f"[done] outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
