#!/usr/bin/env python3
"""Optimize a global rigid alignment of a cut coil set (JAX autodiff).

This example is inspired by “free-space” coil optimization workflows (e.g. FOCUS/FOCUSADD),
where one ultimately cares about the field produced by a discrete set of coils and how that
field changes under geometric perturbations.

Workflow:
  1) run REGCOIL_JAX to obtain a current potential Φ on a winding surface
  2) cut filamentary coils as Φ-contours (REGCOIL-style)
  3) optionally optimize per-coil currents to match the surface-current field
  4) optimize a *global* rigid transform (rotation about z + translation) applied to the coil set
     to reduce the normal-field error on the plasma surface
  5) output figures and ParaView files (coils, field lines, soft Poincaré candidates)

All outputs are written to a gitignored directory:
  `outputs_optimize_rigid_alignment_<timestamp>/`
"""

from __future__ import annotations

import argparse
import os
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
    ap.add_argument("--subsample", type=int, default=1024, help="Number of plasma points used in the objective (speed).")

    ap.add_argument("--optimize_currents_first", action="store_true", help="First optimize per-coil currents (recommended).")
    ap.add_argument("--current_steps", type=int, default=200)
    ap.add_argument("--current_lr", type=float, default=2.0e-2)

    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=5.0e-2)
    ap.add_argument("--rot_reg", type=float, default=5.0e-2, help="Penalty weight on rotation angle (rad^2).")
    ap.add_argument("--shift_reg", type=float, default=5.0e1, help="Penalty weight on translation norm (m^2).")
    ap.add_argument("--seg_batch", type=int, default=2048)

    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--no_vtk", action="store_true")
    ap.add_argument("--fieldline_steps", type=int, default=1200)
    ap.add_argument("--fieldline_ds", type=float, default=0.03)
    ap.add_argument("--poincare_phi0", type=float, default=0.0)
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Missing input: {input_path}")
    if not input_path.name.startswith("regcoil_in."):
        raise SystemExit("Input must be named regcoil_in.*")

    out_dir = here.parent / f"outputs_optimize_rigid_alignment_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_copy = out_dir / input_path.name
    input_copy.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Platform selection must happen before regcoil_jax lazily imports JAX.
    if args.platform:
        os.environ["JAX_PLATFORM_NAME"] = str(args.platform)
    os.environ.setdefault("JAX_ENABLE_X64", "True")

    from regcoil_jax.run import run_regcoil

    res = run_regcoil(str(input_copy), verbose=True)
    out_nc = Path(res.output_nc)

    ds = netCDF4.Dataset(str(out_nc), "r")
    nfp = int(ds.variables["nfp"][()])
    net_pol = float(ds.variables["net_poloidal_current_Amperes"][()])
    ilam = int(res.chosen_idx) if res.chosen_idx is not None else int(len(ds.variables["lambda"][:]) - 1)

    theta_p = np.asarray(ds.variables["theta_plasma"][:], dtype=float)
    zeta_p = np.asarray(ds.variables["zeta_plasma"][:], dtype=float)
    theta_c = np.asarray(ds.variables["theta_coil"][:], dtype=float)
    zeta_c = np.asarray(ds.variables["zeta_coil"][:], dtype=float)

    Btot = np.asarray(ds.variables["Bnormal_total"][ilam], dtype=float)  # (nzeta, ntheta)
    Bplasma = np.asarray(ds.variables["Bnormal_from_plasma_current"][:], dtype=float)
    Bnet = np.asarray(ds.variables["Bnormal_from_net_coil_currents"][:], dtype=float)
    Bsv = Btot - (Bplasma + Bnet)  # surface-current contribution (coils should reproduce)

    Phi = np.asarray(ds.variables["current_potential"][ilam], dtype=float)  # (nzeta_coil, ntheta_coil)
    r_coil_full = np.asarray(ds.variables["r_coil"][:], dtype=float)  # (nzetal, ntheta, 3)
    r_plasma_full = np.asarray(ds.variables["r_plasma"][:], dtype=float)  # (nzetal, ntheta, 3)
    ds.close()

    nzeta = int(zeta_p.size)
    ntheta = int(theta_p.size)
    r_plasma = r_plasma_full[:nzeta]

    from regcoil_jax.surface_utils import unit_normals_from_r_zt3

    nunit = unit_normals_from_r_zt3(r_zt3=r_plasma)

    points_full = r_plasma.reshape(-1, 3)
    normals_full = nunit.reshape(-1, 3)
    target_full = Bsv.reshape(-1)

    # Subsample for speed.
    rng = np.random.default_rng(0)
    n_total = int(points_full.shape[0])
    n_sub = int(min(max(int(args.subsample), 16), n_total))
    idx = rng.choice(n_total, size=n_sub, replace=False)
    points = points_full[idx]
    normals = normals_full[idx]
    target = target_full[idx]

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

    I_use = I0
    if bool(args.optimize_currents_first):
        from regcoil_jax.coil_current_optimization import optimize_coil_currents_to_match_bnormal

        optI = optimize_coil_currents_to_match_bnormal(
            segs0,
            points=points,
            normals_unit=normals,
            target_bnormal=target,
            coil_currents0=I0,
            steps=int(args.current_steps),
            lr=float(args.current_lr),
            net_current_target=float(np.sum(I0)),
            seg_batch=int(args.seg_batch),
        )
        I_use = np.asarray(optI.coil_currents, dtype=float)

    write_makecoil_filaments(out_dir / "coils_initial", filaments_xyz=coils.filaments_xyz, coil_currents=I_use, nfp=nfp)

    # ------------------------------------------------------------
    # JAX rigid-transform optimization
    # ------------------------------------------------------------
    import jax
    import jax.numpy as jnp

    from regcoil_jax.optimize import minimize_adam

    seg_mid0 = jnp.asarray(segs0.seg_midpoints, dtype=jnp.float64)
    seg_dl0 = jnp.asarray(segs0.seg_dls, dtype=jnp.float64)
    seg_fid = jnp.asarray(segs0.seg_filament, dtype=jnp.int32)
    n_fil = int(segs0.n_filaments)

    Ij = jnp.asarray(I_use, dtype=jnp.float64)
    ptsj = jnp.asarray(points, dtype=jnp.float64)
    nj = jnp.asarray(normals, dtype=jnp.float64)
    tj = jnp.asarray(target, dtype=jnp.float64)

    def rotate_z(x: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
        c = jnp.cos(phi)
        s = jnp.sin(phi)
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        xr0 = c * x0 - s * x1
        xr1 = s * x0 + c * x1
        return jnp.stack([xr0, xr1, x2], axis=1)

    def loss(p: jnp.ndarray) -> jnp.ndarray:
        # p = [phi, tx, ty, tz]
        phi = p[0]
        shift = p[1:4]
        mid = rotate_z(seg_mid0, phi) + shift[None, :]
        dl = rotate_z(seg_dl0, phi)
        segs = FilamentSegments(seg_midpoints=mid, seg_dls=dl, seg_filament=seg_fid, n_filaments=n_fil)
        b = bnormal_from_segments(segs, points=ptsj, normals_unit=nj, filament_currents=Ij, seg_batch=int(args.seg_batch))
        mse = jnp.mean((b - tj) ** 2)
        reg = float(args.rot_reg) * (phi * phi) + float(args.shift_reg) * jnp.sum(shift * shift)
        return mse + reg

    p0 = jnp.zeros((4,), dtype=jnp.float64)
    opt = minimize_adam(loss, p0, steps=int(args.steps), lr=float(args.lr), jit=True)
    p1 = np.asarray(opt.x, dtype=float)
    print(f"[rigid] optimized params: phi_z={p1[0]:.6e} rad  shift=({p1[1]:.6e},{p1[2]:.6e},{p1[3]:.6e}) m", flush=True)

    # Report MSE on full plasma grid.
    def apply_to_points_np(xyz: np.ndarray, phi: float, shift: np.ndarray) -> np.ndarray:
        c = float(np.cos(phi))
        s = float(np.sin(phi))
        out = np.array(xyz, copy=True)
        x = xyz[:, 0]
        y = xyz[:, 1]
        out[:, 0] = c * x - s * y
        out[:, 1] = s * x + c * y
        out[:, 2] = xyz[:, 2]
        out = out + shift[None, :]
        return out

    seg_mid_al = np.asarray(apply_to_points_np(np.asarray(segs0.seg_midpoints), float(p1[0]), p1[1:4]))
    seg_dl_al = np.asarray(apply_to_points_np(np.asarray(segs0.seg_dls), float(p1[0]), np.zeros((3,), dtype=float)))
    segs_al = FilamentSegments(
        seg_midpoints=jnp.asarray(seg_mid_al, dtype=jnp.float64),
        seg_dls=jnp.asarray(seg_dl_al, dtype=jnp.float64),
        seg_filament=seg_fid,
        n_filaments=n_fil,
    )

    b0 = np.asarray(bnormal_from_segments(segs0, points=points_full, normals_unit=normals_full, filament_currents=Ij, seg_batch=int(args.seg_batch)))
    b1 = np.asarray(bnormal_from_segments(segs_al, points=points_full, normals_unit=normals_full, filament_currents=Ij, seg_batch=int(args.seg_batch)))
    mse0 = float(np.mean((b0 - target_full) ** 2))
    mse1 = float(np.mean((b1 - target_full) ** 2))
    print(f"[report] MSE(Bn) on full plasma grid: before={mse0:.6e} after={mse1:.6e}", flush=True)

    # ------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------
    if not args.no_figures:
        plt = _setup_matplotlib()

        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.plot(np.asarray(opt.loss_history), "-k", lw=1.6)
        ax.set_xlabel("Adam step")
        ax.set_ylabel("loss (MSE + regularization)")
        ax.set_title("Rigid coilset alignment optimization")
        fig.tight_layout()
        fig.savefig(out_dir / "loss_history.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.hist(b0 - target_full, bins=60, alpha=0.55, label="before", density=True)
        ax.hist(b1 - target_full, bins=60, alpha=0.55, label="after", density=True)
        ax.set_xlabel(r"$(B_n^{filaments} - B_n^{target})$ [T]")
        ax.set_ylabel("density")
        ax.legend(loc="best")
        ax.set_title("Residual distribution on plasma surface")
        fig.tight_layout()
        fig.savefig(out_dir / "bn_residual_hist.png")
        plt.close(fig)

    # ------------------------------------------------------------
    # ParaView outputs
    # ------------------------------------------------------------
    if not args.no_vtk:
        from regcoil_jax.vtk_io import write_vtp_polydata, write_vtu_point_cloud
        from regcoil_jax.fieldlines_jax import trace_fieldlines_rk4, soft_poincare_candidates

        vtk_dir = out_dir / "vtk"
        vtk_dir.mkdir(exist_ok=True)

        def write_coils_vtp(name: str, *, phi: float, shift: np.ndarray):
            pts_all = []
            lines = []
            offset = 0
            for filament in coils.filaments_xyz:
                filament = np.asarray(filament, dtype=float)
                filament = apply_to_points_np(filament, phi, shift)
                n = filament.shape[0]
                pts_all.append(filament)
                lines.append(list(range(offset, offset + n)) + [offset])
                offset += n
            pts_all = np.concatenate(pts_all, axis=0)
            write_vtp_polydata(vtk_dir / name, points=pts_all, lines=lines, cell_data={"coil_current": np.asarray(I_use, dtype=float)})

        write_coils_vtp("coils_before.vtp", phi=0.0, shift=np.zeros((3,), dtype=float))
        write_coils_vtp("coils_after.vtp", phi=float(p1[0]), shift=np.asarray(p1[1:4], dtype=float))

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
            segs_al,
            starts=starts,
            filament_currents=I_use,
            ds=float(args.fieldline_ds),
            n_steps=int(args.fieldline_steps),
            stop_radius=12.0,
            seg_batch=int(args.seg_batch),
        )
        pts = np.asarray(trace.points)
        pts_all = []
        lines = []
        offset = 0
        for j in range(pts.shape[0]):
            line = pts[j]
            pts_all.append(line)
            lines.append(list(range(offset, offset + line.shape[0])))
            offset += line.shape[0]
        pts_all = np.concatenate(pts_all, axis=0)
        write_vtp_polydata(vtk_dir / "fieldlines_after.vtp", points=pts_all, lines=lines)

        cand, w = soft_poincare_candidates(pts, nfp=int(nfp), phi0=float(args.poincare_phi0))
        cand = np.asarray(cand).reshape(-1, 3)
        w = np.asarray(w).reshape(-1)
        write_vtu_point_cloud(vtk_dir / "poincare_candidates_soft.vtu", points=cand, point_data={"weight": w})

    print(f"[done] outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()

