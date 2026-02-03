#!/usr/bin/env python3
"""Compare coils with and without winding-surface optimization (REGCOIL_JAX + coil cutting).

This pedagogic end-to-end example demonstrates several JAX-native workflows that go beyond the
original Fortran REGCOIL example set:

1) Run REGCOIL_JAX on a VMEC plasma boundary with a *baseline* winding surface.
2) Optimize the winding surface via autodiff (a spatially varying separation field sep(θ,ζ)).
3) Run REGCOIL_JAX again on the optimized winding surface.
4) Cut filamentary coils from the current potential (contours of Φ).
5) Optimize *per-coil currents* after cutting so B·n on the plasma surface stays small.
6) Trace field lines and produce Poincaré plots, overlaid with the target plasma surface.

Outputs:
  - publication-ready figures (.png)
  - ParaView-ready VTK files (.vts/.vtp)

Everything is written to an `outputs_*` folder under this directory (gitignored).

Notes:
  - Cutting coils is not differentiable (contouring), so we treat coil geometry as fixed after cutting.
    The per-coil current optimization is differentiable and uses JAX autodiff through Biot–Savart.
  - The Poincaré plots shown are for the *coil filament approximation* (coil-only field).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from typing import Any

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
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _read_run(nc_path: Path, *, ilambda: int = -1) -> dict[str, np.ndarray]:
    if netCDF4 is None:  # pragma: no cover
        raise RuntimeError("netCDF4 is required")
    ds = netCDF4.Dataset(str(nc_path), "r")
    try:
        lambdas = np.asarray(ds.variables["lambda"][:], dtype=float)
        if ilambda >= 0:
            ilam = int(ilambda)
        else:
            ilam = int(ds.variables["chosen_idx"][()]) if "chosen_idx" in ds.variables else int(len(lambdas) - 1)

        out = dict(
            nfp=int(ds.variables["nfp"][()]),
            ilam=ilam,
            lambdas=lambdas,
            chi2_B=np.asarray(ds.variables["chi2_B"][:], dtype=float),
            chi2_K=np.asarray(ds.variables["chi2_K"][:], dtype=float),
            max_B=np.asarray(ds.variables["max_Bnormal"][:], dtype=float),
            max_K=np.asarray(ds.variables["max_K"][:], dtype=float),
            theta_p=np.asarray(ds.variables["theta_plasma"][:], dtype=float),
            zeta_p=np.asarray(ds.variables["zeta_plasma"][:], dtype=float),
            theta_c=np.asarray(ds.variables["theta_coil"][:], dtype=float),
            zeta_c=np.asarray(ds.variables["zeta_coil"][:], dtype=float),
            r_plasma=np.asarray(ds.variables["r_plasma"][:], dtype=float),  # (nzetal,ntheta,3)
            r_coil=np.asarray(ds.variables["r_coil"][:], dtype=float),  # (nzetal,ntheta,3)
            Phi=np.asarray(ds.variables["current_potential"][:], dtype=float),  # (nlambda,nzeta,ntheta)
            Bn_res=np.asarray(ds.variables["Bnormal_total"][:], dtype=float),  # (nlambda,nzeta,ntheta)
            net_pol=float(ds.variables["net_poloidal_current_Amperes"][()]),
        )
        if "normal_plasma" in ds.variables:
            out["normal_plasma"] = np.asarray(ds.variables["normal_plasma"][:], dtype=float)
        else:
            # save_level=3 does not write normals; for this script, require normals.
            raise RuntimeError("normal_plasma not present in output; set save_level < 3 in the generated inputs.")
        return out
    finally:
        ds.close()


def _plasma_points_and_unit_normals(run: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(run["r_plasma"], dtype=float).reshape(-1, 3)
    N = np.asarray(run["normal_plasma"], dtype=float).reshape(-1, 3)
    n = np.linalg.norm(N, axis=1)
    n = np.where(n == 0.0, 1.0, n)
    nhat = N / n[:, None]
    return pts, nhat


def _cross_section_curve_rz(*, r_plasma: np.ndarray, phi0: float) -> tuple[np.ndarray, np.ndarray]:
    """Approximate the R-Z boundary curve at toroidal plane phi0 by selecting the closest zeta index."""
    r = np.asarray(r_plasma, dtype=float)
    phi = np.arctan2(r[:, :, 1], r[:, :, 0])  # (nzetal,ntheta)
    # Find zeta index with mean phi closest to phi0.
    phi_mean = np.unwrap(np.mean(phi, axis=1))
    idx = int(np.argmin(np.abs(((phi_mean - float(phi0) + np.pi) % (2.0 * np.pi)) - np.pi)))
    x = r[idx, :, 0]
    y = r[idx, :, 1]
    R = np.sqrt(x * x + y * y)
    Z = r[idx, :, 2]
    return R, Z


def main() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    examples_dir = here.parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", type=str, default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--wout", type=str, default=str(examples_dir / "wout_d23p4_tm.nc"))
    ap.add_argument("--out_dir", type=str, default=None)

    # Winding surface optimization (kept small by default).
    ap.add_argument("--opt_steps", type=int, default=25)
    ap.add_argument("--opt_step_size", type=float, default=0.05)
    ap.add_argument("--opt_ntheta", type=int, default=16)
    ap.add_argument("--opt_nzeta", type=int, default=16)
    ap.add_argument("--opt_mpol_sep", type=int, default=3)
    ap.add_argument("--opt_ntor_sep", type=int, default=3)
    ap.add_argument("--separation0", type=float, default=0.5)
    ap.add_argument("--separation_min", type=float, default=0.05)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0e-14, help="lambda used inside winding-surface objective")

    # REGCOIL solve setup for the before/after comparisons.
    ap.add_argument("--eval_ntheta", type=int, default=64)
    ap.add_argument("--eval_nzeta", type=int, default=64)
    ap.add_argument("--eval_nlambda", type=int, default=8)
    ap.add_argument("--eval_lambda_min", type=float, default=1.0e-19)
    ap.add_argument("--eval_lambda_max", type=float, default=1.0e-13)
    ap.add_argument("--mpol_potential", type=int, default=12)
    ap.add_argument("--ntor_potential", type=int, default=12)

    # Coil cutting and current optimization.
    ap.add_argument("--coils_per_half_period", type=int, default=6)
    ap.add_argument("--theta_shift", type=int, default=0)
    ap.add_argument("--current_opt_steps", type=int, default=200)
    ap.add_argument("--current_opt_lr", type=float, default=2e-2)
    ap.add_argument(
        "--lambda_select",
        type=str,
        default="coil_bn_rms",
        choices=["none", "coil_bn_rms", "coil_bn_rms_plus_maxK"],
        help="How to choose the lambda index used for cutting coils (demonstrates 'winding surface current' selection).",
    )
    ap.add_argument("--lambda_select_steps", type=int, default=60, help="Short current-optimization steps for lambda selection scan.")
    ap.add_argument("--lambda_select_alpha", type=float, default=1.0e-3, help="Penalty weight for max_K when using coil_bn_rms_plus_maxK.")

    # Field line / Poincaré visualization.
    ap.add_argument("--phi0", type=float, default=0.0, help="Poincaré plane toroidal angle (radians)")
    ap.add_argument("--fieldline_ds", type=float, default=0.03)
    ap.add_argument("--fieldline_steps", type=int, default=1200)
    ap.add_argument("--poincare_max_points", type=int, default=600)
    ap.add_argument("--n_starts", type=int, default=10, help="Number of field line start points between axis and boundary")

    args = ap.parse_args()

    if netCDF4 is None:  # pragma: no cover
        raise SystemExit("netCDF4 is required (install regcoil_jax[viz])")

    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("JAX_PLATFORM_NAME", args.platform)

    wout_src = Path(args.wout).resolve()
    if not wout_src.exists():
        raise SystemExit(f"Missing wout file: {wout_src}")

    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = examples_dir / f"outputs_compare_wsopt_{stamp}"
    else:
        out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy wout into the output directory so all generated inputs can use relative paths.
    wout_dst = out_dir / wout_src.name
    if wout_dst != wout_src:
        wout_dst.write_bytes(wout_src.read_bytes())

    from regcoil_jax.run import run_regcoil
    from regcoil_jax.winding_surface_optimization import (
        SeparationFieldOptConfig,
        optimize_vmec_offset_separation_field,
        write_optimized_winding_surface_nescin,
    )
    import jax.numpy as jnp

    cfg = SeparationFieldOptConfig(
        mpol_sep=int(args.opt_mpol_sep),
        ntor_sep=int(args.opt_ntor_sep),
        separation_min=float(args.separation_min),
    )

    # -----
    # 1) Optimize sep(θ,ζ) (autodiff)
    # -----
    print("[opt] optimizing winding surface sep(θ,ζ)...", flush=True)
    res = optimize_vmec_offset_separation_field(
        wout_filename=str(wout_dst),
        separation0=float(args.separation0),
        nsteps=int(args.opt_steps),
        step_size=float(args.opt_step_size),
        ntheta=int(args.opt_ntheta),
        nzeta=int(args.opt_nzeta),
        mpol_potential=min(int(args.mpol_potential), 6),
        ntor_potential=min(int(args.ntor_potential), 6),
        lam=float(args.lam),
        config=cfg,
    )

    # Baseline coefficients correspond to a constant separation field.
    sep0_eff = max(float(args.separation0) - float(cfg.separation_min), 1e-6)
    base_baseline = float(np.log(np.expm1(sep0_eff) + 1e-300))
    nmodes = int(res.coeff_sin_history.shape[1])
    zeros = jnp.zeros((nmodes,), dtype=jnp.float64)

    base_opt = float(res.base_history[-1])
    cs_opt = res.coeff_sin_history[-1]
    cc_opt = res.coeff_cos_history[-1]

    nescin_before = out_dir / "nescin.wsopt_baseline"
    nescin_after = out_dir / "nescin.wsopt_optimized"
    write_optimized_winding_surface_nescin(
        path=str(nescin_before),
        wout_filename=str(wout_dst),
        base=base_baseline,
        coeff_sin=zeros,
        coeff_cos=zeros,
        config=cfg,
        ntheta=int(args.eval_ntheta),
        nzeta=int(args.eval_nzeta),
    )
    write_optimized_winding_surface_nescin(
        path=str(nescin_after),
        wout_filename=str(wout_dst),
        base=base_opt,
        coeff_sin=cs_opt,
        coeff_cos=cc_opt,
        config=cfg,
        ntheta=int(args.eval_ntheta),
        nzeta=int(args.eval_nzeta),
    )

    # -----
    # 2) Run REGCOIL_JAX on both winding surfaces (use nescin surface input)
    # -----
    def write_input(path: Path, *, nescin_name: str):
        txt = f"""! Autogenerated by {here.name}
&regcoil_nml
  general_option = 1
  save_level = 2
  regularization_term_option = "chi2_K"

  nlambda = {int(args.eval_nlambda)}
  lambda_min = {float(args.eval_lambda_min):.6e}
  lambda_max = {float(args.eval_lambda_max):.6e}

  ntheta_plasma = {int(args.eval_ntheta)}
  nzeta_plasma  = {int(args.eval_nzeta)}
  ntheta_coil   = {int(args.eval_ntheta)}
  nzeta_coil    = {int(args.eval_nzeta)}

  mpol_potential = {int(args.mpol_potential)}
  ntor_potential = {int(args.ntor_potential)}
  symmetry_option = 1

  geometry_option_plasma = 2
  wout_filename = "{wout_dst.name}"

  geometry_option_coil = 3
  nescin_filename = "{nescin_name}"
/
"""
        path.write_text(txt, encoding="utf-8")

    in_before = out_dir / "regcoil_in.wsopt_baseline"
    in_after = out_dir / "regcoil_in.wsopt_optimized"
    write_input(in_before, nescin_name=nescin_before.name)
    write_input(in_after, nescin_name=nescin_after.name)

    print("[run] regcoil_jax baseline...", flush=True)
    run_regcoil(str(in_before), verbose=True)
    print("[run] regcoil_jax optimized winding surface...", flush=True)
    run_regcoil(str(in_after), verbose=True)

    out_before = out_dir / "regcoil_out.wsopt_baseline.nc"
    out_after = out_dir / "regcoil_out.wsopt_optimized.nc"

    run_b = _read_run(out_before, ilambda=-1)
    run_a = _read_run(out_after, ilambda=-1)

    # -----
    # 3) Cut coils and optimize per-coil currents to minimize B·n on the plasma surface.
    # -----
    from regcoil_jax.coil_cutting import cut_coils_from_current_potential
    from regcoil_jax.biot_savart_jax import segments_from_filaments, bnormal_from_segments
    from regcoil_jax.coil_current_optimization import optimize_coil_currents_to_match_bnormal

    def cut_and_optimize(tag: str, run: dict[str, np.ndarray]) -> dict[str, Any]:
        nfp = int(run["nfp"])
        pts, nhat = _plasma_points_and_unit_normals(run)
        target = np.zeros((pts.shape[0],), dtype=float)

        # Optionally choose lambda based on post-cut performance (discretization-aware).
        ilam = int(run["ilam"])
        scan_rms = None
        scan_score = None
        if args.lambda_select != "none":
            scan_rms = np.full((run["Phi"].shape[0],), np.nan, dtype=float)
            scan_score = np.full((run["Phi"].shape[0],), np.nan, dtype=float)

            for j in range(int(run["Phi"].shape[0])):
                cut_j = cut_coils_from_current_potential(
                    current_potential_zt=run["Phi"][j],
                    theta=run["theta_c"],
                    zeta=run["zeta_c"],
                    r_coil_zt3_full=run["r_coil"],
                    theta_shift=int(args.theta_shift),
                    coils_per_half_period=int(args.coils_per_half_period),
                    nfp=int(nfp),
                    net_poloidal_current_Amperes=float(run["net_pol"]),
                )
                segs_j = segments_from_filaments(filaments_xyz=cut_j.filaments_xyz)
                opt_j = optimize_coil_currents_to_match_bnormal(
                    segs_j,
                    points=pts,
                    normals_unit=nhat,
                    target_bnormal=target,
                    coil_currents0=cut_j.coil_currents,
                    steps=int(args.lambda_select_steps),
                    lr=float(args.current_opt_lr),
                    l2_reg=1e-6,
                    seg_batch=2048,
                )
                bn_j = np.asarray(
                    bnormal_from_segments(
                        segs_j,
                        points=pts,
                        normals_unit=nhat,
                        filament_currents=opt_j.coil_currents,
                        seg_batch=2048,
                    )
                )
                scan_rms[j] = float(np.sqrt(np.mean(bn_j * bn_j)))
                if args.lambda_select == "coil_bn_rms_plus_maxK":
                    scan_score[j] = scan_rms[j] + float(args.lambda_select_alpha) * float(run["max_K"][j])
                else:
                    scan_score[j] = scan_rms[j]

            ilam = int(np.nanargmin(scan_score))
            print(f"[{tag}] lambda selection ({args.lambda_select}): ilambda={ilam} lambda={float(run['lambdas'][ilam]):.3e} score={float(scan_score[ilam]):.3e}")

        # Cut coils at the selected lambda index.
        Phi_zt = run["Phi"][ilam]  # (nzeta,ntheta)
        cut = cut_coils_from_current_potential(
            current_potential_zt=Phi_zt,
            theta=run["theta_c"],
            zeta=run["zeta_c"],
            r_coil_zt3_full=run["r_coil"],
            theta_shift=int(args.theta_shift),
            coils_per_half_period=int(args.coils_per_half_period),
            nfp=int(nfp),
            net_poloidal_current_Amperes=float(run["net_pol"]),
        )
        segs = segments_from_filaments(filaments_xyz=cut.filaments_xyz)

        bn0 = np.asarray(
            bnormal_from_segments(
                segs,
                points=pts,
                normals_unit=nhat,
                filament_currents=cut.coil_currents,
                seg_batch=2048,
            )
        )
        rms0 = float(np.sqrt(np.mean(bn0 * bn0)))
        max0 = float(np.max(np.abs(bn0)))

        opt = optimize_coil_currents_to_match_bnormal(
            segs,
            points=pts,
            normals_unit=nhat,
            target_bnormal=target,
            coil_currents0=cut.coil_currents,
            steps=int(args.current_opt_steps),
            lr=float(args.current_opt_lr),
            l2_reg=1e-6,
            seg_batch=2048,
        )
        bn1 = np.asarray(
            bnormal_from_segments(
                segs,
                points=pts,
                normals_unit=nhat,
                filament_currents=opt.coil_currents,
                seg_batch=2048,
            )
        )
        rms1 = float(np.sqrt(np.mean(bn1 * bn1)))
        max1 = float(np.max(np.abs(bn1)))
        print(f"[{tag}] coils={len(cut.filaments_xyz)}  Bn_rms: {rms0:.3e} -> {rms1:.3e}   max|Bn|: {max0:.3e} -> {max1:.3e}")

        return dict(
            ilam=ilam,
            scan_rms=scan_rms,
            scan_score=scan_score,
            cut=cut,
            segs=segs,
            bn_equal=bn0,
            bn_opt=bn1,
            currents_opt=np.asarray(opt.coil_currents, dtype=float),
            rms_equal=rms0,
            rms_opt=rms1,
            max_equal=max0,
            max_opt=max1,
            loss_history=np.asarray(opt.loss_history, dtype=float),
        )

    b = cut_and_optimize("baseline", run_b)
    a = cut_and_optimize("wsopt", run_a)

    # -----
    # 4) Field lines + Poincaré overlay with the target plasma surface.
    # -----
    from regcoil_jax.fieldlines import build_filament_field_multi_current, trace_fieldlines, poincare_points
    from regcoil_jax.vtk_io import write_vtp_polydata, write_vts_structured_grid

    def trace_and_poincare(tag: str, run: dict[str, np.ndarray], result: dict[str, Any]) -> dict[str, Any]:
        fil_field = build_filament_field_multi_current(filaments_xyz=result["cut"].filaments_xyz, coil_currents=result["currents_opt"])

        Rb, Zb = _cross_section_curve_rz(r_plasma=run["r_plasma"], phi0=float(args.phi0))
        axis_R = float(np.mean(Rb))
        axis_Z = float(np.mean(Zb))
        # Choose a boundary point (outboard midplane-ish) to define a radial ray.
        j = int(np.argmax(Rb))
        R_edge = float(Rb[j])
        Z_edge = float(Zb[j])

        t = np.linspace(0.1, 0.98, int(args.n_starts))
        starts = np.stack(
            [
                axis_R + t * (R_edge - axis_R),
                np.zeros_like(t),
                axis_Z + t * (Z_edge - axis_Z),
            ],
            axis=1,
        )
        # Embed starts in the phi0 plane by rotating around z.
        c = np.cos(float(args.phi0))
        s = np.sin(float(args.phi0))
        x = starts[:, 0] * c
        y = starts[:, 0] * s
        starts_xyz = np.stack([x, y, starts[:, 2]], axis=1)

        lines = trace_fieldlines(
            fil_field,
            starts=starts_xyz,
            ds=float(args.fieldline_ds),
            n_steps=int(args.fieldline_steps),
            stop_radius=None,
        )
        pcs = poincare_points(lines, nfp=int(run["nfp"]), phi0=float(args.phi0), max_points_per_line=int(args.poincare_max_points))

        # Write VTK: plasma surface + coils + field lines + Poincaré points
        vtk_dir = out_dir / f"vtk_{tag}"
        vtk_dir.mkdir(exist_ok=True)

        # Plasma surface structured grid at full torus zeta resolution.
        write_vts_structured_grid(vtk_dir / "plasma_surface.vts", points_zt3=np.asarray(run["r_plasma"], dtype=float))

        # Coils as polylines.
        coil_pts = np.concatenate(result["cut"].filaments_xyz, axis=0)
        lines_conn = []
        off = 0
        for pts in result["cut"].filaments_xyz:
            n = int(np.asarray(pts).shape[0])
            lines_conn.append(list(range(off, off + n)) + [off])
            off += n
        write_vtp_polydata(vtk_dir / "coils.vtp", points=coil_pts, lines=lines_conn)

        # Field lines as polylines.
        fl_pts = np.concatenate(lines, axis=0)
        fl_conn = []
        off = 0
        for ln in lines:
            n = int(np.asarray(ln).shape[0])
            fl_conn.append(list(range(off, off + n)))
            off += n
        write_vtp_polydata(vtk_dir / "fieldlines.vtp", points=fl_pts, lines=fl_conn)

        # Poincaré points as a vertex cloud (each line separate file for easy coloring).
        for i, p in enumerate(pcs):
            if p.size == 0:
                continue
            write_vtp_polydata(vtk_dir / f"poincare_line_{i:02d}.vtp", points=p, verts=np.arange(p.shape[0], dtype=np.int64))

        return dict(lines=lines, poincare=pcs, boundary_R=Rb, boundary_Z=Zb)

    vis_b = trace_and_poincare("baseline", run_b, b)
    vis_a = trace_and_poincare("wsopt", run_a, a)

    # -----
    # 5) Figures
    # -----
    plt = _setup_matplotlib()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Optimization history (winding surface)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(np.asarray(res.objective_history), "-o", markersize=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective")
    ax.set_title("Winding surface optimization objective (autodiff)")
    fig.tight_layout()
    fig.savefig(fig_dir / "winding_surface_objective.png")
    plt.close(fig)

    # Lambda scans
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogx(run_b["lambdas"], run_b["max_K"], "-o", label="baseline max_K")
    ax.semilogx(run_a["lambdas"], run_a["max_K"], "-o", label="wsopt max_K")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\max |K|$")
    ax.set_title("Lambda scan comparison")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "lambda_scan_maxK_compare.png")
    plt.close(fig)

    # Poincaré plots overlaid with boundary
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True, sharey=True)
    for ax, tag, vis in [
        (axes[0], "baseline", vis_b),
        (axes[1], "winding-surface optimized", vis_a),
    ]:
        ax.plot(vis["boundary_R"], vis["boundary_Z"], "k-", linewidth=1.2, label="target surface (slice)")
        for pts in vis["poincare"]:
            if pts.size == 0:
                continue
            R = np.sqrt(pts[:, 0] * pts[:, 0] + pts[:, 1] * pts[:, 1])
            Z = pts[:, 2]
            ax.scatter(R, Z, s=3, alpha=0.55)
        ax.set_xlabel("R [m]")
        ax.set_title(f"Poincaré at $\\phi_0$={float(args.phi0):.2f} ({tag})")
    axes[0].set_ylabel("Z [m]")
    axes[0].legend(loc="best")
    fig.suptitle("Coil-only field lines (after per-coil current optimization)")
    fig.tight_layout()
    fig.savefig(fig_dir / "poincare_compare.png")
    plt.close(fig)

    # 3D summary (baseline vs optimized winding surface):
    # plasma surface point cloud + cut coils + a few traced field lines.
    # This figure is intentionally matplotlib-only (no ParaView required).
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    def _plot_3d(ax, *, run: dict[str, np.ndarray], result: dict[str, Any], vis: dict[str, Any], title: str) -> None:
        r = np.asarray(run["r_plasma"], dtype=float).reshape(-1, 3)
        stride = max(1, int(r.shape[0] // 7000))
        rp = r[::stride]
        ax.scatter(rp[:, 0], rp[:, 1], rp[:, 2], s=0.25, alpha=0.06, c="k", linewidths=0)

        for I, coil in zip(result["currents_opt"], result["cut"].filaments_xyz):
            c = "tab:red" if float(I) >= 0.0 else "tab:blue"
            coil = np.asarray(coil, dtype=float)
            ax.plot(coil[:, 0], coil[:, 1], coil[:, 2], color=c, linewidth=1.2, alpha=0.9)

        for ln in vis["lines"]:
            ln = np.asarray(ln, dtype=float)
            ax.plot(ln[:, 0], ln[:, 1], ln[:, 2], color="tab:green", linewidth=0.6, alpha=0.45)

        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_box_aspect([1.0, 1.0, 0.8])

    fig = plt.figure(figsize=(12.6, 5.6), constrained_layout=True)
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_3d(
        ax0,
        run=run_b,
        result=b,
        vis=vis_b,
        title=f"Baseline winding surface\nBn_rms={b['rms_opt']:.2e}, max|Bn|={b['max_opt']:.2e}",
    )
    _plot_3d(
        ax1,
        run=run_a,
        result=a,
        vis=vis_a,
        title=f"Optimized winding surface\nBn_rms={a['rms_opt']:.2e}, max|Bn|={a['max_opt']:.2e}",
    )
    fig.savefig(fig_dir / "wsopt_3d_before_after.png", dpi=220)
    plt.close(fig)

    print(f"[done] outputs: {out_dir}")


if __name__ == "__main__":
    main()
