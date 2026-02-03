#!/usr/bin/env python3
"""Permanent-magnet (dipole lattice) demo: cancel Bnormal_from_plasma_current using dipoles.

This example implements a REGCOIL-PM-like workflow:
  1) Read (or compute) B_plasma · n on the plasma boundary (e.g. from a BNORM file / VMEC).
  2) Place many dipoles ("magnets") on / near the winding surface.
  3) Solve for dipole moments to cancel B_plasma · n (least squares with ridge regularization).

This is differentiable w.r.t the dipole moments and can be used inside higher-level optimization loops.

Outputs:
  - paper-style figures (before/after/residual B·n maps + histograms + magnet moment statistics)
  - ParaView VTK:
      - plasma/winding surfaces (`.vts`) with B·n fields
      - dipole points (`.vtp`) with moment vectors + scalar magnitudes (use Glyph in ParaView)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import re
import shutil

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


def main() -> None:
    here = Path(__file__).resolve()
    examples_dir = here.parent
    t0 = time.perf_counter()

    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", type=str, default="cpu", choices=["cpu", "gpu"])
    ap.add_argument(
        "--input",
        type=str,
        default=str(examples_dir / "regcoil_in.lambda_search_5_with_bnorm"),
        help="Input namelist (must have load_bnorm=.true. so Bnormal_from_plasma_current is available).",
    )
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--dipole_offset", type=float, default=0.35, help="offset distance from winding surface [m]")
    ap.add_argument("--dipole_stride", type=int, default=20, help="downsample stride for dipole placement")
    ap.add_argument("--l2_moment", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=200)
    args = ap.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("JAX_PLATFORM_NAME", args.platform)

    if netCDF4 is None:  # pragma: no cover
        raise SystemExit("netCDF4 is required for this example (pip install regcoil_jax[viz]).")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Missing input: {input_path}")

    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = examples_dir / f"outputs_permanent_magnets_{stamp}"
    else:
        out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run regcoil_jax once to get Bnormal_from_plasma_current and the plasma geometry grid.
    from regcoil_jax.run import run_regcoil

    input_copy = out_dir / input_path.name
    txt_src = input_path.read_text(encoding="utf-8", errors="ignore")

    # Copy dependencies next to the input copy (REGCOIL-style: paths relative to input dir).
    # This keeps each run self-contained, so outputs/VTK/figures can be shared.
    from regcoil_jax.utils import parse_namelist

    inputs = parse_namelist(str(input_path))
    txt_dst = txt_src

    def _copy_and_rewrite(key: str) -> None:
        nonlocal txt_dst
        val = inputs.get(key, None)
        if not val:
            return
        p = Path(str(val))
        if not p.is_absolute():
            p = (input_path.parent / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"{key}={val!r} resolved to {p}, but the file does not exist")
        dst = out_dir / p.name
        shutil.copy2(p, dst)
        # Rewrite only the first occurrence; keep the rest unchanged.
        txt_dst = re.sub(
            rf"({key}\s*=\s*['\"])([^'\"]+)(['\"])",
            rf"\1{p.name}\3",
            txt_dst,
            count=1,
            flags=re.IGNORECASE,
        )

    _copy_and_rewrite("wout_filename")
    _copy_and_rewrite("bnorm_filename")

    input_copy.write_text(txt_dst, encoding="utf-8")
    res = run_regcoil(str(input_copy), verbose=True)

    ds = netCDF4.Dataset(res.output_nc, "r")
    try:
        nfp = int(ds.variables["nfp"][()])
        theta_p = np.asarray(ds.variables["theta_plasma"][:], dtype=float)
        zeta_p = np.asarray(ds.variables["zeta_plasma"][:], dtype=float)
        Bplasma = np.asarray(ds.variables["Bnormal_from_plasma_current"][:], dtype=float)  # (nzeta, ntheta)
        r_plasma_full = np.asarray(ds.variables["r_plasma"][:], dtype=float)  # (nzetal, ntheta, 3)
        r_coil_full = np.asarray(ds.variables["r_coil"][:], dtype=float)  # (nzetal, ntheta, 3)
    finally:
        ds.close()

    nzeta = int(zeta_p.size)
    r_plasma = r_plasma_full[:nzeta]  # (nzeta, ntheta, 3) first field period
    r_coil = r_coil_full[:nzeta]

    from regcoil_jax.surface_utils import unit_normals_from_r_zt3
    from regcoil_jax.vtk_io import write_vts_structured_grid, write_vtp_polydata
    from regcoil_jax.permanent_magnets import place_dipoles_on_winding_surface, solve_dipole_moments_ridge_cg

    nunit = unit_normals_from_r_zt3(r_zt3=r_plasma)
    points = r_plasma.reshape(-1, 3)
    normals_unit = nunit.reshape(-1, 3)
    target_bnormal = (-Bplasma).reshape(-1)  # cancel B_plasma

    # Place dipoles on an offset winding surface.
    # Use winding-surface normals for placement, but target is evaluated on the plasma surface.
    nunit_coil = unit_normals_from_r_zt3(r_zt3=r_coil)
    dip_pos, _ = place_dipoles_on_winding_surface(
        surface_points=r_coil.reshape(-1, 3),
        surface_normals_unit=nunit_coil.reshape(-1, 3),
        offset=float(args.dipole_offset),
        stride=int(args.dipole_stride),
    )

    sol = solve_dipole_moments_ridge_cg(
        points=points,
        normals_unit=normals_unit,
        dipole_positions=dip_pos,
        target_bnormal=target_bnormal,
        l2_moment=float(args.l2_moment),
        maxiter=int(args.maxiter),
    )

    # Residual diagnostics.
    from regcoil_jax.dipoles import dipole_bnormal

    bn_pm = np.asarray(
        dipole_bnormal(points=points, normals_unit=normals_unit, positions=dip_pos, moments=sol.dipole_moments, batch=4096),
        dtype=float,
    )
    bn_plasma = Bplasma.reshape(-1)
    resid = bn_pm + bn_plasma
    resid_rms = float(np.sqrt(np.mean(resid * resid)))
    resid_maxabs = float(np.max(np.abs(resid)))
    print(f"[residual] RMS={resid_rms:.3e}  max|Bn|={resid_maxabs:.3e}  cg_info={sol.cg_info}")

    # ----------------------------
    # ParaView
    # ----------------------------
    vtk_dir = out_dir / "vtk"
    vtk_dir.mkdir(exist_ok=True)

    write_vts_structured_grid(
        vtk_dir / "plasma_surface.vts",
        points_zt3=r_plasma,
        point_data={
            "Bn_plasma": bn_plasma.reshape(nzeta, -1),
            "Bn_pm": bn_pm.reshape(nzeta, -1),
            "Bn_residual": resid.reshape(nzeta, -1),
        },
    )
    write_vts_structured_grid(vtk_dir / "winding_surface.vts", points_zt3=r_coil, point_data={})

    # Moments and summary scalars for ParaView coloring.
    m = np.asarray(sol.dipole_moments, dtype=float)
    m_mag = np.linalg.norm(m, axis=1)
    # Dipole placement uses a stride on the flattened (zeta,theta) winding surface grid.
    n_coil_pts = int(r_coil.reshape(-1, 3).shape[0])
    sel = np.arange(0, n_coil_pts, int(args.dipole_stride), dtype=int)
    nh_sel = nunit_coil.reshape(-1, 3)[sel]
    m_dot_n = np.sum(m * nh_sel, axis=1)

    write_vtp_polydata(
        vtk_dir / "dipoles.vtp",
        points=dip_pos,
        verts=np.arange(dip_pos.shape[0], dtype=np.int64),
        point_data={
            "m": m,
            "m_mag": m_mag,
            "m_dot_ncoil": m_dot_n,
        },
    )

    # ----------------------------
    # Figures
    # ----------------------------
    plt = _setup_matplotlib()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.hist(np.abs(resid), bins=80, alpha=0.85)
    ax.set_xlabel(r"$|B_{\mathrm{plasma}}\cdot n + B_{\mathrm{PM}}\cdot n|$ [T]")
    ax.set_ylabel("count")
    ax.set_title("Residual normal field after PM solve")
    fig.tight_layout()
    fig.savefig(fig_dir / "residual_hist.png")
    plt.close(fig)

    # Paper-style: before/after/residual maps with a shared color scale.
    bn_plasma_zt = bn_plasma.reshape(nzeta, -1)
    bn_pm_zt = bn_pm.reshape(nzeta, -1)
    resid_zt = resid.reshape(nzeta, -1)
    vmax = float(np.max(np.abs(bn_plasma_zt)))
    vmax = max(vmax, float(np.max(np.abs(bn_pm_zt))))
    vmax = max(vmax, float(np.max(np.abs(resid_zt))))
    vmax = float(vmax) if vmax > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    im0 = axes[0].imshow(bn_plasma_zt, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title(r"$B_{\mathrm{plasma}}\cdot n$")
    axes[0].set_xlabel("theta index")
    axes[0].set_ylabel("zeta index")

    axes[1].imshow(bn_pm_zt, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title(r"$B_{\mathrm{PM}}\cdot n$")
    axes[1].set_xlabel("theta index")
    axes[1].set_ylabel("zeta index")

    axes[2].imshow(resid_zt, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[2].set_title(r"$B_{\mathrm{plasma}}\cdot n + B_{\mathrm{PM}}\cdot n$")
    axes[2].set_xlabel("theta index")
    axes[2].set_ylabel("zeta index")

    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.set_label("T")
    fig.savefig(fig_dir / "bn_before_after_residual.png")
    plt.close(fig)

    # Magnet moment magnitude statistics.
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.hist(m_mag, bins=80, alpha=0.85)
    ax.set_xlabel("|m| [A·m^2]")
    ax.set_ylabel("count")
    ax.set_title("Dipole moment magnitudes")
    fig.tight_layout()
    fig.savefig(fig_dir / "moment_magnitude_hist.png")
    plt.close(fig)

    # Sparse (zeta,theta) scatter of dipole magnitudes on the winding-surface grid.
    ntheta_c = int(r_coil.shape[1])
    it = (sel % ntheta_c).astype(int)
    iz = (sel // ntheta_c).astype(int)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    sc = ax.scatter(it, iz, c=m_mag, s=12, cmap="viridis")
    ax.set_xlabel("theta index")
    ax.set_ylabel("zeta index")
    ax.set_title("Dipole placement (colored by |m|)")
    fig.colorbar(sc, ax=ax, shrink=0.85, label=r"A·m$^2$")
    fig.tight_layout()
    fig.savefig(fig_dir / "dipole_magnitude_scatter.png")
    plt.close(fig)

    # ----------------------------
    # Run summary (for reproducibility / paper-style reporting)
    # ----------------------------
    summary = {
        "input": str(input_path),
        "output_nc": str(res.output_nc),
        "output_log": str(res.output_log),
        "nfp": int(nfp),
        "ntheta": int(theta_p.size),
        "nzeta": int(zeta_p.size),
        "dipole_offset_m": float(args.dipole_offset),
        "dipole_stride": int(args.dipole_stride),
        "l2_moment": float(args.l2_moment),
        "maxiter": int(args.maxiter),
        "n_dipoles": int(dip_pos.shape[0]),
        "residual_rms_T": resid_rms,
        "residual_maxabs_T": resid_maxabs,
        "cg_info": sol.cg_info,
        "runtime_s": float(time.perf_counter() - t0),
        "artifacts": {
            "vtk": str(vtk_dir),
            "figures": str(fig_dir),
            "input_copy": str(input_copy),
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[done] outputs: {out_dir}")


if __name__ == "__main__":
    main()
