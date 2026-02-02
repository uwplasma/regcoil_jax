#!/usr/bin/env python3
"""Permanent-magnet (dipole lattice) demo: cancel Bnormal_from_plasma_current using dipoles.

This example implements a REGCOIL-PM-like workflow:
  1) Read (or compute) B_plasma · n on the plasma boundary (e.g. from a BNORM file / VMEC).
  2) Place many dipoles ("magnets") on / near the winding surface.
  3) Solve for dipole moments to cancel B_plasma · n (least squares with ridge regularization).

This is differentiable w.r.t the dipole moments and can be used inside higher-level optimization loops.

Outputs:
  - figures (histogram + 2D map of residual B·n)
  - ParaView VTK (.vts for plasma surface; .vtp for dipoles as points with moment vectors)
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    here = Path(__file__).resolve()
    examples_dir = here.parent

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
    input_copy.write_text(input_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
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

    bn_mag = np.asarray(dipole_bnormal(points=points, normals_unit=normals_unit, positions=dip_pos, moments=sol.dipole_moments, batch=4096), dtype=float)
    resid = bn_mag + Bplasma.reshape(-1)
    print(f"[residual] RMS={np.sqrt(np.mean(resid*resid)):.3e}  max|Bn|={np.max(np.abs(resid)):.3e}  cg_info={sol.cg_info}")

    # ----------------------------
    # ParaView
    # ----------------------------
    vtk_dir = out_dir / "vtk"
    vtk_dir.mkdir(exist_ok=True)

    write_vts_structured_grid(vtk_dir / "plasma_surface.vts", points_zt3=r_plasma, point_data={"Bn_residual": resid.reshape(nzeta, -1)})
    write_vts_structured_grid(vtk_dir / "winding_surface.vts", points_zt3=r_coil, point_data={})

    write_vtp_polydata(
        vtk_dir / "dipoles.vtp",
        points=dip_pos,
        verts=np.arange(dip_pos.shape[0], dtype=np.int64),
        point_data={"m": sol.dipole_moments},
    )

    # ----------------------------
    # Figures
    # ----------------------------
    plt = _setup_matplotlib()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.hist(np.abs(resid), bins=80, alpha=0.85)
    ax.set_xlabel(r"$|B_{\\mathrm{plasma}}\\cdot n + B_{\\mathrm{PM}}\\cdot n|$ [T]")
    ax.set_ylabel("count")
    ax.set_title("Residual normal field after PM solve")
    fig.tight_layout()
    fig.savefig(fig_dir / "residual_hist.png")
    plt.close(fig)

    # 2D map on (zeta,theta) grid
    resid_zt = resid.reshape(nzeta, -1)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    im = ax.imshow(resid_zt, origin="lower", aspect="auto")
    ax.set_xlabel("theta index")
    ax.set_ylabel("zeta index")
    ax.set_title("Residual B·n on plasma surface")
    fig.colorbar(im, ax=ax, shrink=0.85, label="T")
    fig.tight_layout()
    fig.savefig(fig_dir / "residual_map.png")
    plt.close(fig)

    print(f"[done] outputs: {out_dir}")


if __name__ == "__main__":
    main()

