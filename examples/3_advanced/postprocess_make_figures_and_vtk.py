#!/usr/bin/env python3
"""Postprocess a REGCOIL_JAX run into publication-ready figures + ParaView files.

This script is intentionally *pedagogic* and meant to be copied/adapted.

It:
1) (optionally) runs `regcoil_jax` on a `regcoil_in.*` input file
2) reads the resulting `regcoil_out.*.nc`
3) produces:
   - figures (lambda scan + 2D maps of B_n, K, and current potential)
   - a MAKECOIL-style `coils.*` filament file (contours of current potential)
   - VTK PolyData `.vtp` files for ParaView (winding surface, plasma surface, coils, field lines)

Notes:
- Field lines traced here are for the *coil filament* approximation (coil-only field).
- For VMEC-based examples, this is usually sufficient for qualitative visualization.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import netCDF4
except Exception as e:  # pragma: no cover
    netCDF4 = None


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _read_var(ds, name: str) -> np.ndarray:
    if name not in ds.variables:
        raise KeyError(f"Missing variable {name!r} in {ds.filepath()}")
    return np.asarray(ds.variables[name][:])

def _normals_from_r_zt3(*, r_zt3: np.ndarray, nfp: int) -> np.ndarray:
    """Compute non-unit surface normals from a periodic (zeta,theta) grid of points.

    This is used when `normal_plasma` is not present in the netCDF output (REGCOIL save_level=3).
    """
    r = np.asarray(r_zt3, dtype=float)
    if r.ndim != 3 or r.shape[2] != 3:
        raise ValueError(f"Expected r_zt3 shape (nzetal,ntheta,3), got {r.shape}")
    nzetal, ntheta, _ = r.shape
    # Uniform grids: theta spans [0,2π), zeta spans [0,2π) across all field periods.
    dtheta = (2.0 * np.pi) / ntheta
    dzeta = (2.0 * np.pi) / nzetal
    dr_dtheta = (np.roll(r, -1, axis=1) - np.roll(r, 1, axis=1)) / (2.0 * dtheta)
    dr_dzeta = (np.roll(r, -1, axis=0) - np.roll(r, 1, axis=0)) / (2.0 * dzeta)
    # Match REGCOIL convention: normal = dr/dzeta × dr/dtheta (non-unit).
    nvec = np.cross(dr_dzeta, dr_dtheta)
    return nvec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).with_name("regcoil_in.lambda_search_1")),
        help="Path to regcoil_in.* file",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run regcoil_jax before postprocessing (recommended).",
    )
    parser.add_argument("--platform", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--coils_per_half_period", type=int, default=6)
    parser.add_argument("--theta_shift", type=int, default=0)
    parser.add_argument(
        "--ilambda",
        type=int,
        default=-1,
        help="Lambda index to visualize (-1: chosen_idx if present, else last)",
    )
    parser.add_argument("--fieldline_count", type=int, default=8)
    parser.add_argument("--fieldline_steps", type=int, default=800)
    parser.add_argument("--fieldline_ds", type=float, default=0.03)
    parser.add_argument("--fieldline_offset", type=float, default=0.15)
    parser.add_argument("--write_vtu_point_cloud", action="store_true", help="Write a VTU point cloud of |B| from coil filaments.")
    parser.add_argument("--vtu_n", type=int, default=14, help="Resolution per axis for VTU point cloud (N^3 points).")
    parser.add_argument("--vtu_margin", type=float, default=0.6, help="Bounding-box margin (meters) for VTU point cloud.")
    parser.add_argument("--no_figures", action="store_true", help="Skip writing matplotlib figures.")
    parser.add_argument("--no_vtk", action="store_true", help="Skip writing ParaView VTK files.")
    parser.add_argument("--no_coils", action="store_true", help="Skip coil cutting (and thus coils.* output).")
    parser.add_argument("--no_fieldlines", action="store_true", help="Skip coil-only field line tracing.")
    args = parser.parse_args()

    if netCDF4 is None:  # pragma: no cover
        raise SystemExit("netCDF4 is required for this script")

    input_path = Path(args.input).resolve()
    if not input_path.name.startswith("regcoil_in."):
        raise SystemExit("Input must be named regcoil_in.*")

    case = input_path.name[len("regcoil_in.") :]
    out_nc = input_path.with_name(f"regcoil_out.{case}.nc")

    if args.run:
        cmd = [sys.executable, "-m", "regcoil_jax.cli", "--platform", args.platform, "--verbose", str(input_path)]
        print("[postprocess] running:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), check=True)

    if not out_nc.exists():
        raise SystemExit(f"Missing output file: {out_nc}")

    out_dir = input_path.parent
    fig_dir = out_dir / f"figures_{case}"
    vtk_dir = out_dir / f"vtk_{case}"
    if not args.no_figures:
        fig_dir.mkdir(exist_ok=True)
    if not args.no_vtk:
        vtk_dir.mkdir(exist_ok=True)

    # ----------------------------
    # Read output netCDF
    # ----------------------------
    ds = netCDF4.Dataset(str(out_nc), "r")

    lambdas = _read_var(ds, "lambda").astype(float)
    chi2_B = _read_var(ds, "chi2_B").astype(float)
    chi2_K = _read_var(ds, "chi2_K").astype(float)
    max_B = _read_var(ds, "max_Bnormal").astype(float)
    max_K = _read_var(ds, "max_K").astype(float)

    nfp = int(np.asarray(ds.variables["nfp"][()]))
    net_pol = float(np.asarray(ds.variables["net_poloidal_current_Amperes"][()]))

    # Indices / selection
    if args.ilambda >= 0:
        ilam = int(args.ilambda)
    else:
        ilam = int(np.asarray(ds.variables["chosen_idx"][()])) if "chosen_idx" in ds.variables else int(len(lambdas) - 1)

    theta_p = _read_var(ds, "theta_plasma").astype(float)
    zeta_p = _read_var(ds, "zeta_plasma").astype(float)
    theta_c = _read_var(ds, "theta_coil").astype(float)
    zeta_c = _read_var(ds, "zeta_coil").astype(float)

    Btot = _read_var(ds, "Bnormal_total").astype(float)  # (nlambda,nzeta,ntheta)
    K2 = _read_var(ds, "K2").astype(float)  # (nlambda,nzeta,ntheta)
    Phi = _read_var(ds, "current_potential").astype(float)  # (nlambda,nzeta,ntheta)

    r_coil = _read_var(ds, "r_coil").astype(float)  # (nzetal,ntheta,3)
    r_plasma = _read_var(ds, "r_plasma").astype(float)  # (nzetal,ntheta,3)
    if "normal_plasma" in ds.variables:
        normal_plasma = _read_var(ds, "normal_plasma").astype(float)  # (nzetal,ntheta,3)
    else:
        normal_plasma = _normals_from_r_zt3(r_zt3=r_plasma, nfp=int(nfp))

    ds.close()

    # ----------------------------
    # Figures
    # ----------------------------
    if not args.no_figures:
        plt = _setup_matplotlib()

        # Lambda scan
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.semilogx(lambdas, max_K, "-o", label=r"$\\max |K|$")
        ax2 = ax.twinx()
        ax2.semilogx(lambdas, max_B, "-s", color="C1", label=r"$\\max |B_n|$")
        ax.set_xlabel(r"$\\lambda$")
        ax.set_ylabel(r"$\\max |K|$")
        ax2.set_ylabel(r"$\\max |B_n|$")
        ax.axvline(lambdas[ilam], color="k", alpha=0.3, linestyle="--", linewidth=1)
        ax.set_title(f"Lambda scan ({case})")
        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best")
        fig.tight_layout()
        fig.savefig(fig_dir / "lambda_scan.png")
        plt.close(fig)

        # Bnormal map
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        im = ax.pcolormesh(zeta_p, theta_p, Btot[ilam].T, shading="auto")
        fig.colorbar(im, ax=ax, label=r"$B_n$")
        ax.set_xlabel(r"$\\zeta$")
        ax.set_ylabel(r"$\\theta$")
        ax.set_title(f"$B_n$ on plasma (lambda idx {ilam})")
        fig.tight_layout()
        fig.savefig(fig_dir / "Bnormal_plasma.png")
        plt.close(fig)

        # K map
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        Kmag = np.sqrt(K2[ilam])
        im = ax.pcolormesh(zeta_c, theta_c, Kmag.T, shading="auto")
        fig.colorbar(im, ax=ax, label=r"$|K|$")
        ax.set_xlabel(r"$\\zeta$")
        ax.set_ylabel(r"$\\theta$")
        ax.set_title(f"$|K|$ on coil (lambda idx {ilam})")
        fig.tight_layout()
        fig.savefig(fig_dir / "K_coil.png")
        plt.close(fig)

        # Current potential + contours used for coil cutting
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        im = ax.pcolormesh(zeta_c, theta_c, Phi[ilam].T, shading="auto")
        fig.colorbar(im, ax=ax, label=r"$\\Phi$ [A]")
        # Overlay the contour levels used for coil cutting (for pedagogy).
        try:
            data = Phi[ilam].copy()
            if abs(net_pol) > np.finfo(float).eps:
                data = data / net_pol * nfp
            nlevels = int(2 * int(args.coils_per_half_period))
            levels = np.linspace(0.0, 1.0, nlevels, endpoint=False)
            levels = levels + (levels[1] - levels[0]) / 2.0
            ax.contour(zeta_c, theta_c, data.T, levels=levels, colors="k", linewidths=0.6, alpha=0.6)
        except Exception:
            pass
        ax.set_xlabel(r"$\\zeta$")
        ax.set_ylabel(r"$\\theta$")
        ax.set_title(f"Current potential on coil (lambda idx {ilam})")
        fig.tight_layout()
        fig.savefig(fig_dir / "current_potential.png")
        plt.close(fig)

    # ----------------------------
    # Cut coils (filaments) and write MAKECOIL file
    # ----------------------------
    from regcoil_jax.coil_cutting import cut_coils_from_current_potential, write_makecoil_filaments
    from regcoil_jax.fieldlines import build_filament_field, trace_fieldlines, bfield_from_filaments
    from regcoil_jax.vtk_io import (
        flatten_grid_points,
        quad_mesh_connectivity,
        write_vtp_polydata,
        write_vts_structured_grid,
        write_vtu_point_cloud,
    )

    coils = None
    if not args.no_coils:
        coils = cut_coils_from_current_potential(
            current_potential_zt=Phi[ilam],
            theta=theta_c,
            zeta=zeta_c,
            r_coil_zt3_full=r_coil,
            theta_shift=int(args.theta_shift),
            coils_per_half_period=int(args.coils_per_half_period),
            nfp=int(nfp),
            net_poloidal_current_Amperes=float(net_pol),
        )

        coils_path = out_dir / f"coils.{case}"
        write_makecoil_filaments(coils_path, filaments_xyz=coils.filaments_xyz, coil_current=coils.coil_current, nfp=nfp)
        print("[postprocess] wrote:", coils_path)

    # ----------------------------
    # VTK: surfaces
    # ----------------------------
    if args.no_vtk:
        if args.no_figures:
            print("[postprocess] nothing to do (both --no_figures and --no_vtk were set).")
        return

    # Replicate 1-field-period scalar fields across all periods for surface visualization.
    nzeta = zeta_c.size
    nzetal = r_coil.shape[0]
    rep = int(nzetal // nzeta)
    Phi_full = np.tile(Phi[ilam], (rep, 1))
    Kmag_full = np.tile(np.sqrt(K2[ilam]), (rep, 1))

    coil_points = flatten_grid_points(r_coil)
    coil_polys = quad_mesh_connectivity(nzeta=nzetal, ntheta=theta_c.size, periodic_zeta=True, periodic_theta=True)
    write_vtp_polydata(
        vtk_dir / "coil_surface.vtp",
        points=coil_points,
        polys=coil_polys,
        point_data={
            "Phi": Phi_full.reshape(-1),
            "Kmag": Kmag_full.reshape(-1),
        },
    )
    # Also write a structured-grid version (ParaView-friendly for some filters).
    write_vts_structured_grid(
        vtk_dir / "coil_surface.vts",
        points_zt3=r_coil,
        point_data={
            "Phi": Phi_full,
            "Kmag": Kmag_full,
        },
    )

    # Plasma surface VTK
    nzeta_p = zeta_p.size
    nzetal_p = r_plasma.shape[0]
    rep_p = int(nzetal_p // nzeta_p)
    B_full = np.tile(Btot[ilam], (rep_p, 1))
    plasma_points = flatten_grid_points(r_plasma)
    plasma_polys = quad_mesh_connectivity(nzeta=nzetal_p, ntheta=theta_p.size, periodic_zeta=True, periodic_theta=True)
    write_vtp_polydata(
        vtk_dir / "plasma_surface.vtp",
        points=plasma_points,
        polys=plasma_polys,
        point_data={
            "Bnormal": B_full.reshape(-1),
            "absBnormal": np.abs(B_full).reshape(-1),
        },
    )
    write_vts_structured_grid(
        vtk_dir / "plasma_surface.vts",
        points_zt3=r_plasma,
        point_data={
            "Bnormal": B_full,
            "absBnormal": np.abs(B_full),
        },
    )

    # ----------------------------
    # VTK: coil filaments
    # ----------------------------
    if coils is not None:
        pts_all = []
        lines = []
        offset = 0
        for filament in coils.filaments_xyz:
            filament = np.asarray(filament, dtype=float)
            n = filament.shape[0]
            pts_all.append(filament)
            # Close the loop explicitly for visualization.
            idxs = list(range(offset, offset + n)) + [offset]
            lines.append(idxs)
            offset += n

        pts_all = np.concatenate(pts_all, axis=0)
        write_vtp_polydata(vtk_dir / "coils.vtp", points=pts_all, lines=lines)

    # ----------------------------
    # Field lines (coil-only field from filaments)
    # ----------------------------
    field = None
    if coils is not None and (not args.no_fieldlines):
        field = build_filament_field(filaments_xyz=coils.filaments_xyz, coil_current=coils.coil_current)

        # Start points: choose points on the plasma surface (zeta index 0), offset outward along the normal.
        starts = []
        iz = 0
        for k in range(int(args.fieldline_count)):
            it = int(round(k * (theta_p.size / args.fieldline_count))) % theta_p.size
            x = r_plasma[iz, it]
            nvec = normal_plasma[iz, it]
            nhat = nvec / (np.linalg.norm(nvec) + 1e-300)
            starts.append(x + float(args.fieldline_offset) * nhat)
        starts = np.asarray(starts, dtype=float)

        flines = trace_fieldlines(
            field,
            starts=starts,
            ds=float(args.fieldline_ds),
            n_steps=int(args.fieldline_steps),
            stop_radius=10.0,
        )

        pts_all = []
        lines = []
        offset = 0
        for line in flines:
            n = line.shape[0]
            pts_all.append(line)
            lines.append(list(range(offset, offset + n)))
            offset += n
        pts_all = np.concatenate(pts_all, axis=0) if pts_all else np.zeros((0, 3))
        write_vtp_polydata(vtk_dir / "fieldlines.vtp", points=pts_all, lines=lines)

    # ----------------------------
    # Optional VTU point cloud (|B| in a 3D box), coil-filament field only
    # ----------------------------
    if args.write_vtu_point_cloud:
        if field is None:
            raise SystemExit("--write_vtu_point_cloud requires coil cutting + field evaluation (do not set --no_coils).")
        n = int(args.vtu_n)
        margin = float(args.vtu_margin)
        pts = r_plasma.reshape(-1, 3)
        lo = pts.min(axis=0) - margin
        hi = pts.max(axis=0) + margin
        xs = np.linspace(lo[0], hi[0], n)
        ys = np.linspace(lo[1], hi[1], n)
        zs = np.linspace(lo[2], hi[2], n)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        grid = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        B = bfield_from_filaments(field, grid)
        Bmag = np.linalg.norm(B, axis=1)
        write_vtu_point_cloud(
            vtk_dir / "B_point_cloud.vtu",
            points=grid,
            point_data={"B": B, "Bmag": Bmag},
        )

    print("[postprocess] wrote VTK:", vtk_dir)
    if not args.no_figures:
        print("[postprocess] wrote figures:", fig_dir)


if __name__ == "__main__":
    main()
