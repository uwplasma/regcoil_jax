#!/usr/bin/env python3
"""Optimize per-coil currents *after* cutting coils from a REGCOIL_JAX solve.

This example demonstrates an autodiff-based workflow that replaces Fortran adjoint sensitivities:
we differentiate through a Biot–Savart filament model to optimize a set of *discrete* coil currents.

Workflow:
1) run regcoil_jax (programmatically) to obtain a current potential on a winding surface
2) cut filamentary coils as contours of the current potential (REGCOIL-style)
3) optimize an independent current for each filament so the filament model matches the target normal field
   produced by the continuous surface current (B_sv = Btotal - (Bplasma + Bnet))
4) write figures + VTK files (coils, field lines, Poincaré section)
"""

from __future__ import annotations

import argparse
import time
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


def main() -> None:
    if netCDF4 is None:  # pragma: no cover
        raise SystemExit("netCDF4 is required for this example")

    here = Path(__file__).resolve().parent
    default_input = here / "regcoil_in.torus_cli_intermediate"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(default_input), help="Path to regcoil_in.*")
    parser.add_argument("--platform", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--coils_per_half_period", type=int, default=6)
    parser.add_argument("--theta_shift", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--seg_batch", type=int, default=1024)
    parser.add_argument("--no_figures", action="store_true")
    parser.add_argument("--no_vtk", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.name.startswith("regcoil_in."):
        raise SystemExit("Input must be named regcoil_in.*")

    case = input_path.name[len("regcoil_in.") :]
    out_dir = here / f"outputs_optimize_cut_coil_currents_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy the input into the run dir (keeps outputs together).
    input_copy = out_dir / input_path.name
    input_copy.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Run regcoil_jax programmatically (writes regcoil_out.* next to the input file).
    from regcoil_jax.run import run_regcoil

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
    Bsv = Btot - (Bplasma + Bnet)  # target from surface current

    Phi = np.asarray(ds.variables["current_potential"][ilam], dtype=float)  # (nzeta_coil, ntheta_coil)
    r_coil = np.asarray(ds.variables["r_coil"][:], dtype=float)  # (nzetal, ntheta, 3)
    r_plasma = np.asarray(ds.variables["r_plasma"][:], dtype=float)  # (nzetal, ntheta, 3)
    ds.close()

    # Extract first field period geometry (so shapes match Bsv which is per-period).
    nzeta = int(zeta_p.size)
    r_plasma_1 = r_plasma[:nzeta]  # (nzeta, ntheta, 3)

    from regcoil_jax.surface_utils import unit_normals_from_r_zt3

    nunit_1 = unit_normals_from_r_zt3(r_zt3=r_plasma_1)

    points = r_plasma_1.reshape(-1, 3)
    normals_unit = nunit_1.reshape(-1, 3)
    target_bnormal = Bsv.reshape(-1)

    # Cut coils from Phi on the winding surface.
    from regcoil_jax.coil_cutting import cut_coils_from_current_potential, write_makecoil_filaments
    from regcoil_jax.biot_savart_jax import segments_from_filaments
    from regcoil_jax.coil_current_optimization import optimize_coil_currents_to_match_bnormal

    coils = cut_coils_from_current_potential(
        current_potential_zt=Phi,
        theta=theta_c,
        zeta=zeta_c,
        r_coil_zt3_full=r_coil,
        theta_shift=int(args.theta_shift),
        coils_per_half_period=int(args.coils_per_half_period),
        nfp=int(nfp),
        net_poloidal_current_Amperes=float(net_pol),
    )

    segs = segments_from_filaments(filaments_xyz=coils.filaments_xyz)

    # Initial currents (REGCOIL-style: uniform per filament).
    I0 = coils.coil_currents
    net_target = float(np.sum(I0))

    opt = optimize_coil_currents_to_match_bnormal(
        segs,
        points=points,
        normals_unit=normals_unit,
        target_bnormal=target_bnormal,
        coil_currents0=I0,
        steps=int(args.steps),
        lr=float(args.lr),
        net_current_target=net_target,
        seg_batch=int(args.seg_batch),
    )
    I1 = opt.coil_currents

    write_makecoil_filaments(out_dir / f"coils_initial.{case}", filaments_xyz=coils.filaments_xyz, coil_currents=I0, nfp=nfp)
    write_makecoil_filaments(out_dir / f"coils_optimized.{case}", filaments_xyz=coils.filaments_xyz, coil_currents=I1, nfp=nfp)

    # ----------------------------
    # Figures
    # ----------------------------
    if not args.no_figures:
        plt = _setup_matplotlib()

        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(np.asarray(opt.loss_history), "-k")
        ax.set_xlabel("Adam step")
        ax.set_ylabel("loss (MSE + penalties)")
        ax.set_title("Per-coil current optimization")
        fig.tight_layout()
        fig.savefig(out_dir / "loss_history.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(I0, "o-", label="initial")
        ax.plot(I1, "o-", label="optimized")
        ax.set_xlabel("coil index")
        ax.set_ylabel("current [A]")
        ax.legend(loc="best")
        ax.set_title("Per-coil currents")
        fig.tight_layout()
        fig.savefig(out_dir / "coil_currents.png")
        plt.close(fig)

    # ----------------------------
    # VTK: coils + field lines + Poincaré points
    # ----------------------------
    if not args.no_vtk:
        from regcoil_jax.vtk_io import write_vtp_polydata
        from regcoil_jax.fieldlines import build_filament_field_multi_current, trace_fieldlines, poincare_points

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
                idxs = list(range(offset, offset + n)) + [offset]
                lines.append(idxs)
                offset += n
            pts_all = np.concatenate(pts_all, axis=0)
            write_vtp_polydata(
                vtk_dir / name,
                points=pts_all,
                lines=lines,
                cell_data={"coil_current": np.asarray(coil_currents, dtype=float)},
            )

        write_coils_vtp("coils_initial.vtp", I0)
        write_coils_vtp("coils_optimized.vtp", I1)

        field = build_filament_field_multi_current(filaments_xyz=coils.filaments_xyz, coil_currents=I1)

        # Start points: a small ring around zeta index 0.
        starts = []
        iz = 0
        for k in range(8):
            it = int(round(k * (theta_p.size / 8))) % theta_p.size
            x = r_plasma_1[iz, it]
            nh = normals_unit.reshape(nzeta, theta_p.size, 3)[iz, it]
            starts.append(x + 0.15 * nh)
        starts = np.asarray(starts, dtype=float)

        flines = trace_fieldlines(field, starts=starts, ds=0.03, n_steps=800, stop_radius=10.0)
        pts_all = []
        lines = []
        offset = 0
        for line in flines:
            pts_all.append(line)
            lines.append(list(range(offset, offset + line.shape[0])))
            offset += line.shape[0]
        pts_all = np.concatenate(pts_all, axis=0)
        write_vtp_polydata(vtk_dir / "fieldlines_optimized.vtp", points=pts_all, lines=lines)

        psets = poincare_points(flines, nfp=nfp, phi0=0.0, max_points_per_line=400)
        pts = []
        line_id = []
        for j, ps in enumerate(psets):
            if ps.size == 0:
                continue
            pts.append(ps)
            line_id.append(np.full((ps.shape[0],), j, dtype=float))
        if pts:
            pts_all = np.concatenate(pts, axis=0)
            line_id_all = np.concatenate(line_id, axis=0)
            write_vtp_polydata(
                vtk_dir / "poincare_points_optimized.vtp",
                points=pts_all,
                verts=np.arange(pts_all.shape[0], dtype=np.int64),
                point_data={"line_id": line_id_all},
            )

    print("[example] wrote outputs to:", out_dir)


if __name__ == "__main__":
    main()

