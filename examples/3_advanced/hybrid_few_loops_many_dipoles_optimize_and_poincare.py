#!/usr/bin/env python3
"""Hybrid coil design demo: a few simple loops + many dipoles, optimized for B·n ≈ 0.

This example goes beyond the original REGCOIL feature set by adding a *hybrid source model*:

  B_total = B_filaments + B_dipoles

where:
- B_filaments is a standard Biot–Savart field from a small number of closed filament loops.
- B_dipoles is the field from many point dipoles (a differentiable proxy for windowpane coils
  or permanent magnets mounted around the device).

We optimize:
- the current in each filament loop, and
- the dipole moment vector of each dipole,

to minimize mean-square B·n on a target VMEC plasma surface.

Outputs:
  - figures (loss history, B·n histograms, Poincaré plots overlaid with surface slice)
  - ParaView VTK: coils, dipoles (as points with moment vectors), field lines, Poincaré points, plasma surface
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


def _circle_loop(*, R: float, z: float, n: int = 200) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    x = float(R) * np.cos(t)
    y = float(R) * np.sin(t)
    zz = np.full_like(t, float(z))
    return np.stack([x, y, zz], axis=1)


def _cross_section_curve_rz(*, r_plasma_3tz: np.ndarray, phi0: float) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    r = np.asarray(r_plasma_3tz, dtype=float)
    phi = np.arctan2(r[1], r[0])  # (T,Z) in the regcoil_jax internal format
    # We want a zeta index with mean phi close to phi0.
    phi_mean = np.unwrap(np.mean(phi, axis=0))
    idx = int(np.argmin(np.abs(((phi_mean - float(phi0) + np.pi) % (2.0 * np.pi)) - np.pi)))
    x = r[0, :, idx]
    y = r[1, :, idx]
    R = np.sqrt(x * x + y * y)
    Z = r[2, :, idx]
    axis = (float(np.mean(R)), float(np.mean(Z)))
    return R, Z, axis


def main() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    examples_dir = here.parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", type=str, default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--wout", type=str, default=str(examples_dir / "wout_d23p4_tm.nc"))
    ap.add_argument(
        "--bnorm",
        type=str,
        default=str(examples_dir / "bnorm.d23p4_tm"),
        help="BNORM file (Fourier modes for Bnormal_from_plasma_current on the plasma surface).",
    )
    ap.add_argument("--out_dir", type=str, default=None)

    ap.add_argument("--ntheta", type=int, default=48, help="plasma surface grid (theta)")
    ap.add_argument("--nzeta", type=int, default=48, help="plasma surface grid (zeta)")
    ap.add_argument("--dipole_offset", type=float, default=0.45, help="offset distance along normals for dipole positions [m]")
    ap.add_argument("--dipole_stride", type=int, default=24, help="downsample stride for dipole placement")

    ap.add_argument("--n_loops", type=int, default=4)
    ap.add_argument("--loop_radius_offset", type=float, default=1.2)
    ap.add_argument("--loop_z_extent", type=float, default=1.2)

    ap.add_argument("--steps", type=int, default=350)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--l2_current", type=float, default=1e-6)
    ap.add_argument("--l2_moment", type=float, default=1e-8)

    ap.add_argument("--phi0", type=float, default=0.0, help="Poincaré plane toroidal angle [rad]")
    ap.add_argument("--fieldline_ds", type=float, default=0.03)
    ap.add_argument("--fieldline_steps", type=int, default=1600)
    ap.add_argument("--poincare_max_points", type=int, default=800)
    ap.add_argument("--n_starts", type=int, default=14)
    args = ap.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("JAX_PLATFORM_NAME", args.platform)

    wout = Path(args.wout).resolve()
    if not wout.exists():
        raise SystemExit(f"Missing wout file: {wout}")

    bnorm = Path(args.bnorm).resolve()
    if not bnorm.exists():
        raise SystemExit(f"Missing bnorm file: {bnorm}")

    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = examples_dir / f"outputs_hybrid_dipoles_{stamp}"
    else:
        out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy wout locally so the output folder is self-contained.
    wout_dst = out_dir / wout.name
    if wout_dst != wout:
        wout_dst.write_bytes(wout.read_bytes())

    bnorm_dst = out_dir / bnorm.name
    if bnorm_dst != bnorm:
        bnorm_dst.write_bytes(bnorm.read_bytes())

    # -----
    # Build the target plasma surface points/normals (JAX surface builder, but we use numpy arrays).
    # -----
    import jax.numpy as jnp
    from regcoil_jax.io_bnorm import read_bnorm_modes
    from regcoil_jax.io_vmec import compute_curpol_and_G, read_wout_boundary
    from regcoil_jax.geometry_fourier import FourierSurface
    from regcoil_jax.surfaces import plasma_surface_from_inputs
    from regcoil_jax.dipoles import dipole_array_from_surface_offset
    from regcoil_jax.dipole_optimization import optimize_filaments_and_dipoles_to_match_bnormal
    from regcoil_jax.biot_savart_jax import segments_from_filaments, bnormal_from_segments
    from regcoil_jax.hybrid_fields import HybridFieldNumpy, bfield_from_hybrid_numpy
    from regcoil_jax.vtk_io import write_vts_structured_grid, write_vtp_polydata
    from regcoil_jax.fieldlines import poincare_points_for_line

    bound = read_wout_boundary(str(wout_dst), radial_mode="full")
    vmec = FourierSurface(
        nfp=int(bound.nfp),
        lasym=bool(bound.lasym),
        xm=jnp.asarray(bound.xm, dtype=jnp.int32),
        xn=jnp.asarray(bound.xn, dtype=jnp.int32),
        rmnc=jnp.asarray(bound.rmnc, dtype=jnp.float64),
        zmns=jnp.asarray(bound.zmns, dtype=jnp.float64),
        rmns=jnp.asarray(bound.rmns, dtype=jnp.float64),
        zmnc=jnp.asarray(bound.zmnc, dtype=jnp.float64),
    )

    inputs = dict(
        geometry_option_plasma=2,
        ntheta_plasma=int(args.ntheta),
        nzeta_plasma=int(args.nzeta),
        wout_filename=str(wout_dst),
        nfp_imposed=int(vmec.nfp),
    )
    plasma = plasma_surface_from_inputs(inputs, vmec)
    # Flatten points and normals for optimization.
    pts = np.moveaxis(np.asarray(plasma["r"]), 0, -1).reshape((-1, 3))
    nhat = np.moveaxis(np.asarray(plasma["nunit"]), 0, -1).reshape((-1, 3))

    # Target Bn: cancel Bnormal_from_plasma_current (from BNORM modes).
    # This matches the core REGCOIL idea: coils (or hybrid sources) reproduce a desired boundary by canceling
    # the normal field from plasma currents on the plasma surface.
    curpol, _, nfp, _ = compute_curpol_and_G(str(wout_dst))
    if curpol is None:
        curpol = 1.0
        print("[warn] could not infer curpol from wout; using curpol=1.0 (BNORM scaling may be off).")
    m, n, bf = read_bnorm_modes(str(bnorm))
    th = np.asarray(plasma["theta"], dtype=float)
    ze = np.asarray(plasma["zeta"], dtype=float)
    ang = m[:, None, None] * th[None, :, None] + (n[:, None, None] * int(nfp)) * ze[None, None, :]
    Bplasma_TZ = np.sum(bf[:, None, None] * np.sin(ang), axis=0) * float(curpol)
    target = -Bplasma_TZ.reshape((-1,))

    # -----
    # Place dipoles on an offset surface and build a few simple circular loops.
    # -----
    dip_pos, dip_m0 = dipole_array_from_surface_offset(
        surface_points=pts,
        surface_normals_unit=nhat,
        offset=float(args.dipole_offset),
        stride=int(args.dipole_stride),
    )

    # Use a few "large" axisymmetric loops as a baseline source set.
    R_mean = float(np.mean(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)))
    Z_mean = float(np.mean(pts[:, 2]))
    R_loop = R_mean + float(args.loop_radius_offset)
    zs = np.linspace(-float(args.loop_z_extent), float(args.loop_z_extent), int(args.n_loops)) + Z_mean
    loops = [_circle_loop(R=R_loop, z=float(z), n=240) for z in zs]

    segs = segments_from_filaments(filaments_xyz=loops)
    I0 = np.zeros((segs.n_filaments,), dtype=float)

    # -----
    # Optimize currents + dipoles to match target Bn (cancel Bplasma).
    # -----
    print(f"[setup] points={pts.shape[0]} loops={segs.n_filaments} dipoles={dip_pos.shape[0]}")
    print(f"[target] BNORM modes={m.size} nfp={int(nfp)} curpol={float(curpol):.6g}  rms(Bplasma)={np.sqrt(np.mean(Bplasma_TZ*Bplasma_TZ)):.3e}")
    res = optimize_filaments_and_dipoles_to_match_bnormal(
        segs,
        dipole_positions=dip_pos,
        points=pts,
        normals_unit=nhat,
        target_bnormal=target,
        filament_currents0=I0,
        dipole_moments0=dip_m0,
        steps=int(args.steps),
        lr=float(args.lr),
        l2_current=float(args.l2_current),
        l2_moment=float(args.l2_moment),
        seg_batch=4096,
        dipole_batch=4096,
    )

    # Diagnostics: Bn before/after.
    bn0 = np.asarray(
        bnormal_from_segments(segs, points=pts, normals_unit=nhat, filament_currents=I0, seg_batch=4096),
        dtype=float,
    )
    bn1 = np.asarray(
        bnormal_from_segments(segs, points=pts, normals_unit=nhat, filament_currents=res.filament_currents, seg_batch=4096),
        dtype=float,
    )
    # Include dipoles for after:
    # (compute on numpy side for printing; optimization already used hybrid field)
    from regcoil_jax.dipoles import dipole_bnormal
    import jax

    bn_d = np.asarray(
        dipole_bnormal(points=pts, normals_unit=nhat, positions=dip_pos, moments=res.dipole_moments, batch=4096),
        dtype=float,
    )
    bn_h = bn1 + bn_d

    err0 = bn0 - target
    err1 = bn1 - target
    errh = bn_h - target
    print(f"[residual] loops-only RMS {np.sqrt(np.mean(err0*err0)):.3e}")
    print(f"[residual] loops-only (optimized) RMS {np.sqrt(np.mean(err1*err1)):.3e}")
    print(f"[residual] hybrid (loops+dipoles) RMS {np.sqrt(np.mean(errh*errh)):.3e}  max|Bn|={np.max(np.abs(errh)):.3e}")

    # -----
    # Visualization: Poincaré plots and ParaView outputs.
    # -----
    # Build a numpy-side hybrid field for tracing.
    from regcoil_jax.fieldlines import build_filament_field_multi_current

    fil_field = build_filament_field_multi_current(filaments_xyz=loops, coil_currents=res.filament_currents)
    hybrid = HybridFieldNumpy(filaments=fil_field, dipole_positions=dip_pos, dipole_moments=res.dipole_moments)

    # Start points between the cross-section axis estimate and the boundary outboard midplane.
    r3tz = np.asarray(plasma["r"], dtype=float)  # (3,T,Z)
    Rb, Zb, axis = _cross_section_curve_rz(r_plasma_3tz=r3tz, phi0=float(args.phi0))
    axis_R, axis_Z = axis
    j = int(np.argmax(Rb))
    R_edge = float(Rb[j])
    Z_edge = float(Zb[j])
    t = np.linspace(0.1, 0.98, int(args.n_starts))
    starts_rz = np.stack([axis_R + t * (R_edge - axis_R), axis_Z + t * (Z_edge - axis_Z)], axis=1)
    c = np.cos(float(args.phi0))
    s = np.sin(float(args.phi0))
    starts = np.stack([starts_rz[:, 0] * c, starts_rz[:, 0] * s, starts_rz[:, 1]], axis=1)

    def trace_line(x0: np.ndarray) -> np.ndarray:
        x = np.asarray(x0, dtype=float).reshape(3)
        pts_line = [x.copy()]

        def f(x_):
            B = bfield_from_hybrid_numpy(hybrid, x_)
            n = np.linalg.norm(B)
            return B / (n if n != 0.0 else 1.0)

        ds = float(args.fieldline_ds)
        for _ in range(int(args.fieldline_steps)):
            k1 = f(x)
            k2 = f(x + 0.5 * ds * k1)
            k3 = f(x + 0.5 * ds * k2)
            k4 = f(x + ds * k3)
            x = x + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            pts_line.append(x.copy())
        return np.asarray(pts_line, dtype=float)

    lines = [trace_line(x0) for x0 in starts]
    pcs = [poincare_points_for_line(line, nfp=int(vmec.nfp), phi0=float(args.phi0), max_points=int(args.poincare_max_points)) for line in lines]

    vtk_dir = out_dir / "vtk"
    vtk_dir.mkdir(exist_ok=True)
    write_vts_structured_grid(vtk_dir / "plasma_surface.vts", points_zt3=np.moveaxis(r3tz, 0, -1).transpose(1, 0, 2))

    # Coils as polylines
    coil_pts = np.concatenate(loops, axis=0)
    coil_lines = []
    off = 0
    for pts_loop in loops:
        n = int(pts_loop.shape[0])
        coil_lines.append(list(range(off, off + n)) + [off])
        off += n
    write_vtp_polydata(vtk_dir / "loops.vtp", points=coil_pts, lines=coil_lines, point_data={"id": np.arange(coil_pts.shape[0], dtype=float)})

    # Dipoles as point cloud with vector data (use ParaView Glyph filter to visualize).
    m = np.asarray(res.dipole_moments, dtype=float)
    write_vtp_polydata(
        vtk_dir / "dipoles.vtp",
        points=dip_pos,
        verts=np.arange(dip_pos.shape[0], dtype=np.int64),
        point_data={"moment": m, "moment_norm": np.linalg.norm(m, axis=1)},
    )

    # Field lines and Poincaré points
    fl_pts = np.concatenate(lines, axis=0)
    fl_conn = []
    off = 0
    for ln in lines:
        n = int(ln.shape[0])
        fl_conn.append(list(range(off, off + n)))
        off += n
    write_vtp_polydata(vtk_dir / "fieldlines.vtp", points=fl_pts, lines=fl_conn)
    for i, p in enumerate(pcs):
        if p.size == 0:
            continue
        write_vtp_polydata(vtk_dir / f"poincare_{i:02d}.vtp", points=p, verts=np.arange(p.shape[0], dtype=np.int64))

    # -----
    # Figures
    # -----
    plt = _setup_matplotlib()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(res.loss_history)
    ax.set_xlabel("Adam step")
    ax.set_ylabel("loss = mean(B·n)^2 + regularization")
    ax.set_title("Hybrid optimization loss")
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_history.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.hist(np.abs(errh), bins=80, alpha=0.8)
    ax.set_xlabel(r"$|B_{\mathrm{plasma}}\cdot n + B_{\mathrm{hybrid}}\cdot n|$ [T]")
    ax.set_ylabel("count")
    ax.set_title("Residual normal field after optimization")
    fig.tight_layout()
    fig.savefig(fig_dir / "bn_hist.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(Rb, Zb, "k-", linewidth=1.3, label="target surface (slice)")
    for p in pcs:
        if p.size == 0:
            continue
        R = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        Z = p[:, 2]
        ax.scatter(R, Z, s=3, alpha=0.55)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(f"Poincaré at $\\phi_0$={float(args.phi0):.2f} (hybrid)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "poincare.png")
    plt.close(fig)

    print(f"[done] outputs: {out_dir}")


if __name__ == "__main__":
    main()
