#!/usr/bin/env python3
"""Differentiable 3D isosurface extraction demo (soft marching cubes + hard mesh for VTK).

This example complements the 2D contouring demos by showing a 3D pipeline:

1) Build a scalar field φ(x,y,z) on a 3D Cartesian grid.
2) Extract a **differentiable** weighted point cloud of isosurface candidates using
   :func:`regcoil_jax.diff_isosurface.soft_marching_cubes_candidates`.
3) Extract a **non-differentiable** triangle mesh for visualization using a lightweight
   marching-tetrahedra implementation (:func:`regcoil_jax.isosurface_numpy.marching_tetrahedra_mesh`).
4) Write ParaView-ready VTK `.vtp` files and a couple of small figures.

The differentiable candidate extractor is intended for optimization objectives where discrete
topology changes are problematic; the hard mesh extractor is intended for visualization only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


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
    ap.add_argument("--platform", type=str, default="cpu")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--n", type=int, default=32, help="Grid resolution along each axis")
    ap.add_argument("--radius", type=float, default=0.75)
    ap.add_argument("--alpha", type=float, default=160.0, help="Soft straddle sharpness")
    args = ap.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("JAX_PLATFORM_NAME", str(args.platform))

    if args.fast:
        args.n = 18

    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = examples_dir / f"outputs_diff_isosurface_3d_{stamp}"
    else:
        out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vtk").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    import jax
    import jax.numpy as jnp

    from regcoil_jax.diff_isosurface import soft_marching_cubes_candidates
    from regcoil_jax.isosurface_numpy import marching_tetrahedra_mesh
    from regcoil_jax.vtk_io import write_vtp_polydata

    n = int(args.n)
    grid = jnp.linspace(-1.0, 1.0, n, dtype=jnp.float64)
    X, Y, Z = jnp.meshgrid(grid, grid, grid, indexing="ij")
    xyz = jnp.stack([X, Y, Z], axis=-1)

    r0 = jnp.asarray(float(args.radius), dtype=jnp.float64)
    phi = X * X + Y * Y + Z * Z - r0 * r0

    pts, w = soft_marching_cubes_candidates(xyz_ijk3=xyz, phi_ijk=phi, level=0.0, alpha=float(args.alpha))
    radii = jnp.linalg.norm(pts, axis=1)
    wsum = jnp.sum(w) + 1e-300
    r_mean = jnp.sum(w * radii) / wsum

    # A tiny differentiable objective: match the (soft) mean isosurface radius to a target.
    r_target = jnp.asarray(0.8, dtype=jnp.float64)

    def loss(r: jnp.ndarray) -> jnp.ndarray:
        phi_r = X * X + Y * Y + Z * Z - r * r
        pts_r, w_r = soft_marching_cubes_candidates(xyz_ijk3=xyz, phi_ijk=phi_r, level=0.0, alpha=float(args.alpha))
        rr = jnp.linalg.norm(pts_r, axis=1)
        wm = jnp.sum(w_r) + 1e-300
        rbar = jnp.sum(w_r * rr) / wm
        return (rbar - r_target) ** 2

    g = jax.grad(loss)(r0)

    pts_np = np.asarray(pts, dtype=float)
    w_np = np.asarray(w, dtype=float)
    radii_np = np.asarray(radii, dtype=float)

    # Write candidate point cloud.
    write_vtp_polydata(
        out_dir / "vtk" / "sphere_candidates_soft_marching_cubes.vtp",
        points=pts_np,
        verts=np.arange(pts_np.shape[0], dtype=np.int64),
        point_data={"weight": w_np, "radius": radii_np},
    )

    # Write a visualization mesh (hard marching tetrahedra).
    xyz_np = np.asarray(xyz, dtype=float)
    phi_np = np.asarray(phi, dtype=float)
    mesh_pts, mesh_tri = marching_tetrahedra_mesh(xyz_ijk3=xyz_np, phi_ijk=phi_np, level=0.0)
    write_vtp_polydata(out_dir / "vtk" / "sphere_mesh_marching_tetrahedra.vtp", points=mesh_pts, polys=mesh_tri)

    # Figures
    plt = _setup_matplotlib()
    fig = plt.figure(figsize=(8.0, 3.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Show a 2D slice (z~0) of candidate points projected to x-y, colored by weight.
    zmask = np.abs(pts_np[:, 2]) < 0.05
    sel = np.where(zmask)[0]
    if sel.size > 5000:
        sel = sel[:: (sel.size // 5000 + 1)]
    sc = ax1.scatter(pts_np[sel, 0], pts_np[sel, 1], c=w_np[sel], s=4, cmap="viridis")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Soft MC candidates (z≈0 slice)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label="weight")

    ax2.hist(radii_np, bins=60, weights=w_np, density=True, alpha=0.8)
    ax2.axvline(float(args.radius), color="k", lw=1.5, label="iso radius")
    ax2.set_title("Weighted radius histogram")
    ax2.set_xlabel("radius")
    ax2.set_ylabel("density")
    ax2.legend(loc="best")

    fig.suptitle(f"3D differentiable isosurface demo: mean radius≈{float(r_mean):.3f}, grad={float(g):+.3e}")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "soft_marching_cubes_demo.png")
    plt.close(fig)

    summary = dict(
        n=n,
        radius=float(args.radius),
        alpha=float(args.alpha),
        soft_mean_radius=float(r_mean),
        loss_target=float(r_target),
        dloss_dr=float(g),
        n_candidates=int(pts_np.shape[0]),
        n_mesh_points=int(mesh_pts.shape[0]),
        n_mesh_tris=int(mesh_tri.shape[0]),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[isosurface3d] wrote outputs to: {out_dir}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()

