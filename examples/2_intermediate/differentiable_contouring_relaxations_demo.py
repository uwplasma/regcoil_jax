#!/usr/bin/env python3
"""Differentiable contouring relaxations demo (toy, no VMEC required).

This script demonstrates three related utilities:

1) `soft_contour_theta_of_zeta`: a smooth “circular-mean softmax” approximation of θ(ζ) for Φ(ζ,θ)=Φ0.
2) `implicit_coil_polyline_xyz`: a root-finding (Newton) θ(ζ) solver with implicit gradients (IFT).
3) `soft_marching_squares_candidates`: a marching-squares-inspired relaxation that returns a weighted point cloud
   of candidate edge intersections (not a connected polyline).

Outputs (written under `--out_dir`):
  - `figures/theta_of_zeta.png`: θ(ζ) comparison (soft vs implicit)
  - `figures/torus_3d_polylines.png`: 3D plot of the extracted polylines on a torus surface
  - `vtk/torus_surface.vts`: surface structured grid
  - `vtk/implicit_polyline.vtp`: extracted implicit polyline on the surface
  - `vtk/marching_candidates.vtp`: weighted point cloud of candidate intersections (`weight`)
"""

from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--ntheta", type=int, default=80)
    ap.add_argument("--nzeta", type=int, default=80)
    ap.add_argument("--level", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=150.0, help="marching-candidate sharpness")
    ap.add_argument("--beta", type=float, default=2e4, help="soft-contour sharpness")
    args = ap.parse_args()

    import jax.numpy as jnp

    from regcoil_jax.diff_coil_cutting import soft_contour_theta_of_zeta, implicit_coil_polyline_xyz
    from regcoil_jax.diff_isosurface import soft_marching_squares_candidates
    from regcoil_jax.vtk_io import write_vtp_polydata, write_vts_structured_grid

    nfp = int(args.nfp)
    ntheta = int(args.ntheta)
    nzeta = int(args.nzeta)

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, nzeta, endpoint=False)

    # A toy "winding surface": analytic torus embedding on the same (zeta,theta) grid.
    R0 = 6.0
    a = 1.2
    R = R0 + a * jnp.cos(theta)[None, :]
    Z = a * jnp.sin(theta)[None, :]
    phi = zeta[:, None]
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    r_zt3 = jnp.stack([x, y, jnp.broadcast_to(Z, x.shape)], axis=2)

    # A toy scalar field with nontrivial ζ-dependence; this mimics a current potential slice.
    # It has multiple contour branches in general; both θ(ζ) methods pick one branch.
    Phi_zt = jnp.sin(theta)[None, :] + 0.30 * jnp.cos(2.0 * zeta)[:, None] + 0.10 * jnp.sin(theta)[None, :] * jnp.cos(zeta)[:, None]

    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).resolve().parent / f"outputs_diff_contours_{stamp}"
    else:
        out_dir = Path(args.out_dir).resolve()
    fig_dir = out_dir / "figures"
    vtk_dir = out_dir / "vtk"
    fig_dir.mkdir(parents=True, exist_ok=True)
    vtk_dir.mkdir(parents=True, exist_ok=True)

    level = float(args.level)

    theta_soft = soft_contour_theta_of_zeta(phi_zt=Phi_zt, theta=theta, level=level, beta=float(args.beta))
    pts_implicit = implicit_coil_polyline_xyz(phi_zt=Phi_zt, theta=theta, zeta=zeta, r_coil_zt3=r_zt3, level=level, nfp=nfp)

    cand_pts, cand_w = soft_marching_squares_candidates(r_zt3=r_zt3, phi_zt=Phi_zt, level=level, alpha=float(args.alpha))

    # VTK
    write_vts_structured_grid(vtk_dir / "torus_surface.vts", points_zt3=np.asarray(r_zt3, dtype=float), point_data={"Phi": np.asarray(Phi_zt, dtype=float)})

    # Polyline connectivity for the implicit curve (close it by repeating the first point).
    pi = np.asarray(pts_implicit, dtype=float)
    poly_pts = np.vstack([pi, pi[:1]])
    write_vtp_polydata(vtk_dir / "implicit_polyline.vtp", points=poly_pts, lines=[list(range(poly_pts.shape[0]))])

    w_np = np.asarray(cand_w, dtype=float)
    cp = np.asarray(cand_pts, dtype=float)
    write_vtp_polydata(
        vtk_dir / "marching_candidates.vtp",
        points=cp,
        verts=np.arange(cp.shape[0], dtype=np.int64),
        point_data={"weight": w_np},
    )

    # Figures
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(np.asarray(zeta), np.asarray(theta_soft), "-", label="soft θ(ζ)")
    ax.set_xlabel(r"$\zeta$ [rad]")
    ax.set_ylabel(r"$\theta$ [rad]")
    ax.set_title(r"Single-branch contour approximation for $\Phi(\zeta,\theta)=\Phi_0$")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "theta_of_zeta.png")
    plt.close(fig)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7.8, 6.2), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    surf = np.asarray(r_zt3, dtype=float).reshape(-1, 3)
    stride = max(1, surf.shape[0] // 7000)
    s = surf[::stride]
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=0.25, alpha=0.06, c="k", linewidths=0)
    ax.plot(poly_pts[:, 0], poly_pts[:, 1], poly_pts[:, 2], color="tab:red", linewidth=1.6, alpha=0.9, label="implicit polyline")

    # Candidate point cloud (thresholded for visibility).
    mask = w_np > 0.6
    if np.any(mask):
        ax.scatter(cp[mask, 0], cp[mask, 1], cp[mask, 2], s=1.0, alpha=0.25, c="tab:blue", label="candidates (w>0.6)")
    ax.set_title("Differentiable contour relaxations (toy)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="best")
    fig.savefig(fig_dir / "torus_3d_polylines.png", dpi=220)
    plt.close(fig)

    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()

