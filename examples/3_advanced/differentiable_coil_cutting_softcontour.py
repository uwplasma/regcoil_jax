#!/usr/bin/env python3
"""Differentiable coil cutting (approx): soft contour extraction + gradient-based tuning.

Classic REGCOIL coil cutting extracts filaments as exact contours of the winding-surface
current potential Φ(θ,ζ) using non-differentiable geometry operations (marching squares).

This demo shows a *differentiable relaxation*:
  - For each ζ, estimate a single-valued contour θ(ζ) using a softmin over θ:
        θ(ζ) ≈ argmin_θ (Φ(ζ,θ) - level)^2   (softened with a temperature β)
  - Map (ζ, θ(ζ)) to XYZ via differentiable bilinear interpolation on the winding surface.
  - Differentiate through the resulting coil polyline and a Biot–Savart evaluation.

We then optimize two scalar parameters:
  - the contour level `level`
  - the coil current `I`

to reduce the mean-square error between the coil's B·n and the target surface-current
normal field B_sv produced by the REGCOIL solution (for a chosen lambda).

This is a *pedagogic* example; it is not a replacement for robust contour cutting.
See `docs/differentiable_coil_cutting.rst` for limitations and alternatives.
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
    if netCDF4 is None:  # pragma: no cover
        raise SystemExit("netCDF4 is required for this example (pip install regcoil_jax[viz]).")

    here = Path(__file__).resolve().parent
    default_input = here / "regcoil_in.lambda_search_1"

    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", type=str, default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--input", type=str, default=str(default_input), help="Path to regcoil_in.*")
    ap.add_argument("--lambda_index", type=int, default=None, help="Which lambda index to use (default: best RMS(Bn_total))")
    ap.add_argument("--beta", type=float, default=2e4, help="soft contour sharpness")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--length_weight", type=float, default=1e-4)
    ap.add_argument("--n_eval", type=int, default=400, help="number of plasma points used in the objective")
    args = ap.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("JAX_PLATFORM_NAME", args.platform)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Missing input: {input_path}")
    if not input_path.name.startswith("regcoil_in."):
        raise SystemExit("Input must be named regcoil_in.*")

    out_dir = here / f"outputs_diff_cut_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_copy = out_dir / input_path.name
    input_copy.write_text(input_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

    from regcoil_jax.run import run_regcoil

    res = run_regcoil(str(input_copy), verbose=True)
    out_nc = Path(res.output_nc)

    ds = netCDF4.Dataset(str(out_nc), "r")
    try:
        nfp = int(ds.variables["nfp"][()])
        lambdas = np.asarray(ds.variables["lambda"][:], dtype=float)
        Btot = np.asarray(ds.variables["Bnormal_total"][:], dtype=float)
        Bplasma = np.asarray(ds.variables["Bnormal_from_plasma_current"][:], dtype=float)
        Bnet = np.asarray(ds.variables["Bnormal_from_net_coil_currents"][:], dtype=float)
        Phi = np.asarray(ds.variables["current_potential"][:], dtype=float)
        theta_p = np.asarray(ds.variables["theta_plasma"][:], dtype=float)
        zeta_p = np.asarray(ds.variables["zeta_plasma"][:], dtype=float)
        theta_c = np.asarray(ds.variables["theta_coil"][:], dtype=float)
        zeta_c = np.asarray(ds.variables["zeta_coil"][:], dtype=float)
        r_plasma = np.asarray(ds.variables["r_plasma"][:], dtype=float)
        r_coil = np.asarray(ds.variables["r_coil"][:], dtype=float)
    finally:
        ds.close()

    rms_B = np.sqrt(np.mean(Btot * Btot, axis=(1, 2)))
    ilam = int(np.argmin(rms_B)) if args.lambda_index is None else int(args.lambda_index)
    print(f"[lambda] idx={ilam} lambda={lambdas[ilam]:.3e} rms(Bn_total)={rms_B[ilam]:.3e}")

    # B_sv target (surface current) for this lambda.
    Bsv = Btot[ilam] - (Bplasma + Bnet)

    # Use first field period.
    nzeta = int(zeta_p.size)
    r_plasma_1 = r_plasma[:nzeta]
    r_coil_1 = r_coil[:nzeta]

    from regcoil_jax.surface_utils import unit_normals_from_r_zt3

    n_plasma = unit_normals_from_r_zt3(r_zt3=r_plasma_1)
    pts = r_plasma_1.reshape(-1, 3)
    nhat = n_plasma.reshape(-1, 3)
    targ = Bsv.reshape(-1)

    # Downsample eval points for a fast demo objective.
    rng = np.random.default_rng(0)
    sel = rng.choice(pts.shape[0], size=min(int(args.n_eval), pts.shape[0]), replace=False)
    pts_s = pts[sel]
    nhat_s = nhat[sel]
    targ_s = targ[sel]

    # Initialize level at the median of Φ over one period; initialize current to match scale.
    phi0 = Phi[ilam]
    level0 = float(np.median(phi0))
    I0 = 1e5

    import jax
    import jax.numpy as jnp

    from regcoil_jax.diff_coil_cutting import soft_coil_polyline_xyz, bnormal_from_polyline, polyline_length

    jax.config.update("jax_enable_x64", True)

    phi_j = jnp.asarray(phi0, dtype=jnp.float64)
    theta_j = jnp.asarray(theta_c, dtype=jnp.float64)
    zeta_j = jnp.asarray(zeta_c, dtype=jnp.float64)
    rcoil_j = jnp.asarray(r_coil_1, dtype=jnp.float64)
    x_eval = jnp.asarray(pts_s, dtype=jnp.float64)
    n_eval = jnp.asarray(nhat_s, dtype=jnp.float64)
    target_eval = jnp.asarray(targ_s, dtype=jnp.float64)

    # params = [level, logI]
    p0 = jnp.array([level0, jnp.log(float(I0))], dtype=jnp.float64)

    def loss_fn(p: jnp.ndarray) -> jnp.ndarray:
        level = p[0]
        I = jnp.exp(p[1])
        coil_pts = soft_coil_polyline_xyz(
            phi_zt=phi_j,
            theta=theta_j,
            zeta=zeta_j,
            r_coil_zt3=rcoil_j,
            level=level,
            beta=float(args.beta),
            nfp=int(nfp),
        )
        bn = bnormal_from_polyline(coil_points=coil_pts, coil_current=I, eval_points=x_eval, eval_normals_unit=n_eval, seg_batch=1024)
        err = bn - target_eval
        mse = jnp.mean(err * err)
        L = polyline_length(coil_pts)
        return mse + float(args.length_weight) * (L * L)

    # Simple Adam loop.
    from regcoil_jax.optimize import adam_init, adam_step

    state = adam_init(p0)
    p = p0
    hist = []
    for k in range(int(args.steps)):
        val, g = jax.value_and_grad(loss_fn)(p)
        hist.append(val)
        p, state = adam_step(x=p, state=state, grad=g, lr=float(args.lr))

    p = jax.device_get(p)
    hist = np.asarray(jax.device_get(jnp.stack(hist)), dtype=float)
    level_opt = float(p[0])
    I_opt = float(np.exp(p[1]))
    print(f"[opt] level: {level0:.6g} -> {level_opt:.6g}, I: {I0:.3e} -> {I_opt:.3e}, loss: {hist[0]:.3e} -> {hist[-1]:.3e}")

    # Export before/after coil polylines.
    coil0 = np.asarray(
        soft_coil_polyline_xyz(phi_zt=phi_j, theta=theta_j, zeta=zeta_j, r_coil_zt3=rcoil_j, level=level0, beta=float(args.beta), nfp=int(nfp))
    )
    coil1 = np.asarray(
        soft_coil_polyline_xyz(phi_zt=phi_j, theta=theta_j, zeta=zeta_j, r_coil_zt3=rcoil_j, level=level_opt, beta=float(args.beta), nfp=int(nfp))
    )

    # Figures + VTK
    plt = _setup_matplotlib()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(hist)
    ax.set_xlabel("Adam step")
    ax.set_ylabel("loss")
    ax.set_title("Differentiable (soft) coil cutting: tune level + current")
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_history.png")
    plt.close(fig)

    if True:
        from regcoil_jax.vtk_io import write_vts_structured_grid, write_vtp_polydata

        vtk_dir = out_dir / "vtk"
        vtk_dir.mkdir(exist_ok=True)
        write_vts_structured_grid(vtk_dir / "plasma_surface.vts", points_zt3=r_plasma_1, point_data={"Bsv": Bsv})
        write_vts_structured_grid(vtk_dir / "winding_surface.vts", points_zt3=r_coil_1, point_data={})
        write_vtp_polydata(vtk_dir / "coil_soft_before.vtp", points=coil0, lines=[list(range(coil0.shape[0]))])
        write_vtp_polydata(vtk_dir / "coil_soft_after.vtp", points=coil1, lines=[list(range(coil1.shape[0]))])

    print(f"[done] outputs: {out_dir}")


if __name__ == "__main__":
    main()
