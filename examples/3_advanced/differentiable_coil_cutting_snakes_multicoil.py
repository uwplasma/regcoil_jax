#!/usr/bin/env python3
"""Differentiable, topology-fixed multi-coil extraction ("snakes") + current tuning.

This example demonstrates a practical way to get **end-to-end differentiability**
from a REGCOIL current potential Φ(θ,ζ) to a *filamentary* coil set:

Instead of discrete contour extraction (marching squares), we represent each coil as a
curve θ_k(ζ) sampled on the ζ grid, and optimize θ_k(ζ) so that:

  Φ(ζ, θ_k(ζ)) ≈ level_k

This is a differentiable relaxation with fixed topology (a fixed number of coils), so it
avoids non-differentiable split/merge events. Once we have coil polylines, we optimize
an independent current for each coil to match the REGCOIL surface-current normal field B_sv.

Outputs:
  - loss history figure
  - ParaView VTK files for plasma surface, winding surface, coils (before/after)

See `docs/differentiable_coil_cutting.rst` for discussion and limitations.
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
    ap.add_argument("--steps_theta", type=int, default=120, help="steps for theta(ζ) contour relaxation")
    ap.add_argument("--steps_joint", type=int, default=80, help="joint theta + currents refinement steps")
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--level_weight", type=float, default=50.0, help="weight on level-set mismatch term")
    ap.add_argument("--smooth_weight", type=float, default=5e-3)
    ap.add_argument("--repulsion_weight", type=float, default=3e-3)
    ap.add_argument("--repulsion_alpha", type=float, default=40.0)
    ap.add_argument("--n_eval", type=int, default=800, help="number of plasma points used in the B·n objective")
    args = ap.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("JAX_PLATFORM_NAME", args.platform)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Missing input: {input_path}")
    if not input_path.name.startswith("regcoil_in."):
        raise SystemExit("Input must be named regcoil_in.*")

    out_dir = here / f"outputs_diff_snakes_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_copy = out_dir / input_path.name
    input_copy.write_text(input_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

    from regcoil_jax.utils import parse_namelist
    from regcoil_jax.run import run_regcoil

    inputs = parse_namelist(str(input_path))
    coils_per_half = int(inputs.get("coils_per_half_period", 3))
    theta_shift = int(inputs.get("theta_shift", 0))
    K = int(2 * coils_per_half)  # coils per field period

    res = run_regcoil(str(input_copy), verbose=True)
    out_nc = Path(res.output_nc)

    ds = netCDF4.Dataset(str(out_nc), "r")
    try:
        nfp = int(ds.variables["nfp"][()])
        net_pol = float(ds.variables["net_poloidal_current_Amperes"][()])
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

    # B_sv target (surface current) for this lambda: Btot = Bsv + (Bplasma + Bnet).
    Bsv = Btot[ilam] - (Bplasma + Bnet)

    # One field period in zeta (plasma and coil grids can differ).
    nzeta_p = int(zeta_p.size)
    nzeta_c = int(zeta_c.size)
    r_plasma_1 = r_plasma[:nzeta_p]
    r_coil_1 = r_coil[:nzeta_c]

    from regcoil_jax.surface_utils import unit_normals_from_r_zt3

    n_plasma = unit_normals_from_r_zt3(r_zt3=r_plasma_1)
    pts = r_plasma_1.reshape(-1, 3)
    nhat = n_plasma.reshape(-1, 3)
    targ = Bsv.reshape(-1)

    # Downsample eval points for speed.
    rng = np.random.default_rng(0)
    sel = rng.choice(pts.shape[0], size=min(int(args.n_eval), pts.shape[0]), replace=False)
    pts_s = pts[sel]
    nhat_s = nhat[sel]
    targ_s = targ[sel]

    import jax
    import jax.numpy as jnp

    from regcoil_jax.diff_coil_cutting import (
        bnormal_from_coil_curves,
        coil_curves_objective,
        coil_curves_polyline_xyz,
        replicate_coils_across_nfp,
        polyline_length,
    )
    from regcoil_jax.optimize import minimize_adam

    jax.config.update("jax_enable_x64", True)

    # Normalize Φ as the reference contour-cutting script does: (Φ / net_pol) * nfp.
    phi0 = jnp.asarray(Phi[ilam], dtype=jnp.float64)
    if abs(net_pol) > 0:
        phi0 = (phi0 / float(net_pol)) * float(nfp)
    else:
        phi0 = phi0 / (jnp.max(jnp.abs(phi0)) + 1e-30)

    # Match the contour-cutting shift by rolling theta index.
    phi0 = jnp.roll(phi0, shift=int(theta_shift), axis=1)
    theta_c_j = jnp.roll(jnp.asarray(theta_c, dtype=jnp.float64), shift=int(theta_shift))

    # Contour levels in [0,1) centered in each bin.
    levels0 = (jnp.arange(K, dtype=jnp.float64) + 0.5) / float(K)

    nz = int(zeta_c.size)
    theta_kz0 = jnp.tile((2.0 * jnp.pi * (jnp.arange(K, dtype=jnp.float64) + 0.5) / float(K))[:, None], (1, nz))

    rcoil_j = jnp.asarray(r_coil_1, dtype=jnp.float64)
    x_eval = jnp.asarray(pts_s, dtype=jnp.float64)
    n_eval = jnp.asarray(nhat_s, dtype=jnp.float64)
    target_eval = jnp.asarray(targ_s, dtype=jnp.float64)

    def level_loss(theta_kz: jnp.ndarray) -> jnp.ndarray:
        return coil_curves_objective(
            theta_kz=theta_kz,
            phi_zt=phi0,
            theta_grid=theta_c_j,
            levels=levels0,
            smooth_weight=float(args.smooth_weight),
            repulsion_weight=float(args.repulsion_weight),
            repulsion_alpha=float(args.repulsion_alpha),
        )

    # Stage 1: fit θ_k(ζ) curves to the Φ level sets (no Biot–Savart yet).
    print(f"[stage1] optimizing θ_k(ζ) for {K} coils (one field period), steps={args.steps_theta}")
    res_theta = minimize_adam(lambda x: level_loss(x.reshape((K, nz))), theta_kz0.reshape((-1,)), steps=int(args.steps_theta), lr=float(args.lr))
    theta_kz1 = res_theta.x.reshape((K, nz))

    # Stage 2: joint refinement of θ_k(ζ) and per-coil currents to match B_sv on the plasma surface.
    # Use identical currents across field periods; replicate geometry across nfp for the Biot–Savart evaluation.
    I0 = jnp.full((K,), 1e5, dtype=jnp.float64)

    def joint_loss(x: jnp.ndarray) -> jnp.ndarray:
        th = x[: K * nz].reshape((K, nz))
        logI = x[K * nz :].reshape((K,))
        I = jnp.exp(logI)
        coils = coil_curves_polyline_xyz(theta_kz=th, r_coil_zt3=rcoil_j, theta_grid=theta_c_j)  # (K,nz,3)
        coils_full = replicate_coils_across_nfp(coils_kz3=coils, nfp=int(nfp))
        I_full = jnp.tile(I, (int(nfp),))
        bn = bnormal_from_coil_curves(coils_kz3=coils_full, coil_currents=I_full, eval_points=x_eval, eval_normals_unit=n_eval, seg_batch=2048)
        err = bn - target_eval
        mse = jnp.mean(err * err)
        lvl = level_loss(th)
        # Mild length penalty to discourage pathological wrinkling.
        L = jax.vmap(polyline_length)(coils)
        return mse + float(args.level_weight) * lvl + 1e-6 * jnp.mean(L * L)

    x0 = jnp.concatenate([theta_kz1.reshape((-1,)), jnp.log(I0)], axis=0)
    print(f"[stage2] joint θ+I refinement, steps={args.steps_joint}")
    res_joint = minimize_adam(joint_loss, x0, steps=int(args.steps_joint), lr=float(args.lr))
    x1 = res_joint.x
    theta_kz2 = x1[: K * nz].reshape((K, nz))
    I2 = np.asarray(np.exp(np.asarray(x1[K * nz :])), dtype=float)

    # Build before/after coil polylines (full torus).
    coils0 = np.asarray(replicate_coils_across_nfp(coils_kz3=coil_curves_polyline_xyz(theta_kz=theta_kz0, r_coil_zt3=rcoil_j, theta_grid=theta_c_j), nfp=int(nfp)))
    coils2 = np.asarray(replicate_coils_across_nfp(coils_kz3=coil_curves_polyline_xyz(theta_kz=theta_kz2, r_coil_zt3=rcoil_j, theta_grid=theta_c_j), nfp=int(nfp)))

    # Write figures + VTK.
    plt = _setup_matplotlib()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    loss_hist = np.asarray(jax.device_get(res_joint.loss_history), dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(loss_hist)
    ax.set_xlabel("Adam step")
    ax.set_ylabel("joint loss")
    ax.set_title("Differentiable multi-coil cutting (snakes) + current tuning")
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_history.png")
    plt.close(fig)

    from regcoil_jax.vtk_io import write_vts_structured_grid, write_vtp_polydata

    vtk_dir = out_dir / "vtk"
    vtk_dir.mkdir(exist_ok=True)
    write_vts_structured_grid(vtk_dir / "plasma_surface.vts", points_zt3=r_plasma_1, point_data={"Bsv_target": Bsv})
    write_vts_structured_grid(vtk_dir / "winding_surface.vts", points_zt3=r_coil_1, point_data={"phi_norm": np.asarray(phi0)})

    # Coils as PolyData polylines. Each coil is one polyline line-cell.
    ncoils_full = int(coils0.shape[0])
    lines = [list(range(i * nz, (i + 1) * nz)) for i in range(ncoils_full)]
    write_vtp_polydata(vtk_dir / "coils_snakes_before.vtp", points=coils0.reshape(-1, 3), lines=lines)
    write_vtp_polydata(vtk_dir / "coils_snakes_after.vtp", points=coils2.reshape(-1, 3), lines=lines)

    np.savetxt(out_dir / "coil_currents_after.txt", I2)
    print(f"[done] outputs: {out_dir}")


if __name__ == "__main__":
    main()
