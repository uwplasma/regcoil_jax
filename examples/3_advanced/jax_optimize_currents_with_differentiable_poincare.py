#!/usr/bin/env python3
"""Optimize per-coil currents using a differentiable (soft) Poincaré penalty.

This example demonstrates a practical use for a differentiable Poincaré surrogate:

  - Start from a REGCOIL current potential Φ and cut a filament coil set (discrete step).
  - Optimize per-coil currents I_k with autodiff through:
      currents → Biot–Savart → JAX fieldline tracing → soft Poincaré candidate points
  - Penalize the distance of soft Poincaré points to the plasma cross-section at φ=0 (zeta index 0).

The objective is a weighted sum:

  L(I) = w_bn * mean((B·n - B_target)^2) + w_poi * mean( w * dist^2_to_slice ) + l2 * ||I-I0||^2

Notes:
- The Poincaré term here uses the **coil-only** field from filaments, intended as a qualitative constraint.
- Exact Poincaré crossing extraction is non-differentiable; we use `soft_poincare_candidates`.
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


def _copy_aux_files_next_to_input(*, input_path: Path, dst_dir: Path) -> Path:
    """Copy referenced auxiliary files (wout, bnorm, etc) next to a copied input.

    This mirrors the behavior of tests, keeping examples self-contained in an output directory.
    """
    from regcoil_jax.utils import parse_namelist

    inputs = parse_namelist(str(input_path))

    def _maybe_copy(key: str):
        if key not in inputs:
            return
        val = str(inputs[key])
        src = Path(val) if os.path.isabs(val) else (input_path.parent / val)
        if not src.exists():
            return
        dst = dst_dir / src.name
        if dst.exists():
            return
        dst.write_bytes(src.read_bytes())

    for k in [
        "wout_filename",
        "bnorm_filename",
        "nescin_filename",
        "nescout_filename",
        "shape_filename_plasma",
        "efit_filename",
    ]:
        _maybe_copy(k)

    dst_input = dst_dir / input_path.name
    txt = dst_input.read_text(encoding="utf-8", errors="ignore")
    # Rewrite known filename keys to just basenames if they were copied.
    for k in [
        "wout_filename",
        "bnorm_filename",
        "nescin_filename",
        "nescout_filename",
        "shape_filename_plasma",
        "efit_filename",
    ]:
        if k not in inputs:
            continue
        val = str(inputs[k])
        base = Path(val).name
        txt = txt.replace(val, base)
    dst_input.write_text(txt, encoding="utf-8")
    return dst_input


def main() -> None:
    if netCDF4 is None:  # pragma: no cover
        raise SystemExit("netCDF4 is required for this example (pip install regcoil_jax[viz]).")

    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", type=str, default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--input", type=str, default="examples/3_advanced/regcoil_in.lambda_search_1")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--n_lines", type=int, default=6)
    ap.add_argument("--n_steps_line", type=int, default=400)
    ap.add_argument("--ds", type=float, default=0.03)
    ap.add_argument("--poi_sigma", type=float, default=0.05)
    ap.add_argument("--w_bn", type=float, default=1.0)
    ap.add_argument("--w_poi", type=float, default=0.2)
    ap.add_argument("--delta_max", type=float, default=0.5, help="Max fractional even/odd current split (via tanh).")
    args = ap.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("JAX_PLATFORM_NAME", args.platform)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Missing input: {input_path}")
    if not input_path.name.startswith("regcoil_in."):
        raise SystemExit("Input must be named regcoil_in.*")

    out_dir = input_path.parent / f"outputs_opt_currents_soft_poincare_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_copy = out_dir / input_path.name
    input_copy.write_text(input_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    input_copy = _copy_aux_files_next_to_input(input_path=input_path, dst_dir=out_dir)

    from regcoil_jax.run import run_regcoil

    res = run_regcoil(str(input_copy), verbose=True)
    out_nc = Path(res.output_nc)

    ds = netCDF4.Dataset(str(out_nc), "r")
    try:
        nfp = int(ds.variables["nfp"][()])
        ilam = int(ds.variables["chosen_idx"][()]) if "chosen_idx" in ds.variables else -1
        theta_p = np.asarray(ds.variables["theta_plasma"][:], dtype=float)
        zeta_p = np.asarray(ds.variables["zeta_plasma"][:], dtype=float)
        theta_c = np.asarray(ds.variables["theta_coil"][:], dtype=float)
        zeta_c = np.asarray(ds.variables["zeta_coil"][:], dtype=float)
        r_plasma = np.asarray(ds.variables["r_plasma"][:], dtype=float)  # (nzetal,ntheta,3)
        r_coil = np.asarray(ds.variables["r_coil"][:], dtype=float)
        Phi = np.asarray(ds.variables["current_potential"][:], dtype=float)[ilam]
        Btot = np.asarray(ds.variables["Bnormal_total"][:], dtype=float)[ilam]
        Bplasma = np.asarray(ds.variables["Bnormal_from_plasma_current"][:], dtype=float) if "Bnormal_from_plasma_current" in ds.variables else None
        Bnet = np.asarray(ds.variables["Bnormal_from_net_coil_currents"][:], dtype=float) if "Bnormal_from_net_coil_currents" in ds.variables else None
        net_pol = float(ds.variables["net_poloidal_current_Amperes"][()])
    finally:
        ds.close()

    # One field period slices.
    nzeta_p = int(zeta_p.size)
    nzeta_c = int(zeta_c.size)
    r_plasma_1 = r_plasma[:nzeta_p]
    r_coil_1 = r_coil[:nzeta_c]

    # Target B·n: use the REGCOIL surface-current field B_sv for this lambda (when available).
    # REGCOIL writes: Bnormal_total = Bsv + (Bnormal_from_plasma_current + Bnormal_from_net_coil_currents).
    if (Bplasma is not None) and (Bnet is not None):
        Bsv = Btot - (Bplasma + Bnet)
        target_bn = Bsv.reshape(-1)
    else:
        # Fallback: target zero for pedagogic coil-only objective.
        target_bn = np.zeros_like(Btot.reshape(-1))

    # Coil cutting (discrete): use REGCOIL-style contours.
    from regcoil_jax.coil_cutting import cut_coils_from_current_potential

    coils = cut_coils_from_current_potential(
        current_potential_zt=Phi,
        theta=theta_c,
        zeta=zeta_c,
        r_coil_zt3_full=r_coil,
        theta_shift=0,
        coils_per_half_period=6,
        nfp=int(nfp),
        net_poloidal_current_Amperes=float(net_pol),
    )

    # Build evaluation points/normals for B·n objective.
    from regcoil_jax.surface_utils import unit_normals_from_r_zt3

    n_plasma = unit_normals_from_r_zt3(r_zt3=r_plasma_1)
    pts_bn_full = r_plasma_1.reshape(-1, 3)
    nhat_bn_full = n_plasma.reshape(-1, 3)
    target_bn_full = target_bn.reshape(-1)
    # Downsample for speed.
    rng = np.random.default_rng(0)
    sel = rng.choice(pts_bn_full.shape[0], size=min(800, pts_bn_full.shape[0]), replace=False)
    pts_bn = pts_bn_full[sel]
    nhat_bn = nhat_bn_full[sel]
    target_bn = target_bn_full[sel]

    # Plasma cross-section at phi0=0 (zeta index 0).
    iz0 = 0
    slice_xyz = r_plasma_1[iz0]  # (ntheta,3)
    slice_RZ = np.stack([np.sqrt(slice_xyz[:, 0] ** 2 + slice_xyz[:, 1] ** 2), slice_xyz[:, 2]], axis=1)
    # Close the polyline.
    slice_RZ = np.vstack([slice_RZ, slice_RZ[:1]])

    # Seed fieldlines near the slice.
    starts = []
    for it in np.linspace(0, theta_p.size - 1, int(args.n_lines), dtype=int, endpoint=True):
        x = r_plasma_1[iz0, it]
        # outward radial offset in XY:
        rxy = x.copy()
        rxy[2] = 0.0
        n = np.linalg.norm(rxy[:2]) + 1e-12
        starts.append(x + 0.15 * (rxy / n))
    starts = np.asarray(starts, dtype=float)

    import jax
    import jax.numpy as jnp

    from regcoil_jax.biot_savart_jax import bnormal_from_segments, segments_from_filaments
    from regcoil_jax.fieldlines_jax import (
        soft_poincare_candidates,
        softmin_squared_distance_to_polyline_2d,
        trace_fieldlines_rk4,
    )
    from regcoil_jax.optimize import minimize_adam

    jax.config.update("jax_enable_x64", True)

    segs = segments_from_filaments(filaments_xyz=coils.filaments_xyz)
    I_base = jnp.asarray(coils.coil_currents, dtype=jnp.float64)
    ncoils = int(I_base.shape[0])
    # Fixed topology parameterization: overall scale + even/odd split.
    group = jnp.where((jnp.arange(ncoils) % 2) == 0, 1.0, -1.0)
    p0 = jnp.asarray([0.0, 0.0], dtype=jnp.float64)  # [log_scale, raw_delta]

    pts_bn_j = jnp.asarray(pts_bn, dtype=jnp.float64)
    nhat_bn_j = jnp.asarray(nhat_bn, dtype=jnp.float64)
    target_bn_j = jnp.asarray(target_bn, dtype=jnp.float64)
    starts_j = jnp.asarray(starts, dtype=jnp.float64)
    slice_RZ_j = jnp.asarray(slice_RZ, dtype=jnp.float64)

    def currents_from_params(p: jnp.ndarray) -> jnp.ndarray:
        log_scale = p[0]
        raw_delta = p[1]
        delta = float(args.delta_max) * jnp.tanh(raw_delta)
        # Keep topology fixed; allow sign flips if delta is large enough (research use).
        return I_base * jnp.exp(log_scale) * (1.0 + delta * group)

    def _loss_terms(I: jnp.ndarray):
        # B·n term
        bn = bnormal_from_segments(segs, points=pts_bn_j, normals_unit=nhat_bn_j, filament_currents=I, seg_batch=2048)
        loss_bn = jnp.mean((bn - target_bn_j) ** 2)

        # Fieldlines + soft Poincaré candidates
        traced = trace_fieldlines_rk4(
            segs,
            starts=starts_j,
            filament_currents=I,
            ds=float(args.ds),
            n_steps=int(args.n_steps_line),
            stop_radius=10.0,
            seg_batch=2048,
        )
        cand_xyz, w = soft_poincare_candidates(traced.points, nfp=int(nfp), phi0=0.0, alpha=60.0, beta=60.0, gamma=10.0)
        # Flatten candidates across lines/segments
        cand = cand_xyz.reshape((-1, 3))
        w_flat = w.reshape((-1,))
        R = jnp.sqrt(cand[:, 0] * cand[:, 0] + cand[:, 1] * cand[:, 1])
        Z = cand[:, 2]
        rz = jnp.stack([R, Z], axis=1)
        d2 = softmin_squared_distance_to_polyline_2d(rz, polyline_xy=slice_RZ_j, beta=300.0)
        loss_poi = jnp.sum(w_flat * d2) / (jnp.sum(w_flat) + 1e-12)
        return loss_bn, loss_poi

    def loss_fn(p: jnp.ndarray) -> jnp.ndarray:
        I = currents_from_params(p)
        loss_bn, loss_poi = _loss_terms(I)
        return float(args.w_bn) * loss_bn + float(args.w_poi) * loss_poi

    I0 = currents_from_params(p0)
    bn0, poi0 = jax.device_get(_loss_terms(I0))
    print(f"[init] mse_Bn={float(bn0):.6e}  poi_d2={float(poi0):.6e}")

    res_opt = minimize_adam(loss_fn, p0, steps=int(args.steps), lr=float(args.lr), jit=True)
    p_opt = np.asarray(jax.device_get(res_opt.x), dtype=float)
    I_opt = np.asarray(jax.device_get(currents_from_params(jnp.asarray(p_opt))), dtype=float)
    hist = np.asarray(jax.device_get(res_opt.loss_history), dtype=float)
    bn1, poi1 = jax.device_get(_loss_terms(jnp.asarray(I_opt)))
    print(f"[opt]  loss: {hist[0]:.6e} -> {hist[-1]:.6e}")
    print(f"[final] mse_Bn={float(bn1):.6e}  poi_d2={float(poi1):.6e}")

    # Save results.
    np.savetxt(out_dir / "coil_currents_base.txt", np.asarray(coils.coil_currents, dtype=float))
    np.savetxt(out_dir / "coil_currents_optimized.txt", I_opt)
    np.savetxt(out_dir / "params_optimized.txt", p_opt)

    # Figures and VTK: show weighted candidate cloud before/after (threshold in ParaView).
    plt = _setup_matplotlib()
    fig_dir = out_dir / "figures"
    vtk_dir = out_dir / "vtk"
    fig_dir.mkdir(exist_ok=True)
    vtk_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(hist)
    ax.set_xlabel("Adam step")
    ax.set_ylabel("loss")
    ax.set_title("Optimize coil currents with soft Poincaré penalty")
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_history.png")
    plt.close(fig)

    def _write_cloud(name: str, I: jnp.ndarray):
        traced = trace_fieldlines_rk4(segs, starts=starts_j, filament_currents=I, ds=float(args.ds), n_steps=int(args.n_steps_line), stop_radius=10.0)
        cand_xyz, w = soft_poincare_candidates(traced.points, nfp=int(nfp), phi0=0.0)
        cand = np.asarray(jax.device_get(cand_xyz.reshape((-1, 3))), dtype=float)
        w_flat = np.asarray(jax.device_get(w.reshape((-1,))), dtype=float)
        R = np.sqrt(cand[:, 0] ** 2 + cand[:, 1] ** 2)
        Z = cand[:, 2]
        from regcoil_jax.vtk_io import write_vtp_polydata

        write_vtp_polydata(
            vtk_dir / f"{name}.vtp",
            points=cand,
            verts=np.arange(cand.shape[0], dtype=np.int64),
            point_data={"weight": w_flat, "R": R, "Z": Z},
        )

        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        sc = ax.scatter(R, Z, s=6, c=np.clip(w_flat, 0.0, 1.0), cmap="viridis", alpha=0.8, linewidths=0)
        ax.plot(slice_RZ[:, 0], slice_RZ[:, 1], "k-", lw=1.2, alpha=0.7, label="plasma slice")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(name)
        fig.colorbar(sc, ax=ax, label="soft weight")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.png")
        plt.close(fig)

    _write_cloud("soft_poincare_before", jnp.asarray(I0))
    _write_cloud("soft_poincare_after", jnp.asarray(I_opt))

    print(f"[done] outputs: {out_dir}")


if __name__ == "__main__":
    main()
