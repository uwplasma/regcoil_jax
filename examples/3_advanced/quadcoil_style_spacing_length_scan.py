#!/usr/bin/env python3
"""Quadcoil-style diagnostics demo: coil spacing and total coil length from ∇Φ on the winding surface.

This script illustrates how "coil quality" metrics can be estimated *without cutting coils* by using
the winding-surface current potential Φ(θ,ζ) and its surface gradient.

Key idea (coarea-inspired):
  - Coils are Φ-contours on the winding surface.
  - The local spacing between adjacent contours is approximately:
        Δs ≈ ΔΦ / |∇_s Φ|
    where |∇_s Φ| is the surface gradient magnitude.
  - The total contour length can be estimated by:
        L_total ≈ (1/ΔΦ) ∫ |∇_s Φ| dA

These diagnostics are differentiable w.r.t the current-potential coefficients (they depend on Φ only,
not on contouring).

Outputs:
  - figures: tradeoff curves across lambda (chi2_B, chi2_K, min coil spacing, length estimate)
  - optional: cut coils for the selected lambda and write ParaView VTK
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
        help="REGCOIL-style namelist input (must be named regcoil_in.*).",
    )
    ap.add_argument("--coils_per_half_period", type=int, default=6, help="used for ΔΦ estimates (same as coil cutting)")
    ap.add_argument("--min_spacing", type=float, default=0.05, help="target minimum coil spacing estimate [m]")
    ap.add_argument("--cut_coils", action="store_true", help="cut coils for the selected lambda and write VTK/filaments")
    ap.add_argument("--theta_shift", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default=None)
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
        out_dir = examples_dir / f"outputs_quadcoil_style_{stamp}"
    else:
        out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run regcoil_jax to get output files (Phi, solutions, diagnostics).
    from regcoil_jax.run import run_regcoil

    input_copy = out_dir / input_path.name
    input_copy.write_text(input_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    res = run_regcoil(str(input_copy), verbose=True)

    ds = netCDF4.Dataset(res.output_nc, "r")
    try:
        lambdas = np.asarray(ds.variables["lambda"][:], dtype=float)
        chi2_B = np.asarray(ds.variables["chi2_B"][:], dtype=float)
        chi2_K = np.asarray(ds.variables["chi2_K"][:], dtype=float)
        solutions = np.asarray(ds.variables["solution"][:], dtype=float)  # (nlambda, nbasis)
        nfp = int(ds.variables["nfp"][()])
        net_pol = float(ds.variables["net_poloidal_current_Amperes"][()])
        r_coil_full = np.asarray(ds.variables["r_coil"][:], dtype=float)  # (nzetal, ntheta, 3)
        theta_c = np.asarray(ds.variables["theta_coil"][:], dtype=float)
        zeta_c = np.asarray(ds.variables["zeta_coil"][:], dtype=float)
        Phi_zt = np.asarray(ds.variables["current_potential"][:], dtype=float)  # (nlambda, nzeta, ntheta)
    finally:
        ds.close()

    # Compute ∇Φ-based metrics directly from (Phi grid, coil surface geometry),
    # so this works for *any* geometry_option_coil (VMEC, EFIT, FOCUS, analytic, ...).
    import jax
    import jax.numpy as jnp

    from regcoil_jax.quadcoil_objectives import quadcoil_metrics_from_phi_grid

    jax.config.update("jax_enable_x64", True)

    nzeta = int(zeta_c.size)
    r_coil_1 = jnp.asarray(r_coil_full[:nzeta], dtype=jnp.float64)
    metrics = []
    for i in range(Phi_zt.shape[0]):
        metrics.append(
            quadcoil_metrics_from_phi_grid(
                phi_zt=jnp.asarray(Phi_zt[i], dtype=jnp.float64),
                r_coil_zt3=r_coil_1,
                nfp=int(nfp),
                net_poloidal_current_Amperes=float(net_pol),
                coils_per_half_period=int(args.coils_per_half_period),
            )
        )

    min_spacing = np.array([m.coil_spacing_min for m in metrics], dtype=float)
    length_est = np.array([m.total_contour_length_est for m in metrics], dtype=float)

    # Choose a lambda that satisfies the spacing constraint, preferring lowest chi2_B among feasible ones.
    feasible = min_spacing >= float(args.min_spacing)
    if np.any(feasible):
        idx = int(np.argmin(np.where(feasible, chi2_B, np.inf)))
        reason = f"min_spacing >= {float(args.min_spacing):.3g} m"
    else:
        idx = int(np.argmin(chi2_B))
        reason = "no feasible lambda; picked min chi2_B"

    chosen = metrics[idx]
    print(
        f"[choose] idx={idx} lambda={lambdas[idx]:.3e} ({reason}) "
        f"min_spacing={chosen.coil_spacing_min:.3e}m length_est={chosen.total_contour_length_est:.3e}m"
    )

    # ----------------------------
    # Figures
    # ----------------------------
    plt = _setup_matplotlib()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.loglog(chi2_K, chi2_B, "o-", ms=3)
    ax.plot([chi2_K[idx]], [chi2_B[idx]], "ro", ms=6, label="chosen")
    ax.set_xlabel(r"$\\chi^2_K$")
    ax.set_ylabel(r"$\\chi^2_B$")
    ax.set_title("REGCOIL tradeoff curve")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "tradeoff_chi2.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.semilogx(lambdas, min_spacing, "o-", ms=3)
    ax.axhline(float(args.min_spacing), color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.plot([lambdas[idx]], [min_spacing[idx]], "ro", ms=6)
    ax.set_xlabel(r"$\\lambda$")
    ax.set_ylabel("estimated min coil spacing [m]")
    ax.set_title("Coil spacing estimate from ∇Φ")
    fig.tight_layout()
    fig.savefig(fig_dir / "coil_spacing_est.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.semilogx(lambdas, length_est, "o-", ms=3)
    ax.plot([lambdas[idx]], [length_est[idx]], "ro", ms=6)
    ax.set_xlabel(r"$\\lambda$")
    ax.set_ylabel("estimated total contour length [m]")
    ax.set_title("Total coil length estimate from ∇Φ (coarea-inspired)")
    fig.tight_layout()
    fig.savefig(fig_dir / "coil_length_est.png")
    plt.close(fig)

    # ----------------------------
    # Optional: cut coils and write VTK/filaments for the chosen lambda
    # ----------------------------
    if args.cut_coils:
        from regcoil_jax.coil_cutting import cut_coils_from_current_potential, write_makecoil_filaments
        from regcoil_jax.vtk_io import write_vtp_polydata, write_vts_structured_grid

        coils = cut_coils_from_current_potential(
            current_potential_zt=Phi_zt[idx],
            theta=theta_c,
            zeta=zeta_c,
            r_coil_zt3_full=r_coil_full,
            theta_shift=int(args.theta_shift),
            coils_per_half_period=int(args.coils_per_half_period),
            nfp=int(nfp),
            net_poloidal_current_Amperes=float(net_pol),
        )
        write_makecoil_filaments(out_dir / "coils.makecoil", filaments_xyz=coils.filaments_xyz, coil_currents=coils.coil_currents, nfp=int(nfp))

        vtk_dir = out_dir / "vtk"
        vtk_dir.mkdir(exist_ok=True)
        # Winding surface:
        write_vts_structured_grid(vtk_dir / "winding_surface.vts", points_zt3=r_coil_full, point_data={})
        # Coils:
        pts = np.concatenate(coils.filaments_xyz, axis=0)
        conn = []
        off = 0
        for ln in coils.filaments_xyz:
            n = int(ln.shape[0])
            conn.append(list(range(off, off + n)))
            off += n
        write_vtp_polydata(vtk_dir / "coils.vtp", points=pts, lines=conn)

    print(f"[done] outputs: {out_dir}")


if __name__ == "__main__":
    main()
