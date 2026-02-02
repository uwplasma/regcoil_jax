#!/usr/bin/env python3
"""JAX-native fieldline tracing + soft Poincaré objective + gradients.

This script demonstrates an *autodiff-capable* Poincaré pipeline:

  coil currents → Biot–Savart B(x) → field line integration → soft Poincaré statistics

Exact Poincaré section extraction (detecting discrete crossings) is non-differentiable. Here we use
smooth weights (see `regcoil_jax.fieldlines_jax.poincare_section_weights`) and define a toy loss:

  loss = mean_lines (R̄_line - R_target)^2

where R̄_line is a soft section mean radius and R_target is the mean plasma radius at zeta index 0.

This is intended as a minimal, pedagogic proof-of-differentiability; it is not a physics-grade objective.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    import netCDF4  # noqa: F401
except Exception:  # pragma: no cover
    netCDF4 = None


def main() -> None:
    if netCDF4 is None:  # pragma: no cover
        raise SystemExit("netCDF4 is required for this example (pip install regcoil_jax[viz]).")

    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    here = Path(__file__).resolve().parent
    input_path = here / "regcoil_in.lambda_search_1"
    if not input_path.exists():
        raise SystemExit(f"Missing input: {input_path}")

    from regcoil_jax.run import run_regcoil

    res = run_regcoil(str(input_path), verbose=False)
    out_nc = Path(res.output_nc)

    ds = netCDF4.Dataset(str(out_nc), "r")
    try:
        nfp = int(ds.variables["nfp"][()])
        ilam = int(ds.variables["chosen_idx"][()]) if "chosen_idx" in ds.variables else -1
        theta_p = np.asarray(ds.variables["theta_plasma"][:], dtype=float)
        r_plasma = np.asarray(ds.variables["r_plasma"][:], dtype=float)  # (nzetal,ntheta,3)
        r_coil = np.asarray(ds.variables["r_coil"][:], dtype=float)
        theta_c = np.asarray(ds.variables["theta_coil"][:], dtype=float)
        zeta_c = np.asarray(ds.variables["zeta_coil"][:], dtype=float)
        Phi = np.asarray(ds.variables["current_potential"][:], dtype=float)[ilam]
        net_pol = float(ds.variables["net_poloidal_current_Amperes"][()])
    finally:
        ds.close()

    # Cut coils (non-differentiable step; we're differentiating wrt currents afterward).
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

    # Seed points: on plasma surface at zeta index 0, offset outward slightly.
    # Use a small set for a fast demo.
    iz0 = 0
    starts = []
    for it in np.linspace(0, theta_p.size - 1, 6, dtype=int, endpoint=True):
        x = r_plasma[iz0, it]
        # simple radial offset in XY plane
        rxy = x.copy()
        rxy[2] = 0.0
        n = np.linalg.norm(rxy[:2]) + 1e-12
        starts.append(x + 0.1 * (rxy / n))
    starts = np.asarray(starts, dtype=float)

    # Target radius: mean plasma radius at zeta index 0.
    R_target = float(np.mean(np.sqrt(r_plasma[iz0, :, 0] ** 2 + r_plasma[iz0, :, 1] ** 2)))

    import jax
    import jax.numpy as jnp

    from regcoil_jax.biot_savart_jax import segments_from_filaments
    from regcoil_jax.fieldlines_jax import poincare_section_weights, poincare_weighted_RZ, trace_fieldlines_rk4

    jax.config.update("jax_enable_x64", True)

    segs = segments_from_filaments(filaments_xyz=coils.filaments_xyz)
    starts_j = jnp.asarray(starts, dtype=jnp.float64)
    I0 = jnp.asarray(coils.coil_currents, dtype=jnp.float64)

    def loss_fn(I: jnp.ndarray) -> jnp.ndarray:
        traced = trace_fieldlines_rk4(
            segs,
            starts=starts_j,
            filament_currents=I,
            ds=0.03,
            n_steps=500,
            stop_radius=10.0,
            seg_batch=2048,
        )
        w = poincare_section_weights(traced.points, nfp=int(nfp), phi0=0.0, sigma=0.06)
        Rm, _Zm = poincare_weighted_RZ(traced.points, w)
        return jnp.mean((Rm - float(R_target)) ** 2)

    val0, g0 = jax.value_and_grad(loss_fn)(I0)
    print(f"R_target={R_target:.6f}  loss(I0)={float(val0):.6e}  ||grad||={float(jnp.linalg.norm(g0)):.6e}")

    # One small gradient step, just to show it moves.
    I1 = I0 - 1e2 * g0
    val1 = loss_fn(I1)
    print(f"loss(I1)={float(val1):.6e}")


if __name__ == "__main__":
    main()

