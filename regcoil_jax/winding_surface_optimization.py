from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .geometry_fourier import FourierSurface
from .io_vmec import read_wout_boundary
from .surfaces import plasma_surface_from_inputs, coil_surface_from_inputs
from .build_matrices_jax import build_matrices
from .solve_jax import solve_one_lambda, diagnostics


@dataclass(frozen=True)
class SeparationOptResult:
    separation_history: jnp.ndarray  # (nsteps+1,)
    objective_history: jnp.ndarray   # (nsteps+1,)


def _as_fourier_surface(bound) -> FourierSurface:
    return FourierSurface(
        nfp=int(bound.nfp),
        lasym=bool(bound.lasym),
        xm=jnp.asarray(bound.xm, dtype=jnp.int32),
        xn=jnp.asarray(bound.xn, dtype=jnp.int32),
        rmnc=jnp.asarray(bound.rmnc, dtype=jnp.float64),
        zmns=jnp.asarray(bound.zmns, dtype=jnp.float64),
        rmns=jnp.asarray(bound.rmns, dtype=jnp.float64),
        zmnc=jnp.asarray(bound.zmnc, dtype=jnp.float64),
    )


def optimize_vmec_offset_separation(
    *,
    wout_filename: str,
    separation0: float,
    nsteps: int = 20,
    step_size: float = 0.05,
    # Small grids by default to keep this runnable on CPU in a reasonable time.
    ntheta: int = 16,
    nzeta: int = 16,
    mpol_potential: int = 6,
    ntor_potential: int = 6,
    lam: float = 1.0e-14,
    objective: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
) -> SeparationOptResult:
    """Autodiff-based optimization of the VMEC offset-surface separation parameter.

    This is a minimal “winding surface optimization” example that stays within JAX:
    it differentiates through surface construction, matrix build, and the linear solve.

    The objective defaults to chi2_B(lam) + 1e-20 * chi2_K(lam) to keep the problem
    well-scaled without dominating the physics target.
    """
    if objective is None:
        def objective(chi2_B, chi2_K, max_B, max_K):
            return chi2_B + (1.0e-20 * chi2_K)

    bound = read_wout_boundary(wout_filename, radial_mode="full")
    vmec = _as_fourier_surface(bound)

    base_inputs = dict(
        # Geometry / grids:
        geometry_option_plasma=2,
        geometry_option_coil=2,
        wout_filename=wout_filename,
        ntheta_plasma=int(ntheta),
        nzeta_plasma=int(nzeta),
        ntheta_coil=int(ntheta),
        nzeta_coil=int(nzeta),
        # Potential basis:
        mpol_potential=int(mpol_potential),
        ntor_potential=int(ntor_potential),
        symmetry_option=1,
        # Regularization:
        regularization_term_option="chi2_K",
        # No extra target field:
        load_bnorm=False,
        # Currents are taken from the input (not VMEC) here; this is just a demo.
        net_poloidal_current_amperes=1.0,
        net_toroidal_current_amperes=0.0,
        curpol=1.0,
        save_level=3,
    )

    plasma = plasma_surface_from_inputs(base_inputs, vmec)

    def run_one(separation: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        inputs = dict(base_inputs)
        inputs["separation"] = separation
        coil = coil_surface_from_inputs(inputs, plasma, vmec)
        mats = build_matrices(inputs, plasma, coil)
        sol = solve_one_lambda(mats, jnp.asarray(lam, dtype=jnp.float64))
        chi2_B, chi2_K, max_B, max_K = diagnostics(mats, sol[None, :])
        return chi2_B[0], chi2_K[0], max_B[0], max_K[0]

    def obj_from_separation(separation_unconstrained: jnp.ndarray) -> jnp.ndarray:
        separation = jax.nn.softplus(separation_unconstrained)
        chi2_B, chi2_K, max_B, max_K = run_one(separation)
        return objective(chi2_B, chi2_K, max_B, max_K)

    vng = jax.value_and_grad(obj_from_separation)

    sep0 = jnp.asarray(separation0, dtype=jnp.float64)
    # Softplus parameterization to enforce separation > 0. Interpret separation0 as
    # the physical initial value and map to an unconstrained raw parameter.
    s_raw = jnp.log(jnp.expm1(sep0) + 1e-300)

    sep_hist = [sep0]
    obj_hist = [obj_from_separation(s_raw)]

    for _ in range(int(nsteps)):
        _, grad = vng(s_raw)
        # Simple gradient descent; callers can wrap with more sophisticated optimizers if desired.
        s_raw = s_raw - (jnp.asarray(step_size, dtype=jnp.float64) * grad)
        sep_hist.append(jax.nn.softplus(s_raw))
        obj_hist.append(obj_from_separation(s_raw))

    return SeparationOptResult(
        separation_history=jnp.stack(sep_hist, axis=0),
        objective_history=jnp.stack(obj_hist, axis=0),
    )
