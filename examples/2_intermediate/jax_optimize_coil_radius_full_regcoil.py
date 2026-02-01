#!/usr/bin/env python3
"""Differentiate through the full REGCOIL pipeline (analytic torus geometry).

This example demonstrates one of the main advantages of a JAX port:
you can compute gradients through:

- geometry (here: the coil minor radius `a_coil`)
- matrix assembly (Biot–Savart-style kernels)
- the linear solve (`jnp.linalg.solve`)

and then use those gradients in an outer-loop optimization.

This script keeps the problem small so it runs quickly on CPU.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from regcoil_jax.build_matrices_jax import build_matrices
from regcoil_jax.geometry_torus import torus_xyz_and_derivs
from regcoil_jax.solve_jax import diagnostics, solve_one_lambda


def _surface_dict(*, nfp: int, ntheta: int, nzeta: int, R0: float, a: jnp.ndarray) -> dict:
    theta = (2.0 * jnp.pi) * jnp.arange(ntheta) / ntheta
    zeta = (2.0 * jnp.pi / nfp) * jnp.arange(nzeta) / nzeta
    r, rth, rze, nunit, normN = torus_xyz_and_derivs(theta, zeta, R0, a)
    return dict(nfp=nfp, theta=theta, zeta=zeta, r=r, rth=rth, rze=rze, nunit=nunit, normN=normN)


def main():
    jax.config.update("jax_enable_x64", True)
    try:
        jax.config.update("jax_default_matmul_precision", "highest")
    except Exception:
        pass

    # Small-ish grid for quick runs.
    nfp = 1
    ntheta = 24
    nzeta = 24

    # Geometry: simple circular tori.
    R0_plasma = 3.0
    a_plasma = 1.0
    R0_coil = 3.0

    # Current settings (same as the compareToMatlab1 examples).
    net_pol = 1.4
    net_tor = 0.3

    # Potential basis controls the number of unknowns.
    mpol_potential = 8
    ntor_potential = 8
    symmetry_option = 3

    # Pick one regularization strength to keep this example simple.
    lam = jnp.asarray(1.0e-10)

    def objective(a_coil: jnp.ndarray) -> jnp.ndarray:
        plasma = _surface_dict(nfp=nfp, ntheta=ntheta, nzeta=nzeta, R0=R0_plasma, a=jnp.asarray(a_plasma))
        coil = _surface_dict(nfp=nfp, ntheta=ntheta, nzeta=nzeta, R0=R0_coil, a=a_coil)

        inputs = dict(
            ntheta_plasma=ntheta,
            nzeta_plasma=nzeta,
            ntheta_coil=ntheta,
            nzeta_coil=nzeta,
            mpol_potential=mpol_potential,
            ntor_potential=ntor_potential,
            symmetry_option=symmetry_option,
            net_poloidal_current_amperes=net_pol,
            net_toroidal_current_amperes=net_tor,
            load_bnorm=False,
        )
        mats = build_matrices(inputs, plasma, coil)
        sol = solve_one_lambda(mats, lam)
        chi2_B, _chi2_K, _max_B, _max_K = diagnostics(mats, sol[None, :])
        return chi2_B[0]

    obj_jit = jax.jit(objective)
    val_and_grad = jax.jit(jax.value_and_grad(objective))

    a = jnp.asarray(1.7)
    _ = obj_jit(a).block_until_ready()  # compile

    t0 = time.time()
    val0 = float(obj_jit(a).block_until_ready())
    t1 = time.time()
    print(f"init: a_coil={float(a):.6f}  chi2_B={val0:.6e}  eval_s={(t1-t0):.3f}")

    lr = 5e-2
    for k in range(12):
        val, g = val_and_grad(a)
        a = a - lr * g
        print(f"step {k:02d}: a_coil={float(a):.6f}  chi2_B={float(val):.6e}  dchi2/da={float(g):.6e}")


if __name__ == "__main__":
    main()

