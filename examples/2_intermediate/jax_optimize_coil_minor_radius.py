#!/usr/bin/env python3
from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from regcoil_jax.constants import mu0, pi, twopi
from regcoil_jax.geometry_torus import torus_xyz_and_derivs
from regcoil_jax.kernel import inductance_and_h_sum


def _flatten_3TZ_to_N3(x_3tz):
    return jnp.reshape(jnp.moveaxis(x_3tz, 0, -1), (-1, 3))


def bnet_for_tori(*, ntheta: int, nzeta: int, nfp: int, R0_plasma: float, a_plasma: float, R0_coil: float, a_coil: jnp.ndarray,
                  net_poloidal_current: float, net_toroidal_current: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Bnormal_from_net_coil_currents on a circular torus plasma surface.

    This example intentionally bypasses the full REGCOIL solve and focuses on a JAX-friendly,
    differentiable physics kernel (the h field / Bnet computation).
    """
    theta = twopi * jnp.arange(ntheta) / ntheta
    zeta = (twopi / nfp) * jnp.arange(nzeta) / nzeta

    rP, rthP, rzeP, nunitP, normNP = torus_xyz_and_derivs(theta, zeta, R0_plasma, a_plasma)
    rC, rthC, rzeC, nunitC, normNC = torus_xyz_and_derivs(theta, zeta, R0_coil, a_coil)

    # Full (non-unit) normals, matching REGCOIL conventions.
    NP_full = nunitP * normNP[None, :, :]
    NC_full = nunitC * normNC[None, :, :]

    rP_N3 = _flatten_3TZ_to_N3(rP)
    NP_N3 = _flatten_3TZ_to_N3(NP_full)
    rC_N3 = _flatten_3TZ_to_N3(rC)
    NC_N3 = _flatten_3TZ_to_N3(NC_full)

    factor_for_h = net_poloidal_current * rthC - net_toroidal_current * rzeC
    f_h_N3 = _flatten_3TZ_to_N3(factor_for_h)

    _, h_sum = inductance_and_h_sum(rP_N3, NP_N3, rC_N3, NC_N3, f_h_N3, nfp=nfp)

    dth = twopi / ntheta
    dze = (twopi / nfp) / nzeta
    h = h_sum * (dth * dze * mu0 / (8.0 * pi * pi))

    Bnet = h.reshape(ntheta, nzeta) / normNP
    return Bnet, normNP


def main():
    jax.config.update("jax_enable_x64", True)

    ntheta = 24
    nzeta = 24
    nfp = 1

    R0_plasma = 3.0
    a_plasma = 1.0
    R0_coil = 3.0

    net_pol = 1.4
    net_tor = 0.3

    def objective(a_coil):
        Bnet, normN = bnet_for_tori(
            ntheta=ntheta,
            nzeta=nzeta,
            nfp=nfp,
            R0_plasma=R0_plasma,
            a_plasma=a_plasma,
            R0_coil=R0_coil,
            a_coil=a_coil,
            net_poloidal_current=net_pol,
            net_toroidal_current=net_tor,
        )
        dth = twopi / ntheta
        dze = (twopi / nfp) / nzeta
        chi2_B = nfp * dth * dze * jnp.sum((Bnet * Bnet) * normN)
        return chi2_B

    obj_jit = jax.jit(objective)
    grad_jit = jax.jit(jax.grad(objective))

    a = jnp.asarray(1.7)
    _ = obj_jit(a).block_until_ready()  # compile

    t0 = time.time()
    val0 = float(obj_jit(a).block_until_ready())
    g0 = float(grad_jit(a).block_until_ready())
    t1 = time.time()
    print(f"init: a_coil={float(a):.6f}  chi2_B={val0:.6e}  d(chi2_B)/da={g0:.6e}  eval_s={(t1-t0):.3f}")

    lr = 1e-1
    for k in range(15):
        g = grad_jit(a)
        a = a - lr * g
        val = float(obj_jit(a).block_until_ready())
        print(f"step {k:02d}: a_coil={float(a):.6f}  chi2_B={val:.6e}")


if __name__ == "__main__":
    main()

