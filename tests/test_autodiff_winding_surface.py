from __future__ import annotations

import jax
import jax.numpy as jnp

from regcoil_jax.geometry_fourier import FourierSurface
from regcoil_jax.surfaces import coil_surface_from_inputs


def test_offset_surface_separation_is_differentiable():
    # Keep parity-style numerical behavior in tests.
    jax.config.update("jax_enable_x64", True)

    # Minimal FourierSurface representing a circular torus boundary (nfp=1).
    vmec_like = FourierSurface(
        nfp=1,
        lasym=False,
        xm=jnp.asarray([0, 1], dtype=jnp.int32),
        xn=jnp.asarray([0, 0], dtype=jnp.int32),
        rmnc=jnp.asarray([10.0, 0.5], dtype=jnp.float64),
        zmns=jnp.asarray([0.0, 0.5], dtype=jnp.float64),
        rmns=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        zmnc=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
    )

    plasma_min = {"nfp": 1}

    def area_proxy(separation: jnp.ndarray) -> jnp.ndarray:
        inputs = dict(
            geometry_option_coil=2,
            ntheta_coil=4,
            nzeta_coil=4,
            separation=separation,
            # Keep the Fourier fit small for speed:
            max_mpol_coil=4,
            max_ntor_coil=4,
            mpol_coil_filter=4,
            ntor_coil_filter=4,
        )
        coil = coil_surface_from_inputs(inputs, plasma_min, vmec_like)
        return jnp.sum(coil["normN"])

    s0 = jnp.asarray(0.25, dtype=jnp.float64)
    val = area_proxy(s0)
    g = jax.grad(area_proxy)(s0)

    assert jnp.isfinite(val)
    assert jnp.isfinite(g)
    # For a smooth offset, the area proxy should depend on separation.
    assert jnp.abs(g) > 0
