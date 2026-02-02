from __future__ import annotations

import numpy as np


def test_fieldline_trace_is_differentiable_wrt_currents():
    import jax
    import jax.numpy as jnp

    from regcoil_jax.biot_savart_jax import segments_from_filaments
    from regcoil_jax.fieldlines_jax import poincare_section_weights, poincare_weighted_RZ, trace_fieldlines_rk4

    jax.config.update("jax_enable_x64", True)

    # Simple square loop in the XY plane (single filament).
    pts = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    segs = segments_from_filaments(filaments_xyz=[pts])

    starts = jnp.asarray([[2.0, 0.0, 0.2], [2.0, 0.0, -0.2]], dtype=jnp.float64)
    nfp = 1

    def loss(I0):
        I = jnp.asarray([I0], dtype=jnp.float64)
        traced = trace_fieldlines_rk4(segs, starts=starts, filament_currents=I, ds=0.02, n_steps=200, stop_radius=10.0)
        w = poincare_section_weights(traced.points, nfp=nfp, phi0=0.0, sigma=0.08)
        Rm, Zm = poincare_weighted_RZ(traced.points, w)
        # Nontrivial differentiable scalar.
        return jnp.mean(Rm * Rm + Zm * Zm)

    g = jax.grad(loss)(jnp.asarray(1.0e5, dtype=jnp.float64))
    assert np.isfinite(float(g))

