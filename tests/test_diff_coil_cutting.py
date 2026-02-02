from __future__ import annotations

import numpy as np


def test_soft_contour_is_differentiable():
    import jax
    import jax.numpy as jnp

    from regcoil_jax.diff_coil_cutting import soft_contour_theta_of_zeta

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(0)
    nzeta, ntheta = 12, 16
    phi = jnp.asarray(rng.normal(size=(nzeta, ntheta)), dtype=jnp.float64)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
    level = 0.1

    def f(x):
        th = soft_contour_theta_of_zeta(phi_zt=phi + x, theta=theta, level=level, beta=500.0)
        return jnp.sum(jnp.sin(th))

    g = jax.grad(f)(jnp.asarray(0.0, dtype=jnp.float64))
    assert np.isfinite(float(g))

