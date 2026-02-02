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


def test_multicoil_relaxation_objective_has_gradients():
    import jax
    import jax.numpy as jnp

    from regcoil_jax.diff_coil_cutting import coil_curves_objective

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(0)
    nzeta, ntheta = 10, 24
    K = 5
    phi = jnp.asarray(rng.normal(size=(nzeta, ntheta)), dtype=jnp.float64)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
    levels = jnp.linspace(0.0, 1.0, K, endpoint=False) + 0.5 / float(K)
    theta_kz0 = jnp.asarray(rng.uniform(low=0.0, high=2.0 * np.pi, size=(K, nzeta)), dtype=jnp.float64)

    def f(x):
        th = x.reshape((K, nzeta))
        return coil_curves_objective(theta_kz=th, phi_zt=phi, theta_grid=theta, levels=levels, smooth_weight=1e-2, repulsion_weight=1e-2)

    g = jax.grad(f)(theta_kz0.reshape((-1,)))
    assert np.all(np.isfinite(np.asarray(g)))
