from __future__ import annotations

import jax
import jax.numpy as jnp

from regcoil_jax.diff_isosurface import soft_marching_cubes_candidates


def test_soft_marching_cubes_candidates_sphere_grad():
    n = 14
    grid = jnp.linspace(-1.0, 1.0, n, dtype=jnp.float64)
    X, Y, Z = jnp.meshgrid(grid, grid, grid, indexing="ij")
    xyz = jnp.stack([X, Y, Z], axis=-1)  # (n,n,n,3)

    level = 0.0
    r_target = 0.75

    def loss(r: jnp.ndarray) -> jnp.ndarray:
        r = jnp.asarray(r, dtype=jnp.float64)
        phi = X * X + Y * Y + Z * Z - r * r
        pts, w = soft_marching_cubes_candidates(xyz_ijk3=xyz, phi_ijk=phi, level=level, alpha=150.0)
        radii = jnp.linalg.norm(pts, axis=1)
        wsum = jnp.sum(w) + 1e-300
        r_mean = jnp.sum(w * radii) / wsum
        return (r_mean - r_target) ** 2

    # Basic sanity: finite weights and points.
    pts0, w0 = soft_marching_cubes_candidates(xyz_ijk3=xyz, phi_ijk=X * X + Y * Y + Z * Z - 0.6**2, level=level, alpha=150.0)
    assert pts0.ndim == 2 and int(pts0.shape[1]) == 3
    assert w0.ndim == 1
    assert int(pts0.shape[0]) == int(w0.shape[0])
    assert jnp.all(jnp.isfinite(pts0))
    assert jnp.all(jnp.isfinite(w0))
    assert float(jnp.min(w0)) >= 0.0
    assert float(jnp.max(w0)) <= 1.0 + 1e-12

    g = jax.grad(loss)(jnp.asarray(0.6, dtype=jnp.float64))
    assert jnp.isfinite(g)
    assert float(jnp.abs(g)) > 1e-6

    # JIT should work too (static shapes).
    g2 = jax.jit(jax.grad(loss))(jnp.asarray(0.6, dtype=jnp.float64))
    assert jnp.isfinite(g2)

