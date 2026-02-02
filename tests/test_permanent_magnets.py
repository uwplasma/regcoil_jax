from __future__ import annotations

import numpy as np


def test_solve_dipole_moments_cg_recovers_target():
    import jax
    import jax.numpy as jnp

    from regcoil_jax.dipoles import dipole_bnormal
    from regcoil_jax.permanent_magnets import solve_dipole_moments_ridge_cg

    jax.config.update("jax_enable_x64", True)

    rng = np.random.default_rng(0)
    # Points on a sphere: normals are radial.
    N = 80
    x = rng.normal(size=(N, 3))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    points = 1.5 * x
    normals = x

    M = 6
    dip_pos = rng.normal(size=(M, 3))
    dip_pos = 3.0 * dip_pos / np.linalg.norm(dip_pos, axis=1, keepdims=True)
    true_m = rng.normal(size=(M, 3)) * 2e3

    target = np.asarray(dipole_bnormal(points=points, normals_unit=normals, positions=dip_pos, moments=true_m, batch=1024))
    res = solve_dipole_moments_ridge_cg(
        points=points,
        normals_unit=normals,
        dipole_positions=dip_pos,
        target_bnormal=target,
        l2_moment=0.0,
        batch=1024,
        tol=1e-12,
        maxiter=200,
    )
    assert res.cg_info in (0, -1)  # -1 can occur when cg hits maxiter but still returns a usable iterate
    pred = np.asarray(dipole_bnormal(points=points, normals_unit=normals, positions=dip_pos, moments=res.dipole_moments, batch=1024))
    rel = np.linalg.norm(pred - target) / np.linalg.norm(target)
    assert rel < 5e-11


def test_optimize_fixed_magnitude_orientations_smoke():
    import jax
    import jax.numpy as jnp

    from regcoil_jax.dipoles import dipole_bnormal
    from regcoil_jax.permanent_magnets import optimize_dipole_orientations_fixed_magnitude

    jax.config.update("jax_enable_x64", True)

    rng = np.random.default_rng(1)
    N = 60
    points = rng.normal(size=(N, 3))
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    normals = points.copy()

    M = 5
    dip_pos = rng.normal(size=(M, 3))
    dip_pos = 2.5 * dip_pos / np.linalg.norm(dip_pos, axis=1, keepdims=True)

    m0 = 5e3
    true_dir = rng.normal(size=(M, 3))
    true_dir = true_dir / np.linalg.norm(true_dir, axis=1, keepdims=True)
    true_m = m0 * true_dir
    target = np.asarray(dipole_bnormal(points=points, normals_unit=normals, positions=dip_pos, moments=true_m, batch=1024))

    # Start from a poor initial orientation.
    v0 = np.ones((M, 3))
    out = optimize_dipole_orientations_fixed_magnitude(
        points=points,
        normals_unit=normals,
        dipole_positions=dip_pos,
        target_bnormal=target,
        moment_magnitude=m0,
        v0=v0,
        steps=40,
        lr=5e-2,
        batch=1024,
    )
    assert out.loss_history.size == 40
    mags = np.linalg.norm(out.dipole_moments, axis=1)
    assert np.allclose(mags, m0, rtol=3e-3, atol=0.0)
