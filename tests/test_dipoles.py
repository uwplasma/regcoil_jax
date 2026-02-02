from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from regcoil_jax.biot_savart_jax import segments_from_filaments
from regcoil_jax.dipoles import dipole_bfield, dipole_bnormal
from regcoil_jax.dipole_optimization import optimize_filaments_and_dipoles_to_match_bnormal


def test_dipole_field_matches_known_axis_values():
    # Enable x64 for stable tolerances (does not affect other tests).
    jax.config.update("jax_enable_x64", True)

    pos = jnp.array([[0.0, 0.0, 0.0]])
    mom = jnp.array([[0.0, 0.0, 2.0]])  # A m^2

    # On x-axis: B = -mu0/(4pi) * m / r^3 in z-direction
    pts = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    B = np.asarray(dipole_bfield(points=pts, positions=pos, moments=mom, eps=0.0, batch=8), dtype=float)
    expected = np.array([[0.0, 0.0, -2.0e-7], [0.0, 0.0, -(2.0e-7) / 8.0]])
    assert np.allclose(B, expected, rtol=1e-12, atol=1e-14)

    # On z-axis: B = +mu0/(4pi) * 2m / r^3 in z-direction
    pts = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    B = np.asarray(dipole_bfield(points=pts, positions=pos, moments=mom, eps=0.0, batch=8), dtype=float)
    expected = np.array([[0.0, 0.0, +4.0e-7], [0.0, 0.0, +(4.0e-7) / 8.0]])
    assert np.allclose(B, expected, rtol=1e-12, atol=1e-14)


def test_optimize_dipole_moments_recovers_target_bnormal():
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(0)

    # No filament coils in this unit test; we optimize only dipole moments.
    segs = segments_from_filaments(filaments_xyz=[])
    assert segs.n_filaments == 0

    # Random points/normals, but keep them away from the dipole positions for stability.
    pts = rng.normal(size=(64, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts = 3.0 * pts
    nhat = rng.normal(size=(64, 3))
    nhat = nhat / np.linalg.norm(nhat, axis=1, keepdims=True)

    dip_pos = np.array([[0.5, 0.0, 0.0], [-0.4, 0.2, 0.1], [0.1, -0.3, -0.2]], dtype=float)
    true_m = np.array([[0.0, 0.0, 8.0], [1.0, -2.0, 0.5], [-1.0, 0.5, 2.0]], dtype=float)

    target = np.asarray(
        dipole_bnormal(points=pts, normals_unit=nhat, positions=dip_pos, moments=true_m, eps=1e-9, batch=256),
        dtype=float,
    )

    res = optimize_filaments_and_dipoles_to_match_bnormal(
        segs,
        dipole_positions=dip_pos,
        points=pts,
        normals_unit=nhat,
        target_bnormal=target,
        filament_currents0=np.zeros((0,), dtype=float),
        dipole_moments0=np.zeros_like(true_m),
        steps=180,
        lr=5e-2,
        l2_current=0.0,
        l2_moment=0.0,
        seg_batch=64,
        dipole_batch=256,
    )

    pred = np.asarray(
        dipole_bnormal(points=pts, normals_unit=nhat, positions=dip_pos, moments=res.dipole_moments, eps=1e-9, batch=256),
        dtype=float,
    )
    # The objective is convex in m for fixed positions, so we should recover the targets closely.
    assert np.sqrt(np.mean((pred - target) ** 2)) < 5e-7


def test_dipole_bfield_batching_matches_unbatched():
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(0)

    points = rng.normal(size=(17, 3))
    positions = rng.normal(size=(25, 3))
    moments = rng.normal(size=(25, 3))

    B_full = np.asarray(dipole_bfield(points=points, positions=positions, moments=moments, eps=0.0, batch=25), dtype=float)
    B_batched = np.asarray(dipole_bfield(points=points, positions=positions, moments=moments, eps=0.0, batch=8), dtype=float)
    assert np.allclose(B_full, B_batched, rtol=1e-12, atol=1e-14)
