from __future__ import annotations

import numpy as np


def test_biot_savart_jax_matches_numpy():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from regcoil_jax.biot_savart_jax import segments_from_filaments, bfield_from_segments
    from regcoil_jax.fieldlines import build_filament_field_multi_current, bfield_from_filaments

    # Two simple closed filaments (circles) with different currents.
    t = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    c1 = np.stack([1.0 + 0.2 * np.cos(t), 0.2 * np.sin(t), 0.05 * np.sin(2 * t)], axis=1)
    c2 = np.stack([1.4 + 0.2 * np.cos(t), 0.2 * np.sin(t), -0.05 * np.sin(2 * t)], axis=1)
    filaments = [c1, c2]
    I = np.array([1.1e5, -0.7e5], dtype=float)

    # Compare fields at a few points.
    pts = np.array(
        [
            [1.1, 0.0, 0.0],
            [1.2, 0.1, 0.02],
            [1.5, -0.2, 0.1],
            [1.0, 0.2, -0.1],
        ],
        dtype=float,
    )

    # NumPy reference
    field_np = build_filament_field_multi_current(filaments_xyz=filaments, coil_currents=I)
    B_np = bfield_from_filaments(field_np, pts)

    # JAX
    segs = segments_from_filaments(filaments_xyz=filaments)
    B_jax = np.asarray(bfield_from_segments(segs, points=jnp.asarray(pts), filament_currents=jnp.asarray(I), seg_batch=256))

    assert B_jax.shape == B_np.shape
    assert np.allclose(B_jax, B_np, rtol=2e-12, atol=1e-10)

