from __future__ import annotations

import numpy as np

from regcoil_jax.isosurface_numpy import marching_tetrahedra_mesh


def test_marching_tetrahedra_mesh_sphere_smoke():
    n = 14
    grid = np.linspace(-1.0, 1.0, n)
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing="ij")
    xyz = np.stack([X, Y, Z], axis=-1)
    r0 = 0.7
    phi = X * X + Y * Y + Z * Z - r0 * r0
    pts, tri = marching_tetrahedra_mesh(xyz_ijk3=xyz, phi_ijk=phi, level=0.0)
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert tri.ndim == 2 and tri.shape[1] == 3
    assert tri.size > 0
    radii = np.linalg.norm(pts, axis=1)
    r_mean = float(np.mean(radii))
    assert abs(r_mean - r0) < 0.25

