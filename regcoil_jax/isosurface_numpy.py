from __future__ import annotations

from typing import Any

import numpy as np


def marching_tetrahedra_mesh(
    *,
    xyz_ijk3: Any,
    phi_ijk: Any,
    level: float = 0.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract an isosurface triangle mesh using a simple marching-tetrahedra scheme (NumPy).

    This helper is meant for **visualization output** (e.g. VTK `.vtp`) without relying on
    external heavy dependencies.

    Notes:
      - This is *not differentiable* (Python control flow + discrete case handling).
      - Points are *not deduplicated*; each generated triangle has its own vertices.
      - The algorithm is robust for modest grids (e.g. 24^3) used in examples/tests.

    Args:
      xyz_ijk3: (nx, ny, nz, 3) grid vertex coordinates.
      phi_ijk:  (nx, ny, nz) scalar field at vertices.
      level: isovalue.
      eps: stabilizer for interpolation denominators.
    Returns:
      points: (N,3) triangle vertices (with duplicates).
      tris:   (M,3) triangle connectivity indexing into `points`.
    """
    xyz = np.asarray(xyz_ijk3, dtype=float)
    phi = np.asarray(phi_ijk, dtype=float)
    if xyz.ndim != 4 or xyz.shape[3] != 3:
        raise ValueError("xyz_ijk3 must be (nx, ny, nz, 3)")
    if phi.ndim != 3:
        raise ValueError("phi_ijk must be (nx, ny, nz)")
    if phi.shape != xyz.shape[:3]:
        raise ValueError("phi_ijk shape must match xyz_ijk3 first three dims")
    lvl = float(level)
    eps = float(eps)

    nx, ny, nz, _ = xyz.shape
    if nx < 2 or ny < 2 or nz < 2:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.int64)

    # Cube vertex ordering:
    # 0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0) 4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
    cube_offsets = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.int64,
    )

    # Decompose each cube into 6 tetrahedra using the body diagonal (0 -> 6).
    tetras = [
        (0, 5, 1, 6),
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
    ]

    tetra_edges = [
        (0, 1),
        (1, 2),
        (2, 0),
        (0, 3),
        (1, 3),
        (2, 3),
    ]

    points: list[np.ndarray] = []
    tris: list[list[int]] = []

    def interp(p0, p1, f0, f1) -> np.ndarray:
        t = (lvl - f0) / (f1 - f0 + eps)
        t = float(np.clip(t, 0.0, 1.0))
        return (1.0 - t) * p0 + t * p1

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # Gather cube vertices
                idx = cube_offsets + np.array([i, j, k], dtype=np.int64)[None, :]
                p8 = xyz[idx[:, 0], idx[:, 1], idx[:, 2], :]
                f8 = phi[idx[:, 0], idx[:, 1], idx[:, 2]]

                for tet in tetras:
                    pv = np.stack([p8[t] for t in tet], axis=0)  # (4,3)
                    fv = np.array([f8[t] for t in tet], dtype=float)  # (4,)

                    below = fv < lvl
                    n_below = int(np.sum(below))
                    if n_below == 0 or n_below == 4:
                        continue

                    # Collect intersection points on edges that cross the level.
                    ipts = []
                    for a, b in tetra_edges:
                        fa, fb = fv[a], fv[b]
                        if (fa < lvl) == (fb < lvl):
                            continue
                        ipts.append(interp(pv[a], pv[b], fa, fb))

                    if len(ipts) < 3:
                        continue

                    # Emit triangles. For 4 intersection points (a quad), split into two triangles.
                    base = len(points)
                    points.extend(ipts)
                    if len(ipts) == 3:
                        tris.append([base + 0, base + 1, base + 2])
                    else:
                        tris.append([base + 0, base + 1, base + 2])
                        tris.append([base + 0, base + 2, base + 3])

    if not points:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.int64)

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    tri = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    return pts, tri

