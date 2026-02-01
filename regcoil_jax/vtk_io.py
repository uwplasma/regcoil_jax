from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _as_array_1d(x: Any, *, dtype) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={arr.shape}")
    return arr


def _as_points(x: Any) -> np.ndarray:
    pts = np.asarray(x, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points shaped (N,3), got shape={pts.shape}")
    return pts


def write_vtp_polydata(
    path: str | Path,
    *,
    points: Any,
    polys: Any | None = None,
    lines: Any | None = None,
    point_data: dict[str, Any] | None = None,
    cell_data: dict[str, Any] | None = None,
) -> None:
    """Write a minimal VTK XML PolyData (`.vtp`) file.

    This writer uses ASCII arrays for portability and avoids external VTK/meshio deps.

    Args:
      points: (N,3) float array
      polys:  (M,k) int array of polygon connectivity (k vertices per poly) OR list[list[int]]
      lines:  list[list[int]] of polyline vertex indices
      point_data: dict of arrays with first dimension N
      cell_data: dict of arrays with first dimension M (for polys) or L (for lines); write only when provided.
    """
    path = Path(path)
    pts = _as_points(points)
    n_points = int(pts.shape[0])

    polys_arr = None
    if polys is not None:
        polys_arr = np.asarray(polys, dtype=np.int64)
        if polys_arr.ndim != 2:
            raise ValueError(f"polys must be (M,k) int array, got shape={polys_arr.shape}")
        if np.any(polys_arr < 0) or np.any(polys_arr >= n_points):
            raise ValueError("polys contains out-of-range point indices")

    lines_list = None
    if lines is not None:
        lines_list = [np.asarray(line, dtype=np.int64) for line in lines]
        for line in lines_list:
            if line.ndim != 1:
                raise ValueError("Each line must be a 1D array of indices")
            if np.any(line < 0) or np.any(line >= n_points):
                raise ValueError("lines contains out-of-range point indices")

    n_polys = int(polys_arr.shape[0]) if polys_arr is not None else 0
    n_lines = int(len(lines_list)) if lines_list is not None else 0

    def _fmt_f(arr: np.ndarray) -> str:
        return " ".join(f"{x:.16e}" for x in arr.reshape(-1))

    def _fmt_i(arr: np.ndarray) -> str:
        return " ".join(str(int(x)) for x in arr.reshape(-1))

    def _write_data_arrays(f, data: dict[str, Any], *, n_expected: int, indent: str):
        for name, arr_any in data.items():
            arr = np.asarray(arr_any)
            if arr.shape[0] != n_expected:
                raise ValueError(f"{name}: first dimension {arr.shape[0]} != expected {n_expected}")
            if arr.ndim == 1:
                ncomp = 1
                payload = _fmt_f(arr.astype(float))
            elif arr.ndim == 2:
                ncomp = int(arr.shape[1])
                payload = _fmt_f(arr.astype(float))
            else:
                raise ValueError(f"{name}: expected 1D or 2D array, got shape={arr.shape}")
            f.write(
                f'{indent}<DataArray type="Float64" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">\n'
            )
            f.write(f"{indent}  {payload}\n")
            f.write(f"{indent}</DataArray>\n")

    with path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <PolyData>\n")
        f.write(
            f'    <Piece NumberOfPoints="{n_points}" NumberOfVerts="0" NumberOfLines="{n_lines}" '
            f'NumberOfStrips="0" NumberOfPolys="{n_polys}">\n'
        )

        # PointData
        if point_data:
            f.write("      <PointData>\n")
            _write_data_arrays(f, point_data, n_expected=n_points, indent="        ")
            f.write("      </PointData>\n")
        else:
            f.write("      <PointData/>\n")

        # CellData
        if cell_data:
            total_cells = n_polys + n_lines
            f.write("      <CellData>\n")
            _write_data_arrays(f, cell_data, n_expected=total_cells, indent="        ")
            f.write("      </CellData>\n")
        else:
            f.write("      <CellData/>\n")

        # Points
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write(f"          {_fmt_f(pts.astype(float))}\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")

        # Lines
        if lines_list is not None:
            connectivity = np.concatenate(lines_list, axis=0) if n_lines else np.zeros((0,), dtype=np.int64)
            offsets = np.cumsum(np.array([len(line) for line in lines_list], dtype=np.int64)) if n_lines else np.zeros((0,), dtype=np.int64)
            f.write("      <Lines>\n")
            f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
            f.write(f"          {_fmt_i(connectivity)}\n")
            f.write("        </DataArray>\n")
            f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
            f.write(f"          {_fmt_i(offsets)}\n")
            f.write("        </DataArray>\n")
            f.write("      </Lines>\n")
        else:
            f.write("      <Lines/>\n")

        # Polys
        if polys_arr is not None:
            connectivity = polys_arr.reshape(-1)
            offsets = np.arange(1, n_polys + 1, dtype=np.int64) * polys_arr.shape[1]
            f.write("      <Polys>\n")
            f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
            f.write(f"          {_fmt_i(connectivity)}\n")
            f.write("        </DataArray>\n")
            f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
            f.write(f"          {_fmt_i(offsets)}\n")
            f.write("        </DataArray>\n")
            f.write("      </Polys>\n")
        else:
            f.write("      <Polys/>\n")

        f.write("    </Piece>\n")
        f.write("  </PolyData>\n")
        f.write("</VTKFile>\n")


def quad_mesh_connectivity(*, nzeta: int, ntheta: int, periodic_zeta: bool = True, periodic_theta: bool = True) -> np.ndarray:
    """Return quad connectivity for a (nzeta, ntheta) grid (zeta-major indexing)."""
    if nzeta < 2 or ntheta < 2:
        raise ValueError("Need at least 2x2 grid for quads")

    z_max = nzeta if periodic_zeta else nzeta - 1
    t_max = ntheta if periodic_theta else ntheta - 1

    polys = []
    for iz in range(z_max - 1):
        for it in range(t_max - 1):
            iz1 = (iz + 1) % nzeta
            it1 = (it + 1) % ntheta
            i00 = iz * ntheta + it
            i01 = iz * ntheta + it1
            i11 = iz1 * ntheta + it1
            i10 = iz1 * ntheta + it
            polys.append([i00, i01, i11, i10])
    return np.asarray(polys, dtype=np.int64)


def flatten_grid_points(r_zt3: Any) -> np.ndarray:
    """Flatten (nzeta, ntheta, 3) grid to (N,3) points (zeta-major)."""
    r = np.asarray(r_zt3, dtype=float)
    if r.ndim != 3 or r.shape[2] != 3:
        raise ValueError(f"Expected r_zt3 shape (nzeta,ntheta,3), got {r.shape}")
    return r.reshape(-1, 3)


def write_vts_structured_grid(
    path: str | Path,
    *,
    points_zt3: Any,
    point_data: dict[str, Any] | None = None,
) -> None:
    """Write a minimal VTK XML StructuredGrid (`.vts`) file for a 2D (zeta,theta) surface.

    We treat the surface as a structured grid with extents:
      x = zeta index  (0..nzeta-1)
      y = theta index (0..ntheta-1)
      z = 0

    VTK expects points ordered with x fastest, then y, then z.
    """
    path = Path(path)
    r = np.asarray(points_zt3, dtype=float)
    if r.ndim != 3 or r.shape[2] != 3:
        raise ValueError(f"Expected points_zt3 shape (nzeta,ntheta,3), got {r.shape}")
    nzeta, ntheta, _ = r.shape

    # Order: y (theta) outer, x (zeta) inner -> flatten (ntheta,nzeta,3) with zeta fastest.
    pts = r.transpose(1, 0, 2).reshape(-1, 3)

    def _fmt_f(arr: np.ndarray) -> str:
        return " ".join(f"{x:.16e}" for x in arr.reshape(-1))

    def _write_point_data(f):
        if not point_data:
            f.write("      <PointData/>\n")
            return
        f.write("      <PointData>\n")
        for name, arr_any in point_data.items():
            arr = np.asarray(arr_any)
            # Accept either already-flat (N,...) or grid-shaped (nzeta,ntheta,...) and re-order to match points.
            if arr.shape[0] == nzeta and arr.shape[1] == ntheta and arr.ndim >= 2:
                arr2 = arr.transpose(1, 0, *range(2, arr.ndim)).reshape(pts.shape[0], *arr.shape[2:])
            else:
                arr2 = arr
            if arr2.shape[0] != pts.shape[0]:
                raise ValueError(f"{name}: expected {pts.shape[0]} points, got {arr2.shape}")
            if arr2.ndim == 1:
                ncomp = 1
                payload = _fmt_f(arr2.astype(float))
            elif arr2.ndim == 2:
                ncomp = int(arr2.shape[1])
                payload = _fmt_f(arr2.astype(float))
            else:
                raise ValueError(f"{name}: expected 1D or 2D point array, got shape={arr2.shape}")
            f.write(
                f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">\n'
            )
            f.write(f"          {payload}\n")
            f.write("        </DataArray>\n")
        f.write("      </PointData>\n")

    with path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'  <StructuredGrid WholeExtent="0 {nzeta-1} 0 {ntheta-1} 0 0">\n')
        f.write(f'    <Piece Extent="0 {nzeta-1} 0 {ntheta-1} 0 0">\n')
        _write_point_data(f)
        f.write("      <CellData/>\n")
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write(f"          {_fmt_f(pts)}\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("    </Piece>\n")
        f.write("  </StructuredGrid>\n")
        f.write("</VTKFile>\n")


def write_vtu_point_cloud(
    path: str | Path,
    *,
    points: Any,
    point_data: dict[str, Any] | None = None,
) -> None:
    """Write a minimal VTK XML UnstructuredGrid (`.vtu`) point cloud using VTK_VERTEX cells."""
    path = Path(path)
    pts = _as_points(points)
    n = int(pts.shape[0])

    # One VTK_VERTEX cell per point.
    connectivity = np.arange(n, dtype=np.int64)
    offsets = np.arange(1, n + 1, dtype=np.int64)
    types = np.full((n,), 1, dtype=np.uint8)  # VTK_VERTEX

    def _fmt_f(arr: np.ndarray) -> str:
        return " ".join(f"{x:.16e}" for x in arr.reshape(-1))

    def _fmt_i(arr: np.ndarray) -> str:
        return " ".join(str(int(x)) for x in arr.reshape(-1))

    def _write_point_data(f):
        if not point_data:
            f.write("      <PointData/>\n")
            return
        f.write("      <PointData>\n")
        for name, arr_any in point_data.items():
            arr = np.asarray(arr_any)
            if arr.shape[0] != n:
                raise ValueError(f"{name}: expected {n} points, got {arr.shape}")
            if arr.ndim == 1:
                ncomp = 1
                payload = _fmt_f(arr.astype(float))
            elif arr.ndim == 2:
                ncomp = int(arr.shape[1])
                payload = _fmt_f(arr.astype(float))
            else:
                raise ValueError(f"{name}: expected 1D or 2D point array, got shape={arr.shape}")
            f.write(
                f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">\n'
            )
            f.write(f"          {payload}\n")
            f.write("        </DataArray>\n")
        f.write("      </PointData>\n")

    with path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfCells="{n}">\n')
        _write_point_data(f)
        f.write("      <CellData/>\n")
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write(f"          {_fmt_f(pts)}\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        f.write(f"          {_fmt_i(connectivity)}\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        f.write(f"          {_fmt_i(offsets)}\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write(f"          {_fmt_i(types)}\n")
        f.write("        </DataArray>\n")
        f.write("      </Cells>\n")
        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")
