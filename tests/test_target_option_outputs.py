from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

try:
    import netCDF4
except Exception:  # pragma: no cover
    netCDF4 = None


def test_target_option_arrays_written_only_on_success(tmp_path: Path):
    if netCDF4 is None:  # pragma: no cover
        raise RuntimeError("netCDF4 is required for tests")

    # This input is designed to *not* converge (exit_code=-1 in the reference Fortran output),
    # so lp_norm_K should remain at the netCDF fill value rather than being overwritten.
    root = Path(__file__).resolve().parents[1]
    src = root / "examples" / "2_intermediate" / "regcoil_in.lambda_search_option4_torus_lp_norm_K"
    assert src.exists()

    dst = tmp_path / src.name
    shutil.copy2(src, dst)

    from regcoil_jax.run import run_regcoil

    run_regcoil(str(dst), verbose=False)

    out_nc = tmp_path / "regcoil_out.lambda_search_option4_torus_lp_norm_K.nc"
    assert out_nc.exists()

    ds = netCDF4.Dataset(str(out_nc), "r")
    try:
        exit_code = int(ds.variables["exit_code"][()])
        assert exit_code != 0
        assert "lp_norm_K" in ds.variables
        v = np.array(ds.variables["lp_norm_K"][:], dtype=float)
        fill = getattr(netCDF4, "default_fillvals", {}).get("f8", 9.969209968386869e36)
        assert np.all(v == float(fill))
    finally:
        ds.close()

