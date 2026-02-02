from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import netCDF4
except Exception:  # pragma: no cover
    netCDF4 = None


def test_k_zeta_regularization_smoke(tmp_path: Path):
    if netCDF4 is None:  # pragma: no cover
        raise RuntimeError("netCDF4 is required for tests")

    from regcoil_jax.run import run_regcoil

    input_rel = "examples/1_simple/regcoil_in.torus_K_zeta_regularization"
    input_src = Path(__file__).resolve().parents[1] / input_rel
    input_dst = tmp_path / input_src.name
    input_dst.write_text(input_src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

    run_regcoil(str(input_dst), verbose=False)

    suffix = input_dst.name[len("regcoil_in") :]  # includes leading "."
    out_nc = tmp_path / f"regcoil_out{suffix}.nc"
    out_log = tmp_path / f"regcoil_out{suffix}.log"
    assert out_nc.exists()
    assert out_log.exists()

    ds = netCDF4.Dataset(str(out_nc), "r")
    try:
        assert "chi2_B" in ds.variables
        assert "chi2_K" in ds.variables
        assert "K2" in ds.variables
        chi2_K = np.array(ds.variables["chi2_K"][:], dtype=float)
        assert np.all(np.isfinite(chi2_K))
    finally:
        ds.close()

