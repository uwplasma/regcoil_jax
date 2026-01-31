from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import netCDF4
except Exception as e:  # pragma: no cover
    netCDF4 = None


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
BASELINES_PATH = HERE.parent / "baselines.json"


def _run_regcoil_jax(input_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    # Reduce noise / make failures easier to interpret in CI:
    env.setdefault("JAX_TRACEBACK_FILTERING", "off")
    return subprocess.run(
        [sys.executable, "-m", "regcoil_jax.cli", "--platform", "cpu", "--verbose", str(input_path)],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def _copy_example(tmp_dir: Path, input_name: str) -> Path:
    src = EXAMPLES_DIR / input_name
    dst = tmp_dir / input_name
    shutil.copy2(src, dst)

    # VMEC-based case needs the wout file next to the input.
    if "wout_d23p4_tm.nc" in src.read_text(encoding="utf-8", errors="ignore"):
        shutil.copy2(EXAMPLES_DIR / "wout_d23p4_tm.nc", tmp_dir / "wout_d23p4_tm.nc")

    return dst


def _expected_output_paths(tmp_dir: Path, input_name: str) -> tuple[Path, Path]:
    suffix = input_name[len("regcoil_in") :]  # includes leading "."
    out_nc = tmp_dir / f"regcoil_out{suffix}.nc"
    out_log = tmp_dir / f"regcoil_out{suffix}.log"
    return out_nc, out_log


def _load_baselines() -> dict:
    return json.loads(BASELINES_PATH.read_text(encoding="utf-8"))


def _read_nc_scalars(path: Path) -> dict[str, np.ndarray]:
    if netCDF4 is None:  # pragma: no cover
        raise RuntimeError("netCDF4 is required for tests")
    keys = ["lambda", "chi2_B", "chi2_K", "max_Bnormal", "max_K"]
    ds = netCDF4.Dataset(str(path), "r")
    out = {}
    for k in keys:
        assert k in ds.variables, f"missing variable {k} in {path}"
        out[k] = np.array(ds.variables[k][:], dtype=float)
    ds.close()
    return out


def _assert_close(actual: np.ndarray, expected: np.ndarray, *, rtol: float = 1e-9, atol: float = 1e-11):
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"max|Δ|={np.max(np.abs(actual-expected))} "
        f"max|Δ|/max|expected|={np.max(np.abs(actual-expected)) / (np.max(np.abs(expected)) + 1e-300)}"
    )


def _assert_output_self_consistent(path: Path):
    ds = netCDF4.Dataset(str(path), "r")

    nfp = int(ds.variables["nfp"][()]) if "nfp" in ds.variables else 1

    # Plasma consistency checks
    assert "theta_plasma" in ds.variables
    assert "zeta_plasma" in ds.variables
    assert "norm_normal_plasma" in ds.variables
    assert "Bnormal_total" in ds.variables

    theta_p = np.array(ds.variables["theta_plasma"][:], dtype=float)
    zeta_p = np.array(ds.variables["zeta_plasma"][:], dtype=float)
    normN_p = np.array(ds.variables["norm_normal_plasma"][:], dtype=float)  # (nzeta, ntheta)
    Btot = np.array(ds.variables["Bnormal_total"][:], dtype=float)  # (nlambda, nzeta, ntheta)

    dth_p = float(theta_p[1] - theta_p[0]) if theta_p.size > 1 else 0.0
    dze_p = float(zeta_p[1] - zeta_p[0]) if zeta_p.size > 1 else 0.0

    max_B_from_field = np.max(np.abs(Btot), axis=(1, 2))
    max_B = np.array(ds.variables["max_Bnormal"][:], dtype=float)
    _assert_close(max_B_from_field, max_B, rtol=1e-9, atol=1e-11)

    chi2_B_from_field = nfp * dth_p * dze_p * np.sum((Btot * Btot) * normN_p[None, :, :], axis=(1, 2))
    chi2_B = np.array(ds.variables["chi2_B"][:], dtype=float)
    _assert_close(chi2_B_from_field, chi2_B, rtol=1e-8, atol=1e-10)

    # Coil consistency checks
    assert "theta_coil" in ds.variables
    assert "zeta_coil" in ds.variables
    assert "norm_normal_coil" in ds.variables
    assert "K2" in ds.variables

    theta_c = np.array(ds.variables["theta_coil"][:], dtype=float)
    zeta_c = np.array(ds.variables["zeta_coil"][:], dtype=float)
    normN_c = np.array(ds.variables["norm_normal_coil"][:], dtype=float)  # (nzeta, ntheta)
    K2 = np.array(ds.variables["K2"][:], dtype=float)  # (nlambda, nzeta, ntheta)

    dth_c = float(theta_c[1] - theta_c[0]) if theta_c.size > 1 else 0.0
    dze_c = float(zeta_c[1] - zeta_c[0]) if zeta_c.size > 1 else 0.0

    max_K_from_field = np.sqrt(np.max(K2, axis=(1, 2)))
    max_K = np.array(ds.variables["max_K"][:], dtype=float)
    _assert_close(max_K_from_field, max_K, rtol=1e-8, atol=1e-10)

    chi2_K_from_field = nfp * dth_c * dze_c * np.sum(K2 * normN_c[None, :, :], axis=(1, 2))
    chi2_K = np.array(ds.variables["chi2_K"][:], dtype=float)
    _assert_close(chi2_K_from_field, chi2_K, rtol=1e-8, atol=1e-8)

    ds.close()


def test_examples_match_baselines(tmp_path: Path):
    baselines = _load_baselines()
    cases = [
        ("axisymmetrySanityTest_chi2K_regularization", "regcoil_in.axisymmetrySanityTest_chi2K_regularization"),
        ("compareToMatlab1", "regcoil_in.compareToMatlab1"),
        ("compareToMatlab1_option1", "regcoil_in.compareToMatlab1_option1"),
        ("lambda_search_1", "regcoil_in.lambda_search_1"),
    ]

    for case_name, input_name in cases:
        input_path = _copy_example(tmp_path, input_name)
        out_nc, out_log = _expected_output_paths(tmp_path, input_name)

        res = _run_regcoil_jax(input_path)
        assert res.returncode == 0, f"{case_name} failed.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

        assert out_nc.exists(), f"missing {out_nc}"
        assert out_log.exists(), f"missing {out_log}"

        actual = _read_nc_scalars(out_nc)
        expected = {k: np.array(v, dtype=float) for k, v in baselines[case_name].items()}

        for k in expected:
            _assert_close(actual[k], expected[k])

        _assert_output_self_consistent(out_nc)

        # For general_option=5 (lambda search), the last lambda is the chosen one and should be
        # written to the summary log.
        if case_name == "lambda_search_1":
            log_txt = out_log.read_text(encoding="utf-8", errors="ignore")
            assert "chosen_idx=" in log_txt
            assert f"chosen_idx={len(expected['lambda']) - 1}" in log_txt
