from __future__ import annotations

import json
import os
import re
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
    dst = tmp_dir / src.name
    shutil.copy2(src, dst)

    # VMEC-based case needs the wout file next to the input.
    txt = src.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"wout_filename\s*=\s*['\"]([^'\"]+)['\"]", txt, flags=re.IGNORECASE)
    if m:
        wout = m.group(1)
        wout_src = Path(wout) if os.path.isabs(wout) else (src.parent / wout)
        if wout_src.exists():
            shutil.copy2(wout_src, tmp_dir / Path(wout).name)

    # BNORM-based cases need the bnorm file next to the input.
    m = re.search(r"bnorm_filename\s*=\s*['\"]([^'\"]+)['\"]", txt, flags=re.IGNORECASE)
    if m:
        bnorm = m.group(1)
        bnorm_src = Path(bnorm) if os.path.isabs(bnorm) else (src.parent / bnorm)
        if bnorm_src.exists():
            shutil.copy2(bnorm_src, tmp_dir / Path(bnorm).name)

    # NESCOIL-based cases need the nescin file next to the input.
    m = re.search(r"nescin_filename\s*=\s*['\"]([^'\"]+)['\"]", txt, flags=re.IGNORECASE)
    if m:
        nescin = m.group(1)
        nescin_src = Path(nescin) if os.path.isabs(nescin) else (src.parent / nescin)
        if nescin_src.exists():
            shutil.copy2(nescin_src, tmp_dir / Path(nescin).name)

    # Surface-table / FOCUS cases need shape_filename_plasma next to the input.
    m = re.search(r"shape_filename_plasma\s*=\s*['\"]([^'\"]+)['\"]", txt, flags=re.IGNORECASE)
    if m:
        shape = m.group(1)
        shape_src = Path(shape) if os.path.isabs(shape) else (src.parent / shape)
        if shape_src.exists():
            shutil.copy2(shape_src, tmp_dir / Path(shape).name)

    return dst


def _expected_output_paths(tmp_dir: Path, input_basename: str) -> tuple[Path, Path]:
    assert input_basename.startswith("regcoil_in."), f"unexpected input file basename: {input_basename}"
    suffix = input_basename[len("regcoil_in") :]  # includes leading "."
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
        ("axisymmetrySanityTest_chi2K_regularization", "1_simple/regcoil_in.axisymmetrySanityTest_chi2K_regularization"),
        ("axisymmetrySanityTest_Laplace_Beltrami_regularization", "1_simple/regcoil_in.axisymmetrySanityTest_Laplace_Beltrami_regularization"),
        ("compareToMatlab1", "1_simple/regcoil_in.compareToMatlab1"),
        ("compareToMatlab1_option1", "1_simple/regcoil_in.compareToMatlab1_option1"),
        ("plasma_option_6_fourier_table", "1_simple/regcoil_in.plasma_option_6_fourier_table"),
        ("plasma_option_7_focus_embedded_bnorm", "1_simple/regcoil_in.plasma_option_7_focus_embedded_bnorm"),
        ("lambda_search_1", "3_advanced/regcoil_in.lambda_search_1"),
        ("lambda_search_2_current_density_target_too_low", "3_advanced/regcoil_in.lambda_search_2_current_density_target_too_low"),
        ("lambda_search_3_current_density_target_too_high", "3_advanced/regcoil_in.lambda_search_3_current_density_target_too_high"),
        ("lambda_search_4_chi2_B", "3_advanced/regcoil_in.lambda_search_4_chi2_B"),
        ("lambda_search_5_with_bnorm", "3_advanced/regcoil_in.lambda_search_5_with_bnorm"),
        ("compareToMatlab2_geometry_option_coil_3", "3_advanced/regcoil_in.compareToMatlab2_geometry_option_coil_3"),
    ]

    expected_exit_codes = {
        # regcoil_auto_regularization_solve.f90 conventions:
        #  -2: target too low (unachievable), -3: target too high (unachievable)
        "lambda_search_1": 0,
        "lambda_search_2_current_density_target_too_low": -2,
        "lambda_search_3_current_density_target_too_high": -3,
        "lambda_search_4_chi2_B": 0,
        "lambda_search_5_with_bnorm": 0,
    }

    for case_name, input_name in cases:
        input_path = _copy_example(tmp_path, input_name)
        out_nc, out_log = _expected_output_paths(tmp_path, input_path.name)

        res = _run_regcoil_jax(input_path)
        assert res.returncode == 0, f"{case_name} failed.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

        assert out_nc.exists(), f"missing {out_nc}"
        assert out_log.exists(), f"missing {out_log}"

        actual = _read_nc_scalars(out_nc)
        expected = {k: np.array(v, dtype=float) for k, v in baselines[case_name].items()}

        for k in expected:
            # Most parity cases match extremely tightly (1e-9 relative). For larger, less-conditioned
            # systems we allow a slightly looser tolerance (still much smaller than "physics" tolerances).
            if case_name in ("compareToMatlab2_geometry_option_coil_3",):
                _assert_close(actual[k], expected[k], rtol=1e-7, atol=1e-9)
            else:
                _assert_close(actual[k], expected[k])

        _assert_output_self_consistent(out_nc)

        # For Laplace–Beltrami inputs, ensure the extra diagnostics are present and finite.
        if "laplace_beltrami" in case_name.lower():
            ds = netCDF4.Dataset(str(out_nc), "r")
            assert "chi2_Laplace_Beltrami" in ds.variables
            assert "Laplace_Beltrami2" in ds.variables
            chi = np.array(ds.variables["chi2_Laplace_Beltrami"][:], dtype=float)
            assert np.all(np.isfinite(chi))
            ds.close()

        # For general_option=5 (lambda search), validate exit_code and (when applicable) the chosen lambda.
        if case_name in expected_exit_codes:
            log_txt = out_log.read_text(encoding="utf-8", errors="ignore")
            assert f"exit_code={expected_exit_codes[case_name]}" in log_txt
            if expected_exit_codes[case_name] == 0:
                assert "chosen_idx=" in log_txt
                assert f"chosen_idx={len(expected['lambda']) - 1}" in log_txt
            else:
                assert "chosen_idx=" not in log_txt
