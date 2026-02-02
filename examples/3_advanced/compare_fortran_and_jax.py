#!/usr/bin/env python3
"""Run Fortran REGCOIL and regcoil_jax on the same input and compare outputs.

This script is meant for local development only (CI does not have the Fortran binary).

Example:
  python examples/3_advanced/compare_fortran_and_jax.py \
    --input examples/3_advanced/regcoil_in.lambda_search_1
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import netCDF4
except Exception:
    netCDF4 = None


def _copy_deps(input_src: Path, dst_dir: Path) -> None:
    txt = input_src.read_text(encoding="utf-8", errors="ignore")

    def _copy_from_key(key: str) -> None:
        m = re.search(rf"{key}\s*=\s*['\"]([^'\"]+)['\"]", txt, flags=re.IGNORECASE)
        if not m:
            return
        p = m.group(1)
        src = Path(p) if os.path.isabs(p) else (input_src.parent / p)
        if src.exists():
            shutil.copy2(src, dst_dir / src.name)

    for k in ("wout_filename", "bnorm_filename", "nescin_filename", "shape_filename_plasma"):
        _copy_from_key(k)


def _expected_output_path(input_basename: str) -> str:
    assert input_basename.startswith("regcoil_in."), f"expected regcoil_in.* input, got {input_basename}"
    suffix = input_basename[len("regcoil_in") :]  # includes leading "."
    return f"regcoil_out{suffix}.nc"


def _compare_scalar_arrays(fortran_nc: Path, jax_nc: Path) -> None:
    if netCDF4 is None:
        raise SystemExit("netCDF4 is required to compare outputs")
    keys = ["lambda", "chi2_B", "chi2_K", "max_Bnormal", "max_K"]
    fds = netCDF4.Dataset(str(fortran_nc), "r")
    jds = netCDF4.Dataset(str(jax_nc), "r")
    try:
        print("Scalar arrays:")
        for k in keys:
            if k not in fds.variables or k not in jds.variables:
                print(f"  - {k}: missing in one file")
                continue
            fa = np.array(fds.variables[k][:], dtype=float)
            ja = np.array(jds.variables[k][:], dtype=float)
            max_abs = float(np.max(np.abs(fa - ja)))
            denom = float(np.max(np.abs(fa)) + 1e-300)
            print(f"  - {k}: max|Δ|={max_abs:.6e}  rel={max_abs/denom:.6e}")
    finally:
        fds.close()
        jds.close()


def main() -> int:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    default_fortran = repo_root.parent / "regcoil" / "regcoil"

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help="Path to regcoil_in.* file (in this repo's examples/)")
    ap.add_argument(
        "--fortran_exe",
        type=str,
        default=str(default_fortran),
        help=f"Path to Fortran regcoil executable (default: {default_fortran})",
    )
    ap.add_argument("--keep_tmp", action="store_true", help="Keep the temporary run directory")
    args = ap.parse_args()

    input_src = (repo_root / args.input).resolve() if not os.path.isabs(args.input) else Path(args.input).resolve()
    if not input_src.exists():
        print(f"ERROR: input file not found: {input_src}")
        return 2

    fortran_exe = Path(args.fortran_exe).resolve()
    if not fortran_exe.exists():
        print(f"ERROR: Fortran executable not found: {fortran_exe}")
        print("Tip: build it in ../regcoil/ first, or pass --fortran_exe.")
        return 2

    tmpdir_obj = tempfile.TemporaryDirectory(prefix="regcoil_compare_")
    tmpdir = Path(tmpdir_obj.name)
    try:
        input_dst = tmpdir / input_src.name
        shutil.copy2(input_src, input_dst)
        _copy_deps(input_src, tmpdir)

        out_name = _expected_output_path(input_src.name)
        out_fortran = tmpdir / out_name.replace(".nc", ".fortran.nc")
        out_jax = tmpdir / out_name.replace(".nc", ".jax.nc")

        # Fortran REGCOIL writes regcoil_out.*.nc in the same directory as input by default.
        print(f"[fortran] running: {fortran_exe} {input_dst.name}")
        r = subprocess.run([str(fortran_exe), input_dst.name], cwd=str(tmpdir), capture_output=True, text=True)
        if r.returncode != 0:
            print("Fortran regcoil failed:")
            print(r.stdout)
            print(r.stderr)
            return r.returncode
        produced = tmpdir / out_name
        if not produced.exists():
            print(f"ERROR: Fortran did not produce expected output {produced}")
            return 2
        produced.rename(out_fortran)

        # regcoil_jax CLI.
        env = os.environ.copy()
        env.setdefault("JAX_ENABLE_X64", "True")
        env.setdefault("JAX_PLATFORM_NAME", "cpu")
        env.setdefault("JAX_TRACEBACK_FILTERING", "off")
        print(f"[jax] running: python -m regcoil_jax.cli --platform cpu {input_dst.name}")
        r = subprocess.run(
            [sys.executable, "-m", "regcoil_jax.cli", "--platform", "cpu", str(input_dst.name)],
            cwd=str(tmpdir),
            env=env,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print("regcoil_jax failed:")
            print(r.stdout)
            print(r.stderr)
            return r.returncode
        produced = tmpdir / out_name
        if not produced.exists():
            print(f"ERROR: regcoil_jax did not produce expected output {produced}")
            return 2
        produced.rename(out_jax)

        print(f"Wrote:\n  - {out_fortran}\n  - {out_jax}")
        _compare_scalar_arrays(out_fortran, out_jax)

        if netCDF4 is not None:
            fds = netCDF4.Dataset(str(out_fortran), "r")
            jds = netCDF4.Dataset(str(out_jax), "r")
            try:
                vset_f = set(fds.variables.keys())
                vset_j = set(jds.variables.keys())
                if vset_f != vset_j:
                    print("WARNING: variable set differs:")
                    print(f"  missing in jax: {sorted(vset_f - vset_j)}")
                    print(f"  extra in jax:   {sorted(vset_j - vset_f)}")
            finally:
                fds.close()
                jds.close()

        return 0
    finally:
        if args.keep_tmp:
            print(f"[tmp] kept: {tmpdir}")
        else:
            tmpdir_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

