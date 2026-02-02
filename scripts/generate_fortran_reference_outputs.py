#!/usr/bin/env python3
"""Generate and store reference REGCOIL (Fortran) output files for parity tests.

This script runs the compiled Fortran `regcoil` executable on selected example inputs
and copies the resulting `regcoil_out.*.nc` files into `tests/fortran_outputs/`.

It is intended to be run manually by repo maintainers when updating outputs or adding cases.
CI should *not* need the Fortran executable because the reference `.nc` files are committed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from regcoil_jax.utils import parse_namelist, resolve_existing_path


def _copy_deps(*, src_input: Path, inputs: dict, tmp_dir: Path) -> None:
    """Copy auxiliary files referenced by the namelist next to the input in tmp_dir.

    We only copy files that are actually needed by the chosen geometry/options to avoid
    failing on inputs that include unused filenames (some parity inputs do this).
    """
    src_dir = src_input.parent

    gpl = int(inputs.get("geometry_option_plasma", 0))
    gcl = int(inputs.get("geometry_option_coil", 0))
    load_bnorm = bool(inputs.get("load_bnorm", False))

    def _copy_one(key: str) -> None:
        val = inputs.get(key, None)
        if not val:
            return
        p = Path(str(val))
        if not p.is_absolute():
            p = (src_dir / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"{key}={val!r} resolved to {p}, but the file does not exist")
        shutil.copy2(p, tmp_dir / p.name)

    # VMEC wout file is needed for plasma options 2/3/4 and coil options 2/4.
    if gpl in (2, 3, 4) or gcl in (2, 4):
        _copy_one("wout_filename")

    # NESCOIL winding surface file (coil option 3)
    if gcl == 3:
        _copy_one("nescin_filename")

    # Surface shape files for plasma options 6/7
    if gpl in (6, 7):
        _copy_one("shape_filename_plasma")

    # BNORM file is needed only when load_bnorm is explicitly enabled and the plasma shape
    # is not a FOCUS file with embedded Bn coefficients.
    if load_bnorm and gpl != 7:
        _copy_one("bnorm_filename")

    # EFIT is not used by the parity test set yet, but support copying if present.
    if "efit_filename" in inputs:
        _copy_one("efit_filename")


def run_one(*, fortran_exe: Path, input_path: Path, out_dir: Path) -> Path:
    input_path = Path(resolve_existing_path(str(input_path))).resolve()
    if not input_path.name.startswith("regcoil_in."):
        raise ValueError(f"Input file must be named regcoil_in.*, got {input_path.name}")

    inputs = parse_namelist(str(input_path))

    case = input_path.name[len("regcoil_in.") :]
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"regcoil_out.{case}.nc"

    with tempfile.TemporaryDirectory(prefix=f"regcoil_fortran_{case}.") as td:
        tmp_dir = Path(td)
        shutil.copy2(input_path, tmp_dir / input_path.name)
        _copy_deps(src_input=input_path, inputs=inputs, tmp_dir=tmp_dir)

        subprocess.run([str(fortran_exe), input_path.name], cwd=str(tmp_dir), check=True)

        src_out = tmp_dir / f"regcoil_out.{case}.nc"
        if not src_out.exists():
            raise RuntimeError(f"Fortran run did not produce expected output: {src_out}")
        shutil.copy2(src_out, dest)

    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fortran_exe",
        type=str,
        default=str((Path(__file__).resolve().parents[1] / ".." / "regcoil" / "regcoil").resolve()),
        help="Path to compiled Fortran `regcoil` executable",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str((Path(__file__).resolve().parents[1] / "tests" / "fortran_outputs").resolve()),
        help="Destination directory for reference .nc files",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input files (regcoil_in.*). If omitted, uses the default parity test set.",
    )
    args = parser.parse_args()

    fortran_exe = Path(args.fortran_exe).resolve()
    if not fortran_exe.exists():
        raise SystemExit(f"Missing Fortran executable: {fortran_exe}")

    out_dir = Path(args.out_dir).resolve()

    if args.inputs:
        inputs = [Path(p) for p in args.inputs]
    else:
        # Mirror tests/test_examples.py
        inputs = [
            Path("examples/1_simple/regcoil_in.axisymmetrySanityTest_chi2K_regularization"),
            Path("examples/1_simple/regcoil_in.axisymmetrySanityTest_Laplace_Beltrami_regularization"),
            Path("examples/1_simple/regcoil_in.compareToMatlab1"),
            Path("examples/1_simple/regcoil_in.compareToMatlab1_option1"),
            Path("examples/1_simple/regcoil_in.plasma_option_6_fourier_table"),
            Path("examples/1_simple/regcoil_in.plasma_option_7_focus_embedded_bnorm"),
            Path("examples/3_advanced/regcoil_in.lambda_search_1"),
            Path("examples/3_advanced/regcoil_in.lambda_search_2_current_density_target_too_low"),
            Path("examples/3_advanced/regcoil_in.lambda_search_3_current_density_target_too_high"),
            Path("examples/3_advanced/regcoil_in.lambda_search_4_chi2_B"),
            Path("examples/3_advanced/regcoil_in.lambda_search_5_with_bnorm"),
            Path("examples/3_advanced/regcoil_in.compareToMatlab2_geometry_option_coil_3"),
        ]

    for p in inputs:
        dest = run_one(fortran_exe=fortran_exe, input_path=p, out_dir=out_dir)
        print(dest)


if __name__ == "__main__":
    main()
