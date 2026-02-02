from __future__ import annotations

from pathlib import Path


def _run_input(tmp_path: Path, input_rel: str):
    from regcoil_jax.run import run_regcoil

    input_src = Path(__file__).resolve().parents[1] / input_rel
    input_text = input_src.read_text(encoding="utf-8", errors="ignore")

    # Copy any referenced auxiliary files into tmp_path so regcoil_jax can resolve paths
    # relative to the input file directory (matching Fortran behavior).
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    # EFIT
    if "efit_filename" in input_text:
        efit_src = examples_dir / "2_intermediate" / "efit_circle.gfile"
        if efit_src.exists():
            (tmp_path / efit_src.name).write_bytes(efit_src.read_bytes())
    # VMEC
    if "wout_filename" in input_text:
        # Replace any relative ../ paths with a local basename, and copy the referenced wout file.
        wout_src = examples_dir / "3_advanced" / "wout_d23p4_tm.nc"
        if wout_src.exists():
            (tmp_path / wout_src.name).write_bytes(wout_src.read_bytes())
            input_text = input_text.replace("../3_advanced/wout_d23p4_tm.nc", wout_src.name)

    input_dst = tmp_path / input_src.name
    input_dst.write_text(input_text, encoding="utf-8")
    run_regcoil(str(input_dst), verbose=False)
    suffix = input_dst.name[len("regcoil_in") :]  # includes leading "."
    out_nc = tmp_path / f"regcoil_out{suffix}.nc"
    out_log = tmp_path / f"regcoil_out{suffix}.log"
    assert out_nc.exists()
    assert out_log.exists()
    return out_nc


def test_plasma_option_5_efit_smoke(tmp_path: Path):
    # geometry_option_plasma=5: EFIT LCFS.
    _run_input(tmp_path, "examples/2_intermediate/regcoil_in.plasma_option_5_efit_lcfs")


def test_plasma_option_4_vmec_straight_fieldline_smoke(tmp_path: Path):
    # geometry_option_plasma=4: VMEC boundary on half grid + straight-field-line theta.
    _run_input(tmp_path, "examples/2_intermediate/regcoil_in.plasma_option_4_vmec_straight_fieldline")
