#!/usr/bin/env python3
"""Intermediate end-to-end runner (CLI + optional figures/VTK).

Default input: examples/2_intermediate/regcoil_in.torus_cli_intermediate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    default_input = Path(__file__).with_name("regcoil_in.torus_cli_intermediate")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(default_input), help="Path to regcoil_in.*")
    parser.add_argument("--platform", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--postprocess", action="store_true", help="Also write figures/VTK using the shared postprocess script.")
    parser.add_argument("--no_figures", action="store_true", help="Skip writing matplotlib figures (postprocess only).")
    parser.add_argument("--no_vtk", action="store_true", help="Skip writing ParaView VTK files (postprocess only).")
    parser.add_argument("--no_coils", action="store_true", help="Skip coil cutting (postprocess only).")
    parser.add_argument("--no_fieldlines", action="store_true", help="Skip field line tracing (postprocess only).")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.name.startswith("regcoil_in."):
        raise SystemExit("Input must be named regcoil_in.*")

    project_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "-m", "regcoil_jax.cli", "--platform", args.platform, "--verbose", str(input_path)]
    print("[examples/2_intermediate] running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(project_root), check=True)

    if not args.postprocess:
        return

    post = project_root / "examples" / "3_advanced" / "postprocess_make_figures_and_vtk.py"
    cmd = [sys.executable, str(post), "--input", str(input_path)]
    if args.no_figures:
        cmd.append("--no_figures")
    if args.no_vtk:
        cmd.append("--no_vtk")
    if args.no_coils:
        cmd.append("--no_coils")
    if args.no_fieldlines:
        cmd.append("--no_fieldlines")
    print("[examples/2_intermediate] postprocess:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(project_root), check=True)


if __name__ == "__main__":
    main()

