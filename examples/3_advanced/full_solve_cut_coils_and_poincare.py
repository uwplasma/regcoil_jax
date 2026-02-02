#!/usr/bin/env python3
"""One-command “full workflow” demo: run REGCOIL_JAX → cut coils → field lines → Poincaré → VTK/figures.

This is a convenience wrapper around `postprocess_make_figures_and_vtk.py` that enables
Poincaré outputs by default.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    default_input = here / "regcoil_in.lambda_search_1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(default_input))
    parser.add_argument("--platform", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--no_figures", action="store_true")
    parser.add_argument("--no_vtk", action="store_true")
    parser.add_argument("--no_fieldlines", action="store_true")
    parser.add_argument("--poincare_phi0", type=float, default=0.0)
    args = parser.parse_args()

    post = here / "postprocess_make_figures_and_vtk.py"
    cmd = [
        sys.executable,
        str(post),
        "--run",
        "--platform",
        args.platform,
        "--input",
        str(Path(args.input).resolve()),
        "--poincare",
        "--poincare_phi0",
        str(float(args.poincare_phi0)),
    ]
    if args.no_figures:
        cmd.append("--no_figures")
    if args.no_vtk:
        cmd.append("--no_vtk")
    if args.no_fieldlines:
        cmd.append("--no_fieldlines")

    project_root = here.parents[1]
    print("[full demo] running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(project_root), check=True)


if __name__ == "__main__":
    main()

