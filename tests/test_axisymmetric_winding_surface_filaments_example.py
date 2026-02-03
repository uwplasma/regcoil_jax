from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]


def test_axisymmetric_winding_surface_filament_opt_smoke(tmp_path: Path):
    """Smoke-run the axisymmetric winding-surface filament optimization example.

    This is intentionally tiny (`--fast`) so it can run in CI. The goal is to
    validate that:
      - the end-to-end autodiff optimization runs without error,
      - the script writes the expected figure/VTK outputs,
      - the normal-field objective improves at least modestly.
    """
    ex = PROJECT_ROOT / "examples" / "3_advanced" / "axisymmetric_winding_surface_filament_optimization.py"
    assert ex.exists()

    out_dir = tmp_path / "out"

    env = os.environ.copy()
    env.setdefault("JAX_TRACEBACK_FILTERING", "off")
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("JAX_ENABLE_X64", "True")
    env.setdefault("MPLBACKEND", "Agg")

    proc = subprocess.run(
        [
            sys.executable,
            str(ex),
            "--fast",
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, f"example failed.\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"

    # Expected outputs
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "figures" / "loss_history.png").exists()
    assert (out_dir / "figures" / "bn_before_after.png").exists()
    assert (out_dir / "figures" / "coils_3d.png").exists()
    assert (out_dir / "vtk" / "winding_surface_full.vts").exists()
    assert (out_dir / "vtk" / "plasma_full.vts").exists()
    assert (out_dir / "vtk" / "coils_before.vtp").exists()
    assert (out_dir / "vtk" / "coils_after.vtp").exists()

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    chi2_0 = float(summary["metrics_before"]["chi2_B"])
    chi2_1 = float(summary["metrics_after"]["chi2_B"])
    assert chi2_0 > 0.0
    assert chi2_1 > 0.0
    # Loose but meaningful: should improve.
    assert chi2_1 < chi2_0, f"expected improvement: chi2 before={chi2_0} after={chi2_1}"

