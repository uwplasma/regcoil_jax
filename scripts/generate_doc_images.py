#!/usr/bin/env python3
"""Generate documentation images under docs/_static/.

This script is intended for maintainers. The repository commits the resulting PNGs so that:
  - docs builds are fast and do not depend on running expensive examples, and
  - users see representative figures immediately.

Usage examples (from repo root):

  python scripts/generate_doc_images.py lambda_scan

  # Heavier (VMEC + winding-surface optimization + coil cutting + field lines):
  python scripts/generate_doc_images.py wsopt_3d --platform cpu --opt_steps 60
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np


def _require(pkg: str) -> None:
    try:
        __import__(pkg)
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Missing dependency {pkg!r}. Install with: pip install '.[docs,viz]'") from e


def _write_namelist_override(src: Path, dst: Path, overrides: dict[str, str]) -> None:
    txt = src.read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()
    out = []
    seen = set()
    for line in lines:
        stripped = line.strip()
        key = stripped.split("=", 1)[0].strip().lower() if "=" in stripped else None
        if key and key in overrides:
            out.append(f"  {key} = {overrides[key]}")
            seen.add(key)
        else:
            out.append(line)
    # Append missing overrides before the namelist terminator if possible.
    missing = [k for k in overrides.keys() if k not in seen]
    if missing:
        inserted = False
        out2 = []
        for line in out:
            if line.strip() == "/":
                for k in missing:
                    out2.append(f"  {k} = {overrides[k]}")
                inserted = True
            out2.append(line)
        out = out2 if inserted else out + [f"  {k} = {overrides[k]}" for k in missing]
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")


def generate_lambda_scan(*, repo_root: Path) -> Path:
    _require("netCDF4")
    import netCDF4  # noqa: F401
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from regcoil_jax.run import run_regcoil

    examples_dir = repo_root / "examples"
    src = examples_dir / "1_simple" / "regcoil_in.compareToMatlab1"
    if not src.exists():
        raise FileNotFoundError(src)

    with tempfile.TemporaryDirectory(prefix="regcoil_jax_docs_") as td:
        td = Path(td)
        inp = td / "regcoil_in.lambda_scan_docs"
        _write_namelist_override(
            src,
            inp,
            overrides={
                "general_option": "1",
                "nlambda": "80",
                "lambda_min": "1.0e-20",
                "lambda_max": "1.0e-10",
                "save_level": "2",
            },
        )
        res = run_regcoil(str(inp), verbose=False)

        import netCDF4

        ds = netCDF4.Dataset(res.output_nc, "r")
        try:
            lam = np.asarray(ds.variables["lambda"][:], dtype=float)
            chi2_B = np.asarray(ds.variables["chi2_B"][:], dtype=float)
            chi2_K = np.asarray(ds.variables["chi2_K"][:], dtype=float)
            max_B = np.asarray(ds.variables["max_Bnormal"][:], dtype=float)
            max_K = np.asarray(ds.variables["max_K"][:], dtype=float)
        finally:
            ds.close()

    # Plot a more informative “lambda scan” summary than a single line plot.
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    ax = axes[0, 0]
    lam_safe = np.asarray(lam, dtype=float)
    lam_safe = np.where(np.isfinite(lam_safe) & (lam_safe > 0.0), lam_safe, 1.0e-300)
    sc = ax.scatter(chi2_K, chi2_B, c=np.log10(lam_safe), s=18, cmap="viridis")
    ax.set_xlabel(r"$\chi^2_K$")
    ax.set_ylabel(r"$\chi^2_B$")
    ax.set_title("Tradeoff curve (each point is one λ)")
    fig.colorbar(sc, ax=ax, shrink=0.9, label=r"$\log_{10}(\lambda)$")

    ax = axes[0, 1]
    ax.semilogx(lam, chi2_B, "-", label=r"$\chi^2_B$")
    ax.semilogx(lam, chi2_K, "-", label=r"$\chi^2_K$")
    ax.set_xlabel(r"$\lambda$")
    ax.set_title(r"Objective terms vs $\lambda$")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1, 0]
    ax.semilogx(lam, max_B, "-")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\max |B_n|$")
    ax.set_title(r"Max normal field vs $\lambda$")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1, 1]
    ax.semilogx(lam, max_K, "-")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\max |K|$")
    ax.set_title(r"Max current density vs $\lambda$")
    ax.grid(True, which="both", alpha=0.25)

    out = repo_root / "docs" / "_static" / "lambda_scan_rich.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def generate_wsopt_3d(*, repo_root: Path, platform: str, opt_steps: int) -> Path:
    _require("netCDF4")
    import subprocess

    script = repo_root / "examples" / "3_advanced" / "compare_winding_surface_optimization_cut_coils_currents_poincare.py"
    if not script.exists():
        raise FileNotFoundError(script)

    # Use a temp output directory and then copy just the final summary image.
    with tempfile.TemporaryDirectory(prefix="regcoil_jax_wsopt_docs_") as td:
        out_dir = Path(td)
        cmd = [
            "python",
            str(script),
            "--platform",
            platform,
            "--out_dir",
            str(out_dir),
            "--opt_steps",
            str(int(opt_steps)),
            "--opt_ntheta",
            "48",
            "--opt_nzeta",
            "48",
            "--eval_ntheta",
            "56",
            "--eval_nzeta",
            "56",
            "--eval_nlambda",
            "6",
            "--current_opt_steps",
            "70",
            "--fieldline_steps",
            "450",
            "--n_starts",
            "6",
            "--poincare_max_points",
            "250",
        ]
        subprocess.check_call(cmd, cwd=str(repo_root))

        fig = out_dir / "figures" / "wsopt_3d_before_after.png"
        if not fig.exists():
            raise FileNotFoundError(f"Expected {fig} to exist (did the example write the 3D figure?)")

        out = repo_root / "docs" / "_static" / "wsopt_3d_before_after.png"
        out.write_bytes(fig.read_bytes())
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lambda_scan", help="Generate docs/_static/lambda_scan_rich.png")
    p.set_defaults(cmd="lambda_scan")

    p = sub.add_parser("wsopt_3d", help="Generate docs/_static/wsopt_3d_before_after.png (heavier)")
    p.add_argument("--platform", type=str, default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--opt_steps", type=int, default=60)
    p.set_defaults(cmd="wsopt_3d")

    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.cmd == "lambda_scan":
        out = generate_lambda_scan(repo_root=repo_root)
        print(f"[ok] wrote {out}")
        return

    if args.cmd == "wsopt_3d":
        out = generate_wsopt_3d(repo_root=repo_root, platform=args.platform, opt_steps=int(args.opt_steps))
        print(f"[ok] wrote {out}")
        return

    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
