from __future__ import annotations

import os
from pathlib import Path

import pytest


RUN_SLOW = os.environ.get("REGCOIL_JAX_RUN_SLOW", "").strip().lower() in ("1", "true", "yes", "on")


@pytest.mark.skipif(not RUN_SLOW, reason="Set REGCOIL_JAX_RUN_SLOW=1 to run slow example tests.")
def test_slow_examples_smoke(tmp_path: Path):
    """Smoke-run the most expensive example inputs.

    These are intentionally skipped by default to keep CI runtime reasonable.
    """
    from regcoil_jax.run import run_regcoil

    # Reuse the copy/self-consistency helpers from the parity test module.
    from .test_examples import _copy_example, _expected_output_paths, _assert_output_self_consistent

    cases = [
        # Paper-style / high-resolution examples:
        "3_advanced/regcoil_in.regcoilPaper_figure10d_but_geometry_option_coil_2_loRes",
        "3_advanced/regcoil_in.regcoilPaper_figure10d_but_geometry_option_coil_4_loRes",
        "3_advanced/regcoil_in.regcoilPaper_figure10d_originalAngle_loRes",
        # Very slow (128x128) examples:
        "3_advanced/regcoil_in.regcoilPaper_figure10d_but_geometry_option_coil_2",
        "3_advanced/regcoil_in.regcoilPaper_figure10d_originalAngle",
        "3_advanced/regcoil_in.regcoilPaper_figure10d_constArclengthAngle",
        "3_advanced/regcoil_in.regcoilPaper_figure3_NF4",
    ]

    for input_name in cases:
        input_path = _copy_example(tmp_path, input_name)
        out_nc, out_log = _expected_output_paths(tmp_path, input_path.name)

        run_regcoil(str(input_path), verbose=False)

        assert out_nc.exists(), f"missing {out_nc}"
        assert out_log.exists(), f"missing {out_log}"

        _assert_output_self_consistent(out_nc)

