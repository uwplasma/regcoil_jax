from __future__ import annotations

import os
import sys

project = "regcoil_jax"
copyright = "2026"
author = "regcoil_jax contributors"

# Make the package importable for autodoc.
sys.path.insert(0, os.path.abspath(".."))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

autosummary_generate = True

try:
    import sphinx_rtd_theme  # noqa: F401
    html_theme = "sphinx_rtd_theme"
except Exception:  # pragma: no cover
    # Allow docs to build in minimal environments; CI installs the docs extra.
    html_theme = "alabaster"
