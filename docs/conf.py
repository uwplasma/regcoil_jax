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
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

autosummary_generate = True

# External links (used throughout the docs to link to source files and references).
extlinks = {
    # Link to the GitHub repo at the pinned main branch.
    # Usage:
    #   :src:`regcoil_jax/run.py`
    #   :ex:`examples/3_advanced/regcoil_in.lambda_search_1`
    "src": ("https://github.com/uwplasma/regcoil_jax/blob/main/%s", "%s"),
    "ex": ("https://github.com/uwplasma/regcoil_jax/blob/main/%s", "%s"),
    "tree": ("https://github.com/uwplasma/regcoil_jax/tree/main/%s", "%s"),
    "doi": ("https://doi.org/%s", "doi:%s"),
    "arxiv": ("https://arxiv.org/abs/%s", "arXiv:%s"),
}

# Linkcheck configuration (used by `python -m sphinx -b linkcheck ...`).
# Many academic publishers enforce bot protection that returns 403 to automated checkers.
# We still keep the links clickable for real users in the rendered docs.
linkcheck_allowed_redirects = {
    # DOIs routinely redirect through publisher landing pages; this is expected.
    r"https://doi\.org/.*": r".*",
}

# APS blocks automated checks from some environments (403), but these links work in browsers.
linkcheck_ignore = [
    r"https://doi\.org/10\.1103/PhysRevLett\.124\.095001",
]

try:
    import sphinx_rtd_theme  # noqa: F401
    html_theme = "sphinx_rtd_theme"
except Exception:  # pragma: no cover
    # Allow docs to build in minimal environments; CI installs the docs extra.
    html_theme = "alabaster"
