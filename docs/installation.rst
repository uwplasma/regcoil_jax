Installation
============

Quick install (recommended for most users)
------------------------------------------

Clone the repository, then install it with pip from the repo root::

  git clone https://github.com/uwplasma/regcoil_jax.git
  cd regcoil_jax
  pip install .

This installs the `regcoil_jax` package and the `regcoil_jax` CLI entrypoint.

Optional extras
---------------

For plotting / ParaView (VTK) postprocessing examples::

  pip install '.[viz]'

For building the docs locally::

  pip install '.[docs]'
  sphinx-build -b html docs docs/_build/html

Development install
-------------------

Editable install (recommended for development)::

  pip install -e '.[dev]'
