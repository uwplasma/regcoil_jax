Usage
=====

Run the CLI on CPU::

  regcoil_jax --platform cpu --verbose examples/regcoil_in.lambda_search_1

Outputs
-------

For an input file named ``regcoil_in.<case>``, the CLI writes next to the input:

- ``regcoil_out.<case>.nc``
- ``regcoil_out.<case>.log``

Tests
-----

From the project root::

  pytest -q

