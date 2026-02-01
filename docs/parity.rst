Parity with Fortran REGCOIL
==========================

The current parity target is the scalar diagnostic arrays:

- ``lambda``
- ``chi2_B``
- ``chi2_K``
- ``max_Bnormal``
- ``max_K``

See ``PORTING_NOTES.md`` in the repository root for:

- what was broken and how it was fixed
- how to run the reference Fortran code locally
- how to compare outputs

The parity test suite includes both:
- analytic-geometry examples (fast)
- VMEC-based lambda-search examples, including a `load_bnorm=.true.` BNORM case
