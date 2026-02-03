# simsopt VMEC reference cases (small)

This folder contains a **small, curated subset** of VMEC `wout_*.nc` files copied from
`simsopt/tests/test_files` in the local workspace.

These files are intentionally small (order ~10–100 kB) so they can be used for
examples and unit tests without bloating the repository.

Files:
- `wout_circular_tokamak_reference.nc` (axisymmetric)
- `wout_purely_toroidal_field_reference.nc` (simple non-stellarator sanity case)
- `wout_li383_low_res_reference.nc` (low-resolution stellarator-like case)

If you want to add more cases, prefer low-resolution (small) `wout` files and
update the tests to keep runtime reasonable.

