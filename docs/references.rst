References and further reading
==============================

Core methods
------------

- Landreman, M. (2017). *An improved current potential method for fast computation of stellarator coil shapes* (REGCOIL).
  Nuclear Fusion **57** 046003. DOI: 10.1088/1741-4326/aa57d4
- Merkel, P. (1987). *Solution of Stellarator Boundary Value Problems with External Currents* (NESCOIL).
  Nuclear Fusion **27** 867. DOI: 10.1088/0029-5515/27/5/018

Quadratic-constraint coil optimization
--------------------------------------

- Hergt, J. M. J., Paul, E. J., Hudson, S. R., et al. (2024/2025).
  *Global stellarator coil optimization with quadratic constraints and objectives* (“quadcoil”).
  Nuclear Fusion **65** 016006. DOI: 10.1088/1741-4326/ada810 (preprint: arXiv:2408.08267)
- Rutkowski, H., Hergt, J. M. J., Hudson, S. R., et al. (2023).
  *Coil metric constraints for stellarator coil design*.
  Nuclear Fusion **63** 016017. DOI: 10.1088/1741-4326/aca98d

Permanent magnets / dipole coillets
-----------------------------------

- Helander, P., Drevlak, M., Feng, Y., et al. (2020).
  *Stellarator construction with permanent magnets*.
  Physical Review Letters **124** 095001. DOI: 10.1103/PhysRevLett.124.095001
- Drevlak, M., Helander, P., Hennig, T., et al. (2024).
  *Optimization of permanent magnets for stellarators*.
  Computer Physics Communications **300** 109127. DOI: 10.1016/j.cpc.2024.109127

Differentiable geometry extraction (general context)
----------------------------------------------------

These are not required for typical regcoil_jax workflows, but are useful background for differentiable
contour/isosurface extraction ideas:

- Remelli et al. (2020). *MeshSDF: Differentiable Iso-Surface Extraction*. NeurIPS 2020.
- Liao et al. (2018). *Deep Marching Cubes: Learning Explicit Surface Representations*. CVPR 2018.
- Shen et al. (2021). *Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis* (DMTet). NeurIPS 2021.
- Shen et al. (2023). *Flexible Isosurface Extraction for Gradient-Based Mesh Optimization* (FlexiCubes). SIGGRAPH 2023.

Differentiable relaxations of discrete operators
------------------------------------------------

- Grover et al. (2019). *Stochastic Optimization of Sorting Networks via Continuous Relaxations* (NeuralSort). arXiv:1903.08850.
