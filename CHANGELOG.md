# Changelog

## 0.1.0 (unreleased)

The first public release of MTEX for Python, a beta: the names follow MTEX for MATLAB, and the
interface may still change before 1.0.

* crystal geometry: vectors, rotations, point groups, crystal and specimen frames, Miller indices,
  orientations and misorientations, fundamental regions, grids, fibres, planes, slip systems,
  crystal shapes
* EBSD maps: import of `.ang`, `.ctf`, `.cpr`/`.crc`, `.osc`, tables and the vendors' HDF5 files, and
  export; grains and boundaries, their measures and shape parameters, intragranular quantities and
  `calcGND`, denoising and filling, parent grain reconstruction, pseudo symmetry cleaning, simulated
  maps, grid layouts and spatial transformations, TrueEBSD
* maps of volumes: `EBSD3`, grains in 3D, the DREAM.3D and Neper readers
* pole figures: import of the common formats, ODF reconstruction
* functions on the circle, the sphere and the orientation space: harmonic expansions, radial basis
  functions, density estimation with automatic halfwidths, Bingham distributions, texture
  characteristics, vector and axis fields
* tensors: elasticity, wave velocities, plasticity, the Taylor model, optics
* plots: pole figures, inverse pole figures, ODF sections, colour keys, EBSD and grain maps, 3D scenes
* `importWizard()`, a window for the import of EBSD data (experimental)
* `mtexdata` for the sample data sets of the joint data repository
