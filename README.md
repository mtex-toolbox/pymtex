# MTEX for Python

MTEX is the toolbox for crystallographic texture analysis. This package is MTEX in Python,
with the names, the syntax and the numbers of the MATLAB toolbox:

* crystal geometry: point groups, crystal frames, Miller indices, orientations and misorientations
* EBSD maps: import from every major vendor format, grains, boundaries, intragranular quantities,
  denoising, parent grain reconstruction, maps of volumes
* pole figures: import, and ODF reconstruction from them
* functions on the sphere and on the orientation space: density estimation, harmonic expansions,
  texture characteristics
* tensors: elasticity, plasticity, the Taylor model, optics
* publication ready plots: pole figures, inverse pole figures, ODF sections, maps, 3D scenes

## Installation

```
pip install "mtex[all]"
```

The plain `pip install mtex` needs NumPy, SciPy and finufft only. The extras add the rest:
`numba` compiled kernels (much faster, strongly recommended), `plot` matplotlib, `io` the HDF5
formats, `plot3d` PyVista for the 3D scenes, `wizard` the import window; `all` is all of them.
MTEX runs on Python 3.12 or later on Linux, macOS and Windows.

## A first look

```python
from mtex import *

# an EBSD map of forsterite, enstatite and diopside
ebsd = mtexdata('forsterite')

# grains by a 10 degree threshold
grains = calcGrains(ebsd['indexed'], angle=10 * degree, minPixel=5)

# the orientation density of forsterite and its pole figures
ori = ebsd['Forsterite'].orientations
odf = calcDensity(ori, halfwidth=10 * degree)
plotPDF(odf, Miller([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ori.CS))
```

`mtexdata` downloads the sample data sets on first use from
[mtex-toolbox/mtex-data](https://github.com/mtex-toolbox/mtex-data).

## Documentation

The documentation is the [MTEX documentation](https://mtex-toolbox.github.io/documentation.html),
every page in MATLAB and in Python. Coming from MATLAB, the rules are few: a flag is a keyword
(`'antipodal'` becomes `antipodal=True`), indexing takes brackets (`ebsd('indexed')` becomes
`ebsd['indexed']`), and a method is called with parentheses (`rot.angle()`).

## Licence

MTEX is free software under the GNU General Public License, version 2; see `LICENSE`.
The MATLAB toolbox is at [mtex-toolbox/mtex](https://github.com/mtex-toolbox/mtex).
