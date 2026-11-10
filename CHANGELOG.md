# MTEX for Python release notes

## 1.0.0 (unreleased)

### MTEX comes to Python

MTEX 1.0 brings crystallographic texture analysis to Python scripts and Jupyter
notebooks, with the familiar objects and function names of MTEX for MATLAB. It runs
without a MATLAB installation. Where a MATLAB name was a misnomer or a workaround, Python
uses a clearer one, such as `plotPF` for `plotPDF` and `thinning=False` for `'nothinning'`;
[MTEX in MATLAB and in Python](https://mtex-toolbox.github.io/MATLABtoPython_py.html)
lists them.

This first stable release has been cross-checked against the MATLAB toolbox and the
Julia port. The comparisons also uncovered bugs in MATLAB, improving the toolbox
alongside the Python implementation, and they caught defects of the port itself before
the release, such as a 30 degree turn of hexagonal maps converted between `.ctf` and
`.ang`.

### Pores, pockets, cracks and voids

`calcGrains` sorts the notIndexed regions wider than the closing into kinds and closes each
kind with a radius of its own: `alphaPore`, `alphaPocket` and `alphaCrack` replace `alpha` on
the cells of their kind. By default a pore, a region enclosed by one grain, is filled whole and
becomes part of that grain, a crack stays open where `alpha` would have bridged it, and a
pocket between several grains closes with `alpha` as before, so `calcGrains(ebsd, alphaCrack=15)`
bridges the cracks up to 30 pixels wide and `alphaPore=0` keeps the pores as notIndexed grains.
`notIndexedKinds(ebsd)` returns the kind of every pixel and a table of the regions, and
`calcGrains(ebsd, kinds=kinds)` takes the classification back, edited or not. The reach of the
cells across a filled region is measured after the closing, so a crack bridged whole tiles, and
a grain whose boundary does not close is named in an error. `calcGrains(ebsd, delaunay=True)`
fills the gaps by the alpha complex of the indexed pixels instead of the closing with a disk,
exact where the closing is a raster approximation, with a circumradius bound of 2.2 by
default. The port follows MTEX's `calcGrains` and is checked against it on the peridotite
map, grain by grain and pixel by pixel (`bench/calcgrains_ops.py`); the map,
`mtexdata('peridotite')`, is new, and the page [Cracks and Pores](https://mtex-toolbox.github.io/GrainReconstructionCracks_py.html)
shows the kinds on it.

### Faster grain reconstruction

Grain reconstruction runs substantially faster in our benchmarks against MATLAB,
reducing the wait when working with large EBSD maps and trying different reconstruction
parameters. The accelerated routines are included when installing with `mtex[all]`.

### A familiar working environment

VS Code with the Python and Jupyter extensions provides an editor, an interactive
command window, a variable explorer and plots. Create a Python environment (3.12 or
later) in VS Code, then install MTEX and the notebook kernel in its terminal:

```sh
pip install "mtex[all]" ipykernel
```

Run script sections marked with `# %%` using Shift+Enter, much like sections in MATLAB.
See [Working as in MATLAB: VS Code](README.md#working-as-in-matlab-vs-code) for the desktop
equivalents, or [Installing MTEX for Python](https://mtex-toolbox.github.io/Installation_py.html)
for setup on Windows, macOS and Linux.

### Your MTEX workflow in Python

Loading data, reconstructing grains and plotting results follow the familiar MTEX
workflow. Import the commands once at the start of a script or notebook:

```python
from mtex import *

ebsd = mtexdata('forsterite')
grains = calcGrains(ebsd, angle=15 * degree, minPixel=5)
plot(grains)
```

The new syntax writes `grainId` directly into `ebsd`. Phase selections use brackets,
such as `ebsd['Forsterite']`, and options are written as keywords, such as `antipodal=True`.

See [MTEX in MATLAB and in Python](https://mtex-toolbox.github.io/MATLABtoPython_py.html)
for translating an existing analysis, including indexing and default settings.

To import your own measurements, `importWizard()` opens a window where you can inspect
the phases, reference frame and map. It generates the Python import script so that you
can repeat the import in your analysis.

### Tutorials that become your own scripts

The documentation is included in the package. `openDoc` takes the place of opening an
MTEX example in the MATLAB editor, with a choice of Python script or Jupyter notebook:

```python
openDoc('EBSDTutorial')
openDoc('EBSDTutorial', notebook=True)
```

Run the tutorial section by section and adapt it to your measurements. In a notebook,
the explanations, commands and resulting figures stay together in one document.
Tutorial text is available offline; `mtexdata` downloads the sample data on first use.

MTEX is free software under the GNU General Public License, version 2.
