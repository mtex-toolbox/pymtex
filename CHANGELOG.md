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
alongside the Python implementation.

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
