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
plotPF(odf, Miller([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ori.CS))
saveFigure('pf.png')
```

In a notebook, in IPython or in the interactive window of VS Code the pole figures appear as
they are drawn. A script run with `python first.py` ends before a figure could be seen, which
is why the last line writes them to `pf.png`.

`mtexdata` downloads the sample data sets on first use from
[mtex-toolbox/mtex-data](https://github.com/mtex-toolbox/mtex-data).

## Working as in MATLAB: VS Code

MATLAB is one program; in Python the editor is separate. [VS Code](https://code.visualstudio.com/)
with two extensions plays the part of the MATLAB desktop:

1. Install VS Code and, from its *Extensions* panel (Ctrl+Shift+X), the **Python** and the
   **Jupyter** extensions published by Microsoft.
2. Open your project folder (*File > Open Folder...*), press Ctrl+Shift+P, choose
   **Python: Create Environment**, then *Venv* and a Python 3.12 or later. This makes a folder
   `.venv`, the counterpart of MTEX on the MATLAB path, but per project.
3. In *Terminal > New Terminal* (its prompt starts with `(.venv)`) type
   `pip install "mtex[all]" ipykernel`.

Then the pieces of the MATLAB desktop are:

| MATLAB | VS Code |
|---|---|
| a section `%%` in the editor | a cell `# %%` in a `.py` file; **Shift+Enter** runs it and moves on, **Ctrl+Enter** runs it again |
| the command window | the input box at the bottom of the *Interactive* window (Shift+Enter runs), or Ctrl+Shift+P, **Jupyter: Create Interactive Window** |
| the workspace | *Variables* in the toolbar of the interactive window |
| `clear all` | *Restart* in the toolbar of the interactive window |
| figure windows | figures inline in the interactive window; run `%matplotlib qt` once for separate windows that zoom and turn |

```python
# %%
from mtex import *

# %%
ebsd = mtexdata('forsterite')
ebsd

# %%
plot(ebsd)
```

The first time, VS Code asks for a kernel: choose the one named `.venv`. The page
[Installing MTEX for Python](https://mtex-toolbox.github.io/Installation_py.html) goes through
every step for Windows, macOS and Linux, with conda and uv as alternatives and what to do when
something goes wrong.

## Documentation

The documentation is the [MTEX documentation](https://mtex-toolbox.github.io/documentation.html),
every page in MATLAB and in Python, switched by the *MATLAB | Python* button at the top of a
page; the Python version of a page is `<Page>_py.html`. Coming from MATLAB, read
[MTEX in MATLAB and in Python](https://mtex-toolbox.github.io/MATLABtoPython_py.html) first. The
rules are few: a flag is a keyword (`'antipodal'` becomes `antipodal=True`), indexing takes
brackets (`ebsd('indexed')` becomes `ebsd['indexed']`), and a method is called with
parentheses (`rot.angle()`). `help(calcGrains)`, or hovering over a name in VS Code, shows the
help of a function.

Every page is a script, and the pages come with the package, so they also work offline:

```python
openDoc()                                  # every page, by chapter
openDoc('Tutorials')                       # the pages of one chapter
openDoc('EBSDTutorial')                    # the page as EBSDTutorial.py, opened in VS Code
openDoc('EBSDTutorial', notebook=True)     # the page as the Jupyter notebook EBSDTutorial.ipynb
```

The script has a `# %%` cell for every section, the text of the page as `# %% [markdown]`
cells; run it with Shift+Enter cell by cell and the interactive window rebuilds the page. The
notebook opens in VS Code, JupyterLab or Jupyter Notebook with the results below each cell.
The file is written into the current folder (or `folder=`) and never overwritten, so your
changes are safe; `overwrite=True` brings back the original. The name of a page is the last
part of its web address: `.../EBSDTutorial_py.html` is `EBSDTutorial`. Every page on the
website also has a *download as script* link.

## The import wizard

The import wizard opens EBSD files in a window, shows the phases, the reference frame and the
map, and writes the Python script that imports them (it needs the `wizard` extra, which
`mtex[all]` includes). Start it from Python

```python
importWizard()                   # browse the current folder
importWizard('data/ebsd')        # a folder, or a file to open at once
ebsd = importWizard()            # the map it imported, when the window is closed
```

or from a terminal with the environment active: `mtex-wizard [folder or file]`, or
`python -m mtex.wizard`. See [EBSD Import](https://mtex-toolbox.github.io/EBSDImport_py.html).

## Settings

A preference holds for the session; `save=True` keeps it for every later one, stored in
`mtex_settings.json` in your configuration folder (`~/.config/mtex` on Linux and macOS,
`%APPDATA%\mtex` on Windows), MATLAB's `mtex_settings.m`:

```python
plottingConvention.default('y↑→x', save=True)   # maps and pole figures drawn with y up
setMTEXpref('displayRows', 40, save=True)        # rows a long array shows
settingsFile()                                   # where the file is
```

A fresh installation has no file and draws with y down (`y↓→x`). The environment variable
`MTEX_SETTINGS` names another file, or with an empty value switches the file off.

## Licence

MTEX is free software under the GNU General Public License, version 2; see `LICENSE`.
The MATLAB toolbox is at [mtex-toolbox/mtex](https://github.com/mtex-toolbox/mtex).
