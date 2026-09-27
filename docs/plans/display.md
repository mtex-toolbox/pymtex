# The display of objects

MTEX's compact display is one of its strengths: a bold class name, the frames in parentheses,
a few aligned `key: value` lines, then the data as a table. The Python port has a display for
nearly every class (93 own, 95 inherited, 26 without: plans, solvers, figure handles), but the
classes grew their layouts one by one. This plan makes them one layout, MATLAB's, and improves
it where Python allows it.

Reference: the output of `frameSymmetry` for the same objects, probed on 2026-09-27
(`crystalFrame`, `specimenFrame`, `vector3d`, `Miller`, `rotation`, `orientation`,
misorientation, `fibre`, `slipSystem`, `stiffnessTensor`, `plottingConvention`, `SO3FunRBF`,
`SO3FunHarmonic`, `S2FunHarmonic`, `S2Grid`, `SO3Grid`, `sphericalRegion`,
`orientationRegion`, kernels, `quaternion`).

## What differs today

- The `key: value` lines are not aligned; MATLAB pads the keys to one width
  (`mineral : Forsterite`, `symmetry: mmm`).
- Sizes are written `5`, `3 x 3 x 3 x 3`; MATLAB writes `5 × 1`, `3 × 3 × 3 × 3`.
- The header names a frame inconsistently: an orientation shows its crystal frame by the
  mineral (`Iron → specimen`), a misorientation by the group (`m3̅m → m3̅m`), a crystal frame
  repeats its mineral in the header and in the first line. MATLAB names a crystal frame by the
  mineral, else the group, and the specimen frame by its plotting convention (`Iron → y↑→x`).
- `crystalFrame` shows `alpha, beta, gamma` whenever the lattice is not cubic, also for the
  hexagonal 90°, 90°, 120°; MATLAB only when they are not the lattice's defaults.
- `SO3FunRBF` does not list its centres and weights; MATLAB shows the table under a bold
  component heading. `orientationRegion` misses the mineral and the vertices, `S2FunHarmonic`
  its antipodal flag.
- A long array shows only its size, where MATLAB offers a link "show Euler angles".

## Decisions

- One layout module, `geometry/display.py`: `header(obj, frames)`, `fields(pairs)` (the aligned
  block), `sizeString` with `×`, `table` with head and tail rows. Every `__repr__` builds from it.
- Compact as today: no blank line between the header, the fields and the table (MATLAB's
  blank lines come from `disp(' ')`, not from design).
- Frames in the header: the crystal frame by `char()` (mineral, else group), the specimen frame
  by its convention, a named specimen frame by name and convention (`rolling, TD←RD↑`).
- A long array shows its first and last rows around `⋮`, as NumPy and the doc build do, which is
  what MATLAB's link gives in one click; the limit is a preference, `displayRows`.

## Improvements over MATLAB

- Jupyter: `_repr_html_` on the base classes, the class name bold, every frame in the header
  a `<details>` that opens the frame's own display, MATLAB's hyperlinks without a click back into
  the session, the table a real table.
- IPython terminal: `_repr_pretty_` prints the class name bold. `repr()` itself stays plain
  text, so logs, doctests and the doc build see no escape codes.
- The 26 classes without a display get a one-line one (`SO3FourierPlan (bandwidth 64, 1000
  nodes)`, `mtexFigure (2 × 2, ...)`).

## Steps

1. `display.py`: `fields`, `header`, `sizeString` with `×`, truncated `table`, the preference
   `displayRows`; `tests/test_display.py`.
2. The geometry classes onto it: frames, vectors and Miller, quaternion to orientation, fibre,
   slip systems, regions, grids; the frame labels made consistent.
3. The functions, kernels, tensors, pole figures onto it; the RBF centre table.
4. The maps: EBSD, grains, boundaries, and the maps3d classes.
5. The 26 without a display.
6. `_repr_html_` and `_repr_pretty_`.
7. The tests that pin display strings updated; the doc pages rebuilt, their outputs checked
   against MATLAB's once more.

## Status

Plan written 2026-09-27; steps 1 to 7 built the same day. `geometry/display.py` holds the layout
(`display`, `header`, `fields`, `aligned`, `frameLabel`, `framesLabel`, `table` with `takeRows`,
`textTable`, the preference `displayRows`, default 20); the displays that still build their own
lines pass through `aligned`. `plotting/richdisplay.py` gives IPython the HTML display (the name
bold, each frame of the header with its own display as a tooltip) and the terminal one (the name
bold); `repr()` is unchanged plain text. The tables of a notebook stay `pre` text rather than HTML
tables: the text tables are already aligned and read the same everywhere. S2Fun, SO3Fun and
grainBoundaryCriterion are abstract and keep no display.

Step 7: all 237 pages rebuilt, none with a new error. In 76 pages numbers moved; six of them,
built again on the commit before the display work, print the numbers they print now, so the moves
seen there come from other changes (the halfwidth selection, unseeded random draws), and the rest
were not rebuilt that way. The headers against MATLAB's diaries found
`triplePointList` and `shape2d` still naming the specimen frame `specimen`, fixed; the
differences that stay are deliberate and listed under the conventions of `docs/findings.md`.
