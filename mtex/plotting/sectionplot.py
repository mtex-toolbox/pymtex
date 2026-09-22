# the matplotlib drawing of ODF sections: a grid of charts, one cell per section (MATLAB
# @SO3Fun/plotSection, @orientation/plotSection, @ODFSections/plot; docs/plans/plotting.md S2)
#
# A section object hands out charts and the orientations at their points; this file lays
# the charts out, evaluates a function or places a list of orientations on them, and draws
# with the machinery of sphericalplot.py.
from __future__ import annotations

import numpy as np

from ..geometry.crystalshape import crystalShape
from ..geometry.orientation import orientation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .mtexfigure import gcm, mtexFigure
from .sections import ODFSections, SECTION_OPTIONS, newODFSectionPlot
from .sphericalplot import (FILL_FLAGS, _arrows, _filled, _markers, _rangeOf, _split, sphericalPlot)

FIELD_COLOR = (0.72, 0.72, 0.72)
OUTER_COLOR = (0.75, 0.75, 0.75)


def _isSO3Fun(f):
  return hasattr(f, "pdf") and hasattr(f, "CS") and hasattr(f, "SS") and not isinstance(f, orientation)


# one cell per section, in a grid of the given width; the axis angle sections rescale
# theirs, so that equal areas stand for equal orientation volumes
def _sectionAxes(oS, columns, axesKw, whereKw, field=True, onto=False):
  n = oS.numSections
  charts = [oS.sectionCharts(k) for k in range(n)]
  wide = max(len(c) for c in charts)
  # a cell is as tall as its chart is, so a row of half discs leaves no band of white
  aspect = max((lambda b: (b[3] - b[2]) / (b[1] - b[0]))(c[0].bounds()) for c in charts)
  # layout=[rows, cols] names the grid, as MATLAB's option does
  layout = whereKw.get("layout")
  ncols = (int(layout[1]) if layout is not None else columns if columns else _gridColumns(n, aspect)) * wide
  nrows = -(-n * wide // ncols)
  # a held figure of the same sections takes the plot on top of what it has
  held = gcm()
  if held is not None and (held.hold or onto) and len(held.children) == n * wide:
    axes = iter(held.children)
    plots = []
    for k in range(n):
      plots.append([next(axes)._mtexSphericalPlot for _ in charts[k]])
      for _ in range(wide - len(charts[k])):
        next(axes)
    return held, plots
  from .mtexfigure import figureSize
  figSize = figureSize(whereKw.get("figSize"), (min(3.2 * ncols, 16), min(3.2 * aspect * nrows + 0.4 * nrows, 12)))
  fig = mtexFigure(n * wide, columns=ncols, figSize=figSize)
  fig.sections = oS
  labels = oS.axisLabels
  outer = oS.outerSector
  noLabel = axesKw.pop("noLabel", labels is not None or outer is not None)
  plots = []
  for k in range(n):
    cell = []
    for j, proj in enumerate(charts[k]):
      ax = fig.nextAxis()
      sP = sphericalPlot(ax, proj, noLabel=noLabel, title=oS.sectionLabel(k) if j == 0 else None, **axesKw)
      if labels is not None:
        ax.set_xlabel(labels[0], fontsize=sP.fontSize)
        ax.set_ylabel(labels[1], fontsize=sP.fontSize)
      # the sector every section lies in, so that the rescaled ones have a reference
      if outer is not None:
        for x, y in proj.withRegion(outer).boundary():
          ax.plot(x, y, color=OUTER_COLOR, linewidth=1.0, zorder=4)
      # and the field the section counts its coordinate from
      if field:
        f = oS.sectionField(k)
        if f is not None:
          _arrows(sP, f[0], f[1], resolution=15 * degree, arrowSize=0.55, color=FIELD_COLOR, zorder=3)
      _scaleLimits(ax, oS.sectionScale(k))
      cell.append(sP)
    for _ in range(wide - len(cell)):
      fig.nextAxis().set_axis_off()
    plots.append(cell)
  return fig, plots


# the width of the grid: the divisor of the count nearest to a slightly wide picture, so
# that no cell of the last row is left empty
# the columns of a grid of n cells of the given height to width ratio: the figure about
# 1.6 times as wide as tall, so wide Euler sections stack in two columns as MATLAB's do
def _gridColumns(n, aspect=1.0):
  if n <= 3 and aspect >= 0.5:
    return n
  want = np.sqrt(1.6 * n * min(aspect, 1.0))
  divisors = [c for c in range(2, min(n - 1, 8) + 1) if n % c == 0]
  return int(np.ceil(np.sqrt(n))) if not divisors else min(divisors, key=lambda c: abs(c - want))


# the same picture seen from further away, keeping its centre
def _scaleLimits(ax, s):
  if abs(s - 1) < 1e-9:
    return
  xlo, xhi = ax.get_xlim()
  ylo, yhi = ax.get_ylim()
  cx, cy, w, h = (xlo + xhi) / 2, (ylo + yhi) / 2, s * (xhi - xlo) / 2, s * (yhi - ylo) / 2
  ax.set_xlim(cx - w, cx + w)
  ax.set_ylim(cy - h, cy + h)


# every equivalent of every orientation is a marker, so a long list is thinned first,
# deterministically (MATLAB 'points', 'all')
def _thinned(ori, values, points, all_):
  n = ori.size
  if all_ or n <= points:
    return ori.flatten(), None if values is None else np.asarray(values).ravel()
  print(f"plot {points} random orientations out of {n} given orientations")
  idx = np.unique(np.round(np.linspace(0, n - 1, points)).astype(int))
  return ori.flatten()[idx], None if values is None else np.asarray(values).ravel()[idx]


def plotSection(x, *args, **kw):
  """plotSection(odf), plotSection(odf, 'sigma'), plotSection(ori, oS, sections=9)

  An SO3Fun or a list of orientations sliced into sections. Without a section object the
  default is the phi2 sections against a specimen frame and the axis angle sections between
  two crystals, as in MTEX; a string picks another family ('phi1', 'phi2', 'alpha',
  'gamma', 'sigma', 'pf', 'axisAngle') and the keywords of the family name its sections.
  """
  # a function on the sphere along a circle: the polar section
  from ..geometry.vector3d import vector3d
  if args and isinstance(args[0], vector3d) and not hasattr(x, "frameA"):
    from .polarplot import plotSectionS2
    return plotSectionS2(x, *args, **kw)
  kw.pop("silent", None)
  kind, shapes = None, None
  for i, a in enumerate(args):
    if isinstance(a, (str, ODFSections)):
      kind = a
    elif i > 0 and isinstance(args[i - 1], str) and args[i - 1].lower() in ("phi1", "phi2", "alpha", "gamma", "sigma", "omega", "axisangle"):
      # plotSection(ori, 'axisAngle', angles): the numbers after a family are its sections
      kw["angles" if args[i - 1].lower() == "axisangle" else args[i - 1]] = a
    elif isinstance(a, crystalShape):
      # plotSection(ori, ori * cS, oS): a crystal shape per orientation drawn at its place
      shapes = a
    else:
      # plotSection(ori, values, 'sigma'): a value per orientation colours the markers
      kw["values"] = a
  if isinstance(kind, ODFSections):
    oS = kind
  else:
    CS1, CS2 = (x.CS, x.SS)
    # misorientations with grain exchange are drawn on the antipodal sections
    if getattr(x, "antipodal", False) and CS1 is CS2:
      kw.setdefault("antipodal", True)
    oS = newODFSectionPlot(CS1, CS2, kind, **{k: kw[k] for k in SECTION_OPTIONS if k in kw})
  for k in SECTION_OPTIONS:
    kw.pop(k, None)
  # MATLAB's coordinates='off' hides the ticks of the Euler sections, xlabel and ylabel name the axes
  decor = {k: kw.pop(k) for k in ("coordinates", "xlabel", "ylabel") if k in kw}
  if _isSO3Fun(x):
    hs = _plotSectionFun(x, oS, **kw)
  elif shapes is not None:
    hs = _plotSectionShapes(x, shapes, oS, **kw)
  else:
    hs = _plotSectionOri(x, oS, **kw)
  if decor:
    _decorate(gcm(), decor)
  return hs


def _decorate(fig, decor):
  for ax in fig.children:
    if str(decor.get("coordinates", "on")).lower() == "off":
      ax.set_xticks([])
      ax.set_yticks([])
    if "xlabel" in decor:
      ax.set_xlabel(decor["xlabel"])
    if "ylabel" in decor:
      ax.set_ylabel(decor["ylabel"])


# the sections of the current figure, which annotate, quiver and the fibre draw into
def _currentSections():
  fig = gcm()
  oS = getattr(fig, "sections", None)
  if oS is None:
    raise ValueError("no section plot to draw into: plotSection first")
  return oS


def plotSections(oS, columns=0, field=True, **kw):
  """plot(oS): the charts of a section plot with the reference field of every section"""
  projKw, axesKw, whereKw = _split(kw)
  fig, plots = _sectionAxes(oS, columns, axesKw, whereKw, field=field)
  return fig


def annotateSection(ori, *args, **kw):
  """annotate(ori) on a section plot: every orientation marked in the section it lies in"""
  oS = _currentSections()
  kw.pop("add2all", None)
  values = next((a for a in args if not isinstance(a, str)), None)
  kw.setdefault("MarkerSize", 8)
  return _plotSectionOri(ori, oS, values=values, all=True, onto=True, **kw)


def plotFibreSection(f, points=1000, lineColor="k", **kw):
  """plot(f) on a section plot: the fibre as a chain of dots through the sections"""
  oS = _currentSections()
  kw.pop("add2all", None)
  for k in ("lineWidth", "LineWidth", "linewidth"):
    kw.pop(k, None)
  ori = f.flatten().orientation(points) if hasattr(f, "flatten") else f.orientation(points)
  kw.setdefault("MarkerSize", 1.5)
  kw.setdefault("MarkerFaceColor", lineColor)
  kw.setdefault("MarkerEdgeColor", lineColor)
  return _plotSectionOri(ori, oS, all=True, onto=True, **kw)


def quiverSection(ori, w, color="k", lineWidth=1.0, resolution=10 * degree, **kw):
  """quiver(ori, w) on a section plot: an arrow along w at every orientation, w a specimen
  direction per orientation"""
  oS = _currentSections()
  projKw, axesKw, whereKw = _split(kw)
  lineWidth = kw.pop("lineWidth", lineWidth)
  fig, plots = _sectionAxes(oS, 0, axesKw, whereKw, onto=True)
  vs, sec, src = oS.sectionData(ori)
  w = w if isinstance(w, vector3d) else vector3d(w)
  ws = vector3d(np.broadcast_to(w.data.reshape(-1, 3), (ori.size, 3)) if w.size == 1 else w.data.reshape(-1, 3))
  hs = []
  for k, cell in enumerate(plots):
    ind = np.flatnonzero(sec == k)
    if len(ind) == 0:
      continue
    for sP in cell:
      hs.append(_arrows(sP, vs[ind], ws[src[ind]], resolution=resolution, normalize=True, color=color, lineWidth=lineWidth, **kw))
  return hs


# the crystal shapes seen along the screen normal, as polygons in the chart about the
# position of their orientation; the shape list has one shape per orientation
def _plotSectionShapes(ori, cS, oS, faceColor=None, faceAlpha=1.0, edgeColor="k", lineWidth=0.5, colored=None, colormap="viridis",
                       columns=0, field=True, **kw):
  from .shapeplot import faceColors, _draw
  from .space import visibleFaces
  projKw, axesKw, whereKw = _split(kw)
  fig, plots = _sectionAxes(oS, columns, axesKw, whereKw, field=field)
  vs, sec, src = oS.sectionData(ori.flatten())
  if len(cS) != ori.size:
    raise ValueError("one crystal shape per orientation: plotSection(ori, ori * cS, oS)")
  pC = getattr(oS, "pC", None)
  out = pC.outOfScreen.data if pC is not None else np.array([0.0, 0.0, 1.0])
  east = pC.east.data if pC is not None else np.array([1.0, 0.0, 0.0])
  north = np.cross(out, east)
  V = cS.V.data.reshape(len(cS), cS.numVertices, 3)
  P = np.stack([V @ east, V @ north], axis=-1)
  vis = visibleFaces(cS, out).reshape(len(cS), cS.numFaces)
  colors = faceColors(cS, faceColor, colored, colormap)
  hs = []
  for k, cell in enumerate(plots):
    ind = np.flatnonzero(sec == k)
    if len(ind) == 0:
      continue
    for sP in cell:
      x0, y0 = sP.proj.project(vs[ind])
      x0, y0 = np.asarray(x0, dtype=float).ravel(), np.asarray(y0, dtype=float).ravel()
      polys, cols = [], []
      for j, (x, y) in enumerate(zip(x0, y0)):
        if not np.isfinite(x):
          continue
        i = src[ind[j]]
        for fi in np.flatnonzero(vis[i]):
          f = cS.F[fi]
          polys.append(P[i, f[f >= 0]] + [x, y])
          cols.append(colors[i, fi])
      if polys:
        hs.append(_draw("chart", sP, polys, np.asarray(cols), edgeColor, lineWidth, faceAlpha, None))
  return hs


def _plotSectionFun(f, oS, resolution=2.5 * degree, columns=0, colorbar=False, colorRange="equal", minmax=False,
                    field=True, **kw):
  flag = next((s for s in FILL_FLAGS if kw.pop(s, False)), None)
  style = "pcolor" if flag in (None, "smooth") else flag
  projKw, axesKw, whereKw = _split(kw)
  fig, plots = _sectionAxes(oS, columns, axesKw, whereKw, field=field)
  grids = [[sP.proj.grid(resolution) for sP in cell] for cell in plots]
  values = []
  for k, cell in enumerate(grids):
    vals = []
    for V, X, Y in cell:
      vals.append(np.real(np.asarray(f(oS.orientationsAt(k, V.flatten())), dtype=complex)).reshape(X.shape) if V.size else np.zeros((0, 0)))
    values.append(vals)
  if isinstance(colorRange, str) and colorRange == "equal":
    colorRange = _rangeOf([v for cell in values for v in cell])
  hs = []
  for cell, gs, vs in zip(plots, grids, values):
    for sP, (_, X, Y), v in zip(cell, gs, vs):
      hs.append(_filled(sP, X, Y, v, style=style, colorRange=colorRange, **kw) if v.size else None)
      if minmax:
        sP.annotateMinMax()
  fig.colorRange = tuple(colorRange)
  if colorbar:
    fig.colorbar()
  return hs


def _plotSectionOri(ori, oS, columns=0, values=None, colorbar=False, colorRange=None, points=2000, all=False,
                    field=True, MarkerSize=4, onto=False, add2all=False, **kw):
  projKw, axesKw, whereKw = _split(kw)
  ori, values = _thinned(ori, values, points, all)
  # add2all draws into the sections already there, as MATLAB's flag does
  fig, plots = _sectionAxes(oS, columns, axesKw, whereKw, field=field, onto=onto or add2all)
  vs, sec, src = oS.sectionData(ori)
  if values is not None and colorRange is None and len(values):
    colorRange = (float(np.nanmin(values)), float(np.nanmax(values)))
  hs = []
  for k, cell in enumerate(plots):
    ind = np.flatnonzero(sec == k)
    if len(ind) == 0:
      continue
    dirs = vs[ind]
    vals = None if values is None else values[src[ind]]
    for sP in cell:
      hs.append(_markers(sP, dirs, vals, colorRange=colorRange, MarkerSize=MarkerSize, **kw))
  if colorbar and values is not None:
    fig.colorbar()
  return hs


__all__ = ["plotKeySections", "plotSection", "plotSections", "annotateSection", "plotFibreSection", "quiverSection"]


# plot(key), plot(key, 'sigma', sections=6): the colours of an orientation colour key over
# the sections of orientation space, MATLAB @orientationColorKey/plot
def plotKeySections(key, *args, resolution=2.5 * degree, columns=0, field=False, **kw):
  kind = next((a for a in args if isinstance(a, (str, ODFSections))), None)
  oS = kind if isinstance(kind, ODFSections) else newODFSectionPlot(key.CS1, key.CS2, kind, **{k: kw[k] for k in SECTION_OPTIONS if k in kw})
  for k in SECTION_OPTIONS:
    kw.pop(k, None)
  projKw, axesKw, whereKw = _split(kw)
  fig, plots = _sectionAxes(oS, columns, axesKw, whereKw, field=field)
  hs = []
  for k, cell in enumerate(plots):
    for sP in cell:
      V, X, Y = sP.proj.grid(resolution)
      if not V.size:
        continue
      rgb = key.orientation2color(oS.orientationsAt(k, V.flatten())).reshape(X.shape + (3,))
      rgba = np.concatenate([np.clip(np.nan_to_num(rgb, nan=1.0), 0, 1), np.all(np.isfinite(rgb), axis=-1)[..., None].astype(float)], axis=-1)
      # a chart whose region does not reach the whole grid leaves NaN coordinates; the mesh
      # is drawn over the blocks of columns that are finite, as a filled plot is. Its cells
      # follow the curved chart, so matplotlib's warning about their edges is expected
      import warnings

      from .sphericalplot import _blocks
      with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")
        for j0, j1 in _blocks(X):
          hs.append(sP.ax.pcolormesh(X[:, j0:j1], Y[:, j0:j1], rgba[:, j0:j1], shading="nearest", zorder=2))
  return hs
