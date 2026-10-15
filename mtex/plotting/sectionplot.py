# the matplotlib drawing of ODF sections: a grid of charts, one cell per section (MATLAB
# @SO3Fun/plotSection, @orientation/plotSection, @ODFSections/plot; docs/plans/plotting.md S2)
#
# A section object hands out charts and the orientations at their points; this file lays
# the charts out, evaluates a function or places a list of orientations on them, and draws
# with the machinery of sphericalplot.py.
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..functions.so3funclass import SO3Fun
from ..geometry.crystalshape import crystalShape
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d, asVector3d
from .keywords import lineKeywords, refuseVariants
from .mtexfigure import gcm, mtexFigure
from .sections import ODFSections, SECTION_OPTIONS, newODFSectionPlot
from .sphericalplot import (FILL_FLAGS, fillFlag, _arrows, _filled, _markers, _rangeOf, _split, sphericalPlot)
from ..geometry.arguments import refuseFlags

if TYPE_CHECKING:
  from typing import Unpack

  from .keywords import PlotKeywords

FIELD_COLOR = (0.72, 0.72, 0.72)
OUTER_COLOR = (0.75, 0.75, 0.75)


# one cell per section, in a grid of the given width; the axis angle sections rescale
# theirs, so that equal areas stand for equal orientation volumes
def _sectionAxes(oS, columns, axesKw, whereKw, field=True, onto=False):
  n = oS.numSections
  charts = [oS.sectionCharts(k) for k in range(n)]
  wide = max(len(c) for c in charts)
  # a cell is as tall as its chart is, so a row of half discs leaves no band of white
  aspect = max(_aspect(c[0].bounds()) for c in charts)
  # layout=[rows, cols] names the grid, as MATLAB's option does
  layout = whereKw.get("layout")
  if layout is not None:
    ncols = int(layout[1]) * wide
  else:
    ncols = (columns or _gridColumns(n, aspect, wide)) * wide
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
  labels = oS.axisLabels
  # the panels go into the axes nextAxis prepared and those after it, or the first of a
  # fresh newMtexFigure, where the layout has room for them
  fig = ax = None
  if held is not None and not held.hold and (held.pending is not None or not held.children):
    free = held.nrows * held.ncols - len(held.children) + (held.pending is not None)
    if n * wide <= free:
      fig, ax = held, held.takeAxis()
  # a row takes its panel and, with axis labels, room for the title, the ticks and the label
  margin = 0.9 if labels is not None else 0.4
  figSize = figureSize(whereKw.get("figSize"), (min(3.2 * ncols, 16), min(3.2 * aspect * nrows + margin * nrows, 12)))
  if fig is None:
    fig = mtexFigure(n * wide, columns=ncols, figSize=figSize)
  fig.sections = oS
  outer = oS.outerSector
  axisLabels = axesKw.pop("axisLabels", labels is None and outer is None)
  plots = []
  for k in range(n):
    cell = []
    for j, proj in enumerate(charts[k]):
      ax = fig.nextAxis() if ax is None or (k, j) != (0, 0) else ax
      sP = sphericalPlot(ax, proj, axisLabels=axisLabels, title=oS.sectionLabel(k) if j == 0 else None, **axesKw)
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


# the height to width ratio of a chart's bounds (xmin, xmax, ymin, ymax)
def _aspect(b):
  return (b[3] - b[2]) / (b[1] - b[0])


# the columns of a grid of n cells of the given height to width ratio: the figure about
# 1.6 times as wide as tall, so wide Euler sections stack in two columns as MATLAB's do
def _gridColumns(n, aspect=1.0, wide=1):
  # up to four round sections side by side, three of two panels each, as MATLAB lays them
  # out on a wide screen
  if n <= (4 if wide == 1 else 3) and aspect >= 0.5:
    return n
  # up to three wide sections stack in one column, as MATLAB's do
  if n <= 3:
    return 1
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
# deterministically (MATLAB 'points', 'all'); the NaN orientations draw nothing and count not
def _thinned(ori, values, points, all_):
  ori = ori.flatten()
  # one value per orientation, or one row, an RGB colour
  if values is not None:
    values = np.asarray(values)
    values = values.ravel() if values.size == ori.size else values.reshape(ori.size, -1)
  keep = ~np.any(np.isnan(ori.data.reshape(ori.size, -1)), axis=1)
  if not keep.all():
    ori, values = ori[keep], None if values is None else values[keep]
  n = ori.size
  if all_ or n <= points:
    return ori, values
  print(f"plot {points} random orientations out of {n} given orientations")
  idx = np.unique(np.round(np.linspace(0, n - 1, points)).astype(int))
  return ori[idx], None if values is None else values[idx]


def plotSection(x, *args, **kw: Unpack[PlotKeywords]):
  """plotSection(odf), plotSection(odf, 'sigma'), plotSection(ori, oS, sections=9)

  An SO3Fun or a list of orientations sliced into sections. Without a section object the
  default is the phi2 sections against a specimen frame and the axis angle sections between
  two crystals, as in MTEX; a string picks another family ('phi1', 'phi2', 'alpha',
  'gamma', 'sigma', 'pf', 'axisAngle') and the keywords of the family name its sections.

  Example
  Plot a model ODF in three specified Bunge phi2 sections. The remaining two Euler
  angles vary within each panel; intensities are in multiples of a random distribution.

    import numpy as np
    import matplotlib.pyplot as plt
    from mtex import crystalFrame, orientation, degree, unimodalODF, plotSection, mtexColorbar

    cs = crystalFrame('m-3m')
    centre = orientation.byEuler(15*degree, 35*degree, 45*degree, cs)
    odf = unimodalODF(centre, halfwidth=15*degree)
    plotSection(odf, 'phi2', np.array([0, 45, 90])*degree, resolution=5*degree, contourf=True)
    mtexColorbar(title='mrd')
    plt.show()

  A section is a slice of the ODF. A pole figure is a projection of it; use plotPF for that view.
  """
  # a function on the sphere along a circle: the polar section
  from .mtexfigure import firstOfList
  if isinstance(x, SO3Fun):
    x = firstOfList(x)
  if args and isinstance(args[0], vector3d) and not isinstance(x, orientation) and not isinstance(x, SO3Fun):
    from .polarplot import plotSectionS2
    return plotSectionS2(x, *args, **kw)
  kw.pop("verbose", None)
  kind, shapes = None, None
  for i, a in enumerate(args):
    if isinstance(a, str) and (a in FILL_FLAGS or a == "complete"):
      refuseFlags((a,), "plotSection")
    elif isinstance(a, (str, ODFSections)):
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
  options = _sectionOptions(kw)
  if isinstance(kind, ODFSections):
    oS = kind
  else:
    frameA, frameB = x.CS, x.SS
    # complete: the sections of the whole orientation space, the frames without their groups
    if kw.pop("complete", False):
      frameA, frameB = frameA.stripSym(), frameB.stripSym()
    # misorientations with grain exchange are drawn on the antipodal sections
    if getattr(x, "antipodal", False) and frameA is frameB:
      options.setdefault("antipodal", True)
    oS = newODFSectionPlot(frameA, frameB, kind, **options)
  # MATLAB's coordinates='off' hides the ticks of the Euler sections, xlabel and ylabel name the axes
  decor = {k: kw.pop(k) for k in ("coordinates", "xlabel", "ylabel") if k in kw}
  if isinstance(x, SO3Fun):
    hs = _plotSectionFun(x, oS, **kw)
  elif shapes is not None:
    hs = _plotSectionShapes(x, shapes, oS, **kw)
  else:
    hs = _plotSectionOri(x, oS, **kw)
  if decor:
    _decorate(gcm(), decor)
  return hs


# the options of the sections, popped from the keywords
def _sectionOptions(kw):
  return {k: kw.pop(k) for k in SECTION_OPTIONS if k in kw}


# the sections some of the orientations fall into: k, the charts of section k and the
# indices of the orientations in it
def _perSection(plots, sec):
  for k, cell in enumerate(plots):
    ind = np.flatnonzero(sec == k)
    if len(ind):
      yield k, cell, ind


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


def plotSections(oS, columns=0, field=True, **kw: Unpack[PlotKeywords]):
  """the empty charts of a section plot, one per section, with the reference field of every
  section; MATLAB's ODFSections/plot

    plot(oS)                                       # the charts of the sections
    fig = plotSections(oS, columns=2)              # in two columns
    plotSections(oS, field=False)                  # without the reference fields

  A section plot of orientations or of a function draws into these charts; the figure keeps
  oS as its sections, so that later plots of orientations, fibres or annotations go into the
  section each lies in. Sections that rescale their coordinate show the sector all of them
  lie in. The other keywords are those of the spherical charts, figSize and layout among
  them.

  Input
    oS - ODFSections, as phi2Sections or sigmaSections
  Output
    fig - mtexFigure
  Keywords
    columns - the number of columns of sections; by default a grid near a slightly wide
        picture
    field - draw the field each section counts its coordinate from, where it has one
  See also
    plotSection, phi2Sections, sigmaSections, newODFSectionPlot"""
  projKw, axesKw, whereKw = _split(kw)
  fig, plots = _sectionAxes(oS, columns, axesKw, whereKw, field=field)
  return fig


def annotateSection(ori, *args, **kw: Unpack[PlotKeywords]):
  """annotate(ori) on a section plot: every orientation marked in the section it lies in"""
  oS = _currentSections()
  kw.pop("add2all", None)
  values = next((a for a in args if not isinstance(a, str)), None)
  kw.setdefault("markerSize", 8)
  return _plotSectionOri(ori, oS, values=values, all=True, onto=True, **kw)


def plotFibreSection(f, points=1000, lineColor="k", lineWidth=1.5, **kw: Unpack[PlotKeywords]):
  """a fibre on the current section plot, as lines through the sections it passes; what
  plot(f) does on a section plot, MATLAB's line of the orientations of the fibre

    odf.plotSection(); plot(f)                     # the fibre over the sections of an ODF
    plot(f, points=200, lineColor='r', lineWidth=2)  # coarser, red and thicker
    plotFibreSection(f)                            # the same, called directly

  The fibre is sampled at evenly spaced orientations and every symmetric equivalent of them
  is put into the section it lies in; neighbours along the fibre in one section are joined by
  a line, which breaks where the fibre leaves the section or the chart. Further keywords go to
  the lines.

  Input
    f - fibre
  Output
    h - list of the line handles
  Keywords
    points - the number of orientations along the fibre
    lineColor - the colour of the lines
    lineWidth - the width of the lines
  See also
    fibre, plotSection, annotateSection"""
  oS = _currentSections()
  kw.pop("add2all", None)
  lineColor = kw.pop("color", lineColor)
  ori = f.flatten().orientation(points) if hasattr(f, "flatten") else f.orientation(points)
  _, axesKw, whereKw = _split(kw)
  kw = lineKeywords(kw)
  fig, plots = _sectionAxes(oS, 0, axesKw, whereKw, onto=True)
  vs, sec, src = oS.sectionData(ori)
  hs = []
  for k, cell, ind in _perSection(plots, sec):
    # the points of one equivalent come in the order of the fibre: a chain breaks where the
    # next point is not the next one of the fibre
    gap = (np.diff(ind) != 1) | (np.diff(src[ind]) != 1)
    for sP in cell:
      x, y = sP.proj.project(vs[ind])
      x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
      # and where it leaves the chart and comes back elsewhere
      step = np.hypot(np.diff(x), np.diff(y))
      jump = np.flatnonzero(gap | (step > 0.25 * np.nanmax(np.abs(np.concatenate([x, y])), initial=1.0)))
      x, y = np.insert(x, jump + 1, np.nan), np.insert(y, jump + 1, np.nan)
      hs.append(sP.ax.plot(x, y, color=lineColor, linewidth=lineWidth, zorder=7, **kw))
  return hs


def quiverSection(ori, w, *args, color="k", lineWidth=1.0, resolution=10 * degree, **kw: Unpack[PlotKeywords]):
  """arrows on a section plot: of a function on the sphere along one of its circles, or at
  orientations in the sections of orientation space; MATLAB's S2Fun/quiverSection

    quiverSection(sF, vF, zvector)                 # the field vF along the circle normal to z
    quiverSection(sF, vF, zvector, np.pi/3)        # along the small circle at 60 degree from z
    quiverSection(sF, vF, zvector, normalized=True)  # every vector of the same length
    quiverSection(ori, xvector)                    # an arrow along x at every orientation
    quiverSection(ori, w, color='r', resolution=5*degree)  # a direction per orientation

  With a function on the sphere sF, the field vF (an S2VectorField, an S2AxisField or a
  function of the directions) is evaluated at 36 points of the circle normal to sec, or of
  the small circle at the polar angle eta from sec. The vectors are drawn on the curve
  plotSection(sF, sec) draws, seen down sec. The longest arrow is scaled to the spacing of
  the points (autoScaleFactor=0.9); an axis field draws segments without heads. This form
  also takes displayName and how2plot.

  With orientations, the arrows go into the sections of the current section plot, which
  plotSection draws first. Every orientation gets an arrow along the specimen direction w in
  the section it lies in; w is one direction for all or one per orientation.

  Input
    sF - S2Fun, the function whose section is drawn
    vF - S2VectorField, S2AxisField, a vector3d per point of the circle, or a function
    sec - vector3d, the normal of the section
    eta - the polar angle of the circle from sec, optional, pi/2 the great circle
    ori - orientation, on a section plot
    w - vector3d, a specimen direction for all or one per orientation
  Keywords
    color - the colour of the arrows
    lineWidth - the width of the arrows
    resolution - the spacing the arrows are scaled to, an angle: every arrow is half of it
        long (orientations only)
  See also
    plotSection, quiver, S2FunHarmonic.grad, S2FunVectorField, OrientationVisualizationSections"""
  from ..geometry.quaternion import quaternion
  if not isinstance(ori, quaternion):
    from .polarplot import quiverSectionS2
    return quiverSectionS2(ori, w, *args, color=color, lineWidth=lineWidth, **kw)
  oS = _currentSections()
  _, axesKw, whereKw = _split(kw)
  kw = lineKeywords(kw)
  fig, plots = _sectionAxes(oS, 0, axesKw, whereKw, onto=True)
  vs, sec, src = oS.sectionData(ori)
  w = asVector3d(w)
  ws = vector3d(np.broadcast_to(w.data.reshape(-1, 3), (ori.size, 3)) if w.size == 1 else w.data.reshape(-1, 3))
  hs = []
  for k, cell, ind in _perSection(plots, sec):
    for sP in cell:
      hs.append(_arrows(sP, vs[ind], ws[src[ind]], resolution=resolution, normalize=True, color=color, lineWidth=lineWidth, **kw))
  return hs


def quiverSectionField(F, *args, color="k", lineWidth=1.0, resolution=10 * degree, **kw: Unpack[PlotKeywords]):
  """plot of a vector field on SO(3) as arrows in the sections of orientation space

    plot(G)                                        # arrows in phi2 sections, G = odf.grad()
    quiverSectionField(G, 'phi1')                  # in phi1 sections
    quiverSectionField(G, resolution=15*degree, color='b')  # a coarser grid of blue arrows

  At the nodes of a grid in every section the field is evaluated, and the arrow shows where
  a small turn about the field's axis moves the orientation within the section, its length
  the size of the field. The arrows go into the sections of a held figure, the sections given
  as an ODFSections object or named by a string, or new phi2 sections; Euler angle and pole
  figure sections (sigma) are drawn, the others refused.

  Input
    G - SO3VectorField, a vector field on SO(3), as the gradient of an ODF
  Keywords
    color - the colour of the arrows
    lineWidth - the width of the arrows
    resolution - the spacing of the grid of arrows in every section, an angle
  See also
    SO3FunHarmonic.grad, plotSection, quiverSection, SO3FunVectorField"""
  from ..geometry.grids.s2grid import equispacedS2Grid
  from ..geometry.kernels import qmul
  from .sections import eulerSections, pfSections
  refuseVariants(kw)
  oS = next((a for a in args if isinstance(a, ODFSections)), None)
  held = gcm()
  if oS is None:
    oS = held.sections if held is not None and held.hold else None
  if oS is None:
    oS = newODFSectionPlot(F.CS, F.SS, next((a for a in args if isinstance(a, str)), None))
  if not isinstance(oS, (eulerSections, pfSections)):
    raise NotImplementedError("a vector field on SO(3) is drawn into Euler angle or pole figure sections")
  projKw, axesKw, whereKw = _split(kw)
  kw.pop("add2all", None)
  fig, plots = _sectionAxes(oS, 0, axesKw, whereKw, onto=True)
  if isinstance(oS, eulerSections):
    # a regular grid over the box of a section, the cell centres
    theta = np.arange(resolution / 2, oS.maxPhi, resolution)
    azim = np.arange(resolution / 2, oS.maxAzimuth, resolution)
    T, A = np.meshgrid(theta, azim, indexing="ij")
    vs = vector3d.byPolar(T.ravel(), A.ravel())
  else:
    # the pole grid inside the sector of the sections
    vs = equispacedS2Grid(resolution=resolution)
    vs = vector3d(vs.data.reshape(-1, 3)[oS.sR.checkInside(vs).ravel()])
  t = 1e-4
  hs = []
  for k, cell in enumerate(plots):
    ori = oS.orientationsAt(k, vs)
    v = F.eval(ori)
    v = np.asarray(v.data if isinstance(v, vector3d) else v, dtype=float).reshape(-1, 3)
    if isinstance(oS, pfSections):
      # a left turn about v moves the pole ori * h1, which is the point of the chart, along v x pole
      w = vector3d(np.cross(v, vs.normalize().data.reshape(-1, 3)))
    else:
      # the left turn exp(t v) ori, and where it lands in the two angles of the section
      size = np.linalg.norm(v, axis=1)
      axis = v / np.where(size > 0, size, 1.0)[:, None]
      step = np.column_stack([np.cos(t * size / 2), np.sin(t * size / 2)[:, None] * axis])
      moved = qmul(step, ori.data.reshape(-1, 4))
      _, th, az = oS._triples(moved)
      dth = (th - T.ravel()) / t
      daz = (np.mod(az - A.ravel() + np.pi, 2 * np.pi) - np.pi) / t
      w = vector3d.byPolar(T.ravel() + 1e-6 * dth, A.ravel() + 1e-6 * daz) - vs
      w = vector3d(w.data / 1e-6)
    for sP in cell:
      hs.append(_arrows(sP, vs, w, resolution=resolution, arrowSize=0.9, color=color, lineWidth=lineWidth, **kw))
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
  for k, cell, ind in _perSection(plots, sec):
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
  style = fillFlag(kw, "pcolor")
  projKw, axesKw, whereKw = _split(kw)
  fig, plots = _sectionAxes(oS, columns, axesKw, whereKw, field=field)
  grids = [[sP.proj.grid(resolution) for sP in cell] for cell in plots]

  # the sections of a complete plot live in the frames without their groups; the function
  # is asked in its own frames; values on the nodes of makeGrid come as a plain function
  def at(o):
    if not hasattr(f, "CS"):
      return f(o)
    if getattr(o, "CS", None) is not f.CS or getattr(o, "SS", None) is not f.SS:
      o = orientation(rotation(o.data), f.CS, f.SS)
    return f(o)
  def valuesAt(k, V, X):
    if not V.size:
      return np.zeros((0, 0))
    return np.real(np.asarray(at(oS.orientationsAt(k, V.flatten())), dtype=complex)).reshape(X.shape)
  values = [[valuesAt(k, V, X) for V, X, _ in cell] for k, cell in enumerate(grids)]
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
                    field=True, markerSize=4, onto=False, add2all=False, **kw):
  projKw, axesKw, whereKw = _split(kw)
  ori, values = _thinned(ori, values, points, all)
  # add2all draws into the sections already there, as MATLAB's flag does
  fig, plots = _sectionAxes(oS, columns, axesKw, whereKw, field=field, onto=onto or add2all)
  vs, sec, src = oS.sectionData(ori)
  # a row per orientation is its RGB colour, a number its value on the colour map
  rgb = values is not None and values.ndim == 2
  if values is not None and not rgb and colorRange is None and len(values):
    colorRange = (float(np.nanmin(values)), float(np.nanmax(values)))
  hs = []
  for k, cell, ind in _perSection(plots, sec):
    dirs = vs[ind]
    vals = None if values is None else values[src[ind]]
    for sP in cell:
      if rgb:
        hs.append(_markers(sP, dirs, rgb=vals, markerSize=markerSize, **kw))
      else:
        hs.append(_markers(sP, dirs, vals, colorRange=colorRange, markerSize=markerSize, **kw))
  if colorbar and values is not None and not rgb:
    fig.colorbar()
  return hs


def plotKeySections(key, *args, resolution=2.5 * degree, columns=0, field=False, **kw: Unpack[PlotKeywords]):
  """plot of an orientation colour key, its colours over the sections of orientation space;
  MATLAB's orientationColorKey/plot

    plot(key)                                      # the colours in the default sections
    plotKeySections(key, 'sigma', sections=6)      # six sigma sections
    plotKeySections(key, resolution=5*degree, columns=3, field=True)  # coarser, in three columns

  Every section is filled with the colours key.orientation2color gives the orientations of
  its grid; orientations the key leaves without a colour stay white. An antipodal key, one
  with grain exchange symmetry, is drawn over the region that symmetry halves.

  The sections follow the frames of the key, frameA and frameB. They are named by a string
  ('phi2', 'phi1', 'sigma', ...) or given as an ODFSections object, and the keywords of the
  sections (sections, phi2, maxPhi, ...) choose them as for plotSection. A key on the
  sphere, as ipfHSVKey, draws its inverse pole figure with plot(key); plotKeySections draws
  it in sections too.

  Input
    key - an orientation colour key, as BungeColorKey, axisAngleColorKey or spotColorKey;
        ipfHSVKey through plotKeySections itself
  Output
    h - the list of matplotlib meshes drawn
  Keywords
    resolution - the spacing of the grid the colours are computed on, an angle
    columns - the number of columns of the sections, 0 to choose it from the figure
    field - draw into every section the arrows of the reference field its coordinate is
        counted from
  See also
    orientationColorKey.orientation2color, plotSection, ipfColorKey, BungeColorKey, EBSDOrientationPlots,
    OrientationVisualizationSections"""
  kind = next((a for a in args if isinstance(a, (str, ODFSections))), None)
  # an antipodal key, one with grain exchange, is drawn over the region that symmetry halves
  if getattr(key, "antipodal", False):
    kw.setdefault("antipodal", True)
  options = _sectionOptions(kw)
  oS = kind if isinstance(kind, ODFSections) else newODFSectionPlot(key.frameA, key.frameB, kind, **options)
  import warnings

  from .sphericalplot import _blocks
  projKw, axesKw, whereKw = _split(kw)
  fig, plots = _sectionAxes(oS, columns, axesKw, whereKw, field=field)
  hs = []
  for k, cell in enumerate(plots):
    for sP in cell:
      V, X, Y = sP.proj.grid(resolution)
      if not V.size:
        continue
      rgb = key.orientation2color(oS.orientationsAt(k, V.flatten())).reshape(X.shape + (3,))
      alpha = np.all(np.isfinite(rgb), axis=-1)[..., None].astype(float)
      rgba = np.concatenate([np.clip(np.nan_to_num(rgb, nan=1.0), 0, 1), alpha], axis=-1)
      # a chart whose region does not reach the whole grid leaves NaN coordinates; the mesh
      # is drawn over the blocks of columns that are finite, as a filled plot is. Its cells
      # follow the curved chart, so matplotlib's warning about their edges is expected
      with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")
        for j0, j1 in _blocks(X):
          hs.append(sP.ax.pcolormesh(X[:, j0:j1], Y[:, j0:j1], rgba[:, j0:j1], shading="nearest", zorder=2))
  return hs


__all__ = ["plotKeySections", "plotSection", "plotSections", "annotateSection", "plotFibreSection", "quiverSection", "quiverSectionField"]
