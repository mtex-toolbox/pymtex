# the pole figures of orientation lists and of measured pole figures (MATLAB
# @orientation/plotPDF, plotIPDF, @PoleFigure/plot, plotDiff; docs/plans/plotting.md S6)
#
# A row of panels, one per crystal or specimen direction: the poles of a list as markers on
# the sector of their frame, coloured by a value per orientation when given, or the density
# of the poles as a surface when a style asks for one; a measured pole figure as its
# specimen directions coloured by intensity, each panel on its own colour range, since the
# counts of two measurements are not comparable.
from __future__ import annotations

import numpy as np

from ..geometry.orientation import orientation
from ..geometry.sphericalregion import projectIntoRegion, sphericalRegion
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .mtexfigure import gcm, mtexFigure
from .poles import dynamicMarkerSize, inversePoles, poles
from .spherical import hasSymmetry, makeSphericalProjection, plotRegion
from .sphericalplot import fillFlag, intoParents, _directionTitle, _markers, _plotRow, _rangeOf, _split, sphericalPlot
from . import sphericalplot


def _asList(h):
  if isinstance(h, vector3d):
    return [h[i] for i in range(h.size)] if h.size > 1 else [h]
  return [x if isinstance(x, vector3d) else vector3d(x) for x in h]


# a long list is thinned before it is drawn, deterministically, as MTEX thins it too
_HINTED = [False]


def sampleMessage(points, n):
  """the sample says its size, as MATLAB does, and how to change it once per session"""
  print(f"  I'm plotting {points} random orientations out of {n} given orientations")
  if not _HINTED[0]:
    print('  You can specify the number of points by the option "points".\n  The option "all" ensures that all data are plotted')
    _HINTED[0] = True


# the values flattened with the orientations, a colour per orientation kept as a row
def _flatValues(values):
  if values is None:
    return None
  v = np.asarray(values)
  return v.reshape(-1, 3) if v.ndim >= 2 and v.shape[-1] == 3 else v.reshape(-1)


def _thinned(o, values, points):
  n = o.size
  values = _flatValues(values)
  if points is None or points == "all" or points <= 0 or points >= n:
    return o.flatten(), values
  sampleMessage(points, n)
  take = np.sort(np.random.default_rng(0).permutation(n)[:points])
  return o.flatten()[take], None if values is None else values[take]


# a value per orientation carried onto every one of its equivalents
def _spread(values, ps):
  if values is None:
    return None
  v = np.asarray(values)
  n = ps.shape[-1]
  # a colour per orientation is a row, a value a number
  return np.repeat(v.reshape(-1, 3), n, axis=0) if v.ndim == 2 and v.shape[-1] == 3 else np.repeat(v.reshape(-1), n)


# a row of lists of directions, each with a value per direction or none: markers on the
# sector of the frame the directions live in, or the density of the directions, or the
# interpolant of the values, when a style asks for a surface
def _dataRow(vss, titles, valuess=None, style="scatter", colorbar=False, colorRange=None, MarkerSize=None, halfwidth=5 * degree,
             noTitle=False, minmax=False, **kw):
  if style != "scatter":
    from ..functions.density import calcDensity, interpolate
    fs = [calcDensity(vs, halfwidth=halfwidth) if vals is None else interpolate(vs, np.asarray(vals, dtype=float).reshape(-1))
          for vs, vals in zip(vss, valuess or [None] * len(vss))]
    return _plotRow(fs, titles, style=style, colorbar=colorbar, colorRange="individual" if colorRange is None else colorRange,
                    noTitle=noTitle, minmax=minmax, **kw)
  projKw, axesKw, whereKw = _split(kw)
  complete = projKw.pop("complete", False)
  charts = []
  for vs, title in zip(vss, titles):
    ax = projKw.get("antipodal", vs.antipodal)
    region = sphericalRegion(frame=vs.frame) if complete else plotRegion(vs.frame, fundamentalRegion=True, antipodal=ax,
                                                                             how2plot=projKw.get("how2plot"))
    pk = dict(projKw, antipodal=ax, complete=True) if complete else dict(projKw, antipodal=ax)
    projs = makeSphericalProjection(region, **pk)
    # a sphere drawn as two hemispheres is titled once and each half says which it is
    for i, p in enumerate(projs):
      charts.append((vs, region, p, None if noTitle or i else title, len(projs) == 2))
  # a held figure of as many charts takes the markers on top of what it has
  fig = gcm()
  if whereKw.get("parent") is not None:
    fig, sPs = intoParents(whereKw["parent"], [(c[2], c[3]) for c in charts], axesKw)
  elif fig is not None and fig.hold and len(fig.children) == len(charts) and all(hasattr(ax, "_mtexSphericalPlot") for ax in fig.children):
    sPs = [ax._mtexSphericalPlot for ax in fig.children]
  # the axes nextAxis prepared takes one chart, a fresh figure of newMtexFigure its charts
  elif fig is not None and fig.pending is not None and len(charts) == 1:
    sPs = [sphericalPlot(fig.takeAxis(), charts[0][2], **axesKw, title=charts[0][3])]
  elif fig is not None and not fig.children and not fig.hold:
    sPs = [sphericalPlot(fig.nextAxis(), p, **axesKw, title=t) for _, _, p, t, _ in charts]
  else:
    fig = mtexFigure(len(charts), figSize=whereKw.get("figSize"))
    sPs = [sphericalPlot(fig.nextAxis(), p, **axesKw, title=t) for _, _, p, t, _ in charts]
  if len(sPs) == len(charts):
    for sP, c in zip(sPs, charts):
      if c[4]:
        sP.hemisphereLabel()
  individual = isinstance(colorRange, str) and colorRange == "individual"
  # 'equal' is one range over the row, the default for markers
  if isinstance(colorRange, str) and colorRange == "equal":
    colorRange = None
  if valuess is not None and colorRange is None:
    colorRange = _rangeOf([np.asarray(v, dtype=float) for v in valuess])
  hs = []
  valuess = valuess or [None] * len(vss)
  ranges = {}
  for sP, (vs, region, _, _, _) in zip(sPs, charts):
    k = next(i for i, x in enumerate(vss) if x is vs)
    vals = valuess[k]
    # one marker per orbit: the pole in the sector stands for all its equivalents
    w = projectIntoRegion(vs, region) if hasSymmetry(vs.frame) and not complete else vs
    # the values were spread over the equivalents; a sector keeping one pole per orbit takes one back
    if vals is not None and w.size < vs.size and vs.size % w.size == 0:
      vals = np.asarray(vals)[::vs.size // w.size]
    rgb = vals if vals is not None and np.ndim(vals) == 2 and np.shape(vals)[-1] == 3 else None
    vals = None if rgb is not None else vals
    cr = ranges.setdefault(k, _rangeOf([np.asarray(vals, dtype=float)])) if individual and vals is not None else colorRange
    size = MarkerSize if MarkerSize is not None else dynamicMarkerSize(w.size)
    # a size per orientation is repeated over its equivalents, as a value is
    if np.ndim(size) > 0 and np.size(size) != w.size and np.size(size) > 0:
      size = np.repeat(np.asarray(size, dtype=float).reshape(-1), max(w.size // np.size(size), 1))
    hs.append(_markers(sP, w, vals, rgb=rgb, colorRange=None if vals is None else cr, MarkerSize=size, **kw))
    if minmax:
      sP.annotateMinMax()
  if colorbar and any(v is not None for v in valuess):
    # into axes named by parent the bars go beside those axes only
    if whereKw.get("parent") is not None:
      fig.colorbars += [fig.fig.colorbar(sP.mappable, ax=sP.ax, shrink=0.7) for sP in sPs if getattr(sP, "mappable", None) is not None]
    else:
      fig.colorbar()
  return hs


# the directions and the data of plotPDF(ori, h), plotPDF(ori, data, h) and plotPDF(ori, h,
# data): a direction is a vector3d or a list of them, everything else is data
def _pickData(o, args):
  h, data = None, None
  for a in args:
    isDirection = isinstance(a, vector3d) or (isinstance(a, (list, tuple)) and len(a) > 0 and isinstance(a[0], vector3d))
    if isDirection and h is None:
      h = a
    else:
      data = a
  if h is None:
    raise ValueError("the directions of the pole figures are needed")
  return h, data


# a fibre is the curve of its orientations, drawn as dots in its line colour
def _fibreAsCurve(o, points, all, kw):
  from ..geometry.fibre import fibre
  if not isinstance(o, fibre):
    return o, points, all
  color = kw.pop("lineColor", kw.pop("LineColor", "k"))
  kw.pop("lineWidth", None)
  kw.pop("LineWidth", None)
  kw.setdefault("MarkerSize", 1.5)
  kw.setdefault("MarkerFaceColor", color)
  kw.setdefault("MarkerEdgeColor", color)
  o = o.flatten().orientation(points or 1000) if hasattr(o, "flatten") else o.orientation(points or 1000)
  return o, None, True


# MATLAB's positional styles: a string is a flag ('upper', 'antipodal', 'contourf'), an RGB
# triple beside anything but three orientations the marker colour
def _positionalStyle(o, args, kw):
  out = []
  for a in args:
    if isinstance(a, str):
      kw[a] = True
    elif isinstance(a, (tuple, list, np.ndarray)) and len(a) == 3 and all(isinstance(x, (int, float, np.number)) for x in a) \
        and np.size(o) != 3:
      kw.setdefault("MarkerFaceColor", np.asarray(a, dtype=float))
    else:
      out.append(a)
  return tuple(out)


def plotPDF(o, *args, points=None, all=False, superposition=None, **kw):
  """plotPDF(ori, h), plotPDF(ori, data, h), plotPDF(ori, h, points=100, contourf=True, antipodal=True)

  The pole figures of a list of orientations: every orientation applied to every equivalent
  of h, as markers on the specimen sector, coloured by a value per orientation when given;
  a style draws the density of the poles instead. An ODF draws its pole density functions.
  """
  o, points, all = _fibreAsCurve(o, points, all, kw)
  if not isinstance(o, orientation):
    return sphericalplot.plotPDF(o, *args, superposition=superposition, **kw)
  args = _positionalStyle(o, args, kw)
  # plotPDF(ori, 'add2all') without directions, or into held pole figures: the poles of the
  # directions they already show
  fig = sphericalplot.gcm()
  if (kw.get("add2all") or (fig is not None and fig.hold)) and not any(isinstance(a, vector3d) or (isinstance(a, (list, tuple)) and a and isinstance(a[0], vector3d)) for a in args):
    kw["add2all"] = True
    return sphericalplot.plot(o, **kw)
  h, data = _pickData(o, args)
  hs = _asList(h)
  flag = fillFlag(kw, None)
  style = "scatter" if flag is None else "pcolor" if flag == "smooth" else flag
  antipodal = kw.get("antipodal", False)
  # the budget is in markers, and a marker stands for an orbit
  budget = "all" if all else points if points is not None else max(10000 // (o.CS.numSym() * o.SS.numSym()), 1)
  if style == "scatter":
    o, data = _thinned(o, data, budget)
  ps = [poles(o.flatten(), x, antipodal=antipodal) for x in hs]
  out = _dataRow([p.flatten() for p in ps], [x.char() for x in hs], None if data is None else [_spread(data, p) for p in ps],
                 style=style, **kw)
  sphericalplot.tagPoleAxes(hs, len(out) if isinstance(out, list) else len(hs))
  return out


def plotIPDF(o, *args, points=None, all=False, **kw):
  """plotIPDF(ori, r), plotIPDF(ori, data, r): the inverse pole figures of a list of orientations,
  the crystal directions along r as markers on the crystal sector"""
  o, points, all = _fibreAsCurve(o, points, all, kw)
  if not isinstance(o, orientation):
    return sphericalplot.plotIPDF(o, *args, **kw)
  r, data = _pickData(o, _positionalStyle(o, args, kw))
  rs = _asList(r)
  flag = fillFlag(kw, None)
  style = "scatter" if flag is None else "pcolor" if flag == "smooth" else flag
  antipodal = kw.get("antipodal", False)
  budget = "all" if all else points if points is not None else max(100000 // (o.CS.numSym() * o.SS.numSym()), 1)
  if style == "scatter":
    o, data = _thinned(o, data, budget)
  hs = [inversePoles(o.flatten(), x, antipodal=antipodal) for x in rs]
  return _dataRow([h.flatten() for h in hs], [_directionTitle(x) for x in rs], None if data is None else [_spread(data, h) for h in hs],
                  style=style, **kw)


# ------------------------------------------------------------------ measured pole figures

def _pfTitle(pf, i):
  return pf.allH[i].char()


def plotPoleFigure(pf, colorRange="individual", bandwidth=None, **kw):
  """plot(pf), plot(pf, contourf=True), plot(pf, minmax=True): a measured pole figure, or a
  data set of them in a row, the specimen directions as markers coloured by intensity"""
  from ..functions.s2fun import S2FunHarmonic
  flag = fillFlag(kw, None)
  titles = [_pfTitle(pf, i) for i in range(pf.numPF)]
  if flag is not None:
    style = "pcolor" if flag == "smooth" else flag
    fs = [S2FunHarmonic(pf.select(i), bandwidth=bandwidth) for i in range(pf.numPF)]
    return _plotRow(fs, titles, style=style, colorRange="individual" if colorRange == "individual" else colorRange, **kw)
  rs = [vector3d(pf.allR[i].data.reshape(-1, 3), frame=pf.allR[i].frame, antipodal=pf.allR[i].antipodal) for i in range(pf.numPF)]
  return _dataRow(rs, titles, [pf.allI[i].ravel() for i in range(pf.numPF)], colorRange=colorRange, **kw)


def plotDiff(pf, other, mode="relative", epsilon=None, **kw):
  """plotDiff(pf, odf), plotDiff(pf, pf2, 'l1'): the difference between a measured pole figure
  and a recalculated one, or two measurements, as a pole figure of the error; plotDiff(odf, pf)
  the same, plotDiff(odf1, odf2) the sections of their difference"""
  from ..functions.polefigure import calcErrorPF
  # plotDiff(odf, pf): the same error, plotDiff(odf1, odf2) the sections of the difference
  if not hasattr(pf, "allH"):
    if not hasattr(other, "allH"):
      from .sectionplot import plotSection
      return plotSection(pf - other, **kw)
    pf, other = other, pf
    kw.setdefault("colorRange", "equal")
  for m in ("RP", "l1", "l2", "relative"):
    if kw.pop(m, False):
      mode = m
  d = calcErrorPF(pf, other, mode, epsilon)
  kw.setdefault("colorbar", True)
  return plotPoleFigure(d, **kw)


__all__ = ["plotPDF", "plotIPDF", "plotPoleFigure", "plotDiff"]
