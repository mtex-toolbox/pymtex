# the pole figures of orientation lists and of measured pole figures (MATLAB
# @orientation/plotPF, plotIPF, @PoleFigure/plot, plotDiff; docs/plans/plotting.md S6)
#
# A row of panels, one per crystal or specimen direction: the poles of a list as markers on
# the sector of their frame, coloured by a value per orientation when given, or the density
# of the poles as a surface when a style asks for one; a measured pole figure as its
# specimen directions coloured by intensity, each panel on its own colour range, since the
# counts of two measurements are not comparable.
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..functions.polefigure import PoleFigure
from ..geometry.orientation import orientation
from ..geometry.sphericalregion import projectIntoRegion, sphericalRegion
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .mtexfigure import gcm, mtexFigure
from .poles import dynamicMarkerSize, inversePoles, poles
from .spherical import hasSymmetry, makeSphericalProjection, plotRegion
from .sphericalplot import fillFlag, intoParents, _asList, _directionTitle, _markers, _plotRow, _rangeOf, _split, sphericalPlot
from . import sphericalplot
from ..geometry.arguments import refuseFlags

if TYPE_CHECKING:
  from typing import Unpack

  from .keywords import PlotKeywords


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
def _dataRow(vss, titles, valuess=None, style="scatter", colorbar=False, colorRange=None, markerSize=None, halfwidth=5 * degree,
             title=True, minmax=False, **kw):
  if style != "scatter":
    from ..functions.density import calcDensity, interpolate
    fs = [calcDensity(vs, halfwidth=halfwidth) if vals is None else interpolate(vs, np.asarray(vals, dtype=float).reshape(-1))
          for vs, vals in zip(vss, valuess or [None] * len(vss))]
    return _plotRow(fs, titles, style=style, colorbar=colorbar, colorRange="individual" if colorRange is None else colorRange,
                    title=title, minmax=minmax, **kw)
  projKw, axesKw, whereKw = _split(kw)
  complete = projKw.pop("complete", False)
  charts = []
  for vs, name in zip(vss, titles):
    ax = projKw.get("antipodal", vs.antipodal)
    region = sphericalRegion(frame=vs.frame) if complete else plotRegion(vs.frame, fundamentalRegion=True, antipodal=ax,
                                                                             how2plot=projKw.get("how2plot"))
    pk = dict(projKw, antipodal=ax, complete=True) if complete else dict(projKw, antipodal=ax)
    projs = makeSphericalProjection(region, **pk)
    # a sphere drawn as two hemispheres is titled once and each half says which it is
    for i, p in enumerate(projs):
      charts.append((vs, region, p, None if not title or i else name, len(projs) == 2))
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
    cr = colorRange
    if individual and vals is not None:
      cr = ranges.setdefault(k, _rangeOf([np.asarray(vals, dtype=float)]))
    size = markerSize if markerSize is not None else dynamicMarkerSize(w.size)
    # a size per orientation is repeated over its equivalents, as a value is
    if np.ndim(size) > 0 and np.size(size) != w.size and np.size(size) > 0:
      size = np.repeat(np.asarray(size, dtype=float).reshape(-1), max(w.size // np.size(size), 1))
    hs.append(_markers(sP, w, vals, rgb=rgb, colorRange=None if vals is None else cr, markerSize=size, **kw))
    if minmax:
      sP.annotateMinMax()
  if colorbar and any(v is not None for v in valuess):
    # into axes named by parent the bars go beside those axes only
    if whereKw.get("parent") is not None:
      fig.colorbars += [fig.fig.colorbar(sP.mappable, ax=sP.ax, shrink=0.7) for sP in sPs if sP.mappable is not None]
    else:
      fig.colorbar()
  return hs


# the directions and the data of plotPF(ori, h), plotPF(ori, data, h) and plotPF(ori, h,
# data): a direction is a vector3d or a list of them, everything else is data
def _pickData(o, args):
  h, data = None, None
  for a in args:
    if _isDirections(a) and h is None:
      h = a
    else:
      data = a
  if h is None:
    raise ValueError("the directions of the pole figures are needed")
  return h, data


# whether an argument names directions, a vector3d or a list of them
def _isDirections(a):
  return isinstance(a, vector3d) or (isinstance(a, (list, tuple)) and len(a) > 0 and isinstance(a[0], vector3d))


# a fibre is the curve of its orientations, drawn as dots in its line colour
def _fibreAsCurve(o, points, all, kw):
  from ..geometry.fibre import fibre
  if not isinstance(o, fibre):
    return o, points, all
  color = kw.pop("lineColor", "k")
  kw.pop("lineWidth", None)
  kw.setdefault("markerSize", 1.5)
  kw.setdefault("markerFaceColor", color)
  kw.setdefault("markerEdgeColor", color)
  o = o.flatten().orientation(points or 1000) if hasattr(o, "flatten") else o.orientation(points or 1000)
  return o, None, True


# MATLAB's positional styles: a string is a flag ('upper', 'antipodal', 'contourf'), an RGB
# triple beside anything but three orientations the marker colour
def _positionalStyle(o, args, kw):
  out = []
  refuseFlags(args, "plot")
  for a in args:
    if isinstance(a, (tuple, list, np.ndarray)) and len(a) == 3 and all(isinstance(x, (int, float, np.number)) for x in a) \
        and np.size(o) != 3:
      kw.setdefault("markerFaceColor", np.asarray(a, dtype=float))
    else:
      out.append(a)
  return tuple(out)


def plotPF(o, *args, points=None, all=False, superposition=None, **kw: Unpack[PlotKeywords]):
  """pole figures, one per crystal direction: of an ODF, of a list of orientations or of a
  measured PoleFigure; MATLAB's plotPDF

    plotPF(odf, h)                                 # the pole figures of an ODF, h a list of Miller
    plotPF(odf, h, superposition=[0.5, 0.5])       # one pole figure, the superposition of h
    plotPF(ori, h)                                 # the poles of orientations as markers
    plotPF(ori, ori.angle(), h)                    # the markers coloured by a value per orientation
    plotPF(ori, h, contourf=True, antipodal=True)  # the density of the poles, filled contours
    plotPF(ori, h, points=50)                      # at most 50 orientations drawn
    plotPF(ori, h, all=True)                       # every orientation drawn
    plotPF(f, Miller(1, 0, 0, f.CS))               # a fibre as the curve of its poles
    plotPF(pf)                                     # a measured pole figure

  A pole figure shows where the crystal direction h, with every equivalent of it, points in
  the specimen. For an ODF it is the density of these poles, the Radon transform odf.radon(h),
  drawn with the keywords of a spherical function: contourf, contour, pcolor, colorRange
  ('equal' for one range over all pole figures, each its own by default), resolution,
  antipodal (both hemispheres superposed), complete, upper and lower. A positional '3d' draws
  it on the sphere in space.

  For a list of orientations every orientation is applied to every equivalent of h and the
  poles are drawn as markers on the specimen sector, coloured by a value or an RGB colour per
  orientation when one is given; contourf=True or another filled style draws their density.
  A large list is thinned to a random selection, about 10000 markers by default. A fibre is
  drawn as the dots of 1000 orientations along it. Into held pole figures, or with
  add2all=True, the poles of the directions already shown are added.

  A measured PoleFigure draws its intensities as markers, or, with a filled style, the
  function its harmonic expansion gives.

  Input
    odf - SO3Fun, an ODF
    ori - orientation, or a fibre
    data - a value or an RGB colour per orientation, optional
    h - Miller or a list of Miller, the crystal directions, one pole figure each
    pf - PoleFigure, measured pole figures
  Keywords
    points - the number of orientations drawn at most, a random selection
    all - draw every orientation, however many
    superposition - the weights of the directions h in one superposed pole figure, a list of
        arrays for one superposed pole figure per entry of h, as plotPF(odf, pf.allH,
        superposition=pf.c)

  Example
  Compare poles of three explicit orientations with the pole density of a model ODF.
  Each plot contains the cubic {100} and {111} families; antipodal=True identifies opposite poles.

    import numpy as np
    import matplotlib.pyplot as plt
    from mtex import crystalFrame, orientation, Miller, degree, plotPF, unimodalODF, mtexColorbar

    cs = crystalFrame('m-3m')
    h = [Miller(1, 0, 0, cs), Miller(1, 1, 1, cs)]
    ori = orientation.byEuler(np.array([0, 15, 30])*degree, 30*degree, 45*degree, cs)
    plotPF(ori, h, all=True, antipodal=True, markerSize=5)
    centre = orientation.byEuler(15*degree, 30*degree, 45*degree, cs)
    odf = unimodalODF(centre, halfwidth=15*degree)
    plotPF(odf, h, antipodal=True, contourf=True, resolution=5*degree, colorRange='equal')
    mtexColorbar(title='mrd')
    plt.show()

  The first figure shows discrete poles. The second shows a continuous density in multiples
  of a random distribution; it is the projection of the model ODF, not a fit to those three orientations.

  See also
    plotIPF, SO3Fun.radon, PoleFigure, calcPoleFigure, annotate, ODFPoleFigure,
    OrientationPoleFigure, PoleFigurePlot"""
  if isinstance(o, PoleFigure):
    return _plotPoleFigure(o, *args, **kw)
  o, points, all = _fibreAsCurve(o, points, all, kw)
  if not isinstance(o, orientation):
    return sphericalplot.plotPF(o, *args, superposition=superposition, **kw)
  args = _positionalStyle(o, args, kw)
  # plotPF(ori, 'add2all') without directions, or into held pole figures: the poles of the
  # directions they already show
  fig = sphericalplot.gcm()
  if (kw.get("add2all") or (fig is not None and fig.hold)) and not any(_isDirections(a) for a in args):
    kw["add2all"] = True
    return sphericalplot.plot(o, **kw)
  h, data = _pickData(o, args)
  hs = _asList(h)
  style = fillFlag(kw, "scatter")
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


def plotIPF(o, *args, points=None, all=False, **kw: Unpack[PlotKeywords]):
  """inverse pole figures, one per specimen direction: of an ODF or of a list of orientations;
  MATLAB's plotIPDF

    plotIPF(odf, [xvector, zvector])               # the inverse pole figures of an ODF
    plotIPF(odf, zvector, antipodal=True, complete=True)  # over the whole sphere
    plotIPF(ori, [xvector, zvector])               # the crystal directions as markers
    plotIPF(ori, ori.angle(), r)                   # the markers coloured by a value per orientation
    plotIPF(ori, r, points=100)                    # at most 100 orientations drawn
    plotIPF(ori, r, all=True)                      # every orientation drawn
    plotIPF(ori, r, contourf=True)                 # the density of the directions, filled contours

  An inverse pole figure shows which crystal directions point along the specimen direction r,
  all their equivalents folded into the fundamental sector of the crystal. What is drawn
  depends on the first argument:

  - an ODF: the density odf.radon(r=r), drawn with the keywords of a spherical function,
  contourf, contour, pcolor, colorRange, resolution, antipodal, complete.
  - a list of orientations: the crystal directions as markers, coloured by a value or an RGB
  colour per orientation when one is given; with contourf=True or another filled style, their
  density.
  - a fibre: the dots of 1000 orientations along it.

  A large list of orientations is thinned to a random selection of about 100000 markers
  divided by the order of the symmetries; points sets another number, all=True keeps every
  orientation (MATLAB's points='all').

  Input
    odf - SO3Fun, an ODF
    ori - orientation, or a fibre
    data - a value or an RGB colour per orientation, optional
    r - vector3d or a list of them, the specimen directions, one inverse pole figure each
  Keywords
    points - the number of orientations drawn at most, a random selection
    all - draw every orientation, however many
  See also
    plotPF, ipfColorKey, inversePoles, SO3Fun.radon, PlotTypes"""
  o, points, all = _fibreAsCurve(o, points, all, kw)
  if not isinstance(o, orientation):
    return sphericalplot.plotIPF(o, *args, **kw)
  r, data = _pickData(o, _positionalStyle(o, args, kw))
  rs = _asList(r)
  style = fillFlag(kw, "scatter")
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


def _plotPoleFigure(pf, colorRange="individual", bandwidth=None, **kw: Unpack[PlotKeywords]):
  """measured pole figures in a row, the specimen directions as markers coloured by intensity;
  MATLAB's PoleFigure/plot

    plot(pf)                                       # every pole figure of the data set
    plot(pf.select(0))                             # only the first pole figure
    pf.plot(contourf=True)                         # filled contours of the harmonic expansion
    pf.plot(contourf=True, bandwidth=16)           # of an expansion to degree 16
    pf.plot(minmax=True)                           # with the minimum and the maximum
    pf.plot(colorRange='equal')                    # one colour range over all pole figures

  With contourf, contour, pcolor or smooth (MATLAB's flags, keywords here) each pole figure is
  first expanded into spherical harmonics, S2FunHarmonic(pf.select(i)), and drawn as a
  function on the sphere. The other keywords are those of plot of vector3d; mtexColorbar adds
  the colour bars.

  Input
    pf - PoleFigure
  Keywords
    colorRange - 'individual', each pole figure its own range, 'equal', one range over all,
        or (lo, hi)
    bandwidth - the harmonic degree of the expansion a filled style draws; by default that of
        S2FunHarmonic
  See also
    plotPF, PoleFigure.plotDiff, mtexColorbar, PoleFigurePlot"""
  from ..functions.s2fun import S2FunHarmonic
  flag = fillFlag(kw, None)
  titles = [_pfTitle(pf, i) for i in range(pf.numPF)]
  if flag is not None:
    fs = [S2FunHarmonic(pf.select(i), bandwidth=bandwidth) for i in range(pf.numPF)]
    return _plotRow(fs, titles, style=flag, colorRange=colorRange, **kw)
  rs = [vector3d(pf.allR[i].data.reshape(-1, 3), frame=pf.allR[i].frame, antipodal=pf.allR[i].antipodal) for i in range(pf.numPF)]
  return _dataRow(rs, titles, [pf.allI[i].ravel() for i in range(pf.numPF)], colorRange=colorRange, **kw)


def plotDiff(pf, other, mode="relative", epsilon=None, **kw: Unpack[PlotKeywords]):
  """the difference between measured pole figures and recalculated ones, or two measurements,
  drawn as pole figures of the error; MATLAB's plotDiff

    plotDiff(pf, odf)                              # the damped relative error of a recalculation
    plotDiff(pf, pf2)                              # against other pole figures
    pf.plotDiff(odf, 'RP')                         # the RP error, where the recalculation exceeds 1
    pf.plotDiff(odf, mode='l1')                    # the absolute difference
    pf.plotDiff(odf, epsilon=1)                    # another damping
    plotDiff(odf, pf)                              # the same, with one colour range
    plotDiff(odf1, odf2)                           # the ODF sections of the difference of two ODFs

  The error is calcErrorPF(pf, other, mode, epsilon), drawn with a colour bar; MATLAB's flags
  'RP', 'l1', 'l2' are taken as keywords set to True too. Where neither argument is a
  PoleFigure the difference of the two functions is drawn by plotSection. The other keywords
  are those of plot(pf).

  Input
    pf - PoleFigure, the measurement
    other - SO3Fun, the ODF of the recalculation, or PoleFigure on the same directions
  Keywords
    mode - 'relative', 'RP', 'l1' or 'l2', as calcErrorPF
    epsilon - the damping of 'relative' (0.5) or the threshold of 'RP' (1)
  See also
    calcErrorPF, PoleFigure.calcError, PoleFigure.plot, plotSection"""
  from ..functions.polefigure import calcErrorPF
  # plotDiff(odf, pf): the same error, plotDiff(odf1, odf2) the sections of the difference
  if not isinstance(pf, PoleFigure):
    if not isinstance(other, PoleFigure):
      from .sectionplot import plotSection
      return plotSection(pf - other, **kw)
    pf, other = other, pf
    kw.setdefault("colorRange", "equal")
  for m in ("RP", "l1", "l2", "relative"):
    if kw.pop(m, False):
      mode = m
  d = calcErrorPF(pf, other, mode, epsilon)
  kw.setdefault("colorbar", True)
  return _plotPoleFigure(d, **kw)


__all__ = ["plotPF", "plotIPF", "plotDiff"]
