# the drawing of tensor quantities on the sphere (MATLAB @stiffnessTensor/plotSeismicVelocities,
# @S2Fun/plot with 'rgb'; docs/plans/plotting.md S1)
#
# The seismic overview: six charts of the wave velocities of a stiffness tensor with their
# extrema marked and the polarisations drawn, in MATLAB's layout. The rgb plot: three
# functions on the sphere as the colour channels of one surface, which is how the
# interference colours of a refractive index tensor are shown.
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..geometry.grids.s2grid import equispacedS2Grid
from ..geometry.session import getMTEXpref
from ..geometry.sphericalregion import sphericalRegion
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .colormaps import mtexColorMap
from .mtexfigure import hold, mtexColorbar, mtexFigure, mtexTitle, nextAxis
from .sphericalplot import _rgbMesh, _blocks, _split, _targets, _valuesOn, plot, quiver

if TYPE_CHECKING:
  from typing import Unpack

  from .keywords import PlotKeywords


# the largest and the smallest value of a function on the sphere and where they are taken,
# read off a fine grid: the markers of the overview need no more than that
def _extrema(f, resolution=1 * degree):
  g = equispacedS2Grid(resolution=resolution)
  V = vector3d(g.data.reshape(-1, 3), frame=getattr(f, "frame", None), isNormalized=True)
  vals = np.real(np.asarray(f(V), dtype=complex)).reshape(-1)
  hi, lo = int(np.nanargmax(vals)), int(np.nanargmin(vals))
  def at(k):
    return vector3d(V.data[k], frame=V.frame, antipodal=bool(getattr(f, "antipodal", False)))
  return (float(vals[hi]), at(hi)), (float(vals[lo]), at(lo))


# the percentage anisotropy MATLAB reports under every chart
def _anisotropy(hi, lo):
  return 200 * (hi - lo) / (hi + lo)


def plotSeismicVelocities(C, symmetriseMinMax=False, markerSize=10, lower=False, colormap="red2blue", density=None, **kw: Unpack[PlotKeywords]):
  """the overview of the seismic wave velocities of a stiffness tensor, six pole figures;
  MATLAB's stiffnessTensor/plotSeismicVelocities

    plotSeismicVelocities(C)                         # the overview, upper hemisphere
    plotSeismicVelocities(C, lower=True, symmetriseMinMax=True)  # lower, extrema with their equivalents
    plotSeismicVelocities(C, density=3.355)          # with a density not stored on C

  Six charts in two rows: Vp, the S-wave anisotropy 200 (Vs1 - Vs2) / (Vs1 + Vs2), Vs1 with
  the polarisation of the fast S-wave, Vs2 with that of the slow one, Vp/Vs1 and Vp/Vs2,
  each with its maximum as a black square and its minimum as a white circle, the
  anisotropy in percent under it, red slow to blue fast as seismology colours it. The
  velocities are in km/s for a stiffness in GPa and a density in g/cm^3.

  Input
    C - stiffnessTensor, with its density stored (C.density) or given
  Output
    fig - the mtexFigure of the six charts
  Keywords
    symmetriseMinMax - mark every crystal direction equivalent to the extrema
    markerSize - the size of the markers of the extrema
    lower - the lower hemisphere instead of the upper
    colormap - the colour map of the charts
    density - the density in g/cm^3; by default C.density
  See also
    velocity, stiffnessTensor, plotPF, WaveVelocities, CPOSeismicProperties"""
  from ..analysis.elasticity import velocity
  from ..functions.handles import S2FunHandle
  from ..functions.s2axisfield import S2AxisFieldHandle
  # the six charts read the same grids, so the Christoffel problem is solved once per grid
  cache = {}

  def vel(v):
    key = (v.data.shape, v.data.tobytes())
    if key not in cache:
      cache[key] = velocity(C, vector3d(v.data), density=density)
    return cache[key]

  def fun(g):
    return S2FunHandle(lambda v: np.asarray(g(vel(v))).reshape(v.shape), C.frame, antipodal=True, bandwidth=32)
  def field(k):
    return S2AxisFieldHandle(lambda v: vel(v)[k].data, C.frame, bandwidth=32)
  panels = [(fun(lambda w: w.vp), "Vp (km/s)", "Vp Anisotropy", None),
            (fun(lambda w: 200 * (w.vs1 - w.vs2) / (w.vs1 + w.vs2)), "S-wave anisotropy (%)", "Max Vs Anisotropy", None),
            (fun(lambda w: w.vs1), "Vs1 (km/s)", "Vs1 Anisotropy", field(4)),
            (fun(lambda w: w.vs2), "Vs2 (km/s)", "Vs2 Anisotropy", field(5)),
            (fun(lambda w: w.vp / w.vs1), "Vp/Vs1", "Vp/Vs1 Anisotropy", None),
            (fun(lambda w: w.vp / w.vs2), "Vp/Vs2", "Vp/Vs2 Anisotropy", None)]
  hemisphere = {"lower": True} if lower else {"upper": True}
  black = dict(marker="s", markerSize=markerSize, markerEdgeColor="white", markerFaceColor="black")
  white = dict(marker="o", markerSize=markerSize, markerEdgeColor="black", markerFaceColor="white")
  fig = mtexFigure(6, layout=(2, 3), figSize=kw.pop("figSize", (12, 7.5)))
  for k, (f, title, label, pol) in enumerate(panels):
    nextAxis(fig=fig)
    plot(f, contourf=True, complete=True, axisLabels=False, **hemisphere, **kw)
    (hi, pHi), (lo, pLo) = _extrema(f)
    hold(True)
    if pol is not None:
      quiver(pol, color="k", lineWidth=2, **hemisphere)
    for p, marker in ((pHi, black), (pLo, white)):
      plot(p.symmetrise() if symmetriseMinMax and p.isCrystalDirection else p, **marker, **hemisphere)
    hold(False)
    mtexTitle(title)
    value = hi if k == 1 else _anisotropy(hi, lo)
    # the chart hides its axes, so the label goes under it as text
    ax = fig.children[-1]
    ax.text(0.5, -0.03, f"{label} = {value:.1f}", transform=ax.transAxes, ha="center", va="top", fontsize=getMTEXpref("FontSize", 13) - 2)
  mtexColorMap(colormap, fig)
  mtexColorbar(fig)
  return fig


# ------------------------------------------------------------------ rgb

def plotRGB(channels, resolution=0.5 * degree, **kw: Unpack[PlotKeywords]):
  """plot((r, g, b), rgb=True): three functions on the sphere as the colour channels of one
  surface, their values clipped to [0, 1]"""
  r, g, b = channels
  projKw, axesKw, whereKw = _split(kw)
  projKw.setdefault("antipodal", bool(getattr(r, "antipodal", False)))
  # complete drops the symmetry sector, as for a function
  region = sphericalRegion(frame=getattr(r, "frame", None)) if projKw.get("complete", False) else r
  sPs = _targets(region, projKw, axesKw, whereKw)
  hs = []
  for sP in sPs:
    V, X, Y = sP.proj.grid(resolution)
    if V.size == 0:
      continue
    rgb = np.clip(np.stack([_valuesOn(f, V) for f in (r, g, b)], axis=-1), 0, 1)
    hs += [_rgbMesh(sP.ax, X[:, j0:j1], Y[:, j0:j1], rgb[:, j0:j1]) for j0, j1 in _blocks(X)]
  return hs


def isRGB(x):
  """a tuple or list of three functions on the sphere, the channels of an rgb plot"""
  from .sphericalplot import _isFunction
  return isinstance(x, (tuple, list)) and len(x) == 3 and all(_isFunction(f) for f in x)


__all__ = ["plotSeismicVelocities", "plotRGB"]
