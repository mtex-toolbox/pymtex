# polar plots (MATLAB @vector3d/histogram, @S1Fun/plot, @S2Fun/plotSection, @shape2d/plot,
# plottingConvention/setView)
#
#   histogram(v, numBin, weights=)       a rose diagram of directions in the plane, axes doubled
#   plot(sF1)                            a function on the circle as a polar line
#   plotSection(sF2, sec)                a function on the sphere along the great circle normal to sec
#   plot(shape)                          a characteristic shape as its outline in polar coordinates
#   setView(pC, ax)                      the polar axes turned as the plotting convention says
#
# Every one of these is a curve over the angle in the plane of the map, drawn in a polar
# axes whose zero and sense of rotation follow the plotting convention, as MATLAB's
# setView does. matplotlib is imported inside the functions.
from __future__ import annotations

import numpy as np

from ..geometry.plottingconvention import plottingConvention
from ..geometry.session import getMTEXpref
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .mtexfigure import gcm, mtexFigure


# a polar axes: the parent named, the held one, the one nextAxis prepared turned polar, a
# fresh figure's first axes, or a new figure
def polarAxes(parent=None):
  if parent is not None:
    return parent
  fig = gcm()
  if fig is not None and fig.hold and fig.children and fig.children[-1].name == "polar":
    return fig.children[-1]
  if fig is not None and fig.pending is not None:
    ax = fig.takeAxis()
    spec = ax.get_subplotspec()
    ax.remove()
    fig.children.remove(ax)
    ax = fig.fig.add_subplot(spec, projection="polar")
    fig.children.append(ax)
    return ax
  if fig is None or fig.children or fig.hold:
    fig = mtexFigure(1, figSize=(4.5, 4.5))
  ax = fig.fig.add_subplot(fig.nrows, fig.ncols, len(fig.children) + 1, projection="polar")
  fig.children.append(ax)
  return ax


# the zero of the angle where the convention's east lies and its sense from east to north
def setView(pC=None, ax=None):
  """setView(pC): a polar axes turned as the plotting convention says"""
  pC = plottingConvention.default if pC is None else plottingConvention(pC) if not isinstance(pC, plottingConvention) else pC
  if ax is None:
    fig = gcm()
    ax = fig.children[-1] if fig is not None and fig.children else None
  if ax is None or ax.name != "polar":
    return
  e = pC.east.data.reshape(3)
  n = pC.north.data.reshape(3)
  k = int(np.argmax(np.abs(e[:2])))
  ax.set_theta_zero_location({(0, 1): "E", (0, -1): "W", (1, 1): "N", (1, -1): "S"}[(k, int(np.sign(e[k])))])
  ax.set_theta_direction(1 if np.cross(e, n)[2] > 0 else -1)
  ax.grid(True)


# ------------------------------------------------------------------ rose diagrams

def histogram(v, numBin=None, weights=None, antipodal=None, **kw):
  """histogram(v), histogram(v, numBin, weights=w): the rose diagram of the azimuths of
  directions; an axis counts at its azimuth and opposite"""
  v = v if isinstance(v, vector3d) else vector3d(v)
  rho = np.asarray(v.rho, dtype=float).reshape(-1)
  w = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)
  if antipodal or (antipodal is None and v.antipodal):
    rho = np.concatenate([np.mod(rho, np.pi), np.mod(rho, np.pi) + np.pi])
    w = None if w is None else np.concatenate([w, w])
  numBin = 36 if numBin is None else int(numBin)
  edges = np.linspace(0, 2 * np.pi, numBin + 1)
  if w is not None:
    w = np.where(np.isfinite(w), w, 0.0)
  counts, _ = np.histogram(np.mod(rho, 2 * np.pi), bins=edges, weights=w)
  if w is not None and np.sum(w) > 0:
    counts = counts / np.sum(w)
  ax = polarAxes(kw.pop("parent", None))
  h = ax.bar(0.5 * (edges[:-1] + edges[1:]), counts, width=np.diff(edges), bottom=0, align="center",
             color=kw.pop("color", (0.0, 0.447, 0.741)), edgecolor=kw.pop("edgeColor", "k"), linewidth=0.5, alpha=kw.pop("alpha", 0.8), **kw)
  ax.set_yticklabels([])
  setView(kw.get("how2plot", v.how2plot), ax)
  return h


# ------------------------------------------------------------------ functions on the circle

def plotS1(sF, lineWidth=1.0, DisplayName=None, color=None, notPolar=False, how2plot=None, **kw):
  """plot(sF1): a function on the circle as a polar line, or over the angle with notPolar"""
  omega = np.linspace(0, 2 * np.pi, 361)
  d = np.real(np.asarray(sF(omega), dtype=complex)).reshape(-1)
  if notPolar:
    import matplotlib.pyplot as plt
    fig = gcm()
    ax = kw.pop("parent", None) or (fig.children[-1] if fig is not None and fig.hold and fig.children else mtexFigure(1).nextAxis())
    h = ax.plot(omega, d, linewidth=lineWidth, label=DisplayName, color=color, **kw)
    ax.set_xlim(0, 2 * np.pi)
    return h[0]
  ax = polarAxes(kw.pop("parent", None))
  h = ax.plot(omega, d, linewidth=lineWidth, label=DisplayName, color=color, **kw)
  ax.set_rmin(0)
  setView(how2plot, ax)
  return h[0]


# ------------------------------------------------------------------ sections of the sphere

# the great circle normal to sec, or the small circle at the angle eta from it, as the
# polar angle omega counted from an orthogonal of sec
def _sectionCircle(sec, eta=np.pi / 2, omega=None):
  s = (sec.data if isinstance(sec, vector3d) else np.asarray(sec, dtype=float)).reshape(3)
  s = s / np.linalg.norm(s)
  u = vector3d(s).orth().data.reshape(3)
  w = np.cross(s, u)
  omega = np.linspace(0, 2 * np.pi, 361) if omega is None else np.asarray(omega, dtype=float)
  ring = np.cos(eta) * s + np.sin(eta) * (np.cos(omega)[:, None] * u + np.sin(omega)[:, None] * w)
  return omega, ring


def plotSectionS2(sF, sec, eta=np.pi / 2, omega=None, lineWidth=1.0, DisplayName=None, color=None, how2plot=None, **kw):
  """plotSection(sF, sec): a function on the sphere along the circle normal to sec, as a
  polar line over the angle in that plane"""
  omega, ring = _sectionCircle(sec, eta, omega)
  fr = getattr(sF, "frame", None)
  d = np.real(np.asarray(sF(vector3d(ring, frame=fr, isNormalized=True)), dtype=complex)).reshape(-1)
  ax = polarAxes(kw.pop("parent", None))
  h = ax.plot(omega, d, linewidth=lineWidth, label=DisplayName, color=color, **kw)
  ax.set_rmin(0)
  pC = getattr(sF, "how2plot", None) if how2plot is None else how2plot
  setView(pC, ax)
  return h[0]


# ------------------------------------------------------------------ shapes

def plotShape(shape, lineWidth=1.0, DisplayName=None, color=None, normalize=False, plain=False, how2plot=None, **kw):
  """plot(shape): the outline of a characteristic shape in polar coordinates about its centre"""
  V = np.asarray(shape.V, dtype=float).reshape(-1, 2)
  V = V - V.mean(axis=0)
  if normalize:
    V = V / np.sqrt(abs(float(np.asarray(shape.area).reshape(-1)[0])))
  V = np.concatenate([V, V[:1]])
  theta = np.arctan2(V[:, 1], V[:, 0])
  rho = np.linalg.norm(V, axis=1)
  ax = polarAxes(kw.pop("parent", None))
  h = ax.plot(theta, rho, linewidth=lineWidth, label=DisplayName, color=color, **kw)
  ax.set_rmin(0)
  if plain:
    ax.grid(False)
    ax.set_yticks([])
  pC = getattr(shape.frame, "how2plot", None) if how2plot is None else how2plot
  setView(pC, ax)
  if plain:
    ax.grid(False)
  return h[0]


__all__ = ["setView", "polarAxes"]
