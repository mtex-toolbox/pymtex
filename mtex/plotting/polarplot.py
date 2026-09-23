# polar plots (MATLAB @vector3d/histogram, @S1Fun/plot, @S2Fun/plotSection and quiverSection, @shape2d/plot,
# plottingConvention/setView)
#
#   histogram(v, numBin, weights=)       a rose diagram of directions in the plane, axes doubled
#   plot(sF1)                            a function on the circle as a polar line
#   plotSection(sF2, sec)                a function on the sphere along the great circle normal to sec,
#                                        seen down sec; polarplot=True as a polar line
#   quiverSection(sF2, vF, sec)          the vectors of a field at the ring of plotSection
#   plot(shape)                          a characteristic shape as its outline in polar coordinates
#   setView(pC, ax)                      the polar axes turned as the plotting convention says
#
# Every one of these is a curve over the angle in the plane of the map, drawn in a polar
# axes whose zero and sense of rotation follow the plotting convention, as MATLAB's
# setView does; a section is the curve in space seen down its normal, in a plain axes.
# matplotlib is imported inside the functions.
from __future__ import annotations

import numpy as np

from ..geometry.funcs import angle
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
  # a polar axes the user made with plt.subplot takes the plot, as MATLAB's subplot does
  import matplotlib.pyplot as plt
  if plt.get_fignums() and plt.gcf().axes:
    cur = plt.gca()
    if cur.name == "polar" and (fig is None or plt.gcf() is not fig.fig or cur not in fig.children):
      return cur
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
  """setView(pC), pC.setView(): a polar axes turned as the plotting convention says, a 3D
  axes looking down its out of screen direction"""
  pC = plottingConvention.default if pC is None else plottingConvention(pC) if not isinstance(pC, plottingConvention) else pC
  if ax is None:
    fig = gcm()
    ax = None if fig is None or not fig.children else fig.current if getattr(fig, "current", None) in fig.children else fig.children[-1]
  if ax is not None and ax.name == "3d":
    from .spaceplot import _camera
    _camera(ax, pC)
    return
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

def plotS1(sF, *args, lineWidth=1.0, DisplayName=None, color=None, notPolar=False, how2plot=None, **kw):
  """plot(sF1): a function on the circle as a polar line, or over the angle with notPolar;
  a positional MATLAB line style, plot(sF1, '--'), as matplotlib's format"""
  import matplotlib.pyplot as plt
  lineWidth = kw.pop("linewidth", kw.pop("LineWidth", lineWidth))
  DisplayName = kw.pop("displayName", DisplayName)
  notPolar = kw.pop("noPolar", notPolar)
  fmt = [a for a in args if isinstance(a, str)]
  omega = np.linspace(0, 2 * np.pi, 361)
  d = np.real(np.asarray(sF(omega), dtype=complex)).reshape(-1)
  fig = gcm()
  # an axes the user made with plt.subplot takes the line, as MATLAB's subplot does
  user = None
  if plt.get_fignums() and (fig is None or plt.gcf() is not fig.fig or (plt.gcf().axes and plt.gca() not in fig.children)):
    user = plt.gca() if plt.gcf().axes else None
  parent = kw.pop("parent", None) or user
  if notPolar:
    ax = parent or (fig.children[-1] if fig is not None and fig.hold and fig.children else mtexFigure(1).nextAxis())
    h = ax.plot(omega, d, *fmt, linewidth=lineWidth, label=DisplayName, color=color, **kw)
    ax.set_xlim(0, 2 * np.pi)
    return h[0]
  ax = polarAxes(parent if parent is not None and parent.name == "polar" else None)
  h = ax.plot(omega, d, *fmt, linewidth=lineWidth, label=DisplayName, color=color, **kw)
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


def plotSectionS2(sF, sec, eta=np.pi / 2, omega=None, lineWidth=1.0, DisplayName=None, color=None, how2plot=None, polarplot=False,
                  **kw):
  """plotSection(sF, sec): a function on the sphere along the circle normal to sec, the ring
  pushed out to the values and seen down sec; polarplot=True as a polar line over the angle"""
  omega, ring = _sectionCircle(sec, eta, omega)
  fr = getattr(sF, "frame", None)
  d = np.real(np.asarray(sF(vector3d(ring, frame=fr, isNormalized=True)), dtype=complex)).reshape(-1)
  lineWidth = kw.pop("linewidth", kw.pop("LineWidth", lineWidth))
  pC = getattr(sF, "how2plot", None) if how2plot is None else how2plot
  pC = plottingConvention(plottingConvention.default if pC is None else pC)
  if polarplot:
    ax = polarAxes(kw.pop("parent", None))
    x, y = omega, d
  else:
    ax = _sectionAxes(kw.pop("parent", None))
    # the camera looks down the normal of the section, as MATLAB turns the convention
    s = vector3d((sec.data if isinstance(sec, vector3d) else np.asarray(sec, dtype=float)).reshape(3))
    if angle(pC.outOfScreen, s, antipodal=True) > degree:
      pC.outOfScreen = s
    P = d[:, None] * ring
    x, y = P @ pC.east.normalize().data.reshape(3), P @ pC.north.normalize().data.reshape(3)
  # MATLAB's color 'interp': the line coloured by its values, a mappable for the colour bar
  if isinstance(color, str) and color == "interp":
    from types import SimpleNamespace

    from matplotlib.collections import LineCollection

    from .colormaps import mtexColormap
    pts = np.stack([x, y], axis=-1)
    h = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), linewidths=lineWidth, cmap=mtexColormap(kw.pop("colormap", None)),
                       label=DisplayName, capstyle="round", **kw)
    h.set_array((d[:-1] + d[1:]) / 2)
    ax.add_collection(h)
    ax._mtexLinePlot = SimpleNamespace(mappable=h)
    # the rings drawn into one axes share its colour range and map, as MATLAB's surfaces do:
    # a change to the one the figure knows reaches the others
    rings = ax.__dict__.setdefault("_mtexRings", [])
    rings.append(h)
    lo, hi = min(np.nanmin(c.get_array()) for c in rings), max(np.nanmax(c.get_array()) for c in rings)
    for c in rings:
      c.set_clim(lo, hi)
    h.callbacks.connect("changed", lambda m: _followRing(ax, m))
    ax.autoscale_view()
  else:
    h = ax.plot(x, y, linewidth=lineWidth, label=DisplayName, color=color, **kw)[0]
  if polarplot:
    ax.set_rmin(0)
    setView(pC, ax)
  return h


def _followRing(ax, m):
  if getattr(ax, "_mtexSyncing", False):
    return
  ax._mtexSyncing = True
  try:
    for c in ax._mtexRings:
      if c is not m:
        c.set_cmap(m.get_cmap())
        c.set_clim(*m.get_clim())
  finally:
    ax._mtexSyncing = False


def quiverSectionS2(sF, vF, sec, eta=np.pi / 2, color=None, lineWidth=0.5, autoScaleFactor=0.9, normalized=False,
                    DisplayName=None, how2plot=None, **kw):
  """quiverSection(sF, vF, sec): the vectors of vF along the circle normal to sec, drawn at
  the ring of plotSection(sF, sec) and seen down sec; axes as segments without heads"""
  kw.pop("doNotDraw", None)
  _, ring = _sectionCircle(sec, eta, np.linspace(0, 2 * np.pi, 36))
  fr = getattr(sF, "frame", None)
  S2 = vector3d(ring, frame=fr, isNormalized=True)
  v = vF(S2) if callable(vF) and not hasattr(vF, "eval") else vF.eval(S2)
  antipodal = bool(getattr(v, "antipodal", False))
  V = np.asarray(v.data if isinstance(v, vector3d) else v, dtype=float).reshape(-1, 3)
  if normalized:
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
  d = np.real(np.asarray(sF(S2), dtype=complex)).reshape(-1)
  P = d[:, None] * ring
  # MATLAB's quiver3 scale: the longest arrow autoScaleFactor times the mean spacing
  n = np.sqrt(len(P))
  spacing = np.sqrt(np.sum(((P.max(axis=0) - P.min(axis=0)) / n) ** 2))
  longest = np.max(np.linalg.norm(V, axis=1)) / spacing if spacing > 0 else 0
  V = V * (autoScaleFactor / longest if longest > 0 else autoScaleFactor)
  lineWidth = kw.pop("linewidth", kw.pop("LineWidth", lineWidth))
  pC = getattr(sF, "how2plot", None) if how2plot is None else how2plot
  pC = plottingConvention(plottingConvention.default if pC is None else pC)
  s = vector3d((sec.data if isinstance(sec, vector3d) else np.asarray(sec, dtype=float)).reshape(3))
  if angle(pC.outOfScreen, s, antipodal=True) > degree:
    pC.outOfScreen = s
  e, nth = pC.east.normalize().data.reshape(3), pC.north.normalize().data.reshape(3)
  ax = _sectionAxes(kw.pop("parent", None))
  x, y, u, w = P @ e, P @ nth, V @ e, V @ nth
  opt = dict(angles="xy", scale_units="xy", scale=1, color=color, width=lineWidth * 0.004, zorder=3, **kw)
  hs = []
  if antipodal:
    headless = dict(headwidth=0, headlength=0, headaxislength=0)
    hs.append(ax.quiver(x, y, -u, -w, label="_nolegend_", **opt, **headless))
    hs.append(ax.quiver(x, y, u, w, label=DisplayName, **opt, **headless))
  else:
    # short heads, MATLAB's MaxHeadSize of a fifth of the arrow
    hs.append(ax.quiver(x, y, u, w, label=DisplayName, headwidth=2.5, headlength=3, headaxislength=2.7, **opt))
  ax.autoscale_view()
  return hs


# a plain axes, equal in its units and without axis lines, for a curve seen in space
def _sectionAxes(parent=None):
  if parent is not None:
    return parent
  fig = gcm()
  if fig is not None and fig.hold and fig.children and getattr(fig.children[-1], "_mtexSection", False):
    return fig.children[-1]
  if fig is None or (fig.children and fig.pending is None) or fig.hold:
    fig = mtexFigure(1, figSize=(4.5, 4.5))
  ax = fig.takeAxis()
  ax._mtexSection = True
  ax.set_aspect("equal", adjustable="datalim")
  ax.set_axis_off()
  return ax


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
