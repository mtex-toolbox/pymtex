# the matplotlib drawing of three dimensional plots (MATLAB @vector3d/plot3d, scatter3d,
# @S2Fun/plot3d, @orientationRegion/plot, @orientation/scatter; docs/plans/plotting.md S5)
#
# space.py says what to draw: the sphere as a mesh with its grid and the axes of the frame,
# the surface of an orientation region in axis angle space, the points orientations are.
# matplotlib's 3D axes is the camera; it is imported inside the functions.
from __future__ import annotations

import numpy as np

from ..geometry.orientation import orientation
from ..geometry.orientationregion import eulerRegion, fundamentalRegion, orientationRegion
from ..geometry.quaternion import quaternion
from ..geometry.tangent import SO3TangentVector
from ..geometry.representations import quat2euler, quatAngle, quatAxis
from ..geometry.session import getMTEXpref
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .colormaps import mtexColormap
from .mtexfigure import gcm, mtexFigure
from .space import axisAnglePoints, frameAxes, regionLines, regionSurface, regionWalls, sphereGridLines, sphereMesh
from .sphericalplot import MARKER_COLOR, _isField, _isFunction, _isAxisField, _rangeOf

SPHERE_COLOR = (0.93, 0.93, 0.93)
GRID3_COLOR = (0.75, 0.75, 0.75)


class spacePlot:
  """a plot in space: the 3D axes, the frame it is drawn in, the mappable of its colours"""

  def __init__(self, ax, frame=None):
    self.ax, self.frame, self.mappable = ax, frame, None
    ax._mtexSpacePlot = self


# a 3D axes in a new figure, or the current one when it is held, looking at the unit sphere
# or at a shape of the given extent
def _spaceAxes(frame, limits=1.3, decorated=False, figSize=None, parent=None, azimuth=None, elevation=None, title=None):
  if parent is not None:
    ax = parent
  else:
    fig = gcm()
    if fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexSpacePlot"):
      return fig.children[-1]._mtexSpacePlot
    from .mtexfigure import figureSize
    fig = mtexFigure(1, figSize=figureSize(figSize, (5.5, 5.5)))
    # the markers are drawn after the region they lie in, whatever matplotlib makes of
    # the depth of one big translucent surface
    ax = fig.fig.add_subplot(1, 1, 1, projection="3d", computed_zorder=False)
    fig.children.append(ax)
  sP = spacePlot(ax, frame)
  ax.set_box_aspect((1, 1, 1))
  lim = (-limits, limits) if np.isscalar(limits) else limits
  ax.set_xlim(*lim)
  ax.set_ylim(*lim)
  ax.set_zlim(*lim)
  if not decorated:
    ax.set_axis_off()
  ax.view_init(elev=22.5 if elevation is None else elevation / degree, azim=-130 if azimuth is None else azimuth / degree)
  if title is not None:
    ax.set_title(title, fontsize=getMTEXpref("FontSize", 13))
  return sP


# the sphere with its grid and the axes of the frame, the ground every plot in space stands
# on; with values the sphere is coloured at its faces, with radial pushed out by them
def _sphere(sP, resolution=5 * degree, values=None, colormap="WhiteJet", colorRange=None, radial=False, grid=True, noLabel=False,
            grid_res=30 * degree, alpha=1.0):
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  V, F = sphereMesh(resolution)
  R = np.abs(values)[:, None] * V if radial else V
  poly = Poly3DCollection(R[F], linewidths=0.2, alpha=alpha)
  if values is None:
    poly.set_facecolor(SPHERE_COLOR)
  else:
    lo, hi = _rangeOf([values]) if colorRange is None else colorRange
    from matplotlib.colors import Normalize
    norm = Normalize(lo, hi)
    cmap = mtexColormap(colormap)
    faceValues = np.mean(values[F], axis=1)
    poly.set_facecolor(cmap(norm(faceValues)))
    poly.set_cmap(cmap)
    poly.set_norm(norm)
    poly.set_array(faceValues)
    sP.mappable = poly
  sP.ax.add_collection3d(poly)
  # the faces take their own colour as their edge, so the seams between them close
  poly.set_edgecolor("face")
  if grid:
    for l in sphereGridLines(grid_res):
      sP.ax.plot(1.003 * l[:, 0], 1.003 * l[:, 1], 1.003 * l[:, 2], color=GRID3_COLOR, linewidth=0.6)
  if not noLabel:
    for d, name in frameAxes(sP.frame):
      sP.ax.text(*(1.15 * d), name, ha="center", va="center", fontsize=getMTEXpref("FontSize", 13))
  return poly


def _keys(kw):
  axesKw = {k: kw.pop(k) for k in ("figSize", "parent", "azimuth", "elevation", "title") if k in kw}
  return axesKw


def plot3d(x, *args, **kw):
  """plot3d(v), plot3d(v, data), plot3d(sF), plot3d(vF), plot3d(oR), plot3d(ori)

  Directions as markers on the sphere in space, a function as the sphere coloured, a field
  as arrows on it, the fundamental region of orientation space as its boundary in axis angle
  space, and a list of orientations as points in it.
  """
  if isinstance(x, orientationRegion):
    return plotRegion3d(x, **kw)
  if isinstance(x, SO3TangentVector):
    return quiver3(x, *args, **kw)
  if isinstance(x, quaternion):
    return scatterOrientations(x, *args, **kw)
  from ..geometry.fibre import fibre
  if isinstance(x, fibre):
    return plotFibre3d(x, *args, **kw)
  if _isField(x):
    return quiver3d(x, **kw)
  if _isFunction(x):
    return plotFunction3d(x, **kw)
  return scatter3d(x, *args, **kw)


def scatter3d(v, data=None, MarkerSize=6, color=MARKER_COLOR, colormap="WhiteJet", colorRange=None, grid=True, noLabel=False,
              label=None, colorbar=False, **kw):
  """scatter3d(v), plot3d(v, data): directions as markers on the sphere in space"""
  v = v if isinstance(v, vector3d) else vector3d(v)
  axesKw = _keys(kw)
  sP = _spaceAxes(v.frame, **axesKw)
  if not sP.ax.collections:
    _sphere(sP, grid=grid, noLabel=noLabel)
  P = 1.01 * v.normalize().data.reshape(-1, 3)
  if v.antipodal:
    P = np.concatenate([P, -P])
  if data is None:
    h = sP.ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=MarkerSize ** 2, c=[color] if isinstance(color, tuple) else color, edgecolors="k",
                      linewidths=0.4, depthshade=False, **kw)
  else:
    vals = np.asarray(data, dtype=float).reshape(-1)
    vals = np.concatenate([vals, vals]) if v.antipodal else vals
    lo, hi = _rangeOf([vals]) if colorRange is None else colorRange
    h = sP.ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=MarkerSize ** 2, c=vals, cmap=mtexColormap(colormap), vmin=lo, vmax=hi,
                      edgecolors="k", linewidths=0.4, depthshade=False, **kw)
    sP.mappable = h
    if colorbar:
      sP.ax.figure.colorbar(h, ax=sP.ax, shrink=0.6)
  if label is not None:
    texts = [label] * len(P) if isinstance(label, str) else list(label)
    for p, t in zip(P, texts):
      sP.ax.text(*(1.08 * p), str(t), fontsize=getMTEXpref("FontSize", 13) - 2)
  return h


def plotFunction3d(f, resolution=5 * degree, colormap="WhiteJet", colorRange=None, colorbar=False, radial=False, grid=None,
                   noLabel=None, grid_res=30 * degree, **kw):
  """plot3d(sF, radial=True): a function as the sphere coloured, or pushed out by its values"""
  axesKw = _keys(kw)
  V, _ = sphereMesh(resolution)
  fr = getattr(f, "frame", None)
  vals = np.real(np.asarray(f(vector3d(V, frame=fr, isNormalized=True)), dtype=complex)).reshape(-1)
  limits = max(1e-12, 1.1 * np.max(np.abs(vals))) if radial else 1.3
  sP = _spaceAxes(fr, limits=limits, **axesKw)
  h = _sphere(sP, resolution, vals, colormap, colorRange, radial, grid=(not radial) if grid is None else grid,
              noLabel=radial if noLabel is None else noLabel, grid_res=grid_res)
  if colorbar:
    sP.ax.figure.colorbar(h, ax=sP.ax, shrink=0.6)
  return h


def quiver3d(vF, resolution=15 * degree, color="k", arrowSize=0.75, grid=True, noLabel=False, **kw):
  """plot3d(vF): a vector or axis field as arrows on the sphere in space"""
  axesKw = _keys(kw)
  fr = getattr(vF, "frame", None)
  sP = _spaceAxes(fr, **axesKw)
  _sphere(sP, grid=grid, noLabel=noLabel)
  V, _ = sphereMesh(resolution)
  W = vF(vector3d(V, frame=fr, isNormalized=True)).data.reshape(-1, 3)
  # the longest arrow spans the grid spacing, the rest in proportion
  scale = 0.9 * resolution * arrowSize / max(np.max(np.linalg.norm(W, axis=1)), 1e-12)
  D = scale * W
  P = 1.01 * V
  if _isAxisField(vF):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    h = Line3DCollection(np.stack([P - D / 2, P + D / 2], axis=1), colors=color, linewidths=1.0)
    sP.ax.add_collection3d(h)
    return h
  return sP.ax.quiver(P[:, 0], P[:, 1], P[:, 2], D[:, 0], D[:, 1], D[:, 2], color=color, arrow_length_ratio=0.3, linewidths=1.0, **kw)


# ------------------------------------------------------------------ orientation space

def _regionLabels(ax):
  ax.set_xlabel("x / °")
  ax.set_ylabel("y / °")
  ax.set_zlabel("z / °")


# the surface of a fundamental region: light grey, so that the markers inside it show
REGION_COLOR = (0.8, 0.8, 0.8)


# the ball of the whole rotation space in axis angle space, the half turns on its surface,
# as a light wire sphere of radius 180 degrees
def _ball(sP, radius=180.0):
  u, w = np.meshgrid(np.linspace(0, 2 * np.pi, 37), np.linspace(0, np.pi, 19))
  X, Y, Z = radius * np.cos(u) * np.sin(w), radius * np.sin(u) * np.sin(w), radius * np.cos(w)
  sP.ax.plot_surface(X, Y, Z, color=SPHERE_COLOR, alpha=0.05, linewidth=0, shade=False)
  sP.ax.plot_wireframe(X, Y, Z, color=GRID3_COLOR, linewidth=0.4)


def plotRegion3d(oR, resolution=5 * degree, color=REGION_COLOR, alpha=0.25, lines=True, lineColor="k", **kw):
  """plot(oR): the fundamental region in axis angle space, in degrees, its surface with the
  parallels and meridians on it"""
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  axesKw = _keys(kw)
  limit = 1.15 * oR.maxAngle() / degree
  sP = _spaceAxes(oR.frameA, limits=limit, decorated=True, **axesKw)
  _regionLabels(sP.ax)
  return _region(sP, oR, resolution, color, alpha, lines, lineColor)


def _region(sP, oR, resolution=5 * degree, color=REGION_COLOR, alpha=0.25, lines=True, lineColor="k"):
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  P, F = regionSurface(oR, resolution)
  P = P / degree
  keep = np.all(np.isfinite(P[F]).reshape(len(F), -1), axis=1)
  poly = Poly3DCollection(P[F[keep]], facecolors=color, alpha=alpha, linewidths=0, zorder=1)
  sP.ax.add_collection3d(poly)
  for w in regionWalls(oR):
    pts = w[np.all(np.isfinite(w), axis=1)] / degree
    if len(pts) >= 3:
      sP.ax.add_collection3d(Poly3DCollection([pts], facecolors=color, alpha=alpha, linewidths=0, zorder=1))
  if lines:
    for l in regionLines(oR):
      q = l / degree
      sP.ax.plot(q[:, 0], q[:, 1], q[:, 2], color=lineColor, alpha=0.5, linewidth=0.6, zorder=2)
  return poly


# the points orientations are, in the coordinates asked for: axis angle and Euler angles in
# degrees, Rodrigues vectors as they are
def orientationPoints(o, coordinates="axisAngle", fundamentalRegion=True):
  if coordinates == "Euler":
    if fundamentalRegion:
      a, b, c = o.flatten().project2EulerFR()
      return np.stack([a, b, c], axis=1) / degree
    a, b, c = quat2euler(o.data.reshape(-1, 4), "Bunge")
    return np.stack([a, b, c], axis=1) / degree
  q = (o.flatten().project2FundamentalRegion() if fundamentalRegion else o.flatten()).data.reshape(-1, 4)
  if coordinates == "axisAngle":
    omega = quatAngle(q)
    return np.where(omega[:, None] < 1e-12, 0.0, omega[:, None] * quatAxis(q)) / degree
  if coordinates == "Rodrigues":
    return q[:, 1:] / np.where(np.abs(q[:, :1]) < 1e-12, 1e-12, q[:, :1])
  raise ValueError("coordinates are 'axisAngle', 'Euler' or 'Rodrigues'")


# MATLAB's marker options on a scatter in space
def _markerStyle(kw, color, MarkerSize):
  for k in ("MarkerFaceColor", "MarkerColor", "markerfacecolor"):
    if k in kw:
      color = kw.pop(k)
  edge = kw.pop("MarkerEdgeColor", kw.pop("markeredgecolor", None))
  lw = kw.pop("linewidth", kw.pop("LineWidth", None))
  MarkerSize = kw.pop("markersize", MarkerSize)
  style = {"s": MarkerSize ** 2, "depthshade": False, "zorder": 5}
  name = kw.pop("DisplayName", kw.pop("displayName", None))
  if name is not None:
    style["label"] = name
  if edge is not None:
    style["edgecolors"] = edge
    style["linewidths"] = 1.0 if lw is None else lw
  return color, style


# the flags of the coordinates of orientation space, as MATLAB names them
def _spaceFlags(args, Euler, axisAngle, Rodrigues):
  data, region, complete = None, None, False
  for a in args:
    if isinstance(a, str):
      flags = {"euler": "Euler", "bunge": "Euler", "axisangle": "axisAngle", "rodrigues": "Rodrigues"}
      if a.lower() in flags:
        Euler, axisAngle, Rodrigues = (flags[a.lower()] == x for x in ("Euler", "axisAngle", "Rodrigues"))
      elif a.lower() == "noboundary":
        region = False
      elif a.lower() == "complete":
        complete = True
      elif a.lower() != "filled":
        raise ValueError("the space is 'axisAngle', 'Euler', 'Bunge' or 'Rodrigues'")
    else:
      data = a
  return data, Euler, axisAngle, Rodrigues, region, complete


def scatterOrientations(o, *args, axisAngle=False, Rodrigues=False, Euler=False, region=True, fundamentalRegion=True,
                        ignoreFundamentalRegion=False, points=None, MarkerSize=5, color=MARKER_COLOR, colormap="WhiteJet",
                        colorRange=None, colorbar=False, **kw):
  """scatter(ori), plot3d(ori), scatter(ori, 'Rodrigues'), scatter(ori, 'Euler'), scatter(ori, data, 'axisAngle')

  Orientations as points in axis angle space, in Rodrigues space or in Euler angle space,
  brought into the fundamental region unless ignoreFundamentalRegion, with the boundary of
  the region behind them in axis angle space unless 'noBoundary'; the space named by a
  positional string as MATLAB names it, 'Bunge' being 'Euler'. A bare rotation has no
  region: the whole Euler box, the ball of axis angle space.
  """
  data, Euler, axisAngle, Rodrigues, noRegion, complete = _spaceFlags(args, Euler, axisAngle, Rodrigues)
  region = region and noRegion is None and not complete
  ignoreFundamentalRegion = ignoreFundamentalRegion or complete
  color, style = _markerStyle(kw, color, MarkerSize)
  axesKw = _keys(kw)
  # an orientation with its frames scatters in the Euler region by default, a bare rotation
  # in axis angle space, as MATLAB does; a list is sampled to 2000 points unless asked otherwise
  isOri = hasattr(o, "CS")
  euler = Euler or (not axisAngle and not Rodrigues and isOri)
  coordinates = "Rodrigues" if Rodrigues else "Euler" if euler else "axisAngle"
  reduced = isOri and fundamentalRegion and not ignoreFundamentalRegion
  region = region and isOri
  o = o.flatten()
  # an orientation that is not there, a NaN, is left out with its value
  keep = ~np.isnan(o.data).any(axis=-1)
  if not np.all(keep):
    o = o[keep]
    data = None if data is None else np.asarray(data).reshape(-1)[keep]
  points = 2000 if points is None else points
  if points < o.size:
    print(f"plot {points} random orientations out of {o.size} given orientations")
    take = np.sort(np.random.default_rng(0).permutation(o.size)[:points])
    o = o[take]
    data = None if data is None else np.asarray(data).reshape(-1)[take]
  P = orientationPoints(o, coordinates, reduced)
  oR = fundamentalRegion_(o) if isOri else None
  frame = o.CS if isOri else None
  fig = gcm()
  held = fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexSpacePlot")
  if coordinates == "Euler":
    bounds = np.array(eulerRegion(o.CS, o.SS) if reduced else (2 * np.pi, np.pi, 2 * np.pi)) / degree
    sP = _spaceAxes(frame, limits=(0, float(bounds.max())), decorated=True, **axesKw)
    # held axes keep their box
    if not held:
      sP.ax.set_xlim(0, bounds[0])
      sP.ax.set_ylim(0, bounds[1])
      sP.ax.set_zlim(0, bounds[2])
      sP.ax.set_box_aspect(tuple(bounds / bounds.max()))
      sP.ax.set_xlabel("φ1 / °")
      sP.ax.set_ylabel("Φ / °")
      sP.ax.set_zlabel("φ2 / °")
  else:
    finite = P[np.all(np.isfinite(P), axis=1)]
    bound = float(np.max(np.abs(finite))) if len(finite) else 1.0
    if coordinates == "axisAngle" and (region or not reduced):
      bound = max(bound, oR.maxAngle() / degree if region else 180.0)
    sP = _spaceAxes(frame, limits=1.1 * bound, decorated=True, **axesKw)
    if coordinates == "axisAngle":
      _regionLabels(sP.ax)
      if region:
        _region(sP, oR)
      elif not reduced and noRegion is None:
        _ball(sP)
    else:
      sP.ax.set_xlabel("r1")
      sP.ax.set_ylabel("r2")
      sP.ax.set_zlabel("r3")
  # the coordinates of the axes, for what is drawn into them later
  sP.coordinates, sP.reduced = coordinates, reduced
  if data is None:
    h = sP.ax.scatter(P[:, 0], P[:, 1], P[:, 2], color=color, **style)
  else:
    vals = np.asarray(data, dtype=float).reshape(-1)
    lo, hi = _rangeOf([vals]) if colorRange is None else colorRange
    h = sP.ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=vals, cmap=mtexColormap(colormap), vmin=lo, vmax=hi, **style)
    sP.mappable = h
    if colorbar:
      sP.ax.figure.colorbar(h, ax=sP.ax, shrink=0.6)
  return h


# quiver3(t): a tangent vector as an arrow at its reference in orientation space, in the
# coordinates of the held axes or axis angle space; the arrow ends where the vector, scaled
# to the longest of the list, points to, as MATLAB draws it
def quiver3(t, *args, lineWidth=1.5, color="b", normalize=False, axisAngle=False, Rodrigues=False, Euler=False, maxHeadSize=None, **kw):
  _, Euler, axisAngle, Rodrigues, _, _ = _spaceFlags(args, Euler, axisAngle, Rodrigues)
  lineWidth = kw.pop("LineWidth", kw.pop("linewidth", lineWidth))
  axesKw = _keys(kw)
  ref = t.oriRef
  isOri = hasattr(ref, "CS")
  fig = gcm()
  held = fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexSpacePlot")
  if held:
    sP = fig.children[-1]._mtexSpacePlot
    coordinates, reduced = getattr(sP, "coordinates", "axisAngle"), getattr(sP, "reduced", isOri)
  else:
    coordinates = "Rodrigues" if Rodrigues else "Euler" if Euler else "axisAngle"
    reduced = isOri
    o = ref if ref.ndim else ref.reshape(1)
    scatterOrientations(o, coordinates, MarkerSize=0.1, **axesKw)
    sP = gcm().children[-1]._mtexSpacePlot
  scale = 1.0 if normalize else float(np.max(t.norm()))
  t = t.normalize() if normalize else t / max(scale, 1e-12)
  P1 = orientationPoints(t.oriRef, coordinates, reduced)
  P2 = orientationPoints(t.exp(), coordinates, reduced)
  D = P2 - P1
  return sP.ax.quiver(P1[:, 0], P1[:, 1], P1[:, 2], D[:, 0], D[:, 1], D[:, 2], color=color, linewidths=lineWidth, arrow_length_ratio=0.25, **kw)


# arrow3d(v): solid arrows from the origin, or from anchor, to the tips v, into the held
# axes or a new figure in space, the axes widened to hold them (MATLAB @vector3d/arrow3d)
def arrow3d(v, faceColor="k", anchor=None, arrowWidth=None, edgeColor="none", labeled=False, label=None, **kw):
  from matplotlib.colors import to_rgba
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  from .sphericalplot import _nameOf
  v = v if isinstance(v, vector3d) else vector3d(v)
  faceColor = kw.pop("FaceColor", faceColor)
  if labeled and label is None:
    label = [_nameOf(v.flatten()[i]) for i in range(v.size)]
  axesKw = _keys(kw)
  fig = gcm()
  held = fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexSpacePlot")
  sP = _spaceAxes(v.frame, **axesKw)
  D = v.data.reshape(-1, 3)
  base = np.zeros_like(D) if anchor is None else np.broadcast_to((anchor.data if isinstance(anchor, vector3d) else np.asarray(anchor, dtype=float)).reshape(-1, 3), D.shape)
  scale = max(float(np.max(np.linalg.norm(D, axis=1))), 1e-12)
  r = 0.035 * scale if arrowWidth is None else float(arrowWidth)
  t = np.linspace(0, 2 * np.pi, 25)[:-1]
  faces = []
  for b, d in zip(base, D):
    L = float(np.linalg.norm(d))
    if L < 1e-12:
      continue
    n = d / L
    u = np.cross(n, [0, 0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    ring = np.cos(t)[:, None] * u + np.sin(t)[:, None] * w
    head = min(0.2 * scale, 0.5 * L)
    shaftEnd = b + n * (L - head)
    tip = b + d
    c0, c1, c2 = b + r * ring, shaftEnd + r * ring, shaftEnd + 2.2 * r * ring
    k = np.arange(len(t))
    j = (k + 1) % len(t)
    faces += [[c0[i], c0[j[i]], c1[j[i]], c1[i]] for i in k]
    faces += [[c1[i], c1[j[i]], c2[j[i]], c2[i]] for i in k]
    faces += [[c2[i], c2[j[i]], tip] for i in k]
    faces.append(list(c0[::-1]))
  poly = Poly3DCollection(faces, facecolors=to_rgba(faceColor), shade=True, **kw)
  if edgeColor != "none":
    poly.set_edgecolor(edgeColor)
  sP.ax.add_collection3d(poly)
  if label is not None:
    texts = [label] * len(D) if isinstance(label, str) else list(label)
    for b, d, t in zip(base, D, texts):
      tip = b + 1.12 * d
      sP.ax.text(tip[0], tip[1], tip[2], str(t), fontsize=getMTEXpref("FontSize", 13), ha="center", va="center")
  # the axes fit every arrow drawn so far, at equal scale, as MATLAB's do
  pts = np.concatenate([base, base + D])
  lo, hi = pts.min(axis=0) - 0.1 * scale, pts.max(axis=0) + 0.1 * scale
  if held:
    cur = np.array([sP.ax.get_xlim(), sP.ax.get_ylim(), sP.ax.get_zlim()])
    lo, hi = np.minimum(lo, cur[:, 0]), np.maximum(hi, cur[:, 1])
  sP.ax.set_xlim(lo[0], hi[0])
  sP.ax.set_ylim(lo[1], hi[1])
  sP.ax.set_zlim(lo[2], hi[2])
  sP.ax.set_box_aspect(tuple(hi - lo), zoom=1.3)
  return poly


# plot(f): a fibre as its curve through orientation space, in the coordinates of the held
# axes or Euler angles, as MATLAB draws it
def plotFibre3d(f, *args, lineWidth=1.5, color="k", axisAngle=False, Rodrigues=False, Euler=False, fundamentalRegion=True,
                ignoreFundamentalRegion=False, region=True, points=1000, **kw):
  _, Euler, axisAngle, Rodrigues, noRegion, complete = _spaceFlags(args, Euler, axisAngle, Rodrigues)
  color = kw.pop("lineColor", kw.pop("LineColor", color))
  lineWidth = kw.pop("LineWidth", kw.pop("linewidth", lineWidth))
  coordinates = "Rodrigues" if Rodrigues else "axisAngle" if axisAngle else "Euler"
  o = f.flatten().orientation(points) if f.ndim > 0 else f.orientation(points)
  reduced = fundamentalRegion and not ignoreFundamentalRegion and not complete
  P = orientationPoints(o, coordinates, reduced)
  axesKw = _keys(kw)
  fig = gcm()
  held = fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexSpacePlot")
  if coordinates == "Euler" or held:
    sP = _spaceAxes(o.CS, limits=(0, float(np.nanmax(np.abs(P)))), decorated=True, **axesKw)
    if not held:
      bounds = np.array(eulerRegion(o.CS, o.SS) if reduced else (2 * np.pi, np.pi, 2 * np.pi)) / degree
      sP.ax.set_xlim(0, bounds[0])
      sP.ax.set_ylim(0, bounds[1])
      sP.ax.set_zlim(0, bounds[2])
      sP.ax.set_box_aspect(tuple(bounds / bounds.max()))
      sP.ax.set_xlabel("φ1 / °")
      sP.ax.set_ylabel("Φ / °")
      sP.ax.set_zlabel("φ2 / °")
  else:
    # the ball of axis angle space, or the fundamental region with its boundary
    oR = fundamentalRegion_(o)
    bound = float(np.nanmax(np.abs(P)))
    if coordinates == "axisAngle":
      bound = max(bound, (oR.maxAngle() if reduced else np.pi) / degree)
    sP = _spaceAxes(o.CS, limits=1.1 * bound, decorated=True, **axesKw)
    if coordinates == "axisAngle":
      _regionLabels(sP.ax)
      if region and noRegion is None:
        _region(sP, oR) if reduced else _ball(sP)
    else:
      sP.ax.set_xlabel("r1")
      sP.ax.set_ylabel("r2")
      sP.ax.set_zlabel("r3")
  sP.coordinates, sP.reduced = coordinates, reduced
  # a curve that wraps through the region is drawn without the jumps
  jump = np.linalg.norm(np.diff(P, axis=0), axis=1) > 5 * np.nanmedian(np.linalg.norm(np.diff(P, axis=0), axis=1))
  Q = P.copy()
  Q[1:][jump] = np.nan
  return sP.ax.plot(Q[:, 0], Q[:, 1], Q[:, 2], linewidth=lineWidth, color=color, **kw)[0]


def fundamentalRegion_(o):
  return fundamentalRegion(o.CS, o.SS, antipodal=o.antipodal)


__all__ = ["plot3d", "scatter3d", "spacePlot", "arrow3d", "quiver3"]
