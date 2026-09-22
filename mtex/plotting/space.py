# three dimensional plots, the geometry without graphics (MATLAB @vector3d/plot3d,
# @orientationRegion/plot, @orientation/scatter; docs/plans/plotting.md S5)
#
# The sphere in space as a mesh, the grid and the axes a frame names; orientation space as
# axis angle space, where the fundamental region is star shaped about the identity, so its
# surface is the sphere mesh pushed out to the largest angle about every axis.
from __future__ import annotations

import numpy as np

from ..geometry.frames import crystalFrame, specimenFrame
from ..geometry.representations import quatAngle, quatAxis
from ..geometry.units import degree
from ..geometry.vector3d import vector3d


# the unit sphere as a mesh: rings of constant polar angle joined by quads split into
# triangles, fans at the poles, every triangle wound so that its normal points outwards;
# the vertices as an (n, 3) array and the faces as (m, 3) index triples
def sphereMesh(resolution=5 * degree):
  nTheta = max(2, int(round(np.pi / resolution)))
  nRho = max(3, int(round(2 * np.pi / resolution)))
  theta = np.pi * np.arange(1, nTheta)[:, None] / nTheta
  rho = 2 * np.pi * np.arange(nRho)[None, :] / nRho
  ring = np.stack([np.sin(theta) * np.cos(rho), np.sin(theta) * np.sin(rho), np.cos(theta) * np.ones_like(rho)], axis=-1).reshape(-1, 3)
  V = np.concatenate([[[0.0, 0.0, 1.0]], ring, [[0.0, 0.0, -1.0]]])
  south = len(V) - 1
  at = lambda i, j: 1 + i * nRho + np.mod(j, nRho)
  j = np.arange(nRho)
  faces = [np.stack([np.zeros(nRho, dtype=int), at(0, j), at(0, j + 1)], axis=1),
           np.stack([np.full(nRho, south), at(nTheta - 2, j + 1), at(nTheta - 2, j)], axis=1)]
  for i in range(nTheta - 2):
    a, b, c, d = at(i, j), at(i, j + 1), at(i + 1, j + 1), at(i + 1, j)
    faces.append(np.stack([a, d, c], axis=1))
    faces.append(np.stack([a, c, b], axis=1))
  return V, np.concatenate(faces).astype(int)


# the circle of directions at an angle about a direction, closed; the plane at 90 degrees
def smallCircle(n, opening=90 * degree, points=361):
  v = (n.data if isinstance(n, vector3d) else np.asarray(n, dtype=float)).reshape(3)
  v = v / np.linalg.norm(v)
  u = vector3d(v).orth().data.reshape(3)
  w = np.cross(v, u)
  t = np.linspace(0, 2 * np.pi, points)
  return np.cos(opening) * v + np.sin(opening) * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * w)


# the small circles about the pole and the meridians, as polylines of directions
def sphereGridLines(step=30 * degree, points=91):
  lines = [smallCircle([0, 0, 1], theta, points) for theta in np.arange(step, np.pi - step + 1e-9, step)]
  t = np.linspace(0, np.pi, points // 2 + 1)
  for rho in np.arange(0, 2 * np.pi - step + 1e-9, step):
    lines.append(np.stack([np.sin(t) * np.cos(rho), np.sin(t) * np.sin(rho), np.cos(t)], axis=-1))
  return lines


# the axes a frame names, as unit directions with their labels: X, Y, Z of a specimen frame,
# the lattice axes a, b, c of a crystal frame
def frameAxes(fr):
  if isinstance(fr, crystalFrame):
    axes = [np.asarray(fr.sym.axes[k], dtype=float) for k in range(3)]
    return [(a / np.linalg.norm(a), n) for a, n in zip(axes, fr.axesNames)]
  names = fr.axesNames if fr is not None else ("x", "y", "z")
  return [(np.eye(3)[k], names[k]) for k in range(3)]


# ------------------------------------------------------------------ orientation space

# the orientations as points of axis angle space, each brought into the fundamental region
# of its frames first: axis times angle, in radians, shape (n, 3)
def axisAnglePoints(o):
  q = o.flatten().project2FundamentalRegion().data.reshape(-1, 4)
  omega = quatAngle(q)
  return np.where(omega[:, None] < 1e-12, 0.0, omega[:, None] * quatAxis(q))


# the largest rotation angle inside the region about every axis of an (n, 3) array, zero
# where the axis lies outside the sector the region is cut down to: the bound
# dot(q, N_i) <= 0 at q = (cos(w/2), sin(w/2) v) reads tan(w/2) dot(v, a_i) <= tan(w_i/4)
def maxAngles(oR, V):
  V = np.asarray(V, dtype=float).reshape(-1, 3)
  V = V / np.linalg.norm(V, axis=1, keepdims=True)
  omega = np.full(len(V), np.pi)
  for n in oR.N:
    a = n[1:]
    s = np.linalg.norm(a)
    c = V @ a / s
    ok = c > 1e-9
    omega[ok] = np.minimum(omega[ok], 2 * np.arctan(-n[0] / s / c[ok]))
  if oR.sector is not None:
    omega[~oR.sector.checkInside(vector3d(V))] = 0.0
  return omega


# the points of the region's boundary about every axis, NaN where the axis lies outside
def surfacePoints(oR, V):
  V = np.asarray(V, dtype=float).reshape(-1, 3)
  omega = maxAngles(oR, V)
  P = omega[:, None] * V
  P[omega <= 0] = np.nan
  return P


# the boundary of an orientation region in axis angle space: the sphere mesh pushed out to
# the largest angle about every axis, in radians, with the faces of the mesh
def regionSurface(oR, resolution=5 * degree):
  V, F = sphereMesh(resolution)
  return surfacePoints(oR, V), F


# the walls of a region cut down to an axis sector: the plane of every boundary arc of the
# sector from the identity out to the surface, one polygon per arc
def regionWalls(oR, resolution=2 * degree):
  if oR.sector is None:
    return []
  return [np.concatenate([np.zeros((1, 3)), surfacePoints(oR, arc.data.reshape(-1, 3))]) for arc in oR.sector.boundary(resolution)]


# the parallels and meridians drawn on the surface, which is how a viewer reads its shape
def regionLines(oR, step=15 * degree):
  return [surfacePoints(oR, l) for l in sphereGridLines(step)]


# ------------------------------------------------------------------ crystal shapes

# the faces a viewer along view sees: those whose outward normal points at the viewer, a
# mask over the faces, one row per shape of a list
def visibleFaces(cS, view):
  v = (view.data if isinstance(view, vector3d) else np.asarray(view, dtype=float)).reshape(3)
  n = cS.faceNormal().data.reshape(-1, cS.numFaces, 3)
  vis = n @ (v / np.linalg.norm(v)) > 1e-9
  return vis[0] if len(cS) == 1 else vis


# the shapes seen along the normal of a plane, as polygons on the screen: the vertices in
# the coordinates of the plane under the layout, for the front faces; a layout that flips
# the plane is seen from the other side, so the other faces are the front ones. Returns
# the polygons with the shape and the face each one is
def shapeOutlines(cS, plane, layout=None):
  M = np.eye(2) if layout is None else np.asarray(layout, dtype=float)
  n = plane.normal * (-1 if np.linalg.det(M) < 0 else 1)
  vis = visibleFaces(cS, n).reshape(-1, cS.numFaces)
  P = plane.to2d(cS.V.data.reshape(-1, cS.numVertices, 3)) @ M.T
  polys, shape, face = [], [], []
  for k, i in zip(*np.nonzero(vis)):
    f = cS.F[i]
    polys.append(P[k, f[f >= 0]])
    shape.append(k)
    face.append(i)
  return polys, np.array(shape, dtype=int), np.array(face, dtype=int)


# one crystal shape per grain of the shape's phase: rotated by the grain's mean
# orientation, scaled by the square root of its area and put at its centroid, as one list
# of shapes over the grains that take it, whose indices come with it; on an EBSD map one
# shape per measurement of the phase, rotated by its orientation, at the shape's own size.
# A phase takes the shape when its frame fits the shape's crystal frame the way an
# orientation fits its frame: another lattice is left out
def shapesOnMap(x, cS, scale=1.0):
  from ..geometry.products import fitFrame
  if not isinstance(cS.CS, crystalFrame):
    raise ValueError("the shape has to be given in a crystal frame")
  onGrains = hasattr(x, "meanOrientation")
  parts, ids = [], []
  for p in x.indexedPhaseId:
    fr = x.CSList[int(p)]
    if not isinstance(fr, crystalFrame):
      continue
    try:
      fitFrame(cS.CS, fr)
    except ValueError:
      continue
    at = np.nonzero(x.phaseId.reshape(-1) == p)[0]
    sub = x[at] if onGrains else x[np.isin(np.arange(x.phaseId.size).reshape(x.phaseId.shape), at)]
    if onGrains:
      size = scale * np.sqrt(np.maximum(sub.area, 0))
      ok = size > 0
      if not np.any(ok):
        continue
      sub, size = sub[ok], size[ok]
      s = sub.meanOrientation * cS * size + sub.centroid
    else:
      ok = np.ones(len(at), dtype=bool)
      s = sub.orientations.reshape(-1) * cS * scale + sub.pos.reshape(-1)
    parts.append(s)
    ids.append(at[ok])
  if not parts:
    return None, np.zeros(0, dtype=int)
  s = parts[0]
  if len(parts) > 1:
    V = np.concatenate([q.V.data.reshape(-1, cS.numVertices, 3) for q in parts])
    s = s._new(vector3d(V, frame=s.V.frame, dispStyle="xyz"))
  return s, np.concatenate(ids)


__all__ = ["sphereMesh", "sphereGridLines", "smallCircle", "frameAxes", "axisAnglePoints", "maxAngles", "regionSurface",
           "regionWalls", "regionLines", "visibleFaces", "shapeOutlines", "shapesOnMap"]
