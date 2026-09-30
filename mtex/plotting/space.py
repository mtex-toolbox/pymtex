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


def sphereMesh(resolution=5 * degree):
  """the unit sphere as a mesh: rings of constant polar angle joined by quads split into
  triangles, fans at the poles, every triangle wound so that its normal points outwards;
  the vertices as an (n, 3) array and the faces as (m, 3) index triples"""
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


def smallCircle(n, opening=90 * degree, points=361):
  """the circle of directions at an angle about a direction, closed; the plane at 90 degrees"""
  v = (n.data if isinstance(n, vector3d) else np.asarray(n, dtype=float)).reshape(3)
  v = v / np.linalg.norm(v)
  u = vector3d(v).orth().data.reshape(3)
  w = np.cross(v, u)
  t = np.linspace(0, 2 * np.pi, points)
  return np.cos(opening) * v + np.sin(opening) * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * w)


def sphereGridLines(step=30 * degree, points=91):
  """the small circles about the pole and the meridians, as polylines of directions"""
  lines = [smallCircle([0, 0, 1], theta, points) for theta in np.arange(step, np.pi - step + 1e-9, step)]
  t = np.linspace(0, np.pi, points // 2 + 1)
  for rho in np.arange(0, 2 * np.pi - step + 1e-9, step):
    lines.append(np.stack([np.sin(t) * np.cos(rho), np.sin(t) * np.sin(rho), np.cos(t)], axis=-1))
  return lines


def frameAxes(fr):
  """the axes a frame names, as unit directions with their labels: X, Y, Z of a specimen frame,
  the lattice axes a, b, c of a crystal frame"""
  if isinstance(fr, crystalFrame):
    axes = [np.asarray(fr.sym.axes[k], dtype=float) for k in range(3)]
    return [(a / np.linalg.norm(a), n) for a, n in zip(axes, fr.axesNames)]
  names = fr.axesNames if fr is not None else ("X", "Y", "Z")
  return [(np.eye(3)[k], names[k]) for k in range(3)]


# ------------------------------------------------------------------ orientation space

def axisAnglePoints(o):
  """the orientations as points of axis angle space, each brought into the fundamental region
  of its frames first: axis times angle, in radians, shape (n, 3)"""
  q = o.flatten().projectIntoFundamentalRegion().data.reshape(-1, 4)
  omega = quatAngle(q)
  return np.where(omega[:, None] < 1e-12, 0.0, omega[:, None] * quatAxis(q))


def maxAngles(oR, V):
  """the largest rotation angle inside the region about every axis of an (n, 3) array, zero
  where the axis lies outside the sector the region is cut down to: the bound
  dot(q, N_i) <= 0 at q = (cos(w/2), sin(w/2) v) reads tan(w/2) dot(v, a_i) <= tan(w_i/4)"""
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


def surfacePoints(oR, V):
  """the points of the region's boundary about every axis, NaN where the axis lies outside"""
  V = np.asarray(V, dtype=float).reshape(-1, 3)
  omega = maxAngles(oR, V)
  P = omega[:, None] * V
  P[omega <= 0] = np.nan
  return P


def regionSurface(oR, resolution=5 * degree):
  """the boundary of an orientation region in axis angle space: the sphere mesh pushed out to
  the largest angle about every axis, in radians, with the faces of the mesh"""
  V, F = sphereMesh(resolution)
  return surfacePoints(oR, V), F


def regionWalls(oR, resolution=2 * degree):
  """the walls of a region cut down to an axis sector: the plane of every boundary arc of the
  sector from the identity out to the surface, one polygon per arc"""
  if oR.sector is None:
    return []
  return [np.concatenate([np.zeros((1, 3)), surfacePoints(oR, arc.data.reshape(-1, 3))]) for arc in oR.sector.boundary(resolution)]


def regionLines(oR, step=15 * degree):
  """the parallels and meridians drawn on the surface, which is how a viewer reads its shape"""
  return [surfacePoints(oR, l) for l in sphereGridLines(step)]


# ------------------------------------------------------------------ isosurfaces

# the corners of a cube and its split into six tetrahedra about the diagonal 0-6, the same
# in every cube, so that the split faces of neighbours agree
_CUBE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
_TETRAS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]


# the triangles of every sign pattern of a tetrahedron, as triples of edges (a, b): one
# triangle cut off a single corner, two cutting a pair of corners from the other pair
def _tetraCases():
  cases = {}
  for mask in range(1, 15):
    inside = [k for k in range(4) if mask >> k & 1]
    outside = [k for k in range(4) if not mask >> k & 1]
    if len(inside) == 2:
      (a, b), (c, d) = inside, outside
      cases[mask] = [((a, c), (a, d), (b, d)), ((a, c), (b, d), (b, c))]
    else:
      odd, rest = (inside[0], outside) if len(inside) == 1 else (outside[0], inside)
      cases[mask] = [tuple((odd, o) for o in rest)]
  return cases


_CASES = _tetraCases()


def isosurface(x, y, z, V, level):
  """the isosurface V = level of values on the grid of the axes x, y, z (V indexed [i, j, k]
  along x, y, z) as a list of triangles, an (n, 3, 3) array, by marching tetrahedra"""
  X = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
  nx, ny, nz = V.shape
  i, j, k = (a.ravel() for a in np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij"))
  corners = [(i + d[0], j + d[1], k + d[2]) for d in _CUBE]
  P = np.stack([X[c] for c in corners], axis=1)
  F = np.stack([V[c] for c in corners], axis=1) - level
  tris = []
  for t in _TETRAS:
    Pt, Ft = P[:, t], F[:, t]
    mask = np.sum((Ft > 0) * (1 << np.arange(4)), axis=1)
    for m, faces in _CASES.items():
      rows = mask == m
      if not rows.any():
        continue
      Pm, Fm = Pt[rows], Ft[rows]
      for face in faces:
        pts = []
        for a, b in face:
          s = Fm[:, a] / (Fm[:, a] - Fm[:, b])
          pts.append(Pm[:, a] + s[:, None] * (Pm[:, b] - Pm[:, a]))
        tris.append(np.stack(pts, axis=1))
  return np.concatenate(tris) if tris else np.zeros((0, 3, 3))


# ------------------------------------------------------------------ crystal shapes

def visibleFaces(cS, view):
  """the faces a viewer along view sees: those whose outward normal points at the viewer, a
  mask over the faces, one row per shape of a list"""
  v = (view.data if isinstance(view, vector3d) else np.asarray(view, dtype=float)).reshape(3)
  n = cS.faceNormal().data.reshape(-1, cS.numFaces, 3)
  vis = n @ (v / np.linalg.norm(v)) > 1e-9
  return vis[0] if len(cS) == 1 else vis


def shapeOutlines(cS, plane, layout=None):
  """the shapes seen along the normal of a plane, as polygons on the screen: the vertices in
  the coordinates of the plane under the layout, for the front faces; a layout that flips
  the plane is seen from the other side, so the other faces are the front ones. Returns
  the polygons with the shape and the face each one is"""
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


def shapesOnMap(x, cS, scale=1.0):
  """one crystal shape per grain of the shape's phase: rotated by the grain's mean
  orientation, scaled by the square root of its area and put at its centroid, as one list
  of shapes over the grains that take it, whose indices come with it; on an EBSD map one
  shape per measurement of the phase, rotated by its orientation, at the shape's own size.
  A phase takes the shape when its frame fits the shape's crystal frame the way an
  orientation fits its frame: another lattice is left out"""
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
