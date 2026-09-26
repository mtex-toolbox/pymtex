# the section of the grains of a volume with a plane (MATLAB grain3d/slice, ../master)
#
#   grains2 = sliceGrains(grains3, plane3d(N, P0))    grain2d; prop Id3d is the grain cut
#
# Every face that crosses the plane gives a segment between the points where two of its
# edges cross it; the segment is turned so that the first grain of the face, the one its
# normal points out of, lies on its left. A grain the plane cuts into several pieces gives a
# 2D grain per piece: the loops of every grain are the connected parts of its segments, the
# anticlockwise ones pieces and the clockwise ones holes, each hole going to the piece that
# holds it. The plane is moved by 0.001 towards the origin, as MATLAB moves it, so that no
# vertex of a voxel mesh lies on it.
from __future__ import annotations

import numpy as np

from ..geometry.plane import plane3d
from ..geometry.vector3d import vector3d
from ..maps.boundary import OUTSIDE, _walkOrder, grainBoundary
from ..maps.grains import grain2d
from ..maps.lattice import mapPlane


def sliceGrains(grains, plane, P0=None):
  if not isinstance(plane, plane3d):
    plane = plane3d(plane, P0 if P0 is not None else 0)
  N = plane.N.data.reshape(3)
  d = float(np.asarray(plane.d).reshape(-1)[0])
  d += 0.001 * (-1 if d > 0 else 1)
  gB = grains.boundary
  V = gB.allV
  dist = V @ N - d
  # the corners whose edge to the next corner crosses the plane
  u, v = gB._idx, gB._idx[gB._nextCorner()]
  face = gB._faceOfCorner()
  cross = (dist[u] > 0) != (dist[v] > 0)
  keep = np.bincount(face[cross], minlength=len(gB)) == 2
  cross &= keep[face]
  if not cross.any():
    raise ValueError("the plane does not cut the grains")
  e = np.sort(np.stack([u[cross], v[cross]], axis=1), axis=1)
  E, eId = np.unique(e, axis=0, return_inverse=True)
  t = dist[E[:, 0]] / (dist[E[:, 0]] - dist[E[:, 1]])
  P = V[E[:, 0]] + t[:, None] * (V[E[:, 1]] - V[E[:, 0]])
  f = face[cross]
  order = np.argsort(f, kind="stable")
  F = eId.reshape(-1)[order].reshape(-1, 2)
  faces = f[order][::2]
  # the segments turned to have the first grain of the face on their left
  pl = mapPlane(P, e1=_inPlane(N))
  P2 = pl.to2d(P)
  n = gB._vectorArea()[faces]
  tan = P[F[:, 1]] - P[F[:, 0]]
  swap = np.einsum("ij,ij->i", np.cross(pl.normal, tan), n) > 0
  F[swap] = F[swap][:, ::-1]
  g = gB.grainId[faces]
  newId, oldId = _pieces(F, g, P2, grains.id)
  # both sides in 2D grain ids, in the order of the 2D boundary: phases, then grains
  side = np.stack([newId[:, 0], newId[:, 1]], axis=1)
  ph = gB.phaseId[faces].astype(np.int16)
  eid = gB.ebsdId[faces].copy()
  mori = gB.misrotation.data.reshape(-1, 4)[faces].copy()
  ph = np.where(side > 0, ph, OUTSIDE)
  flip = (ph[:, 0] > ph[:, 1]) | ((ph[:, 0] == ph[:, 1]) & (side[:, 0] > side[:, 1]))
  for a in (side, ph, eid):
    a[flip] = a[flip][:, ::-1]
  F[flip] = F[flip][:, ::-1]
  mori[flip, 1:] *= -1

  def leftOf(r):
    a, b = P2[F[r, 0]], P2[F[r, 1]]
    t = b - a
    return (a + b) / 2 + 1e-6 * np.stack([-t[:, 1], t[:, 0]], axis=1)

  idx, F = _walkOrder(F, P2, leftOf)
  b2 = grainBoundary(P2, F, side[idx], eid[idx], ph[idx], mori[idx], grains.CSList, pl, grains.frame, scanUnit=gB.scanUnit)
  z = np.zeros((0, 2), dtype=np.int64)
  inner = grainBoundary(P2, z, z, z, z, np.zeros((0, 4)), grains.CSList, pl, grains.frame, scanUnit=gB.scanUnit)
  pos = grains.id2ind(oldId)
  return grain2d(np.arange(1, len(oldId) + 1), grains.phaseId[pos], grains.CSList, np.ones(len(oldId)),
                 grains.meanRotation.data.reshape(-1, 4)[pos], grains.GOS[pos], grains.frame, {"Id3d": oldId}, boundary=b2,
                 innerBoundary=inner, phaseMap=grains.phaseMap)


# the first direction of the plane's coordinates: the one of x, y, z furthest from the normal
def _inPlane(N):
  return np.eye(3)[np.argmin(np.abs(N))]


# the 2D grain of both sides of every segment: a grain's segments, each taken with the grain
# on its left, fall into connected parts; the anticlockwise parts are pieces, numbered in the
# order of the grains and then of the pieces a grain has more of, the clockwise parts are
# holes of the piece that holds them
def _pieces(F, g, P2, ids):
  from scipy.sparse import coo_matrix
  from scipy.sparse.csgraph import connected_components
  half = np.concatenate([np.stack([F[:, 0], F[:, 1], g[:, 0], np.arange(len(F)), np.zeros(len(F), dtype=np.int64)], axis=1),
                         np.stack([F[:, 1], F[:, 0], g[:, 1], np.arange(len(F)), np.ones(len(F), dtype=np.int64)], axis=1)])
  half = half[half[:, 2] > 0]
  newId = np.zeros(F.shape, dtype=np.int64)
  nV = len(P2)
  # the parts: half segments of one grain that share a vertex
  key = half[:, 2] * nV
  a, b = key + half[:, 0], key + half[:, 1]
  nodes, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
  m = len(half)
  G = coo_matrix((np.ones(m), (inv[:m], inv[m:])), shape=(len(nodes), len(nodes)))
  _, comp = connected_components(G, directed=False)
  part = comp[inv[:m]]
  # the signed area of every part, positive around a piece
  x0, y0 = P2[half[:, 0], 0], P2[half[:, 0], 1]
  x1, y1 = P2[half[:, 1], 0], P2[half[:, 1], 1]
  area = np.bincount(part, weights=x0 * y1 - x1 * y0) / 2
  grainOf = np.zeros(comp.max() + 1, dtype=np.int64)
  grainOf[part] = half[:, 2]
  parts = np.unique(part)
  outer = parts[area[parts] > 0]
  first, extra = [], []
  for gid in ids:
    mine = outer[grainOf[outer] == gid]
    if len(mine):
      first.append(mine[0])
      extra.extend(mine[1:])
  pieceOrder = first + extra
  label = np.zeros(comp.max() + 1, dtype=np.int64)
  label[pieceOrder] = np.arange(1, len(pieceOrder) + 1)
  # a hole to the piece of its grain that holds its first vertex
  for h in parts[area[parts] <= 0]:
    cand = [p for p in outer if grainOf[p] == grainOf[h]]
    if len(cand) == 1:
      label[h] = label[cand[0]]
      continue
    q = P2[half[part == h][0, 0]]
    for p in cand:
      if _inside(q, P2[half[part == p][:, 0]], P2[half[part == p][:, 1]]):
        label[h] = label[p]
        break
  newId[half[:, 3], half[:, 4]] = label[part]
  oldId = grainOf[pieceOrder]
  return newId, oldId


# a point inside the loop of segments (a, b), by the crossing number
def _inside(q, a, b):
  c = (a[:, 1] > q[1]) != (b[:, 1] > q[1])
  x = a[c, 0] + (q[1] - a[c, 1]) * (b[c, 0] - a[c, 0]) / (b[c, 1] - a[c, 1])
  return int(np.sum(x > q[0])) % 2 == 1


__all__ = []
