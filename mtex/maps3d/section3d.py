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
from ..maps.boundary import OUTSIDE, _sideSwap, _walkOrder, grainBoundary
from ..maps.grains import grain2d, _at, _positionTable
from ..maps.segment import components, _firstAppearanceLabels
from ..maps.lattice import mapPlane


def sliceGrains(grains, plane, P0=None):
  """the section of the grains of a volume with a plane, the 2D grains of the pieces the plane
  cuts with the grain of the volume in the property Id3d; MATLAB's slice of grain3d

    grains2 = sliceGrains(grains3, plane3d(N, P0))
    grains2 = sliceGrains(grains3, N, P0)"""
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
    return _noGrains(grains, N, d)
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
  side = newId.copy()
  ph = gB.phaseId[faces].astype(np.int16)
  eid = gB.ebsdId[faces].copy()
  mori = gB.misrotation.data.reshape(-1, 4)[faces].copy()
  ph = np.where(side > 0, ph, OUTSIDE)
  flip = _sideSwap(ph, side)
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


# the empty section of a plane that cuts no grain, a grain2d in that plane, as MATLAB's
def _noGrains(grains, N, d):
  e1 = _inPlane(N)
  e1 = e1 - (e1 @ N) * N
  p0 = d * N
  pl = mapPlane(np.stack([p0, p0 + e1, p0 + np.cross(N, e1)]), e1=e1)
  z = np.zeros((0, 2), dtype=np.int64)
  gB = grains.boundary
  b2 = [grainBoundary(np.zeros((0, 2)), z, z, z, z, np.zeros((0, 4)), grains.CSList, pl, grains.frame, scanUnit=gB.scanUnit) for _ in range(2)]
  return grain2d(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8), grains.CSList, np.ones(0), np.zeros((0, 4)), np.zeros(0),
                 grains.frame, {"Id3d": np.zeros(0, dtype=np.int64)}, boundary=b2[0], innerBoundary=b2[1], phaseMap=grains.phaseMap)


# the first direction of the plane's coordinates: the one of x, y, z furthest from the normal
def _inPlane(N):
  return np.eye(3)[np.argmin(np.abs(N))]


# the 2D grain of both sides of every segment: a grain's segments, each taken with the grain
# on its left, are walked into loops, at a vertex the one leaving it first clockwise from the
# way back, so that pieces touching in a vertex stay apart; the anticlockwise loops are
# pieces, numbered in the order of the grains and then of the pieces a grain has more of, the
# clockwise loops are holes of the piece that holds them
def _pieces(F, g, P2, ids):
  half = np.concatenate([np.stack([F[:, 0], F[:, 1], g[:, 0], np.arange(len(F)), np.zeros(len(F), dtype=np.int64)], axis=1),
                         np.stack([F[:, 1], F[:, 0], g[:, 1], np.arange(len(F)), np.ones(len(F), dtype=np.int64)], axis=1)])
  half = half[half[:, 2] > 0]
  newId = np.zeros(F.shape, dtype=np.int64)
  m = len(half)
  # the half segments leaving every vertex of a grain, by the angle they leave in
  d = P2[half[:, 1]] - P2[half[:, 0]]
  phi = np.arctan2(d[:, 1], d[:, 0])
  order = np.lexsort((phi, half[:, 0], half[:, 2]))
  key = np.stack([half[order, 2], half[order, 0]], axis=1)
  start = np.flatnonzero(np.r_[True, np.any(key[1:] != key[:-1], axis=1)])
  count = np.diff(np.r_[start, m])
  # the next half segment of a loop: of the grain, leaving the end, first clockwise from back
  nxt = np.empty(m, dtype=np.int64)
  pos = np.searchsorted(key[:, 0] * (len(P2) + 1) + key[:, 1],
                        half[:, 2] * (len(P2) + 1) + half[:, 1])
  grp = np.searchsorted(start, pos, side="right") - 1
  single = count[grp] == 1
  nxt[single] = order[start[grp[single]]]
  for i in np.flatnonzero(~single):
    cand = order[start[grp[i]]:start[grp[i]] + count[grp[i]]]
    back = np.arctan2(-d[i, 1], -d[i, 0])
    nxt[i] = cand[np.argmin(np.mod(back - phi[cand], 2 * np.pi))]
  # the loops, the cycles of nxt, numbered by their first half segment
  labels, n = _firstAppearanceLabels(components(m, np.arange(m), nxt))
  part = labels.astype(np.int64) - 1
  # the signed area of every loop, positive around a piece
  x0, y0 = P2[half[:, 0], 0], P2[half[:, 0], 1]
  x1, y1 = P2[half[:, 1], 0], P2[half[:, 1], 1]
  area = np.bincount(part, weights=x0 * y1 - x1 * y0, minlength=n) / 2
  grainOf = np.zeros(n, dtype=np.int64)
  grainOf[part] = half[:, 2]
  parts = np.arange(n)
  outer = parts[area > 0]
  # the pieces of the grains asked for in their order, the first piece of every grain before the
  # further ones
  r = _at(_positionTable(ids), grainOf[outer])
  rows = np.flatnonzero(r >= 0)
  rows = rows[np.argsort(r[rows], kind="stable")]
  isFirst = np.diff(r[rows], prepend=-1) != 0
  pieceOrder = np.concatenate([outer[rows[isFirst]], outer[rows[~isFirst]]])
  label = np.zeros(n, dtype=np.int64)
  label[pieceOrder] = np.arange(1, len(pieceOrder) + 1)
  # the half segments of every loop, in their order
  byPart = np.argsort(part, kind="stable")
  bounds = np.searchsorted(part[byPart], np.arange(n + 1))

  def halves(p):
    return half[byPart[bounds[p]:bounds[p + 1]]]

  # a hole to the smallest piece of its grain that holds it, tested at a vertex off the piece,
  # since a hole may touch it
  for h in parts[area <= 0]:
    cand = outer[grainOf[outer] == grainOf[h]]
    if len(cand) == 1:
      label[h] = label[cand[0]]
      continue
    vh = halves(h)[:, 0]
    holding = []
    for p in cand:
      hp = halves(p)
      off = np.setdiff1d(vh, hp[:, 0])
      if len(off) and _inside(P2[off[0]], P2[hp[:, 0]], P2[hp[:, 1]]):
        holding.append(p)
    if holding:
      label[h] = label[min(holding, key=lambda p: area[p])]
  newId[half[:, 3], half[:, 4]] = label[part]
  oldId = grainOf[pieceOrder]
  return newId, oldId


# a point inside the loop of segments (a, b), by the crossing number
def _inside(q, a, b):
  c = (a[:, 1] > q[1]) != (b[:, 1] > q[1])
  x = a[c, 0] + (q[1] - a[c, 1]) * (b[c, 0] - a[c, 0]) / (b[c, 1] - a[c, 1])
  return int(np.sum(x > q[0])) % 2 == 1


__all__ = []
