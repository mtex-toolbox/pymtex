# the equispaced orientation grid (MATLAB @SO3Grid, equispacedSO3Grid): an S2Grid of
# (alpha, beta), the images of z, with a ladder of gamma per node, so a query costs three
# binary searches. The grid covers one fundamental region of its frame pair and the query
# is symmetrised instead: over one representative per coset of the rotations about z, the
# z rotations themselves being absorbed by the periods of alpha and gamma. Everything
# derived from the grid that is not a subgrid is a plain orientation list.
#
# The misorientation angle w between a query (a, b, g) and a node (a', b', g') is affine in
# cos of the gamma difference, cos w = (A - 1 + B cos(g' - dg)) / 2, with A, B, dg depending
# on b, b', a - a' only (MATLAB SO3Grid.c), which makes the gamma search one-dimensional.
from __future__ import annotations

import numpy as np
from scipy import sparse

from ..orientation import orientation
from ..quaternion import qmul, rotation
from ..representations import euler2quat, quatAngle, quatAxis
from ..symreduce import reductionGroups
from ..units import degree
from ..vector3d import vector3d, zvector
from .s1grid import S1Grid
from .s2grid import equispacedS2Grid


class SO3Grid(orientation):

  def __init__(self, rot, frameA, frameB, alphabeta, alpha, beta, gamma, resolution, center=None):
    orientation.__init__(self, rot, frameA, frameB)
    self.alphabeta = alphabeta
    self.alpha, self.beta = np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)
    self.cb, self.sb = np.cos(self.beta), np.sin(self.beta)
    self.gamma = list(gamma)
    self.offset = np.concatenate([[0], np.cumsum([len(g) for g in gamma])])
    self.res = float(resolution)
    self.center = rotation.id if center is None else center

  # anything derived from the grid is a plain orientation list
  def _new(self, data, **attrs):
    obj = object.__new__(orientation)
    obj.data = data
    for name in self._carried:
      setattr(obj, name, attrs.pop(name, getattr(self, name)))
    if attrs:
      raise TypeError(f"orientation carries no attribute {', '.join(attrs)}")
    return obj

  @property
  def resolution(self):
    return self.res

  @property
  def nnodes(self):
    return len(self.alpha)

  def node(self, k):
    return np.arange(self.offset[k], self.offset[k + 1])

  @property
  def alphaPeriod(self):
    return self.alphabeta.rhoGrid[0].period

  @property
  def nZ(self):
    return int(np.round(2 * np.pi / self.alphaPeriod))

  # the rotations selected by a mask; nodes that empty out are dropped
  def subGrid(self, mask):
    mask = np.asarray(mask)
    if mask.dtype != bool:
      m = np.zeros(self.size, dtype=bool)
      m[mask] = True
      mask = m
    keep = np.array([bool(np.any(mask[self.node(k)])) for k in range(self.nnodes)])
    gamma = [self.gamma[k].subGrid(mask[self.node(k)]) for k in range(self.nnodes) if keep[k]]
    return SO3Grid(rotation._fromData(self.data[mask]), self.frameA, self.frameB, self.alphabeta.subGrid(keep),
                   self.alpha[keep], self.beta[keep], gamma, self.res, self.center)

  # a rotation of the whole grid keeps the structure when the specimen group survives it
  def __rmul__(self, other):
    from ..products import dropGroup
    if isinstance(other, rotation) and not isinstance(other, orientation) and other.ndim == 0:
      fr = dropGroup(other, self.frameB)
      if fr is self.frameB:
        return SO3Grid(rotation._fromData(qmul(other.data, self.data)), self.frameA, self.frameB, self.alphabeta,
                       self.alpha, self.beta, self.gamma, self.res, other * self.center)
    return orientation.__rmul__(self, other)

  # the nearest grid rotation of every orientation, or the k nearest, or all within a
  # radius, up to the symmetries of the grid's frames, through the quaternion k-d tree of
  # orientationtree.py (MATLAB SO3Grid/find). The angles are the reduced misorientation ones
  def find(self, o, epsOrK=None):
    from .orientationtree import findInGrid
    return findInGrid(self, o, epsOrK)

  def __repr__(self):
    return f"SO3Grid ({self._frameSummary()})\n  size: {self.size}\n  resolution: {self.res / degree:.4g}°"


# whether rotations are about z, and one rotation per coset of the rotations about z: right
# cosets p Z for the group acting on the right, left cosets Z p for the one on the left
def _isAboutZ(q):
  return (quatAngle(q) < 1e-8) | (np.abs(quatAxis(q)[..., 2]) > 1 - 1e-8)




# ------------------------------------------------------------------ construction

def equispacedSO3Grid(CS, SS=None, resolution=5 * degree, points=None, maxAngle=np.pi, center=None):
  """nearly equidistant orientations covering one fundamental region of the frame pair: an
  equispaced grid of the images of z at the resolution, with a ladder of the third angle at
  each; `points` picks the resolution for that count; with `maxAngle` only the ball around
  `center` is kept, a ball smaller than the fundamental region being a local grid"""
  from ..frames import specimenFrame
  from .localgrids import localOrientationGrid
  SS = specimenFrame.default if SS is None else SS
  ctr = orientation(rotation.id, CS, SS) if center is None else center
  L, R = reductionGroups(orientation(rotation(), CS, SS))
  if maxAngle < np.pi / 2 / int(np.sum(_isAboutZ(R))) and center is None:
    return localOrientationGrid(CS, SS, maxAngle=maxAngle, resolution=resolution)
  res = float(resolution) if points is None else np.cbrt(8 * np.pi ** 2 / (len(L) * len(R)) / points)
  g = _buildSO3Grid(CS, SS, L, R, res)
  if points is not None:
    g = _buildSO3Grid(CS, SS, L, R, res * np.cbrt(g.size / points))
  if maxAngle < np.pi:
    g = g.subGrid(g.angle(ctr) <= maxAngle)
  return g


def _buildSO3Grid(CS, SS, L, R, res):
  from ..orientationregion import fundamentalRegion
  nZL, nZR = int(np.sum(_isAboutZ(L))), int(np.sum(_isAboutZ(R)))
  ap, gp = 2 * np.pi / nZL, 2 * np.pi / nZR
  oR = fundamentalRegion(CS, SS)
  bmax = np.pi if len(oR.N) == 0 else min(np.pi, oR.maxAngle() + 3 * res)
  noCenter = int(np.round(2 * np.pi / res)) % 2 == 0
  ab = equispacedS2Grid(resolution=res, maxTheta=bmax, minRho=0, maxRho=ap, rhoPeriod=ap, noCenter=noCenter)
  beta, alpha, _ = ab.polar()
  ng = max(int(np.round(gp / res)), 1)
  resg = gp / ng
  gamma, rots = [], []
  for k in range(len(alpha)):
    # the ladder centred on -alpha, since near beta = 0 only alpha + gamma matters
    g0 = -alpha[k] - gp / 2
    pts = g0 + np.arange(ng) * resg
    gamma.append(S1Grid(pts, min=g0, max=g0 + gp, periodic=True))
    rots.append(euler2quat(np.full(ng, alpha[k]), np.full(ng, beta[k]), pts, "Matthies"))
  rot = rotation._fromData(np.concatenate(rots) if rots else np.zeros((0, 4)))
  g = SO3Grid(rot, CS, SS, ab, alpha, beta, gamma, resg)
  if len(oR.N) == 0:
    return g
  # the box is one cell of the rotations about z, the region is not: a node stays when one
  # of its copies under those rotations lies in the region; then the pairs the boundary
  # holds twice are merged, searched for among the nodes near a face only
  zL = euler2quat(np.zeros(nZL), np.zeros(nZL), np.arange(nZL) * ap, "Matthies")
  zR = euler2quat(np.zeros(nZR), np.zeros(nZR), np.arange(nZR) * gp, "Matthies")
  keep, near = _regionMembership(oR, g.data, zL, zR, res / 2)
  g = g.subGrid(keep)
  return _mergeClose(g, res / 2, near[keep])


def _regionMembership(oR, rot, zL, zR, delta):
  keep = np.zeros(len(rot), dtype=bool)
  near = np.zeros(len(rot), dtype=bool)
  band = np.sin(delta / 2) + 1e-3
  sector = oR.sector is not None
  for l in zL:
    for r in zR:
      p = qmul(qmul(l[None], rot), r[None])
      inside = oR.checkInside(p)
      keep |= inside
      if sector:
        near |= inside
      else:
        near |= inside & np.any(np.abs(p @ oR.N.T) <= band, axis=1)
  return keep, near


# of two rotations closer than delta up to symmetry, found with the grid's own search, the
# later one goes: what stays is at least delta apart and nothing was further than delta
def _mergeClose(g, delta, candidates):
  idx = np.flatnonzero(candidates)
  if idx.size == 0:
    return g
  I, D = g.find(g[idx], delta)
  I = I.tocoo()
  pairs = sorted(((min(i, j), max(i, j)) for c, i in zip(I.row, I.col) for j in [idx[c]] if i != j), key=lambda p: p[1])
  drop = np.zeros(g.size, dtype=bool)
  for i, j in pairs:
    if not drop[i]:
      drop[j] = True
  return g.subGrid(~drop) if np.any(drop) else g


__all__ = ["SO3Grid", "equispacedSO3Grid"]
