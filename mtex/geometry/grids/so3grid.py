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

import copy
from typing import TYPE_CHECKING
import numpy as np
from scipy import sparse

from ..orientation import orientation
from ..quaternion import qmul, rotation
from ..representations import euler2quat, quatAngle, quatAxis
from ..kernels import regionMembership
from ..symreduce import reductionGroups
from ..units import degree
from ..vector3d import vector3d, zvector
from .s1grid import S1Grid
from .s2grid import equispacedS2Grid
from ..display import display, header


class SO3Grid(orientation):
  """the equispaced grid of orientations over one fundamental region of its frame pair, MATLAB's
  SO3Grid: an S2 grid of the images of z with a ladder of gamma per node, so a search costs
  three binary searches; built by equispacedSO3Grid, and anything derived from it is a plain
  orientation list

    S3G = equispacedSO3Grid(cs, ss, resolution=5*degree)"""

  # the ladders of gamma as one flat array, node k holding gammaPoints[offset[k]:offset[k + 1]]
  # on the circle [gammaMin[k], gammaMin[k] + gammaPeriod)
  def __init__(self, rot, frameA, frameB, alphabeta, alpha, beta, gammaPoints, offset, gammaMin, gammaPeriod,
               resolution, center=None):
    orientation.__init__(self, rot, frameA, frameB)
    self.alphabeta = alphabeta
    self.alpha, self.beta = np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)
    self.cb, self.sb = np.cos(self.beta), np.sin(self.beta)
    self.gammaPoints, self.offset = np.asarray(gammaPoints, dtype=float), np.asarray(offset, dtype=np.int64)
    self.gammaMin, self.gammaPeriod = np.asarray(gammaMin, dtype=float), float(gammaPeriod)
    self.res = float(resolution)
    self.center = rotation.id if center is None else center

  @property
  def gamma(self):
    """the ladder of every node as an S1Grid"""
    return [S1Grid(self.gammaPoints[self.offset[k]:self.offset[k + 1]], min=self.gammaMin[k],
                   max=self.gammaMin[k] + self.gammaPeriod, periodic=True) for k in range(self.numNodes)]

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
    """the resolution the grid was built for, in radians"""
    return self.res

  @property
  def numNodes(self):
    """the number of nodes of the S2 grid of (alpha, beta), each carrying a ladder of gamma"""
    return len(self.alpha)

  def node(self, k):
    """the indices of the orientations on the gamma ladder of node k"""
    return np.arange(self.offset[k], self.offset[k + 1])

  @property
  def alphaPeriod(self):
    """the period of alpha, 2 pi over the order of the rotations about z the frames share"""
    return self.alphabeta.rhoGrid[0].period

  @property
  def nZ(self):
    """the number of rotations about z absorbed by the period of alpha"""
    return int(np.round(2 * np.pi / self.alphaPeriod))

  def subGrid(self, mask):
    """the rotations selected by a mask; nodes that empty out are dropped"""
    mask = np.asarray(mask)
    if mask.dtype != bool:
      m = np.zeros(self.size, dtype=bool)
      m[mask] = True
      mask = m
    counts = np.add.reduceat(mask, self.offset[:-1]) if self.size else np.zeros(self.numNodes, dtype=np.int64)
    keep = counts > 0
    offset = np.concatenate([[0], np.cumsum(counts[keep])])
    return SO3Grid(rotation._fromData(self.data[mask]), self.frameA, self.frameB, self.alphabeta.subGrid(keep),
                   self.alpha[keep], self.beta[keep], self.gammaPoints[mask], offset, self.gammaMin[keep],
                   self.gammaPeriod, self.res, self.center)

  # a rotation of the whole grid keeps the structure when the specimen group survives it
  def __rmul__(self, other):
    from ..products import dropGroup
    if isinstance(other, rotation) and not isinstance(other, orientation) and other.ndim == 0:
      fr = dropGroup(other, self.frameB)
      if fr is self.frameB:
        return SO3Grid(rotation._fromData(qmul(other.data, self.data)), self.frameA, self.frameB, self.alphabeta,
                       self.alpha, self.beta, self.gammaPoints, self.offset, self.gammaMin, self.gammaPeriod,
                       self.res, other * self.center)
    return orientation.__rmul__(self, other)

  def find(self, o, epsOrK=None):
    """the nearest grid rotation of every orientation, or the k nearest, or all within a
    radius, up to the symmetries of the grid's frames, through the quaternion k-d tree of
    orientationtree.py (MATLAB SO3Grid/find): cos(w/2) of the reduced misorientation angle w
    for the nearest, the angles for the k nearest and the radius"""
    from .orientationtree import findInGrid
    return findInGrid(self, o, epsOrK)

  def radius(self, o, eps):
    """every grid rotation within eps of each query, up to symmetry: the sparse incidence
    (nq x n) and the sparse cos(w/2) of the reduced angles; the structured search where the
    queries carry the grid's frames and numba is there, the tree otherwise"""
    from ..kernels import backend
    from .orientationtree import orientationTree
    if backend() != "numba" or not isinstance(o, orientation) or o.frameA is not self.frameA or o.frameB is not self.frameB:
      return orientationTree(self).radius(o, eps)
    from .gridsearch import flatSO3, so3Radius
    if getattr(self, "_flat", None) is None:
      L, R = reductionGroups(self)
      self._flat = flatSO3(self), _cosetReps(L, left=True), _cosetReps(R, left=False), int(np.sum(_isAboutZ(L)))
    flat, Lr, Rr, nZL = self._flat
    conj = np.array([1.0, -1.0, -1.0, -1.0])
    c, q = self.center.data.reshape(4) * conj, o.data.reshape(-1, 4)
    Q = qmul(c, np.stack([q, q * conj], axis=1) if o.antipodal else q[:, None, :])
    indptr, cols, vals = so3Radius(flat, Q, Lr, Rr, nZL, eps)
    shape = (len(q), self.size)
    return (sparse.csr_matrix((np.ones(len(cols), dtype=bool), cols, indptr), shape=shape),
            sparse.csr_matrix((vals, cols.copy(), indptr.copy()), shape=shape))

  def __repr__(self):
    return display(header("SO3Grid", self._frameSummary()), [("size", self.size), ("resolution", f"{self.res / degree:.4g}°")])

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    @property
    def frameA(self) -> Any:
      ...
    @frameA.setter
    def frameA(self, value) -> None:
      ...
    @property
    def frameB(self) -> Any:
      ...
    @frameB.setter
    def frameB(self, value) -> None:
      ...


# whether rotations are about z, and one rotation per coset of the rotations about z: right
# cosets p Z for the group acting on the right, left cosets Z p for the one on the left
def _isAboutZ(q):
  return (quatAngle(q) < 1e-8) | (np.abs(quatAxis(q)[..., 2]) > 1 - 1e-8)


def _cosetReps(G, left):
  Z = G[_isAboutZ(G)]
  reps, covered = [], np.zeros((0, 4))
  for p in G:
    if len(covered) and np.max(np.abs(covered @ p)) > 1 - 1e-8:
      continue
    reps.append(p)
    covered = np.concatenate([covered, qmul(Z, p[None]) if left else qmul(p[None], Z)])
  return np.array(reps).reshape(-1, 4)




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


# the last grids built, by their frames and resolution, as MATLAB keeps its last one: max(odf),
# calcODF and the plots ask for the same grid again and again. A caller gets a copy with its own
# coordinates, the ladders shared, which nothing changes in place; a grid above GRID_CACHE_NODES
# is not kept
_GRIDS = {}
GRID_CACHE_SIZE, GRID_CACHE_NODES = 4, 2_000_000


def _buildSO3Grid(CS, SS, L, R, res):
  key = (CS, SS, round(float(res), 12))
  g = _GRIDS.pop(key, None)
  if g is None:
    g = _newSO3Grid(CS, SS, L, R, res)
    if g.size > GRID_CACHE_NODES:
      return g
  _GRIDS[key] = g
  while len(_GRIDS) > GRID_CACHE_SIZE:
    _GRIDS.pop(next(iter(_GRIDS)))
  out = copy.copy(g)
  out.data = g.data.copy()
  return out


def _newSO3Grid(CS, SS, L, R, res):
  from ..orientationregion import fundamentalRegion
  nZL, nZR = int(np.sum(_isAboutZ(L))), int(np.sum(_isAboutZ(R)))
  ap, gp = 2 * np.pi / nZL, 2 * np.pi / nZR
  oR = fundamentalRegion(CS, SS)
  bmax = np.pi if len(oR.N) == 0 else min(np.pi, oR.maxAngle() + 3 * res)
  center = int(np.round(2 * np.pi / res)) % 2 != 0
  ab = equispacedS2Grid(resolution=res, maxTheta=bmax, minRho=0, maxRho=ap, rhoPeriod=ap, center=center)
  beta, alpha, _ = ab.polar()
  ng = max(int(np.round(gp / res)), 1)
  resg = gp / ng
  # the ladders centred on -alpha, since near beta = 0 only alpha + gamma matters
  g0 = -alpha - gp / 2
  pts = (g0[:, None] + np.arange(ng) * resg).reshape(-1)
  node = np.repeat(np.arange(len(alpha)), ng)
  rot = rotation._fromData(euler2quat(alpha[node], beta[node], pts, "Matthies").reshape(-1, 4))
  g = SO3Grid(rot, CS, SS, ab, alpha, beta, pts, np.arange(len(alpha) + 1) * ng, g0, gp, resg)
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
  band = np.sin(delta / 2) + 1e-3
  if oR.sector is None:
    # dot(l q r, n) = dot(q, l* n r*): the normals turned, one product for every copy
    conj = np.array([1.0, -1.0, -1.0, -1.0])
    N = qmul(qmul((zL * conj)[:, None, None], oR.N[None, None]), (zR * conj)[None, :, None]).reshape(-1, 4)
    return regionMembership(rot, N, len(zL) * len(zR), 1e-3, band)   # checkInside's tolerance
  keep = np.zeros(len(rot), dtype=bool)
  for l in zL:
    for r in zR:
      keep |= oR.checkInside(qmul(qmul(l[None], rot), r[None]))
  return keep, keep.copy()


# of two rotations closer than delta up to symmetry, found with the grid's own search, the
# later one goes: what stays is at least delta apart and nothing was further than delta
def _mergeClose(g, delta, candidates):
  idx = np.flatnonzero(candidates)
  if idx.size == 0:
    return g
  I, _ = g.radius(g[idx], delta)
  I = I.tocoo()
  a, b = I.col, idx[I.row]
  a, b = a[a != b], b[a != b]
  lo, hi = np.minimum(a, b), np.maximum(a, b)
  order = np.argsort(hi, kind="stable")
  drop = np.zeros(g.size, dtype=bool)
  for i, j in zip(lo[order].tolist(), hi[order].tolist()):
    if not drop[i]:
      drop[j] = True
  return g.subGrid(~drop) if np.any(drop) else g


def regularSO3Grid(CS, SS=None, resolution=5 * degree, sections=None):
  """the Euler angle raster over the Euler box of the frames, MATLAB's regularSO3Grid: sections
  of phi2 from zero below its bound, and on each the (Phi, phi1) grid with both ends of Phi
  and phi1 from zero below its bound, all at the resolution"""
  from ..frames import specimenFrame
  from ..orientationregion import eulerRegion
  SS = specimenFrame.default if SS is None else SS
  maxPhi1, maxPhi, maxPhi2 = eulerRegion(CS, SS)
  nsec = int(round(maxPhi2 / resolution)) if sections is None else int(sections)
  phi2 = np.linspace(0, maxPhi2, nsec + 1)[:-1]
  Phi = np.linspace(0, maxPhi, int(round(maxPhi / resolution)) + 1)
  phi1 = np.linspace(0, maxPhi1, int(round(maxPhi1 / resolution)) + 1)[:-1]
  B, A, C = np.meshgrid(Phi, phi1, phi2, indexing="ij")
  return orientation.byEuler(A, B, C, "Bunge", CS, SS)


__all__ = ["SO3Grid", "equispacedSO3Grid", "regularSO3Grid"]
