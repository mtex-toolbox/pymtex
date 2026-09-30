# clustering directions and orientations (MATLAB vector3d/calcCluster, orientation/calcCluster,
# extern/classix; docs/plans/functions.md F6, ported 2026-09-22)
#
#   cId, center = calcCluster(v)                                     CLASSIX on the coordinates
#   cId, center = calcCluster(ori, method='classix', radius=, minPoints=)
#   cId, center = calcCluster(ori)                                   the modes of the density, MATLAB's default
#   cId, center = calcCluster(ori, method='hierarchical', maxAngle=15*degree, numCluster=)
#   labels = classix(X, radius, minPts=, mergeScale=1.5)
#
# CLASSIX (Chen and Guettel, Fast and explainable clustering based on sorting,
# arXiv:2202.01456): the data is centred, scaled by its median norm and sorted along its
# first principal axis; every point joins the first group whose centre lies within radius;
# groups whose centres lie within mergeScale times the radius are merged; a cluster below
# minPts points is dissolved into the nearest larger one. Directions cluster on their
# coordinates, axes on the rank one tensor that identifies v with -v, orientations on the
# isometric embedding that carries the symmetry. The hierarchical method merges the closest
# pair of centres until the count or the angle is reached; the density method takes the
# modes the orientations ascend to.
from __future__ import annotations

import numpy as np

from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from ..geometry.install import install


# ------------------------------------------------------------------ CLASSIX

def classix(X, radius, minPts=1, mergeScale=1.5, mergeTiny=True):
  """one cluster label per row of X, counted from zero"""
  X = np.asarray(X, dtype=float)
  if X.ndim == 1:
    X = X[:, None]
  n, d = X.shape
  if n == 0:
    return np.zeros(0, dtype=int)
  if radius <= 0:
    raise ValueError("the radius must be positive")
  # centre by the mean, scale by the median norm
  x = X - X.mean(axis=0)
  scale = np.median(np.linalg.norm(x, axis=1))
  x = x / (scale if scale > 0 else 1.0)
  # the scores along the first principal axis, oriented deterministically
  if d == 1:
    u = x[:, 0].copy()
  else:
    vals, vecs = np.linalg.eigh(x.T @ x)
    u = x @ vecs[:, np.argmax(np.abs(vals))]
  u = u * (1.0 if u[0] == 0 else -np.sign(u[0]))
  perm = np.argsort(u, kind="stable")
  x, u = x[perm], u[perm]
  halfN = np.sum(x * x, axis=1) / 2
  halfR2 = radius ** 2 / 2
  # aggregation: the first group within the radius takes the point
  label = np.full(n, -1)
  centres, sizes = [], []
  for i in range(n):
    if label[i] >= 0:
      continue
    lab = len(centres)
    centres.append(i)
    sizes.append(1)
    label[i] = lab
    hi = np.searchsorted(u, u[i] + radius, side="right")
    if hi <= i + 1:
      continue
    window = slice(i + 1, hi)
    close = (label[window] < 0) & (halfN[window] - x[window] @ x[i] <= halfR2 - halfN[i])
    idx = np.flatnonzero(close) + i + 1
    label[idx] = lab
    sizes[lab] += idx.size
  centres, sizes = np.array(centres), np.array(sizes)
  ng = len(centres)
  # merge the group centres within mergeScale times the radius by union find
  gx, gu, gh = x[centres], u[centres], halfN[centres]
  parent = np.arange(ng)

  def find(a):
    while parent[a] != a:
      parent[a] = parent[parent[a]]
      a = parent[a]
    return a

  mr = mergeScale * radius
  for i in range(ng):
    if not mergeTiny and sizes[i] < minPts:
      continue
    hi = np.searchsorted(gu, gu[i] + mr, side="right")
    if hi <= i + 1:
      continue
    window = slice(i + 1, hi)
    close = gh[window] - gx[window] @ gx[i] <= mr ** 2 / 2 - gh[i]
    if not mergeTiny:
      close &= sizes[window] >= minPts
    ri = find(i)
    for j in np.flatnonzero(close) + i + 1:
      rj = find(j)
      if ri != rj:
        if ri < rj:
          parent[rj] = ri
        else:
          parent[ri] = rj
          ri = rj
  gLabel, counts = _renumber(np.array([find(k) for k in range(ng)]), sizes)
  # dissolve the clusters below minPts into the nearest group of a large enough cluster
  if np.any(counts < minPts):
    fixed = gLabel.copy()
    large = counts[fixed] >= minPts
    if np.any(large):
      for c in np.flatnonzero(counts < minPts):
        for g in np.flatnonzero(fixed == c):
          dist = gh - gx @ gx[g]
          dist[~large] = np.inf
          gLabel[g] = fixed[int(np.argmin(dist))]
      gLabel, counts = _renumber(gLabel, sizes)
  out = np.empty(n, dtype=int)
  out[perm] = gLabel[label]
  return out


# the labels renamed to 0..K-1, in order of appearance, with the points per cluster
def _renumber(gLabel, sizes):
  uniq, inv = np.unique(gLabel, return_inverse=True)
  counts = np.bincount(inv, weights=sizes, minlength=len(uniq)).astype(int)
  return inv, counts


def knnRadius(X, k=10, sub=1500):
  """the median distance to the tenth nearest neighbour of a subsample, the estimate CLASSIX suggests"""
  from scipy.spatial import cKDTree
  X = np.asarray(X, dtype=float)
  n = len(X)
  if n <= k + 1:
    return 1.0
  idx = np.arange(n) if n <= sub else np.unique(np.round(np.linspace(0, n - 1, sub)).astype(int))
  d, _ = cKDTree(X).query(X[idx], k=k + 1)
  return float(np.median(d[:, -1]))


# ------------------------------------------------------------------ the clusters of directions and orientations

def calcCluster(x, *args, method=None, radius=None, minPoints=None, mergeScale=1.5, numCluster=None, maxAngle=None,
                halfwidth=5 * degree, weights=None, resolution=2.5 * degree, verbose=False):
  """the clusters of directions or orientations: one cluster id per element, counted from
  zero, and the centre of every cluster

    cId, center = calcCluster(v)                                          directions by CLASSIX
    cId, center = calcCluster(ori, method='classix', radius=0.1, minPoints=15)
    cId, center = calcCluster(ori)                                        the modes of the density
    cId, center = calcCluster(ori, method='hierarchical', maxAngle=15*degree)
    cId, center = calcCluster(v, method='hierarchical', numCluster=5)
  """
  for a in args:
    if isinstance(a, str):
      method = a
  isOri = isinstance(x, orientation)
  if method is None:
    method = "odf" if isOri else "classix"
  if method == "classix":
    return _classixCluster(x, radius, minPoints, mergeScale)
  if method == "hierarchical":
    return _hierarchical(x, numCluster, maxAngle, weights, resolution)
  if method == "odf":
    if not isOri:
      raise ValueError("the density method clusters orientations; use 'classix' or 'hierarchical' for directions")
    return _densityCluster(x, halfwidth, weights, numCluster)
  raise ValueError("the method is 'classix', 'hierarchical' or 'odf'")


def _classixCluster(x, radius, minPoints, mergeScale):
  n = x.size
  if isinstance(x, orientation):
    from .embedding import embedding
    X = embedding(x.flatten()).double()
    r = 0.5 * knnRadius(X) / np.max(np.std(X, axis=0)) if radius is None else float(radius)
  else:
    v = x.flatten()
    xyz = v.normalize().data
    if v.antipodal:
      X = np.stack([xyz[:, 0] ** 2, xyz[:, 1] ** 2, xyz[:, 2] ** 2, xyz[:, 0] * xyz[:, 1], xyz[:, 0] * xyz[:, 2], xyz[:, 1] * xyz[:, 2]], axis=1)
    else:
      X = xyz
    r = 2 * knnRadius(X) if radius is None else float(radius)
  mp = max(1, round(n / 100)) if minPoints is None else int(minPoints)
  cId = classix(X, r, minPts=mp, mergeScale=mergeScale)
  return cId.reshape(x.shape), _centers(x, cId)


# the mean of every cluster, as the elements are: an axis mean for axes, a symmetry reduced
# mean for orientations
def _centers(x, cId):
  K = int(cId.max()) + 1 if cId.size else 0
  flat = x.flatten()
  if isinstance(x, orientation):
    q = np.stack([rotation(flat[cId == c].mean().data).data.reshape(4) for c in range(K)]) if K else np.zeros((0, 4))
    return orientation(rotation(q), x.CS, x.SS, antipodal=x.antipodal)
  data = np.stack([flat[cId == c].mean().data.reshape(3) for c in range(K)]) if K else np.zeros((0, 3))
  return vector3d(data, antipodal=x.antipodal, frame=x.frame)


# the modes of the density the orientations ascend to, the largest numCluster of them
def _densityCluster(o, halfwidth, weights, numCluster):
  from ..functions.characteristics import calcComponents
  from ..functions.density import calcDensity
  flat = o.flatten()
  odf = calcDensity(flat, weights=weights, halfwidth=halfwidth)
  modes, vol, cId = calcComponents(odf, seed=flat, weights=weights)
  K = min(20 if numCluster is None else int(numCluster), modes.size)
  cId = np.asarray(cId).astype(float)
  cId[cId >= K] = np.nan
  center = modes[:K]
  # the points too far from their centre, beyond twice the mean angle, are dropped
  for i in range(K):
    mask = cId == i
    if not np.any(mask):
      continue
    omega = flat[mask].angle(center[i])
    far = np.flatnonzero(mask)[omega > 2 * np.mean(omega)]
    cId[far] = np.nan
  return cId.reshape(o.shape), center


# the closest pair of centres merged, the mean weighted by the points behind them, until
# the number of clusters or the angle between the centres is reached; a large list is
# binned on a grid first
def _hierarchical(x, numCluster, maxAngle, weights, resolution):
  flat = x.flatten()
  n = flat.size
  w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()
  isOri = isinstance(x, orientation)
  if n > 1000 and isOri:
    from ..geometry.grids.so3grid import equispacedSO3Grid
    g = equispacedSO3Grid(x.CS, x.SS, resolution=resolution)
    ind, _ = g.find(flat)
    gw = np.bincount(np.asarray(ind).ravel(), weights=w, minlength=g.size)
    keep = gw > 0
    _, center = _hierarchical(g.subGrid(keep), numCluster, maxAngle, gw[keep], resolution)
    cId = _nearest(flat, center)
    center = _centers(x, cId)
    cId = _nearest(flat, center)
    counts = np.bincount(cId, minlength=center.size)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    cId = rank[cId]
    center = center[order][counts[order] > 0]
    return cId.reshape(x.shape), center
  nTarget = 2 if numCluster is None else int(numCluster)
  cosMax = -np.inf if maxAngle is None else np.cos(maxAngle)
  members = [[i] for i in range(n)]
  centers = [flat[i] for i in range(n)]
  d = _similarity(flat, flat)
  np.fill_diagonal(d, -np.inf)
  alive = np.ones(n, dtype=bool)
  while np.count_nonzero(alive) > nTarget:
    k = int(np.argmax(d))
    i, j = divmod(k, n)
    if d[i, j] < cosMax:
      break
    members[i] += members[j]
    members[j] = []
    alive[j] = False
    sel = np.array(members[i])
    centers[i] = flat[sel].mean(weights=w[sel])
    dd = _similarity(centers[i], flat._new(np.stack([c.data.reshape(-1) for c in centers]))).ravel()
    dd[~alive] = -np.inf
    dd[i] = -np.inf
    d[i, :] = dd
    d[:, i] = dd
    d[j, :] = -np.inf
    d[:, j] = -np.inf
  cId = np.empty(n, dtype=int)
  keep = np.flatnonzero(alive)
  for c, i in enumerate(keep):
    cId[members[i]] = c
  center = flat._new(np.stack([centers[i].data.reshape(-1) for i in keep]))
  return cId.reshape(x.shape), center


# the cosine of the angle between every pair, reduced by the symmetry
def _similarity(a, b):
  A, B = a.reshape(-1, 1), b.reshape(1, -1)
  return np.cos(A.angle(B))


def _nearest(x, center):
  return np.argmin(x.reshape(-1, 1).angle(center.reshape(1, -1)), axis=1)


def _install():
  install((vector3d, orientation), calcCluster)


_install()

__all__ = ["calcCluster", "classix", "knnRadius"]
