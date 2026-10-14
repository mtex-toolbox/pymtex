# the traces of subregions of a map (MATLAB @EBSD/calcTraces, @grain2d/calcTraces)
#
#   traces, rel, cSize = calcTraces(ebsd, clusterId, 'Radon')      the default, or 'Fourier'
#   traces, rel, cSize = calcTraces(grains, clusterId)              the characteristic shape, or 'calliper', 'hist'
#
# A cluster is the set of pixels, or grains, with one value of every column of clusterId;
# the default is one cluster of all indexed pixels, or of all grains. The trace of a
# cluster is the direction its lamellae run along, an axis in the plane of the map; rel is
# how much the winning direction stands out, zero for none. The outputs have the shape of
# the largest id of every column, NaN and zero where a cluster has minClusterSize pixels or
# fewer. Reference: Nyyssonen et al., arXiv 2201.02103.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import columnMax, columnMin
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .decomposition import _gather
from .ebsd import EBSD
from .grains import grain2d, _at, _positionTable
from ..geometry.install import install


# the columns of a cluster id per measurement or grain, the shape of the output and the
# size of every cluster, from the weights of the members
def _clusters(clusterId, n, weights):
  cid = np.ones((n, 1), dtype=np.int64) if clusterId is None else np.asarray(clusterId).reshape(n, -1)
  cid = np.where(np.isfinite(cid), cid, 0).astype(np.int64)
  sz = tuple(int(m) for m in np.maximum(columnMax(cid), 0))
  cSize = np.zeros(sz)
  ok = np.all(cid > 0, axis=1)
  np.add.at(cSize, tuple((cid[ok] - 1).T), weights[ok])
  return cid, sz, cSize


# the clusters above minClusterSize, each with its members in their order: the index of the
# cluster into the output and the rows of clusterId that carry it
def _members(cid, sz, cSize, minClusterSize):
  ok = np.flatnonzero(np.all(cid > 0, axis=1))
  key = np.ravel_multi_index(tuple((cid[ok] - 1).T), sz) if len(ok) else np.zeros(0, dtype=np.int64)
  order = np.argsort(key, kind="stable")
  rows, key = ok[order], key[order]
  big = np.flatnonzero((cSize > minClusterSize).reshape(-1))
  lo, hi = np.searchsorted(key, big), np.searchsorted(key, big, side="right")
  return [(np.unravel_index(k, sz), rows[a:b]) for k, a, b in zip(big, lo, hi)]


# the axes of angles in the plane of a map as directions of the specimen, NaN where not computed
def _axes(plane, omega, frame):
  d = np.cos(omega)[..., None] * plane.e1 + np.sin(omega)[..., None] * plane.e2
  return vector3d(d, frame=frame, antipodal=True)


# ------------------------------------------------------------------ the map

def _ebsdTraces(ebsd, clusterId=None, method="Radon", minClusterSize=100, cutOff=6):
  l = ebsd.lattice
  if l is None:
    raise ValueError("the traces are computed on a map measured on a lattice")
  if method.lower() not in ("radon", "fourier"):
    raise ValueError(f"the method of calcTraces on a map is 'Radon' or 'Fourier', not {method!r}")
  radon = method.lower() == "radon"
  n = ebsd.size
  if clusterId is None:
    clusterId = (ebsd.isIndexed & ~ebsd.isPadding).reshape(-1).astype(np.int64)
  cid, sz, cSize = _clusters(clusterId, n, np.ones(n))
  P = l.plane.to2d(ebsd.pos.data.reshape(-1, 3))
  omega = np.full(sz, np.nan)
  rel = np.zeros(sz)
  for ic, pix in _members(cid, sz, cSize, minClusterSize):
    omega[ic], rel[ic] = _radonTrace(P[pix], l, cutOff) if radon else _fourierTrace(l.ij[pix], l.A)
  return _axes(l.plane, omega, ebsd.frame), rel, cSize


# the Radon transform of the cluster at every degree, each pixel split into four subpixels
# sharing their value linearly between the two nearest bins of unit width; the trace is
# perpendicular to the projection with the most energy at the higher frequencies, those
# up to cutOff left out as the shape of the cluster dominates them
def _radonTrace(P, l, cutOff):
  h = l.dxy
  sub = np.array([[a, b] for a in (-0.25, 0.25) for b in (-0.25, 0.25)]) @ l.A.T
  X = ((P[:, None, :] + sub[None, :, :]).reshape(-1, 2) - (columnMin(P) + columnMax(P)) / 2) / h
  rLast = int(np.ceil(np.max(np.linalg.norm(X, axis=1)))) + 1
  theta = np.arange(180) * degree
  s = X @ np.stack([np.cos(theta), np.sin(theta)]) + rLast
  lo = np.floor(s).astype(np.int64)
  frac = s - lo
  R = np.zeros((2 * rLast + 2, len(theta)))
  col = np.broadcast_to(np.arange(len(theta)), s.shape)
  np.add.at(R, (lo, col), (1 - frac) / 4)
  np.add.at(R, (lo + 1, col), frac / 4)
  R = R[:2 * rLast + 1]
  Z = np.abs(np.fft.rfft(R, axis=0))[cutOff + 1:rLast + 1].sum(axis=0)
  k = int(np.argmax(Z))
  if not Z[k] > 0:
    return np.nan, 0.0
  return theta[k] + np.pi / 2, (Z[k] - Z.mean()) / Z[k]


# the second moments of the Fourier transform of the cluster image within half the inner
# radius; the energy lies across the lamellae, so the trace is the axis of the smaller moment
def _fourierTrace(ij, A):
  lo = columnMin(ij)
  shape = tuple(columnMax(ij) - lo + 1)
  X = np.zeros(shape)
  X[tuple((ij - lo).T)] = 1
  Y = np.abs(np.fft.fftshift(np.fft.fft2(X - X.mean())))
  Y = (Y - Y.min()) / (np.ptp(Y) or 1)
  m = (np.array(shape) - 1) / 2
  d = np.stack(np.meshgrid(*[np.arange(k) - c for k, c in zip(shape, m)], indexing="ij"), axis=-1).reshape(-1, 2)
  Y = Y.reshape(-1) * (np.linalg.norm(d, axis=1) < np.min(m + 1) / 2)
  M = Y[:, None] * d
  lam, V = np.linalg.eigh(M.T @ M)
  # a cluster filling its bounding box has no direction
  if not lam[1] > 0:
    return np.nan, 0.0
  t = A @ V[:, 0]
  return np.arctan2(t[1], t[0]), (lam[1] - lam[0]) / lam[1]


# ------------------------------------------------------------------ the grains

def _grainTraces(grains, clusterId=None, method="shape", minClusterSize=100):
  if method not in ("shape", "calliper", "hist"):
    raise ValueError(f"the method of calcTraces on grains is 'shape', 'calliper' or 'hist', not {method!r}")
  gB = grains.boundary
  cid, sz, cSize = _clusters(clusterId, len(grains), np.asarray(grains.numPixel, dtype=float))
  omega = np.full(sz, np.nan)
  rel = np.zeros(sz)
  # the cluster of the grain on either side of every segment, and the segments of every cluster
  # in their order, a segment inside a cluster once
  ok = np.all(cid > 0, axis=1)
  clusterOf = np.full(len(grains), -1, dtype=np.int64)
  clusterOf[ok] = np.ravel_multi_index(tuple((cid[ok] - 1).T), sz) if ok.any() else 0
  side = _gather(clusterOf, _at(_positionTable(grains.id), gB.grainId), -1)
  seg = np.repeat(np.arange(len(gB)), 2)[side.reshape(-1) >= 0]
  key = side.reshape(-1)[side.reshape(-1) >= 0]
  order = np.lexsort((seg, key))
  seg, key = seg[order], key[order]
  once = np.ones(len(seg), dtype=bool)
  once[1:] = (seg[1:] != seg[:-1]) | (key[1:] != key[:-1])
  seg, key = seg[once], key[once]
  for k in np.flatnonzero((cSize > minClusterSize).reshape(-1)):
    ic = np.unravel_index(k, sz)
    F = gB.F[seg[np.searchsorted(key, k):np.searchsorted(key, k, side="right")]]
    if len(F) == 0:
      continue
    lS = gB.V[F[:, 0]] - gB.V[F[:, 1]]
    omega[ic], rel[ic] = _histTrace(lS) if method == "hist" else _shapeTrace(lS, method == "calliper")
  return _axes(gB.plane, omega, grains.frame), rel, cSize


# the maximum of the density of the segment directions, weighted by their lengths
def _histTrace(lS):
  from ..functions.s1fun import calcDensity
  f = calcDensity(np.arctan2(lS[:, 1], lS[:, 0]), weights=np.linalg.norm(lS, axis=1), sigma=10 * degree, antipodal=True)
  phi = np.linspace(0, 2 * np.pi, 360)
  v = np.real(f.eval(phi))
  k = int(np.argmax(v))
  return phi[k], (v[k] - 1) / v[k]


# the characteristic shape of the segments, sorted by angle and summed up: its principal
# axis, or the direction across its shortest calliper
def _shapeTrace(lS, calliper):
  lS = np.where(lS[:, 1:2] < 0, -lS, lS)
  dxy = np.vstack([lS, -lS])
  dxy = dxy[np.argsort(np.arctan2(dxy[:, 1], dxy[:, 0]), kind="stable")]
  xy = np.cumsum(dxy, axis=0)
  xy = xy - xy.mean(axis=0)
  if calliper:
    d = xy - np.roll(xy, -int(np.round(len(xy) / 2)), axis=0)
    delta = np.linalg.norm(d, axis=1)
    a, ib = delta.max(), int(np.argmin(delta))
    return np.pi / 2 + np.arctan2(d[ib, 1], d[ib, 0]), (a - delta[ib]) / a
  # the vertices weighted by half the length of their adjacent edges, as principalComponents
  dist = np.linalg.norm(xy[:-1] - xy[1:], axis=1)
  dist = 0.5 * (dist + np.roll(dist, 1))
  w = xy[:-1] * dist[:, None] * np.sum(xy[:-1] ** 2, axis=1, keepdims=True) ** 0.25
  lam, V = np.linalg.eigh(w.T @ w)
  a, b = np.sqrt(lam[1]), np.sqrt(lam[0])
  return np.mod(np.arctan2(V[1, 1], V[0, 1]), np.pi), (a - b) / a


def calcTraces(x, clusterId=None, *flags, **kwargs):
  """the traces of subregions of a map or of sets of grains, the direction their lamellae run
  along; MATLAB's calcTraces

    traces, rel, cSize = calcTraces(ebsd)                       # one trace of all indexed pixels, by the Radon transform
    traces, rel, cSize = calcTraces(ebsd, ebsd.phaseId, 'Fourier', minClusterSize=50)  # a trace per phase, Fourier based
    traces, rel, cSize = calcTraces(grains, grains.phaseId)    # a trace per phase from the characteristic shape
    traces, rel, cSize = calcTraces(grains, grains.phaseId, 'calliper')  # the shortest calliper of the shape instead

  A cluster is the set of pixels, or grains, with one value of every column of clusterId, as
  MATLAB's clusterId = [grainId, variantId]; ids of zero or NaN belong to no cluster. By
  default all indexed pixels, or all grains, form one cluster. The outputs have the shape of
  the largest id of every column. A cluster of minClusterSize pixels or fewer (default 100)
  keeps a NaN trace and a reliability of zero.

  On a map, measured on a lattice, the method is:

  - 'Radon', the default: perpendicular to the projection of the cluster with the most
  energy at high frequencies
  - 'Fourier': the axis of the smaller second moment of the cluster image's Fourier transform

  On grains it is:

  - 'shape', the default: the principal axis of the characteristic shape of the boundaries
  - 'calliper': the direction across the shortest calliper of that shape
  - 'hist': the maximum of the length weighted density of the segment directions

  The method is a positional string or method=. A cluster filling a rectangle has no
  direction: 'Fourier' then gives NaN, 'Radon' a side of the rectangle with high reliability.

  Input
    x - EBSD on a lattice, or grain2d
  Output
    traces - vector3d, antipodal directions in the plane of the map, one per cluster
    rel - the reliability, how much the winning direction stands out, zero for none
    cSize - the size of every cluster, pixels on a map, summed numPixel of the grains
  Keywords
    clusterId - one or more columns of ids, a row per pixel or grain, naming the cluster
        of each; by default one cluster
  References
    Determination of child phase habit planes from two-dimensional reconstructed parent
        phase orientation maps, arXiv 2201.02103 (2022)
  See also
    grainBoundary.characteristicShape, parentGrainReconstructor"""
  if isinstance(clusterId, str):
    clusterId, flags = None, (clusterId,) + flags
  if flags:
    kwargs["method"] = flags[0]
  if isinstance(x, EBSD):
    return _ebsdTraces(x, clusterId, **kwargs)
  if isinstance(x, grain2d):
    return _grainTraces(x, clusterId, **kwargs)
  raise TypeError(f"calcTraces is defined for EBSD maps and grains, not {type(x).__name__}")


install((EBSD, grain2d), calcTraces)

__all__ = ["calcTraces"]
