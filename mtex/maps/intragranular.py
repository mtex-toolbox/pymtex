# intragranular quantities: how the orientation changes from pixel to pixel (MATLAB
# @EBSD/gradient, gradientX, gradientY, gradientZ, curvature, KAM, calcGROD, grainMean,
# spatialProfile, calcMisorientation; Julia intragranular.jl)
#
#   gx, gy = ebsd.gradient(), ebsd.gradient(stencil='full'), ebsd.gradient(v), ebsd.gradientX()
#   ebsd.curvature(), ebsd.KAM(order=2, threshold=10 * degree, max=True, weights=f, radius=r)
#   ebsd.calcGROD(grains), ebsd.calcGROD(grains, rotRef), ebsd.grainMean(prop, grains, method, ulim=, llim=)
#   ebsd.spatialProfile(xy), ebsd.spatialProfile(x, y), ebsd.calcMisorientation(ebsd2, sampleSize=, minDistance=)
#
# Every one of these reads the map through its lattice, so a rotated, a sheared, a
# hexagonal grid and a list that is on a lattice without being stored as one are the same
# code. One gradient, three neighbourhoods, one solver: the one sided difference and the two
# least squares fits are the same normal equations over a different set of neighbours. The
# gradient is taken along the axes of the plane of the map and returned as a framed vector,
# the left tangent vector at the orientation of its own pixel, whose components live in the
# specimen frame. A difference across a grain boundary is not a derivative: a pair is
# compared only when both sides are indexed, of one phase and, where the map knows its
# grains, of one grain.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import backend, closestImage, qmul
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.symreduce import chordAngle
from ..geometry.tangent import rotationVector
from ..geometry.vector3d import vector3d
from .ebsd import EBSD
from ..geometry.install import install

IDENTITY = np.array([[1.0, 0.0, 0.0, 0.0]])


def _conj(q):
  return q * np.array([1, -1, -1, -1], dtype=q.dtype)


# ------------------------------------------------------------------ the neighbourhood

def mapLattice(ebsd):
  """the lattice a map is measured on, or an error naming what is missing"""
  if ebsd.lattice is None:
    raise ValueError("the map is not on a lattice, so it has no neighbours")
  return ebsd.lattice


def latticeLookup(l):
  """the measurement at every lattice index, -1 where the lattice carries none; built once per
  index array, which a lattice never changes, and read only"""
  cached = getattr(l, "_lookup", None)
  if cached is not None and cached[0] is l.ij:
    return cached[1]
  lk = np.full((int(l.ij[:, 0].max()) + 1, int(l.ij[:, 1].max()) + 1), -1, dtype=np.int64)
  lk[l.ij[:, 0], l.ij[:, 1]] = np.arange(len(l.ij))
  lk.flags.writeable = False
  l._lookup = (l.ij, lk)
  return lk


def neighbourAt(lk, ij, off):
  """the measurement at every index shifted by an offset, -1 where the map has none, and
  whether the shifted index is still inside the lattice"""
  i, j = ij[:, 0] + off[0], ij[:, 1] + off[1]
  inside = (i >= 0) & (i < lk.shape[0]) & (j >= 0) & (j < lk.shape[1])
  nb = np.full(len(ij), -1, dtype=np.int64)
  nb[inside] = lk[i[inside], j[inside]]
  return nb, inside


class mapPairs:
  """what a comparison of two measurements needs, gathered once: a pair the orientation field
  is continuous across is indexed on both sides, of one phase and of one grain"""

  def __init__(self, ebsd):
    self.rot = np.ascontiguousarray(ebsd.rotations.data, dtype=float).reshape(-1, 4)
    # a NaN rotation is NaN in every coordinate, so the first tells
    self.indexed = ebsd.isIndexed.reshape(-1) & ~np.isnan(self.rot[:, 0])
    self.phase = ebsd.phaseId.reshape(-1)
    self.grain = ebsd.prop["grainId"].reshape(-1) if "grainId" in ebsd.prop else None
    self.G = {int(p): ebsd.CSList[int(p)].sym.properGroup().rotations().data.reshape(-1, 4).astype(float)
              for p in np.unique(self.phase) if ebsd.CSList.isIndexed(int(p))}

  def groupTable(self):
    """the proper groups as one table indexed by phase, and their sizes"""
    top = int(self.phase.max(initial=0)) + 1
    width = max([len(G) for G in self.G.values()] + [1])
    Gall = np.zeros((top, width, 4))
    nG = np.zeros(top, dtype=np.int64)
    for p, G in self.G.items():
      Gall[p, :len(G)] = G
      nG[p] = len(G)
    return Gall, nG

  def arrays(self):
    """what the kernels read: the grain ids, or a dummy where the map has none"""
    grain = np.zeros(len(self.rot), dtype=np.int64) if self.grain is None else np.ascontiguousarray(self.grain, dtype=np.int64)
    return np.ascontiguousarray(self.indexed), np.ascontiguousarray(self.phase, dtype=np.int64), grain, self.grain is not None

  def comparable(self, a, b):
    """the mask of the pairs (a, b) the orientation field is continuous across: b a measurement,
    both indexed, of one phase and, where the map has grains, of one grain"""
    ok = (b >= 0)
    bb = np.maximum(b, 0)
    ok &= self.indexed[a] & self.indexed[bb] & (self.phase[a] == self.phase[bb])
    if self.grain is not None:
      ok &= self.grain[a] == self.grain[bb]
    return ok

  def closest(self, a, b):
    """the image of the orientation of b closest to that of a, phase by phase"""
    out = np.empty((len(a), 4))
    for p, G in self.G.items():
      sel = self.phase[a] == p
      if sel.any():
        out[sel] = closestImage(self.rot[a[sel]], self.rot[b[sel]], IDENTITY, G)[1]
    return out

  def tangentTo(self, a, b):
    """the left tangent vector pointing from a to b, in specimen coordinates"""
    return rotationVector(qmul(self.closest(a, b), _conj(self.rot[a])))

  def misAngle(self, a, b):
    """the misorientation angle of a pair, from the chord to the closest element"""
    return chordAngle(self.rot[a], self.closest(a, b))


def neighbourBall(l, order=None, radius=None):
  """the offsets reachable in that many steps over the stencil of the lattice, or within a
  physical distance over as many steps as it takes to reach it"""
  st = [tuple(int(x) for x in o) for o in l.stencil]
  if radius is not None:
    step = min(np.linalg.norm(l.A @ np.array(o, dtype=float)) for o in st)
    order = int(np.ceil(radius / step)) + 1
  seen, front = {(0, 0)}, [(0, 0)]
  for _ in range(order):
    nxt = []
    for f in front:
      for s in st:
        v = (f[0] + s[0], f[1] + s[1])
        if v not in seen:
          seen.add(v)
          nxt.append(v)
    front = nxt
  offs = np.array(sorted(v for v in seen if v != (0, 0)), dtype=np.int64).reshape(-1, 2)
  if radius is not None:
    offs = offs[np.linalg.norm(offs @ l.A.T, axis=1) <= radius]
  return offs


# ------------------------------------------------------------------ the gradient

STENCILS = ("oneSided", "lattice", "full")


def gradient(ebsd, direction=None, stencil="oneSided"):
  """the derivative of the orientation along the two axes of the plane of the map, per unit
  length, as a pair of framed vectors; or along a direction of the specimen, which has
  none where it leaves the plane. The stencil says which neighbours it is taken from: the
  one sided difference along the two lattice directions, stepping backwards only where the
  map ends and NaN where the neighbour it asks for is missing inside the map; the whole
  stencil of the lattice, the central difference in the interior; or the same with the
  diagonals. A pixel whose neighbours do not span two directions has none, and is NaN"""
  if direction is not None:
    gx, gy = gradient(ebsd, stencil=stencil)
    pl = mapLattice(ebsd).plane
    d = np.asarray(direction.data if isinstance(direction, vector3d) else direction, dtype=float).reshape(3)
    d = d / np.linalg.norm(d) if np.linalg.norm(d) else d
    if abs(d @ pl.normal) > 1e-10:
      return vector3d(np.full(ebsd.shape + (3,), np.nan), frame=ebsd.frame)
    return vector3d((d @ pl.e1) * gx.data + (d @ pl.e2) * gy.data, frame=ebsd.frame)
  # MATLAB's name of the axial stencil
  stencil = "lattice" if stencil == "1hop" else stencil
  if stencil not in STENCILS:
    raise ValueError(f"the stencil is 'oneSided', 'lattice' ('1hop') or 'full', not {stencil!r}")
  l = mapLattice(ebsd)
  lk, m = latticeLookup(l), mapPairs(ebsd)
  if stencil == "oneSided":
    offs = [(1, 0), (0, 1)]
  elif stencil == "lattice":
    offs = [tuple(int(x) for x in o) for o in l.stencil]
  else:
    offs = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]
  offs = np.array(offs, dtype=np.int64).reshape(-1, 2)
  D = offs @ l.A.T.astype(float)
  gx = gy = None
  if backend() == "numba":
    from .kernels import pixelKernel
    loop = pixelKernel("gradient")
    if loop is not None:
      n = len(l.ij)
      gx, gy = np.empty((n, 3)), np.empty((n, 3))
      indexed, phase, grain, hasGrain = m.arrays()
      Gall, nG = m.groupTable()
      loop(m.rot, indexed, phase, grain, hasGrain, lk, np.ascontiguousarray(l.ij, dtype=np.int64), offs, np.ascontiguousarray(D), stencil == "oneSided",
           Gall, nG, gx, gy)
  if gx is None:
    gx, gy = _gradientNumpy(l, lk, m, offs, D, stencil == "oneSided")
  return vector3d(gx.reshape(ebsd.shape + (3,)), frame=ebsd.frame), vector3d(gy.reshape(ebsd.shape + (3,)), frame=ebsd.frame)


# the same normal equations assembled with bincount over all pixels, one offset at a time
def _gradientNumpy(l, lk, m, offs, D, oneSided):
  n = len(l.ij)
  m11 = m12 = m22 = np.zeros(n)
  r1, r2 = np.zeros((n, 3)), np.zeros((n, 3))
  bad = ~m.indexed
  k = np.arange(n)
  for o, d0 in zip(offs, D):
    o = (int(o[0]), int(o[1]))
    nb, inside = neighbourAt(lk, l.ij, o)
    d = np.broadcast_to(d0, (n, 2)).copy()
    if oneSided:
      # the one sided stencil steps backwards where the map ends, and nowhere else
      back = ~inside
      nb[back] = neighbourAt(lk, l.ij[back], (-o[0], -o[1]))[0]
      d[back] *= -1
    ok = m.comparable(k, nb)
    if oneSided:
      bad |= ~ok
    if not ok.any():
      continue
    h = m.tangentTo(k[ok], nb[ok])
    d = d[ok]
    m11 = m11 + np.bincount(k[ok], d[:, 0] ** 2, n)
    m12 = m12 + np.bincount(k[ok], d[:, 0] * d[:, 1], n)
    m22 = m22 + np.bincount(k[ok], d[:, 1] ** 2, n)
    r1[ok] += d[:, :1] * h
    r2[ok] += d[:, 1:] * h
  # two independent neighbour directions are what makes the fit solvable
  det = m11 * m22 - m12 ** 2
  s = ~bad & (det > 1e-10 * m11 * m22) & (det > 0)
  gx, gy = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
  gx[s] = (m22[s, None] * r1[s] - m12[s, None] * r2[s]) / det[s, None]
  gy[s] = (m11[s, None] * r2[s] - m12[s, None] * r1[s]) / det[s, None]
  return gx, gy


def gradientX(ebsd, stencil="oneSided"):
  """the derivative of the orientation along the specimen x axis per unit length, gradient(ebsd,
  xvector); the stencil as for gradient"""
  return gradient(ebsd, [1.0, 0.0, 0.0], stencil=stencil)


def gradientY(ebsd, stencil="oneSided"):
  """the derivative of the orientation along the specimen y axis per unit length, gradient(ebsd,
  yvector); the stencil as for gradient"""
  return gradient(ebsd, [0.0, 1.0, 0.0], stencil=stencil)


def gradientZ(ebsd, stencil="oneSided"):
  """the derivative of the orientation along the specimen z axis per unit length, gradient(ebsd,
  zvector), which a planar map has only where z lies in its plane"""
  return gradient(ebsd, [0.0, 0.0, 1.0], stencil=stencil)


# ------------------------------------------------------------------ the curvature

def curvature(ebsd, stencil="oneSided"):
  """the curvature tensor of every pixel, kappa = d omega / d x, in the specimen frame per scan
  unit, its rows the components of the rotation vector and its columns the directions of the
  specimen. A two dimensional map says nothing about how the orientation changes along its
  normal, so every column the normal reaches is unknown and NaN"""
  from ..tensors.tensor import curvatureTensor
  gx, gy = gradient(ebsd, stencil=stencil)
  pl = mapLattice(ebsd).plane
  unknown = np.abs(pl.normal) > 1e-10
  a, b = np.ascontiguousarray(gx.data, dtype=float), np.ascontiguousarray(gy.data, dtype=float)
  loop = None
  if backend() == "numba":
    from .kernels import pixelKernel
    loop = pixelKernel("planeTensor")
  if loop is not None:
    M = np.empty(a.shape + (3,))
    loop(a.reshape(-1, 3), b.reshape(-1, 3), np.asarray(pl.e1, dtype=float), np.asarray(pl.e2, dtype=float), unknown, M.reshape(-1, 3, 3))
  else:
    M = a[..., :, None] * pl.e1[None, :] + b[..., :, None] * pl.e2[None, :]
    M[..., :, unknown] = np.nan
  return curvatureTensor(M, ebsd.frame, unit=f"1/{ebsd.scanUnit}")


# ------------------------------------------------------------------ the weighted Burgers vector

def weightedBurgersVec(ebsd, *args, windowSize=1, stencil="oneSided", gradient=False):
  """weightedBurgersVec(ebsd), weightedBurgersVec(ebsd, windowSize=2), weightedBurgersVec(ebsd,
  'gradient', stencil='full'): the net Burgers vector crossing the map per unit area, after
  Wheeler et al. (2009), a vector per pixel in the specimen frame per scan unit. The integral
  method sums the crystal coordinates of the specimen x and y axes around a square loop of
  2 windowSize + 1 pixels, on the gridded map, NaN where the loop leaves the grain or the
  map; the gradient method is the column of Nye's tensor along the normal of the map"""
  from scipy.ndimage import correlate, maximum_filter, minimum_filter
  from ..analysis.mechanics import dislocationDensity
  if gradient or "gradient" in args:
    alpha = dislocationDensity(curvature(ebsd, stencil=stencil))
    return vector3d(alpha.M[..., :, 2], frame=ebsd.frame)
  if ebsd.ndim != 2:
    raise ValueError("the integral method runs on the rows and columns of a map: gridify it first")
  # every orientation the equivalent nearest the last measured one of its grain, as MATLAB's
  # projectIntoFundamentalRegion of a map, so that the loop sums no jumps between equivalents
  # a map without grains is one grain
  gid = np.asarray(ebsd.prop["grainId"]) if "grainId" in ebsd.prop else np.ones(ebsd.shape)
  gid = np.where(ebsd.isIndexed, gid, 0).reshape(-1).astype(np.int64)
  last = np.zeros(gid.max(initial=0) + 1, dtype=np.int64)
  np.maximum.at(last, gid, np.arange(gid.size))
  oris = ebsd.orientations
  ori = oris.projectIntoFundamentalRegion(oris.reshape(-1)[last[gid]].reshape(ebsd.shape))
  oriX, oriY = ((ori.inv() * vector3d(v, frame=ebsd.frame)).data for v in ([1.0, 0, 0], [0, 1.0, 0]))
  w = int(windowSize)
  fY = np.zeros((2 * w + 1, 2 * w + 1))
  fY[:, 0], fY[:, -1] = -1, 1
  fY[[0, -1], 0], fY[[0, -1], -1] = -0.5, 0.5
  fX = -fY.T
  W = np.stack([correlate(oriX[..., c], fX, mode="constant") + correlate(oriY[..., c], fY, mode="constant") for c in range(3)], axis=-1)
  # a loop reaching into another grain, a missing pixel or over the rim has no value
  g = np.where(ebsd.isIndexed, gid.reshape(ebsd.shape).astype(float), -1.0)
  W[maximum_filter(g, 2 * w + 1, mode="constant", cval=-2) != minimum_filter(g, 2 * w + 1, mode="constant", cval=-2)] = np.nan
  d = float(np.min(np.linalg.norm(mapLattice(ebsd).A, axis=0)))
  W = ori * vector3d(W, frame=ori.CS)
  return vector3d(W.data / (4 * w ** 2 * d), frame=ebsd.frame)


# ------------------------------------------------------------------ KAM

def KAM(ebsd, order=1, radius=None, threshold=np.inf, max=False, weights=None):
  """the kernel average misorientation: the mean misorientation angle of a pixel to its
  neighbours, NaN where it has none to compare with. The neighbours are the ones within
  order steps over the lattice stencil, or within a physical radius; threshold drops a pair
  above an angle, max takes the largest angle instead of the mean, and weights is a
  function of the distance"""
  l = mapLattice(ebsd)
  offs = neighbourBall(l, order=order, radius=radius)
  w = np.ones(len(offs)) if weights is None else np.array([float(weights(np.linalg.norm(l.A @ o))) for o in offs])
  lk, m = latticeLookup(l), mapPairs(ebsd)
  out = None
  if backend() == "numba":
    from .kernels import pixelKernel
    loop = pixelKernel("kam")
    if loop is not None:
      out = np.empty(len(l.ij))
      indexed, phase, grain, hasGrain = m.arrays()
      Gall, nG = m.groupTable()
      loop(m.rot, indexed, phase, grain, hasGrain, lk, np.ascontiguousarray(l.ij, dtype=np.int64), np.ascontiguousarray(offs, dtype=np.int64),
           np.ascontiguousarray(w, dtype=float), float(threshold), bool(max), Gall, nG, out)
  if out is None:
    out = _kamNumpy(l, lk, m, offs, w, threshold, max)
  return out.reshape(ebsd.shape)


# the same sums one offset at a time over all pixels
def _kamNumpy(l, lk, m, offs, w, threshold, maximal):
  n = len(l.ij)
  k = np.arange(n)
  acc, cnt, best = np.zeros(n), np.zeros(n), np.zeros(n)
  found = np.zeros(n, dtype=bool)
  for o, wo in zip(offs, w):
    nb = neighbourAt(lk, l.ij, (int(o[0]), int(o[1])))[0]
    ok = m.comparable(k, nb)
    if not ok.any():
      continue
    ang = m.misAngle(k[ok], nb[ok])
    keep = np.isfinite(ang) & (ang <= threshold)
    idx = k[ok][keep]
    found[idx] = True
    best[idx] = np.maximum(best[idx], ang[keep])
    acc[idx] += wo * ang[keep]
    cnt[idx] += wo
  out = np.full(n, np.nan)
  out[found] = best[found] if maximal else acc[found] / cnt[found]
  return out


# ------------------------------------------------------------------ GROD and grain means

def calcGROD(ebsd, grains=None, rotRef=None):
  """the grain reference orientation deviation, mis2mean: the misorientation of every pixel
  from the mean orientation of its grain, or from a given reference. The reference is
  taken over the symmetric equivalent closest to the pixel, so the deviation is the small
  rotation it is meant to be"""
  n = ebsd.size
  if rotRef is None:
    if grains is None:
      raise ValueError("calcGROD needs the grains, or a reference rotation")
    if "grainId" not in ebsd.prop:
      raise ValueError("the map carries no grain id; reconstruct it with calcGrains(ebsd)")
    byId = np.full(int(grains.id.max(initial=0)) + 1, -1, dtype=np.int64)
    byId[grains.id] = np.arange(len(grains))
    gid = ebsd.prop["grainId"].reshape(-1).astype(np.int64)
    pos = np.where((gid > 0) & (gid < len(byId)), byId[np.clip(gid, 0, len(byId) - 1)], -1)
    # one gather from the mean rotations behind a NaN row, which a pixel without a grain takes
    means = np.vstack([np.full((1, 4), np.nan), grains.meanRotation.data.reshape(-1, 4)])
    ref = np.take(means, pos + 1, axis=0)
  else:
    ref = np.broadcast_to(np.asarray(rotRef.data if isinstance(rotRef, rotation) else rotRef, dtype=float).reshape(-1, 4), (n, 4))
  ids = ebsd.indexedPhaseId
  if len(ids) != 1:
    raise ValueError("a grain reference deviation mixes no phases, so select one first: calcGROD(ebsd['name'], grains)")
  cs = ebsd.CSList[int(ids[0])]
  G = cs.sym.properGroup().rotations().data.reshape(-1, 4).astype(float)
  rot = np.ascontiguousarray(ebsd.rotations.data, dtype=float).reshape(-1, 4)
  # a NaN rotation is NaN in every coordinate, so the first tells
  ind = ebsd.isIndexed.reshape(-1) & ~np.isnan(ref[:, 0])
  out = None
  if backend() == "numba":
    from .kernels import pixelKernel
    loop = pixelKernel("grod")
    if loop is not None:
      out = np.empty((n, 4))
      loop(rot, np.ascontiguousarray(ref, dtype=float), np.ascontiguousarray(ind), G, len(G), out)
  if out is None:
    out = np.full((n, 4), np.nan)
    img = closestImage(rot[ind], ref[ind], IDENTITY, G)[1]
    out[ind] = qmul(_conj(img), rot[ind])
  return orientation._fromRotation(out.reshape(ebsd.shape + (4,)), (cs, cs))


def grainMean(ebsd, prop, grains=None, method=np.mean, ulim=np.inf, llim=-np.inf):
  """the average of a property over each grain: one value per grain when the grains are
  given, and the value of its own grain at every pixel when they are not; method is the
  reduction, llim and ulim the range of values it is taken over"""
  if "grainId" not in ebsd.prop:
    raise ValueError("the map carries no grain id; reconstruct it with calcGrains(ebsd)")
  v = np.asarray(prop, dtype=float).reshape(-1)
  if len(v) != ebsd.size:
    raise ValueError(f"the property has {len(v)} values for {ebsd.size} measurements")
  gid = ebsd.prop["grainId"].reshape(-1).astype(np.int64)
  ok = (gid > 0) & (v >= llim) & (v <= ulim)
  n = int(gid.max(initial=0))
  per = np.full(n + 1, np.nan)
  order = np.argsort(gid[ok], kind="stable")
  g, vs = gid[ok][order], v[ok][order]
  bounds = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1], [True]]))
  for a, b in zip(bounds[:-1], bounds[1:]):
    per[g[a]] = method(vs[a:b])
  if grains is not None:
    ids = grains.id.astype(np.int64)
    return np.where(ids <= n, per[np.minimum(ids, n)], np.nan)
  return np.where(gid > 0, per[gid], np.nan).reshape(ebsd.shape)


# ------------------------------------------------------------------ profiles and pairs

def spatialProfile(ebsd, lineXY, y=None, radius=None):
  """the measurements along a polyline, in the order they are met, and the distance along the
  line of each of them, as MATLAB's pair; a pixel belongs to the line when it lies within
  radius of it, half a step by default. The line is an (N, 2) array of x and y, or x and y
  as two lists"""
  if y is not None:
    lineXY = np.stack([np.asarray(lineXY, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)], axis=1)
  Q = np.asarray(lineXY.data if isinstance(lineXY, vector3d) else lineXY, dtype=float)
  l = mapLattice(ebsd)
  pl = l.plane
  if Q.ndim != 2 or Q.shape[1] not in (2, 3) or len(Q) < 2 or not np.all(np.isfinite(Q)):
    raise ValueError("the profile is a finite (N, 2) array of at least two points")
  Q = pl.to2d(Q) if Q.shape[1] == 3 else Q
  P = pl.to2d(ebsd.pos.data.reshape(-1, 3))
  r = l.dxy / 2 if radius is None else float(radius)
  idx, dist = [], []
  base = 0.0
  for a, b in zip(Q[:-1], Q[1:]):
    L = np.linalg.norm(b - a)
    if L == 0:
      continue
    u = (b - a) / L
    d = P - a
    t = d @ u
    hit = np.flatnonzero((t >= 0) & (t <= L) & (np.abs(u[0] * d[:, 1] - u[1] * d[:, 0]) <= r))
    for k in hit[np.argsort(t[hit], kind="stable")]:
      if idx and idx[-1] == k:
        continue
      idx.append(int(k))
      dist.append(base + t[k])
    base += L
  return ebsd[np.array(idx, dtype=int)], np.array(dist)


def calcMisorientation(ebsd, other=None, sampleSize=100000, minDistance=None, rng=None):
  """uncorrelated misorientations: random pairs of measurements far enough apart that they say
  nothing about one another, which is what an uncorrelated distribution is compared
  against. Both maps must carry one indexed phase; the pair is written as a boundary
  writes it, inv(o2) o1 carrying a direction of the first phase into the second"""
  other = ebsd if other is None else other
  # the pixels of the one indexed phase of each map, of which only the drawn ones are read
  p1 = ebsd._onePhase()
  idx1 = np.flatnonzero(ebsd.phaseId.reshape(-1) == p1)
  p2, idx2 = (p1, idx1) if other is ebsd else (other._onePhase(), None)
  idx2 = np.flatnonzero(other.phaseId.reshape(-1) == p2) if idx2 is None else idx2
  P1, P2 = ebsd.pos.data.reshape(-1, 3), other.pos.data.reshape(-1, 3)
  rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
  # far enough apart is a hundredth of the map by default, as in MTEX
  if minDistance is None:
    # column by column: NumPy's reduction over the long axis of an (n, 3) array is twenty times slower
    cols = [P1[:, k] if len(idx1) == len(P1) else P1[idx1, k] for k in range(3)]
    d0 = np.linalg.norm([c.max() - c.min() for c in cols]) / 100
  else:
    d0 = float(minDistance)
  i1, i2 = idx1[rng.integers(0, len(idx1), sampleSize)], idx2[rng.integers(0, len(idx2), sampleSize)]
  keep = np.linalg.norm(P1[i1] - P2[i2], axis=1) > d0
  i1, i2 = i1[keep], i2[keep]
  # inv(o1) * o2 as MATLAB's o1 .\ o2: the second crystal's coordinates into the first's
  q = qmul(_conj(ebsd.rotations.data.reshape(-1, 4)[i1].astype(float)), other.rotations.data.reshape(-1, 4)[i2].astype(float))
  cs1, cs2 = ebsd.CSList[p1], other.CSList[p2]
  return orientation._fromRotation(q, (cs2, cs1), antipodal=cs1 is cs2)


for _name, _f in (("gradient", gradient), ("gradientX", gradientX), ("gradientY", gradientY), ("gradientZ", gradientZ), ("curvature", curvature),
                  ("KAM", KAM), ("calcGROD", calcGROD), ("grainMean", grainMean), ("spatialProfile", spatialProfile), ("calcMisorientation", calcMisorientation), ("weightedBurgersVec", weightedBurgersVec)):
  install(EBSD, _f, _name)

__all__ = ["neighbourBall"]
