# the texture characteristics of an ODF (MATLAB @SO3Fun: textureindex, entropy, volume,
# fibreVolume, max, min, calcComponents, calcMDF, calcAngleDistribution, calcAxisDistribution,
# calcMIndex, discreteSample, calcError, calcKernel, centerSpecimen; docs/plans/functions.md F6)
#
#   textureindex(odf), entropy(odf), volume(odf, center, radius), fibreVolume(odf, h, r, radius)
#   values, modes = max(odf, numLocal=, resolution=, accuracy=);  min(odf)
#   modes, weights, centerId = calcComponents(odf, seed=, weights=, tolerance=, radius=, exact=)
#   calcMDF(odf1, odf2, bandwidth=), calcAngleDistribution(odf | cs1, cs2 | mori), calcAxisDistribution(odf, v)
#   calcMIndex(odf), discreteSample(odf, n, rng=), discreteSample(fun, n, range=), calcError(odf1, odf2, 'l2'), calcKernel(ori | v | ebsd, 'UCV')
from __future__ import annotations

import functools
import math
import types

import numpy as np

from ..geometry._array import MtexArray
from ..geometry.fibre import fibre
from ..geometry.frames import crystalFrame, referenceFrame
from ..geometry.kernels import backend
from ..geometry.grids.localgrids import localOrientationGrid, localS2Grid
from ..geometry.grids.fibonacci import fibonacciS2Grid
from ..geometry.grids.s2grid import equispacedS2Grid
from ..geometry.grids.so3grid import equispacedSO3Grid
from ..geometry.orientation import orientation
from ..geometry.quaternion import qrot, quaternion, rotation
from ..geometry.session import getMTEXpref
from ..geometry.symreduce import reductionGroups, symDots
from ..geometry.tangent import SO3TangentVector
from ..geometry.units import degree
from ..geometry.vector3d import vector3d, xvector, yvector, zvector
from .bingham import S2FunBingham, SO3FunBingham, binghamSample
from .kernels import rbfSample
from .funops import conv
from .handles import S2FunHandle, S2FunSum, SO3FunComposition, SO3FunGeneric, SO3FunHandle
from .kernelfunctions import SO3DeLaValleePoussinKernel
from .quadrature import bandwidthFor, quadratureS2Grid, quadratureSO3Grid
from .rbf import SO3FunCBF, SO3FunRBF, _rotationsFor, uniformODF
from .s2fun import S2FunHarmonic
from .s2funtri import S2FunTri
from .so3fun import SO3FunHarmonic
from ..geometry.install import install

SO3_KINDS = (SO3FunHarmonic, SO3FunRBF, SO3FunCBF, SO3FunGeneric)
S2_KINDS = (S2FunHarmonic, S2FunHandle, S2FunSum, S2FunBingham, S2FunTri)


def _isSO3(f):
  return isinstance(f, SO3_KINDS)


def _ori(r, f):
  return orientation._fromRotation(r.data if isinstance(r, quaternion) else np.asarray(r), (f.CS, f.SS), f.antipodal)


# ------------------------------------------------------------------ scalars


def textureindex(f):
  """the texture index of a function on SO(3) or on the sphere, the squared L2 norm of its harmonic
  expansion"""
  return (f if isinstance(f, SO3FunHarmonic) else SO3FunHarmonic(f)).norm() ** 2 if _isSO3(f) else S2FunHarmonic(f).norm() ** 2


def entropy(f, bandwidth=None):
  """-int f log f by the exact rule at a raised bandwidth, since f log f is not bandlimited"""
  if _isSO3(f):
    L0 = max(2 * f.bandwidth, 64) if bandwidth is None else int(bandwidth)
    L = bandwidthFor(min(L0, 2 * getMTEXpref("maxSO3Bandwidth")), f.CS, f.SS)
    g = quadratureSO3Grid(L, f.CS, f.SS)
    v = np.real(f(g if isinstance(f, SO3FunHarmonic) else g.nodes()))
    return float(-np.sum(g.weights * np.where(v > 0, v * np.log(np.where(v > 0, v, 1.0)), 0.0)))
  L0 = max(2 * f.bandwidth, 128) if bandwidth is None else int(bandwidth)
  g = quadratureS2Grid(min(L0, 2 * getMTEXpref("maxS2Bandwidth")))
  v = np.real(f(g if isinstance(f, S2FunHarmonic) else g.nodes()))
  return float(-np.sum(g.weights * np.where(v > 0, v * np.log(np.where(v > 0, v, 1.0)), 0.0)) / (4 * np.pi))


def volume(f, center, radius, resolution=None):
  """the share of orientations within a radius of a centre: the mean of f on a local grid in
  the ball, times the ball's share of the fundamental region (its Haar measure times the
  number of symmetric copies, at most one); a list of orientations gives the count instead"""
  if isinstance(f, orientation):
    if isinstance(center, fibre):
      return float(np.count_nonzero(center.angle(f) <= radius) / f.size)
    return float(np.count_nonzero(f.angle(center) <= radius) / f.size)
  if isinstance(center, fibre):
    return fibreVolume(f, center, radius)
  if center.size > 1:
    return np.array([volume(f, center[i], radius, resolution) for i in range(center.size)]).reshape(center.shape)
  if isinstance(f, S2_KINDS):
    return _volumeS2(f, center, radius, resolution)
  _rotationsFor(f, center)
  res = min(1.25 * degree, radius / 30) if resolution is None else resolution
  L, R = reductionGroups(center)
  g = localOrientationGrid(center, maxAngle=radius, resolution=res, respectSymmetry=False)
  if g.size == 0:
    return 0.0
  share = min(1.0, len(L) * len(R) * (2 if f.antipodal else 1) * (radius - math.sin(radius)) / math.pi)
  return float(np.mean(np.real(f(g))) * share)


# the share of a spherical density within a cap: its mean over the cap times the cap's share
# of the sphere, over the orbit of the centre under the group and both caps of an axis, S2Fun/volume
def _volumeS2(f, center, radius, resolution=None):
  from ..geometry.grids.localgrids import localS2Grid
  res = min(0.1 * degree, radius / 200) if resolution is None else resolution
  g = localS2Grid(center, maxTheta=radius, resolution=res, respectSymmetry=False)
  if g.size == 0:
    return 0.0
  orbit = center.symmetrise().size if getattr(getattr(f, "frame", None), "sym", None) is not None else 1
  share = min(1.0, orbit * (2 if f.antipodal else 1) * (1 - math.cos(radius)) / 2)
  return float(np.mean(np.real(f(g))) * share)


def fibreVolume(f, h, r=None, radius=None, resolution=None):
  """the share of crystal directions within a radius of h among those mapped to r: the mean
  of the inverse pole figure over the cap, times the cap's share of the sphere over the orbit of h
  the share of the tube about the fibre: the function averaged over the orientations that
  send h within the radius of r, u = g h on a cap about r and the turn about u equispaced,
  times the tube's share of orientation space over the copies of h under the group"""
  if isinstance(h, fibre):
    h, r, radius = h.h, h.r, r
  res = min(1 * degree, radius / 30) if resolution is None else resolution
  # the cap about r as rings of equal width, every point weighted by the area it stands for
  m = max(int(np.ceil(radius / res)), 1)
  dtheta = radius / m
  theta = (np.arange(m) + 0.5) * dtheta
  counts = np.maximum(np.round(2 * np.pi * np.sin(theta) / dtheta).astype(int), 1)
  th = np.repeat(theta, counts)
  ph = np.concatenate([np.linspace(0, 2 * np.pi, c, endpoint=False) for c in counts])
  wcap = np.repeat(np.sin(theta) * dtheta / counts, counts)
  local = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
  rd = vector3d(r.data.reshape(3)).normalize().data.reshape(3)
  u = qrot(np.tile(rotation.map(np.array([0.0, 0.0, 1.0]), rd).data.reshape(1, 4), (len(local), 1)), local)
  hd = np.broadcast_to(vector3d(h.data.reshape(3)).normalize().data, u.shape)
  toU = rotation.map(hd, u)
  n = max(int(np.ceil(2 * np.pi / res)), 36)
  omega = np.linspace(0, 2 * np.pi, n, endpoint=False)
  about = rotation.byAxisAngle(vector3d(np.repeat(u, n, axis=0)), np.tile(omega, len(u)))
  q = (about * rotation(np.repeat(toU.data, n, axis=0))).data
  g = orientation._fromRotation(q, (f.CS, f.SS), getattr(f, "antipodal", False))
  vals = np.real(f(g)).reshape(len(u), n).mean(axis=1)
  copies = h.symmetrise(unique=True).size if isinstance(h, vector3d) and h.frame is not None else 1
  return float(min(1.0, np.sum(wcap * vals) / np.sum(wcap) * copies * (1 - math.cos(radius)) / 2))


# ------------------------------------------------------------------ misorientations


def calcMDF(f1, f2=None, bandwidth=32):
  """the misorientation distribution of two ODFs, the density of inv(o1) o2 for independent
  grains, (inv f1 * f2)(g) = int f1(S g^-1) f2(S) dS, with exchange symmetry for one ODF"""
  h1 = SO3FunHarmonic(f1, bandwidth=min(bandwidth, f1.bandwidth)).truncate(bandwidth)
  h2 = h1 if f2 is None or f2 is f1 else SO3FunHarmonic(f2, bandwidth=min(bandwidth, f2.bandwidth)).truncate(bandwidth)
  m = conv(h1.conj().inv(), h2)
  return m.withExchange() if f2 is None or f2 is f1 else m


# the rotations whose own angle is the reduced one, decided in bulk against the identity.
# The orbit of a generic rotation has |L| such members, its conjugates by the specimen
# group, which carry the images of its axis under that group, so integrating over this set
# with the weight |R| gives the specimen-symmetric axis density and the angle density
def _minimalAngle(q, L, R):
  return np.abs(q[:, 0]) >= symDots(q, np.array([1.0, 0, 0, 0]), L, R) - 1e-10


def _framesOf(f):
  return reductionGroups(orientation(rotation.id, f.CS, f.SS))


def calcAngleDistribution(f, *args, omega=None, edges=None, resolution=0.5 * degree):
  """the distribution of the misorientation angle, MATLAB's order (density, omega):
    calcAngleDistribution(cs1, cs2)   the uniform one, the area of the axis sector at each
                                      angle times sin^2(omega/2), with mean one over the angles
    calcAngleDistribution(mori)       a histogram of the disorientation angles in percent
    calcAngleDistribution(f)          the uniform one times the mean of f over the axes at each
                                      angle, found by sampling the axes with a count following
                                      sin^2(omega/2) and keeping the rotations of minimal angle"""
  from ..geometry.orientationregion import fundamentalRegion
  if args and isinstance(args[0], referenceFrame):
    SS, args = args[0], args[1:]
  else:
    SS = None
  if args and omega is None:
    omega = np.asarray(args[0], dtype=float).ravel()
  if isinstance(f, referenceFrame):
    oR = fundamentalRegion(f, SS)
    w = np.linspace(0, oR.maxAngle(), 200) if omega is None else omega[omega < oR.maxAngle() + 1e-8]
    # the area of the axis sector at each angle, counted on a Fibonacci grid, which has no
    # rings to alias with the circular caps of the sector
    V = fibonacciS2Grid(1000000).data.reshape(-1, 3)
    V = V[oR.axisSector(0.0).checkInside(V)]
    ad = np.array([np.count_nonzero(oR.axisSector(x).checkInside(V)) for x in w]) * np.sin(w / 2) ** 2
    return ad / np.mean(ad), w
  if isinstance(f, orientation):
    top = fundamentalRegion(f.CS, f.SS).maxAngle()
    e = np.linspace(0, top, int(min(50, max(5, f.size / 20)))) if edges is None else np.asarray(edges, dtype=float)
    counts, _ = np.histogram(f.angle().ravel(), e)
    return 100 * counts / np.sum(counts), (e[:-1] + e[1:]) / 2
  L, R = _framesOf(f)
  top = fundamentalRegion(f.CS, f.SS).maxAngle()
  w = np.linspace(0, top, 200) if omega is None else omega[omega < top + 1e-8]
  rots, which = [], []
  for k, x in enumerate(w):
    n = max(4, round(4 / 3 * math.sin(x / 2) ** 2 / resolution ** 2))
    axes = fibonacciS2Grid(n)
    rots.append(rotation.byAxisAngle(vector3d(axes.data), x).data.reshape(-1, 4))
    which.append(np.full(len(rots[-1]), k))
  q, which = np.concatenate(rots), np.concatenate(which)
  inside = _minimalAngle(q, L, R)
  v = np.zeros(len(q))
  v[inside] = np.maximum(np.real(f(_ori(q[inside], f))), 0)
  counts = np.bincount(which, minlength=len(w))
  uniform = np.sin(w / 2) ** 2 * np.bincount(which, weights=inside.astype(float), minlength=len(w)) / counts
  density = np.sin(w / 2) ** 2 * np.bincount(which, weights=v, minlength=len(w)) / counts
  return density / np.mean(uniform), w


def calcMIndex(f, resolution=1 * degree):
  """the M-index: half the total variation between the angle distributions of the uniform
  and the uncorrelated misorientation distribution"""
  m = calcMDF(f)
  pu, _ = calcAngleDistribution(uniformODF(f.CS, f.CS, antipodal=True), resolution=resolution)
  pm, _ = calcAngleDistribution(m, resolution=resolution)
  return float(np.sum(np.abs(pu / np.sum(pu) - pm / np.sum(pm))) / 2)


# the density of the rotation axis: adf(a) = 2 copies/pi int f(a, omega) sin^2(omega/2) d omega
# over the angles for which (a, omega) lies in the fundamental region and in the window asked,
# by a rectangle rule; a spherical function in the crystal frame when the frames coincide
def _axisDensity(f, resolution=2.5 * degree, minAngle=0.0, maxAngle=np.pi):
  L, R = _framesOf(f)
  copies = len(R)
  omega = np.arange(resolution / 2, np.pi, resolution)
  window = (omega >= max(minAngle, 0)) & (omega <= min(maxAngle, np.pi))

  def value(vs):
    d = vector3d(vs.data.reshape(-1, 3)).normalize()
    q = rotation.byAxisAngle(d[None, :], omega[:, None]).data.reshape(-1, 4)
    use = (_minimalAngle(q, L, R).reshape(len(omega), d.size)) & window[:, None]
    v = np.zeros((len(omega), d.size))
    idx = np.flatnonzero(use.ravel())
    if idx.size:
      v.ravel()[idx] = np.real(f(_ori(q[idx], f)))
    out = 2 * copies / np.pi * np.sum(v * np.sin(omega / 2)[:, None] ** 2, axis=0) * resolution
    return out.reshape(vs.shape)

  return value


# the axis density of uniformly distributed rotations in the fundamental region: the angle
# integral of sin^2(omega/2) up to the largest angle about the axis, (w - sin w) / 2, which
# the planes of the region give in closed form; mean one over the sphere
def _uniformAxisDensity(oR):
  def wmax(xyz):
    w = np.full(len(xyz), np.pi)
    for n in oR.N:
      a = n[1:]
      s = np.linalg.norm(a)
      c = xyz @ a / s
      pos = c > 1e-9
      w[pos] = np.minimum(w[pos], 2 * np.arctan(-n[0] / s / c[pos]))
    return w

  V = fibonacciS2Grid(200000).data.reshape(-1, 3)
  w = wmax(V)
  scale = 1 / np.mean(w - np.sin(w))

  def value(vs):
    xyz = vector3d(vs.data.reshape(-1, 3)).normalize().data.reshape(-1, 3)
    w = wmax(xyz)
    return (scale * (w - np.sin(w))).reshape(vs.shape)

  return value


def calcAxisDistribution(f, v=None, *args, bandwidth=64, **kwargs):
  """the axis distribution of an SO(3) function, an S2 function of the rotation axes symmetric under
  the rotations the two frames share; with v its values at the axes v; for two frames the axis
  distribution of the uniform distribution, exactly

    adf = calcAxisDistribution(odf)
    value = calcAxisDistribution(odf, v)
    adf = calcAxisDistribution(odf, minAngle=20*degree, maxAngle=40*degree)
    adf = calcAxisDistribution(cs1, cs2)"""
  if isinstance(f, referenceFrame):
    # the uniform distribution exactly, not as a truncated expansion
    from ..geometry.orientationregion import fundamentalRegion
    CS = f
    SS = v if isinstance(v, referenceFrame) else CS
    v = None if isinstance(v, referenceFrame) else v
    value = _uniformAxisDensity(fundamentalRegion(CS, SS))
    if v is not None:
      return value(v if isinstance(v, vector3d) else vector3d(v))
    return S2FunHandle(value, CS if CS is SS else None, antipodal=True)
  if v is not None:
    return _axisDensity(f, **kwargs)(v if isinstance(v, vector3d) else vector3d(v))
  # the axes live in the first crystal frame, symmetric under the rotations the two frames
  # share, and antipodal only with grain exchange
  antipodal = bool(getattr(f, "antipodal", False))
  frame = None
  if isinstance(f.CS, crystalFrame) and isinstance(f.SS, crystalFrame):
    g = f.CS.sym.disjoint(f.SS.sym).properGroup()
    frame = f.CS if f.CS is f.SS and not antipodal else f.CS.sibling(g.Laue() if antipodal else g)
  return S2FunHarmonic(S2FunHandle(_axisDensity(f, **kwargs), frame, bandwidth=bandwidth, antipodal=antipodal))


def calcError(f, g, mode="l1", resolution=5 * degree):
  """the distance of two SO(3) functions on the equispaced grid of the resolution: 'l1' is half
  the mean absolute difference, for densities the fraction of the volume that would have to
  move to turn one into the other; 'l2' the norm of the difference over the norm of g"""
  mode = str(mode).lower()
  grid = equispacedSO3Grid(f.CS, f.SS, resolution=resolution)
  d1 = np.real(f(grid))
  d2 = np.real(g(orientation._fromRotation(grid.data, (g.CS, g.SS), g.antipodal)))
  if mode == "l1":
    return float(np.sum(np.abs(d1 - d2)) / d1.size / 2)
  if mode == "l2":
    return float(np.linalg.norm(d1 - d2) / np.linalg.norm(d2))
  raise ValueError(f"the error mode is 'l1' or 'l2', not {mode!r}")


# ------------------------------------------------------------------ extrema
# the best seeds of an equispaced grid, refined on a shrinking local grid around each until
# the resolution reaches the accuracy asked, then merged within the seed resolution;
# derivative free, so every representation works


# greedy clusters by the symmetry reduced angle: each orientation joins the cluster of the first
# of their number within the tolerance of it; the neighbourhoods from one radius search of the
# quaternion tree over the symmetric images, of the orientations among only when given (the
# pairs of two others are taken as apart). An orientation without neighbours leads its own
# cluster, so the loop runs over the others and the clusters are numbered by their leaders
def _clusterOrientations(o, tol, among=None):
  from scipy import sparse
  from ..geometry.grids.orientationtree import OrientationTree
  o = o.flatten()
  n = o.size
  among = np.arange(n) if among is None else np.asarray(among, dtype=int)
  I, _ = OrientationTree(o).radius(o[among], tol)
  I = I.tocoo()
  rows, cols = among[I.row], I.col
  apart = rows != cols
  A = sparse.csr_matrix((np.ones(int(apart.sum()), dtype=bool), (rows[apart], cols[apart])), shape=(n, n))
  A = (A + A.T).tocsr()
  indptr, indices = A.indptr, A.indices
  leader = np.arange(n)
  done = np.zeros(n, dtype=bool)
  for i in np.flatnonzero(np.diff(indptr)):
    if done[i]:
      continue
    near = indices[indptr[i]:indptr[i + 1]]
    near = near[~done[near]]
    leader[near], done[near], done[i] = i, True, True
  return np.unique(leader, return_inverse=True)[1]


# the basins of a function on an SO(3) grid: every node points to its highest neighbour within
# the angle, itself at a peak, the larger number deciding a tie so that no pointers cycle; the
# pointers followed to their peaks by doubling
def _gridBasins(grid, v, angle):
  from ..geometry.grids.orientationtree import gridRadius
  I, _ = gridRadius(grid, grid, angle)
  n = grid.size
  rank = np.empty(n, dtype=int)
  rank[np.lexsort((np.arange(n), v))] = np.arange(n)
  byRank = np.argsort(rank)
  best = rank.copy()
  rowOf = np.repeat(np.arange(n), np.diff(I.indptr))
  np.maximum.at(best, rowOf, rank[I.indices])
  up = byRank[best]
  while True:
    nxt = up[up]
    if np.array_equal(nxt, up):
      return up
    up = nxt


# the member of each cluster with the largest value, the first of their number on a tie
def _bestOfClusters(ids, vals, nc):
  order = np.lexsort((np.arange(len(ids)), -vals, ids))
  first = np.concatenate([[True], ids[order][1:] != ids[order][:-1]])
  rep = np.empty(nc, dtype=int)
  rep[ids[order][first]] = order[first]
  return rep


# the seeds of the ascent: the nodes that beat every neighbour within the angle whose cosine
# is c, the neighbourhood taken over the copies of the grid under the group (a node at the
# rim of the fundamental region has its uphill neighbour across it) and over the sign of a
# quaternion or an axis; the best 10 numLocal of them
def _bestSeeds(cand, v, numLocal, c, copies=None):
  from scipy.spatial import cKDTree
  tree = cKDTree(cand)
  r = math.sqrt(max(2 - 2 * c, 0.0))
  best = np.full(len(v), -np.inf)
  for pts in ([cand] if copies is None else copies):
    for sign in (1.0, -1.0):
      for i, near in enumerate(tree.query_ball_point(sign * pts, r)):
        if near:
          best[i] = max(best[i], float(np.max(v[near])))
  order = np.argsort(v)[::-1]
  seeds = [int(i) for i in order if v[i] >= best[i]]
  return np.array(seeds[:10 * numLocal])


# the copies of the grid nodes under the group elements, one array per element
def _copiesSO3(cand, f):
  # the proper rotations of the groups: an improper element maps no rotation onto the grid
  proper = lambda fr: (fr.properGroup() if hasattr(fr, "properGroup") else fr).sym.rot.flatten()
  Ls, Rs = proper(f.CS), proper(f.SS)
  q = rotation(cand)
  return [(rotation(Ls[i]) * q * rotation(Rs[j])).data.reshape(-1, 4) for i in range(Ls.size) for j in range(Rs.size)]


def _copiesS2(cand, fr):
  if fr is None or not hasattr(fr, "sym"):
    return None
  els = fr.sym.rot.flatten()
  return [(els[i] * vector3d(cand)).data.reshape(-1, 3) for i in range(els.size)]


def _localMaximaSO3(f, numLocal=1, resolution=5 * degree, accuracy=0.25 * degree, startingNodes=None):
  g = equispacedSO3Grid(f.CS, f.SS, resolution=resolution) if startingNodes is None else startingNodes
  cand = rotation(g).data.reshape(-1, 4)
  v = np.real(f(_ori(cand, f))).ravel()
  keep = _bestSeeds(cand, v, numLocal, math.cos(1.5 * resolution / 2), _copiesSO3(cand, f))
  modes, vals = cand[keep].copy(), v[keep].copy()
  res = resolution
  while res > accuracy:
    res /= 1.25
    shifts = rotation(localOrientationGrid(f.CS, f.SS, maxAngle=2 * res, resolution=res / 2, respectSymmetry=False))
    for k in range(len(modes)):
      trial = shifts * rotation(modes[k])
      tv = np.real(f(_ori(trial, f))).ravel()
      i = int(np.argmax(tv))
      if tv[i] > vals[k]:
        modes[k], vals[k] = trial.data.reshape(-1, 4)[i], tv[i]
  o = _ori(modes, f)
  ids = _clusterOrientations(o, 1.5 * resolution)
  reps = [int(np.flatnonzero(ids == k)[np.argmax(vals[ids == k])]) for k in np.unique(ids)]
  order = np.argsort(vals[reps])[::-1][:min(numLocal, len(reps))]
  sel = np.array(reps)[order]
  # the values of the modes found, from the function itself: a sum of many kernels is
  # searched through its harmonic form, which can undershoot the function
  best = _ori(modes[sel], f)
  return np.real(f(best)).ravel(), best


def _localMaximaS2(f, numLocal=1, resolution=2.5 * degree, accuracy=0.05 * degree):
  fr, antipodal = f.frame, f.antipodal
  g = equispacedS2Grid(resolution=resolution, antipodal=antipodal) if fr is None else equispacedS2Grid(fr, resolution=resolution, antipodal=antipodal)
  cand = g.data.reshape(-1, 3)
  v = np.real(f(vector3d(cand))).ravel()
  keep = _bestSeeds(cand, v, numLocal, math.cos(1.5 * resolution), _copiesS2(cand, fr))
  modes, vals = cand[keep].copy(), v[keep].copy()
  res = resolution
  while res > accuracy:
    res /= 1.25
    for k in range(len(modes)):
      cap = localS2Grid(vector3d(modes[k]), maxTheta=2 * res, resolution=res / 2, respectSymmetry=False)
      trial = cap.data.reshape(-1, 3)
      tv = np.real(f(vector3d(trial))).ravel()
      i = int(np.argmax(tv))
      if tv[i] > vals[k]:
        modes[k], vals[k] = trial[i], tv[i]
  # merge within the seed resolution, over the images under the group since the refinement
  # from seeds either side of a sector boundary can reach one mode and its copy
  els = None if fr is None else fr.sym.rot.flatten()
  c = math.cos(1.5 * resolution)

  def near(a, b):
    imgs = a[None, :] if els is None else (els * vector3d(a)).data.reshape(-1, 3)
    d = imgs @ b
    return bool(np.any(d > c) or (antipodal and np.any(-d > c)))

  reps = []
  for i in np.argsort(vals)[::-1]:
    if not any(near(modes[r], modes[i]) for r in reps):
      reps.append(i)
  reps = reps[:min(numLocal, len(reps))]
  pts = vector3d(modes[reps], frame=fr, antipodal=antipodal)
  return vals[reps], pts


# the largest value and where it is taken; numLocal > 1 gives the numLocal best local maxima
# max(f, g), min(f, 1): the pointwise extremum of two functions, or of a function and a number,
# expanded by quadrature at the larger bandwidth, antipodal when both are
def _pointwiseS2(op, f, g):
  isFun = isinstance(g, S2_KINDS)
  L = max(f.bandwidth, g.bandwidth) if isFun else f.bandwidth
  fun = (lambda v: op(np.real(f(v)), np.real(g(v)))) if isFun else (lambda v: op(np.real(f(v)), g))
  antipodal = f.antipodal and (g.antipodal if isFun else True)
  return S2FunHarmonic(fun, f.frame, bandwidth=L, antipodal=antipodal, real=True)


# the pointwise extremum of two functions on SO(3), or of one and a number, a handle
# evaluated when asked, as MATLAB's
def _pointwiseSO3(op, f, g):
  isFun = _isSO3(g)
  fun = (lambda o: op(np.real(f.eval(o)), np.real(g.eval(o)))) if isFun else (lambda o: op(np.real(f.eval(o)), g))
  L = max(f.bandwidth, g.bandwidth) if isFun else f.bandwidth
  return SO3FunHandle(fun, f.CS, f.SS, bandwidth=L)


# a list of functions on SO(3): along an axis the pointwise extremum, MATLAB's min(f, [], d),
# else the extremum of every function, the values and the positions
def _listExtremaSO3(f, op, one, numLocal, kwargs):
  from .funops import mapFun
  axis = kwargs.pop("axis", None)
  if axis is not None:
    return mapFun(lambda x: op(np.real(x), axis=1 + int(axis)), f, bandwidth=f.bandwidth)
  pairs = [one(f[k], numLocal=numLocal, **kwargs) for k in np.ndindex(f.shape)]
  return np.array([p[0] for p in pairs]).reshape(f.shape), [p[1] for p in pairs]


# a handle of several values reduced over one axis of its values, which follow the axes of
# the directions it is evaluated at
def _reduceHandle(f, op, axis):
  def fun(v):
    vals = np.asarray(f.eval(v))
    return op(vals, axis=np.ndim(v.data) - 1 + axis)
  return S2FunHandle(fun, f.frame, bandwidth=f.bandwidth, antipodal=f.antipodal, real=f.isReal, symmetrise=False)


def max_(f, *args, numLocal=1, **kwargs):
  """the maximum of a function on SO(3) or the sphere, the value and the position, numLocal local maxima
  from a grid refined down to the accuracy; with a second function or a number the pointwise
  maximum; with axis= the maximum of a list of functions along an axis

    value, ori = max(odf)
    values, oris = max(odf, numLocal=5, resolution=5*degree, accuracy=0.25*degree)
    f = max(odf1, odf2)"""
  # a pole figure clips or reports its intensities
  if hasattr(f, "allI"):
    return f.max(*args)
  # a list of functions reduced along an axis, MATLAB's max(sF, [], d)
  if kwargs.get("axis") is not None and isinstance(f, S2FunHarmonic):
    return f._reduce(np.max, kwargs["axis"])
  if kwargs.get("axis") is not None and isinstance(f, S2FunHandle):
    return _reduceHandle(f, np.max, kwargs["axis"])
  # the pointwise maximum with another function or a number
  if args and isinstance(f, S2_KINDS) and (isinstance(args[0], S2_KINDS) or np.isscalar(args[0])):
    return _pointwiseS2(np.maximum, f, args[0])
  if args and _isSO3(f) and (_isSO3(args[0]) or np.isscalar(args[0])):
    return _pointwiseSO3(np.maximum, f, args[0])
  if isinstance(f, SO3FunHarmonic) and f.shape:
    return _listExtremaSO3(f, np.max, max_, numLocal, kwargs)
  # a function on the circle knows its own extrema
  if hasattr(f, "_harmonicS1") or type(f).__name__ == "S1FunHarmonic":
    return f.max(**kwargs)
  if not (_isSO3(f) or isinstance(f, S2_KINDS)):
    return max(f, *args, **kwargs)
  v, p = (_localMaximaSO3 if _isSO3(f) else _localMaximaS2)(f, numLocal=numLocal, **kwargs)
  return (float(v[0]), p[0]) if numLocal == 1 else (v, p)


def min_(f, *args, numLocal=1, **kwargs):
  """the minimum of a function on SO(3) or the sphere, the value and the position, numLocal local minima;
  with a second function or a number the pointwise minimum; with axis= along an axis of a list

    value, ori = min(odf)
    f = min(odf, 0)"""
  if hasattr(f, "allI"):
    return f.min(*args)
  if kwargs.get("axis") is not None and isinstance(f, S2FunHarmonic):
    return f._reduce(np.min, kwargs["axis"])
  if kwargs.get("axis") is not None and isinstance(f, S2FunHandle):
    return _reduceHandle(f, np.min, kwargs["axis"])
  if args and isinstance(f, S2_KINDS) and (isinstance(args[0], S2_KINDS) or np.isscalar(args[0])):
    return _pointwiseS2(np.minimum, f, args[0])
  if args and _isSO3(f) and (_isSO3(args[0]) or np.isscalar(args[0])):
    return _pointwiseSO3(np.minimum, f, args[0])
  if isinstance(f, SO3FunHarmonic) and f.shape:
    return _listExtremaSO3(f, np.min, min_, numLocal, kwargs)
  if hasattr(f, "_harmonicS1") or type(f).__name__ == "S1FunHarmonic":
    return f.min(**kwargs)
  if not (_isSO3(f) or isinstance(f, S2_KINDS)):
    return min(f, *args, **kwargs)
  v, p = max_(-1 * f, numLocal=numLocal, **kwargs)
  return -v, p


# ------------------------------------------------------------------ components
# steepest ascent from seeds along the gradient with a fixed step ladder that halves when no
# step improves, clusters merged within a tolerance, the weights of the seeds accumulated per mode.
# Unless exact, a seed lighter than a mode that is kept and on a slope below FLAT_SLOPE of the
# steepest at the start stops where it is: the ripples of a harmonic expansion are such seeds

FLAT_SLOPE = 1e-3


# the gradient that drives the ascent, as a function of orientations: a radial function with
# many centres through its harmonic form, as its evaluation goes; the field of a harmonic
# function built once
def _gradientField(f):
  if isinstance(f, SO3FunRBF):
    from .rbf import HARMONIC_EVAL
    L, R = reductionGroups(f.center)
    if f.center.size ** 2 * len(L) * len(R) > HARMONIC_EVAL and f.psi.bandwidth <= getMTEXpref("maxSO3Bandwidth"):
      if getattr(f, "_harmonicCache", None) is None:
        f._harmonicCache = f._harmonic()
      return f._harmonicCache.grad()
  if isinstance(f, (SO3FunRBF, SO3FunCBF)):
    return f.grad
  return (f if isinstance(f, SO3FunHarmonic) else SO3FunHarmonic(f)).grad()


def calcComponents(f, seed=None, weights=None, tolerance=1.5 * degree, maxIter=100, radius=np.inf, exact=False, silent=False):
  """the modal orientations of an SO(3) function by steepest ascent from seeds, their volumes and which
  mode each seed flowed into; modes within the tolerance merge, and modes of less than one percent
  are dropped unless exact=True; the seeds are the centres of a radial function or a 2.5 degree grid

    modes, weights, centerId = calcComponents(odf)
    modes, weights, centerId = calcComponents(odf, seed=ori, radius=10*degree, exact=True)"""
  grid = None
  if seed is None and isinstance(f, SO3FunRBF):
    seed, weights = f.center, f.weights
  elif seed is None:
    grid = seed = equispacedSO3Grid(f.CS, f.SS, resolution=2.5 * degree)
  rots = rotation(seed).data.reshape(-1, 4)
  w = np.ones(len(rots)) if weights is None else np.asarray(weights, dtype=float).ravel()
  keep = w > 0
  rots, w = rots[keep], w[keep] / np.sum(w[keep])
  fg = _gradientField(f)
  # which mode each seed has flowed into; seeds that meet are one mode from then on, so the
  # working set shrinks as the ascent proceeds. The nodes of the grid climb to their peaks on
  # the grid first, and only the peaks ascend
  if grid is not None:
    v = np.real(f(grid)).ravel()
    peaks, labels = np.unique(_gridBasins(grid, v, 1.5 * grid.res), return_inverse=True)
    modes, vals = rots[peaks], v[peaks]
  else:
    modes, labels = rots.copy(), np.arange(len(rots))
    vals = np.real(f(_ori(modes, f))).ravel()
  steps = np.full(len(modes), 5 * degree)
  active = np.ones(len(modes), dtype=bool)
  steepest = None
  for _ in range(maxIter):
    idx = np.flatnonzero(active)
    if idx.size == 0:
      break
    o = _ori(modes[idx], f)
    g = fg(o)
    n = np.linalg.norm(g.data, axis=-1)
    steepest = n.max() if steepest is None else steepest
    if not exact:
      flat = (np.bincount(labels, w, minlength=len(modes))[idx] < 0.01) & (n < FLAT_SLOPE * steepest)
      if flat.any():
        active[idx[flat]] = False
        idx, o, g, n = idx[~flat], o[~flat], g[~flat], n[~flat]
        if idx.size == 0:
          break
    dirs = g.data / np.where(n > 0, n, 1.0)[:, None]
    trial1 = SO3TangentVector(vector3d((2 * steps[idx] / 3)[:, None] * dirs), o, "left").exp()
    trial2 = SO3TangentVector(vector3d(steps[idx][:, None] * dirs), o, "left").exp()
    v1, v2 = np.real(f(trial1)).ravel(), np.real(f(trial2)).ravel()
    t1, t2 = rotation(trial1).data.reshape(-1, 4), rotation(trial2).data.reshape(-1, 4)
    best = np.argmax(np.stack([vals[idx], v1, v2]), axis=0)
    # on a tie in a flat region the gradient still says where the mass is
    best[(best == 0) & (v1 == vals[idx]) & (v2 == vals[idx]) & (n > 0)] = 2
    stay = best == 0
    halve = stay & (steps[idx] > 0.05 * degree)
    steps[idx[halve]] /= 2
    active[idx[stay & ~halve]] = False
    for b, t, v in ((1, t1, v1), (2, t2, v2)):
      go = best == b
      modes[idx[go]], vals[idx[go]] = t[go], v[go]
    if stay.all():
      if not np.any(active):
        break
      continue
    # join the modes that have come within the tolerance of each other: a pair that has not
    # moved was looked at before
    ids = _clusterOrientations(_ori(modes, f), tolerance, among=idx[~stay])
    nc = ids.max() + 1
    if nc < len(modes):
      rep = _bestOfClusters(ids, vals, nc)
      keepActive = np.bincount(ids, weights=active, minlength=nc) > 0
      keepStep = np.zeros(nc)
      np.maximum.at(keepStep, ids, steps)
      labels = ids[labels]
      modes, vals, steps, active = modes[rep], vals[rep], keepStep, keepActive
  # clusters, the representative with the largest value, the weights within the radius
  ids = _clusterOrientations(_ori(modes, f), tolerance)
  nclust = ids.max() + 1
  modeRots = modes[_bestOfClusters(ids, vals, nclust)]
  wsum = np.zeros(nclust)
  seedIds = ids[labels]
  if np.isinf(radius):
    np.add.at(wsum, seedIds, w)
  else:
    d = _ori(rots, f).angle(_ori(modeRots[seedIds], f))
    np.add.at(wsum, seedIds[d < radius], w[d < radius])
  order = np.argsort(wsum)[::-1]
  remap = np.empty(nclust, dtype=int)
  remap[order] = np.arange(nclust)
  centerId = remap[seedIds]
  wsum, modeRots = wsum[order], modeRots[order]
  if not exact:
    thr = min(0.01, 0.5 * (wsum.max() if wsum.size else 0.0))
    keepm = wsum > thr
    newId = np.cumsum(keepm) - 1
    centerId = np.where(keepm[centerId], newId[centerId], -1)
    wsum, modeRots = wsum[keepm], modeRots[keepm]
  return _ori(modeRots, f), wsum, centerId


# ------------------------------------------------------------------ sampling


def discreteSample(f, n, resolution=None, rng=None, withoutReplacement=False, range=(0, 1)):
  """the nodes of a grid drawn with probability proportional to f, moved by a random rotation
  within the grid resolution and by a random element of each group; a radial basis function
  exactly: a centre by its weight, the angle by inverse transform of the kernel density, a
  uniform axis, a group element on each side; a Bingham distribution exactly by rejection; a
  sum of kernel forms term by term"""
  rng = np.random.default_rng() if rng is None else rng
  n = int(n)
  # a density given as a Python function of real numbers is sampled on 10000 points of range
  if isinstance(f, (types.FunctionType, functools.partial)):
    x = np.linspace(range[0], range[1], 10000)
    p = np.maximum(np.asarray(f(x), dtype=float), 0)
    return x[rng.choice(len(x), size=n, p=p / p.sum())]
  # a list of orientations or directions is sampled from itself, MATLAB's discreteSample(ori, n)
  if isinstance(f, MtexArray):
    flat = f.flatten()
    k = rng.choice(flat.size, size=min(n, flat.size) if withoutReplacement else n, replace=not withoutReplacement)
    return flat[k]
  if isinstance(f, S2_KINDS):
    return _discreteSampleS2(f, n, 1 * degree if resolution is None else resolution, rng)
  if isinstance(f, SO3FunComposition) and all(isinstance(t, SO3FunRBF) for t in f.components):
    m = np.array([t.mean() for t in f.components])
    counts = np.bincount(rng.choice(len(m), size=n, p=m / m.sum()), minlength=len(m))
    parts = [discreteSample(t, c, rng=rng).data.reshape(-1, 4) for t, c in zip(f.components, counts) if c > 0]
    return _ori(np.concatenate(parts) if parts else np.zeros((0, 4)), f)
  L, R = _framesOf(f)
  if isinstance(f, SO3FunRBF):
    omega = np.linspace(0, np.pi, 4001)
    pdf = f.psi(np.cos(omega / 2)) * np.sin(omega / 2) ** 2
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    probs = np.concatenate([f.weights * f.psi.mean(), [f.c0]])
    cs = f.center.data.reshape(-1, 4)
    if backend() == "numba":
      return _ori(rbfSample(n, rng, cs, probs, omega, cdf, L, R, f.antipodal), f)
    k = rng.choice(len(probs), size=n, p=probs / probs.sum())
    out = rotation.rand(n, rng=rng).data
    fromCentre = k < len(cs)
    w = omega[np.minimum(np.searchsorted(cdf, rng.random(n)), len(omega) - 1)]
    turned = rotation(cs[k[fromCentre]]) * rotation.byAxisAngle(vector3d.rand(int(fromCentre.sum()), rng=rng), w[fromCentre])
    out[fromCentre] = turned.data
    if f.antipodal:
      flip = rng.random(n) < 0.5
      out[flip] = rotation(out[flip]).inv().data
  elif isinstance(f, SO3FunBingham):
    out = binghamSample(f.kappa, f.A, n, rng)
    if f.antipodal:
      flip = rng.random(n) < 0.5
      out[flip] = rotation(out[flip]).inv().data
  else:
    res = 1.5 * degree if resolution is None else resolution
    g = equispacedSO3Grid(f.CS, f.SS, resolution=res)
    p = np.maximum(np.real(f(g)).ravel(), 0)
    idx = rng.choice(g.size, size=n, p=p / p.sum())
    w = 1.5 * res * np.cbrt(rng.random(n))
    out = (rotation.byAxisAngle(vector3d.rand(n, rng=rng), w) * rotation(g.data.reshape(-1, 4)[idx])).data
  left = rotation(L[rng.integers(len(L), size=n)])
  right = rotation(R[rng.integers(len(R), size=n)])
  return _ori((left * rotation(out) * right).data, f)


def _discreteSampleS2(f, n, resolution, rng):
  fr, antipodal = f.frame, f.antipodal
  g = equispacedS2Grid(resolution=resolution) if fr is None else equispacedS2Grid(fr, resolution=resolution, antipodal=antipodal)
  cand = g.data.reshape(-1, 3)
  p = np.maximum(np.real(f(vector3d(cand))).ravel(), 0)
  idx = rng.choice(len(cand), size=n, p=p / p.sum())
  v = rotation.byAxisAngle(vector3d.rand(n, rng=rng), 1.5 * resolution * np.sqrt(rng.random(n))) * vector3d(cand[idx])
  if fr is not None:
    els = (fr.sym.Laue() if antipodal else fr.sym).rot.flatten()
    v = vector3d(np.stack([(els[i] * v[j]).data for j, i in enumerate(rng.integers(els.size, size=n))]))
  return vector3d(v.data, frame=fr, antipodal=antipodal)


# ------------------------------------------------------------------ kernel selection and specimen centering


def calcKernel(o, method="UCV", samplingSize=1000, rng=None, *, groups=None, weights=None, bandwidth=None, halfwidths=None,
               noiseFloor=True):
  """the de la Vallee Poussin kernel a density estimate of orientations or directions should use
  (MATLAB calcKernel): 'UCV', the default, and 'conservative' select from the degree energies of
  the sample and are bounded below by a noise floor unless noiseFloor=False
  (docs/plans/halfwidth.md); 'klcv' scores ten candidate halfwidths by Kullback-Leibler cross
  validation, 'ruleOfThumb' reads the nearest neighbour disorientations, 'magicRule' the sample
  size and the symmetries alone; directions take the first two only. groups are the grain ids
  of pixels, which the energies fold by; a map brings its own"""
  from .halfwidth import degreeEnergies, selectHalfwidth
  from .kernelfunctions import S2DeLaValleePoussinKernel
  if not isinstance(o, (quaternion, vector3d)) and hasattr(o, "orientations"):
    groups = o.prop.get("grainId") if groups is None and "grainId" in o.prop else groups
    o = o.orientations
  directions = isinstance(o, vector3d)
  # the method however it is capitalised, MATLAB's 'RuleOfThumb' included
  method = {"klcv": "klcv", "ruleofthumb": "ruleOfThumb", "magicrule": "magicRule", "conservative": "conservative",
            "ucv": "UCV"}.get(str(method).lower(), method)
  if method in ("conservative", "UCV"):
    # the selection draws subsamples and folds; a fixed seed makes the default reproducible
    rng = np.random.default_rng(0) if rng is None else rng
    L = (32 if method == "conservative" else (256 if directions else 64)) if bandwidth is None else int(bandwidth)
    # UCV needs no errors of the energies, so independent points take one transform
    folds = 1 if method == "UCV" and groups is None else 8
    E = degreeEnergies(o, L, weights=weights, groups=groups, folds=folds, rng=rng)
    h = selectHalfwidth(E, method, halfwidths, noiseFloor=noiseFloor)
    return S2DeLaValleePoussinKernel(halfwidth=h) if directions else SO3DeLaValleePoussinKernel(halfwidth=h)
  rng = np.random.default_rng() if rng is None else rng
  if directions:
    raise ValueError(f"the kernel of directions is selected by 'UCV' or 'conservative', not {method!r}")
  # the cross validation and the rules count pixels as independent draws
  if isinstance(o, orientation) and o.notIID:
    raise ValueError(f"{method!r} counts the pixels of a map as independent draws, which neighbouring pixels of one "
                     "grain are not: select with 'conservative' or 'UCV' and the grain ids, or from the grain mean "
                     "orientations")
  o = o.flatten()
  N = o.size
  if N == 0:
    raise ValueError("no orientations to select a kernel from")
  # above five thousand a subsample is scored and the concentration scaled for the whole
  fak = 1.0
  if N > 5000:
    fak = (N / 5000) ** (1 / 7)
    o = o[rng.permutation(N)[:5000]]
  n = o.size
  nL = o.SS.sym.properGroup().rotations().size
  nR = o.CS.sym.properGroup().rotations().size
  if method == "magicRule":
    return SO3DeLaValleePoussinKernel(3 * (nL * nR * N) ** (2 / 7))
  if method == "ruleOfThumb":
    nearest = np.empty(n)
    for k in range(n):
      d = o.angle(o[k]).ravel()
      d[k] = np.pi
      nearest[k] = np.min(np.where(d < 0.005, np.pi, d))
    res = np.quantile(nearest, 0.9)
    psi = SO3DeLaValleePoussinKernel(halfwidth=max(1.5 * res, 2 * degree))
    return SO3DeLaValleePoussinKernel(fak * psi.kappa)
  if method == "klcv":
    candidates = [SO3DeLaValleePoussinKernel(halfwidth=30 * degree / 2 ** (k / 3)) for k in range(1, 11)]
    sN = min(n, samplingSize)
    t = np.stack([np.cos(o.angle(o[j]).ravel() / 2) for j in range(sN)])
    scores = []
    for psi in candidates:
      vals = psi(t)
      vals[np.arange(sN), np.arange(sN)] = 0
      fj = np.sum(vals, axis=1) / n / nR
      scores.append(np.mean(np.log(np.maximum(fj, np.finfo(float).tiny))))
    best = candidates[int(np.argmax(scores))]
    return SO3DeLaValleePoussinKernel(fak * best.kappa)
  raise ValueError(f"the kernel selection method is 'UCV', 'conservative', 'klcv', 'ruleOfThumb' or 'magicRule', not {method!r}")


def centerSpecimen(odf, v0=None, *args, delta=45 * degree, bandwidth=16, verbose=False):
  """two perpendicular twofold axes of an assumed orthotropic texture, found and rotated onto
  the nearest specimen axes (MATLAB centerSpecimen): the mismatch of a candidate axis is the
  norm of the difference between the harmonic series of the density and the series turned
  by 180 degrees about it; the first axis is searched on a cap about v0, the second on the
  fan perpendicular to it. Returns the centred series, the correction applied, and the axes"""
  # MATLAB's flags 'Fourier' and 'silent' change nothing here: the search runs on the harmonic form
  if isinstance(v0, str):
    v0 = None
  from ..geometry.frames import specimenFrame
  f0 = SO3FunHarmonic(odf, bandwidth=bandwidth)
  # the specimen group is what is being looked for, so the series drops it
  f = SO3FunHarmonic(f0._fhat, f0.CS, specimenFrame.default, real=f0.isReal, antipodal=f0.antipodal, symmetrise=False, _internal=True)
  scale = f.norm()
  v0 = xvector if v0 is None else v0

  def mismatch(vs):
    return np.array([(f - f.rotate(rotation.byAxisAngle(vector3d(v), np.pi))).norm() / scale for v in vs.data.reshape(-1, 3)])

  def best(cands):
    vals = mismatch(cands)
    k = int(np.argmin(vals))
    if verbose:
      print(f"  fit: {(1 - vals[k]) * 100:.1f}%")
    return vector3d(cands.data.reshape(-1, 3)[k])

  def cap(center, opening, resolution):
    return rotation.map(zvector, center) * vector3d(equispacedS2Grid(resolution=resolution, maxTheta=opening).data)

  if verbose:
    print("  searching for a first two fold symmetry axis")
  v1 = best(cap(v0, delta, 5 * degree))
  v1 = best(cap(v1, 2.5 * degree, 0.75 * degree))
  if verbose:
    print("  searching for a second two fold symmetry axis")
  o = v1.orth()

  def fan(rng_):
    return rotation.byAxisAngle(v1, rng_) * o

  v2 = best(fan(np.arange(-45, 46, 5) * degree))
  o = v2
  v2 = best(fan(np.arange(-5, 5.5, 0.5) * degree))
  # the rotation carrying the two axes onto the specimen axes nearest to them
  axes = np.concatenate([np.eye(3), -np.eye(3)])
  closest = lambda u: axes[int(np.argmax(axes @ u))]
  u1, u2 = v1.data.reshape(3), v2.data.reshape(3)
  a1, a2 = closest(u1), closest(u2)
  w2 = u2 - np.dot(u2, u1) * u1
  w2 /= np.linalg.norm(w2)
  M = np.stack([a1, a2, np.cross(a1, a2)], axis=1) @ np.stack([u1, w2, np.cross(u1, w2)], axis=1).T
  rot = rotation.byMatrix(M)
  return f0.rotate(rot), rot, v1, vector3d(w2)


# ------------------------------------------------------------------ the methods of the function kinds


install(SO3_KINDS + S2_KINDS, max_, "max")
install(SO3_KINDS + S2_KINDS, min_, "min")
for _f in (entropy, textureindex, discreteSample):
  install(SO3_KINDS + S2_KINDS, _f)
for _f in (volume, fibreVolume, calcComponents, calcMDF, calcAngleDistribution, calcAxisDistribution, calcMIndex, calcError, centerSpecimen):
  install(SO3_KINDS, _f)

__all__ = ["textureindex", "entropy", "volume", "fibreVolume", "calcMDF", "calcAngleDistribution", "calcMIndex",
           "calcAxisDistribution", "calcError", "calcComponents", "discreteSample", "calcKernel", "centerSpecimen"]
