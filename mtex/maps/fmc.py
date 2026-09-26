# fast multiscale clustering, the FMC grain boundary criterion (MATLAB @gbcFMC and its
# private FMC_MTEX, FMC_Coarsen, FMC_interpret, FMC_report; branch ADR 0004 for the defaults)
#
#   grains = calcGrains(ebsd, fmc=3)                     cmaha, the one parameter that acts as a threshold
#   grains = calcGrains(ebsd, fmc=1.5, minPixel=10, verbose=True)
#   grains = calcGrains(ebsd, gbcFMC(3, quatmax=5 * degree, maxDelta=15 * degree))
#
# FMC clusters the whole neighbour graph of a phase at once, as an algebraic multigrid
# coarsening in the tradition of segmentation by weighted aggregation. One step dilutes the
# weak edges, picks seeds every node couples strongly to, interpolates the other nodes into
# them softly, forms the Galerkin coarse graph P'WP, gathers the statistics of every
# aggregate (mean orientation, centroid, spread and, where it pays for its parameters by
# minimum description length, a lattice gradient), rescales the coarse couplings by the
# residual misorientation measured against the aggregates' own spread, and scores every
# aggregate by its saliency, external over internal coupling. No merge is committed on the
# way: every pixel is read off at the scale where its aggregate is most salient, nested
# choices are joined, spatially separate parts split, and regions below minPixel absorbed
# by their most strongly coupled neighbour. The criterion then answers 1 for a pair in one
# cluster and 0 otherwise; each indexed phase is clustered on its own.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import backend as kernelBackend, closestImage
from ..geometry.orientationregion import maxAngle
from ..geometry.units import degree
from .criteria import _bestDot, grainBoundaryCriterion

KAPPA = 4             # the outlier radius in standard deviations of an aggregate's own spread
MEMBER = 0.05         # the least membership that counts a node into an aggregate's statistics
WTHRESHOLD = 1e-3     # a residual this small, in degree, leaves a coupling as it is


class gbcFMC(grainBoundaryCriterion):
  """gbcFMC(cmaha=3, cmaha0=0.05, quatmax=5 * degree, alpha=0.2, gammaW=10, maxDelta=inf, minPixel=1, verbose=False)

  cmaha the sharpness of the coarse rebias, larger giving more grains; cmaha0 the decay of
  the initial edge weight per degree of misorientation; quatmax the ceiling of the outlier
  radius; alpha the seed aggressiveness; gammaW the inverse edge dilution; maxDelta the
  misorientation above which an edge is cut; minPixel the smallest region left standing;
  verbose prints the hierarchy of every phase."""

  handlesMinPixel = True

  def __init__(self, cmaha=3.0, cmaha0=0.05, quatmax=5 * degree, alpha=0.2, gammaW=10.0, maxDelta=np.inf, minPixel=1, verbose=False):
    self.cmaha, self.cmaha0, self.quatmax, self.alpha, self.gammaW = float(cmaha), float(cmaha0), float(quatmax), float(alpha), float(gammaW)
    self.maxDelta, self.minPixel, self.verbose = float(maxDelta), int(minPixel), bool(verbose)

  def setMinPixel(self, minPixel):
    out = gbcFMC(**{k: getattr(self, k) for k in ("cmaha", "cmaha0", "quatmax", "alpha", "gammaW", "maxDelta", "verbose")})
    out.minPixel = int(minPixel)
    return out

  def evaluate(self, ebsd, i, j):
    i, j = np.asarray(i, dtype=np.int64).reshape(-1), np.asarray(j, dtype=np.int64).reshape(-1)
    ph = ebsd.phaseId.reshape(-1)
    out = np.zeros(len(i))
    for p in range(len(ebsd.CSList)):
      fr = ebsd.CSList[p]
      pair = (ph[i] == p) & (ph[j] == p)
      if not pair.any():
        continue
      if not fr.isIndexed:
        out[pair] = 1                             # nothing was measured, nothing separates
        continue
      isP = np.flatnonzero(ph == p)
      loc = np.full(ph.size, -1, dtype=np.int64)
      loc[isP] = np.arange(len(isP))
      ip, jp = loc[i[pair]], loc[j[pair]]
      label = self.clusterPhase(ebsd, isP, ip, jp, fr)
      out[pair] = (label[ip] == label[jp]) & (label[ip] > 0)
    return out

  # FMC on the pixels isP of one phase over the pair list (ip, jp) within them
  def clusterPhase(self, ebsd, isP, ip, jp, fr):
    q = ebsd.rotations.data.reshape(-1, 4)[isP].astype(np.float64)
    P3 = ebsd.pos.data.reshape(-1, 3)[isP].astype(np.float64)
    l = ebsd.lattice
    pos = l.plane.to2d(P3) if l is not None else P3[:, :2]
    if not np.all(np.isfinite(pos)):
      pos = np.zeros((len(q), 2))
    hier = coarsenAll(self, q, pos, ip, jp, fr)
    label, rep = interpret(hier, self.minPixel)
    if self.verbose:
      report(self, fr.mineral, hier, rep)
    return label

  def __repr__(self):
    return f"gbcFMC (cmaha {self.cmaha:g}, quatmax {self.quatmax / degree:g}°" + (f", minPixel {self.minPixel}" if self.minPixel > 1 else "") + ")"


# ------------------------------------------------------------------ the hierarchy

class _state:
  pass


def _sparse(i, j, v, shape):
  from scipy.sparse import coo_matrix
  M = coo_matrix((v, (i, j)), shape=shape).tocsr()
  M.sum_duplicates()
  M.eliminate_zeros()
  return M


# the proper group of a frame as rows of quaternions, the right factors of the equivalents
def _group(fr):
  return fr.sym.properGroup().rotations().data.reshape(-1, 4).astype(np.float64)


# the equivalents of q closest to the references
def _project(q, ref, G):
  return closestImage(ref, q, np.array([[1.0, 0, 0, 0]]), G, False)[1]


# the whole coarsening of one phase: the memberships and saliencies of every scale, the
# pixel couplings without the maxDelta cut and what the report prints
def coarsenAll(crit, q, pos, ip, jp, fr):
  N = len(q)
  a, b = np.minimum(ip, jp), np.maximum(ip, jp)
  key = np.unique(a * N + b)
  i, j = key // N, key % N
  G = _group(fr)
  d = _bestDot(fr, q[i], q[j]) if len(i) else np.zeros(0)
  delta = 2 * np.degrees(np.real(np.arccos(np.minimum(d, 1.0))))
  finite = delta[np.isfinite(delta)]
  s = _state()
  s.G_sym = G
  s.sigma = max(np.median(finite) / np.sqrt(4 * 2.3659738843), 1e-3) if len(finite) else 1e-3
  s.maxAng = maxAngle(fr) / degree
  s.cmaha, s.cmaha0, s.quatmax, s.gammaW, s.alpha = crit.cmaha, crit.cmaha0, crit.quatmax / degree, crit.gammaW, crit.alpha
  w = np.exp(-crit.cmaha0 * delta)
  w[~np.isfinite(delta)] = 0
  wCut = np.where(delta > np.degrees(crit.maxDelta), 0.0, w)
  W = _sparse(i, j, wCut, (N, N))
  s.W = (W + W.T).tocsr()
  W0 = _sparse(i, j, w, (N, N))
  W0 = (W0 + W0.T).tocsr()
  A = _sparse(i, j, np.ones(len(i)), (N, N))
  s.A_D = ((A + A.T) > 0).astype(np.float64).tocsr()
  s.O = q.copy()
  s.pos = pos.copy()
  s.M = np.zeros((N, 10))
  s.Qvar = np.zeros(N)
  s.QvarTot = np.zeros(N)
  s.Gr = np.zeros((N, 6))
  s.v = np.ones(N)
  s.level = 1
  s.sizeW, s.sizeWnext = 0, N
  hier = _state()
  hier.Ps, hier.Sals, hier.numClusters, hier.nLin = [None], [np.full(N, np.inf)], [N], [0]
  hier.W0, hier.A_D, hier.sigma = W0, s.A_D.copy(), s.sigma
  while s.sizeW != s.sizeWnext:
    coarsen(s)
    s.level += 1
    hier.Ps.append(s.P)
    hier.Sals.append(s.Sal)
    hier.numClusters.append(s.W.shape[0])
    hier.nLin.append(s.nLin)
  return hier


# one coarsening step
def coarsen(s):
  s.sizeW = s.sizeWnext
  diluteEdges(s)
  coarseSeeds(s)
  interpolation(s)
  coarseGraph(s)
  aggregateStatistics(s)
  biasWeights(s)
  saliency(s)
  s.v = s.vcoarse


# drop the edges far weaker than the average coupling of both their ends (Kushnir 2006)
def diluteEdges(s):
  W = s.W.tocoo()
  sumW = np.asarray(s.W.sum(axis=1)).ravel() - s.W.diagonal()
  n = np.asarray((s.W > 0).sum(axis=0)).ravel()
  with np.errstate(divide="ignore", invalid="ignore"):
    weak = (W.data < sumW[W.row] / (s.gammaW * n[W.row])) & (W.data < sumW[W.col] / (s.gammaW * n[W.col]))
  s.W = _sparse(W.row, W.col, np.where(weak, 0.0, W.data), (s.sizeW, s.sizeW))


# the seeds: in order of decreasing coupling at the finest scale and of decreasing size
# above it, a node becomes a seed when its coupling to the seeds so far is below alpha of its total
def coarseSeeds(s):
  W = s.W.tocsc()
  W.sort_indices()
  alphaSum = s.alpha * (np.asarray(W.sum(axis=1)).ravel() - W.diagonal())
  order = np.argsort(-(np.asarray(W.sum(axis=0)).ravel() if s.level == 1 else s.v), kind="stable")
  seed = np.zeros(s.sizeW, dtype=np.bool_)
  loop = None
  if kernelBackend() == "numba":
    from .kernels import grainKernel
    loop = grainKernel("fmcSeeds")
  if loop is not None:
    loop(order.astype(np.int64), W.indptr.astype(np.int64), W.indices.astype(np.int64), W.data.astype(np.float64), alphaSum, seed)
  else:
    for k in order:
      rows = W.indices[W.indptr[k]:W.indptr[k + 1]]
      seed[k] = W.data[W.indptr[k]:W.indptr[k + 1]][seed[rows]].sum() < alphaSum[k]
  s.C = np.flatnonzero(seed)
  s.F = ~seed
  s.sizeWnext = len(s.C)


# the interpolation: a fine node in the seeds it couples to, row normalised, a seed in
# itself; a node coupled to no seed becomes a seed of its own
def interpolation(s):
  from scipy.sparse import diags, hstack
  nC = s.sizeWnext
  P = diags(s.F.astype(np.float64)) @ s.W[:, s.C] + _sparse(s.C, np.arange(nC), np.ones(nC), (s.sizeW, nC))
  Psum = np.asarray(P.sum(axis=1)).ravel()
  # every entry divided by its row sum, which stays finite where the sum is subnormal
  P = P.tocoo()
  P = _sparse(P.row, P.col, P.data / Psum[P.row], (s.sizeW, nC))
  lone = np.flatnonzero(Psum <= 0)
  if len(lone):
    P = hstack([P, _sparse(lone, np.arange(len(lone)), np.ones(len(lone)), (s.sizeW, len(lone)))]).tocsr()
    s.C = np.concatenate([s.C, lone])
  s.sizeWnext = len(s.C)
  P.eliminate_zeros()
  s.P = P


# the Galerkin coarse graph and the sizes of the aggregates
def coarseGraph(s):
  P = s.P
  s.W = (P.T @ s.W @ P).tocsr()
  s.A_D = (P.T @ s.A_D @ P).tocsr()
  s.vcoarse = P.T @ s.v


# the tangent vectors of b in the frame of a, in degree, both rows of unit quaternions
def tangent(a, b):
  w = np.sum(a * b, axis=1)
  x = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] - a[:, 2] * b[:, 3] + a[:, 3] * b[:, 2]
  y = a[:, 0] * b[:, 2] + a[:, 1] * b[:, 3] - a[:, 2] * b[:, 0] - a[:, 3] * b[:, 1]
  z = a[:, 0] * b[:, 3] - a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1] - a[:, 3] * b[:, 0]
  sgn = np.where(w < 0, -1.0, 1.0)
  w, x, y, z = w * sgn, x * sgn, y * sgn, z * sgn
  nv = np.sqrt(x ** 2 + y ** 2 + z ** 2)
  big = nv > 1e-12
  f = np.full(nv.shape, 2.0)
  f[big] = 2 * np.arctan2(nv[big], w[big]) / nv[big]
  return np.stack([x, y, z], axis=1) * (f / degree)[:, None]


def _acc(j, v, n):
  return np.bincount(j, weights=v, minlength=n)[:n]


# the statistics of every aggregate from its seed and the fine nodes interpolating into it,
# all members at once: the mean orientation with the outliers beyond the aggregate's radius
# taken out against the converged mean, the centroid, the moment sums carried up the
# hierarchy, and the spread around a constant or, where minimum description length pays
# for the six parameters, a linear orientation field whose gradient is the lattice curvature
def aggregateStatistics(s):
  C = s.C
  nC = s.sizeWnext
  Pc = s.P.tocsc()
  Pc.sort_indices()
  Pc = Pc.tocoo()
  ii, jj, pp = Pc.row, Pc.col, Pc.data
  t = (pp > MEMBER) & (ii != C[jj])
  i, j, p = ii[t], jj[t], pp[t].copy()
  Oc = s.O[C]
  O = _project(s.O[i], Oc[j], s.G_sym) if len(i) else np.zeros((0, 4))
  vi = s.v[i] * p
  vc = s.v[C]
  rad = KAPPA * np.sqrt(np.maximum(s.Qvar[C], s.sigma ** 2))
  rad = np.minimum(np.maximum(rad, 0.4 * s.quatmax), s.quatmax)
  thr = np.cos(rad / 2 * degree)
  mu = Oc
  keep = np.ones(len(i), dtype=bool)
  for _ in range(2):
    d = np.sum(O * mu[j], axis=1)
    keep = np.abs(d) > thr[j]
    sgn = np.where(d < 0, -1.0, 1.0)
    w = vi * sgn * keep
    acc = vc[:, None] * Oc + np.stack([_acc(j, w * O[:, c], nC) for c in range(4)], axis=1)
    nrm = np.linalg.norm(acc, axis=1)
    mu = acc / np.where(nrm > 0, nrm, 1)[:, None]
  p[~keep] = 0
  kp = np.flatnonzero(keep)
  jm = np.concatenate([j[kp], np.arange(nC)])
  wm = np.concatenate([vi[kp], vc])
  xm = np.vstack([s.pos[i[kp]], s.pos[C]])
  Mm = np.vstack([s.M[i[kp]] * p[kp, None], s.M[C]])
  tm = tangent(mu[jm], np.vstack([O[kp], Oc]))
  n = _acc(jm, wm, nC)
  n[n <= 0] = 1
  cx = _acc(jm, wm * xm[:, 0], nC) / n
  cy = _acc(jm, wm * xm[:, 1], nC) / n
  dx, dy = xm[:, 0] - cx[jm], xm[:, 1] - cy[jm]
  Sxx = _acc(jm, Mm[:, 0] + wm * dx ** 2, nC)
  Sxy = _acc(jm, Mm[:, 1] + wm * dx * dy, nC)
  Syy = _acc(jm, Mm[:, 2] + wm * dy ** 2, nC)
  Sxv = np.stack([_acc(jm, Mm[:, 3 + c] + wm * dx * tm[:, c], nC) for c in range(3)], axis=1)
  Syv = np.stack([_acc(jm, Mm[:, 6 + c] + wm * dy * tm[:, c], nC) for c in range(3)], axis=1)
  Sv = np.stack([_acc(jm, wm * tm[:, c], nC) for c in range(3)], axis=1)
  Stt = _acc(jm, Mm[:, 9] + wm * np.sum(tm ** 2, axis=1), nC)
  sseConst = np.maximum(Stt - np.sum(Sv ** 2, axis=1) / n, 0)
  # the linear model in centred coordinates, one 2 x 2 solve per component
  det = Sxx * Syy - Sxy ** 2
  ok = (det > 0) & (n >= 12)
  idet = np.zeros(nC)
  idet[ok] = 1 / det[ok]
  gx = (Syy[:, None] * Sxv - Sxy[:, None] * Syv) * idet[:, None]
  gy = (-Sxy[:, None] * Sxv + Sxx[:, None] * Syv) * idet[:, None]
  gain = np.minimum(np.maximum(np.sum(gx * Sxv + gy * Syv, axis=1), 0), sseConst)
  sseLin = sseConst - gain
  # L = SSE / (2 sigma^2) + k log(maxAngle / sigma) + k/2 log(n), k = 3 constant, 9 linear
  sigma = s.sigma
  bits = 3 * np.log(max(s.maxAng, 2 * sigma) / sigma)
  lConst = sseConst / (2 * sigma ** 2) + bits + 1.5 * np.log(np.maximum(n, 2))
  lLin = sseLin / (2 * sigma ** 2) + 3 * bits + 4.5 * np.log(np.maximum(n, 2))
  useLin = ok & (lLin < lConst)
  sse = np.where(useLin, sseLin, sseConst)
  gx[~useLin] = 0
  gy[~useLin] = 0
  s.Qvar = np.maximum(sse / n, sigma ** 2)
  s.QvarTot = np.maximum(sseConst / n, sigma ** 2)
  s.nLin = int(useLin.sum())
  s.P = _sparse(np.concatenate([i, ii[~t]]), np.concatenate([j, jj[~t]]), np.concatenate([p, pp[~t]]), (s.sizeW, nC))
  s.O = mu
  s.pos = np.stack([cx, cy], axis=1)
  s.M = np.column_stack([Sxx, Sxy, Syy, Sxv, Syv, Stt])
  s.Gr = np.hstack([gx, gy])


# the coarse couplings rescaled by how surprising the misorientation between two aggregates
# is: the residual neither side's gradient explains, against the smaller of their spreads
def biasWeights(s):
  from scipy.sparse import triu
  nC = s.sizeWnext
  U = triu(s.W).tocoo()
  i, j = U.row, U.col
  Oi = s.O[i]
  Oj = _project(s.O[j], Oi, s.G_sym) if len(i) else np.zeros((0, 4))
  v = tangent(Oi, Oj)
  dc = s.pos[j] - s.pos[i]
  predI = s.Gr[i, :3] * dc[:, :1] + s.Gr[i, 3:] * dc[:, 1:]
  predJ = s.Gr[j, :3] * dc[:, :1] + s.Gr[j, 3:] * dc[:, 1:]
  res = np.minimum(np.linalg.norm(v - predI, axis=1), np.linalg.norm(v - predJ, axis=1))
  small = res <= WTHRESHOLD
  sq = np.maximum(np.sqrt(np.minimum(s.QvarTot[i], s.QvarTot[j])), WTHRESHOLD)
  bias = np.where(small, 1.0, np.exp(-s.cmaha * (res - sq) / sq))
  B = _sparse(i, j, bias, (nC, nC))
  B = B + triu(B, 1).T
  s.W = s.W.multiply(B).tocsr()


# the saliency of every aggregate, external over internal coupling
def saliency(s):
  dW, dA = s.W.diagonal(), s.A_D.diagonal()
  with np.errstate(divide="ignore", invalid="ignore"):
    s.Sal = 2 * ((np.asarray(s.W.sum(axis=0)).ravel() - dW) * dA) / (dW * (np.asarray(s.A_D.sum(axis=1)).ravel() - dA))


# ------------------------------------------------------------------ reading the hierarchy

# the row maximum of a sparse matrix of nonnegative entries and the first column attaining
# it, zero and column zero for an empty row, over the stored entries at once
def _rowMax(B):
  B = B.tocsr()
  B.sum_duplicates()
  n = B.shape[0]
  cnt = np.diff(B.indptr)
  mx = np.zeros(n)
  arg = np.zeros(n, dtype=np.int64)
  has = cnt > 0
  if has.any():
    mx[has] = np.maximum.reduceat(B.data, B.indptr[:-1][has])
    row = np.repeat(np.arange(n), cnt)
    hit = np.flatnonzero(B.data == mx[row])
    r, first = np.unique(row[hit], return_index=True)
    arg[r] = B.indices[hit[first]]
  return mx, arg


def _components(n, a, b):
  from scipy.sparse.csgraph import connected_components
  return connected_components(_sparse(a, b, np.ones(len(a)), (n, n)), directed=False)[1]


# one label per pixel from the hierarchy: every pixel read off at the scale where its
# dominant aggregate is most salient against the median of that scale, a chosen aggregate
# joined to the first chosen one above it along the dominant parents, the regions split
# into their connected parts and the ones below minPixel absorbed, rim first, by the full
# sized neighbour they couple to most strongly; 0 for a pixel no scale reads off
def interpret(hier, minPixel=1):
  nc = hier.numClusters
  numS, N = len(nc), nc[0]
  A_D = ((hier.A_D + hier.A_D.T) > 0).tocsr()
  offset = np.zeros(numS, dtype=np.int64)
  for sc in range(2, numS):
    offset[sc] = offset[sc - 1] + nc[sc - 1]
  numAgg = int(offset[-1] + nc[-1]) if numS > 1 else 0
  bestSal = np.full(N, np.inf)
  bestLab = np.full(N, -1, dtype=np.int64)
  bestP = np.zeros(N)
  scaleOf = np.zeros(numAgg, dtype=np.int64)
  indexOf = np.zeros(numAgg, dtype=np.int64)
  parent = [None] * numS
  B = hier.Ps[1] if numS > 1 else None
  for sc in range(1, numS):
    if sc > 1:
      B = (B @ hier.Ps[sc]).tocsr()
    scaleOf[offset[sc]:offset[sc] + nc[sc]] = sc
    indexOf[offset[sc]:offset[sc] + nc[sc]] = np.arange(nc[sc])
    p, a = _rowMax(B)
    sal = np.asarray(hier.Sals[sc], dtype=float)
    good = sal[np.isfinite(sal) & (sal > 0)]
    ref = np.median(good) if len(good) else 1.0
    sal = sal / (ref if ref > 0 else 1.0)
    ok = p > 0
    q = np.full(N, np.inf)
    q[ok] = sal[a[ok]]
    q[np.isnan(q)] = np.inf
    take = ok & (q < bestSal)
    bestSal[take], bestLab[take], bestP[take] = q[take], offset[sc] + a[take], p[take]
    if sc < numS - 1:
      parent[sc] = _rowMax(hier.Ps[sc + 1])[1]
  sel = bestLab >= 0
  rep = _state()
  rep.readPerScale = np.bincount(scaleOf[bestLab[sel]], minlength=numS)
  rep.numStranded = int((~sel).sum())
  # a chosen aggregate contained in a chosen one above it describes the same region
  chosen = np.unique(bestLab[sel])
  isChosen = np.zeros(numAgg, dtype=bool)
  isChosen[chosen] = True
  cur, active = indexOf[chosen].copy(), np.ones(len(chosen), dtype=bool)
  src, dst = [], []
  for sc in range(2, numS):
    up = active & (scaleOf[chosen] < sc)
    cur[up] = parent[sc - 1][cur[up]]
    hit = up & isChosen[offset[sc] + np.where(up, cur, 0)]
    src.append(chosen[hit])
    dst.append(offset[sc] + cur[hit])
    active &= ~hit
  if numAgg:
    comp = _components(numAgg, np.concatenate(src + [np.zeros(0, dtype=np.int64)]), np.concatenate(dst + [np.zeros(0, dtype=np.int64)]))
    bestLab[sel] = comp[bestLab[sel]]
  # the spatial split
  U = A_D.tocoo()
  up = U.row < U.col
  a, b = U.row[up], U.col[up]
  same = (bestLab[a] == bestLab[b]) & (bestLab[a] >= 0)
  label = _components(N, a[same], b[same]) + 1
  label[bestLab < 0] = 0
  # absorb the undersized regions, the strongest couplings first
  thr = max(minPixel, 2)
  mx = hier.W0.max() if hier.W0.nnz else 0
  mx = mx if mx > 0 else 1
  label0 = label.copy()
  cand0 = A_D.multiply(hier.W0).tocsr()
  for lv in mx * np.array([0.9, 0.7, 0.5, 0.3, 0.15, 0.05, 0]):
    for _ in range(2 * thr + 20):
      nLab = int(label.max())
      if nLab == 0:
        break
      cnt = np.bincount(label, minlength=nLab + 1)
      under = (label == 0) | (cnt[label] < thr)
      if not under.any():
        break
      target = np.where((label > 0) & (cnt[label] >= thr), label, 0)
      cand = cand0[under].multiply((target > 0)[None, :]).tocsr()
      best, who = _rowMax(cand)
      take = (best > lv) & (best > 0)
      if not take.any():
        break
      rows = np.flatnonzero(under)
      label[rows[take]] = target[who[take]]
  rep.numRegions = len(np.unique(label0[label0 > 0])) + int((label0 == 0).sum())
  rep.numGrains = len(np.unique(label[label > 0])) + int((label == 0).sum())
  rep.numAbsorbed = int((label != label0).sum())
  rep.numUnassigned = int((label == 0).sum())
  return label, rep


# what one run did: how the graph coarsened, where the aggregates took a gradient and at
# which scale the pixels were read off
def report(crit, mineral, hier, rep):
  par = f"cmaha {crit.cmaha:g}, cmaha0 {crit.cmaha0:g}, quatmax {crit.quatmax / degree:g}, alpha {crit.alpha:g}, gammaW {crit.gammaW:g}"
  if crit.minPixel > 1:
    par += f", minPixel {crit.minPixel:g}"
  if np.isfinite(crit.maxDelta):
    par += f", maxDelta {np.degrees(crit.maxDelta):g}"
  lines = [f"\n fast multiscale clustering of {mineral or 'unnamed phase'}", f"   {hier.numClusters[0]} pixels, noise {hier.sigma:.3g} degree",
           f"   {par}\n", " scale   aggregates   gradients   pixels read", " ---------------------------------------------"]
  for sc, n in enumerate(hier.numClusters):
    if sc == 0:
      lines.append(f" {1:5d}   {n:10d}   {'-':>9}   {'-':>11}")
    else:
      lines.append(f" {sc + 1:5d}   {n:10d}   {hier.nLin[sc]:9d}   {rep.readPerScale[sc]:11d}")
  if rep.numStranded:
    lines.append(f"\n {rep.numStranded} pixels read off at no scale")
  tail = f"\n {rep.numRegions} regions -> {rep.numGrains} grains"
  if rep.numAbsorbed:
    tail += f", {rep.numAbsorbed} pixels absorbed into a neighbour"
  if rep.numUnassigned:
    tail += f", {rep.numUnassigned} left unassigned"
  lines.append(tail + "\n")
  print("\n".join(lines))


__all__ = ["gbcFMC"]
