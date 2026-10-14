# orientation relationships between a parent and a child phase (MATLAB
# geometry/misorientation/calcParent2Child, calcVariantId, calcParent, round2Miller;
# Julia parent2child.jl, ADR 0005 of the MATLAB branch)
#
#   p2c = calcParent2Child(c2c, p2c0), calcParent2Child(childOri, p2c0), ... local=True, quantile=0.8
#   p2c, fit = calcParent2Child(c2c, p2c0, fit=True)
#   variantId, packetId, bainId, fit = calcVariantId(parentOri, childOri, p2c)
#   parentOri, fit = calcParent(childOri, p2c), calcParent(childOri, parentOri, p2c), ids, fit = calcParent(childPairs, p2c, id=True)
#   hkl, uvw = round2Miller(ori); n1, n2, d1, d2 = round2Miller(mori)
#
# For the relationship p mapping parent to child, the child variants are p S_k over the
# parent group and the child-to-child misorientations c_k = p S_k^-1 p^-1; their angles do
# not move with p, only their axes. The fit is the trimmed chordal misfit of the
# misorientations to the nearest c_k, minimised by a fixed point iteration with
# backtracking, from the start given and from the best basins of a scan over the
# misorientation space unless local. Everything is deterministic: the subsamples stride
# the angle sorted list, so two calls on the same data agree and the order of the data
# does not matter.
from __future__ import annotations

from collections import namedtuple

import numpy as np

from .frames import crystalFrame, specimenFrame
from .miller import Miller
from .orientation import orientation
from .kernels import bestDot, closestAngle, closestImage, qconj, threads, variantAngles
from .quaternion import qmul, quaternion, rotation
from .symmetryops import projectIntoFundamentalRegionOrientation
from .symreduce import chordAngle, reductionGroups, symReduce
from .units import degree
from .vector3d import vector3d, xvector, zvector
from .display import aligned

VariantIds = namedtuple("VariantIds", ["variantId", "packetId", "bainId", "fit"])


def _vprint(verbose, *args):
  if verbose:
    print(*args)


# ------------------------------------------------------------------ the misfit and the vote

# the child variants without the identity and the child to child misorientations they give
def _childToChild(p2c):
  V = p2c.variants().flatten()
  c2c = orientation(rotation._fromData(qmul(p2c.data[None], V.inv().data)), p2c.frameB, p2c.frameB)
  keep = c2c.angle(symmetry=False) > 1e-3 * degree
  return V[keep], c2c[keep]


# the nearest c_k of every misorientation and the angle to it, reduced over the groups: the
# images l c_k r in a k-d tree over the double cover, whose Euclidean distance is the chord
# 2 sin(w/4); the c_k equivalent under the groups share their images, and the first of them
# names them. A non finite misorientation gets the first and no angle
def _nearestC2C(mori, c2c):
  from scipy.spatial import cKDTree
  L, R = reductionGroups(c2c)
  ex = mori.antipodal or c2c.antipodal
  c = c2c.data.reshape(-1, 4)
  first = np.argmax(symReduce(c[:, None], c[None, :], L, R, ex)[0] > 1 - 1e-9, axis=1)
  reps = np.flatnonzero(first == np.arange(len(c)))
  refs = np.concatenate([c[reps], qconj(c[reps])]) if ex else c[reps]
  es = qmul(qmul(L[:, None, None], refs[None, :, None]), R[None, None, :])
  label = np.broadcast_to(np.tile(reps, 2 if ex else 1)[None, :, None], es.shape[:-1]).reshape(-1)
  es = es.reshape(-1, 4)
  q = mori.data.reshape(-1, 4)
  ok = np.isfinite(q).all(axis=1)
  d, i = cKDTree(np.concatenate([es, -es])).query(q[ok], workers=threads())
  variant, omega = np.zeros(len(q), dtype=np.int64), np.full(len(q), np.nan)
  variant[ok] = label[i % len(es)]
  omega[ok] = 4 * np.arcsin(np.minimum(d / 2, 1.0))
  return variant, omega


# the residual of every misorientation: the nearest c_k, its angle, the chordal residual
# 1 - cos capped at tau, the h inliers cut on the value so that ties keep the set order
# independent; a non finite angle counts as the worst residual
_Misfit = namedtuple("_Misfit", ["V", "c2c", "omega", "variant", "ind", "fit"])


def _misfit(mori, p2c, h, tau):
  V, c2c = _childToChild(p2c)
  variant, omega = _nearestC2C(mori, c2c)
  r = np.where(np.isfinite(omega), np.minimum(1 - np.cos(omega), tau), tau)
  kth = min(h, len(r)) - 1
  ind = r <= np.partition(r, kth)[kth]
  return _Misfit(V, c2c, omega, variant, ind & (r < tau), float(np.mean(r[ind])))


# one vote per inlier: with its variant it measures p directly, proj(m, c_k) p S_k; the
# chordal mean of the votes is the update
def _vote(mori, p2c, s):
  votes = []
  for k in np.unique(s.variant[s.ind]):
    sel = s.ind & (s.variant == k)
    proj = mori[sel].projectIntoFundamentalRegion(s.c2c[k])
    votes.append(qmul(proj.data.reshape(-1, 4), s.V[k].data.reshape(1, 4)))
  if not votes:
    return None
  return p2c._new(quaternion(np.concatenate(votes)).mean().data)


# a deterministic subsample: every k-th of the angle sorted list
def _subSample(mori, n):
  if mori.size <= n:
    return mori
  order = np.argsort(mori.angle(symmetry=False), kind="stable")
  return mori[order[np.unique(np.round(np.linspace(0, mori.size - 1, n)).astype(int))]]


# the fixed point iteration from one start, every step backtracked until it descends
def _fitLocal(mori, p2c, h, tau, maxIteration, verbose=False):
  s = _misfit(mori, p2c, h, tau)
  _report(p2c, s.fit, verbose)
  for _ in range(maxIteration):
    trial = _vote(mori, p2c, s)
    if trial is None:
      break
    step, ok = 1.0, s.fit * (1 + 1e-9)
    while True:
      new = p2c._new(quaternion(np.stack([p2c.data.reshape(4), trial.data.reshape(4)])).mean(weights=[1 - step, step]).data)
      sNew = _misfit(mori, new, h, tau)
      if sNew.fit <= ok or step < 2.0 ** -6:
        break
      step /= 2
    if sNew.fit > ok:
      break
    moved = float(new.angle(p2c, symmetry=False))
    p2c, s = new, sNew
    _report(p2c, s.fit, verbose)
    if moved < 0.01 * degree:
      break
  return p2c, s


def _report(p2c, fit, verbose):
  if verbose:
    e = np.degrees(p2c.Euler()).reshape(-1)
    print(f"  ({e[0]:6.2f}°,{e[1]:6.2f}°,{e[2]:6.2f}°)  {np.degrees(np.arccos(1 - fit)):.4g}")


# the seeds of the local fits: the relationships of a grid scored on a small subsample,
# the best few far enough apart, the start given always among them
def _globalSearch(mori, p2c, h, tau, searchResolution, scanSample, numLocal, maxIteration, verbose):
  from .grids.so3grid import equispacedSO3Grid
  sub = _subSample(mori, scanSample)
  hSub = max(1, int(np.ceil(h / mori.size * sub.size)))
  g = orientation(equispacedSO3Grid(p2c.frameA, p2c.frameB, resolution=searchResolution)).flatten()
  g = orientation.cat(p2c.flatten(), g)
  # conjugating the parent group directly: the duplicates variants would drop cannot change
  # a minimum over the variants
  S = p2c.frameA.sym.properGroup().rotations().data.reshape(-1, 4)
  S = S[2 * np.arccos(np.clip(np.abs(S[:, 0]), 0, 1)) > 1e-3 * degree]
  _vprint(verbose, f"  scanning {g.size} orientations")
  fit = np.zeros(g.size)
  block = max(1, int(round(2e6 / (sub.size * len(S)))))
  for a in range(0, g.size, block):
    b = min(a + block, g.size)
    q = g.data[a:b].reshape(-1, 1, 4)
    c2c = orientation(rotation._fromData(qmul(qmul(q, qconj(S)[None]), qconj(q))), p2c.frameB, p2c.frameB)
    omega = sub.reshape(sub.size, 1, 1).angle(c2c.reshape(1, b - a, len(S)))
    r = np.sort(np.minimum(1 - np.cos(np.min(omega, axis=2)), tau), axis=0)
    fit[a:b] = np.mean(r[:hSub], axis=0)
  seeds = [p2c]
  for j in np.argsort(fit, kind="stable"):
    if len(seeds) >= numLocal + 1:
      break
    if all(float(sd.angle(g[j])) > searchResolution for sd in seeds):
      seeds.append(g[j])
  # a candidate only has to reach its basin, so a subsample serves; every candidate is
  # scored on all the data, and the caller refines the winner there
  pSub = _subSample(mori, 3000)
  hLocal = int(np.ceil(h / mori.size * pSub.size))
  best, bestFit = p2c, np.inf
  for sd in seeds:
    pj, _ = _fitLocal(pSub, sd, hLocal, tau, maxIteration)
    s = _misfit(mori, pj, h, tau)
    if s.fit < bestFit:
      best, bestFit = pj, s.fit
  return best


def calcParent2Child(mori, p2c, local=False, quantile=0.9, threshold=np.inf, maxIteration=30, maxSample=50000,
                     searchResolution=4 * degree, scanSample=250, numLocal=5, verbose=True, fit=False):
  """the parent to child orientation relationship that fits child to child misorientations
  best, refined from an initial guess

    p2c = calcParent2Child(c2c, p2c0)                           # from child to child misorientations
    p2c = calcParent2Child(childOri, p2c0)                      # from pairs of child orientations
    p2c = calcParent2Child(c2c, p2c0, quantile=0.8)             # the fit on the best 80 percent
    p2c = calcParent2Child(c2c, p2c0, local=True)               # iterate from p2c0 only
    p2c, fit = calcParent2Child(c2c, p2c0, fit=True)            # with the misfit of every misorientation

  Two children of one parent grain are related by a misorientation the relationship predicts,
  V_i^-1 V_j for two of its variants. The fitted relationship minimises the trimmed chordal
  misfit of the measured misorientations to the nearest predicted one, taken over the share
  quantile of the smallest misfits, by a fixed point iteration.

  With local=True the iteration starts from p2c0 alone. By default it also starts from the
  best basins of a scan over all relationships, so that a poor start still finds the global
  minimum; p2c0 refined on its own remains a candidate.

  Pairs of child orientations, an (n, 2) array in the specimen frame, are turned into their
  misorientations. The progress is reported unless verbose=False.

  Input
    c2c - orientation, child to child misorientations
    childOri - orientation, (n, 2) pairs of child orientations
    p2c0 - orientation, the initial guess of the parent to child relationship
  Output
    p2c - orientation, the fitted parent to child relationship
    fit - the disorientation angle of every misorientation to the nearest predicted one
  Keywords
    local - iterate from p2c0 only, without the scan
    quantile - the share of the misorientations the fit is based on
    threshold - misorientations misfitting by more than this angle are outliers
    maxIteration - the largest number of iterations
    maxSample - fit on at most this many misorientations, a random subsample of the rest
    searchResolution - the resolution of the grid of the scan
    scanSample - the number of misorientations that rank the basins of the scan
    numLocal - the number of basins of the scan fitted locally
    verbose - report the progress
    fit - also return the misfit of every misorientation, MATLAB's second output
  References
    T. Nyyssonen et al., Crystallography, morphology, and martensite transformation of prior
        austenite in intercritically annealed high-aluminum steel, Metall. Mater. Trans. A 49
        (2018), doi:10.1007/s11661-018-4904-9
  See also
    calcParent, parentGrainReconstructor.calcParent2Child, misfit, OrientationRelationshipFit"""
  if isinstance(mori.frameB, specimenFrame):
    mori = mori.reshape(-1, 2)
    mori = mori[:, 0].inv() * mori[:, 1]
  mori = mori.flatten()
  if p2c.frameB is not mori.frameA:
    raise ValueError("the misorientations are between the child frame of the relationship")
  if mori.size == 0:
    raise ValueError("the relationship is fitted to child to child misorientations, and none are given")
  tau = 1 - np.cos(min(threshold, np.pi))
  moriAll = mori
  mori = _subSample(mori, maxSample)
  h = int(np.ceil(quantile * mori.size))
  _vprint(verbose, " optimizing parent to child orientation relationship")
  start = p2c
  if not local:
    p2c = _globalSearch(mori, p2c, h, tau, searchResolution, scanSample, numLocal, maxIteration, verbose)
  p2c, s = _fitLocal(mori, p2c, h, tau, maxIteration, verbose)
  # the candidates were refined on a subsample, so the start refined on all the data may
  # still reach a lower minimum; it stays a candidate
  if not local:
    p0, s0 = _fitLocal(mori, start, h, tau, maxIteration)
    if s0.fit < s.fit:
      p2c = p0
  if fit:
    return p2c, _misfit(moriAll, p2c, moriAll.size, tau).omega
  return p2c


def misfit(mori, p2c):
  """misfit(mori, p2c): the angle of every child to child misorientation to the nearest one
  the relationship predicts"""
  mori = mori.flatten()
  return _misfit(mori, p2c, mori.size, 2.0).omega


# ------------------------------------------------------------------ variants, packets, Bain groups

# the packet or Bain group of every variant: the member of the parent family h1 the
# variant carries onto the child direction h2, one based
def _familyId(p2c, h1, h2):
  V = p2c.variants().flatten()
  d = np.stack([(V * h1[k]).dot(h2) for k in range(h1.size)], axis=1)
  return np.argmax(d, axis=1) + 1


def packetPlanes(CS):
  """the four {111} planes of the parent phase cs that the packets of the variants are sorted by"""
  return Miller([[1, 1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, -1]], CS)


def bainPlanes(CS):
  """the three {100} directions of the parent phase cs that the Bain groups of the variants are
  sorted by"""
  return Miller([[0, 0, 1], [1, 0, 0], [0, 1, 0]], CS)


# the variant of every child nearest it and the angle: the child c of the parent P is closest
# to the variant k where m = inv(P) c is closest to an image inv(V_k) inv(r) over the child
# group, the chord being the same; the images in the order of the variants, so that the first
# variant wins a tie as the argmin over the variants did. A specimen group or another child
# frame takes the table of all the angles
def _nearestVariant(parentOri, childOri, p2c):
  L, R = reductionGroups(childOri)
  if len(L) > 1 or childOri.antipodal or childOri.frameA is not p2c.frameB or childOri.frameB is not parentOri.frameB:
    a = p2c.variants(parentOri).reshape(childOri.size, -1).angle(childOri.reshape(-1, 1))
    variantId = np.argmin(a, axis=1)
    return variantId, a[np.arange(len(a)), variantId]
  par = projectIntoFundamentalRegionOrientation(parentOri).data.reshape(-1, 4)
  m = qmul(qconj(par), childOri.data.reshape(-1, 4))
  V = p2c.variants().data.reshape(-1, 4)
  es = qmul(qconj(V)[:, None], qconj(R)[None, :]).reshape(-1, 4)
  k = bestDot(m, es)[1]
  return k // len(R), chordAngle(m, es[k])


def calcVariantId(parentOri, childOri, p2c, packet=None, bain=None):
  """the variant, packet and Bain group of every child orientation with respect to its parent,
  MATLAB's calcVariantId

    variantId, packetId, bainId, fit = calcVariantId(parentOri, childOri, p2c)  # the {111} packets and {001} Bain groups
    hklParent = Miller([[1, 1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, -1]], p2c.CS)
    variantId, packetId, bainId, fit = calcVariantId(parentOri, childOri, p2c, packet=(hklParent, Miller(1, 0, 1, p2c.SS)))  # packets by planes of one's own

  For every child the variant of its parent it belongs to, one based, with the misfit, and
  the packet and Bain group of that variant: the packet by the parent {111} plane the
  variant maps onto the child (011), the Bain group by the parent {001} plane mapped onto
  the child (100), unless packet=(hklParent, hklChild) or bain=(hklParent, hklChild) name
  other families. MATLAB computes the packet and Bain ids only when the planes are given; the
  port always does, with these defaults.

  Input
    parentOri - orientation of the parent phase, one or one per child
    childOri - orientation of the child phase
    p2c - orientation, the parent to child orientation relationship
  Output
    variantId - the one based variant ids, in the order of p2c.variants()
    packetId - the one based packet ids
    bainId - the one based Bain group ids
    fit - the misorientation angle between the child and its variant, in radians
  Keywords
    packet - (hklParent, hklChild), the parent planes (one Miller per packet) and the child
        plane they map onto
    bain - (hklParent, hklChild), the parent directions (one Miller per Bain group) and the
        child direction they map onto
  See also
    orientation.variants, calcParent, packetPlanes, bainPlanes, MartensiteVariants"""
  childOri = childOri.flatten()
  parentOri = parentOri.flatten()
  if parentOri.size == 1 and childOri.size > 1:
    parentOri = parentOri._new(np.broadcast_to(parentOri.data.reshape(1, 4), (childOri.size, 4)).copy())
  if parentOri.size != childOri.size:
    raise ValueError("one parent per child orientation")
  variantId, fit = _nearestVariant(parentOri, childOri, p2c)
  h1, h2 = packet if packet is not None else (packetPlanes(p2c.frameA), Miller(1, 0, 1, p2c.frameB))
  packetId = _familyId(p2c, h1, h2)[variantId]
  h1, h2 = bain if bain is not None else (bainPlanes(p2c.frameA), Miller(1, 0, 0, p2c.frameB))
  bainId = _familyId(p2c, h1, h2)[variantId]
  return VariantIds(variantId + 1, packetId, bainId, fit)


# ------------------------------------------------------------------ the parent of children

def calcParent(childOri, *args, numFit=1, id=False, weights=None, threshold=10 * degree):
  """the parent orientations of child orientations under a parent to child orientation
  relationship, with their fit

    parentOri, fit = calcParent(childOri, p2c)                 # the parent of the children of one grain
    parentOri, fit = calcParent(childOri, p2c, weights=w)      # the children weighted
    parentOri, fit = calcParent(childOri, parentOri, p2c)      # the parent variant of every child
    parentId, fit = calcParent(childOri, parentOri, p2c, id=True)  # its id instead
    parentId, fit = calcParent(childOri, parentOri, p2c, numFit=2, id=True)  # the two best
    parentOri, fit = calcParent(childPairs, p2c)               # the variant pair of every boundary
    parentOri, fit = calcParent(childTriples, p2c)             # the variant triple of every triple point

  The form of the input decides what is reconstructed:

  - many children of one grain: every child proposes its parent variants, and the proposal
  the others agree with best is refined as their weighted mean.
  - children with a parent orientation, one or one per child: every child takes the parent
  variant nearest that parent.
  - pairs or triples of children, an (n, 2) or (n, 3) array as a boundary or a triple point
  gives them: the parent variants that agree best on one parent, the fit their disorientation.

  numFit returns the best few fits instead of one, along a last axis. With id=True the one
  based ids of the parent variants come back, childOri * p2c.variants(parent=True)[id - 1]
  being the parent.

  Input
    childOri - orientation of the child phase; (n, 2) pairs or (n, 3) triples of them
    parentOri - orientation of the parent phase, one or one per child
    p2c - orientation, the parent to child orientation relationship, or its inverse
  Output
    parentOri - orientation of the parent phase
    parentId - the one based parent variant ids
    fit - the disorientation angle to the relationship, in radians
  Keywords
    numFit - the number of best fits to return
    id - return the parent variant ids instead of the orientations
    weights - the weights of the children of one grain in the mean
    threshold - the largest disagreement of the first pair of a triple that is considered
  See also
    calcParent2Child, orientation.variants, parentGrainReconstructor, calcVariantId,
    MaParentGrainReconstruction"""
  parentOri = None
  if len(args) == 2:
    parentOri, p2c = args
  else:
    (p2c,) = args
  if p2c.frameB is not childOri.frameA:
    p2c = p2c.inv()
  PV = p2c.variants(parent=True).flatten()
  numV = PV.size
  if parentOri is not None:
    # every child with its parent: the parent variant their misorientation is nearest to,
    # reduced over the parent group alone, so that child * PV[id] is the parent as stored
    child = childOri.flatten()
    par = parentOri.flatten()
    if par.size == 1 and child.size > 1:
      par = par._new(np.broadcast_to(par.data.reshape(1, 4), (child.size, 4)).copy())
    if child.size == 0:
      fit = np.zeros((0, numV))
      return _asParents(child, PV, *_best(fit, numFit, (numV,)), id)
    mori = _stripChild(child.inv() * par)
    fit = mori.reshape(-1, 1).angle(_stripChild(PV).reshape(1, -1))
    return _asParents(childOri.flatten(), PV, *_best(fit, numFit, (numV,)), id)
  if childOri.ndim == 2 and childOri.shape[1] in (2, 3):
    # the angle between the parent candidates of two children depends on their misorientation
    # alone, so each pair is one table over the variant pairs
    pv = childOri[:, :1] * PV[:1].reshape(1, 1)
    L, G = reductionGroups(pv)
    q = childOri.data.reshape(-1, childOri.shape[1], 4)

    def table(a, b):
      if len(L) > 1:
        c1 = childOri[:, a].reshape(-1, 1) * PV.reshape(1, -1)
        c2 = childOri[:, b].reshape(-1, 1) * PV.reshape(1, -1)
        return np.asarray(c1.reshape(-1, numV, 1).angle(c2.reshape(-1, 1, numV)))
      return variantAngles(qmul(qconj(q[:, a]), q[:, b]), PV.data.reshape(-1, 4), G)

    if childOri.shape[1] == 2:
      # pairs of children: the variant pair whose parents agree best
      return _asParents(childOri, PV, *_best(table(0, 1).transpose(0, 2, 1).reshape(-1, numV * numV), numFit, (numV, numV)), id)
    # triples of children: the largest of the three pairwise disagreements, where the first
    # pair agrees within the threshold, in blocks that bound the table of variant triples
    f12, f23, f13 = table(0, 1), table(1, 2), table(0, 2)
    n = len(q)
    ids, vals = [], []
    if n == 0:
      i, v = _best(np.zeros((0, numV ** 3)), numFit, (numV, numV, numV))
      ids, vals = [i], [v]
    for s0 in range(0, n, 256):
      b = slice(s0, min(n, s0 + 256))
      fit = np.broadcast_to(f12[b, :, :, None], (f12[b].shape[0], numV, numV, numV)).copy()
      ok = np.any(f12[b] < threshold, axis=1)
      fit = np.where(ok[:, None, :, None], np.maximum(fit, f23[b, None, :, :]), fit)
      ok = np.any(fit < threshold, axis=2)
      fit = np.where(ok[:, :, None, :], np.maximum(fit, f13[b, :, None, :]), fit)
      i, v = _best(fit.transpose(0, 3, 2, 1).reshape(fit.shape[0], -1), numFit, (numV, numV, numV))
      ids.append(i)
      vals.append(v)
    return _asParents(childOri, PV, np.concatenate(ids), np.concatenate(vals), id)
  # many children of one parent
  childOri = childOri.flatten()
  if childOri.size == 0:
    raise ValueError("the parent of a grain is reconstructed from its child orientations, and none are given")
  w = np.ones(childOri.size) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
  w = w / np.sum(w)
  seed = childOri[int(np.argmax(w))]
  cand = qmul(seed.data.reshape(1, 4), PV.data.reshape(-1, 4))
  # every candidate parent at once: the parent variant of every child nearest it, the parents
  # these give, their mean and how far they lie from it; the first best candidate wins
  n, cq, pv = childOri.size, childOri.data.reshape(-1, 4), PV.data.reshape(-1, 4)
  if not np.isfinite(cq).all():
    return _parentOfManyLoop(childOri, cand, PV, p2c, w, id)
  child = childOri._new(np.tile(cq, (numV, 1)))
  ids, _ = calcParent(child, orientation(rotation._fromData(np.repeat(cand, n, axis=0)), p2c.frameA, childOri.frameB), p2c, id=True)
  ids = np.asarray(ids).reshape(numV, n)
  P = qmul(np.tile(cq, (numV, 1)), pv[ids.reshape(-1) - 1])
  L, R = reductionGroups(orientation(rotation._fromData(P[:1]), p2c.frameA, childOri.frameB))
  m = _orientationMeans(P.reshape(numV, n, 4), w, L, R)
  f = np.sum(closestAngle(P, np.repeat(m, n, axis=0), L, R).reshape(numV, n) * w, axis=1)
  if not np.isfinite(f).any():
    return None, np.inf
  k = int(np.argmin(np.where(np.isnan(f), np.inf, f)))
  return (ids[k] if id else orientation(rotation._fromData(m[k]), p2c.frameA, childOri.frameB)), float(f[k])


# the candidates one at a time, for children of which some have no orientation
def _parentOfManyLoop(childOri, cand, PV, p2c, w, id):
  best, bestFit, bestIds = None, np.inf, None
  for k in range(len(cand)):
    ids, _ = calcParent(childOri, orientation(rotation._fromData(cand[k]), p2c.frameA, childOri.frameB), p2c, id=True)
    pOri = orientation(rotation._fromData(qmul(childOri.data.reshape(-1, 4), PV.data.reshape(-1, 4)[ids - 1])), p2c.frameA, childOri.frameB)
    m = pOri.mean(weights=w, robust=True)
    f = float(np.sum(pOri.angle(m) * w))
    if f < bestFit:
      best, bestFit, bestIds = m, f, ids
  return (bestIds if id else best), bestFit


# the mean orientation of every list of a stack (K, n, 4), as orientationMean with weights and
# robust: three times every element moved to its equivalent closest to the running mean and the
# robust quaternion mean taken
def _orientationMeans(Q, w, L, R, iterations=3):
  K, n = Q.shape[:2]
  m = Q[:, 0].copy()
  for _ in range(iterations):
    moved = closestImage(np.repeat(m, n, axis=0), Q.reshape(-1, 4), L, R, False)[1].reshape(K, n, 4)
    m = _robustMeans(moved, w)
  return m


# the quantile of every row without its NaN entries, as np.nanquantile row by row: the rows of
# as many entries together over their sorted values, NaN sorting last
def _rowQuantile(x, q):
  s, cnt = np.sort(x, axis=1), np.sum(~np.isnan(x), axis=1)
  out = np.full(len(x), np.nan)
  for c in np.unique(cnt[cnt > 0]):
    r = cnt == c
    out[r] = np.quantile(s[r, :c], q, axis=1)
  return out


# the robust weighted mean of every list of a stack, as quaternion.mean(robust=True): the
# principal eigenvector of the moment matrix, then the elements beyond 2.5 times the 0.8
# quantile of their angles to it dropped and the mean taken again, while more than four remain
def _robustMeans(q, w):
  K, n = q.shape[:2]
  keep = np.ones((K, n), dtype=bool)
  active = np.ones(K, dtype=bool)
  m = np.empty((K, 4))
  while active.any():
    a = np.flatnonzero(active)
    qa, ka = q[a], keep[a]
    T = np.einsum("kni,knj->kij", qa * np.where(ka, w, 0.0)[..., None], qa)
    m[a] = np.linalg.eigh(T)[1][..., -1]
    d = np.sum(qa * m[a][:, None], axis=-1)
    h = np.linalg.norm(m[a][:, None] - np.where(d < 0, -1.0, 1.0)[..., None] * qa, axis=-1)
    omega = 4 * np.arcsin(np.minimum(h / 2, 1.0))
    cut = 2.5 * _rowQuantile(np.where(ka, omega, np.nan), 0.8)
    drop = ka & ~(omega <= cut[:, None])
    stop = (ka.sum(axis=1) <= 4) | ~drop.any(axis=1)
    keep[a[~stop]] &= ~drop[~stop]
    active[a[stop]] = False
  return m


# the ids as they are, or the parents they name: every child times its parent variant
def _asParents(childOri, PV, ids, fit, id):
  if id:
    return ids, fit
  q = childOri.data.reshape(childOri.shape[0], childOri.shape[1], 4) if childOri.ndim == 2 else childOri.data.reshape(-1, 1, 4)
  ids = np.asarray(ids)
  if ids.ndim == 1:
    parents = qmul(q[:, 0], PV.data.reshape(-1, 4)[ids - 1])
  elif ids.ndim == 2 and childOri.ndim == 1:
    parents = qmul(q[:, :1], PV.data.reshape(-1, 4)[ids - 1])
  elif ids.ndim == 2:
    parents = qmul(q, PV.data.reshape(-1, 4)[ids - 1])
  else:
    parents = qmul(q[:, None], PV.data.reshape(-1, 4)[ids - 1])
  return orientation(rotation._fromData(parents), PV.frameA, childOri.frameB), fit


def _stripChild(o):
  return orientation(rotation._fromData(o.data), o.frameA, o.frameB.stripSym())


# the best fits of every row as one based ids into the variant table, and the values; a row
# of several variants is laid out with the first running fastest, which is how MATLAB
# breaks ties between fits. The best few of finite rows are taken one at a time, the first
# minimum on a tie as in the stable sort, which a row of thousands of variant triples does not need
def _best(fit, numFit, dims):
  if numFit == 1:
    k = np.argmin(fit, axis=1)
    value = fit[np.arange(len(fit)), k]
  elif numFit >= fit.shape[1] or not np.isfinite(fit).all():
    k = np.argsort(fit, axis=1, kind="stable")[:, :numFit]
    value = np.take_along_axis(fit, k, axis=1)
  else:
    rest, rows, ks = fit.copy(), np.arange(len(fit)), []
    for _ in range(numFit):
      ks.append(np.argmin(rest, axis=1))
      rest[rows, ks[-1]] = np.inf
    k = np.stack(ks, axis=1)
    value = np.take_along_axis(fit, k, axis=1)
  ids = np.stack(np.unravel_index(k, dims, order="F"), axis=-1) + 1
  return (ids[..., 0] if len(dims) == 1 else ids), value


# ------------------------------------------------------------------ Miller indices of an orientation

def allHKL(maxIndex):
  """every index triple up to maxIndex, coprime, without (0, 0, 0)"""
  r = np.arange(-maxIndex, maxIndex + 1)
  h, k, l = (a.reshape(-1) for a in np.meshgrid(r, r, r, indexing="ij"))
  keep = np.gcd(np.gcd(h, k), l) == 1
  return np.stack([h[keep], k[keep], l[keep]], axis=1)


def _roundSet(v, style, maxIndex):
  from .miller import roundIndices
  return roundIndices(v, style, maxIndex)


def round2Miller(mori, penalty=0.002, maxIndex=None, nextFit=1):
  """the lattice planes and directions of low indices an orientation or a misorientation aligns

    hkl, uvw = round2Miller(ori)                           # the plane normal along z and the direction along x
    n1, n2, d1, d2 = round2Miller(mori)                    # the parallel planes and directions of two crystals
    n1, n2, d1, d2 = round2Miller(mori, penalty=0.01)      # smaller indices preferred more
    n1, n2, d1, d2 = round2Miller(mori, maxIndex=6)        # indices up to 6
    n1, n2, d1, d2 = round2Miller(mori, nextFit=2)         # the second best fit

  For an orientation, the plane (hkl) along the specimen z and the direction [uvw] along the
  specimen x, each rounded on its own to the nearest low indices, as MATLAB's; the two need
  not be perpendicular. penalty and nextFit do not apply.

  For a misorientation, the planes n1, n2 and the directions d1, d2, d1 in n1, with
  mori * n1 = n2 and mori * d1 = d2 as nearly as indices up to maxIndex allow. Each candidate
  scores its misfit angle plus penalty times the size of its indices, and the lowest score
  wins. The result displays as a table of the parallel planes and directions with the angle
  of the relation to the misorientation; a list of misorientations gives lists.

  Input
    ori - orientation, crystal to specimen
    mori - orientation, a misorientation between two crystals
  Output
    hkl, uvw - Miller, the plane normal and the direction
    n1, n2, d1, d2 - Miller, the parallel planes and directions
  Keywords
    penalty - the weight of the size of the indices against the misfit angle
    maxIndex - the largest index, 4 for a misorientation and 12 for an orientation by default
    nextFit - which fit to return, 1 the best
  See also
    CSL, orientation.map, orientation.KurdjumovSachs, Miller.round"""
  if isinstance(mori.frameB, specimenFrame) or not isinstance(mori.frameB, crystalFrame):
    # the plane and the direction are rounded in their own indices, over MATLAB's twelve
    inv = mori.inv()
    hkl = _roundSet(inv * zvector, "hkl", 12 if maxIndex is None else maxIndex)
    uvw = _roundSet(inv * xvector, "uvw", 12 if maxIndex is None else maxIndex)
    return _stripped(hkl, "hkl"), _stripped(uvw, "uvw")
  maxIndex = 4 if maxIndex is None else maxIndex
  if mori.size > 1:
    parts = [round2Miller(mori.flatten()[i], penalty, maxIndex, nextFit) for i in range(mori.size)]
    return tuple(vector3d(np.stack([p[j].data.reshape(3) for p in parts]), frame=parts[0][j].frame, dispStyle=parts[0][j].dispStyle) for j in range(4))
  cs = mori.frameA
  H = allHKL(maxIndex).astype(float)
  # all plane normals, carried over and rounded there
  n1 = Miller(H, cs)
  n2 = mori * n1
  rn2 = _roundSet(n2, "hkl", maxIndex)
  hkl2 = rn2.hkl.reshape(-1, 3)
  omegaH = rn2.angle(n2, symmetry=False) + (H[:, 0] ** 2 + H[:, 1] ** 2 + 0.995 * H[:, 2] ** 2 + hkl2 ** 2 @ [1, 1, 0.995]
                                             + 0.01 * np.sum(hkl2 < -0.1, axis=1)) * penalty
  # all directions, the same way
  d1 = Miller(H, cs, "uvw")
  d2 = mori * d1
  rd2 = _roundSet(d2, "uvw", maxIndex)
  uvw2 = rd2.uvw.reshape(-1, 3)
  omegaD = rd2.angle(d2, symmetry=False) + (1.05 * np.sum(H ** 2, axis=1) + np.sum(uvw2 ** 2, axis=1)
                                             + 0.01 * np.sum(uvw2 < -0.1, axis=1)) * penalty
  # directions have to lie in the planes, before and after
  ortho1 = np.abs(np.pi / 2 - n1.reshape(-1, 1).angle(d1.reshape(1, -1), symmetry=False)) > 1e-5
  ortho2 = np.abs(np.pi / 2 - rn2.reshape(-1, 1).angle(rd2.reshape(1, -1), symmetry=False)) > 1e-5
  fit = omegaH[:, None] + omegaD[None, :] + 100 * ortho1 + 100 * ortho2
  order = np.argsort(fit.reshape(-1), kind="stable")
  ih, idx = np.unravel_index(order[nextFit - 1], fit.shape)
  # the four winners are the planes and directions this one rotation takes onto each
  # other, not the families they come from
  out = (_stripped(n1[ih], "hkl"), _stripped(rn2[ih], "hkl"), _stripped(d1[idx], "uvw"), _stripped(rd2[idx], "uvw"))
  return millerRelation(out, mori)


class millerRelation(tuple):
  """n1, n2, d1, d2 of round2Miller(mori), displayed as MATLAB's table of the parallel planes,
  the parallel directions and the angle to the misorientation they make"""

  def __new__(cls, parts, mori):
    obj = super().__new__(cls, parts)
    obj.mori = mori
    return obj

  def __repr__(self):
    from .orientation import orientation
    n1, n2, d1, d2 = self
    fit = orientation.map(n1, n2, d1, d2).angle(self.mori.reshape(()), symmetry=True) / degree
    planes, dirs = f"{n1.char()} || {n2.char()}", f"{d1.char()} || {d2.char()}"
    return (aligned(f"{'plane parallel':>{len(planes)}}   {'direction parallel':>{len(dirs)}}     fit\n"
            f"{planes}   {dirs}  {round(float(fit), 2):>4g}°"))


# a plane or direction as one of its own, the group gone but the four index style kept
def _stripped(v, style):
  out = vector3d(v.data.reshape(3), frame=v.frame.stripSym())
  out.dispStyle = ("hkil" if style == "hkl" else "UVTW") if v.lattice.isTriHex else style
  # the smallest integers in the style shown, [011-1] rather than [033-3]
  idx = np.asarray(getattr(out, out.dispStyle), dtype=float).reshape(-1)
  r = np.round(idx)
  if np.all(np.abs(idx - r) < 1e-6) and np.any(r != 0):
    g = np.gcd.reduce(np.abs(r).astype(int))
    if g > 1:
      out = out / g
      out.dispStyle = ("hkil" if style == "hkl" else "UVTW") if v.lattice.isTriHex else style
  return out


__all__ = ["calcParent2Child", "misfit", "calcVariantId", "calcParent", "round2Miller", "allHKL", "packetPlanes", "bainPlanes"]
