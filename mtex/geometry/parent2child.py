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
from math import gcd

import numpy as np

from .frames import crystalFrame, specimenFrame
from .miller import Miller
from .orientation import orientation
from .kernels import variantAngles
from .quaternion import qmul, quaternion, rotation
from .symreduce import reductionGroups
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


# the angle of every row to every column, both lists reduced over their groups
def _angleOuter(a, b):
  return a.reshape(a.size, 1).angle(b.reshape(1, b.size))


# the residual of every misorientation: the nearest c_k, its angle, the chordal residual
# 1 - cos capped at tau, the h inliers cut on the value so that ties keep the set order
# independent; a non finite angle counts as the worst residual
def _misfit(mori, p2c, h, tau):
  V, c2c = _childToChild(p2c)
  a = _angleOuter(mori, c2c)
  variant = np.argmin(np.where(np.isfinite(a), a, np.inf), axis=1)
  omega = a[np.arange(len(a)), variant]
  r = np.where(np.isfinite(omega), np.minimum(1 - np.cos(omega), tau), tau)
  rs = np.sort(r)
  ind = r <= rs[min(h, len(rs)) - 1]
  return dict(V=V, c2c=c2c, omega=omega, variant=variant, ind=ind & (r < tau), fit=float(np.mean(r[ind])))


# one vote per inlier: with its variant it measures p directly, proj(m, c_k) p S_k; the
# chordal mean of the votes is the update
def _vote(mori, p2c, s):
  votes = []
  for k in np.unique(s["variant"][s["ind"]]):
    sel = s["ind"] & (s["variant"] == k)
    proj = mori[sel].project2FundamentalRegion(s["c2c"][k])
    votes.append(qmul(proj.data.reshape(-1, 4), s["V"][k].data.reshape(1, 4)))
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
  _report(p2c, s["fit"], not verbose)
  for _ in range(maxIteration):
    trial = _vote(mori, p2c, s)
    if trial is None:
      break
    step, ok = 1.0, s["fit"] * (1 + 1e-9)
    while True:
      new = p2c._new(quaternion(np.stack([p2c.data.reshape(4), trial.data.reshape(4)])).mean(weights=[1 - step, step]).data)
      sNew = _misfit(mori, new, h, tau)
      if sNew["fit"] <= ok or step < 2.0 ** -6:
        break
      step /= 2
    if sNew["fit"] > ok:
      break
    moved = float(new.angle(p2c, symmetry=False))
    p2c, s = new, sNew
    _report(p2c, s["fit"], not verbose)
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
  g = orientation.cat(p2c.flatten(), g) if hasattr(orientation, "cat") else _cat(p2c, g)
  # conjugating the parent group directly: the duplicates variants would drop cannot change
  # a minimum over the variants
  S = p2c.frameA.sym.properGroup().rotations().data.reshape(-1, 4)
  S = S[2 * np.arccos(np.clip(np.abs(S[:, 0]), 0, 1)) > 1e-3 * degree]
  _vprint(not verbose, f"  scanning {g.size} orientations")
  fit = np.zeros(g.size)
  block = max(1, int(round(2e6 / (sub.size * len(S)))))
  for a in range(0, g.size, block):
    b = min(a + block, g.size)
    q = g.data[a:b].reshape(-1, 1, 4)
    c2c = orientation(rotation._fromData(qmul(qmul(q, _conj(S)[None]), _conj(q))), p2c.frameB, p2c.frameB)
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
    if s["fit"] < bestFit:
      best, bestFit = pj, s["fit"]
  return best


def _conj(q):
  return q * np.array([1.0, -1, -1, -1])


def _cat(a, b):
  return a._new(np.concatenate([a.data.reshape(-1, 4), b.data.reshape(-1, 4)]))


def calcParent2Child(mori, p2c, local=False, quantile=0.9, threshold=np.inf, maxIteration=30, maxSample=50000,
                     searchResolution=4 * degree, scanSample=250, numLocal=5, verbose=False, fit=False):
  """calcParent2Child(c2c, p2c0), calcParent2Child(childOri, p2c0), calcParent2Child(c2c, p2c0, local=True)

  The parent to child orientation relationship fitted to child to child misorientations,
  or to pairs of child orientations: the trimmed chordal misfit of the misorientations to
  the nearest child to child misorientation the relationship predicts, minimised by a
  fixed point iteration from p2c0 and, unless local, from the best basins of a scan over
  the misorientation space. quantile is the share of the misorientations the fit is based
  on, misorientations misfitting by more than threshold are treated as outliers; with
  fit=True the disorientation angle of every misorientation to the result comes too.
  """
  if isinstance(mori.frameB, specimenFrame):
    mori = mori.reshape(-1, 2)
    mori = mori[:, 0].inv() * mori[:, 1]
  mori = mori.flatten()
  if p2c.frameB is not mori.frameA:
    raise ValueError("the misorientations are between the child frame of the relationship")
  tau = 1 - np.cos(min(threshold, np.pi))
  moriAll = mori
  mori = _subSample(mori, maxSample)
  h = int(np.ceil(quantile * mori.size))
  _vprint(not verbose, " optimizing parent to child orientation relationship")
  start = p2c
  if not local:
    p2c = _globalSearch(mori, p2c, h, tau, searchResolution, scanSample, numLocal, maxIteration, not verbose)
  p2c, s = _fitLocal(mori, p2c, h, tau, maxIteration, not verbose)
  # the candidates were refined on a subsample, so the start refined on all the data may
  # still reach a lower minimum; it stays a candidate
  if not local:
    p0, s0 = _fitLocal(mori, start, h, tau, maxIteration)
    if s0["fit"] < s["fit"]:
      p2c = p0
  if fit:
    return p2c, _misfit(moriAll, p2c, moriAll.size, tau)["omega"]
  return p2c


def misfit(mori, p2c):
  """misfit(mori, p2c): the angle of every child to child misorientation to the nearest one
  the relationship predicts"""
  mori = mori.flatten()
  return _misfit(mori, p2c, mori.size, 2.0)["omega"]


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


def calcVariantId(parentOri, childOri, p2c, packet=None, bain=None):
  """variantId, packetId, bainId, fit = calcVariantId(parentOri, childOri, p2c)

  For every child the variant of its parent it belongs to, one based, with the misfit, and
  the packet and Bain group of that variant: the packet by the parent {111} plane the
  variant maps onto the child (011), the Bain group by the parent {001} plane mapped onto
  the child (100), unless packet=(hklParent, hklChild) or bain=(hklParent, hklChild) name
  other families.
  """
  childOri = childOri.flatten()
  parentOri = parentOri.flatten()
  if parentOri.size == 1 and childOri.size > 1:
    parentOri = parentOri._new(np.broadcast_to(parentOri.data.reshape(1, 4), (childOri.size, 4)).copy())
  if parentOri.size != childOri.size:
    raise ValueError("one parent per child orientation")
  cv = p2c.variants(parentOri)
  a = cv.reshape(childOri.size, -1).angle(childOri.reshape(-1, 1))
  variantId = np.argmin(a, axis=1)
  fit = a[np.arange(len(a)), variantId]
  h1, h2 = packet if packet is not None else (packetPlanes(p2c.frameA), Miller(1, 0, 1, p2c.frameB))
  packetId = _familyId(p2c, h1, h2)[variantId]
  h1, h2 = bain if bain is not None else (bainPlanes(p2c.frameA), Miller(1, 0, 0, p2c.frameB))
  bainId = _familyId(p2c, h1, h2)[variantId]
  return VariantIds(variantId + 1, packetId, bainId, fit)


# ------------------------------------------------------------------ the parent of children

def calcParent(childOri, *args, numFit=1, id=False, weights=None, threshold=10 * degree):
  """parentOri, fit = calcParent(childOri, p2c)               the parent of the children of one grain
  parentOri, fit = calcParent(childOri, parentOri, p2c)       the parent variant of every child
  parentOri, fit = calcParent(childPairs, p2c)                the variant pair of every boundary
  parentOri, fit = calcParent(childTriples, p2c)              the variant triple of every triple point

  The parent of many children of one grain: every child proposes its parent variants, the
  proposal the others agree with best is refined as their weighted mean. For a child with a
  parent orientation, one or one for all, the candidate of the child nearest it; for pairs and
  triples of children the candidates that agree best on a parent, and their disorientation;
  numFit gives the best few instead of one. With id=True the one based ids of the parent
  variants instead of the orientations, child * PV[id] being the parent. The fit is the
  disorientation to the relationship.
  """
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
      return variantAngles(qmul(_conj(q[:, a]), q[:, b]), PV.data.reshape(-1, 4), G)

    if childOri.shape[1] == 2:
      # pairs of children: the variant pair whose parents agree best
      return _asParents(childOri, PV, *_best(table(0, 1).transpose(0, 2, 1).reshape(-1, numV * numV), numFit, (numV, numV)), id)
    # triples of children: the largest of the three pairwise disagreements, where the first
    # pair agrees within the threshold, in blocks that bound the table of variant triples
    f12, f23, f13 = table(0, 1), table(1, 2), table(0, 2)
    n = len(q)
    ids, vals = [], []
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
  w = np.ones(childOri.size) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
  w = w / np.sum(w)
  seed = childOri[int(np.argmax(w))]
  candidates = orientation(rotation._fromData(qmul(seed.data.reshape(1, 4), PV.data.reshape(-1, 4))), p2c.frameA, childOri.frameB)
  best, bestFit, bestIds = None, np.inf, None
  for k in range(numV):
    ids, _ = calcParent(childOri, candidates[k], p2c, id=True)
    pOri = orientation(rotation._fromData(qmul(childOri.data.reshape(-1, 4), PV.data.reshape(-1, 4)[ids - 1])), p2c.frameA, childOri.frameB)
    m = pOri.mean(weights=w, robust=True)
    f = float(np.sum(pOri.angle(m) * w))
    if f < bestFit:
      best, bestFit, bestIds = m, f, ids
  return (bestIds if id else best), bestFit


# the ids as they are, or the parents they name: every child times its parent variant
def _asParents(childOri, PV, ids, fit, id):
  if id:
    return ids, fit
  q = childOri.data.reshape(childOri.shape[0], -1, 4) if childOri.ndim == 2 else childOri.data.reshape(-1, 1, 4)
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
# breaks ties between fits
def _best(fit, numFit, dims):
  if numFit == 1:
    k = np.argmin(fit, axis=1)
    value = fit[np.arange(len(fit)), k]
  else:
    order = np.argsort(fit, axis=1, kind="stable")[:, :numFit]
    value = np.take_along_axis(fit, order, axis=1)
    k = order
  ids = np.stack(np.unravel_index(k, dims, order="F"), axis=-1) + 1
  return (ids[..., 0] if len(dims) == 1 else ids), value


# ------------------------------------------------------------------ Miller indices of an orientation

def allHKL(maxIndex):
  """every index triple up to maxIndex, coprime, without (0, 0, 0)"""
  r = np.arange(-maxIndex, maxIndex + 1)
  h, k, l = (a.reshape(-1) for a in np.meshgrid(r, r, r, indexing="ij"))
  g = np.array([gcd(gcd(abs(int(a)), abs(int(b))), abs(int(c))) for a, b, c in zip(h, k, l)])
  keep = g == 1
  return np.stack([h[keep], k[keep], l[keep]], axis=1)


def _roundSet(v, style, maxIndex):
  from .miller import roundIndices
  return roundIndices(v, style, maxIndex)


def round2Miller(mori, penalty=0.002, maxIndex=None, nextFit=1):
  """hkl, uvw = round2Miller(ori): the plane and the direction of the crystal along the
  specimen z and x, as their nearest low indices; n1, n2, d1, d2 = round2Miller(mori): the
  parallel planes and the parallel directions of the two crystals the misorientation
  relates, the smallest indices that fit best, penalty weighing the size of the indices"""
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
  cs, csB = mori.frameA, mori.frameB
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
