# parent grain reconstruction (MATLAB @parentGrainReconstructor, Julia reconstruction.jl)
#
#   job = parentGrainReconstructor(ebsd, grains, p2c)
#   job.calcParent2Child(); fit, pairs = job.calcGBFit()
#   job.calcGBVotes(p2c=True, threshold=2*degree), job.calcTPVotes(minFit=2.5*degree, maxFit=5*degree)
#   job.calcParentFromVote(minProb=0.5)
#   job.calcVariantGraph(), job.clusterVariantGraph(), job.calcGraph(), job.clusterGraph(),
#   job.calcParentFromGraph(), job.mergeByGraph()                                    (variantgraph.py)
#   job.mergeSimilar(threshold=7.5*degree), job.mergeInclusions(maxSize=50), job.revert(ind)
#   job.calcVariants(), job.calcParentEBSD(), job.parentGrains, job.transformedGrains, job.packetId
#
# A child grain of mean orientation o has one parent candidate per variant, o * PV_k with PV
# the parent variants of the relationship; the reconstruction picks one per grain so that
# neighbouring grains agree. Two routes meet in job.votes: the votes of the boundaries and
# of the triple points, and the clusters of the variant graph (arXiv:2201.02103). The votes
# are MATLAB's table, the best numFit variants of every grain sorted with their accumulated
# probability (or fit, or count), so that a threshold of a MATLAB script means the same here.
# A job is the one handle of the port: a reconstruction is a sequence of steps over a state
# that grows, each judged before the next is taken; the steps change the job and return
# nothing.
from __future__ import annotations

from typing import TYPE_CHECKING
from types import SimpleNamespace

import numpy as np
from scipy.special import erf

from ..geometry.kernels import qconj
from .grains import _at, _idPositions, _pairKey, grain2d
from ..geometry.orientation import orientation
from ..geometry.parent2child import calcParent, calcParent2Child, calcVariantId, round2Miller
from ..geometry.quaternion import qmul
from ..geometry.units import degree
from .phaselist import phaseList
from ..geometry.display import aligned



def _ori(q, frameA, frameB):
  return orientation._fromRotation(np.asarray(q, dtype=float).reshape(-1, 4), (frameA, frameB))


def probabilityOf(fit, threshold, tol):
  """the probability that a misfit of that angle still means one parent: the cumulative
  Gaussian around the threshold over the width of the tolerance"""
  return 1 - 0.5 * (1 + erf(2 * (np.asarray(fit, dtype=float) - threshold) / tol))


# MTEX's quantile, which shadows MATLAB's: the round(n q)-th of the n sorted values without
# nan, no interpolation
def _quantile(x, q):
  x = np.sort(np.asarray(x, dtype=float).reshape(-1))
  x = x[~np.isnan(x)]
  return x[max(1, int(np.floor(len(x) * q + 0.5))) - 1] if len(x) else np.nan


# a number as MATLAB's xnum2str writes it: the decimals by its size, trailing zeros dropped
def _xnum(x):
  if np.isnan(x):
    return "NaN"
  a = abs(x)
  if a > 1e6 or (a <= 1e-4 and a > 0):
    s = f"{x:.2g}"
  elif a >= 10:
    s = f"{x:.0f}"
  else:
    digits = 1 if a >= 1 else 2 if a > 0.1 else 3 if a > 0.01 else 4 if a > 0.001 else 5
    s = f"{x:.{digits}f}".rstrip("0") if a > 0 else "0"
  s = s.rstrip(".")
  return "0" if s == "-0" else s


class voteTable:
  """MATLAB's table of votes: a row per grain, parentId the best variants of the grain and
  prob, fit or count their value, sorted best first"""

  def __init__(self, parentId, **value):
    (kind, v), = value.items()
    self.parentId = np.asarray(parentId)
    self.kind = kind
    setattr(self, kind, np.asarray(v, dtype=float))

  @property
  def value(self):
    """the value of the votes, prob, fit or count by the kind of the table, one row per grain"""
    return getattr(self, self.kind)

  def __len__(self):
    return len(self.parentId)

  def __repr__(self):
    return aligned(f"voteTable ({len(self)} grains)\n  parentId: {self.parentId.shape}\n  {self.kind}: {self.value.shape}")


def accumVotes(grainInd, parentId, fit, n, numFit=2, threshold=None, tol=None, minFit=np.inf, maxFit=-np.inf,
               weights=None, bestFit=False, count=False):
  """the votes of many (grain, variant, fit) rows turned into a table of n grains: each row is a
  vote with its numFit best variants, the best first. The fits become probabilities, summed
  per grain and variant and scaled by a high quantile of them and by the share of the grain;
  with bestFit the best fit per variant is kept, with count the number of fits below the
  threshold"""
  fit = np.asarray(fit, dtype=float).reshape(-1, numFit)
  grainInd = np.asarray(grainInd, dtype=np.int64).reshape(-1, numFit)
  parentId = np.asarray(parentId, dtype=np.int64).reshape(-1, numFit)
  keep = fit[:, 0] < minFit
  if numFit > 1:
    keep &= fit[:, 1] > maxFit
  fit, grainInd, parentId = fit[keep], grainInd[keep], parentId[keep]
  threshold = min(2 * degree, minFit) if threshold is None else threshold
  numV = int(parentId.max(initial=0))
  g, v, f = grainInd.reshape(-1), parentId.reshape(-1) - 1, fit.reshape(-1)
  if bestFit:
    M = np.full((n, numV), np.inf)
    np.minimum.at(M, (g, v), f)
    M[np.isinf(M)] = np.nan
    order = np.argsort(np.where(np.isnan(M), np.inf, M), axis=1, kind="stable")
    parentId, fit = _padVotes(order + 1, np.take_along_axis(M, order, axis=1), numFit)
    return voteTable(parentId, fit=fit)
  if count:
    M = np.zeros((n, numV))
    np.add.at(M, (g, v), f < threshold)
    M[M.sum(axis=1) == 0] = np.nan
  else:
    tol = threshold if tol is None else tol
    prob = probabilityOf(f, threshold, tol)
    prob[prob < 1e-2] = 0
    if weights is not None and np.size(weights) > 1:
      prob = prob * np.asarray(weights, dtype=float).reshape(-1, numFit)[keep].reshape(-1)
    M = np.zeros((n, numV))
    np.add.at(M, (g, v), prob)
    # the share of every variant in the grain's votes, scaled by how strong its votes are
    with np.errstate(invalid="ignore", divide="ignore"):
      M = _groupQuantile(g, v, prob, n, numV, 1 - 1 / (4 * numFit)) * M / M.sum(axis=1, keepdims=True) if numV else M
  # descending, a row without votes first as MATLAB sorts nan
  order = np.argsort(np.where(np.isnan(M), -np.inf, -M), axis=1, kind="stable")[:, :numFit]
  value = np.take_along_axis(M, order, axis=1)
  parentId, value = _padVotes(order + 1, value, numFit)
  return voteTable(parentId, **{"count" if count else "prob": value})


# a table of fewer than numFit variants, none when no vote passed minFit and maxFit, filled up to
# numFit columns of no variant (id 0) and no value, which no criterion of calcParentFromVote passes
def _padVotes(parentId, value, numFit):
  k = numFit - parentId.shape[1]
  if k <= 0:
    return parentId, value
  n = len(parentId)
  return np.hstack([parentId, np.zeros((n, k), dtype=parentId.dtype)]), np.hstack([value, np.full((n, k), np.nan)])


# the quantile of the values of every (grain, variant) cell, 0 where a cell has none
def _groupQuantile(g, v, x, n, numV, q):
  out = np.zeros((n, numV))
  if len(x) == 0:
    return out
  key = g * numV + v
  order = np.lexsort((x, key))
  key, x = key[order], x[order]
  starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
  sizes = np.diff(np.r_[starts, len(key)])
  pos = np.maximum(1, np.floor(sizes * q + 0.5).astype(np.int64))
  out.reshape(-1)[key[starts]] = x[starts + pos - 1]
  return out


def majorityVote(id, votes, maxId=None, strict=False, weights=None):
  """the most frequent vote cast for every id; MATLAB's majorityVote

    vote, numVotes = majorityVote([1, 1, 1, 1, 1, 2, 3, 2, 3, 2, 3], [5, 9, 5, 9, 9, 4, 1, 4, 2, 4, 1])  # the most frequent vote per id
    vote, numVotes = majorityVote([1, 1, 2, 2, 3], [5, 5, 4, 7, 1], strict=True)  # a vote only where all agree
    vote, numVotes = majorityVote([1, 1, 2], [5, 9, 4], maxId=5)   # a row for every id up to 5
    vote, numVotes = majorityVote([1, 1, 1], [5, 9, 9], weights=[3, 1, 1])  # the vote of the largest summed weight

  The votes are grouped by their id, and every id from 1 to maxId gets one row, at position
  id - 1: the vote cast most often for it, the smallest of a tie, and NaN for an id without
  votes. Votes that are NaN and ids below 1 are left out.

  Input
    id - integer array, the id every vote is cast for, from 1
    votes - array of the same length, the votes, positive integers
  Output
    vote - array of maxId entries, the winning vote of every id, NaN where there is none
    numVotes - array of maxId entries, how often the winning vote was cast; with weights the
        lead of the winning vote, its weight over the sum of its weight and the second's
  Keywords
    maxId - the number of rows, by default the largest id
    strict - give a vote only to an id whose votes all agree, NaN to the others
    weights - array of the length of votes, a weight for every vote; the vote of the largest
        summed weight wins
  See also
    parentGrainReconstructor.calcParentFromVote, parentGrainReconstructor.calcGBVotes"""
  id = np.asarray(id, dtype=np.int64).reshape(-1)
  votes = np.asarray(votes, dtype=float).reshape(-1)
  maxId = int(id.max(initial=0)) if maxId is None else int(maxId)
  vote = np.full(maxId, np.nan)
  numVotes = np.zeros(maxId)
  ok = np.isfinite(votes) & (id > 0)
  id, votes = id[ok], votes[ok]
  if weights is not None:
    w = np.asarray(weights, dtype=float).reshape(-1)[ok]
    from scipy.sparse import coo_matrix
    W = coo_matrix((w, (id - 1, votes.astype(np.int64) - 1)), shape=(maxId, int(votes.max(initial=1)))).toarray()
    best = np.argmax(W, axis=1)
    top = W[np.arange(maxId), best]
    W[np.arange(maxId), best] = 0
    has = top > 0
    vote[has] = best[has] + 1
    numVotes = top / (1e-3 + W.max(axis=1, initial=0) + top)
    return vote, numVotes
  if strict:
    order = np.argsort(id, kind="stable")
    id, votes = id[order], votes[order]
    starts = np.flatnonzero(np.r_[True, id[1:] != id[:-1]])
    first = np.repeat(votes[starts], np.diff(np.r_[starts, len(id)]))
    same = votes == first
    equal = np.logical_and.reduceat(same, starts) if len(starts) else np.zeros(0, dtype=bool)
    g = id[starts] - 1
    vote[g[equal]] = votes[starts][equal]
    numVotes[g[equal]] = np.add.reduceat(same.astype(float), starts)[equal]
    return vote, numVotes
  # the most frequent vote of every id, the smallest of a tie
  pairs, counts = np.unique(np.stack([id, votes], axis=1), axis=0, return_counts=True)
  order = np.lexsort((pairs[:, 1], -counts, pairs[:, 0]))
  pairs, counts = pairs[order], counts[order]
  starts = np.flatnonzero(np.r_[True, pairs[1:, 0] != pairs[:-1, 0]])
  g = pairs[starts, 0].astype(np.int64) - 1
  vote[g] = pairs[starts, 1]
  numVotes[g] = counts[starts]
  return vote, numVotes


# the row of every boundary segment, by its grain ids gid, in a list of grain pairs, -1 where
# it is none of them
def _segmentRows(pairs, gid):
  M = int(max(gid.max(initial=0), pairs.max(initial=0))) + 1
  key, skey = _pairKey(pairs, M), _pairKey(gid, M)
  order = np.argsort(key)
  at = np.minimum(np.searchsorted(key[order], skey), len(key) - 1)
  return np.where(key[order][at] == skey, order[at], -1)


# the map with every pixel of a grain moved to the equivalent next to the grain's mean
def _projectToGrains(ebsd, grains):
  out = ebsd.copy()
  gid = np.asarray(out.grainId).reshape(-1)
  ph = out.phaseId.reshape(-1)
  rot = out.rotations.data.reshape(-1, 4)
  gp = _at(grains._positions(), gid)
  for p in out.CSList.indexedPhaseId:
    sel = (gp >= 0) & (ph == p)
    sel[sel] &= grains.phaseId[gp[sel]] == p
    if sel.any():
      o = _ori(rot[sel], out.CSList[int(p)], out.frame)
      rot[sel] = o.projectIntoFundamentalRegion(grains.meanRotation.data[gp[sel]]).data.reshape(-1, 4)
  return out


# a phase list with the frames appended that it does not hold yet
def _withFrames(pl, frames):
  missing = [f for f in frames if f is not None and not any(f is g for g in pl.frames)]
  return pl if not missing else phaseList(list(pl.frames) + missing)


def _registerGrains(grains, pl):
  if grains.CSList is pl:
    return grains
  out = grains.copy()
  out.CSList = pl
  for name in ("boundary", "innerBoundary"):
    gB = getattr(out, name)
    if gB is not None:
      setattr(out, name, gB._with(CSList=pl))
  return out


class parentGrainReconstructor:
  """the state of a parent grain reconstruction: the map and the grains it starts from, the
  orientation relationship and the grains after every step taken; MATLAB's
  parentGrainReconstructor

    job = parentGrainReconstructor(ebsd, grains)       # the parent and child phases guessed
    job = parentGrainReconstructor(ebsd, grains, orientation.KurdjumovSachs(csParent, csChild))  # with an initial orientation relationship
    job = parentGrainReconstructor(ebsd, grains, p2c, useBoundaryOrientations=True)  # votes from the orientations along the boundaries
    job = parentGrainReconstructor(ebsd)               # the grains segmented at 3 degree

  The job keeps the map and the grains as given, ebsdPrior and grainsPrior, the grains at
  the current stage, grains, with parentGrains and childGrains, and for every prior grain the
  id of the current grain it went into, mergeId. The steps of a reconstruction are its
  methods: calcParent2Child refines the orientation relationship p2c, calcGBVotes,
  calcTPVotes or calcGraph with clusterVariantGraph collect votes, calcParentFromVote and
  calcParentFromGraph turn child grains into parent grains, mergeSimilar and
  mergeInclusions merge them, and calcParentEBSD gives the reconstructed map, with
  variantId and packetId.

  The phases are guessed from the map: the most frequent indexed phase is the child and
  the rarest the parent, which job.csParent and job.csChild correct, or setting job.p2c,
  whose frames then name both. The parent frame is added to the phase lists of the job's
  copies when the map carries only the child. The map must carry the grain ids of grains,
  which calcGrains writes into it.

  Input
    ebsd - EBSD, the map of the measured, mostly child, phase, with the grain ids of grains
    grains - grain2d, the child grains of the map; when not given the map is segmented at 3
        degree, on a copy
    p2c - orientation, the initial guess of the parent to child orientation relationship
        (MATLAB's p2c0)
  Output
    job - parentGrainReconstructor
  Keywords
    useBoundaryOrientations - a pair of grains votes with the mean orientations of their
        pixels along the boundary they share rather than with the grain means
    reportFit - show the quantiles of the misfit of the boundaries to p2c when the job is
        displayed
  References
    R. Hielscher, T. Nyyssönen, F. Niessen, A. A. Gazder, The variant graph approach to
        improved parent grain reconstruction, Materialia 22 (2022) 101399
    F. Niessen, T. Nyyssönen, A. A. Gazder, R. Hielscher, Parent grain reconstruction from
        partially or fully transformed microstructures in MTEX, J. Appl. Cryst. 55 (2022)
        180-194
  See also
    parentGrainReconstructor.calcParent2Child, parentGrainReconstructor.calcGBVotes,
    parentGrainReconstructor.calcParentFromVote, calcParent2Child, MaParentGrainReconstruction,
    TiBetaReconstruction, GrainGraphBasedReconstruction"""

  def __init__(self, ebsd, grains=None, p2c=None, useBoundaryOrientations=False, reportFit=True):
    if grains is None:
      from .calcgrains import calcGrains
      ebsd = ebsd.copy()
      grains = calcGrains(ebsd, threshold=3 * degree)
    if "grainId" not in ebsd.prop:
      raise ValueError("the map carries no grain id; reconstruct the grains with calcGrains(ebsd) first")
    self.useBoundaryOrientations = useBoundaryOrientations
    self.reportFit = reportFit
    self.ebsdPrior = _projectToGrains(ebsd, grains)
    self.grainsPrior = grains.copy()
    self.grains = grains.copy()
    self.mergeId = grains.id.astype(np.int64).copy()
    self.votes = None
    self.graph = None
    self._p2c = None
    self._variantMap = None
    # the most frequent indexed phase is the child, the rarest the parent
    ph = ebsd.phaseId[~ebsd.isPadding].reshape(-1)
    idx = ebsd.CSList.indexedPhaseId
    counts = np.bincount(ph, minlength=len(ebsd.CSList))[idx]
    self._csChild = ebsd.CSList[int(idx[np.argmax(counts)])] if len(idx) else None
    self._csParent = ebsd.CSList[int(idx[np.argmin(counts)])] if len(idx) > 1 else None
    if p2c is not None:
      self.p2c = p2c

  # ------------------------------------------------------------------ the relationship and the phases

  @property
  def p2c(self):
    """the parent to child orientation relationship, refined along the reconstruction"""
    return self._p2c

  @p2c.setter
  def p2c(self, p2c):
    """sets the orientation relationship, its frames added to the phase lists of the map and the
    grains, the variant order reset"""
    self._p2c = p2c
    self._variantMap = None
    self._register(p2c.frameA, p2c.frameB)

  # the frames added to the phase lists of the map and of both grain sets
  def _register(self, *frames):
    pl = _withFrames(self.ebsdPrior.CSList, frames)
    if pl is not self.ebsdPrior.CSList:
      self.ebsdPrior.CSList = pl
    self.grainsPrior = _registerGrains(self.grainsPrior, _withFrames(self.grainsPrior.CSList, frames))
    self.grains = _registerGrains(self.grains, _withFrames(self.grains.CSList, frames))

  @property
  def csParent(self):
    """the frame of the parent phase, that of the orientation relationship once it is set"""
    return self._p2c.frameA if self._p2c is not None else self._csParent

  @csParent.setter
  def csParent(self, CS):
    """sets the parent phase, the frame added to the phase lists"""
    self._csParent = CS
    self._register(CS)

  @property
  def csChild(self):
    """the frame of the child phase, that of the orientation relationship once it is set"""
    return self._p2c.frameB if self._p2c is not None else self._csChild

  @csChild.setter
  def csChild(self, CS):
    """sets the child phase, the frame added to the phase lists"""
    self._csChild = CS
    self._register(CS)

  @property
  def parentPhaseId(self):
    """the phase id of the parent phase"""
    return self.grains.CSList.phaseId(self.csParent)

  @property
  def childPhaseId(self):
    """the phase id of the child phase"""
    return self.grains.CSList.phaseId(self.csChild)

  @property
  def variantMap(self):
    """the order the variants are numbered in, kept across a refit of the relationship"""
    n = self.p2c.variants().size
    return np.arange(1, n + 1) if self._variantMap is None or len(self._variantMap) != n else self._variantMap

  @variantMap.setter
  def variantMap(self, vMap):
    """sets the order the variants are numbered in"""
    self._variantMap = np.asarray(vMap, dtype=np.int64)

  # variant ids of the unmapped list as the ids of the mapped one
  def _mapVariants(self, vId):
    vm = self.variantMap
    back = np.zeros(len(vm) + 1)
    back[vm] = np.arange(1, len(vm) + 1)
    vId = np.asarray(vId, dtype=float)
    ok = np.isfinite(vId)
    out = np.full(vId.shape, np.nan)
    out[ok] = back[vId[ok].astype(np.int64)]
    return out

  @property
  def c2c(self):
    """the child to child misorientations of the relationship, the identity included"""
    V = self.p2c.variants().data.reshape(-1, 4)
    return _ori(qmul(self.p2c.data.reshape(1, 4), qconj(V)), self.csChild, self.csChild)

  # ------------------------------------------------------------------ the state

  def _pos(self, ids):
    return self.grains._positions()[np.asarray(ids, dtype=np.int64)]

  @property
  def numChildren(self):
    """the number of prior child grains merged into each grain of the current stage (MATLAB's
    parentGrainReconstructor.numChilds)"""
    return np.bincount(self._pos(self.mergeId), minlength=self.grains.size)

  @property
  def isMerged(self):
    """which prior grains have been merged with others into a parent grain"""
    return self.numChildren[self._pos(self.mergeId)] > 1

  @property
  def isTransformed(self):
    """which prior child grains have been reconstructed into a parent grain"""
    return (self.grainsPrior.phaseId == self.childPhaseId) & (self.grains.phaseId[self._pos(self.mergeId)] == self.parentPhaseId)

  @property
  def isChild(self):
    """which grains of the current stage are child grains"""
    return self.grains.phaseId == self.childPhaseId

  @property
  def isParent(self):
    """which grains of the current stage are parent grains"""
    return self.grains.phaseId == self.parentPhaseId

  @property
  def parentGrains(self):
    """the parent grains at the current stage of the reconstruction"""
    return self.grains[self.isParent]

  @property
  def childGrains(self):
    """the remaining child grains at the current stage of the reconstruction"""
    return self.grains[self.isChild]

  @property
  def transformedGrains(self):
    """the prior child grains that have been reconstructed"""
    return self.grainsPrior[self.isTransformed]

  @property
  def ebsd(self):
    """the map at the current stage of the reconstruction, calcParentEBSD"""
    return self.calcParentEBSD()

  @property
  def parentEBSD(self):
    """the parent phase of the map at the current stage of the reconstruction"""
    return self.ebsd[self.csParent]

  def _priorProp(self, name):
    p = self.grainsPrior.prop.get(name)
    return np.full(self.grainsPrior.size, np.nan) if p is None else p

  @property
  def variantId(self):
    """the reconstructed variant id of every prior grain, NaN where none is computed"""
    return self._priorProp("variantId")

  @property
  def packetId(self):
    """the reconstructed packet id of every prior grain, NaN where none is computed"""
    return self._priorProp("packetId")

  @property
  def bainId(self):
    """the reconstructed Bain group id of every prior grain, NaN where none is computed"""
    return self._priorProp("bainId")

  @property
  def parentId(self):
    """the parent variant every transformed prior grain was reconstructed with"""
    out = np.full(self.grainsPrior.size, np.nan)
    tr = np.flatnonzero(self.isTransformed)
    if len(tr):
      ids, _ = calcParent(self._meanOri(self.grainsPrior, tr, self.csChild),
                          self._meanOri(self.grains, self._pos(self.mergeId[tr]), self.csParent), self.p2c, id=True)
      out[tr] = ids
    return out

  @property
  def hasVariantGraph(self):
    """whether the graph is a variant graph, one node per grain and parent variant"""
    return self.graph is not None and self.graph.shape[0] > 1.5 * self.grains.size

  @property
  def hasGraph(self):
    """whether the graph is a grain graph, one node per grain"""
    return self.graph is not None and self.graph.shape[0] < 1.5 * self.grains.size

  def _meanOri(self, g, idx, frame):
    return _ori(g.meanRotation.data.reshape(-1, 4)[np.asarray(idx, dtype=np.int64)], frame, g.frame)

  # the parent variants of the relationship as quaternions
  def _PV(self):
    return self.p2c.variants(parent=True).data.reshape(-1, 4)

  # the grains after a change of mean orientations and phases, the boundary phases updated
  def _setGrains(self, idx, q, phase):
    g = self.grains.copy()
    g.meanRotation = g.meanRotation._new(g.meanRotation.data.copy())
    g.phaseId = g.phaseId.copy()
    g.meanRotation.data.reshape(-1, 4)[idx] = q
    g.phaseId[idx] = phase
    self.grains = g.update()

  # ------------------------------------------------------------------ the pairs of grains

  # the neighbouring grains of the kinds c2c, p2c (the parent first) or p2p, with the
  # orientations they vote with: their means, or the means of their pixels along the
  # boundary they share, as useBoundaryOrientations or boundaryOrientations says; minDelta
  # drops pairs of orientations too close to tell a parent
  def _pairs(self, kind, minDelta=None, boundaryOrientations=None):
    g = self.grains
    P, C = self.parentPhaseId, self.childPhaseId
    pairs = g.neighbors()
    pos = g._positions()
    a, b = pos[pairs[:, 0]], pos[pairs[:, 1]]
    pa, pb = g.phaseId[a], g.phaseId[b]
    if kind == "c2c":
      keep, frames = (pa == C) & (pb == C), (self.csChild, self.csChild)
    elif kind == "p2p":
      keep, frames = (pa == P) & (pb == P), (self.csParent, self.csParent)
    else:
      keep, frames = ((pa == P) & (pb == C)) | ((pa == C) & (pb == P)), (self.csParent, self.csChild)
      flip = keep & (pa == C)
      pairs[flip] = pairs[flip][:, ::-1]
    pairs = pairs[keep]
    a, b = pos[pairs[:, 0]], pos[pairs[:, 1]]
    q1 = g.meanRotation.data.reshape(-1, 4)[a]
    q2 = g.meanRotation.data.reshape(-1, 4)[b]
    useBoundary = self.useBoundaryOrientations if boundaryOrientations is None else boundaryOrientations
    if useBoundary and len(pairs):
      q1, q2 = self._boundaryMeans(pairs, q1, q2)
    o1, o2 = _ori(q1, frames[0], g.frame), _ori(q2, frames[1], g.frame)
    if minDelta is not None and kind != "p2c" and len(pairs):
      far = np.asarray(o1.angle(o2)).reshape(-1) >= minDelta
      pairs, o1, o2 = pairs[far], o1[far], o2[far]
    return pairs, o1, o2

  # the mean of the pixels of each side along the boundary of every pair, each pixel moved
  # next to the mean of its grain first; a child pixel of a grain that is a parent now
  # counts with the parent it gives nearest that mean
  def _boundaryMeans(self, pairs, q1, q2):
    g, e = self.grains, self.ebsdPrior
    gB = g.boundary
    row = _segmentRows(pairs, gB.grainId)
    ids = np.asarray(e.id).reshape(-1)
    idOrder = np.argsort(ids)
    rot = e.rotations.data.reshape(-1, 4)
    ph = e.phaseId.reshape(-1)
    pos = g._positions()
    acc = np.zeros((len(pairs), 2, 4))
    for col in (0, 1):
      seg = np.flatnonzero((row >= 0) & (gB.ebsdId[:, col] >= 0))
      r = row[seg]
      grain = gB.grainId[seg, col]
      side = np.where(grain == pairs[r, 0], 0, 1)
      pix = idOrder[np.searchsorted(ids[idOrder], gB.ebsdId[seg, col])]
      ref = np.where(side[:, None] == 0, q1[r], q2[r])
      gph = g.phaseId[pos[grain]]
      q = rot[pix].copy()
      ok = ph[pix] == gph
      conv = (ph[pix] == self.childPhaseId) & (gph == self.parentPhaseId)
      if conv.any():
        q[conv] = self._parentOf(q[conv], ref[conv])[0]
        ok |= conv
      for p in np.unique(gph[ok]):
        s = ok & (gph == p)
        q[s] = _ori(q[s], g.CSList[int(p)], g.frame).projectIntoFundamentalRegion(ref[s]).data.reshape(-1, 4)
      q = np.where(np.sum(q * ref, axis=1, keepdims=True) < 0, -q, q)
      np.add.at(acc, (r[ok], side[ok]), q[ok])
    out = [q1.copy(), q2.copy()]
    for side in (0, 1):
      n = np.linalg.norm(acc[:, side], axis=1)
      out[side][n > 0] = acc[n > 0, side] / n[n > 0, None]
    return out[0], out[1]

  # the parent a child orientation gives nearest a reference parent, with the variant and
  # its misfit
  def _parentOf(self, childQ, parentQ):
    child = _ori(childQ, self.csChild, self.grains.frame)
    ids, fit = calcParent(child, _ori(parentQ, self.csParent, self.grains.frame), self.p2c, id=True)
    return qmul(np.asarray(childQ).reshape(-1, 4), self._PV()[ids - 1]), ids, fit

  # ------------------------------------------------------------------ the fit of the relationship

  def calcGBFit(self, p2c=False, c2c=False):
    """the misfit of the boundaries between neighbouring grains to the orientation relationship
    job.p2c, and the ids of the grains on either side; MATLAB's
    parentGrainReconstructor.calcGBFit

      fit, pairs = job.calcGBFit()               # every parent to child and child to child boundary
      fit, p2cPairs = job.calcGBFit(p2c=True)    # the parent to child boundaries alone
      fit, c2cPairs = job.calcGBFit(c2c=True)    # the child to child boundaries alone

    For a pair of child grains the misfit is the angle between their misorientation and the
    nearest child to child misorientation job.p2c admits; for a parent and a child grain the
    angle between their misorientation and job.p2c. With neither keyword both kinds are
    returned, the parent to child pairs first.

    Input
      job - parentGrainReconstructor
    Output
      fit - array of the misfit angles, in radians
      pairs - (n, 2) int array, the ids of the two grains of every boundary
    Keywords
      p2c - only the boundaries between a parent and a child grain
      c2c - only the boundaries between two child grains
    See also
      parentGrainReconstructor.calcParent2Child, parentGrainReconstructor.calcGBVotes,
      MaParentGrainReconstruction"""
    noOpt = not (p2c or c2c)
    fits, pairs = [], []
    if noOpt or p2c:
      pr, oP, oC = self._pairs("p2c")
      if len(pr):
        fits.append(np.asarray((oC.inv() * oP).angle(self.p2c)).reshape(-1))
        pairs.append(pr)
    if noOpt or c2c:
      pr, o1, o2 = self._pairs("c2c")
      if len(pr):
        mori = o1.inv() * o2
        fits.append(np.min(mori.reshape(-1, 1).angle(self.c2c.reshape(1, -1)), axis=1))
        pairs.append(pr)
    if not fits:
      return np.zeros(0), np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(fits), np.concatenate(pairs)

  def calcParent2Child(self, p2c0=None, p2c=False, c2c=False, threshold=None, quantile=None, useP2C=True, **fitOptions):
    """the parent to child orientation relationship job.p2c refined on the boundaries of the
    map; MATLAB's calcParent2Child of the job

      job.calcParent2Child()                         # fitted to the parent to child and the child to child boundaries
      job.calcParent2Child(c2c=True, quantile=0.8)   # to the child to child boundaries alone
      job.calcParent2Child(p2c=True, threshold=3*degree)  # to the parent to child boundaries alone
      job.calcParent2Child(orientation.NishiyamaWassermann(job.csParent, job.csChild))  # from another start

    Two fits are made and combined, weighted by their numbers of boundaries:

    - parent to child: where the parent phase covers more than one percent of the map, the
    misorientations of the parent to child boundaries closer to the current relationship than
    threshold and than the quantile of all of them are averaged, three times.
    - child to child: where the relationship is more than 5 degree from the identity, or with
    c2c, the child to child misorientations above 5 degree are fitted by calcParent2Child of
    the misorientations, the iteration of Nyyssönen et al.

    Further keywords go to calcParent2Child of the misorientations: local, maxIteration,
    searchResolution, verbose, ...

    job.p2c is replaced. The variants keep their numbers through job.variantMap, the variant of
    the old relationship that every new one is closest to.

    Input
      job - parentGrainReconstructor, with job.p2c or p2c0 set
      p2c0 - orientation, the start of the fit instead of job.p2c
    Keywords
      p2c - fit to the parent to child boundaries alone
      c2c - fit to the child to child boundaries alone
      threshold - the misorientations further than this from the start are left out, 5
          degree for the parent to child boundaries
      quantile - the share of the misorientations closest to the start that is fitted, 0.5
          for the parent to child boundaries
      useP2C - combine the fit to the child to child boundaries with that to the parent to
          child ones; False keeps the first alone
    References
      T. Nyyssönen, M. Isakov, P. Peura, V.-T. Kuokkala, Crystallography, morphology, and
          martensite transformation of prior austenite in intercritically annealed
          high-aluminum steel, Metall. Mater. Trans. A 49 (2018), doi:10.1007/s11661-018-4904-9
    See also
      calcParent2Child, parentGrainReconstructor.calcGBFit, orientation.KurdjumovSachs,
      OrientationRelationshipFit, MaParentGrainReconstruction"""
    noOpt = not (p2c or c2c)
    guess = p2c0 if p2c0 is not None else self.p2c
    thr = 5 * degree if threshold is None else threshold
    quant = 0.5 if quantile is None else quantile
    p2cData, nP2C = None, 0
    ph = self.ebsdPrior.phaseId.reshape(-1)
    if np.count_nonzero(ph == self.parentPhaseId) > 0.01 * ph.size and (noOpt or p2c):
      pr, oP, oC = self._pairs("p2c", boundaryOrientations=False)
      mori = oC.inv() * oP
      for _ in range(3):
        omega = np.asarray(mori.angle(guess)).reshape(-1)
        ind = omega < min(thr, _quantile(omega, quant))
        guess = mori[ind].mean()
      p2cData, nP2C = guess, len(pr)
    if (noOpt and guess.angle(symmetry=True) > 5 * degree) or c2c:
      pr, o1, o2 = self._pairs("c2c")
      mori = (o1.inv() * o2).flatten()
      mori = mori[np.asarray(mori.angle()).reshape(-1) >= 5 * degree]
      start = p2c0 if p2c0 is not None else self.p2c
      kw = dict(fitOptions)
      if threshold is not None:
        kw["threshold"] = threshold
      if quantile is not None:
        kw["quantile"] = quantile
      new = calcParent2Child(mori, start, **kw)
      if p2cData is not None and useP2C:
        both = _ori(np.stack([new.data.reshape(4), p2cData.data.reshape(4)]), new.frameA, new.frameB)
        new = both.mean(weights=[len(pr), nP2C])
    else:
      new = guess
    old = self.p2c
    self._p2c = new.reshape(()) if new.size == 1 else new
    self._variantMap = None
    if old is not None:
      Vn, Vo = new.variants(), old.variants()
      if Vn.size == Vo.size:
        # the child variants compared up to the child symmetry alone
        a = _ori(Vn.data, new.frameA.stripSym(), new.frameB).reshape(-1, 1)
        b = _ori(Vo.data, new.frameA.stripSym(), new.frameB).reshape(1, -1)
        self._variantMap = np.argmin(a.angle(b), axis=1) + 1

  # ------------------------------------------------------------------ the votes

  def calcGBVotes(self, ind=None, p2c=False, c2c=False, reconsiderAll=False, numFit=2, threshold=None, tol=None,
                  minFit=np.inf, maxFit=-np.inf, bestFit=False, count=False, curvatureFactor=None):
    """the votes the grain boundaries cast for the parent orientation of their child grains,
    written into job.votes; MATLAB's calcGBVotes

      job.calcGBVotes()                              # from the child to child and the parent to child boundaries
      job.calcGBVotes(p2c=True, threshold=2.5*degree, tol=1.5*degree)  # from the parent to child boundaries alone
      job.calcGBVotes(p2c=True, reconsiderAll=True, threshold=4*degree, tol=1.5*degree)  # every measured child grain anew, the growth step
      job.calcGBVotes(numFit=1, bestFit=True)        # the best fit per variant instead of a probability
      job.calcGBVotes(count=True)                    # the number of fits below the threshold

    A pair of neighbouring child grains votes for the numFit pairs of variants that let both
    come from one parent, a child grain next to a parent grain for the numFit variants of
    that parent that fit it. A fit becomes a probability that it still means one parent: 1
    below threshold - tol/2, 0 above threshold + tol/2, a cumulative Gaussian between. The
    probabilities are summed per grain and variant into job.votes, a table whose rows are
    job.grains, the best variant first, which calcParentFromVote reads.

    With reconsiderAll every measured child grain votes afresh, from its own orientation,
    for the variant that fits its parent neighbours: the growth step of a reconstruction,
    which takes back an earlier wrong decision. It needs the grains as measured, before
    mergeSimilar or after revert.

    Input
      job - parentGrainReconstructor, with job.p2c set
      ind - with reconsiderAll: the positions of the prior grains to vote for; the table
          of their votes is returned and job.votes left as it was
    Output
      votes - with ind: voteTable of the votes of those grains
    Keywords
      p2c - only the parent to child boundaries vote
      c2c - only the child to child boundaries vote
      reconsiderAll - every measured child grain votes for the variant that fits its parent
          neighbours, the decided ones too
      numFit - the number of best fitting variants every boundary votes for
      threshold - the misfit angle at which the probability is one half, by default 2
          degree or minFit when smaller
      tol - the width of the angles over which the probability falls from 1 to 0, by default
          the threshold (MATLAB's tolerance)
      minFit - a boundary votes only when its best fit is below this angle
      maxFit - a boundary votes only when its second best fit is above this angle
      bestFit - the vote of a grain is the best fit per variant, an angle, instead of a
          probability
      count - the vote of a grain is the number of fits per variant below the threshold
      curvatureFactor - with reconsiderAll: weigh a boundary by 1 plus this factor times its
          normalised curvature, so that curved boundaries count more
    See also
      parentGrainReconstructor.calcParentFromVote, parentGrainReconstructor.calcTPVotes,
      parentGrainReconstructor.calcGBFit, MaParentGrainReconstruction,
      TriplePointBasedReconstruction"""
    opts = dict(threshold=threshold, tol=tol, minFit=minFit, maxFit=maxFit, bestFit=bestFit, count=count)
    if reconsiderAll and self.grains.size != self.grainsPrior.size:
      raise ValueError("reconsiderAll votes for the grains as measured: call it before mergeSimilar, or after revert")
    if reconsiderAll:
      votes = self._reconsiderVotes(ind, curvatureFactor, opts)
      if ind is not None:
        return votes
      self.votes = votes
      return None
    noOpt = not (p2c or c2c)
    g = self.grains
    pos = g._positions()
    rows, vids, fits = [], [], []
    if len(self.parentGrains) and (noOpt or p2c):
      pr, oP, oC = self._pairs("p2c")
      if len(pr):
        ids, fit = calcParent(oC, oP, self.p2c, numFit=numFit, id=True)
        rows.append(np.repeat(pos[pr[:, 1]][:, None], numFit, axis=1).reshape(-1, numFit))
        vids.append(np.asarray(ids).reshape(-1, numFit))
        fits.append(np.asarray(fit).reshape(-1, numFit))
    if noOpt or c2c:
      pr, o1, o2 = self._pairs("c2c", minDelta=2 * degree)
      if len(pr):
        both = _ori(np.stack([o1.data.reshape(-1, 4), o2.data.reshape(-1, 4)], axis=1).reshape(-1, 4), self.csChild, g.frame).reshape(-1, 2)
        ids, fit = calcParent(both, self.p2c, numFit=numFit, id=True)
        ids = np.asarray(ids).reshape(-1, numFit, 2)
        fit = np.asarray(fit).reshape(-1, numFit)
        for side in (0, 1):
          rows.append(np.repeat(pos[pr[:, side]][:, None], numFit, axis=1))
          vids.append(ids[:, :, side])
          fits.append(fit)
    if rows:
      grain, vid, fit = np.concatenate(rows), np.concatenate(vids), np.concatenate(fits)
    else:
      grain = vid = np.zeros((0, numFit), dtype=np.int64)
      fit = np.zeros((0, numFit))
    self.votes = accumVotes(grain, vid, fit, g.size, numFit=numFit, **opts)

  # every prior child grain against its parent neighbours, every variant of its measured
  # orientation one column of the vote
  def _reconsiderVotes(self, ind, curvatureFactor, opts):
    g = self.grains
    P, C = self.parentPhaseId, self.childPhaseId
    pos = g._positions()
    sel = np.flatnonzero(self.grainsPrior.phaseId == C) if ind is None else np.asarray(ind, dtype=np.int64).reshape(-1)
    PV = self._PV()
    numV = len(PV)
    pairs = g.neighbors(full=True)
    both = np.concatenate([pairs, pairs[:, ::-1]])
    cur = self.mergeId[sel]
    hit = np.isin(both[:, 0], cur)
    both = both[hit]
    r = _idPositions(cur, both[:, 0])
    nb = pos[both[:, 1]]
    isP = g.phaseId[nb] == P
    fit = np.full((len(both), numV), np.nan)
    if isP.any():
      child = self.grainsPrior.meanRotation.data.reshape(-1, 4)[sel[r[isP]]]
      cand = qmul(child[:, None, :], PV[None, :, :])
      oP = _ori(g.meanRotation.data.reshape(-1, 4)[nb[isP]], self.csParent, g.frame)
      oV = _ori(cand.reshape(-1, 4), self.csParent, g.frame).reshape(-1, numV)
      fit[isP] = np.asarray(oV.angle(oP.reshape(-1, 1))).reshape(-1, numV)
    w = None
    if curvatureFactor is not None and len(both):
      w = np.repeat(self._boundaryWeights(both, curvatureFactor)[:, None], numV, axis=1)
    # the grain a row votes for is its place in the whole list of grains
    grain = np.repeat(sel[r][:, None], numV, axis=1) if ind is None else np.repeat(r[:, None], numV, axis=1)
    n = self.grains.size if ind is None else len(sel)
    vid = np.repeat(np.arange(1, numV + 1)[None, :], len(both), axis=0)
    return accumVotes(grain, vid, fit, n, numFit=numV, weights=w, **opts)

  # the weight of a boundary by its curvature: 1 plus the upper quartile of the capped and
  # normalised curvature of its segments
  def _boundaryWeights(self, pairs, factor):
    gB = self.grains.boundary
    kappa = np.abs(np.asarray(gB.curvature(), dtype=float))
    kappa = np.minimum(kappa, _quantile(kappa, 0.9))
    kappa = factor * kappa / np.nanmax(kappa)
    row = _segmentRows(pairs, gB.grainId)
    ok = (row >= 0) & ~np.isnan(kappa)
    return 1 + _groupQuantile(row[ok], np.zeros(np.count_nonzero(ok), dtype=np.int64), kappa[ok], len(pairs), 1, 0.75)[:, 0]

  def calcTPVotes(self, numFit=2, threshold=None, tol=None, minFit=np.inf, maxFit=-np.inf, bestFit=False, count=False):
    """the votes the triple points of three child grains cast for their parent orientation,
    written into job.votes; MATLAB's calcTPVotes

      job.calcTPVotes()                              # every triple point of three child grains votes
      job.calcTPVotes(minFit=2.5*degree, maxFit=5*degree)  # only the triple points of one clear parent
      job.calcTPVotes(numFit=1, bestFit=True)        # the best fit per variant instead of a probability

    At a triple point where three child grains meet, the parent orientation that fits all
    three, the mean of one variant of each, is computed for the numFit best fitting
    combinations, and each of the three grains votes for its variant, weighted by the fit
    as calcGBVotes weighs it. A triple point fits one parent clearly when its best fit is
    small and its second best large, which minFit and maxFit ask for.

    Input
      job - parentGrainReconstructor, with job.p2c set
    Keywords
      numFit - the number of best fitting parents every triple point votes for
      threshold - the misfit angle at which the probability is one half, by default 2
          degree or minFit when smaller; also the largest misfit a parent is computed for,
          5 degree when not given
      tol - the width of the angles over which the probability falls from 1 to 0, by default
          the threshold (MATLAB's tolerance)
      minFit - a triple point votes only when its best fit is below this angle
      maxFit - a triple point votes only when its second best fit is above this angle
      bestFit - the vote of a grain is the best fit per variant, an angle, instead of a
          probability
      count - the vote of a grain is the number of fits per variant below the threshold
    See also
      parentGrainReconstructor.calcParentFromVote, parentGrainReconstructor.calcGBVotes,
      TriplePointBasedReconstruction"""
    g = self.grains
    tP = g.triplePoints
    C = self.childPhaseId
    sel = np.all(tP.phaseId == C, axis=1)
    gid = tP.grainId[sel]
    pos = g._positions()
    n = len(gid)
    if n:
      ori = _ori(g.meanRotation.data.reshape(-1, 4)[pos[gid.reshape(-1)]], self.csChild, g.frame).reshape(n, 3)
      ids, fit = calcParent(ori, self.p2c, numFit=numFit, threshold=5 * degree if threshold is None else threshold, id=True)
      ids = np.asarray(ids).reshape(n, numFit, 3).transpose(0, 2, 1).reshape(-1, numFit)
      fit = np.repeat(np.asarray(fit).reshape(n, 1, numFit), 3, axis=1).reshape(-1, numFit)
      grain = np.repeat(pos[gid].reshape(-1, 1), numFit, axis=1)
    else:
      ids = grain = np.zeros((0, numFit), dtype=np.int64)
      fit = np.zeros((0, numFit))
    self.votes = accumVotes(grain, ids, fit, g.size, numFit=numFit, threshold=threshold, tol=tol, minFit=minFit,
                            maxFit=maxFit, bestFit=bestFit, count=count)

  def calcParentFromVote(self, minProb=0, minDelta=None, minFit=5 * degree, minAlpha=None, minCount=0, maxCount=None):
    """the child grains whose vote is clear turned into parent grains, the parent orientation
    of the variant they voted for; MATLAB's calcParentFromVote

      job.calcParentFromVote()                       # every grain with a vote
      job.calcParentFromVote(minProb=0.7)            # a probability of at least 0.7
      job.calcParentFromVote(minProb=0.5, minDelta=0.2)  # and ahead of the second by 0.2
      job.calcParentFromVote(minFit=2*degree, minDelta=1*degree)  # after votes of the best fit
      job.calcParentFromVote(minCount=3, maxCount=1)  # after counted votes

    The votes are those of the last calcGBVotes, calcTPVotes or clusterVariantGraph, and the
    criteria read them as they are: probabilities with minProb, minDelta and minAlpha, fits
    of bestFit with minFit, minDelta and minAlpha, counts with minCount and maxCount. A grain
    that meets them takes the parent phase and the parent orientation of the variant it
    voted for; a grain that is a parent already is reconstructed again from its measured
    orientation, a merged parent grain is not. The votes of the grains turned are cleared.

    MATLAB documents minVotes, which neither MATLAB nor this port reads; minAlpha compares
    fits, as MATLAB's code does, not probabilities, as its help says.

    Input
      job - parentGrainReconstructor, with job.votes
    Keywords
      minProb - with probabilities: the best must exceed this
      minDelta - the best must lead the second by more than this: in probability, or in
          angle with fits
      minFit - with fits: the best must be below this angle
      minAlpha - with fits: the best must exceed this factor times the second
      minCount - with counts: the best must reach this count
      maxCount - with counts: the second must stay below this count
    References
      F. Niessen, T. Nyyssönen, A. A. Gazder, R. Hielscher, Parent grain reconstruction from
          partially or fully transformed microstructures in MTEX, J. Appl. Cryst. 55 (2022)
          180-194
    See also
      parentGrainReconstructor.calcGBVotes, parentGrainReconstructor.calcTPVotes,
      parentGrainReconstructor.calcParentFromGraph, MaParentGrainReconstruction"""
    v = self.votes
    if v is None:
      raise ValueError("compute the votes first, with calcGBVotes, calcTPVotes or clusterVariantGraph")
    val = v.value
    second = val[:, 1] if val.shape[1] > 1 else np.full(len(val), np.nan)
    with np.errstate(invalid="ignore"):
      if v.kind == "fit":
        do = val[:, 0] < minFit
        if minDelta is not None:
          do &= second - val[:, 0] > minDelta
        if minAlpha is not None:
          do &= val[:, 0] > minAlpha * second
      elif v.kind == "prob":
        do = (v.parentId[:, 0] > 0) & (val[:, 0] > minProb)
        if minDelta is not None:
          do &= val[:, 0] - second > minDelta
      else:
        do = val[:, 0] >= minCount
        if maxCount is not None:
          do &= second < maxCount
    idx = np.flatnonzero(do)
    # a grain that is a parent already is reconstructed again from its measured orientation, that
    # of the one prior grain it is; a merged parent grain has no one measured orientation and stays
    cur = self._pos(self.mergeId)
    single = np.bincount(cur, minlength=self.grains.size) == 1
    priorOf = np.full(self.grains.size, -1, dtype=np.int64)
    priorOf[cur[single[cur]]] = np.flatnonzero(single[cur])
    isP = self.grains.phaseId[idx] == self.parentPhaseId
    idx, isP = idx[~isP | (priorOf[idx] >= 0)], isP[~isP | (priorOf[idx] >= 0)]
    if len(idx) == 0:
      return
    child = self.grains.meanRotation.data.reshape(-1, 4)[idx].copy()
    child[isP] = self.grainsPrior.meanRotation.data.reshape(-1, 4)[priorOf[idx[isP]]]
    q = qmul(child, self._PV()[v.parentId[idx, 0] - 1])
    self._setGrains(idx, q, self.parentPhaseId)
    val[idx] = np.nan

  # ------------------------------------------------------------------ merging and reverting

  # the prior grains follow the grains they went into
  def _followMerge(self, merged):
    parent = self.grains.parentId
    self.mergeId = parent[self._pos(self.mergeId)].astype(np.int64)
    self.grains = merged.update()
    self.votes = None
    self.graph = None

  def mergeSimilar(self, threshold=5 * degree):
    """neighbouring grains of similar orientation merged; MATLAB's
    parentGrainReconstructor.mergeSimilar

      job.mergeSimilar()                         # within 5 degree
      job.mergeSimilar(threshold=7.5*degree)     # within 7.5 degree

    Neighbouring grains of one phase whose mean orientations are within the threshold become
    one; job.grains and job.mergeId follow the merge, and job.grains.clusterSize counts the
    prior grains in each.

    Input
      job - parentGrainReconstructor
    Keywords
      threshold - the misorientation angle up to which grains are merged
    See also
      parentGrainReconstructor.mergeInclusions, grain2d.merge"""
    self._followMerge(self.grains.merge(threshold=threshold))
    self.grains.clusterSize = np.bincount(self._pos(self.mergeId), minlength=self.grains.size)

  def mergeInclusions(self, maxSize=np.inf):
    """the inclusions merged into the parent grains around them; MATLAB's
    parentGrainReconstructor.mergeInclusions

      job.mergeInclusions()              # every grain enclosed by a parent grain
      job.mergeInclusions(maxSize=50)    # only those of at most 50 pixels

    The child and parent grains lying inside a parent grain, of at most maxSize pixels, become
    part of it, with its orientation; job.grains and job.mergeId follow the merge.

    Input
      job - parentGrainReconstructor
    Keywords
      maxSize - the largest number of pixels of an inclusion that is merged
    See also
      parentGrainReconstructor.mergeSimilar, grain2d.merge"""
    self._followMerge(self.grains.merge(inclusions=self.isChild | self.isParent, host=self.isParent, maxSize=maxSize))

  def revert(self, ind=None):
    """the reconstruction undone, for all grains or those named; MATLAB's
    parentGrainReconstructor.revert

      job.revert()                               # undo the whole reconstruction
      job.revert(job.grains.fit > 5*degree)      # undo the grains a mask selects
      job.revert(job.parentGrains[:5])           # undo specific grains

    The grains named, a mask or positions into job.grains, or grains, are given back the phase
    and the orientation of their prior grains; a grain merged of several comes apart into them,
    the others stay merged. The votes and the variant graph are dropped.

    Input
      job - parentGrainReconstructor
      ind - bool mask or positions into job.grains, or grain2d; all grains when omitted
    See also
      parentGrainReconstructor.calcParentFromVote, GrainGraphBasedReconstruction,
      MaParentGrainReconstruction"""
    self.votes = None
    self.graph = None
    g = self.grains
    if ind is None:
      do = np.ones(g.size, dtype=bool)
    elif isinstance(ind, grain2d):
      do = np.zeros(g.size, dtype=bool)
      do[self._pos(ind.id)] = True
    else:
      ind = np.asarray(ind)
      do = ind.astype(bool) if ind.dtype == bool else np.isin(np.arange(g.size), ind)
    doPrior = do[self._pos(self.mergeId)]
    prior = self.grainsPrior
    if not doPrior.any():
      return
    if doPrior.all():
      self.grains = prior.copy()
      self.mergeId = prior.id.astype(np.int64).copy()
      return
    at = self._pos(self.mergeId)
    if np.bincount(at[doPrior]).max() == 1:
      self._setGrains(at[doPrior], prior.meanRotation.data.reshape(-1, 4)[doPrior], prior.phaseId[doPrior])
      return
    # the reverted prior grains stand alone, the others are merged again as they were
    labels = self.mergeId.copy()
    labels[doPrior] = labels.max() + 1 + np.arange(np.count_nonzero(doPrior))
    merged = prior.merge(labels)
    parent = prior.parentId
    q = merged.meanRotation.data.reshape(-1, 4).copy()
    ph = merged.phaseId.copy()
    npos = merged._positions()[parent]
    q[npos[doPrior]] = prior.meanRotation.data.reshape(-1, 4)[doPrior]
    ph[npos[doPrior]] = prior.phaseId[doPrior]
    q[npos[~doPrior]] = g.meanRotation.data.reshape(-1, 4)[at[~doPrior]]
    ph[npos[~doPrior]] = g.phaseId[at[~doPrior]]
    merged.meanRotation = merged.meanRotation._new(q)
    merged.phaseId = ph
    self.grains = merged.update()
    self.mergeId = parent.astype(np.int64)

  # ------------------------------------------------------------------ the results

  def calcParentEBSD(self, exactOR=False):
    """the reconstructed parent map; MATLAB's parentGrainReconstructor.calcParentEBSD

      parentEBSD = job.calcParentEBSD()              # the parent orientation of every grain
      parentEBSD = job.calcParentEBSD(exactOR=True)  # the parent each pixel's own orientation gives

    A copy of the prior map with the grain ids of the current grains, every child pixel of a
    grain that is a parent now given the parent orientation of its grain and the parent phase.
    The pixels carry variantId, packetId and fit, the misfit of the pixel to its variant; the
    other pixels are NaN there. With exactOR every pixel carries the parent orientation its own
    orientation gives through the relationship, the variant nearest the grain's parent, so the
    fit is zero.

    Input
      job - parentGrainReconstructor
    Output
      parentEBSD - EBSD, with parentEBSD.grainId, variantId, packetId and fit
    Keywords
      exactOR - enforce the exact orientation relationship pixel by pixel
    See also
      parentGrainReconstructor.calcVariants, MaParentGrainReconstruction"""
    ebsd = self.ebsdPrior.copy()
    gid = np.asarray(ebsd.grainId).reshape(-1).copy()
    ph = ebsd.phaseId.reshape(-1)
    pp = _at(self.grainsPrior._positions(), gid)
    inside = (gid > 0) & (pp >= 0)
    gid[inside] = self.mergeId[pp[inside]]
    ebsd.grainId = gid.reshape(ebsd.shape)
    cur = np.full(gid.shape, -1, dtype=np.int64)
    cur[inside] = self._pos(gid[inside])
    now = inside & (ph == self.childPhaseId)
    now[now] &= self.grains.phaseId[cur[now]] == self.parentPhaseId
    if not now.any():
      return ebsd
    rot = ebsd.rotations.data.reshape(-1, 4)
    child = rot[now]
    pm = self.grains.meanRotation.data.reshape(-1, 4)[cur[now]]
    if exactOR:
      pm = self._parentOf(child, pm)[0]
    fr = self.grains.frame
    vId, pId, _, fit = calcVariantId(_ori(pm, self.csParent, fr), _ori(child, self.csChild, fr), self.p2c)
    q = self._throughVariant(_ori(pm, self.csParent, fr).projectIntoFundamentalRegion(), vId)
    fit = np.asarray(fit, dtype=float)
    vId = self._mapVariants(vId)
    pId = np.asarray(pId, dtype=float)
    vId[np.isnan(fit)] = np.nan
    pId[np.isnan(fit)] = np.nan
    for name, val in (("variantId", vId), ("packetId", pId), ("fit", fit)):
      full = np.full(gid.shape, np.nan)
      full[now] = val
      setattr(ebsd, name, full.reshape(ebsd.shape))
    rot[now] = q
    ph[now] = self.parentPhaseId
    return ebsd

  def calcVariants(self, **options):
    """job.calcVariants(): the variant, packet and Bain id of every transformed prior grain
    against the parent it went into, as job.variantId, job.packetId, job.bainId; the parent
    grains take the representative of the last of their children"""
    tr = np.flatnonzero(self.isTransformed)
    if len(tr) == 0:
      return
    child = self._meanOri(self.grainsPrior, tr, self.csChild)
    at = self._pos(self.mergeId[tr])
    parent = self._meanOri(self.grains, at, self.csParent)
    vId, pId, bId, _ = calcVariantId(parent, child, self.p2c, **options)
    for name, val in (("variantId", self._mapVariants(vId)), ("packetId", pId), ("bainId", bId)):
      full = np.array(self._priorProp(name), dtype=float)
      full[tr] = val
      setattr(self.grainsPrior, name, full)
    self._setGrains(at, self._throughVariant(parent.projectIntoFundamentalRegion(), vId), self.parentPhaseId)

  # the parents in the fundamental region taken back through the variants vId and forward
  # through the relationship, the representative of the parent the variants agree on
  def _throughVariant(self, parent, vId):
    V = self.p2c.variants().data.reshape(-1, 4)
    return qmul(qmul(parent.data.reshape(-1, 4), qconj(V[np.asarray(vId) - 1])), self.p2c.data.reshape(1, 4))

  def parent2ChildInfo(self):
    """OR = job.parent2ChildInfo(): the relationship with its closest rational parallel
    planes and directions, their deviations, its axis in both crystals and the angles and
    axes of the variants, printed and returned"""
    if self.p2c is None:
      raise ValueError("no p2c defined; set job.p2c or run job.calcParent2Child()")
    p2c = self.p2c
    n1, n2, d1, d2 = round2Miller(p2c, maxIndex=8)
    rational = orientation.map(n1, n2, d1, d2)
    V = p2c.variants()
    c2c = _ori(qmul(V.data.reshape(-1, 4), qconj(V.data.reshape(-1, 4)[:1])), self.csChild, self.csChild)
    OR = SimpleNamespace(p2c=p2c, p2cRational=rational, plane=SimpleNamespace(parent=n1, child=n2),
                         direction=SimpleNamespace(parent=d1, child=d2),
                         devAngle=SimpleNamespace(plane=float(np.min(np.asarray((p2c * n1.symmetrise()).angle(n2)))),
                                                  direction=float(np.min(np.asarray((p2c * d1.symmetrise()).angle(d2))))),
                         variants=SimpleNamespace(orientation=V, c2c=c2c, angle=np.asarray(c2c.angle()).reshape(-1)))
    print(" -> OR info:")
    print(f"    - OR misorientation angle = {float(p2c.angle()) / degree:.4g}°")
    print(" -> Parallel planes")
    print(f"    - Closest parent plane = {n1.char()}")
    print(f"    - Closest child plane = {n2.char()}")
    print(f"    - Ang. dev. from parallel plane relationship from OR = {OR.devAngle.plane / degree:.4g}°")
    print(" -> Parallel directions")
    print(f"    - Closest parent direction = {d1.char()}")
    print(f"    - Closest child direction = {d2.char()}")
    print(f"    - Ang. dev. from parallel directions relationship from OR = {OR.devAngle.direction / degree:.4g}°")
    print(" -> Angle of unique variants")
    vm = self.variantMap
    for j in range(V.size):
      print(f"    - {j + 1}: {OR.variants.angle[vm[j] - 1] / degree:.2f}°")
    return OR

  # ------------------------------------------------------------------ display

  def __repr__(self):
    g = self.grains
    lines = ["parentGrainReconstructor", ""]
    gs = g.numPixel.astype(float)
    total = max(gs.sum(), 1)
    rows = [("phase", "mineral", "symmetry", "grains", "area", "reconstructed")]
    if self.csParent is not None:
      P = self.parentPhaseId if self.csParent in g.CSList else None
      nP = 0 if P is None else int(np.count_nonzero(g.phaseId == P))
      area = 0 if P is None else 100 * gs[g.phaseId == P].sum() / total
      priorC = np.count_nonzero(self.grainsPrior.phaseId == self.childPhaseId) if self.csChild is not None else 0
      rec = 100 * np.count_nonzero(self.isTransformed) / priorC if priorC and P is not None else 0
      rows.append(("parent", self.csParent.mineral, self.csParent.pointGroup, str(nP), f"{_xnum(area)}%", f"{_xnum(rec)}%"))
    if self.csChild is not None:
      C = self.childPhaseId
      rows.append(("child", self.csChild.mineral, self.csChild.pointGroup, str(int(np.count_nonzero(g.phaseId == C))),
                   f"{_xnum(100 * gs[g.phaseId == C].sum() / total)}%", ""))
    widths = [max(len(r[j]) for r in rows) for j in range(6)]
    lines += ["  " + "  ".join(x.ljust(w) for x, w in zip(r, widths)) for r in rows]
    if self.p2c is not None and self.reportFit:
      lines += ["", f" OR: {self.p2c.char()}"]
      for kind in ("p2c", "c2c"):
        fit = self.calcGBFit(**{kind: True})[0] / degree
        if len(fit):
          qs = ", ".join(f"{_xnum(_quantile(fit, q))}°" for q in (0.2, 0.4, 0.6, 0.8))
          lines.append(f"   {kind} fit: {qs} (quintiles)")
      # the nearest relationship of low indices, when this one is not rational itself
      n1, n2, d1, d2 = round2Miller(self.p2c, maxIndex=5)
      ideal = orientation.map(n1, n2, d1, d2)
      dev = float(ideal.angle(self.p2c)) / degree
      if dev > 1e-3:
        lines.append(f"   closest ideal OR: {ideal.char()} fit: {_xnum(dev)}°")
    if self.hasGraph:
      mId = g.merge(self.graph, testRun=True)
      n = np.bincount(mId)[1:]
      lines += ["", f" graph: {int(n[n > 1].sum())} grains in {int(np.count_nonzero(n > 1))} clusters + "
                    f"{int(np.count_nonzero(n == 1))} single grain clusters"]
    elif self.hasVariantGraph:
      lines += ["", f" variant graph: {self.graph.nnz} entries"]
    if self.votes is not None:
      val = self.votes.value[:, 0]
      val = val[~np.isnan(val)]
      if len(val):
        scale, name, unit = {"fit": (1 / degree, "fit", "°"), "prob": (100, "probabilities", "%"), "count": (1, "counts", "")}[self.votes.kind]
        qs = ", ".join(f"{_xnum(_quantile(val * scale, q))}{unit}" for q in (0.8, 0.6, 0.4, 0.2))
        lines += ["", f" votes: {len(val)} × 1", f"   {name}: {qs} (quintiles)"]
    return aligned("\n".join(lines))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from .variantgraph import (calcGraph, clusterGraph, calcParentFromGraph, mergeByGraph, plotGraph, calcVariantGraph,
                               clusterVariantGraph)


__all__ = ["parentGrainReconstructor", "voteTable", "majorityVote"]
