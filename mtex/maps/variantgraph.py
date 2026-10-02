# the graphs of a parent grain reconstruction (MATLAB calcGraph, clusterGraph,
# calcParentFromGraph, mergeByGraph, calcVariantGraph, clusterVariantGraph; Julia
# reconstruction.jl)
#
#   job.calcGraph(threshold=2.5*degree, tol=2.5*degree); job.clusterGraph(inflationPower=1.6)
#   job.calcParentFromGraph(), job.mergeByGraph(threshold=5*degree), job.plotGraph()
#   job.calcVariantGraph(threshold=3.5*degree, tol=1.5*degree); job.clusterVariantGraph(includeSimilar=True)
#
# The grain graph has a node per grain and an edge where two neighbours fit the
# relationship; its clusters by MCL are the parent grains. The variant graph of
# arXiv:2201.02103 has a node per child grain and variant, one per parent grain, which has
# nothing left to choose but carries the graph through it, and an edge where two candidates
# agree; MCL with the rows of a grain normalised together turns it into votes per grain and
# variant, the table calcParentFromVote reads.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import variantAngles
from ..geometry.parent2child import calcParent
from ..geometry.quaternion import qmul
from ..geometry.units import degree
from .kernels import mclExpandNumba
from .reconstruction import _ori, _quantile, parentGrainReconstructor, probabilityOf, voteTable
from ..geometry.install import install


def _sparse(r, c, v, n):
  from scipy.sparse import coo_matrix
  return coo_matrix((np.asarray(v, dtype=float), (np.asarray(r, dtype=np.int64), np.asarray(c, dtype=np.int64))), shape=(n, n)).tocsr()


# ------------------------------------------------------------------ the grain graph

def calcGraph(job, threshold=2 * degree, tol=1.5 * degree, p2c=False, c2c=False):
  """job.calcGraph(threshold=2.5*degree, tol=2.5*degree): job.graph, a node per grain
  and between neighbours the probability that the misfit of their boundary to the
  relationship still means one parent"""
  fit, pairs = job.calcGBFit(p2c=p2c, c2c=c2c)
  prob = probabilityOf(fit, threshold, tol)
  pos = job.grains._positions()
  a, b = pos[pairs[:, 0]], pos[pairs[:, 1]]
  G = _sparse(a, b, prob, job.grains.size)
  job.graph = G.maximum(G.T).tocsr()


def mclComponents(A, p, maxIter=np.inf):
  """MCL until the columns settle: expansion, inflation by p, pruning, the columns normalised"""
  from scipy.sparse import identity
  A = A.tocsr().copy()
  A.data[A.data <= 1e-4] = 0
  A.eliminate_zeros()
  if A.multiply(A.T).nnz == 0:
    A = A.maximum(A.T).tocsr()
  if not np.any(A.diagonal()):
    A = (A + identity(A.shape[0], format="csr")).tocsr()
  e, i = 1.0, 1
  while e > 1e-3 and i < maxIter:
    i += 1
    A = (A @ A).tocsr()
    A.data = A.data ** p
    A.data[A.data <= 1e-4] = 0
    A.eliminate_zeros()
    s = np.asarray(A.sum(axis=0)).reshape(-1)
    A = (A @ _diagInv(s)).tocsr()
    sq = A.multiply(A)
    e = float(np.max(A.max(axis=0).toarray().reshape(-1) - np.asarray(sq.sum(axis=0)).reshape(-1), initial=0))
  return A


def _diagInv(s):
  from scipy.sparse import diags
  inv = np.zeros_like(s, dtype=float)
  inv[s > 0] = 1 / s[s > 0]
  return diags(inv)


def clusterGraph(job, inflationPower=None, **options):
  """job.clusterGraph(inflationPower=1.6): job.graph clustered by MCL, so that its
  components are the parent grains; a variant graph is clustered by clusterVariantGraph"""
  if job.hasVariantGraph:
    return job.clusterVariantGraph(inflationPower=1.05 if inflationPower is None else inflationPower, **options)
  job.graph = mclComponents(job.graph, 1.6 if inflationPower is None else inflationPower)


# the orientations of the grains as children or as parents, nan for the other phase
def _grainOrientations(job):
  g = job.grains
  q = g.meanRotation.data.reshape(-1, 4)
  wasChild, wasParent = job.isChild, job.isParent
  return q, wasChild, wasParent, g.numPixel.astype(float)


def calcParentFromGraph(job):
  """job.calcParentFromGraph(): every component of job.graph of more than one grain gives
  its child grains the parent orientation that fits them together, or fits the parent
  grains among them; job.grains.fit is the misfit of every grain to it and
  job.grains.clusterSize the size of its component"""
  if job.graph is None:
    raise ValueError("no graph; compute it with job.calcGraph() and cluster it with job.clusterGraph()")
  g = job.grains
  q, wasChild, wasParent, w = _grainOrientations(job)
  mergeId = g.merge(job.graph, testRun=True)
  pos = g._positions()
  mergeId = mergeId if len(mergeId) == g.size else mergeId[pos[g.id]]
  PV = job._PV()
  parentQ = np.where(wasParent[:, None], q, np.nan)
  fit = np.full(g.size, np.inf)
  fit[~wasParent & ~wasChild] = np.nan
  clusterSize = np.ones(g.size)
  fr = g.frame
  for k in np.unique(mergeId):
    ind = mergeId == k
    if wasParent[ind].all():
      fit[ind] = np.nan
      continue
    if np.count_nonzero(ind) <= 1:
      continue
    if wasChild[ind].all():
      child = _ori(q[ind], job.csChild, fr)
      pOri, _ = calcParent(child, job.p2c, weights=w[ind])
      ids, f = calcParent(child, pOri, job.p2c, id=True)
      parentQ[ind] = qmul(q[ind], PV[np.asarray(ids) - 1])
      fit[ind] = f
      clusterSize[ind] = np.count_nonzero(ind)
    elif (wasChild[ind] | wasParent[ind]).all():
      pi = ind & wasParent
      pOri = _ori(q[pi], job.csParent, fr).mean(weights=w[pi], robust=True)
      fit[pi] = np.asarray(_ori(q[pi], job.csParent, fr).angle(pOri)).reshape(-1)
      ci = ind & wasChild
      child = _ori(q[ci], job.csChild, fr)
      ids, f = calcParent(child, pOri, job.p2c, id=True)
      parentQ[ci] = qmul(q[ci], PV[np.asarray(ids) - 1])
      fit[ci] = f
  has = ~np.isnan(parentQ[:, 0])
  job._setGrains(np.flatnonzero(has), parentQ[has], job.parentPhaseId)
  job.grains.fit = fit
  job.grains.clusterSize = clusterSize
  job.graph = None


def mergeByGraph(job, threshold=5 * degree):
  """job.mergeByGraph(threshold=5*degree): every component of job.graph whose grains fit a
  common parent within the threshold becomes one parent grain of that orientation"""
  if job.graph is None:
    raise ValueError("no graph; compute it with job.calcGraph() and cluster it with job.clusterGraph()")
  g = job.grains
  q, wasChild, wasParent, w = _grainOrientations(job)
  mergeId = g.merge(job.graph, testRun=True)
  PV = job._PV()
  fr = g.frame
  labels = np.unique(mergeId)
  recQ = np.full((len(labels), 4), np.nan)
  fit = np.full(len(labels), np.inf)
  for n, k in enumerate(labels):
    ind = mergeId == k
    if np.count_nonzero(ind) <= 1:
      continue
    if wasParent[ind].all():
      recQ[n] = _ori(q[ind], job.csParent, fr).mean(weights=w[ind]).data.reshape(4)
      fit[n] = 0
    elif wasChild[ind].all():
      pOri, fit[n] = calcParent(_ori(q[ind], job.csChild, fr), job.p2c, weights=w[ind])
      recQ[n] = pOri.data.reshape(4)
    elif (wasChild[ind] | wasParent[ind]).all():
      pi, ci = ind & wasParent, ind & wasChild
      pOri = _ori(q[pi], job.csParent, fr).mean(weights=w[pi], robust=True)
      child = _ori(q[ci], job.csChild, fr)
      ids, f = calcParent(child, pOri, job.p2c, id=True)
      cand = qmul(q[ci], PV[np.asarray(ids) - 1])
      allQ = np.concatenate([pOri.data.reshape(1, 4), cand])
      recQ[n] = _ori(allQ, job.csParent, fr).mean(robust=True).data.reshape(4)
      fit[n] = np.max(f)
  good = fit < threshold
  goodNode = good[np.searchsorted(labels, mergeId)]
  G = job.graph.tocsr().copy()
  from scipy.sparse import diags
  D = diags(goodNode.astype(float))
  G = (D @ G @ D).tocsr()
  work = g.copy()
  merged = work.merge(G)
  parent = work.parentId
  job.mergeId = parent[job._pos(job.mergeId)].astype(np.int64)
  job.grains = merged
  mpos = merged._positions()
  at, qs = [], []
  for n in np.flatnonzero(good):
    members = np.flatnonzero(mergeId == labels[n])
    at.append(mpos[parent[members[0]]])
    qs.append(recQ[n])
  if at:
    job._setGrains(np.array(at), np.array(qs), job.parentPhaseId)
  else:
    job.grains = merged.update()
  job.votes = None
  job.graph = None


def plotGraph(job, **kwargs):
  """job.plotGraph(lineWidth=2): the boundaries of the grain graph, the more transparent the
  more likely its two grains share a parent"""
  if job.graph is None:
    raise ValueError("no graph; compute it with job.calcGraph()")
  if job.hasVariantGraph:
    raise NotImplementedError("plotting the variant graph")
  from ..plotting.mapplot import plotBoundary
  g = job.grains
  pairs = g.neighbors()
  pos = g._positions()
  G = job.graph.tocsr()
  v = np.asarray(G[pos[pairs[:, 0]], pos[pairs[:, 1]]]).reshape(-1)
  gB = g.boundary
  row = job._segmentRows(pairs, gB.grainId)
  sel = row >= 0
  return plotBoundary(gB[sel], edgeAlpha=1 - v[row[sel]], **kwargs)


# ------------------------------------------------------------------ the variant graph

def _conj(q):
  return q * np.array([1.0, -1, -1, -1])


# the parent variants with a variant within 8 degrees of an earlier one pointed at it
def _bestFriends(job, PV, tol=8 * degree):
  o = _ori(PV, job.csParent, job.csChild.stripSym())
  close = np.asarray(o.reshape(-1, 1).angle(o.reshape(1, -1))) < tol
  return np.argmax(close, axis=1)


# the angle between the parent variants, the child symmetry ignored
def _variantAngles(job, PV):
  o = _ori(PV, job.csParent, job.csChild.stripSym())
  return np.asarray(o.reshape(-1, 1).angle(o.reshape(1, -1)))


def _layout(job, numV):
  width = numV * job.isChild + job.isParent
  first = np.concatenate([[0], np.cumsum(width)])
  return width, first[:-1], int(first[-1])


def calcVariantGraph(job, threshold=2 * degree, tol=1.5 * degree, p2c=False, p2p=False, c2c=False, mergeSimilar=False,
                     diagonal=True):
  """job.calcVariantGraph(threshold=3.5*degree, tol=1.5*degree): job.graph, the variant
  graph: a node per child grain and parent variant and one per parent grain, and an edge
  where two neighbours agree - child to child over every pair of variants, parent to child
  over the variants that fit the parent, parent to parent over their orientations - with
  the probability of the agreement. mergeSimilar joins the variants within 8 degrees of
  one another into one node; votes of the job restrict every grain to its best variant"""
  g = job.grains
  PV = job._PV()
  numV = len(PV)
  friend = _bestFriends(job, PV) if mergeSimilar else np.arange(numV)
  width, first, nVG = _layout(job, numV)
  pos = g._positions()
  noOpt = not (p2c or p2p or c2c)
  from scipy.sparse import identity, csr_matrix
  G = csr_matrix((nVG, nVG)) if not diagonal else identity(nVG, format="csr")

  def both(r, c, v):
    return _sparse(np.r_[r, c], np.r_[c, r], np.r_[v, v], nVG)

  if len(job.parentGrains) and (p2p or noOpt):
    pr, o1, o2 = job._pairs("p2p")
    if len(pr):
      prob = probabilityOf(np.asarray(o1.angle(o2)).reshape(-1), threshold, tol)
      ok = prob > 0.1
      G = G.maximum(both(first[pos[pr[ok, 0]]], first[pos[pr[ok, 1]]], prob[ok]))
  if len(job.parentGrains) and (p2c or noOpt):
    pr, oP, oC = job._pairs("p2c")
    if len(pr):
      iP, iC = first[pos[pr[:, 0]]], first[pos[pr[:, 1]]]
      cand = _ori(qmul(oC.data.reshape(-1, 1, 4), PV[None]).reshape(-1, 4), job.csParent, g.frame).reshape(-1, numV)
      prob = probabilityOf(np.asarray(cand.angle(oP.reshape(-1, 1))).reshape(-1, numV), threshold, tol)
      for k in range(numV):
        ok = prob[:, k] > 0.1
        G = G.maximum(both(iP[ok], iC[ok] + friend[k], prob[ok, k]))
  if c2c or noOpt:
    pr, o1, o2 = job._pairs("c2c")
    if len(pr):
      i1, i2 = first[pos[pr[:, 0]]], first[pos[pr[:, 1]]]
      m = qmul(_conj(o1.data.reshape(-1, 4)), o2.data.reshape(-1, 4))
      prob = probabilityOf(variantAngles(m, PV, job.csParent.sym.properGroup().rotations().data.reshape(-1, 4)), threshold, tol)
      p, k1, k2 = np.nonzero(prob > 0.1)
      v = prob[p, k1, k2]
      # the variants merged into one node add up, as MATLAB's sparse does
      G = G.maximum(both(i1[p] + friend[k1], i2[p] + friend[k2], v))
  if job.votes is not None and len(job.votes) == g.size:
    # every grain restricted to its best variant and the one merged with it
    friendAll = _bestFriends(job, PV)
    partner = np.zeros(numV, dtype=np.int64) - 1
    lone = ~np.isin(np.arange(numV), friendAll)
    partner[friendAll[lone]] = np.flatnonzero(lone)
    cp = width > 0
    best = job.votes.parentId[cp, 0] - 1
    keep = [first[cp] + np.minimum(best, width[cp] - 1)]
    second = partner[best]
    has = (second >= 0) & (width[cp] > 1)
    keep.append((first[cp] + second)[has])
    d = np.zeros(nVG)
    d[np.concatenate(keep)] = 1
    from scipy.sparse import diags
    D = diags(d)
    G = (D @ G @ D).tocsr()
  if diagonal:
    lone = np.asarray((G > 0).sum(axis=0)).reshape(-1) == 1
    from scipy.sparse import diags
    G = (diags((~lone).astype(float)) @ G).tocsr()
  G.eliminate_zeros()
  job.graph = G


def clusterVariantGraph(job, inflationPower=1.05, numIter=10, cutOff=1e-4, minCluster=4, includeSimilar=False, mergeSimilar=False,
                        pseudoDiagonal=True, sym=False, keepGraph=False, **graphOptions):
  """job.clusterVariantGraph(includeSimilar=True): the variant graph (calcVariantGraph
  first when there is none) clustered by MCL - expansion, inflation by inflationPower,
  pruning below cutOff, the rows of every grain normalised together - over numIter steps;
  the score of every node of a child grain whose cluster has minCluster members at least
  becomes its vote, job.votes. includeSimilar lends a variant a tenth of the score of the
  variants within 5 degrees, mergeSimilar all of it"""
  if job.graph is None:
    job.calcVariantGraph(**graphOptions)
  A = job.graph.tocsr().astype(float)
  job.graph = None
  g = job.grains
  PV = job._PV()
  numV = len(PV)
  width, first, nVG = _layout(job, numV)
  node2grain = np.repeat(np.arange(g.size), width)
  isChildNode = np.repeat(job.isChild, width)

  def normalize(A):
    s = np.asarray(A.sum(axis=1)).reshape(-1)
    t = np.bincount(node2grain, weights=s, minlength=g.size)[node2grain]
    return (_diagInv(t) @ A).tocsr()

  def dropPseudoDiagonal(A):
    A = A.tocoo()
    same = (node2grain[A.row] == node2grain[A.col]) & (A.row != A.col) & isChildNode[A.row]
    A.data[same] = 0
    A = A.tocsr()
    A.eliminate_zeros()
    return A

  # the expansion A A with the inflation and the pruning, compiled, or by scipy
  def expand(A):
    E = mclExpandNumba(A, inflationPower, cutOff)
    if E is not None:
      return E
    A = (A @ A).tocsr()
    A.data = A.data ** inflationPower
    A.data[A.data <= cutOff] = 0
    A.eliminate_zeros()
    return A

  A = normalize(A)
  for _ in range(numIter):
    A = expand(A)
    if not pseudoDiagonal:
      A = dropPseudoDiagonal(A)
    if sym:
      A = normalize(A.T.multiply(A).sqrt().tocsr())
    A = normalize(A)
  if sym:
    A = normalize(A.T.multiply(A).sqrt().tocsr())
  s = np.asarray(A.sum(axis=1)).reshape(-1)
  if inflationPower <= 1:
    s = s - A.diagonal()
  s = s * (np.asarray((A > 0).sum(axis=1)).reshape(-1) >= minCluster)
  score = np.full((g.size, numV), np.nan)
  score[job.isChild] = s[isChildNode].reshape(-1, numV)
  if includeSimilar or mergeSimilar:
    M = (_variantAngles(job, PV) < 5 * degree).astype(float)
    if includeSimilar:
      M = 0.9 * M + 0.1 * np.eye(numV)
    score = score @ M
  order = np.argsort(np.where(np.isnan(score), -np.inf, -score), axis=1, kind="stable")
  job.votes = voteTable(order + 1, prob=np.take_along_axis(score, order, axis=1))
  if keepGraph:
    job.graph = A


for _f in (calcGraph, clusterGraph, calcParentFromGraph, mergeByGraph, plotGraph, calcVariantGraph, clusterVariantGraph):
  install(parentGrainReconstructor, _f)
del _f

__all__ = ["mclComponents"]
