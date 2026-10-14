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

from ..geometry.kernels import qconj, variantAngles
from ..geometry.parent2child import calcParent
from ..geometry.quaternion import qmul
from ..geometry.units import degree
from .kernels import mclExpandNumba
from .reconstruction import _ori, _quantile, _segmentRows, parentGrainReconstructor, probabilityOf, voteTable
from ..geometry.install import install


def _sparse(r, c, v, n):
  from scipy.sparse import coo_matrix
  return coo_matrix((np.asarray(v, dtype=float), (np.asarray(r, dtype=np.int64), np.asarray(c, dtype=np.int64))), shape=(n, n)).tocsr()


# ------------------------------------------------------------------ the grain graph

def calcGraph(job, threshold=2 * degree, tol=1.5 * degree, p2c=False, c2c=False):
  """the graph of the grains for a parent grain reconstruction, job.graph; MATLAB's calcGraph of
  parentGrainReconstructor

    job.calcGraph()                                        # the default probabilities
    job.calcGraph(threshold=2.5*degree, tol=2.5*degree)    # another misfit at probability 1/2
    job.calcGraph(c2c=True)                                # the child to child boundaries only

  A node per grain, and between neighbouring grains the probability that the misfit of their
  boundary to the orientation relationship still means one parent: a cumulative Gaussian of
  the misfit with the mean threshold and the spread tol. The child to child and the parent to
  child boundaries are both used unless one of them is asked for. The job is changed in place.

  Input
    job - parentGrainReconstructor with its orientation relationship job.p2c
  Keywords
    threshold - the misfit at which the probability is 1/2
    tol - the range about threshold over which the probability goes from 1 to 0
    p2c - the parent to child boundaries only
    c2c - the child to child boundaries only
  See also
    parentGrainReconstructor.clusterGraph, parentGrainReconstructor.calcGBFit,
    parentGrainReconstructor.calcVariantGraph, GrainGraphBasedReconstruction"""
  fit, pairs = job.calcGBFit(p2c=p2c, c2c=c2c)
  prob = probabilityOf(fit, threshold, tol)
  pos = job.grains._positions()
  a, b = pos[pairs[:, 0]], pos[pairs[:, 1]]
  G = _sparse(a, b, prob, job.grains.size)
  job.graph = G.maximum(G.T).tocsr()


def mclComponents(A, p, maxIter=np.inf):
  """the components of a weighted graph by Markov clustering, MCL; MATLAB's mclComponents

    A = mclComponents(job.graph, 1.6)                          # until the columns settle
    A = mclComponents(job.graph, 1.6, maxIter=4)               # at most four iterations

  The steps of S. van Dongen's algorithm repeated until the columns settle: the expansion
  A @ A, the inflation of every entry to the power p, the entries up to 1e-4 pruned and the
  columns normalised. Before, the entries up to 1e-4 are pruned, a graph with no symmetric
  entry made symmetric and, where the diagonal is empty, ones put on it. The larger p, the
  smaller the clusters; the nonzero pattern of the result joins the nodes of one cluster.

  Input
    A - sparse adjacency matrix, the weights between 0 and 1
    p - the inflation power
  Output
    A - sparse matrix of the clusters
  Keywords
    maxIter - the iteration bound; the loop stops after maxIter - 1 steps
  References
    S. van Dongen, Graph clustering by flow simulation, PhD thesis, University of Utrecht
        (2000)
  See also
    parentGrainReconstructor.clusterGraph, GrainReconstructionMCL"""
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
  """the graph of a parent grain reconstruction broken into clusters, one per parent grain;
  MATLAB's clusterGraph of parentGrainReconstructor

    job.clusterGraph()                                         # the grain graph by MCL
    job.clusterGraph(inflationPower=1.6)                       # larger for smaller clusters

  A grain graph, from calcGraph, is clustered by mclComponents, so that its components are
  the parent grains, for calcParentFromGraph or mergeByGraph. A variant graph, from
  calcVariantGraph, is clustered by clusterVariantGraph, which takes the remaining keyword
  arguments and an inflation power of 1.05 by default. The job is changed in place.

  Input
    job - parentGrainReconstructor with job.graph
  Output
    job.graph - the clustered graph
  Keywords
    inflationPower - the inflation power of MCL, larger for smaller clusters; by default 1.6
        for a grain graph, 1.05 for a variant graph
  See also
    mclComponents, parentGrainReconstructor.calcGraph,
    parentGrainReconstructor.calcParentFromGraph, parentGrainReconstructor.mergeByGraph,
    parentGrainReconstructor.clusterVariantGraph, GrainGraphBasedReconstruction"""
  if job.hasVariantGraph:
    return job.clusterVariantGraph(inflationPower=1.05 if inflationPower is None else inflationPower, **options)
  job.graph = mclComponents(job.graph, 1.6 if inflationPower is None else inflationPower)


# the positions of the grains of every label, the labels in increasing order and the positions
# of one label increasing
def _clusters(mergeId):
  order = np.argsort(mergeId, kind="stable")
  _, starts = np.unique(mergeId[order], return_index=True)
  return np.split(order, starts[1:])


# the orientations of the grains as children or as parents, nan for the other phase
def _grainOrientations(job):
  g = job.grains
  return g.meanRotation.data.reshape(-1, 4), job.isChild, job.isParent, g.numPixel.astype(float)


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
  for ind in _clusters(mergeId):
    if wasParent[ind].all():
      fit[ind] = np.nan
      continue
    if len(ind) <= 1:
      continue
    if wasChild[ind].all():
      child = _ori(q[ind], job.csChild, fr)
      pOri, _ = calcParent(child, job.p2c, weights=w[ind])
      ids, f = calcParent(child, pOri, job.p2c, id=True)
      parentQ[ind] = qmul(q[ind], PV[np.asarray(ids) - 1])
      fit[ind] = f
      clusterSize[ind] = len(ind)
    elif (wasChild[ind] | wasParent[ind]).all():
      pi = ind[wasParent[ind]]
      pOri = _ori(q[pi], job.csParent, fr).mean(weights=w[pi], robust=True)
      fit[pi] = np.asarray(_ori(q[pi], job.csParent, fr).angle(pOri)).reshape(-1)
      ci = ind[wasChild[ind]]
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
  """the grains of every cluster of the graph merged into a parent grain; MATLAB's
  mergeByGraph of parentGrainReconstructor

    job.mergeByGraph()                                         # merged where they fit within 5 degrees
    job.mergeByGraph(threshold=3*degree)                       # a tighter fit

  The components of job.graph, after clusterGraph, are merged as a test first. For every
  component of more than one grain the parent orientation is fitted: the mean of its parent
  grains, the parent fitted to its child grains by calcParent, or the robust mean of both;
  the misfit is the largest over its child grains. The components that fit within the
  threshold become parent grains of that orientation, the others are left as they were. The
  graph and the votes are removed; the job is changed in place.

  Input
    job - parentGrainReconstructor with a clustered job.graph
  Output
    job.grains - the reconstructed parent grains
  Keywords
    threshold - the largest misfit of a component that is merged
  See also
    parentGrainReconstructor.clusterGraph, parentGrainReconstructor.calcParentFromGraph,
    calcParent, GrainGraphBasedReconstruction"""
  from scipy.sparse import diags
  if job.graph is None:
    raise ValueError("no graph; compute it with job.calcGraph() and cluster it with job.clusterGraph()")
  g = job.grains
  q, wasChild, wasParent, w = _grainOrientations(job)
  mergeId = g.merge(job.graph, testRun=True)
  PV = job._PV()
  fr = g.frame
  labels, firstOf = np.unique(mergeId, return_index=True)
  recQ = np.full((len(labels), 4), np.nan)
  fit = np.full(len(labels), np.inf)
  for n, ind in enumerate(_clusters(mergeId)):
    if len(ind) <= 1:
      continue
    if wasParent[ind].all():
      recQ[n] = _ori(q[ind], job.csParent, fr).mean(weights=w[ind]).data.reshape(4)
      fit[n] = 0
    elif wasChild[ind].all():
      pOri, fit[n] = calcParent(_ori(q[ind], job.csChild, fr), job.p2c, weights=w[ind])
      recQ[n] = pOri.data.reshape(4)
    elif (wasChild[ind] | wasParent[ind]).all():
      pi, ci = ind[wasParent[ind]], ind[wasChild[ind]]
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
  D = diags(goodNode.astype(float))
  G = (D @ G @ D).tocsr()
  work = g.copy()
  merged = work.merge(G)
  parent = work.parentId
  job.mergeId = parent[job._pos(job.mergeId)].astype(np.int64)
  job.grains = merged
  # every good cluster is the merged grain of its first member
  if good.any():
    job._setGrains(merged._positions()[parent[firstOf[good]]], recQ[good], job.parentPhaseId)
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
  row = _segmentRows(pairs, gB.grainId)
  sel = row >= 0
  return plotBoundary(gB[sel], edgeAlpha=1 - v[row[sel]], **kwargs)


# ------------------------------------------------------------------ the variant graph


# the parent variants with a variant within 8 degrees of an earlier one pointed at it
def _bestFriends(job, PV, tol=8 * degree):
  return np.argmax(_variantAngles(job, PV) < tol, axis=1)


# the angle between the parent variants, the child symmetry ignored
def _variantAngles(job, PV):
  o = _ori(PV, job.csParent, job.csChild.stripSym())
  return np.asarray(o.reshape(-1, 1).angle(o.reshape(1, -1)))


# the variant pairs (k1, k2) of every child to child misorientation whose probability exceeds
# 0.1, with it, in chunks of misorientations: the table of the variant angles of one chunk at a
# time, the probability only of the angles below the one where it is 0.1, with a margin
def _variantPairs(m, PV, G, threshold, tol, chunk=1 << 15):
  from scipy.special import erfinv
  limit = threshold + tol / 2 * float(erfinv(0.8)) + 1e-9
  out = [], [], [], []
  for s0 in range(0, len(m), chunk):
    a = variantAngles(m[s0:s0 + chunk], PV, G)
    p, k1, k2 = np.nonzero(a < limit)
    v = probabilityOf(a[p, k1, k2], threshold, tol)
    ok = v > 0.1
    for lst, x in zip(out, (p[ok] + s0, k1[ok], k2[ok], v[ok])):
      lst.append(x)
  return tuple(np.concatenate(x) if x else np.zeros(0, dtype=int) for x in out)


def _layout(job, numV):
  width = numV * job.isChild + job.isParent
  first = np.concatenate([[0], np.cumsum(width)])
  return width, first[:-1], int(first[-1])


def calcVariantGraph(job, threshold=2 * degree, tol=1.5 * degree, p2c=False, p2p=False, c2c=False, mergeSimilar=False,
                     diagonal=True):
  """the variant graph of a parent grain reconstruction, job.graph; MATLAB's calcVariantGraph of
  parentGrainReconstructor

    job.calcVariantGraph()                                     # the default probabilities
    job.calcVariantGraph(threshold=3.5*degree, tol=1.5*degree) # another misfit at probability 1/2
    job.calcVariantGraph(mergeSimilar=True)                    # similar variants as one node
    job.calcVariantGraph(c2c=True)                             # the child to child boundaries only

  A node per child grain and parent variant and one per parent grain. An edge joins two
  nodes of neighbouring grains where they agree: child to child over every pair of variants,
  parent to child over the variants that fit the parent, parent to parent over their
  orientations, with the probability of the agreement, a cumulative Gaussian of the misfit
  with the mean threshold and the spread tol. All three kinds are used unless some are asked
  for. When the job holds votes, every grain is restricted to its best variant. The job is
  changed in place; clusterVariantGraph clusters the graph.

  Input
    job - parentGrainReconstructor with its orientation relationship job.p2c
  Keywords
    threshold - the misfit at which the probability is 1/2
    tol - the range about threshold over which the probability goes from 1 to 0
    p2c - the parent to child boundaries
    p2p - the parent to parent boundaries
    c2c - the child to child boundaries
    mergeSimilar - join the variants within 8 degrees of one another into one node
    diagonal - put every node on the diagonal of the graph, the loops MCL needs
  References
    R. Hielscher, T. Nyyssönen, F. Niessen, A.A. Gazder, The variant graph approach to
        improved parent grain reconstruction, Materialia 22 (2022) 101399
  See also
    parentGrainReconstructor.clusterVariantGraph, parentGrainReconstructor.calcParentFromVote,
    parentGrainReconstructor.calcGraph, MaParentGrainReconstruction"""
  g = job.grains
  PV = job._PV()
  numV = len(PV)
  friend = _bestFriends(job, PV) if mergeSimilar else np.arange(numV)
  width, first, nVG = _layout(job, numV)
  pos = g._positions()
  noOpt = not (p2c or p2p or c2c)
  from scipy.sparse import csr_matrix, diags, identity
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
      m = qmul(qconj(o1.data.reshape(-1, 4)), o2.data.reshape(-1, 4))
      p, k1, k2, v = _variantPairs(m, PV, job.csParent.sym.properGroup().rotations().data.reshape(-1, 4), threshold, tol)
      # the variants merged into one node add up, as MATLAB's sparse does
      G = G.maximum(both(i1[p] + friend[k1], i2[p] + friend[k2], v))
  if job.votes is not None and len(job.votes) == g.size:
    # every grain restricted to its best variant and the one merged with it
    friendAll = _bestFriends(job, PV)
    partner = np.full(numV, -1, dtype=np.int64)
    lone = np.bincount(friendAll, minlength=numV) == 0
    partner[friendAll[lone]] = np.flatnonzero(lone)
    cp = width > 0
    best = job.votes.parentId[cp, 0] - 1
    keep = [first[cp] + np.minimum(best, width[cp] - 1)]
    second = partner[best]
    has = (second >= 0) & (width[cp] > 1)
    keep.append((first[cp] + second)[has])
    d = np.zeros(nVG)
    d[np.concatenate(keep)] = 1
    D = diags(d)
    G = (D @ G @ D).tocsr()
  if diagonal:
    lone = np.asarray((G > 0).sum(axis=0)).reshape(-1) == 1
    G =(diags((~lone).astype(float)) @ G).tocsr()
  G.eliminate_zeros()
  job.graph = G


def clusterVariantGraph(job, inflationPower=1.05, numIter=10, cutOff=1e-4, minCluster=4, includeSimilar=False, mergeSimilar=False,
                        pseudoDiagonal=True, sym=False, keepGraph=False, **graphOptions):
  """the variant graph clustered by the MCL algorithm generalised to the variants of a grain, the
  votes for the parent orientations; MATLAB's clusterVariantGraph of parentGrainReconstructor

    job.clusterVariantGraph()                                  # job.votes from the variant graph
    job.clusterVariantGraph(includeSimilar=True)               # similar variants share their score
    job.clusterVariantGraph(threshold=3*degree)                # the graph built first with these options
    job.calcParentFromVote(minProb=0.5)                        # the parents from the votes

  The variant graph is built by calcVariantGraph first when the job has none, with the
  keywords left over. Then numIter steps of MCL: the expansion, the inflation by
  inflationPower and the pruning below cutOff, the rows of the nodes of one grain normalised
  together. The score of every node of a child grain whose cluster has at least minCluster
  members becomes its vote, job.votes, the variants of every grain ordered by it. The graph is
  removed unless keepGraph. The job is changed in place.

  Input
    job - parentGrainReconstructor with its orientation relationship job.p2c
  Keywords
    inflationPower - the inflation power of MCL
    numIter - the number of iterations
    cutOff - the entries below it are pruned in every iteration
    minCluster - the least size of a cluster whose scores become votes
    includeSimilar - a variant takes 0.9 of the score of the variants within 5 degrees
    mergeSimilar - a variant takes the whole score of the variants within 5 degrees
    pseudoDiagonal - keep the links between the variants of one child grain
    sym - make the graph symmetric in every iteration, the geometric mean of both directions
    keepGraph - keep the clustered graph in job.graph
  References
    R. Hielscher, T. Nyyssönen, F. Niessen, A.A. Gazder, The variant graph approach to
        improved parent grain reconstruction, Materialia 22 (2022) 101399
  See also
    parentGrainReconstructor.calcVariantGraph, parentGrainReconstructor.calcParentFromVote,
    parentGrainReconstructor.clusterGraph, MaParentGrainReconstruction"""
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
