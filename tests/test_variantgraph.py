# M9: the graphs of a parent grain reconstruction, the variant graph of the Julia R4 and R4b
# and the grain graph route of MATLAB's GrainGraphBasedReconstruction, on the synthetic map
# of test_reconstruction
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mtex import calcGrains, degree, parentGrainReconstructor
from mtex.maps.variantgraph import mclComponents
from test_reconstruction import P2C, fitsParent, martensiteMap


@pytest.fixture(scope="module")
def plain():
  e = martensiteMap()
  return e, calcGrains(e, threshold=2 * degree)


def test_R4_the_variant_graph_and_its_clustering(plain):
  e, g = plain
  job = parentGrainReconstructor(e, g, P2C)
  nV = P2C.variants(parent=True).size
  job.calcVariantGraph()
  assert job.graph.shape == (6 * nV, 6 * nV) and job.hasVariantGraph
  job.clusterVariantGraph(minCluster=2)
  assert job.votes.parentId.shape == (6, nV) and np.all(job.votes.prob[:, 0] > 0.3)
  job.calcParentFromVote(minProb=0.1)
  job.mergeSimilar()
  assert len(job.grains) == 2 and np.max(fitsParent(job.grains)) < 2 * degree


def test_R4b_a_grain_that_is_a_parent_carries_the_graph_through_it(plain):
  e, g = plain
  job = parentGrainReconstructor(e, g, P2C)
  job.calcGBVotes()
  job.calcParentFromVote(minProb=0.3)
  job.revert([k for k in range(6) if k != 1])
  assert np.count_nonzero(job.isParent) == 1 and np.count_nonzero(job.isChild) == 5
  nV = P2C.variants(parent=True).size
  job.votes = None
  job.calcVariantGraph()
  assert job.graph.shape[0] == 5 * nV + 1
  # the node of the parent is joined to the children that fit it
  node = int(np.flatnonzero(job.isParent)[0]) * nV
  assert job.graph[node].nnz > 1
  job.clusterVariantGraph(minCluster=2)
  job.calcParentFromVote(minProb=0.1)
  job.mergeSimilar()
  assert len(job.grains) == 2 and np.max(fitsParent(job.grains)) < 2 * degree


def test_the_variant_graph_restricted_to_the_votes(plain):
  e, g = plain
  job = parentGrainReconstructor(e, g, P2C)
  job.calcGBVotes()
  job.calcVariantGraph()
  # every grain keeps its best variant alone, so a node row is empty unless voted for
  nV = P2C.variants(parent=True).size
  alive = np.flatnonzero(np.asarray(job.graph.sum(axis=1)).reshape(-1) > 0)
  assert len(np.unique(alive // nV)) <= 6 and np.all(np.isin(alive % nV + 1, job.votes.parentId[:, 0]))
  # includeSimilar lends a variant a tenth of the score of those within 5 degrees
  job.votes = None
  job.calcVariantGraph()
  job.clusterVariantGraph(minCluster=2, includeSimilar=True)
  assert job.votes.parentId.shape == (6, nV) and job.graph is None


def test_MCL_splits_two_cliques():
  A = np.zeros((6, 6))
  A[:3, :3] = 1
  A[3:, 3:] = 1
  A[2, 3] = A[3, 2] = 0.05
  M = mclComponents(csr_matrix(A), 1.6).toarray()
  assert np.all(M[:3, 3:] < 1e-3) and np.all(M[3:, :3] < 1e-3)


def test_the_grain_graph_route(plain):
  e, g = plain
  job = parentGrainReconstructor(e, g, P2C)
  job.calcGraph(threshold=2.5 * degree, tolerance=2.5 * degree)
  assert job.graph.shape == (6, 6) and job.hasGraph and "graph:" in repr(job)
  job.clusterGraph(inflationPower=1.6)
  job.calcParentFromGraph()
  assert job.graph is None and np.all(job.isParent)
  assert np.all(np.isfinite(job.grains.fit)) and np.max(job.grains.fit) < 2 * degree
  assert np.array_equal(np.sort(job.grains.clusterSize), [3, 3, 3, 3, 3, 3])
  job.mergeSimilar()
  assert len(job.grains) == 2 and np.max(fitsParent(job.grains)) < 2 * degree
  # a graph merged directly into parent grains
  job = parentGrainReconstructor(e, g, P2C)
  job.calcGraph(threshold=2.5 * degree, tolerance=2.5 * degree)
  job.clusterGraph()
  job.mergeByGraph()
  assert len(job.grains) == 2 and np.all(job.isParent) and np.max(fitsParent(job.grains)) < 2 * degree
  assert np.all(job.isTransformed) and np.array_equal(np.sort(job.numChildren), [3, 3])


def test_the_compiled_variant_angles_against_their_oracles():
  from mtex import orientation, setMTEXpref, getMTEXpref
  from mtex.geometry.quaternion import qmul
  from mtex.geometry.kernels import variantAngles
  from test_reconstruction import CSC, CSP
  rng = np.random.default_rng(3)
  o1, o2 = orientation.rand(7, CSC, rng=rng), orientation.rand(7, CSC, rng=rng)
  PV = P2C.variants(parent=True).data.reshape(-1, 4)
  G = CSP.sym.properGroup().rotations().data.reshape(-1, 4)
  m = (o1.inv() * o2).data.reshape(-1, 4)
  fast = variantAngles(m, PV, G)
  keep = getMTEXpref("kernels", "numba")
  setMTEXpref("kernels", "numpy")
  try:
    slow = variantAngles(m, PV, G)
  finally:
    setMTEXpref("kernels", keep)
  c1 = orientation._fromRotation(qmul(o1.data.reshape(-1, 1, 4), PV[None]), (CSP, o1.frameB)).reshape(7, 24, 1)
  c2 = orientation._fromRotation(qmul(o2.data.reshape(-1, 1, 4), PV[None]), (CSP, o2.frameB)).reshape(7, 1, 24)
  assert np.allclose(fast, slow, atol=1e-6) and np.allclose(fast, np.asarray(c1.angle(c2)), atol=1e-6)
