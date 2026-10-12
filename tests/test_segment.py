# M3: union-find and the segmentation, the Julia S1 and S8
import os

import numpy as np
import pytest

from mtex import decompose, gbcAngle, degree
from mtex.maps.segment import components, segment, faceConnectivity, segmentFrom
from mtex.geometry.kernels import _jit
from test_calcgrains import blockMap, SQ3

BACKENDS = ["numpy"] + (["numba"] if _jit() else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_S1_union_find(backend):
  roots = components(10, [0, 1, 4], [1, 2, 5], backend=backend)
  assert roots[0] == roots[1] == roots[2] and roots[4] == roots[5] and roots[0] != roots[4]
  assert len(np.unique(roots)) == 7 and roots[3] != roots[0]
  # no edges, and an edge repeated
  assert len(np.unique(components(5, [], [], backend=backend))) == 5
  assert len(np.unique(components(5, [0, 0, 1], [1, 1, 0], backend=backend))) == 4
  rng = np.random.default_rng(1)
  a, b = rng.integers(0, 2000, 3000), rng.integers(0, 2000, 3000)
  r = components(2000, a, b, backend=backend)
  assert np.all(r[a] == r[b])


@pytest.mark.parametrize("backend", BACKENDS)
def test_S8_the_segmentation_marks_the_faces_it_separates(backend):
  e = blockMap(SQ3, 2, 5)
  c = decompose(e)
  seg = segment(c, e, gbcAngle(5 * degree), backend)
  assert seg.nGrains == 4 and len(seg.connect) == len(c)
  # a face is a boundary of the map, of a grain, or of neither
  assert not np.any(seg.isInner & seg.isOuter) and seg.isOuter.sum() > 0
  assert np.array_equal(np.unique(seg.grainId), np.arange(1, 5))
  # with a low angle threshold added, the boundaries between blocks stay inside
  seg2 = segment(c, e, gbcAngle([90 * degree, 5 * degree]), backend)
  assert seg2.nGrains == 1 and seg2.isInner.sum() > 0
  # the criterion is evaluated once per face and the segmentation runs on it
  connect = faceConnectivity(c, e, gbcAngle(5 * degree))
  assert len(connect) == len(c) and np.array_equal(segmentFrom(c, e, connect, backend).grainId, seg.grainId)
  assert "grains     : 4" in repr(seg)


# Markov clustering of the links (MATLAB's calcGrains(ebsd, 'mcl', [p maxIt])): the grains are
# the components of what it keeps, and a pair it keeps no link for is a boundary
def test_S9_the_segmentation_through_markov_clustering():
  from scipy.sparse import coo_matrix
  from scipy.sparse.csgraph import connected_components
  from mtex import gbcSoft
  from mtex.maps.variantgraph import mclComponents
  e = blockMap(SQ3, 3, 6, step=1.5 * degree)
  c = decompose(e)
  crit = gbcSoft(1 * degree, 0.5 * degree)
  seg = segment(c, e, crit, mcl=(1.4, 4))
  connect = faceConnectivity(c, e, crit)
  p1, p2 = c.site2pixel[c.cells[:, 0]], c.site2pixel[c.cells[:, 1]]
  inside, join = (p1 >= 0) & (p2 >= 0), (p1 >= 0) & (p2 >= 0) & (connect > 0)
  lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
  M = mclComponents(coo_matrix((connect[join], (lo[join], hi[join])), shape=(e.size, e.size)), 1.4, 4).tocsr()
  n, labels = connected_components(M, directed=False)
  assert seg.nGrains == n
  same = labels[:, None] == labels[None, :]
  assert np.array_equal(seg.grainId[:, None] == seg.grainId[None, :], same)
  cut = np.asarray(M[lo[inside], hi[inside]]).reshape(-1) == 0
  g1, g2 = seg.grainId[p1[inside]], seg.grainId[p2[inside]]
  assert np.array_equal(seg.isInner[inside], (g1 == g2) & cut)
  # a scalar is the inflation with MATLAB's four iterations
  assert np.array_equal(segment(c, e, crit, mcl=1.4).grainId, seg.grainId)


LOCAL = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data")


# the counts of MATLAB's doc/Grains/GrainReconstructionMCL.m, read through bench/matlab/probe_mcl.m
@pytest.mark.skipif(not os.path.isfile(os.path.join(LOCAL, "EBSD", "EMSphinx.h5")), reason="the MTEX data folder is not here")
def test_S9_markov_clustering_counts_as_MATLAB():
  from mtex import calcGrains, gbcSoft, inpolygon, mtexdata
  ebsd = mtexdata("EMSphinx", root=LOCAL, download=False)["Iron fcc"]
  ebsd = ebsd[inpolygon(ebsd, [40, 30, 80, 60])]
  soft = lambda: gbcSoft(1 * degree, 0.5 * degree)
  # every reconstruction on its own copy: the cull of minPixel leaves its mark on the map
  counts = [(19, 126, dict(angle=10 * degree)), (19, 96, dict(angle=10 * degree, mcl=(1.2, 4))),
            (40, 3104, dict(criterion=soft())), (45, 919, dict(criterion=soft(), mcl=(1.2, 4)))]
  for nGrains, nInner, kw in counts:
    g = calcGrains(ebsd.copy(), minPixel=10, **kw)
    assert (g.size, g.innerBoundary.size) == (nGrains, nInner)
  assert [calcGrains(ebsd.copy(), criterion=soft(), minPixel=10, mcl=(p, 4)).size for p in (1.1, 1.4, 1.8)] == [45, 46, 52]


def test_labels_count_from_one_in_the_order_values_first_appear():
  from mtex.maps.segment import _firstAppearanceLabels
  labels, n = _firstAppearanceLabels(np.array([7, 3, 7, 9, 3]))
  assert np.array_equal(labels, [1, 2, 1, 3, 2]) and n == 3 and labels.dtype == np.int32
