# M3: union-find and the segmentation, the Julia S1 and S8
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
  assert "grains: 4" in repr(seg)
