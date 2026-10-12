# calcTraces on maps and grains: lamellae of a known direction give it back
import numpy as np
import pytest

from mtex import EBSD, calcGrains, calcTraces, crystalFrame, degree, phaseList, rotation, smoothBoundary, xvector


# a square map of stripes of two variants, running along the angle ang
def stripes(ang, n=80, width=4):
  y, x = np.mgrid[0:n, 0:n].astype(float)
  s = -x * np.sin(ang) + y * np.cos(ang)
  v = (np.floor(s / width) % 2).astype(int).reshape(-1) + 1
  pos = np.stack([x, y, np.zeros_like(x)], -1).reshape(-1, 3)
  rot = rotation.byAxisAngle(xvector, np.where(v == 1, 0, 40 * degree))
  return EBSD(pos, rot, np.ones(n * n, dtype=np.int8), phaseList([crystalFrame("m-3m")])), v


def traceAngle(tr):
  return np.degrees(np.arctan2(tr.y, tr.x)) % 180


def axialError(a, b):
  d = np.abs(a - b) % 180
  return np.minimum(d, 180 - d)


@pytest.mark.parametrize("method, tol", [("Radon", 1.0), ("Fourier", 2.0)])
def test_the_traces_of_a_map(method, tol):
  for ang in (0, 30, 75, 120):
    e, v = stripes(ang * degree)
    tr, rel, cSize = calcTraces(e, v, method)
    assert tr.shape == (2,) and tr.antipodal and cSize.sum() == e.size
    assert np.all(axialError(traceAngle(tr), ang) < tol) and np.all(rel > 0.8)
  # one cluster of the whole map by default, and a cluster too small is left out
  tr, rel, cSize = calcTraces(e, method=method)
  assert tr.shape == (1,) and cSize[0] == e.size
  tr, rel, _ = calcTraces(e, v, method, minClusterSize=e.size)
  assert np.all(np.isnan(tr.x)) and np.all(rel == 0)


@pytest.mark.parametrize("method, tol", [("shape", 1.0), ("calliper", 2.0), ("hist", 2.0)])
def test_the_traces_of_grains(method, tol):
  for ang in (30, 120):
    e, v = stripes(ang * degree, n=120)
    grains = smoothBoundary(calcGrains(e, angle=5 * degree), 10)
    cid = np.zeros(len(grains), dtype=int)
    cid[e.grainId.reshape(-1) - 1] = v
    tr, rel, cSize = calcTraces(grains, cid, method)
    assert np.all(axialError(traceAngle(tr), ang) < tol) and np.all(rel > 0.5)
    assert cSize.sum() == e.size


def test_calcTraces_as_a_method():
  e, v = stripes(30 * degree)
  assert np.allclose(e.calcTraces(v)[0].data, calcTraces(e, v)[0].data)
  with pytest.raises(ValueError):
    calcTraces(e, v, "hist")


# a method of another name is refused also when no cluster is large enough to be traced
def test_calcTraces_refusesAnUnknownMethodWithoutClusters():
  e, v = stripes(0.0)
  with pytest.raises(ValueError):
    calcTraces(e, v, "hist", minClusterSize=e.size)


def test_the_members_of_a_cluster_are_its_rows_in_order():
  from mtex.maps.traces import _clusters, _members
  cid, sz, cSize = _clusters(np.array([2, 1, 2, 0, 1, 2]), 6, np.ones(6))
  assert [(tuple(int(i) for i in ic), r.tolist()) for ic, r in _members(cid, sz, cSize, 2)] == [((1,), [0, 2, 5])]
