# CLASSIX clustering of directions, axes and orientations, the hierarchical merging and the
# clusters by the modes of a density (docs/plans/functions.md F6, after the Julia test)
import numpy as np
import pytest

from mtex import calcCluster, classix, crystalFrame, degree, orientation, rotation, vector3d, xvector, yvector, zvector
from mtex.analysis.cluster import knnRadius

rng = np.random.default_rng(1)


def clouds(centers, n, spread=0.02):
  return vector3d(np.concatenate([c + spread * rng.standard_normal((n, 3)) for c in centers])).normalize()


# three directions 90 degrees apart, each a tight cloud
def threeDirections():
  return clouds([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], 200)


# four tight clouds of orientations about the centres oc
@pytest.fixture(scope="module")
def fourClouds():
  cs = crystalFrame("432")
  oc = orientation.byEuler(np.array([0, 40, 80, 20]) * degree, np.array([0, 30, 60, 80]) * degree, np.array([0, 10, 50, 70]) * degree, cs)
  rots = rotation.cat(*[rotation(oc[k].data) * rotation.byAxisAngle(vector3d.rand(100, rng=rng), 1.5 * degree * np.abs(rng.standard_normal(100)))
                        for k in range(4)])
  return cs, oc, rots, orientation(rots, cs)


# the angle in degrees of every centre to the nearest true centre
def deviation(center, oc):
  return np.rad2deg(center.reshape(-1, 1).angle(oc.reshape(1, -1)).min(axis=1))


def test_calcCluster_findsTheCloudsOfDirections():
  v = threeDirections()
  cId, center = calcCluster(v, radius=0.15, minPoints=20)
  assert cId.shape == v.shape and cId.min() == 0 and cId.max() == 2 and center.size == 3
  assert np.all(np.rad2deg(center.reshape(-1, 1).angle(vector3d.cat(xvector, yvector, zvector).reshape(1, -1)).min(axis=1)) < 3)
  assert np.all(np.bincount(cId) == 200)


def test_calcCluster_defaultsToTheCLASSIXEstimates():
  # the radius and the minimum count default to the CLASSIX estimates
  v = threeDirections()
  cId2, _ = calcCluster(v)
  assert cId2.max() == 2 and np.array_equal(v.calcCluster()[0], cId2)


def test_calcCluster_collapsesAntipodalDirectionsToAxes():
  # antipodal: x, -x, y, -y collapse to two axes
  va = clouds([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]], 150)
  va.antipodal = True
  cIdA, centerA = calcCluster(va, radius=0.15, minPoints=20)
  assert cIdA.max() == 1 and centerA.antipodal and centerA.size == 2


def test_classix_labelsFromZeroAndDissolvesSmallClusters():
  # the raw algorithm: labels from zero, small clusters dissolved, a radius is positive
  X = np.concatenate([rng.standard_normal((50, 2)), rng.standard_normal((50, 2)) + 10, [[100.0, 100.0]]])
  labels = classix(X, 1.0, minPts=5)
  assert labels.shape == (101,) and labels.max() == 1 and labels[-1] in (0, 1)
  assert classix(X, 1.0).max() == 2 and classix(np.zeros((0, 2)), 1.0).size == 0
  with pytest.raises(ValueError):
    classix(X, 0)
  assert knnRadius(X) > 0 and knnRadius(X[:5]) == 1.0


def test_classix_dissolvesInBlocksAsAtOnce(monkeypatch):
  # the small clusters dissolved a few groups at a time give the labels of one block
  import mtex.analysis.cluster as cluster
  X = np.concatenate([rng.standard_normal((300, 2)), rng.standard_normal((300, 2)) + 6, rng.random((60, 2)) * 30])
  whole = classix(X, 0.5, minPts=20)
  monkeypatch.setattr(cluster, "DISSOLVE_BLOCK", 7)
  assert np.array_equal(classix(X, 0.5, minPts=20), whole)


def test_calcCluster_classixFindsTheCloudsOfOrientations(fourClouds):
  cs, oc, _, ori = fourClouds
  # CLASSIX on the embedding
  cId, center = calcCluster(ori, method="classix", radius=0.1, minPoints=15)
  assert cId.max() == 3 and center.size == 4 and np.all(deviation(center, oc) < 5) and center.CS is cs
  # the radius CLASSIX estimates may split a cloud of this tightness; the call runs
  assert calcCluster(ori, "classix")[1].size >= 4


def test_calcCluster_hierarchicalMergesByCountOrAngle(fourClouds):
  cs, oc, rots, ori = fourClouds
  # the hierarchical merging, by count or by angle, and on a binned large list
  cId, center = calcCluster(ori, method="hierarchical", numCluster=4)
  assert cId.max() == 3 and np.all(deviation(center, oc) < 5)
  assert calcCluster(ori, method="hierarchical", maxAngle=10 * degree)[1].size == 4
  big = orientation(rotation.cat(rots, rots, rots), cs)
  cId, center = calcCluster(big, method="hierarchical", maxAngle=10 * degree)
  assert center.size == 4 and cId.shape == big.shape and np.all(deviation(center, oc) < 5)


def test_calcCluster_byDefaultTakesTheModesOfTheDensity(fourClouds):
  _, oc, _, ori = fourClouds
  # the modes of the density, MATLAB's default, with the far points dropped
  cId, center = calcCluster(ori)
  assert center.size == 4 and np.all(deviation(center, oc) < 5) and np.nanmax(cId) == 3
  assert ori.calcCluster(numCluster=2)[1].size == 2


@pytest.mark.parametrize("method", ["odf", "classix", "hierarchical"])
def test_calcCluster_labelsEveryValidOrientationAndNoNaNOne(fourClouds, method):
  ori = fourClouds[3]
  d = ori.data.copy()
  d[:3] = np.nan
  cId, center = calcCluster(orientation(rotation(d), ori.CS), method=method, numCluster=None if method == "classix" else 4)
  # the NaN orientations get no cluster, every other id is a cluster or NaN, never another number
  ok = np.isnan(cId[3:]) | ((cId[3:] >= 0) & (cId[3:] < center.size))
  assert np.all(np.isnan(cId[:3])) and np.all(ok)


def test_calcCluster_refusesAnUnknownMethod(fourClouds):
  ori = fourClouds[3]
  with pytest.raises(ValueError):
    calcCluster(ori, method="mars")
  with pytest.raises(ValueError):
    calcCluster(vector3d.rand(10, rng=rng), method="odf")


def test_calcCluster_hierarchicalMergesDirections():
  # directions merge hierarchically as well
  v = clouds([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], 100)
  cId, center = calcCluster(v, method="hierarchical", numCluster=3)
  assert center.size == 3 and np.all(np.bincount(cId) == 100)
