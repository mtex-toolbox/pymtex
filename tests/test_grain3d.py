# grain3d: the polyhedra of a volume, their volume, centroid, surface and orientation
import numpy as np
import pytest

from mtex import EBSD3square, calcGrains, crystalFrame, degree, phaseList, rotation, vector3d


def grains():
  r = np.zeros((3, 2, 2, 4))
  r[..., 0] = 1
  r[2] = rotation.byAxisAngle(vector3d(0, 0, 1), 30 * degree).data
  e = EBSD3square(rotation(r), np.ones((3, 2, 2), dtype=np.int8), phaseList([crystalFrame("m-3m", mineral="Iron")]), (1.0, 1.0, 1.0))
  return calcGrains(e, angle=10 * degree)


def test_the_measures_of_two_boxes():
  g = grains()
  assert len(g) == 2 and list(g.numPixel) == [8, 4]
  assert np.allclose(g.volume, [8, 4]) and np.allclose(g.surfaceArea, [2 * (4 + 4 + 4), 2 * (4 + 2 + 2)])
  assert np.allclose(g.centroid.data, [[0.5, 0.5, 0.5], [2, 0.5, 0.5]])
  assert np.allclose(g.extent, [-0.5, 2.5, -0.5, 1.5, -0.5, 1.5])
  assert list(g.numFaces) == [48, 32] and g.I_GF.shape == (2, len(g.boundary))
  # every face of a grain is +1 on the outside of it: the column sums are +1 or 0
  s = np.asarray(g.I_GF.sum(axis=0)).reshape(-1)
  assert set(np.unique(s)) <= {0.0, 1.0}
  assert np.all(np.asarray(g.meanOrientation.angle(g.meanOrientation[0])) / degree < [1e-6, 30.0001])
  assert "grain3d" in repr(g)


def test_mixed_phases_refuse_one_mean():
  r = np.zeros((2, 2, 2, 4))
  r[..., 0] = 1
  ph = np.ones((2, 2, 2), dtype=np.int8)
  ph[1] = 2
  e = EBSD3square(rotation(r), ph, phaseList([crystalFrame("m-3m", mineral="A"), crystalFrame("6/mmm", mineral="B")]))
  g = calcGrains(e)
  with pytest.raises(ValueError):
    g.meanOrientation
