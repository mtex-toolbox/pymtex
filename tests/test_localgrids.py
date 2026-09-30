import numpy as np
import pytest

from mtex import localOrientationGrid, localS2Grid, orientation, crystalFrame, Miller, vector3d, degree, angle, unique

cs = crystalFrame("m-3m")
cs2 = crystalFrame("m-3m", [3, 3, 3], mineral="other")


# a ball of 60 degrees around an orientation, with and without the symmetry
@pytest.fixture(scope="module")
def ball():
  c = orientation.byAxisAngle(vector3d.Z, 10 * degree, cs)
  full = localOrientationGrid(c, maxAngle=60 * degree, resolution=7.5 * degree, respectSymmetry=False)
  kept = localOrientationGrid(c, maxAngle=60 * degree, resolution=7.5 * degree)
  return c, full, kept


def test_localOrientationGrid_liesInTheBall(ball):
  c, full, _ = ball
  assert np.all(angle(full, c, symmetry=False) <= 60 * degree + 1e-10)


def test_localOrientationGrid_symmetryKeepsFewerDistinctNodes(ball):
  _, full, kept = ball
  assert kept.size < full.size and unique(kept).size == kept.size


def test_localOrientationGrid_keptSetCoversTheBall(ball):
  # every orientation of the ball has an equivalent within the resolution of the kept set
  _, full, kept = ball
  assert np.max(np.min(angle(kept[None, :], full[:, None]), axis=1)) < 1.05 * 7.5 * degree


def test_localOrientationGrid_smallBallIsNotFolded():
  c = orientation.byAxisAngle(vector3d.Z, 10 * degree, cs)
  assert localOrientationGrid(c, maxAngle=20 * degree, resolution=5 * degree).size == localOrientationGrid(c, maxAngle=20 * degree, resolution=5 * degree, respectSymmetry=False).size


def test_localOrientationGrid_aroundTheIdentityOfAFrame():
  assert localOrientationGrid(cs, 20 * degree, resolution=5 * degree).CS is cs


def test_localOrientationGrid_betweenTwoCubicFramesFoldsToAWedge():
  # the identity between two cubic frames is fixed by 24 pairs: the ball folds to a wedge
  m = localOrientationGrid(cs, cs2, maxAngle=15 * degree, resolution=3 * degree)
  mf = localOrientationGrid(cs, cs2, maxAngle=15 * degree, resolution=3 * degree, respectSymmetry=False)
  assert unique(m).size == m.size and m.size < mf.size / 12
  assert np.max(np.min(angle(m[None, :], mf[:, None]), axis=1)) < 1.01 * 3 * degree


def test_localS2Grid_capAround111FoldsToASixth():
  # the stabiliser 3m folds it to a sixth
  cap = localS2Grid(Miller(1, 1, 1, cs), maxTheta=15 * degree, resolution=3 * degree)
  capf = localS2Grid(Miller(1, 1, 1, cs), maxTheta=15 * degree, resolution=3 * degree, respectSymmetry=False)
  assert cap.frame is cs and unique(cap).size == cap.size and capf.size / 8 < cap.size < capf.size / 4.5
  assert np.max(np.min(angle(cap[None, :], capf[:, None]), axis=1)) < 1.01 * 3 * degree


def test_localS2Grid_capAroundAGeneralDirectionIsFoldedAndCovered():
  cap2 = localS2Grid(Miller(0.1, 0, 1, cs), maxTheta=20 * degree, resolution=3 * degree)
  cap2f = localS2Grid(Miller(0.1, 0, 1, cs), maxTheta=20 * degree, resolution=3 * degree, respectSymmetry=False)
  assert unique(cap2).size == cap2.size and cap2.size < cap2f.size
  assert np.max(np.min(angle(cap2[None, :], cap2f[:, None]), axis=1)) < 1.01 * 3 * degree
