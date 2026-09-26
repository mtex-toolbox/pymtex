# the sections of 3D grains with a plane, against MATLAB master (bench/matlab/probe_grain3d.m)
import os

import numpy as np
import pytest

from mtex import EBSD3, EBSD3square, calcGrains, crystalFrame, degree, phaseList, plane3d, rotation, vector3d

DREAM3D = os.path.join(os.path.dirname(__file__), "..", "..", "master", "data", "EBSD", "SmallIN100_MeshStats.dream3d")


# a U of grain 1 around grain 2 in a 3 x 3 x 1 plate, grain 3 above: a section at mid height
# of the lower layer cuts grain 1 into one U and grain 2 into a square
def volume():
  r = np.zeros((3, 3, 2, 4))
  r[..., 0] = 1
  r[1, 1:, 0] = rotation.byAxisAngle(vector3d(0, 0, 1), 30 * degree).data
  r[:, :, 1] = rotation.byAxisAngle(vector3d(1, 0, 0), 30 * degree).data
  return EBSD3square(rotation(r), np.ones((3, 3, 2), dtype=np.int8), phaseList([crystalFrame("m-3m", mineral="Iron")]))


def test_a_section_through_three_grains():
  g = calcGrains(volume(), angle=10 * degree)
  assert len(g) == 3
  pl = plane3d(vector3d(0, 0, 1), vector3d(0, 0, 0))
  assert list(g.intersected(pl)) == [True, True, False]
  s = g.slice(pl)
  assert len(s) == 2 and np.isclose(s.area.sum(), 9)
  byId = dict(zip(s.prop["Id3d"], s.area))
  assert np.isclose(byId[g.id[g.numPixel == 7][0]], 7) and np.isclose(byId[g.id[g.numPixel == 2][0]], 2)
  # a plane through the middle of both layers along y cuts the U into two pieces
  t = g.slice(plane3d(vector3d(0, 1, 0), vector3d(0, 1.2, 0)))
  ids = list(t.prop["Id3d"])
  assert len(t) == 4 and np.isclose(t.area.sum(), 6) and ids.count(g.id[g.numPixel == 7][0]) == 2


@pytest.mark.skipif(not os.path.exists(DREAM3D), reason="the IN100 volume of ../master/data")
def test_IN100_sections():
  g = calcGrains(EBSD3.load(DREAM3D), angle=10 * degree)
  pl = plane3d(vector3d(0, 0, 1), vector3d(0, 0, -20))
  assert g.intersected(pl).sum() == 63
  s = g.slice(pl)
  # as MATLAB: 67 pieces of the 63 grains, the whole section, the same grains cut twice
  assert len(s) == 67 and np.isclose(s.area.sum(), 625) and np.isclose(s.perimeter.sum(), 907.5)
  assert sorted(s.prop["Id3d"])[:16] == [289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 302, 304]
  # a tilted section covers exactly the section of the box, 809.7134; MATLAB's pieces add up
  # to 814.52 there, more than the section holds
  t = g.slice(plane3d(vector3d(1, 1, 1), vector3d(-25, 22, -17)))
  assert np.isclose(t.area.sum(), 809.71343, atol=1e-4) and len(np.unique(t.prop["Id3d"])) == 124
