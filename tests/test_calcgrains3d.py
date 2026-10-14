# grain reconstruction in a volume, against MATLAB master (bench/matlab/probe_calcgrains3d.m)
import os

import numpy as np
import pytest

from mtex import EBSD3, EBSD3square, calcGrains, crystalFrame, degree, phaseList, rotation, vector3d

DREAM3D = os.path.join(os.path.dirname(__file__), "..", "..", "master", "data", "EBSD", "SmallIN100_MeshStats.dream3d")


# a 4 x 3 x 2 volume of two grains split along x, a lone voxel of a third orientation
def twoGrains():
  r = np.zeros((4, 3, 2, 4))
  r[..., 0] = 1
  r[2:] = rotation.byAxisAngle(vector3d(0, 0, 1), 30 * degree).data
  r[0, 0, 0] = rotation.byAxisAngle(vector3d(1, 0, 0), 20 * degree).data
  return EBSD3square(rotation(r), np.ones((4, 3, 2), dtype=np.int8), phaseList([crystalFrame("m-3m", mineral="Iron")]), (1.0, 2.0, 0.5))


def test_the_grains_and_their_faces():
  e = twoGrains()
  g = calcGrains(e, angle=10 * degree)
  assert len(g) == 3 and sorted(g.numPixel) == [1, 11, 12]
  # grains numbered by their last voxel in MATLAB's order: the lone voxel first
  assert e.grainId[0, 0, 0] == 1 and e.grainId[3, 2, 1] == 3
  # every grain a closed surface with outward normals: the volumes are the voxels'
  assert np.allclose(g.volume, g.numPixel * 1.0 * 2.0 * 0.5)
  assert np.allclose(g.centroid.data[g.numPixel == 1], [0, 0, 0])
  # the surface of the box and the faces between the grains, two triangles each
  box = 2 * (3 * 2 + 4 * 2 + 4 * 3)
  inner = 3 * 2 + 3
  assert len(g.boundary) == 2 * (box + inner) and np.all(g.boundary.area > 0)
  gB = g.boundary[np.all(g.boundary.grainId > 0, axis=1)]
  assert len(gB) == 2 * inner and np.all(gB.phaseId == 1) and np.all(gB.ebsdId >= 0)
  assert np.allclose(np.unique(np.round(np.asarray(gB.misorientation.angle()) / degree, 6)), [20, 30])
  assert np.allclose(g.surfaceArea.sum(), 2 * g.boundary.area[np.all(g.boundary.grainId > 0, axis=1)].sum() + g.boundary.area[g.boundary.grainId[:, 1] == 0].sum())


# a part cut out of a volume has the grains of a volume of its own: they lie where its voxels do
def test_the_grains_of_a_part_of_a_volume_lie_at_its_voxels():
  e = twoGrains()
  part = e[1:4, 1:3, 1:]
  g, whole = calcGrains(part, angle=10 * degree), calcGrains(EBSD3square(part.rotations, part.phaseId, part.CSList, (1.0, 2.0, 0.5),
                                                                          origin=part.pos.data[0, 0, 0]), angle=10 * degree)
  assert np.allclose(g.centroid.data, whole.centroid.data) and np.allclose(g.boundary.allV, whole.boundary.allV)


def test_minPixel_turns_a_lone_voxel_into_a_notIndexed_grain():
  e = twoGrains()
  g = calcGrains(e, angle=10 * degree, minPixel=2)
  assert len(g) == 3 and e.phaseId[0, 0, 0] == 0 and g.phaseId[e.grainId[0, 0, 0] - 1] == 0


def test_a_list_with_a_hole_segments_around_the_padding():
  # the missing voxel is padding, its faces are faces of the surface
  f = twoGrains()
  sub = f[np.arange(f.size) != 5]
  h = calcGrains(sub, angle=10 * degree)
  assert h.numPixel.sum() == 23 and np.allclose(h.volume.sum(), 23)
  assert sub.grainId.shape == (23,) and np.all(sub.grainId > 0)


def test_IN100_as_MATLAB_reconstructs_it(in100):
  e, g = in100
  assert len(g) == 751 and len(g.F) == 748886 and len(g.boundary.idV) == 343561
  assert list(g.numPixel[:10]) == [1, 2, 1, 14, 33, 32, 6, 32, 44, 7]
  assert (e.grainId[0, 0, 0], e.grainId[49, 29, 69]) == (142, 649)
  assert np.isclose(g.volume.sum(), 15625) and int(np.all(g.boundary.grainId > 0, axis=1).sum()) == 628886
  assert np.allclose(g.GOS[:5] / degree, [0, 0.237183, 0, 0.48779, 0.410571], atol=1e-5)
  a = np.asarray(g.boundary[np.all(g.boundary.grainId > 0, axis=1)].misorientation.angle()) / degree
  assert np.isclose(a.mean(), 46.337339, atol=1e-5) and np.isclose(a.max(), 62.203832, atol=1e-5)
  assert np.allclose(g.centroid.data[0], [-37.000004, 17.75, -29.25], atol=1e-5)
  m = g.meanOrientation[:3]
  from mtex import orientation
  ref = orientation(rotation(np.array([[0.5885705, -0.39679621, 0.69542289, -0.11191308], [0.64912626, -0.55708348, 0.51796979, 0.0006153904],
                                       [0.86928923, -0.49407266, 0.0032280125, 0.014765492]])), m.CS)
  assert np.all(np.asarray(m.angle(ref)) < 1e-6)
  e = EBSD3.load(DREAM3D)
  g = calcGrains(e, angle=5 * degree, minPixel=10)
  assert len(g) == 809 and len(g.F) == 757554 and (e.grainId[0, 0, 0], e.grainId[49, 29, 69]) == (149, 700)


def test_the_grains_a_file_stored():
  e = twoGrains()
  # two stored grains, the lone voxel of another orientation inside the first, and a voxel of no grain
  e.grainId = np.ones(e.shape, dtype=np.int64)
  e.grainId[2:] = 2
  e.grainId[1, 0, 0] = 0
  g = calcGrains(e, 'grainId')
  assert sorted(g.numPixel[g.isIndexed]) == [e.grainId[:2].size - 1, e.grainId[2:].size]
  assert np.count_nonzero(~g.isIndexed) == 1 and e.phaseId[1, 0, 0] == 0
