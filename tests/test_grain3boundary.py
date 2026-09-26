# grain3Boundary: the faces of the grains of a volume, their sides, normals and incidences
import numpy as np

from mtex import EBSD3square, calcGrains, crystalFrame, degree, phaseList, rotation, vector3d


def boundary():
  r = np.zeros((2, 1, 1, 4))
  r[..., 0] = 1
  r[1] = rotation.byAxisAngle(vector3d(0, 0, 1), 30 * degree).data
  e = EBSD3square(rotation(r), np.ones((2, 1, 1), dtype=np.int8), phaseList([crystalFrame("m-3m", mineral="Iron")]))
  return calcGrains(e, angle=10 * degree).boundary


def test_the_faces_of_two_voxels():
  gB = boundary()
  # one square between the voxels, ten on the outside, two triangles each
  assert len(gB) == 22 and np.allclose(gB.area, 0.5) and len(gB.idV) == 12
  inner = np.all(gB.grainId > 0, axis=1)
  assert inner.sum() == 2 and np.all(gB.grainId[inner] == [1, 2])
  # the normal of a face between the grains points from the first side to the second
  assert np.allclose(gB.N.data[inner], [1, 0, 0]) and np.allclose(gB.centroid.data[inner][:, 0], 0.5)
  assert np.all(gB.ebsdId[~inner, 1] == -1) and np.all(gB.phaseId[~inner, 1] == -1)
  assert np.allclose(np.asarray(gB[inner].misorientation.angle()) / degree, 30)
  # the incidences
  assert gB.I_FG.shape == (22, 2) and gB.I_FG.sum() == 24 and gB.I_VF.shape == (12, 22)
  assert np.all(np.asarray(gB.I_VG.sum(axis=0)).reshape(-1) == 8)
  assert gB.hasPhase(1, 1).sum() == 2 and "grain3Boundary" in repr(gB)
