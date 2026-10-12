# grain3Boundary: the faces of the grains of a volume, their sides, normals, measures and
# incidences, against MATLAB master (bench/matlab/probe_grain3d.m)

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
  assert len(gB) == 22 and gB.isTriangulated and np.allclose(gB.area, 0.5) and len(gB.idV) == 12
  inner = np.all(gB.grainId > 0, axis=1)
  assert inner.sum() == 2 and np.all(gB.grainId[inner] == [1, 2])
  assert np.allclose(gB.N.data[inner], [1, 0, 0]) and np.allclose(gB.centroid.data[inner][:, 0], 0.5)
  assert np.all(gB.ebsdId[~inner, 1] == -1) and np.all(gB.phaseId[~inner, 1] == -1)
  assert np.allclose(np.asarray(gB[inner].misorientation.angle()) / degree, 30)
  assert np.allclose(gB.perimeter, 2 + np.sqrt(2)) and np.allclose(gB.diameter, np.sqrt(2))
  assert gB.I_FG.shape == (22, 2) and gB.I_FG.sum() == 24 and gB.I_VF.shape == (12, 22)
  assert np.all(np.asarray(gB.I_VG.sum(axis=0)).reshape(-1) == 8)
  assert gB.hasPhase(1, 1).sum() == 2 and gB.hasGrain(1, 2).sum() == 2 and "grain3Boundary" in repr(gB)


def test_the_edges_and_node_types_of_two_voxels():
  gB = boundary()
  E, F2E = gB.edges()
  assert E.shape == (31, 2) and F2E.shape == (22, 3)
  # every vertex is on the surface of the box and touches one or two grains
  assert set(np.unique(gB.nodeType())) == {11, 12}


def test_flip_swaps_the_sides_and_inverts_the_misrotation():
  gB = boundary()
  inner = np.flatnonzero(np.all(gB.grainId > 0, axis=1))
  f = gB.flip(inner)
  assert np.all(f.grainId[inner] == [2, 1]) and np.allclose(f.misrotation.data[inner][:, 1:], -gB.misrotation.data[inner][:, 1:])
  assert np.allclose(gB.flipN(inner).N.data[inner], [-1, 0, 0])


def test_flipN_reverses_the_corners_of_a_face_once_however_often_named():
  gB = boundary()
  assert np.array_equal(gB.flipN([3]).F[3], gB.F[3][::-1]) and np.array_equal(gB.flipN([3, 3]).F, gB.F)


def test_the_fan_of_a_quadrilateral_is_two_triangles_from_its_first_corner():
  from mtex.maps3d.grain3boundary import grain3Boundary
  gB = grain3Boundary(np.eye(4, 3), [[0, 1, 2, 3]], [[1, 0]], None, [[1, -1]], [[1.0, 0, 0, 0]], phaseList([crystalFrame("m-3m")]), None)
  T, face = gB._fanIndex()
  assert np.array_equal(T, [[0, 1, 2], [0, 2, 3]]) and np.array_equal(face, [0, 0])


def test_a_selection_keeps_the_ids_and_the_faces():
  gB = boundary()
  inner = np.flatnonzero(np.all(gB.grainId > 0, axis=1))
  s = gB[inner]
  assert len(s) == 2 and list(s.id) == list(inner + 1) and np.array_equal(s.F, gB.F[inner])


def test_IN100_as_MATLAB_measures_it(in100):
  gB = in100[1].boundary
  assert np.allclose(gB.perimeter[:3], 0.85355339) and np.isclose(gB.perimeter.sum(), 639214.184468)
  assert list(np.bincount(gB.nodeType())) == [0, 0, 240232, 39657, 3513, 153, 4, 0, 0, 0, 0, 47124, 11815, 1036, 27]
  assert len(gB.edges()[0]) == 1092216
