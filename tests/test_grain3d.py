# grain3d: the polyhedra of a volume, their measures, selections, rotation and neighbours,
# against MATLAB master (bench/matlab/probe_grain3d.m)

import numpy as np
from scipy.sparse import csr_matrix

from mtex import EBSD3square, calcGrains, crystalFrame, degree, grain3Boundary, grain3d, phaseList, rotation, vector3d



def grains():
  r = np.zeros((3, 2, 2, 4))
  r[..., 0] = 1
  r[2] = rotation.byAxisAngle(vector3d(0, 0, 1), 30 * degree).data
  e = EBSD3square(rotation(r), np.ones((3, 2, 2), dtype=np.int8), phaseList([crystalFrame("m-3m", mineral="Iron")]), (1.0, 1.0, 1.0))
  return calcGrains(e, angle=10 * degree)


# the unit cube as one grain of six outward quadrilaterals
def cube():
  V = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], dtype=float)
  F = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
  cs = phaseList([crystalFrame("m-3m", mineral="Iron")])
  gB = grain3Boundary(V, F, np.array([[1, 0]] * 6), None, np.array([[1, -1]] * 6), np.tile([1.0, 0, 0, 0], (6, 1)), cs, None)
  from mtex import specimenFrame
  gB.frame = specimenFrame.default
  return grain3d([1], [1], cs, [1], [[1.0, 0, 0, 0]], [0.0], csr_matrix(np.ones((1, 6))), gB, specimenFrame.default)


def test_the_measures_of_two_boxes():
  g = grains()
  assert len(g) == 2 and list(g.numPixel) == [8, 4]
  assert np.allclose(g.volume, [8, 4]) and np.allclose(g.surfaceArea, [24, 16]) and np.allclose(g.surface(), [24, 16])
  assert np.allclose(g.centroid.data, [[0.5, 0.5, 0.5], [2, 0.5, 0.5]])
  assert np.allclose(g.extent, [-0.5, 2.5, -0.5, 1.5, -0.5, 1.5])
  assert list(g.numFaces) == [48, 32] and g.I_GF.shape == (2, len(g.boundary))
  s = np.asarray(g.I_GF.sum(axis=0)).reshape(-1)
  assert set(np.unique(s)) <= {0.0, 1.0}
  assert np.allclose(g.diameter, [np.sqrt(12), np.sqrt(9)])
  assert g.neighbors().tolist() == [[1, 2]] and list(g.numNeighbors) == [1, 1]
  assert "grain3d" in repr(g)


def test_a_selection_keeps_its_faces_and_its_outer_neighbours():
  g = grains()
  one = g[1]
  assert list(one.id) == [2] and np.allclose(one.volume, 4) and len(one.boundary) == 32
  assert list(g["id", 2].id) == [2] and len(g["Iron"]) == 2 and len(g[g.numPixel > 5]) == 1
  assert one.neighbors().shape == (0, 2) and one.neighbors(full=True).tolist() == [[1, 2]]


def test_rotate_turns_the_positions_and_the_orientations():
  g = grains()
  r = g.rotate(rotation.byAxisAngle(vector3d(0, 0, 1), 90 * degree))
  assert np.allclose(r.volume, g.volume) and np.allclose(r.centroid.data[1], [-0.5, 2, 0.5])
  # the orientations turn with the grains, 90 degree about z being a symmetry of m-3m
  assert np.allclose(np.asarray(r.meanOrientation.angle(g.meanOrientation)), 0, atol=1e-6)
  r20 = g.rotate(rotation.byAxisAngle(vector3d(0, 0, 1), 20 * degree))
  assert np.allclose(np.asarray(r20.meanOrientation.angle(g.meanOrientation)) / degree, 20)
  k = g.rotate(rotation.byAxisAngle(vector3d(0, 0, 1), 90 * degree), keepXY=True)
  assert np.allclose(k.centroid.data, g.centroid.data)


def test_a_polygonal_grain():
  c = cube()
  assert not c.boundary.isTriangulated and np.allclose(c.volume, 1) and np.allclose(c.centroid.data, 0.5)
  assert np.allclose(c.surfaceArea, 6) and np.allclose(c.boundary.perimeter, 4) and np.allclose(c.boundary.diameter, np.sqrt(2))
  a, b, cc = c.principalComponents()
  # the ellipsoid of a cube is a ball of the cube's volume
  assert np.allclose([a.norm(), b.norm(), cc.norm()], (3 / (4 * np.pi)) ** (1 / 3))
  t = c.triangulate()
  assert t.boundary.isTriangulated and len(t.boundary) == 12 and np.allclose(t.volume, 1) and np.all(t.boundary.id <= 6)


def test_IN100_as_MATLAB_measures_it(in100):
  g = in100[1]
  assert np.allclose(g.diameter[:5], [0.4330127, 0.61237244, 0.4330127, 1.1726039, 1.9202864])
  a, b, c = g.principalComponents()
  assert np.allclose(a.norm()[:5], [0.15508762, 0.31017525, 0.15508762, 0.47510537, 0.75848807])
  assert np.allclose(b.norm()[:5], [0.15508762, 0.15508762, 0.15508762, 0.4011621, 0.66585186])
  assert np.allclose(c.norm()[:5], [0.15508762, 0.15508762, 0.15508762, 0.27399941, 0.24373561])
  assert np.allclose(a.norm()[699:705], [2.5187008, 0.92907578, 2.9534468, 4.2148139, 2.4972178, 2.2592067])
  assert len(g.neighbors()) == 3899 and list(g.numNeighbors[:10]) == [2, 3, 1, 5, 4, 2, 1, 4, 4, 2]
  s = g[g.numPixel > 1000]
  assert len(s) == 226 and len(s.F) == 657170


def test_orientations_set_and_the_misorientations_follow():
  g = grains()
  cs = crystalFrame("6/mmm", mineral="Mg")
  from mtex import orientation
  o = orientation(rotation.byAxisAngle(vector3d(0, 0, 1), np.array([0, 20]) * degree), cs)
  g.meanOrientation = o
  assert g.CSList[g.phaseId[0]] is cs and np.allclose(np.asarray(g.meanOrientation.angle(o)), 0, atol=1e-9)
  gB = g.boundary[np.all(g.boundary.grainId > 0, axis=1)]
  assert np.allclose(np.asarray(gB.misorientation.angle()) / degree, 20) and np.all(gB.phaseId == g.phaseId[0])


def test_grains_from_faces():
  g = cube()
  h = grain3d.fromFaces(g.boundary.allV, g.boundary.F, g.I_GF, np.array([[1.0, 0, 0, 0]]), g.CSList)
  assert len(h) == 1 and h.phaseId[0] == 1 and np.allclose(h.volume, 1)
  assert np.all(h.boundary.grainId == [1, 0]) and np.all(h.boundary.phaseId == [1, -1])
