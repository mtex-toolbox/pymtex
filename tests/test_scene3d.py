# scene3d: what a 3D plot shows as arrays - the outer faces of grains, boundary faces, the
# slices of a volume, arrows, pictures of maps
import numpy as np

from mtex import EBSD3square, calcGrains, crystalFrame, degree, phaseList, rotation, vector3d
from mtex.plotting.scene3d import arrows3d, boundaryMesh, grainMesh, lines3d, mesh3d, picture3d, scene3d, slices3d, volumeSlices


# a 3 x 2 x 2 box of voxels, two grains of 8 and 4 voxels
def volume():
  r = np.zeros((3, 2, 2, 4))
  r[..., 0] = 1
  r[2] = rotation.byAxisAngle(vector3d(1, 0, 0), 30 * degree).data
  return EBSD3square(rotation(r), np.ones((3, 2, 2), dtype=np.int8), phaseList([crystalFrame("m-3m", mineral="Iron")]), (1.0, 1.0, 1.0),
                     prop={"v": np.arange(12.0).reshape(3, 2, 2)})


def test_a_mesh_keeps_the_vertices_its_faces_use():
  V = np.array([[0, 0, 0], [9, 9, 9], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
  m = mesh3d(V, [0, 4], [0, 2, 3, 4], (1, 0, 0))
  assert len(m) == 1 and m.V.shape == (4, 3) and not np.any(m.V == 9)
  T, face = m.triangles()
  assert T.shape == (2, 3) and list(face) == [0, 0]
  assert len(m.edges()) == 4 and m.faceColor.shape == (1, 3)


# the surface of the box, two triangles per voxel face; the four faces between the grains are inside
def test_grainMesh_isTheOuterFacesOfTheGrains():
  g = calcGrains(volume(), angle=10 * degree)
  m = grainMesh(g)
  assert len(m) == 2 * 2 * (3 * 2 + 3 * 2 + 2 * 2)
  lo, hi = m.bounds()
  assert np.allclose(lo, -0.5) and np.allclose(hi, [2.5, 1.5, 1.5])


# translucent faces show the faces inside too, grey
def test_grainMesh_translucentShowsTheInnerFacesGrey():
  g = calcGrains(volume(), angle=10 * degree)
  m = grainMesh(g)
  t = grainMesh(g, faceAlpha=0.5)
  assert len(t) == len(m) + 2 * 4 and np.sum(np.all(t.faceColor == 0.5, axis=1)) == 8


# one value per grain, and a grain without a value is not drawn
def test_grainMesh_leavesAGrainWithoutAValueOut():
  g = calcGrains(volume(), angle=10 * degree)
  v = grainMesh(g, np.array([1.0, np.nan]))
  # the grain of 2 x 2 x 2 voxels has 20 outer voxel faces
  assert len(v) == 2 * 20 and v.colorRange is not None


def test_grainMesh_takesAFaceColourByName():
  g = calcGrains(volume(), angle=10 * degree)
  assert np.allclose(grainMesh(g, faceColor="red").faceColor, [1, 0, 0])


def test_the_orientations_of_grains_colour_them_apart():
  g = calcGrains(volume(), angle=10 * degree)
  m = grainMesh(g, g.meanOrientation)
  assert len(np.unique(m.faceColor.round(6), axis=0)) == 2


def test_boundaryMesh_isLightBlueByDefault():
  gB = calcGrains(volume(), angle=10 * degree).boundary
  assert np.allclose(boundaryMesh(gB).faceColor, [0.678, 0.847, 0.902], atol=1e-3)


def test_boundaryMesh_byValueLeavesANaNOutAndSpansTheRest():
  gB = calcGrains(volume(), angle=10 * degree).boundary
  vals = np.arange(len(gB), dtype=float)
  vals[0] = np.nan
  m = boundaryMesh(gB, vals)
  assert len(m) == len(gB) - 1 and m.colorRange == (1.0, len(gB) - 1.0)


def test_boundaryMesh_byColourDrawsEveryFace():
  gB = calcGrains(volume(), angle=10 * degree).boundary
  assert len(boundaryMesh(gB, np.tile([0, 1, 0], (len(gB), 1)))) == len(gB)


def test_the_slices_of_a_volume():
  e = volume()
  s = volumeSlices(e, e.prop["v"])
  assert s.visible.shape == (3, 2, 2) and s.at == (1, 1, 1)
  rgb, vis, corners = s.plane(2)
  assert rgb.shape == (3, 2, 3) and vis.all()
  # the rectangle of the slice through the voxel centres at z = 1, across the voxels in x and y
  assert np.allclose(corners, [[-0.5, -0.5, 1], [2.5, -0.5, 1], [-0.5, 1.5, 1], [2.5, 1.5, 1]])
  lo, hi = s.bounds()
  assert np.allclose(lo, -0.5) and np.allclose(hi, [2.5, 1.5, 1.5])
  # the largest value takes the top of the colour map, the smallest the bottom
  assert s.colorRange == (0.0, 11.0)


def test_arrows3d_areShaftsAndConesAndSkipNaN():
  shafts, heads = arrows3d([[0, 0, 0], [1, 0, 0]], [[0, 0, 2], [np.nan, 0, 0]])
  assert len(shafts.segments) == 1 and np.allclose(shafts.segments[0], [[0, 0, 0], [0, 0, 1.5]])
  assert np.isclose(heads.V[:, 2].max(), 2) and len(heads) == 24


def test_arrows3d_antipodalIsAHeadlessLineThroughTheBase():
  both = arrows3d([[0, 0, 0]], [[1, 0, 0]], antipodal=True)
  assert len(both) == 1 and np.allclose(both[0].segments[0], [[-1, 0, 0], [1, 0, 0]])


def test_scene3d_spansItsItems():
  s = scene3d()
  s.add(lines3d([[[0, 0, 0], [1, 2, 3]]]))
  s.add(picture3d(np.ones((2, 2, 4)), [[-1, 0, 0], [0, 0, 0], [-1, 1, 0], [0, 1, 0]]))
  lo, hi = s.bounds()
  assert np.allclose(lo, [-1, 0, 0]) and np.allclose(hi, [1, 2, 3])


def test_scene3d_cameraLooksAlongAUnitDirectionWithUpPerpendicular():
  s = scene3d()
  s.add(lines3d([[[0, 0, 0], [1, 2, 3]]]))
  o, up = s.camera()
  assert np.isclose(np.linalg.norm(o), 1) and np.isclose(o @ up, 0)


def test_slices3d_boundsAreAnArray():
  assert isinstance(slices3d(np.ones((1, 1, 1, 3)), np.ones((1, 1, 1)), np.zeros(3), np.eye(3)).bounds()[0], np.ndarray)
