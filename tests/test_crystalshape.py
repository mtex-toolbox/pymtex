# G7: crystal shapes, the Julia crystalshape.jl numbers and the MATLAB CrystalShapes page
import numpy as np
import pytest

from mtex import Miller, crystalFrame, crystalShape, degree, innerFace, orientation, rotation, specimenFrame, vector3d, zvector
from mtex.geometry.crystalshape import CRYSTAL_SHAPES
from conftest import close


def test_a_cube_and_an_octahedron():
  cs = crystalFrame("m-3m", mineral="cube")
  cube = crystalShape(Miller(1, 0, 0, cs))
  assert cube.numFaces == 6 and cube.numVertices == 8 and len(cube.E) == 12 and np.all(np.sum(cube.F >= 0, axis=1) == 4)
  assert close(np.max(cube.V.norm()), 0.5) and close(cube.volume, (1 / np.sqrt(3)) ** 3, 1e-10) and np.allclose(cube.faceArea, 1 / 3)
  assert close(cube.diameter, 1.0, 1e-10) and cube.CS is cs and cube.frame is cs and len(cube) == 1
  assert np.allclose(cube.faceNormal().dot(cube.faceCenter, noSymmetry=True), 1 / (2 * np.sqrt(3)))
  # the edges are the pairs of vertices at the side of the cube
  assert np.allclose(np.linalg.norm(cube.V.data[cube.E[:, 0]] - cube.V.data[cube.E[:, 1]], axis=1), 1 / np.sqrt(3))
  # the sections through the centre: a square along z, a hexagon along the body diagonal
  assert len(innerFace(cube, zvector)) == 4 and len(cube.innerFace(vector3d(1, 1, 1))) == 6
  assert close(cube.exitDistance(Miller(1, 0, 0, cs)), 1 / (2 * np.sqrt(3)), 1e-10) and close(cube.exitDistance([1, 1, 1]), 0.5, 1e-10)
  octa = crystalShape(Miller(1, 1, 1, cs))
  assert octa.numFaces == 8 and octa.numVertices == 6 and np.all(np.sum(octa.F >= 0, axis=1) == 3)
  # a face too far out disappears: a truncated cube has 14 faces, a far octahedron none
  both = crystalShape(Miller([[1, 0, 0], [1, 1, 1]], cs) / np.array([1, 1.5]))
  assert both.numFaces == 14 and len(both.N) == 2
  assert crystalShape(Miller([[1, 0, 0], [1, 1, 1]], cs) / np.array([1, 10])).numFaces == 6
  with pytest.raises(ValueError):
    crystalShape(vector3d(1, 0, 0))
  r = repr(cube)
  assert "crystalShape" in r and "cube" in r and "faces: 6" in r and "vertices: 8" in r


def test_scaling_moving_and_rotating():
  cs, ss = crystalFrame("m-3m", mineral="cube"), specimenFrame("1")
  cube = crystalShape.cube(cs)
  assert close((2 * cube).volume, 8 * cube.volume, 1e-10) and close((cube * 0.5).diameter, 0.5, 1e-10) and close((cube / 2).diameter, 0.5, 1e-10)
  moved = cube + vector3d(1, 0, 0)
  assert np.allclose(moved.V[0].data, cube.V[0].data + [1, 0, 0]) and close(moved.volume, cube.volume, 1e-12)
  assert np.allclose((cube + [0, 0, 2]).V.data.mean(axis=0), [0, 0, 2]) and np.allclose((cube + [1, 1]).V.data.mean(axis=0), [1, 1, 0])
  # an orientation takes the shape into the specimen frame, the normals with the vertices
  o = orientation.byEuler(30 * degree, 0, 0, cs, ss)
  rc = o * cube
  assert rc.frame is ss and rc.CS is cs and close(rc.volume, cube.volume, 1e-12)
  assert np.allclose(rc.faceNormal()[0].data, (o * cube.faceNormal()[0]).data)
  assert rc.rotate(rotation.byAxisAngle(zvector, 10 * degree)).frame is ss
  # a list of orientations makes a list of shapes sharing the faces
  ol = orientation.byEuler(np.array([0, 30, 60]) * degree, 0, 0, cs, ss)
  L = ol * cube
  assert len(L) == 3 and L.V.shape == (3, 8) and L.faceArea.shape == (3, 6) and L.volume.shape == (3,) and L.F is cube.F
  assert np.allclose(L.volume, cube.volume) and L.faceCenter.shape == (3, 6) and L.faceNormal().shape == (3, 6)
  assert np.allclose(L[1].V.data, (ol[1] * cube).V.data) and L[1:].V.shape == (2, 8) and len(list(L)) == 3
  # a factor and a shift per shape, and a list from a single shape
  L2 = L * np.array([1, 2, 3]) + np.array([[0, 0], [10, 0], [20, 0]])
  assert np.allclose(L2.volume, cube.volume * np.array([1, 8, 27])) and np.allclose(L2.V.data[1].mean(axis=0), [10, 0, 0])
  assert np.allclose(L2.diameter, [1, 2, 3])
  assert (cube * np.array([1, 2])).V.shape == (2, 8) and cube.repmat(4).V.shape == (4, 8)
  assert (np.array([1, 2]) * cube).V.shape == (2, 8) and (vector3d(np.zeros((5, 3))) + cube).V.shape == (5, 8)
  assert (cube + vector3d(np.zeros((5, 3)), frame=cs)).V.shape == (5, 8)
  with pytest.raises(ValueError):
    cube + vector3d(1, 0, 0, frame=ss)
  with pytest.raises(ValueError):
    L.innerFace(zvector)
  # the diameter can be set
  d = crystalShape.cube(cs)
  d.diameter = 3
  assert close(d.diameter, 3, 1e-10)


def test_selecting_faces_and_the_hexagonal_prism():
  cs = crystalFrame("6/mmm", [3, 3, 4.7], mineral="Titanium")
  cS = crystalShape.hex(cs)
  assert cS.numVertices == 12 and cS.numFaces == 8 and len(cS.N) == 2
  # a form selects all its faces, a normal one
  assert cS[Miller(1, 0, -1, 0, cs).symmetrise()].numFaces == 6 and cS[Miller(0, 0, 0, 1, cs)].numFaces == 1
  assert cS.subSet(Miller(0, 0, 0, 1, cs)).numFaces == 1 and cS[Miller(0, 0, 0, 1, cs)].V is cS.V
  with pytest.raises(ValueError):
    cS[Miller(1, 1, -2, 3, cs)]
  # the sections through the centre: a prism plane cuts a rectangle, the basal and a
  # pyramidal plane hexagons
  assert len(cS.innerFace(Miller(1, 0, -1, 0, cs))) == 4 and len(cS.innerFace(Miller(0, 0, 0, 1, cs))) == 6
  assert len(cS.innerFace(Miller(0, 1, -1, 1, cs))) == 6
  # a crystal direction against a shape standing in the specimen does not fit
  o = orientation.byEuler(10 * degree, 20 * degree, 0, cs)
  with pytest.raises(ValueError):
    (o * cS).innerFace(Miller(0, 0, 0, 1, cs))
  assert len((o * cS).innerFace(o * Miller(0, 0, 0, 1, cs))) == 6
  assert crystalShape.tetr(crystalFrame("4/mmm", [1, 1, 0.7])).numFaces == 6


def test_the_quartz_page_and_the_habitus():
  cs = crystalShape.quartz().CS
  m, r, z = Miller(1, 0, -1, 0, cs), Miller(1, 0, -1, 1, cs), Miller(0, 1, -1, 1, cs)
  s2, x2 = Miller(1, 1, -2, 1, cs), Miller(5, 1, -6, 1, cs)
  cat = lambda *vs: vector3d(np.stack([v.data.reshape(3) for v in vs]), frame=cs)
  # the prism lies beyond the two rhombohedra: eight vertices, twelve faces
  cS = crystalShape(cat(m, r, z))
  assert cS.numVertices == 8 and cS.numFaces == 12 and len(cS.N) == 3
  # moved in, the prism truncates them; a smaller z makes the rhombohedra unequal
  assert crystalShape(cat(2 * m, r, z)).numFaces == 18
  a = crystalShape(cat(2 * m, r, 0.9 * z)).faceArea
  assert a.shape == (18,) and len(np.unique(np.round(a, 6))) == 3
  assert crystalShape(cat(2 * m, r, 0.9 * z, 0.7 * s2, 0.3 * x2)).numFaces > 18
  # the habitus derives the distances from the indices
  N = cat(m, r, z, s2, x2)
  h1 = crystalShape(N, 1, [1, 1, 1])
  h2 = crystalShape(N, 1.3, [1, 1.2, 1.1])
  assert h1.habitus == 1 and h2.extension == (1, 1.2, 1.1) and h1.numFaces < h2.numFaces
  assert "habitus" in repr(h2)


def test_the_catalogue():
  assert len(CRYSTAL_SHAPES) >= 45
  fo = crystalShape.forsterite()
  assert isinstance(fo, crystalShape) and fo.numFaces > 6 and fo.volume > 0 and fo.CS.mineral == "forsterite"
  q = crystalShape.quartz(simple=True)
  assert q.numFaces > 6 and q.CS.lattice.isTriHex and q.habitus == 1.2
  assert crystalShape.quartz("simple").numFaces == q.numFaces and crystalShape.quartz().numFaces != q.numFaces
  assert crystalShape.olivine().habitus == 2 and crystalShape.olivine().numFaces > 6
  # the shape on a phase's own frame, whose cell is the real one
  frame = crystalFrame("mmm", [4.756, 10.207, 5.98], mineral="Forsterite")
  own = crystalShape.forsterite(frame)
  assert own.CS is frame and own.numFaces == fo.numFaces
  with pytest.raises(ValueError):
    crystalShape.forsterite(crystalFrame("m-3m"))
  with pytest.raises(ValueError):
    crystalShape.halite(simple=True)
  assert not hasattr(crystalShape, "unobtainium")
  # every entry builds
  for k in CRYSTAL_SHAPES:
    name, simple = (k[:-7], True) if k.endswith("_simple") else (k, False)
    assert getattr(crystalShape, name)(simple=simple).numFaces >= 4
