# G7: crystal shapes, the Julia crystalshape.jl numbers and the MATLAB CrystalShapes page
import numpy as np
import pytest

from mtex import Miller, crystalFrame, crystalShape, degree, innerFace, orientation, rotation, specimenFrame, vector3d, zvector
from mtex.geometry.crystalshape import CRYSTAL_SHAPES
from conftest import close

cs = crystalFrame("m-3m", mineral="cube")
ss = specimenFrame("1")
hx = crystalFrame("6/mmm", [3, 3, 4.7], mineral="Titanium")


def cube():
  return crystalShape(Miller(1, 0, 0, cs))


# three orientations about z applied to the cube: a list of shapes
def cubeList(c):
  ol = orientation.byEuler(np.array([0, 30, 60]) * degree, 0, 0, cs, ss)
  return ol, ol * c


def test_crystalShape_cubeHasSixFacesEightVerticesTwelveEdges():
  c = cube()
  assert c.numFaces == 6 and c.numVertices == 8 and len(c.E) == 12 and np.all(np.sum(c.F >= 0, axis=1) == 4)


def test_crystalShape_cubeMeasures():
  c = cube()
  assert close(np.max(c.V.norm()), 0.5) and close(c.volume, (1 / np.sqrt(3)) ** 3, 1e-10) and np.allclose(c.faceArea, 1 / 3)
  assert close(c.diameter, 1.0, 1e-10) and c.CS is cs and c.frame is cs and len(c) == 1


def test_crystalShape_faceCentresLieOnTheirPlanes():
  c = cube()
  assert np.allclose(c.faceNormal().dot(c.faceCenter, symmetry=False), 1 / (2 * np.sqrt(3)))


def test_crystalShape_edgesAreTheSidesOfTheCube():
  # the edges are the pairs of vertices at the side of the cube
  c = cube()
  assert np.allclose(np.linalg.norm(c.V.data[c.E[:, 0]] - c.V.data[c.E[:, 1]], axis=1), 1 / np.sqrt(3))


def test_innerFace_ofTheCubeIsASquareOrAHexagon():
  # the sections through the centre: a square along z, a hexagon along the body diagonal
  c = cube()
  assert len(innerFace(c, zvector)) == 4 and len(c.innerFace(vector3d(1, 1, 1))) == 6


def test_exitDistance_ofTheCube():
  c = cube()
  assert close(c.exitDistance(Miller(1, 0, 0, cs)), 1 / (2 * np.sqrt(3)), 1e-10) and close(c.exitDistance([1, 1, 1]), 0.5, 1e-10)


def test_crystalShape_octahedronHasEightTriangles():
  octa = crystalShape(Miller(1, 1, 1, cs))
  assert octa.numFaces == 8 and octa.numVertices == 6 and np.all(np.sum(octa.F >= 0, axis=1) == 3)


def test_crystalShape_aFaceTooFarOutDisappears():
  # a truncated cube has 14 faces, a far octahedron none
  both = crystalShape(Miller([[1, 0, 0], [1, 1, 1]], cs) / np.array([1, 1.5]))
  assert both.numFaces == 14 and len(both.N) == 2
  assert crystalShape(Miller([[1, 0, 0], [1, 1, 1]], cs) / np.array([1, 10])).numFaces == 6


def test_crystalShape_refusesNormalsWithoutACrystalFrame():
  with pytest.raises(ValueError):
    crystalShape(vector3d(1, 0, 0))


def test_crystalShape_repr():
  r = repr(cube())
  assert "crystalShape" in r and "cube" in r and "faces   : 6" in r and "vertices: 8" in r


def test_crystalShape_scaling():
  c = crystalShape.cube(cs)
  assert close((2 * c).volume, 8 * c.volume, 1e-10) and close((c * 0.5).diameter, 0.5, 1e-10) and close((c / 2).diameter, 0.5, 1e-10)


def test_crystalShape_movingByAVector():
  c = crystalShape.cube(cs)
  moved = c + vector3d(1, 0, 0)
  assert np.allclose(moved.V[0].data, c.V[0].data + [1, 0, 0]) and close(moved.volume, c.volume, 1e-12)
  assert np.allclose((c + [0, 0, 2]).V.data.mean(axis=0), [0, 0, 2]) and np.allclose((c + [1, 1]).V.data.mean(axis=0), [1, 1, 0])


def test_crystalShape_orientationTakesItIntoTheSpecimenFrame():
  # the normals with the vertices
  c = crystalShape.cube(cs)
  o = orientation.byEuler(30 * degree, 0, 0, cs, ss)
  rc = o * c
  assert rc.frame is ss and rc.CS is cs and close(rc.volume, c.volume, 1e-12)
  assert np.allclose(rc.faceNormal()[0].data, (o * c.faceNormal()[0]).data)
  assert rc.rotate(rotation.byAxisAngle(zvector, 10 * degree)).frame is ss


def test_crystalShape_listOfOrientationsMakesAListSharingTheFaces():
  c = crystalShape.cube(cs)
  ol, L = cubeList(c)
  assert len(L) == 3 and L.V.shape == (3, 8) and L.faceArea.shape == (3, 6) and L.volume.shape == (3,) and L.F is c.F
  assert np.allclose(L.volume, c.volume) and L.faceCenter.shape == (3, 6) and L.faceNormal().shape == (3, 6)
  assert np.allclose(L[1].V.data, (ol[1] * c).V.data) and L[1:].V.shape == (2, 8) and len(list(L)) == 3


def test_crystalShape_aFactorAndAShiftPerShape():
  c = crystalShape.cube(cs)
  _, L = cubeList(c)
  L2 = L * np.array([1, 2, 3]) + np.array([[0, 0], [10, 0], [20, 0]])
  assert np.allclose(L2.volume, c.volume * np.array([1, 8, 27])) and np.allclose(L2.V.data[1].mean(axis=0), [10, 0, 0])
  assert np.allclose(L2.diameter, [1, 2, 3])


def test_crystalShape_aListFromASingleShape():
  c = crystalShape.cube(cs)
  assert (c * np.array([1, 2])).V.shape == (2, 8) and c.repmat(4).V.shape == (4, 8)
  assert (np.array([1, 2]) * c).V.shape == (2, 8) and (vector3d(np.zeros((5, 3))) + c).V.shape == (5, 8)
  assert (c + vector3d(np.zeros((5, 3)), frame=cs)).V.shape == (5, 8)


def test_crystalShape_refusesAShiftInTheSpecimenFrame():
  with pytest.raises(ValueError):
    crystalShape.cube(cs) + vector3d(1, 0, 0, frame=ss)


def test_innerFace_refusesAList():
  _, L = cubeList(crystalShape.cube(cs))
  with pytest.raises(ValueError):
    L.innerFace(zvector)


def test_crystalShape_diameterCanBeSet():
  d = crystalShape.cube(cs)
  d.diameter = 3
  assert close(d.diameter, 3, 1e-10)


def test_crystalShape_hexagonalPrism():
  cS = crystalShape.hex(hx)
  assert cS.numVertices == 12 and cS.numFaces == 8 and len(cS.N) == 2


def test_crystalShape_aFormSelectsAllItsFacesANormalOne():
  cS = crystalShape.hex(hx)
  assert cS[Miller(1, 0, -1, 0, hx).symmetrise()].numFaces == 6 and cS[Miller(0, 0, 0, 1, hx)].numFaces == 1
  assert cS.subSet(Miller(0, 0, 0, 1, hx)).numFaces == 1 and cS[Miller(0, 0, 0, 1, hx)].V is cS.V


def test_crystalShape_selectingAMissingFaceRaises():
  with pytest.raises(ValueError):
    crystalShape.hex(hx)[Miller(1, 1, -2, 3, hx)]


def test_innerFace_ofTheHexagonalPrism():
  # a prism plane cuts a rectangle, the basal and a pyramidal plane hexagons
  cS = crystalShape.hex(hx)
  assert len(cS.innerFace(Miller(1, 0, -1, 0, hx))) == 4 and len(cS.innerFace(Miller(0, 0, 0, 1, hx))) == 6
  assert len(cS.innerFace(Miller(0, 1, -1, 1, hx))) == 6


def test_innerFace_ofAShapeInTheSpecimenNeedsASpecimenDirection():
  # a crystal direction against a shape standing in the specimen does not fit
  cS = crystalShape.hex(hx)
  o = orientation.byEuler(10 * degree, 20 * degree, 0, hx)
  with pytest.raises(ValueError):
    (o * cS).innerFace(Miller(0, 0, 0, 1, hx))
  assert len((o * cS).innerFace(o * Miller(0, 0, 0, 1, hx))) == 6


def test_crystalShape_tetragonalPrism():
  assert crystalShape.tetr(crystalFrame("4/mmm", [1, 1, 0.7])).numFaces == 6


# the quartz forms of the MATLAB page
def quartzForms():
  qs = crystalShape.quartz().CS
  m, r, z = Miller(1, 0, -1, 0, qs), Miller(1, 0, -1, 1, qs), Miller(0, 1, -1, 1, qs)
  s2, x2 = Miller(1, 1, -2, 1, qs), Miller(5, 1, -6, 1, qs)
  cat = lambda *vs: vector3d(np.stack([v.data.reshape(3) for v in vs]), frame=qs)
  return m, r, z, s2, x2, cat


def test_crystalShape_quartzPrismBeyondTheRhombohedra():
  # eight vertices, twelve faces
  m, r, z, s2, x2, cat = quartzForms()
  cS = crystalShape(cat(m, r, z))
  assert cS.numVertices == 8 and cS.numFaces == 12 and len(cS.N) == 3


def test_crystalShape_quartzPrismMovedInTruncatesTheRhombohedra():
  # a smaller z makes the rhombohedra unequal
  m, r, z, s2, x2, cat = quartzForms()
  assert crystalShape(cat(2 * m, r, z)).numFaces == 18
  a = crystalShape(cat(2 * m, r, 0.9 * z)).faceArea
  assert a.shape == (18,) and len(np.unique(np.round(a, 6))) == 3
  assert crystalShape(cat(2 * m, r, 0.9 * z, 0.7 * s2, 0.3 * x2)).numFaces > 18


def test_crystalShape_habitusDerivesTheDistancesFromTheIndices():
  m, r, z, s2, x2, cat = quartzForms()
  N = cat(m, r, z, s2, x2)
  h1 = crystalShape(N, 1, [1, 1, 1])
  h2 = crystalShape(N, 1.3, [1, 1.2, 1.1])
  assert h1.habitus == 1 and h2.extension == (1, 1.2, 1.1) and h1.numFaces < h2.numFaces
  assert "habitus" in repr(h2)


def test_CRYSTAL_SHAPES_catalogueSize():
  assert len(CRYSTAL_SHAPES) >= 45


def test_crystalShape_forsterite():
  fo = crystalShape.forsterite()
  assert isinstance(fo, crystalShape) and fo.numFaces > 6 and fo.volume > 0 and fo.CS.mineral == "forsterite"


def test_crystalShape_quartzSimple():
  q = crystalShape.quartz(simple=True)
  assert q.numFaces > 6 and q.CS.lattice.isTriHex and q.habitus == 1.2
  assert crystalShape.quartz("simple").numFaces == q.numFaces and crystalShape.quartz().numFaces != q.numFaces


def test_crystalShape_olivine():
  assert crystalShape.olivine().habitus == 2 and crystalShape.olivine().numFaces > 6


def test_crystalShape_onAPhasesOwnFrame():
  # the shape on a phase's own frame, whose cell is the real one
  frame = crystalFrame("mmm", [4.756, 10.207, 5.98], mineral="Forsterite")
  own = crystalShape.forsterite(frame)
  assert own.CS is frame and own.numFaces == crystalShape.forsterite().numFaces


def test_crystalShape_catalogueRefusesAFrameOfAnotherGroup():
  with pytest.raises(ValueError):
    crystalShape.forsterite(crystalFrame("m-3m"))


def test_crystalShape_catalogueRefusesAMissingSimpleForm():
  with pytest.raises(ValueError):
    crystalShape.halite(simple=True)


def test_crystalShape_catalogueHasNoOtherNames():
  assert not hasattr(crystalShape, "unobtainium")


@pytest.mark.parametrize("k", sorted(CRYSTAL_SHAPES))
def test_crystalShape_everyCatalogueEntryBuilds(k):
  name, simple = (k[:-7], True) if k.endswith("_simple") else (k, False)
  assert getattr(crystalShape, name)(simple=simple).numFaces >= 4


@pytest.mark.parametrize("group, cell, constructor, vertices, faces", [
  ("6/mmm", [3, 3, 5], crystalShape.hex, 12, 8),
  ("m-3m", [3, 3, 3], crystalShape.cube, 8, 6),
])
def test_float32_lattice_angles_keep_the_crystal_shape(group, cell, constructor, vertices, faces):
  exact = crystalFrame(group, cell, intern=False)
  angles = np.asarray(exact.lattice.defaultAngles).astype(np.float32).astype(float)
  imported = crystalFrame(group, cell, angles, intern=False)
  reference, shape = constructor(exact), constructor(imported)
  assert (shape.numVertices, shape.numFaces) == (vertices, faces)
  np.testing.assert_allclose(shape.volume, reference.volume, rtol=1e-12, atol=0)
