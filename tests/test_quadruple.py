# M4: the quadruple points a square grid invents, the Julia Q1 to Q7 but for the areas of M5
import numpy as np
import pytest

from mtex import EBSD, phaseList, crystalFrame, rotation, zvector, degree, angle, calcGrains, gbcAngle, decompose
from mtex.maps.quadruple import quadruplePoints
from mtex.maps.segment import segment
from conftest import close

CS1 = crystalFrame("1", mineral="test")


# a square map whose pixel (i, j) carries a rotation about z of omega[i, j] degrees
def angleMap(omega, A=np.eye(2), dtype=np.float64):
  ni, nj = omega.shape
  i, j = np.meshgrid(np.arange(ni), np.arange(nj), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  pos = np.stack([A[0, 0] * i + A[0, 1] * j, A[1, 0] * i + A[1, 1] * j, np.zeros(ni * nj)], axis=-1).astype(dtype)
  rot = rotation.byAxisAngle(zvector, (omega[i, j] * degree).astype(dtype))
  return EBSD(pos, rot, np.ones(ni * nj, dtype=np.int8), phaseList(CS1))


# the four quadrants of a 4 x 4 map meeting at its centre: the two on the diagonal of
# positive slope differ by 3 degrees, the other two by more than any threshold
def quadrants(**kw):
  return angleMap(np.array([[0.0 if j < 2 else 30.0 for j in range(4)] if i < 2 else [60.0 if j < 2 else 3.0 for j in range(4)] for i in range(4)]), **kw)


# a checkerboard of 2 x 2 blocks: only the positions tell the two pairings apart
def checker(a, b):
  return angleMap(np.array([[a if (i < 2) == (j < 2) else b for j in range(4)] for i in range(4)]))


def test_Q1_a_diagonal_contact_is_one_grain_or_two():
  e = quadrants()
  g = calcGrains(e.copy(), angle=10 * degree)
  assert len(g) == 4
  c = decompose(e)
  seg = segment(c, e, gbcAngle(10 * degree))
  verts, faces = quadruplePoints(c, seg)
  assert len(verts) == 1 and close(c.V[verts[0]], [1.5, 1.5], tol=1e-9) and faces.shape == (1, 4)
  q = calcGrains(e.copy(), angle=10 * degree, removeQuadruplePoints=True)
  assert len(q) == 3 and q.numPixel.sum() == len(e) and q.numPixel.max() == 8


def test_Q2_no_four_way_contact_survives():
  e = quadrants()
  g = calcGrains(e.copy(), angle=10 * degree)
  q = calcGrains(e.copy(), angle=10 * degree, removeQuadruplePoints=True)
  assert np.any(g.boundary.vertexDegree() == 4) and not np.any(q.boundary.vertexDegree() == 4)


def test_Q3_the_geometry_is_left_where_it_was():
  e = quadrants()
  g = calcGrains(e.copy(), angle=10 * degree)
  q = calcGrains(e.copy(), angle=10 * degree, removeQuadruplePoints=True)
  # a segment between the two grains a split joins moves to the inner boundary
  assert close(q.boundary.perimeter() + q.innerBoundary.perimeter(), g.boundary.perimeter() + g.innerBoundary.perimeter(), tol=1e-6)
  Lg = np.sort(np.concatenate([g.boundary.segLength, g.innerBoundary.segLength]))
  Lq = np.sort(np.concatenate([q.boundary.segLength, q.innerBoundary.segLength]))
  assert len(Lq) == len(Lg) and np.max(np.abs(Lq - Lg)) < 1e-6


def test_Q4_the_pairing_is_chosen_by_the_measurements():
  # both diagonals are connected alike and differ by the same angle, so only the tie break
  # decides, and it reads the positions: the diagonal of positive slope wins
  for a, b in ((0.0, 40.0), (40.0, 0.0)):
    q = calcGrains(checker(a, b), angle=10 * degree, removeQuadruplePoints=True)
    assert len(q) == 3
    big = np.argmax(q.numPixel)
    assert q.numPixel[big] == 8 and close(q.meanRotation[big].angle() / degree, a, tol=1e-6)


def test_Q5_a_hexagonal_map_has_none():
  e = quadrants(A=np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]]))
  g = calcGrains(e.copy(), angle=10 * degree)
  q = calcGrains(e.copy(), angle=10 * degree, removeQuadruplePoints=True)
  assert len(q) == len(g) and q.boundary.perimeter() == g.boundary.perimeter()


def test_Q6_a_map_full_of_them():
  rng = np.random.default_rng(7)
  e = angleMap(90.0 * rng.random((30, 30)))
  g = calcGrains(e.copy(), angle=10 * degree)
  q = calcGrains(e.copy(), angle=10 * degree, removeQuadruplePoints=True)
  assert np.any(g.boundary.vertexDegree() == 4) and not np.any(q.boundary.vertexDegree() == 4)
  assert len(q) <= len(g) and q.numPixel.sum() == len(e)
  total = g.boundary.perimeter() + g.innerBoundary.perimeter()
  assert abs(q.boundary.perimeter() + q.innerBoundary.perimeter() - total) < 1e-6 * total
  assert q.boundary.perimeter() <= g.boundary.perimeter()
  # the walk order holds on every chain of the split boundary
  gB = q.boundary
  cid = gB.chainId
  same = cid[1:] == cid[:-1]
  assert np.all(gB.F[1:, 0][same] == gB.F[:-1, 1][same])


def test_Q7_single_precision():
  e = quadrants(dtype=np.float32)
  q = calcGrains(e.copy(), angle=10 * degree, removeQuadruplePoints=True)
  assert len(q) == 3 and not np.any(q.boundary.vertexDegree() == 4) and q.boundary.V.dtype == np.float32
