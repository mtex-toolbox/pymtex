# M4: the quadruple points a square grid invents, the Julia Q1 to Q7 but for the areas of M5
import numpy as np
import pytest

from mtex import EBSD, phaseList, crystalFrame, rotation, zvector, degree, angle, calcGrains, gbcAngle, decompose
from mtex.maps.quadruple import quadruplePoints, splitQuadruplePoints
from mtex.maps.segment import segment, segmentFrom
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


# a 6 x 6 map of one orientation with two pixels turned by 86 degrees that touch at a corner
def twinPixels():
  om = np.zeros((6, 6))
  om[2, 2] = om[3, 3] = 86.0
  return angleMap(om)


def test_Q1_a_diagonal_contact_is_one_grain_or_two():
  e = quadrants()
  g = calcGrains(e.copy(), angle=10 * degree, removeQuadruplePoints=False)
  assert len(g) == 4
  c = decompose(e)
  seg = segment(c, e, gbcAngle(10 * degree))
  verts, faces = quadruplePoints(c, seg)
  assert len(verts) == 1 and close(c.V[verts[0]], [1.5, 1.5], tol=1e-9) and faces.shape == (1, 4)
  q = calcGrains(e.copy(), angle=10 * degree)
  assert len(q) == 3 and q.numPixel.sum() == len(e) and q.numPixel.max() == 8


# the sorted lengths of the segments of both boundaries with no end within r of the point p
def farLengths(g, p, r):
  L = []
  for b in (g.boundary, g.innerBoundary):
    near = np.linalg.norm(np.asarray(b.V, float)[:, :2] - p, axis=1) < r
    L.append(b.segLength[~near[b.F].any(axis=1)])
  return np.sort(np.concatenate(L))


def test_Q3_the_split_moves_only_the_ends_of_the_four_segments_it_splits():
  e = quadrants()
  g = calcGrains(e.copy(), angle=10 * degree, removeQuadruplePoints=False)
  q = calcGrains(e.copy(), angle=10 * degree)
  p = np.array([1.5, 1.5])
  assert np.allclose(farLengths(g, p, 0.25), farLengths(q, p, 0.25))
  # the two triple points a quarter pixel apart, in the wedges of the cells not joined
  V = np.asarray(q.boundary.V, float)[np.unique(q.boundary.F), :2]
  near = V[np.linalg.norm(V - p, axis=1) < 0.25]
  assert len(near) == 2 and close(np.linalg.norm(near[0] - near[1]), 0.25) and close(np.abs(near - p), 0.125 / np.sqrt(2))


def test_the_pairing_that_merges_two_grains_wins_over_the_smaller_misorientation():
  # the two turned pixels are joined, though the pixels of the other diagonal are closer in
  # orientation to each other: those already lie in one grain
  q = calcGrains(twinPixels(), angle=10 * degree)
  assert len(q) == 2 and sorted(q.numPixel.tolist()) == [2, 34] and not np.any(q.boundary.vertexDegree() == 4)


def test_a_point_whose_diagonals_the_criterion_separates_stays_a_quadruple_point():
  e = angleMap(np.array([[0.0 if j < 2 else 30.0 for j in range(4)] if i < 2 else [60.0 if j < 2 else 90.0 for j in range(4)] for i in range(4)]))
  q = calcGrains(e, angle=10 * degree)
  assert len(q) == 4 and np.count_nonzero(q.boundary.vertexDegree() == 4) == 1


def test_merging_across_the_added_faces_is_the_segmentation_of_the_split_complex():
  rng = np.random.default_rng(3)
  e = angleMap(30.0 * rng.random((20, 20)))
  crit = gbcAngle(10 * degree)
  c = decompose(e)
  seg = segment(c, e, crit)
  c2, seg2 = splitQuadruplePoints(c, seg, e, crit)
  ref = segmentFrom(c2, e, seg2.connect)
  assert len(c2.F) > len(c.F) and seg2.nGrains == ref.nGrains and np.array_equal(seg2.grainId, ref.grainId)
  assert np.array_equal(seg2.isOuter, ref.isOuter) and np.array_equal(seg2.isInner, ref.isInner)


def test_Q4_the_pairing_is_chosen_by_the_measurements():
  # both diagonals are connected alike and differ by the same angle, so only the tie break
  # decides, and it reads the positions: the diagonal of positive slope wins
  for a, b in ((0.0, 40.0), (40.0, 0.0)):
    q = calcGrains(checker(a, b), angle=10 * degree)
    assert len(q) == 3
    big = np.argmax(q.numPixel)
    assert q.numPixel[big] == 8 and close(q.meanRotation[big].angle() / degree, a, tol=1e-6)


def test_Q5_a_hexagonal_map_has_none():
  e = quadrants(A=np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]]))
  g = calcGrains(e.copy(), angle=10 * degree, removeQuadruplePoints=False)
  q = calcGrains(e.copy(), angle=10 * degree)
  assert len(q) == len(g) and q.boundary.perimeter() == g.boundary.perimeter()


def test_Q6_a_map_full_of_them():
  rng = np.random.default_rng(7)
  e = angleMap(90.0 * rng.random((30, 30)))
  g = calcGrains(e.copy(), angle=10 * degree, removeQuadruplePoints=False)
  q = calcGrains(e.copy(), angle=10 * degree)
  # the points left are those where neither diagonal passes the criterion
  assert np.count_nonzero(q.boundary.vertexDegree() == 4) < np.count_nonzero(g.boundary.vertexDegree() == 4)
  assert len(q) < len(g) and q.numPixel.sum() == len(e) and close(q.area.sum(), g.area.sum())
  assert q.boundary.perimeter() < g.boundary.perimeter()
  # the walk order holds on every chain of the split boundary
  gB = q.boundary
  cid = gB.chainId
  same = cid[1:] == cid[:-1]
  assert np.all(gB.F[1:, 0][same] == gB.F[:-1, 1][same])


def test_Q7_single_precision():
  e = quadrants(dtype=np.float32)
  q = calcGrains(e.copy(), angle=10 * degree)
  assert len(q) == 3 and not np.any(q.boundary.vertexDegree() == 4) and q.boundary.V.dtype == np.float32


def test_the_wedge_site_is_the_one_site_two_faces_share():
  from mtex.maps.quadruple import _commonSite
  a, b = np.array([[1, 2], [1, 2], [1, 2]]), np.array([[2, 5], [7, 8], [2, 1]])
  assert np.array_equal(_commonSite(a, b), [2, -1, -1])
