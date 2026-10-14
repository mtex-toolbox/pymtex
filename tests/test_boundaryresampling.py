# M4: the boundary coarsened, simplified and refined, the Julia B14 to B17
import numpy as np
import pytest

from mtex import EBSD, phaseList, crystalFrame, rotation, zvector, degree, calcGrains, setMTEXpref, getMTEXpref, reduce
from mtex import reduceBoundary, simplifyBoundary, refineBoundary
from mtex.maps.boundaryresampling import keepShapeChains, sharedVertices, NO_PIXEL
from mtex.maps.polygon import signedArea2
from conftest import close
from test_boundary import boundaryMap, boundaryOf, diskCircle, SQ4, CS


# an n x n map with one pixel in the middle turned away from the rest
def pixelMap(n):
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  pos = np.stack([i, j, np.zeros(n * n)], axis=-1).astype(float)
  rot = rotation.byAxisAngle(zvector, np.where((i == n // 2) & (j == n // 2), 30 * degree, 0.0))
  return EBSD(pos, rot, np.ones(n * n, dtype=np.int8), phaseList(CS))


def closedArea(gB):
  return signedArea2(gB.V[gB.F[:, 0]]) / 2


def test_B14_coarsening_keeps_every_nth_vertex_and_the_topology_with_it():
  gB = boundaryOf(boundaryMap(SQ4, 2, 10))[0]
  assert len(gB) == 120
  for n in (2, 5):
    r = gB.reduce(n)
    assert len(r) == 120 // n
    # merging segments rather than dropping them keeps the length exactly, since every
    # chain of this map is straight
    assert close(r.perimeter(), 120)
    # the junctions stay where they are, so the network is the same one
    assert r.chainId.max() == 7 and r.triplePoints.isempty
    assert set(r.F.reshape(-1)) <= set(gB.F.reshape(-1))
    cid = r.chainId
    same = cid[1:] == cid[:-1]
    assert np.all(r.F[1:, 0][same] == r.F[:-1, 1][same])
    # a segment that swallowed a run runs between no one pair of pixels
    assert np.all(r.ebsdId == NO_PIXEL)
  assert gB.reduce(1) is gB and reduce(gB, 2).size == 60


def test_B15_simplifying_removes_the_staircase_and_nothing_else():
  gB = boundaryOf(boundaryMap(SQ4, 2, 10))[0]
  # four border sides with a corner in the middle of each, and four arms
  s = gB.simplify()
  assert len(s) == 12 and close(s.perimeter(), 120) and s.triplePoints.isempty
  # a single grain, whose corners are no junctions of its own boundary: the pair of grains
  # changes there, which is what keeps the vertex
  s1 = gB.selectByGrainId([1]).simplify()
  assert len(s1) == 4 and close(s1.segLength, 10)
  # a tolerance of nothing coarsens nothing
  assert gB.simplify(0) is gB and gB.simplify().size == 12
  # a digital circle becomes a polygon of a sixth of the segments: a regular n-gon of radius
  # R stays within R(1 - cos(pi/n)) of its circle, so nine sides already meet the tolerance
  # and the staircase asks for a few more
  cir = diskCircle(12, 57)[0]
  sc = cir.simplify()
  assert 10 <= len(sc) <= 20
  # and the area it encloses moves by less than the tolerance times the perimeter
  assert abs(closedArea(sc) - 441) < 12


def test_the_numba_and_the_numpy_simplification_agree():
  cir = diskCircle(12, 57)[0]
  gB = boundaryOf(boundaryMap(SQ4, 3, 7))[0]
  before = getMTEXpref("kernels", "numba")
  try:
    out = {}
    for kernels in ("numba", "numpy"):
      setMTEXpref("kernels", kernels)
      out[kernels] = [keepShapeChains(b, eps) for b in (cir, gB) for eps in (0.3, 1 / np.sqrt(2), 2.0)]
  finally:
    setMTEXpref("kernels", before)
  for a, b in zip(out["numba"], out["numpy"]):
    assert np.array_equal(a, b)


def test_B16_refining_splits_every_segment_and_keeps_its_vertices():
  gB = boundaryOf(boundaryMap(SQ4, 2, 10))[0]
  r = gB.refine(0.5)
  assert len(r) == 240 and close(r.segLength, 0.5) and close(r.perimeter(), 120)
  assert r.chainId.max() == 7 and r.triplePoints.isempty
  # every vertex is still there, the new ones appended
  assert set(gB.F.reshape(-1)) <= set(r.F.reshape(-1))
  assert np.array_equal(r.V[:len(gB.V)], gB.V)
  assert all(np.count_nonzero(r.chainId == c) in (20, 40) for c in range(8))
  # a piece lies within one original segment and keeps its pair of pixels
  assert np.all(r.ebsdId[r.grainId.min(axis=1) > 0] >= 0)
  # the pieces lie on the old boundary, so a closed chain keeps its area
  cir = diskCircle(12, 57)[0]
  rc = cir.refine(0.5)
  assert len(rc) == 2 * len(cir) and close(closedArea(rc), closedArea(cir))
  # and the corners a simplification kept, so the boundary keeps its length
  sc = cir.simplify()
  assert close(sc.refine(0.5).perimeter(), sc.perimeter())
  # the default spacing is half the median segment
  assert len(cir.refine()) == len(rc)
  # a spacing longer than every segment leaves the boundary as it is
  assert len(gB.refine(100)) == len(gB)


def test_B17_a_grain_list_coarsens_both_of_its_boundaries_at_once():
  # with a low angle threshold the cross between the blocks is an inner boundary that ends
  # on the outer one, at vertices neither may dissolve
  g = calcGrains(boundaryMap(SQ4, 2, 10), angle=[90 * degree, 5 * degree])
  assert len(g) == 1
  shared = sharedVertices(g)
  assert len(shared) == 4 and np.array_equal(g.sharedVertices(), shared)
  for h in (g.simplifyBoundary(), g.reduceBoundary(4), g.refineBoundary(0.5), simplifyBoundary(g), refineBoundary(g)):
    assert np.array_equal(sharedVertices(h), shared)
    assert close(h.area, [400]) and close(h.boundary.perimeter(), 80) and close(h.innerBoundary.perimeter(), 40)
    # one pool of vertices, or the two boundaries can no longer be laid end to end
    assert h.boundary.V is h.innerBoundary.V
  assert len(g.simplifyBoundary().boundary) == 8
  assert len(g.refineBoundary(0.5).innerBoundary) == 80
  # a grain of a single pixel keeps a polygon rather than collapsing onto a line, though a
  # tolerance of the size of the pixel may take half its area
  p = calcGrains(pixelMap(11), angle=5 * degree)
  assert close(np.sort(p.area), [1, 120])
  assert np.all(p.simplifyBoundary().area > 0) and np.all(p.reduceBoundary(4).area > 0)
  # simplifying keeps the corners of the map, so the grains still tile it; reducing thins
  # a corner like a straight run and the outline shrinks
  assert close(p.simplifyBoundary().area.sum(), 121)
  assert p.reduceBoundary(4).area.sum() < 121
  # the ids, the phases and the properties are carried, the loops traced again
  p.score = [1, 2]
  q = p.simplifyBoundary()
  assert np.array_equal(q.id, p.id) and np.array_equal(q.score, [1, 2]) and q.boundarySize.sum() < p.boundarySize.sum()


def test_a_closed_chain_with_fewer_than_three_kept_heads_keeps_its_thirds():
  from mtex.maps.boundaryresampling import keepClosedChains
  gB = diskCircle(5, 16)[0]
  keep = keepClosedChains(gB, np.zeros(len(gB), dtype=bool))
  s, e = np.flatnonzero(gB.isChainStart), np.flatnonzero(gB.isChainEnd)
  expected = np.zeros(len(gB), dtype=bool)
  for a, b in zip(s[gB.isClosed[s]], e[gB.isClosed[s]]):
    expected[a + np.unique(np.rint(np.linspace((b - a + 1) / 3, b - a + 1, 3)).astype(int)) - 1] = True
  assert expected.sum() >= 3 and np.array_equal(keep, expected)
