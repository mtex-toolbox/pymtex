# M4: grain boundaries in walk order, the Julia B1 to B13
import numpy as np
import pytest

from mtex import (EBSD, phaseList, crystalFrame, orientation, rotation, vector3d, zvector, degree, angle, calcGrains, gbcAngle, decompose)
from mtex.maps.boundary import chainOrder, buildBoundary, orderWalk
from mtex.maps.segment import segment
from conftest import close

CS = crystalFrame("m-3m", mineral="test")
SQ4 = np.eye(2)
HEX4 = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])


def _cross2(a, b):
  return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


# a map of nb x nb blocks of bs x bs pixels, each block one grain
def boundaryMap(A, nb, bs):
  n = nb * bs
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  pos = np.stack([A[0, 0] * i + A[0, 1] * j, A[1, 0] * i + A[1, 1] * j, np.zeros(n * n)], axis=-1)
  rot = rotation.byAxisAngle(zvector, (i // bs + (j // bs) * nb) * 20 * degree)
  return EBSD(pos, rot, np.ones(n * n, dtype=np.int8), phaseList(CS))


def boundaryOf(e, theta=5 * degree):
  c = decompose(e)
  s = segment(c, e, gbcAngle(theta))
  return buildBoundary(c, s, e), c, s


# a round grain of radius R in a map of n x n pixels, its boundary a digital circle, the
# round grain to the left of the walk everywhere
def diskCircle(R, n):
  c = (n - 1) / 2
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  inside = (i - c) ** 2 + (j - c) ** 2 <= R ** 2
  pos = np.stack([i, j, np.zeros(n * n)], axis=-1).astype(float)
  rot = rotation.byAxisAngle(zvector, np.where(inside, 0.0, 30 * degree))
  e = EBSD(pos, rot, np.ones(n * n, dtype=np.int8), phaseList(CS))
  gB, _, s = boundaryOf(e)
  d = s.grainId[np.flatnonzero(inside)[0]]
  cir = gB.selectByGrainId([d])
  return cir.flip(cir.grainId[:, 1] == d), np.array([c, c])


def twoPhaseMap(a, b):
  i, j = np.meshgrid(np.arange(20), np.arange(20), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  pos = np.stack([i, j, np.zeros(400)], axis=-1).astype(float)
  return EBSD(pos, rotation.byAxisAngle(zvector, np.zeros(400)), np.where(i < 10, 1, 2).astype(np.int8), phaseList(a, b))


def test_B1_chainOrder_walks_an_edge_list():
  # an open chain of four edges given in scrambled order
  F = np.array([[2, 3], [0, 1], [3, 4], [1, 2]])
  perm, entry = chainOrder(F, 5)
  assert len(perm) == 4 and np.array_equal(np.sort(perm), np.arange(4))
  walk = [(F[k, 0], F[k, 1]) if entry[k] == 1 else (F[k, 1], F[k, 0]) for k in perm]
  assert all(walk[i][1] == walk[i + 1][0] for i in range(3))
  # a closed loop has no terminal and is cut at its lowest vertex
  G = np.array([[0, 1], [1, 2], [2, 0]])
  p2, e2 = chainOrder(G, 3)
  w2 = [(G[k, 0], G[k, 1]) if e2[k] == 1 else (G[k, 1], G[k, 0]) for k in p2]
  assert all(w2[i][1] == w2[i + 1][0] for i in range(2)) and w2[-1][1] == w2[0][0]
  assert len(chainOrder(np.zeros((0, 2), dtype=int), 0)[0]) == 0
  # a junction, an isolated vertex and a disconnected cycle: the adjacency keeps increasing
  # edge order when several walks start at the junction
  H = np.array([[1, 2], [1, 0], [1, 3], [5, 6], [6, 7], [7, 5]])
  p3, e3 = chainOrder(H, 10)
  assert np.array_equal(p3, [1, 0, 2, 3, 4, 5]) and np.array_equal(e3, [1, 2, 1, 1, 1, 1])


def test_B2_the_walk_order_is_an_invariant_of_the_boundary():
  for A in (SQ4, HEX4):
    gB = boundaryOf(boundaryMap(A, 2, 10))[0]
    cid = gB.chainId
    same = cid[1:] == cid[:-1]
    # inside a chain, the head of one segment is the tail of the next, and no chain runs through a junction
    assert np.all(gB.F[1:, 0][same] == gB.F[:-1, 1][same])
    assert not np.any(gB.isJunction[gB.F[1:, 0][same]])
    assert gB.isChainStart.sum() == cid.max() == gB.isChainEnd.sum()
    cs = gB.chainSize
    assert all(cs[k] == np.sum(cid == cid[k]) for k in range(len(cid))) and np.sum(np.bincount(cid)[1:]) == len(gB)


def test_B3_the_first_grain_lies_to_the_left_of_the_walk():
  e = boundaryMap(SQ4, 2, 10)
  gB = boundaryOf(e)[0]
  P = gB.plane.to2d(e.pos.data)
  inner = np.flatnonzero(gB.ebsdId[:, 0] >= 0)
  assert len(inner)
  v1 = gB.V[gB.F[inner, 0]]
  assert np.all(_cross2(gB.V[gB.F[inner, 1]] - v1, P[gB.ebsdId[inner, 0]] - v1) > 0)


def test_B4_a_block_map_has_the_boundaries_it_must():
  # four grains of 10 x 10 on a 20 x 20 map: the outer border is 80 long and cut at four
  # triple points, the inner cross 40 and cut at the quadruple point
  gB = boundaryOf(boundaryMap(SQ4, 2, 10))[0]
  assert close(gB.perimeter(), 120) and gB.chainId.max() == 8
  assert sorted(set(gB.chainSize.tolist())) == [10, 20] and gB.isJunction.sum() == 5
  tp = gB.triplePoints
  assert tp.isempty and tp.grainId.shape == (0, 3)
  assert np.all(gB.grainId[:, 1] > 0) and np.sum(gB.grainId[:, 0] == 0) == 80
  assert "notIndexed" in repr(gB) and "segments" in repr(gB)
  # the length rounded in the scan unit, as MATLAB writes it
  assert " µm  " in repr(gB) and "e+" not in repr(gB)


def test_B5_lengths_midpoints_and_directions():
  e = boundaryMap(SQ4, 2, 10)
  gB = boundaryOf(e)[0]
  assert close(gB.segLength, 1) and close(gB.segLength.sum(), gB.perimeter())
  assert gB.midPoint.size == len(gB) and gB.midPoint.frame is e.frame and gB._mid2d.shape == (len(gB), 2)
  d = gB.direction
  assert len(d) == len(gB) and d.frame is e.frame and d.antipodal
  # the arc length restarts with every chain and ends at the chain's length
  a = gB.arcLength
  assert close(a[gB.isChainStart], gB.segLength[gB.isChainStart]) and close(a[gB.isChainEnd], gB.chainSize[gB.isChainEnd])
  assert close(gB.chainLength, gB.chainSize)
  assert np.sum(gB.chainV == -1) == gB.chainId.max() - 1
  assert len(gB.allV) == len(gB.V) and gB.allV.frame is e.frame


def test_B6_the_misorientation_carries_the_exchange_symmetry():
  e = boundaryMap(SQ4, 2, 10)
  gB = boundaryOf(e)[0]
  k = np.flatnonzero((gB.grainId[:, 0] > 0) & (gB.grainId[:, 1] > 0))[0]
  mo = gB[k].misorientation
  assert mo.antipodal and mo.frameA is CS and mo.frameB is CS and mo.angle() > 5 * degree
  b = np.flatnonzero(gB.grainId[:, 0] == 0)[0]
  with pytest.raises(ValueError):
    gB[b].misorientation


def test_B7_inner_boundaries_are_the_low_angle_ones():
  e = boundaryMap(SQ4, 2, 10)
  c = decompose(e)
  s = segment(c, e, gbcAngle([90 * degree, 5 * degree]))
  outer, inner = buildBoundary(c, s, e), buildBoundary(c, s, e, inner=True)
  assert s.nGrains == 1 and close(inner.perimeter(), 40) and close(outer.perimeter(), 80)
  assert np.all(inner.grainId[:, 0] == inner.grainId[:, 1])
  g = calcGrains(e, angle=[90 * degree, 5 * degree])
  assert len(g) == 1 and close(g.innerBoundary.perimeter(), 40) and close(g.boundary.perimeter(), 80)


def test_B8_subsets_and_the_phases_of_a_segment():
  a, b = crystalFrame("m-3m", mineral="alpha"), crystalFrame("6/mmm", mineral="beta")
  e = twoPhaseMap(a, b)
  gB = boundaryOf(e)[0]
  ab = gB.hasPhase("alpha", "beta")
  assert ab.sum() == 20 and close(gB.perimeter(), 100) and gB.hasPhase("alpha").sum() == 60 and len(gB["beta"]) == 60
  sub = gB[ab]
  assert len(sub) == 20 and sub.V is gB.V and close(sub.perimeter(), 20) and len(gB[3]) == 1
  # the sides are listed by phase, so a selected pair comes in one order
  assert np.all(sub.phaseId[:, 0] == 1) and np.all(sub.phaseId[:, 1] == 2)
  mo = sub.misorientation
  assert len(mo) == 20 and mo.frameA is a and mo.frameB is b and not mo.antipodal
  with pytest.raises(ValueError):
    gB.misorientation
  with pytest.raises(ValueError):
    gB[gB.phaseId[:, 0] < 0].misorientation
  # selecting a pair names its order: the first phase lands on the first side
  sel = gB["alpha", "beta"]
  rev = gB["beta", "alpha"]
  assert np.all(sel.phaseId[:, 0] == 1) and np.all(rev.phaseId[:, 0] == 2)
  assert np.array_equal(rev.grainId[:, 0], sel.grainId[:, 1]) and np.array_equal(rev.F[:, 0], sel.F[:, 1])
  assert close(rev.misrotation.data, (sel.misrotation.data * [1, -1, -1, -1]))
  assert len(gB[a, b]) == 20 and len(gB[b]) == 60


def test_B9_the_twin_misorientation_is_found_on_the_segments_that_carry_it():
  cf = crystalFrame("m-3m", mineral="twinned")
  twin = rotation.byAxisAngle(vector3d(1, 1, 1), 60 * degree)
  i, j = np.meshgrid(np.arange(20), np.arange(20), indexing="xy")
  i = i.reshape(-1)
  pos = np.stack([i, j.reshape(-1), np.zeros(400)], axis=-1).astype(float)
  rot = rotation._fromData(np.where((i < 10)[:, None], [1.0, 0, 0, 0], twin.data))
  e = EBSD(pos, rot, np.ones(400, dtype=np.int8), phaseList(cf))
  gB = boundaryOf(e)[0]
  mori = orientation(twin, cf, cf, antipodal=True)
  ind = gB.isTwinning(mori)
  assert ind.sum() == 20 and np.all(gB.grainId[ind, 0] > 0)
  assert np.all(angle(gB[ind].misorientation, mori) < 1e-10)
  assert not gB.isTwinning(orientation(rotation.byAxisAngle(zvector, 40 * degree), cf, cf, antipodal=True)).any()


def test_B10_the_curvature_of_a_digital_circle():
  for R in (6, 12):
    cir, c = diskCircle(R, 4 * R + 9)
    assert cir.isClosed.all()
    k = cir.curvature()
    assert np.all(np.isfinite(k)) and np.all(k > 0)
    # the binomial filter conserves the total turning, eight for a convex digital curve
    assert close(k.sum(), cir.curvature(0).sum(), tol=1e-9) and close(k.sum(), 8, tol=1e-9)
    assert len(cir) == 8 * R + 4 and abs(k.sum() / len(cir) - 1 / R) < 0.1 / R
    assert close(np.sort(cir.flip().curvature()), np.sort(-k), tol=1e-9)
  # a right angle of unit segments, and the ends of an open chain
  gB = boundaryOf(boundaryMap(SQ4, 2, 10))[0]
  g1 = gB.selectByGrainId([1])
  k0 = g1.curvature(0)
  assert close(np.nanmax(k0), 2 / np.sqrt(5))
  open1 = gB[gB.chainId == 1]
  assert not open1.isClosed.any() and np.array_equal(np.flatnonzero(np.isnan(open1.curvature())), [0, len(open1) - 1])


def test_B11_the_mean_direction_ignores_the_staircase():
  cir, c = diskCircle(12, 57)
  mp = cir._mid2d
  radial = mp - c
  radial /= np.linalg.norm(radial, axis=1, keepdims=True)
  tangent = lambda d: cir.plane.to2d(d.data)
  md = tangent(cir.calcMeanDirection(5))
  md /= np.linalg.norm(md, axis=1, keepdims=True)
  assert np.max(np.abs(np.sum(radial * md, axis=1))) < 0.2
  d1 = tangent(cir.direction)
  d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
  assert np.max(np.abs(np.sum(radial * d1, axis=1))) > 0.99
  assert np.all(cir.calcMeanDirection(0).angle(cir.direction, noSymmetry=True) < 1e-12)
  assert cir.calcMeanDirection(5).frame is cir.frame


def test_B12_flipping_swaps_the_sides_and_reverses_the_walk():
  gB = boundaryOf(boundaryMap(SQ4, 2, 10))[0]
  f = gB.flip()
  assert len(f) == len(gB) and close(f.perimeter(), gB.perimeter())
  assert np.array_equal(np.sort(f.grainId, axis=0), np.sort(gB.grainId[:, ::-1], axis=0))
  cid = f.chainId
  same = cid[1:] == cid[:-1]
  assert np.all(f.F[1:, 0][same] == f.F[:-1, 1][same])
  ff = f.flip()
  assert np.array_equal(ff.F, gB.F) and np.array_equal(ff.grainId, gB.grainId) and np.array_equal(ff.ebsdId, gB.ebsdId)
  both = gB.ebsdId[:, 0] >= 0
  assert close(ff.misrotation.data[both], gB.misrotation.data[both])
  # ordering an ordered boundary changes nothing
  o = orderWalk(gB)
  assert np.array_equal(o.F, gB.F) and np.array_equal(o.grainId, gB.grainId)
  # the neighbours along a chain are its rows, and its ends have none
  prev, nxt, hasPrev, hasNext = gB.chainNeighbors()
  assert (~hasPrev).sum() == (~hasNext).sum() == gB.chainId.max()
  cid = gB.chainId
  assert np.all(prev[1:][cid[1:] == cid[:-1]] == np.arange(1, len(gB))[cid[1:] == cid[:-1]] - 1)


def test_B13_a_line_across_the_map_and_a_box_around_a_block():
  gB = boundaryOf(boundaryMap(SQ4, 2, 10))[0]
  # the map runs from -0.5 to 19.5 with the cross at 9.5
  # MATLAB's [x, y] = intersect(gB, xy1, xy2): per segment, NaN where it is not crossed
  x, y = gB.intersect([-2.0, 5.0], [22.0, 5.0])
  cut = ~np.isnan(x)
  assert x.shape == (len(gB),) and cut.sum() == 3 and close(np.sort(x[cut]), [-0.5, 9.5, 19.5]) and close(y[cut], 5)
  x2, _, seg = gB.intersect([22.0, 5.0], [-2.0, 5.0], segLength=True)
  assert np.array_equal(~np.isnan(x2), cut) and close(seg, [10, 10])
  assert np.all(np.isnan(gB.intersect([-2.0, 25.0], [22.0, 25.0])[0]))
  # a line along the cross cuts only the segments across it
  along = ~np.isnan(gB.intersect([9.5, -2.0], [9.5, 22.0])[0])
  assert np.all(gB.V[gB.F[along, 0], 0] != gB.V[gB.F[along, 1], 0])
  # the segments of the lower left block, the cross excluded
  ind = gB.inpolygon([[-1.0, -1.0], [9.0, -1.0], [9.0, 9.0], [-1.0, 9.0]])
  mp = gB._mid2d[ind]
  assert ind.sum() == 18 and np.all((-1 < mp[:, 0]) & (mp[:, 0] < 9) & (-1 < mp[:, 1]) & (mp[:, 1] < 9))
  assert gB.inpolygon([-1, -1, 10, 10]).sum() == 20


def test_triple_points_chains_and_components():
  # three regions of a square map meeting at one interior point
  n = 12
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  label = np.where(i < 6, 1, np.where(j < 6, 2, 3))
  pos = np.stack([i, j, np.zeros(n * n)], axis=-1).astype(float)
  e = EBSD(pos, rotation.byAxisAngle(zvector, label * 20 * degree), np.ones(n * n, dtype=np.int8), phaseList(CS))
  g = calcGrains(e, angle=5 * degree)
  gB = g.boundary
  tp = g.triplePoints
  assert len(tp) == 1 and np.array_equal(tp.grainId[0], np.sort(g.id)) and np.all(tp.phaseId == 1)
  assert tp.boundaryId.shape == (1, 3) and close(tp.angles.sum(), 2 * np.pi) and close(np.sort(tp.angles[0]), [np.pi / 2, np.pi / 2, np.pi])
  assert tp.hasPhase("test")[0] and tp.hasPhase("test", "test", "test")[0] and len(tp["test"]) == 1 and tp[np.zeros(1, dtype=bool)].isempty
  assert "triplePointList" in repr(tp) and len(tp.allV) == 1
  # the three chains between the grains meet at the point, so one component
  inner = gB[np.all(gB.grainId > 0, axis=1)]
  assert np.all(inner.componentId == 1) and np.all(inner.componentSize == len(inner))
  # the grains of a subset carry the segments they touch
  g1 = g[0]
  assert len(g1.boundary) < len(gB) and np.all(np.any(g1.boundary.grainId == g1.id[0], axis=1))
  assert "triple points: 1" in repr(g)
  # a selection keeps the points of the whole network at its vertices, boundaryId into the whole
  assert len(g1.triplePoints) == 1 and len(gB[:1].triplePoints) <= 1 and gB[np.zeros(len(gB), dtype=bool)].triplePoints.isempty
  arms = gB[tp.boundaryId]
  assert np.array_equal(np.sort(arms.grainId.reshape(-1)), np.sort(gB.grainId[tp.boundaryId.reshape(-1)].reshape(-1)))
  assert len(inner[inner.grainId[:, 0] == g.id[0]].triplePoints) == 1
  # an index matrix, the three arms of each point, keeps its shape for the misorientations
  assert arms.misorientation.shape == (1, 3) and len(arms) == 3


def test_the_compiled_boundary_is_the_numpy_one():
  # square and hexagonal maps with holes and two phases: every field of the outer and the
  # inner boundary, the walk order and the orientation of every segment agree exactly
  from mtex import setMTEXpref, getMTEXpref
  CS2 = crystalFrame("6/mmm", mineral="other")
  for A in (SQ4, HEX4):
    e = boundaryMap(A, 5, 6)
    n = len(e)
    i = np.arange(n)
    ph = np.where(i % 7 == 0, 0, np.where((i // 30) % 3 == 0, 2, 1)).astype(np.int8)
    e = EBSD(e.pos, e.rotations, ph, phaseList(CS, CS2))
    c = decompose(e)
    s = segment(c, e, gbcAngle([60 * degree, 5 * degree]))
    before = getMTEXpref("kernels", "numba")
    try:
      out = {}
      for kernels in ("numba", "numpy"):
        setMTEXpref("kernels", kernels)
        out[kernels] = [buildBoundary(c, s, e), buildBoundary(c, s, e, inner=True)]
    finally:
      setMTEXpref("kernels", before)
    for a, b in zip(out["numba"], out["numpy"]):
      assert len(a) > 0 or len(b) == 0
      for f in ("F", "grainId", "ebsdId", "phaseId"):
        assert np.array_equal(getattr(a, f), getattr(b, f))
      assert np.array_equal(a.misrotation.data, b.misrotation.data, equal_nan=True) and a.V is b.V
