# M5: the polygons the boundary encloses, the Julia G1 to G8, and the area assertions M3
# and M4 left open from check_calcGrainsCases and the Julia S10, Q3 and Q6
import numpy as np
import pytest

from mtex import EBSD, phaseList, crystalFrame, rotation, vector3d, zvector, degree, angle, calcGrains, transform, setMTEXpref, getMTEXpref
from mtex.maps.grains import traceLoops, tracePolygons
from mtex.maps.polygon import signedArea2
from conftest import close
from test_calcgrains import blockMap, squareBlockGrid, hexBlockGrid, SQ3
from test_quadruple import quadrants, angleMap

CSG = crystalFrame("m-3m", mineral="test")
SQ5 = np.eye(2)
HEX5 = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])


# an n x n map whose orientation is given by a function of the pixel index
def mapOf(A, n, blockOf):
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  pos = np.stack([A[0, 0] * i + A[0, 1] * j, A[1, 0] * i + A[1, 1] * j, np.zeros(n * n)], axis=-1)
  return EBSD(pos, rotation.byAxisAngle(zvector, blockOf(i, j) * 20 * degree), np.ones(n * n, dtype=np.int8), phaseList(CSG))


def inclusionMap():
  return mapOf(SQ5, 21, lambda i, j: ((8 <= i) & (i <= 12) & (8 <= j) & (j <= 12)).astype(int))


# every grain's boundary is a disjoint union of closed rings: every vertex it uses is met
# by an even number of the grain's own segments, and every polygon encloses a positive area
def assertClosed(g):
  gB = g.boundary
  gid = np.concatenate([gB.grainId[:, 0], gB.grainId[:, 0], gB.grainId[:, 1], gB.grainId[:, 1]])
  vid = np.concatenate([gB.F[:, 0], gB.F[:, 1], gB.F[:, 0], gB.F[:, 1]])
  keep = gid > 0
  key = gid[keep].astype(np.int64) * (len(gB.V) + 1) + vid[keep]
  deg = np.unique(key, return_counts=True)[1]
  assert np.all(deg % 2 == 0)
  assert np.all(g.area > 0)


def test_G1_loops_close_and_wind_by_the_walk_order():
  V = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
  edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
  ls = traceLoops(edges, V)
  assert len(ls) == 1 and len(ls[0]) == 4 and signedArea2(V[ls[0]]) > 0
  # the same square walked the other way is a hole
  assert signedArea2(V[traceLoops([e[::-1] for e in edges], V)[0]]) < 0
  assert traceLoops([], V) == []


def test_G2_a_block_map_has_the_polygons_it_must():
  for A, cell in ((SQ5, 1.0), (HEX5, np.sqrt(3) / 2)):
    g = calcGrains(mapOf(A, 20, lambda i, j: i // 10 + (j // 10) * 2), angle=5 * degree)
    assert len(g) == 4 and all(len(ls) == 1 for ls in g.loops)
    g.score = np.arange(1, 5)
    assert np.array_equal(g[0:2].score, [1, 2])
    with pytest.raises(ValueError):
      g.score = [1]
    assert close(g.area, 100 * cell) and close(g.area.sum(), 400 * cell)
    assert np.all(g.perimeter > 0) and np.all(g.boundarySize > 0) and np.all(g.numNeighbors >= 2)
    selected = g[[3, 1]]
    assert np.array_equal(selected.numNeighbors, g.numNeighbors[[3, 1]]) and np.array_equal(selected.isBoundary, g.isBoundary[[3, 1]])
    assert len(g[3].neighbors()) == 0 and len(g[3].neighbors(full=True)) == g.numNeighbors[3]
    assert np.array_equal(selected.selectByGrainId([2, 4]).id, [2, 4])
    with pytest.raises(ValueError):
      selected.selectByGrainId(1)
    adjacency = g.neighbors(matrix=True).toarray()
    assert np.array_equal(adjacency, adjacency.T) and not adjacency.diagonal().any()
    assert np.array_equal(adjacency.sum(axis=1), g.numNeighbors)
    assert g.isBoundary.all() and not g.hasHole.any() and close(g.equivalentRadius, np.sqrt(g.area / np.pi))
    assert g.poly[0][0] == g.poly[0][-1]


def test_G3_a_square_grains_measures_are_its_closed_forms():
  g = calcGrains(mapOf(SQ5, 20, lambda i, j: i // 10 + (j // 10) * 2), angle=5 * degree)
  assert close(g.area, 100) and close(g.perimeter, 40) and close(g.diameter, 10 * np.sqrt(2))
  assert close(g.equivalentPerimeter, 2 * np.pi * np.sqrt(100 / np.pi))
  c = g.centroid
  assert c.frame is g.frame
  assert sorted(map(tuple, np.round(c.data[:, :2], 6).tolist())) == [(4.5, 4.5), (4.5, 14.5), (14.5, 4.5), (14.5, 14.5)]


def test_G4_an_inclusion_is_a_hole_in_the_grain_around_it():
  g = calcGrains(inclusionMap(), angle=5 * degree)
  assert len(g) == 2
  big = int(np.argmax(g.numPixel))
  small = 1 - big
  assert g.numPixel[small] == 25 and len(g.loops[big]) == 2 and len(g.loops[small]) == 1
  # the hole is taken off the area, so the two areas tile the map exactly
  assert close(g.area[small], 25) and close(g.area[big], 21 * 21 - 25) and close(g.area.sum(), 21 * 21)
  assert g.hasHole[big] and not g.hasHole[small] and g.isInclusion[small] and not g.isInclusion[big]
  assert not g.isBoundary[small] and g.isBoundary[big]
  assert close(g.perimeter[big], 4 * 21) and close(g.perimeterWithInclusion[big], 4 * 21 + 4 * 5)


def test_G5_a_point_is_inside_the_grain_that_covers_it():
  g = calcGrains(inclusionMap(), angle=5 * degree)
  big = int(np.argmax(g.numPixel))
  small = 1 - big
  inside = g.checkInside([[10.0, 10.0], [2.0, 2.0], [-5.0, -5.0]])
  assert inside.shape == (3, 2)
  assert inside[0, small] and not inside[0, big] and inside[1, big] and not inside[1, small] and not inside[2].any()
  assert np.array_equal(g.inpolygon([10.0, 10.0]), [small]) and np.array_equal(g(10, 10).id, g.id[[small]])
  assert g(-5, -5).size == 0 and np.array_equal(g.inpolygon([2.0, 2.0]), [big])
  assert g.checkInside(vector3d(10.0, 10.0, 0.0)).shape == (1, 2)
  assert g.checkInside(np.zeros((0, 2))).shape == (0, 2)


def test_G6_a_subset_keeps_its_grains_and_their_boundary():
  g = calcGrains(mapOf(SQ5, 20, lambda i, j: i // 10 + (j // 10) * 2), angle=5 * degree)
  sub = g[[0, 1]]
  assert len(sub) == 2 and np.array_equal(sub.id, g.id[[0, 1]]) and np.array_equal(g[0].id, g.id[:1])
  assert close(g[0].area, g.area[:1]) and close(sub.area, g.area[[0, 1]]) and len(sub.boundary) < len(g.boundary)
  assert np.all(np.isin(sub.boundary.grainId[:, 0], sub.id) | np.isin(sub.boundary.grainId[:, 1], sub.id))
  assert len(g["test"]) == 4 and len(g["id", 2, 3]) == 2
  # phase names and a condition together, MATLAB's grains('test', grains.numPixel > 100)
  big = g["test", g.numPixel >= 100]
  assert np.array_equal(big.id, g.id[g.numPixel >= 100]) and len(g["test", np.zeros(len(g), dtype=bool)]) == 0


def test_G7_the_keyhole_form_carries_the_holes():
  g = calcGrains(inclusionMap(), angle=5 * degree)
  big = int(np.argmax(g.numPixel))
  kh = g.keyhole(big)
  outer, hole = g.loops[big]
  assert len(kh) == len(outer) + len(hole) + 3 and kh[0] == outer[0] and kh[len(outer)] == outer[0] and kh[-1] == outer[0]
  small = 1 - big
  assert np.array_equal(g.keyhole(small), np.concatenate([g.loops[small][0], g.loops[small][0][:1]]))


def test_G8_merging_relabels_the_grains_and_keeps_the_area():
  e = mapOf(SQ5, 20, lambda i, j: i // 10 + (j // 10) * 2)
  g = calcGrains(e, angle=5 * degree)
  a, b = g.neighbors()[0]
  m = g.merge([(a, b)])
  parent = g.parentId
  pa, pb = parent[a - 1], parent[b - 1]
  assert len(m) == 3 and pa == pb and len(np.unique(parent)) == 3
  assert set(m.childId[pa - 1]) == {a, b} and all(len(m.childId[k]) == np.sum(parent == m.id[k]) for k in range(len(m)))
  assert close(m.area.sum(), g.area.sum()) and m.numPixel.sum() == g.numPixel.sum()
  assert np.array_equal(e[g.selectByGrainId([a, b])].grainId, e[np.isin(e.grainId, [a, b])].grainId)
  assert len(e[g.selectByGrainId(a)]) == g.numPixel[a - 1] and e[g[np.zeros(4, dtype=bool)]].isempty
  assert close(m.area[pa - 1], g.area[a - 1] + g.area[b - 1]) and m.numPixel[pa - 1] == g.numPixel[a - 1] + g.numPixel[b - 1]
  # the boundary the two shared is inside the merged grain now
  assert len(m.boundary) < len(g.boundary) and len(m.innerBoundary) > len(g.innerBoundary)
  assert np.all(m.boundary.grainId[:, 0] != m.boundary.grainId[:, 1])
  # a threshold above every misorientation joins the whole map, one below joins nothing
  m2 = g.merge(threshold=90 * degree)
  assert len(m2) == 1 and close(m2.area.sum(), g.area.sum()) and m2.numPixel[0] == len(e)
  assert len(g.merge(threshold=1 * degree)) == len(g) and np.array_equal(g.parentId, g.id)
  selected = g[[3, 1]]
  merged = selected.merge([(4, 2)])
  assert len(merged) == 1 and np.array_equal(selected.parentId, [1, 1]) and close(merged.area, [selected.area.sum()])
  assert np.array_equal(merged.numNeighbors, [2]) and merged.isBoundary[0] and merged.boundary.grainId.max() == 3
  assert close(selected.merge(threshold=90 * degree).area, merged.area)
  unchanged = selected.merge(threshold=1 * degree)
  assert np.array_equal(unchanged.id, selected.id) and close(unchanged.area, selected.area) and np.array_equal(selected.parentId, selected.id)
  assert unchanged.childId == [[4], [2]]
  assert close(selected.merge(selected.boundary).area, merged.area)
  with pytest.raises(ValueError):
    g[[0, 0]].merge([(1, 1)])
  # two interior grains merged carry no border and no outside
  inner = calcGrains(mapOf(SQ5, 20, lambda i, j: i // 5 + 4 * (j // 5)), angle=5 * degree)
  ids = [inner(7, 7).id[0], inner(12, 7).id[0]]
  ins = inner.selectByGrainId(ids)
  mi = ins.merge([tuple(ids)])
  assert not mi.isBoundary[0] and not (mi.boundary.grainId == 0).any() and close(mi.area, [ins.area.sum()])


def test_G8b_merging_by_labels_graphs_and_inclusions():
  g = calcGrains(mapOf(SQ5, 20, lambda i, j: i // 10 + (j // 10) * 2), angle=5 * degree)
  # a label per grain, a graph of grains by id, and the labels alone with testRun
  m = g.merge(np.array([1, 1, 2, 2]))
  assert len(m) == 2 and np.array_equal(g.parentId, [1, 1, 2, 2])
  from scipy.sparse import csr_matrix
  A = csr_matrix((np.ones(1), ([0], [1])), shape=(4, 4))
  assert np.array_equal(g.merge(A, testRun=True), [1, 1, 2, 3]) and len(g.merge(A)) == 3
  # an inclusion joins the grain around it, which keeps its phase and orientation
  gi = calcGrains(inclusionMap(), angle=5 * degree)
  big = int(np.argmax(gi.numPixel))
  host = gi.inclusionHost
  assert host[big] == 0 and host[1 - big] == gi.id[big]
  assert len(gi.merge(inclusions=True, maxSize=10)) == 2
  keep = np.zeros(2, dtype=bool)
  assert len(gi.merge(inclusions=True, host=keep)) == 2
  mi = gi.merge(inclusions=True)
  assert len(mi) == 1 and mi.numPixel[0] == 21 * 21 and not mi.hasHole[0]
  assert np.allclose(np.abs(mi.meanRotation.data[0] @ gi.meanRotation.data[big]), 1)
  # a grain turned into another phase takes its boundary along with update
  CS2 = crystalFrame("m-3m", mineral="other")
  two = gi.copy()
  object.__setattr__(two, "CSList", phaseList(CSG, CS2))
  two.phaseId[1 - big] = 2
  u = two.update()
  assert np.all(u.boundary.phaseId[np.any(u.boundary.grainId == gi.id[1 - big], axis=1)].max(axis=1) == 2)

# ------------------------------------------------------------------ the areas M3 and M4 left open

def test_S10_the_rim_closes_and_alpha_bounds_both():
  e = blockMap(SQ3, 1, 7, notIndexed=((3, 3, 0.0), (0, 3, 0.0)))
  ix = e[e.isIndexed]
  for alpha in (3.1, 20.0):
    g = calcGrains(ix, angle=5 * degree, alpha=alpha)
    # 49 cells of map less the quarter the outline gives up at the absorbed rim cell
    assert len(g) == 1 and close(g.area.sum(), 48.75)
  # a bay of radius 4 is wider than the disk, so its middle stays out of the map
  e2 = blockMap(SQ3, 1, 15, notIndexed=((7, 14, 4.0),))
  g2 = calcGrains(e2[e2.isIndexed], angle=5 * degree, alpha=3.1)
  assert g2.area.sum() < 225 - 8


def test_calcGrains_commutes_with_a_transform_and_a_rotation():
  oris = rotation.byAxisAngle(zvector, np.arange(1, 17) * 7 * degree)
  thr = 5 * degree
  for build in (squareBlockGrid, hexBlockGrid):
    e = build(6, 4, oris)
    g0 = calcGrains(e.copy(), angle=thr)
    assertClosed(g0)
    x, y = e.x.reshape(-1), e.y.reshape(-1)
    xc, yc, yh = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2, (y.max() - y.min()) / 2
    scale = lambda p: 1 + 0.02 * (p.y - yc) / yh
    eT = transform(e, lambda p: vector3d(xc + (p.x - xc) * scale(p), p.y, p.z))
    gT = calcGrains(eT, angle=thr)
    # the areas of the distorted map are those of the distorted polygons within 5e-3
    V = g0.V
    VT = np.stack([xc + (V[:, 0] - xc) * (1 + 0.02 * (V[:, 1] - yc) / yh), V[:, 1]], axis=1)
    expected = np.array([sum(signedArea2(VT[l]) for l in ls) / 2 for ls in g0.loops])
    assert len(gT) == len(g0) and np.max(np.abs(np.sort(gT.area) - np.sort(expected)) / np.sort(expected)) < 5e-3
    gR = calcGrains(e.rotate(27 * degree), angle=thr)
    assert len(gR) == len(g0) and np.max(np.abs(np.sort(gR.area) - np.sort(g0.area)) / np.sort(g0.area)) < 1e-3


def test_every_polygon_is_a_closed_ring_of_positive_area():
  rng = np.random.default_rng(3)
  e = angleMap(rng.random((30, 30)) * 360)
  e = EBSD(e.pos, e.rotations, e.phaseId, phaseList(crystalFrame("432", mineral="test")))
  for rq in (False, True):
    g = calcGrains(e.copy(), angle=5 * degree, removeQuadruplePoints=rq)
    assert len(g) >= 100
    assertClosed(g)
    assert close(g.area.sum(), 900)
  # the quadruple point split leaves the area where it was
  e = quadrants()
  g = calcGrains(e.copy(), angle=10 * degree)
  q = calcGrains(e.copy(), angle=10 * degree, removeQuadruplePoints=True)
  assert close(q.area.sum(), g.area.sum(), tol=1e-6) and close(q.area.sum(), 16) and all(ls for ls in q.loops)


def test_the_compiled_and_the_python_tracing_agree():
  # grains with holes, an inclusion and a grain touching itself around a hole of the map
  e = inclusionMap()
  ph = e.phaseId.copy()
  ph[[45, 46, 66, 200, 300, 301, 302]] = 0
  e = EBSD(e.pos, e.rotations, ph, e.CSList)
  g = calcGrains(e, angle=5 * degree)
  before = getMTEXpref("kernels", "numba")
  try:
    setMTEXpref("kernels", "numba")
    a = tracePolygons(g.boundary, g.id)
    setMTEXpref("kernels", "numpy")
    b = tracePolygons(g.boundary, g.id)
  finally:
    setMTEXpref("kernels", before)
  V = g.V
  for la, lb in zip(a, b):
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
      # the same ring, whichever vertex it starts at
      assert len(x) == len(y) and close(signedArea2(V[x]), signedArea2(V[y])) and set(x.tolist()) == set(y.tolist())


def test_id2ind_keeps_the_shape_of_the_ids():
  ebsd = blockMap(np.eye(2), 3, 4)
  grains = calcGrains(ebsd)
  ids = grains.boundary.grainId
  ind = grains.id2ind(ids[ids.min(axis=1) > 0])
  assert ind.ndim == 2 and ind.shape[1] == 2 and np.array_equal(grains.id[ind], ids[ids.min(axis=1) > 0])
  assert isinstance(grains.id2ind(int(grains.id[0])), int)


def test_the_compiled_edge_sort_is_the_stable_argsort():
  # the directed edges of every grain, the first sides before the second, each in row order
  from mtex.maps.kernels import edgesByGrainNumba
  g = calcGrains(inclusionMap(), angle=5 * degree)
  gB = g.boundary
  ids = np.asarray(g.id)[::2]
  posOf = np.full(int(gB.grainId.max()) + 1, -1, dtype=np.int64)
  posOf[ids] = np.arange(len(ids))
  ea, eb, goff = edgesByGrainNumba(gB.F, gB.grainId, posOf, len(ids))
  k = np.concatenate([posOf[gB.grainId[:, 0]], posOf[gB.grainId[:, 1]]])
  keep = k >= 0
  order = np.argsort(k[keep], kind="stable")
  assert np.array_equal(ea, np.concatenate([gB.F[:, 0], gB.F[:, 1]])[keep][order])
  assert np.array_equal(eb, np.concatenate([gB.F[:, 1], gB.F[:, 0]])[keep][order])
  assert np.array_equal(goff, np.concatenate([[0], np.cumsum(np.bincount(k[keep], minlength=len(ids)))]))


def test_the_parallel_tracing_of_many_grains_is_the_python_one():
  # enough grains for every thread, some with holes: the loops of the parallel walk are the
  # rings of the Python walk in the same order, whichever vertex each starts at
  e = blockMap(SQ3, 12, 6, notIndexed=((8, 8, 1.5), (30, 30, 2.5), (50, 20, 1.0)))
  g = calcGrains(e, angle=5 * degree)
  before = getMTEXpref("kernels", "numba")
  try:
    setMTEXpref("kernels", "numba")
    a = tracePolygons(g.boundary, g.id)
    setMTEXpref("kernels", "numpy")
    b = tracePolygons(g.boundary, g.id)
  finally:
    setMTEXpref("kernels", before)
  assert len(a) == len(b) == len(g)
  for la, lb in zip(a, b):
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
      assert len(x) == len(y) and close(signedArea2(g.V[x]), signedArea2(g.V[y])) and set(x.tolist()) == set(y.tolist())
