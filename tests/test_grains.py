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


SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
SQUARE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def test_traceLoops_loopsCloseAndWindByTheWalkOrder():
  ls = traceLoops(SQUARE_EDGES, SQUARE)
  assert len(ls) == 1 and len(ls[0]) == 4 and signedArea2(SQUARE[ls[0]]) > 0
  # the same square walked the other way is a hole
  assert signedArea2(SQUARE[traceLoops([e[::-1] for e in SQUARE_EDGES], SQUARE)[0]]) < 0


def test_traceLoops_noEdgesGiveNoLoops():
  assert traceLoops([], SQUARE) == []


# the four 10 x 10 blocks of a 20 x 20 map on a lattice, and the area of its cell
def fourBlocks(A):
  return calcGrains(mapOf(A, 20, lambda i, j: i // 10 + (j // 10) * 2), angle=5 * degree)


BLOCK_LATTICES = pytest.mark.parametrize("A, cell", ((SQ5, 1.0), (HEX5, np.sqrt(3) / 2)), ids=("square", "hex"))


@BLOCK_LATTICES
def test_calcGrains_blockMapHasOneSimplePolygonPerBlock(A, cell):
  g = fourBlocks(A)
  assert len(g) == 4 and all(len(ls) == 1 for ls in g.loops)
  assert close(g.area, 100 * cell) and close(g.area.sum(), 400 * cell)
  assert np.all(g.perimeter > 0) and np.all(g.boundarySize > 0) and np.all(g.numNeighbors >= 2)
  assert g.isBoundary.all() and not g.hasHole.any() and close(g.equivalentRadius, np.sqrt(g.area / np.pi))
  assert g.poly[0][0] == g.poly[0][-1]


@BLOCK_LATTICES
def test_grain2d_propertyHasOneValuePerGrain(A, cell):
  g = fourBlocks(A)
  g.score = np.arange(1, 5)
  assert np.array_equal(g[0:2].score, [1, 2])
  with pytest.raises(ValueError):
    g.score = [1]


@BLOCK_LATTICES
def test_grain2d_selectionKeepsTheNeighboursOfTheMap(A, cell):
  g = fourBlocks(A)
  selected = g[[3, 1]]
  assert np.array_equal(selected.numNeighbors, g.numNeighbors[[3, 1]]) and np.array_equal(selected.isBoundary, g.isBoundary[[3, 1]])
  assert len(g[3].neighbors()) == 0 and len(g[3].neighbors(full=True)) == g.numNeighbors[3]


@BLOCK_LATTICES
def test_selectByGrainId_refusesIdsNotInTheSelection(A, cell):
  selected = fourBlocks(A)[[3, 1]]
  assert np.array_equal(selected.selectByGrainId([2, 4]).id, [2, 4])
  with pytest.raises(ValueError):
    selected.selectByGrainId(1)


@BLOCK_LATTICES
def test_neighbors_adjacencyMatrixIsSymmetricAndCountsTheNeighbours(A, cell):
  g = fourBlocks(A)
  adjacency = g.neighbors(matrix=True).toarray()
  assert np.array_equal(adjacency, adjacency.T) and not adjacency.diagonal().any()
  assert np.array_equal(adjacency.sum(axis=1), g.numNeighbors)


def test_grain2d_squareGrainsMeasuresAreItsClosedForms():
  g = fourBlocks(SQ5)
  assert close(g.area, 100) and close(g.perimeter, 40) and close(g.diameter, 10 * np.sqrt(2))
  assert close(g.equivalentPerimeter, 2 * np.pi * np.sqrt(100 / np.pi))
  c = g.centroid
  assert c.frame is g.frame
  assert sorted(map(tuple, np.round(c.data[:, :2], 6).tolist())) == [(4.5, 4.5), (4.5, 14.5), (14.5, 4.5), (14.5, 14.5)]


def test_calcGrains_inclusionIsAHoleInTheGrainAroundIt():
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


def test_checkInside_pointIsInsideTheGrainThatCoversIt():
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


def test_grain2d_subsetKeepsItsGrainsAndTheirBoundary():
  g = fourBlocks(SQ5)
  sub = g[[0, 1]]
  assert len(sub) == 2 and np.array_equal(sub.id, g.id[[0, 1]]) and np.array_equal(g[0].id, g.id[:1])
  assert close(g[0].area, g.area[:1]) and close(sub.area, g.area[[0, 1]]) and len(sub.boundary) < len(g.boundary)
  assert np.all(np.isin(sub.boundary.grainId[:, 0], sub.id) | np.isin(sub.boundary.grainId[:, 1], sub.id))


def test_grain2d_selectionByPhaseIdAndCondition():
  g = fourBlocks(SQ5)
  assert len(g["test"]) == 4 and len(g["id", 2, 3]) == 2
  # phase names and a condition together, MATLAB's grains('test', grains.numPixel > 100)
  big = g["test", g.numPixel >= 100]
  assert np.array_equal(big.id, g.id[g.numPixel >= 100]) and len(g["test", np.zeros(len(g), dtype=bool)]) == 0


def test_keyhole_carriesTheHoles():
  g = calcGrains(inclusionMap(), angle=5 * degree)
  big = int(np.argmax(g.numPixel))
  kh = g.keyhole(big)
  outer, hole = g.loops[big]
  assert len(kh) == len(outer) + len(hole) + 3 and kh[0] == outer[0] and kh[len(outer)] == outer[0] and kh[-1] == outer[0]
  small = 1 - big
  assert np.array_equal(g.keyhole(small), np.concatenate([g.loops[small][0], g.loops[small][0][:1]]))


def test_merge_relabelsTheGrainsAndKeepsTheArea():
  e = mapOf(SQ5, 20, lambda i, j: i // 10 + (j // 10) * 2)
  g = calcGrains(e, angle=5 * degree)
  a, b = g.neighbors()[0]
  m = g.merge([(a, b)])
  parent = g.parentId
  pa, pb = parent[a - 1], parent[b - 1]
  assert len(m) == 3 and pa == pb and len(np.unique(parent)) == 3
  assert set(m.childId[pa - 1]) == {a, b} and all(len(m.childId[k]) == np.sum(parent == m.id[k]) for k in range(len(m)))
  assert close(m.area.sum(), g.area.sum()) and m.numPixel.sum() == g.numPixel.sum()
  assert close(m.area[pa - 1], g.area[a - 1] + g.area[b - 1]) and m.numPixel[pa - 1] == g.numPixel[a - 1] + g.numPixel[b - 1]
  # the boundary the two shared is inside the merged grain now
  assert len(m.boundary) < len(g.boundary) and len(m.innerBoundary) > len(g.innerBoundary)
  assert np.all(m.boundary.grainId[:, 0] != m.boundary.grainId[:, 1])


def test_selectByGrainId_selectsTheMeasurementsOfTheGrains():
  e = mapOf(SQ5, 20, lambda i, j: i // 10 + (j // 10) * 2)
  g = calcGrains(e, angle=5 * degree)
  a, b = g.neighbors()[0]
  assert np.array_equal(e[g.selectByGrainId([a, b])].grainId, e[np.isin(e.grainId, [a, b])].grainId)
  assert len(e[g.selectByGrainId(a)]) == g.numPixel[a - 1] and e[g[np.zeros(4, dtype=bool)]].isempty


# a threshold above every misorientation joins the whole map, one below joins nothing
def test_merge_thresholdAboveEveryMisorientationJoinsAllBelowJoinsNothing():
  e = mapOf(SQ5, 20, lambda i, j: i // 10 + (j // 10) * 2)
  g = calcGrains(e, angle=5 * degree)
  m2 = g.merge(threshold=90 * degree)
  assert len(m2) == 1 and close(m2.area.sum(), g.area.sum()) and m2.numPixel[0] == len(e)
  assert len(g.merge(threshold=1 * degree)) == len(g) and np.array_equal(g.parentId, g.id)


def test_merge_ofASelectionStaysWithinIt():
  g = fourBlocks(SQ5)
  selected = g[[3, 1]]
  merged = selected.merge([(4, 2)])
  assert len(merged) == 1 and np.array_equal(selected.parentId, [1, 1]) and close(merged.area, [selected.area.sum()])
  assert np.array_equal(merged.numNeighbors, [2]) and merged.isBoundary[0] and merged.boundary.grainId.max() == 3
  assert close(selected.merge(threshold=90 * degree).area, merged.area)
  unchanged = selected.merge(threshold=1 * degree)
  assert np.array_equal(unchanged.id, selected.id) and close(unchanged.area, selected.area) and np.array_equal(selected.parentId, selected.id)
  assert unchanged.childId == [[4], [2]]
  assert close(selected.merge(selected.boundary).area, merged.area)


def test_merge_refusesRepeatedGrains():
  g = fourBlocks(SQ5)
  with pytest.raises(ValueError):
    g[[0, 0]].merge([(1, 1)])


# two interior grains merged carry no border and no outside
def test_merge_ofInteriorGrainsHasNoOutside():
  inner = calcGrains(mapOf(SQ5, 20, lambda i, j: i // 5 + 4 * (j // 5)), angle=5 * degree)
  ids = [inner(7, 7).id[0], inner(12, 7).id[0]]
  ins = inner.selectByGrainId(ids)
  mi = ins.merge([tuple(ids)])
  assert not mi.isBoundary[0] and not (mi.boundary.grainId == 0).any() and close(mi.area, [ins.area.sum()])


# a label per grain, a graph of grains by id, and the labels alone with testRun
def test_merge_byLabelsOrAGraphOfGrains():
  g = fourBlocks(SQ5)
  m = g.merge(np.array([1, 1, 2, 2]))
  assert len(m) == 2 and np.array_equal(g.parentId, [1, 1, 2, 2])
  from scipy.sparse import csr_matrix
  A = csr_matrix((np.ones(1), ([0], [1])), shape=(4, 4))
  assert np.array_equal(g.merge(A, testRun=True), [1, 1, 2, 3]) and len(g.merge(A)) == 3


# an inclusion joins the grain around it, which keeps its phase and orientation
def test_merge_inclusionJoinsTheGrainAroundIt():
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


# two inclusions in one grain and a third inside the first: each the grain whose hole it borders
def test_inclusionHost_ofNestedInclusionsIsTheGrainAroundEach():
  def blocks(i, j):
    b = np.zeros_like(i)
    b[(4 <= i) & (i <= 12) & (4 <= j) & (j <= 12)] = 1
    b[(7 <= i) & (i <= 9) & (7 <= j) & (j <= 9)] = 2
    b[(16 <= i) & (i <= 20) & (16 <= j) & (j <= 20)] = 3
    return b
  g = calcGrains(mapOf(SQ5, 25, blocks), angle=5 * degree)
  at = {int(round(float(r) / (20 * degree))): k for k, r in enumerate(g.meanRotation.angle())}
  host = g.inclusionHost
  assert host[at[0]] == 0 and host[at[1]] == g.id[at[0]] and host[at[3]] == g.id[at[0]] and host[at[2]] == g.id[at[1]]


# a grain turned into another phase takes its boundary along with update
def test_update_carriesAGrainsNewPhaseToItsBoundary():
  gi = calcGrains(inclusionMap(), angle=5 * degree)
  big = int(np.argmax(gi.numPixel))
  CS2 = crystalFrame("m-3m", mineral="other")
  two = gi.copy()
  object.__setattr__(two, "CSList", phaseList(CSG, CS2))
  two.phaseId[1 - big] = 2
  u = two.update()
  assert np.all(u.boundary.phaseId[np.any(u.boundary.grainId == gi.id[1 - big], axis=1)].max(axis=1) == 2)

# ------------------------------------------------------------------ the areas M3 and M4 left open

# the edge of the map is not the edge of a hole: the rim cell is absorbed as a cell in its
# place further in would be, and the outline gives up a quarter of it
@pytest.mark.parametrize("alpha", (3.1, 20.0))
def test_calcGrains_rimCellIsAbsorbedWhateverAlpha(alpha):
  e = blockMap(SQ3, 1, 7, notIndexed=((3, 3, 0.0), (0, 3, 0.0)))
  g = calcGrains(e[e.isIndexed], angle=5 * degree, alpha=alpha)
  assert len(g) == 1 and g.numPixel[0] == 47 and close(g.area.sum(), 48.75)


# a bay of radius 4 is wider than the disk, so its middle stays out of the map
def test_calcGrains_alphaLeavesABayWiderThanTheDiskOut():
  e2 = blockMap(SQ3, 1, 15, notIndexed=((7, 14, 4.0),))
  g2 = calcGrains(e2[e2.isIndexed], angle=5 * degree, alpha=3.1)
  assert g2.area.sum() < 225 - 8


# a 6 x 6 block grid of 16 grains and its grains
def blockGrains(build):
  e = build(6, 4, rotation.byAxisAngle(zvector, np.arange(1, 17) * 7 * degree))
  return e, calcGrains(e.copy(), angle=5 * degree)


BLOCK_GRIDS = pytest.mark.parametrize("build", (squareBlockGrid, hexBlockGrid), ids=("square", "hex"))


@BLOCK_GRIDS
def test_calcGrains_blockGridGivesClosedPolygons(build):
  e, g0 = blockGrains(build)
  assert g0.isIndexed.sum() == 16
  assertClosed(g0)


@BLOCK_GRIDS
def test_calcGrains_commutesWithATransform(build):
  e, g0 = blockGrains(build)
  x, y = e.x.reshape(-1), e.y.reshape(-1)
  xc, yc, yh = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2, (y.max() - y.min()) / 2
  scale = lambda p: 1 + 0.02 * (p.y - yc) / yh
  eT = transform(e, lambda p: vector3d(xc + (p.x - xc) * scale(p), p.y, p.z))
  gT = calcGrains(eT, angle=5 * degree)
  # the areas of the distorted map are those of the distorted polygons within 5e-3
  V = g0.V
  VT = np.stack([xc + (V[:, 0] - xc) * (1 + 0.02 * (V[:, 1] - yc) / yh), V[:, 1]], axis=1)
  expected = np.array([sum(signedArea2(VT[l]) for l in ls) / 2 for ls in g0.loops])
  assert len(gT) == len(g0) and np.array_equal(np.sort(gT.numPixel), np.sort(g0.numPixel))
  assert np.max(np.abs(np.sort(gT.area) - np.sort(expected)) / np.sort(expected)) < 5e-3


@BLOCK_GRIDS
def test_calcGrains_commutesWithARotation(build):
  e, g0 = blockGrains(build)
  gR = calcGrains(e.rotate(27 * degree), angle=5 * degree)
  assert len(gR) == len(g0) and np.array_equal(np.sort(gR.numPixel), np.sort(g0.numPixel))
  assert np.max(np.abs(np.sort(gR.area) - np.sort(g0.area)) / np.sort(g0.area)) < 1e-3


@pytest.mark.parametrize("rq", (False, True))
def test_calcGrains_everyPolygonIsAClosedRingOfPositiveArea(rq):
  rng = np.random.default_rng(3)
  e = angleMap(rng.random((30, 30)) * 360)
  e = EBSD(e.pos, e.rotations, e.phaseId, phaseList(crystalFrame("432", mineral="test")))
  g = calcGrains(e, angle=5 * degree, removeQuadruplePoints=rq)
  assert len(g) >= 100
  assertClosed(g)
  assert close(g.area.sum(), 900)


# the quadruple point split leaves the area where it was
def test_calcGrains_quadruplePointSplitKeepsTheArea():
  e = quadrants()
  g = calcGrains(e.copy(), angle=10 * degree)
  q = calcGrains(e.copy(), angle=10 * degree, removeQuadruplePoints=True)
  assert close(q.area.sum(), g.area.sum(), tol=1e-6) and close(q.area.sum(), 16) and all(ls for ls in q.loops)


# grains with holes, an inclusion and a grain touching itself around a hole of the map
def holedInclusionGrains():
  e = inclusionMap()
  ph = e.phaseId.copy()
  ph[[45, 46, 66, 200, 300, 301, 302]] = 0
  return calcGrains(EBSD(e.pos, e.rotations, ph, e.CSList), angle=5 * degree)


# enough grains for every thread of the parallel walk, some with holes
def manyGrains():
  return calcGrains(blockMap(SQ3, 12, 6, notIndexed=((8, 8, 1.5), (30, 30, 2.5), (50, 20, 1.0))), angle=5 * degree)


@pytest.mark.parametrize("grains", [holedInclusionGrains, manyGrains])
def test_tracePolygons_compiledIsThePythonOne(grains):
  # the loops of the compiled walk are the rings of the Python walk in the same order,
  # whichever vertex each starts at
  g = grains()
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


def test_id2ind_keepsTheShapeOfTheIds():
  ebsd = blockMap(np.eye(2), 3, 4)
  grains = calcGrains(ebsd)
  ids = grains.boundary.grainId
  ind = grains.id2ind(ids[ids.min(axis=1) > 0])
  assert ind.ndim == 2 and ind.shape[1] == 2 and np.array_equal(grains.id[ind], ids[ids.min(axis=1) > 0])
  assert isinstance(grains.id2ind(int(grains.id[0])), int)


def test_edgesByGrainNumba_isTheStableArgsort():
  pytest.importorskip("numba")
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


# the loop sums of both backends over grains with holes and an inclusion: area, both perimeters,
# centroid and second moments
def loopSumsByBackend(g):
  from mtex.maps.polygon import loopSums
  before = getMTEXpref("kernels", "numba")
  try:
    out = {}
    for be in ("numba", "numpy"):
      setMTEXpref("kernels", be)
      out[be] = loopSums(g.V, g.loops, g.size)
  finally:
    setMTEXpref("kernels", before)
  return out


def test_loopSums_compiledIsThePythonOne():
  g = holedInclusionGrains()
  out = loopSumsByBackend(g)
  assert out["numba"].shape == (g.size, 8) and np.allclose(out["numba"], out["numpy"], rtol=1e-13, atol=1e-12)


# a hole is taken off the area and left out of the outer perimeter only
def test_loopSums_holeIsLeftOutOfTheOuterPerimeterOnly():
  out = loopSumsByBackend(holedInclusionGrains())
  assert np.any(out["numba"][:, 2] > out["numba"][:, 1] + 1e-9)


def test_loopSums_noLoopsSumToZero():
  from mtex.maps.polygon import loopSums
  g = holedInclusionGrains()
  assert np.array_equal(loopSums(g.V, [[] for _ in range(3)], 3), np.zeros((3, 8)))
