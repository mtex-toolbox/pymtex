# the equispaced and regular sphere grids, their search and the grid of a frame
import numpy as np
import pytest

from mtex import S2Grid, equispacedS2Grid, regularS2Grid, crystalFrame, vector3d, rotation, degree, angle, fundamentalSector, symmetrise, Miller

rng = np.random.default_rng(1)
vs = vector3d.rand(200, rng=rng)
cs = crystalFrame("m-3m")


def bruteDots(g, v):
  d = g.data @ v.data
  return np.abs(d) if g.antipodal else d


grids = [equispacedS2Grid(resolution=5 * degree), equispacedS2Grid(resolution=5 * degree, antipodal=True),
         equispacedS2Grid(resolution=4 * degree, hemisphere="upper", center=False), equispacedS2Grid(points=1000)]
gridIds = ["5", "5antipodal", "4upper", "1000points"]


@pytest.fixture(scope="module")
def g5():
  return equispacedS2Grid(resolution=5 * degree)


@pytest.mark.parametrize("g", grids, ids=gridIds)
def test_find_nearestAgainstBruteForce(g):
  d = [bruteDots(g, v) for v in vs]
  ind, dist = g.find(vs)
  assert ind.shape == (200,) and all(abs(d[k][ind[k]] - np.max(d[k])) < 1e-12 for k in range(200))
  assert np.allclose(np.cos(dist), [d[k][ind[k]] for k in range(200)])


@pytest.mark.parametrize("g", grids, ids=gridIds)
def test_neighbourIndices_againstBruteForce(g):
  eps = 1.7 * g.resolution
  assert all(g.neighbourIndices(vs[k], eps).tolist() == np.flatnonzero(bruteDots(g, vs[k]) >= np.cos(eps)).tolist() for k in range(200))


@pytest.mark.parametrize("g", grids, ids=gridIds)
def test_find_radiusAgainstBruteForce(g):
  eps = 1.7 * g.resolution
  I, D = g.find(vs, eps)
  assert I.shape == (200, g.size) and I.dtype == bool
  for k in range(200):
    cols = I[k].indices
    assert np.allclose(np.cos(D[k].data), bruteDots(g, vs[k])[cols])


@pytest.mark.parametrize("g", grids, ids=gridIds)
def test_S2Grid_ringsAddUpToTheSize(g):
  assert g.size == g.offset[-1] == sum(len(r) for r in g.rhoGrid)


def test_sector_grid_wraps_with_its_period():
  gq = equispacedS2Grid(resolution=6 * degree, minRho=0, maxRho=np.pi / 2, rhoPeriod=np.pi / 2)
  copies = lambda v: [rotation.byAxisAngle(vector3d.Z, k * np.pi / 2) * v for k in range(4)]
  dq = [np.max([gq.data @ c.data for c in copies(v)], axis=0) for v in vs]
  ind, _ = gq.find(vs)
  assert all(abs(dq[k][ind[k]] - np.max(dq[k])) < 1e-12 for k in range(200))
  assert all(gq.neighbourIndices(vs[k], 10 * degree).tolist() == np.flatnonzero(dq[k] >= np.cos(10 * degree)).tolist() for k in range(200))


def test_equispacedS2Grid_sizeAndResolution(g5):
  assert 1600 < g5.size < 1700 and abs(g5.resolution - 5 * degree) < 1e-12 and isinstance(g5, S2Grid)


def test_equispacedS2Grid_coversBelowTheResolution(g5):
  assert max(np.arccos(np.max(bruteDots(g5, v))) for v in vs) < g5.resolution


def test_equispacedS2Grid_nearestNeighbourDistanceIsTheResolution(g5):
  I, D = g5.find(g5, 1.5 * g5.resolution)
  nn = np.array([np.min(D[j].data[D[j].indices != j]) for j in range(g5.size)])
  assert np.min(nn) > 0.9 * g5.resolution and np.max(nn) < 1.1 * g5.resolution


def test_equispacedS2Grid_antipodalHasHalfTheNodesAndCovers(g5):
  ga = equispacedS2Grid(resolution=5 * degree, antipodal=True)
  assert ga.antipodal and abs(ga.size / (g5.size / 2) - 1) < 0.02
  assert max(np.arccos(np.max(bruteDots(ga, v))) for v in vs) < ga.resolution


def test_subGrid_byMaskIsAGridThatSearches(g5):
  mask = g5.z > 0.2
  s = g5.subGrid(mask)
  assert isinstance(s, S2Grid) and s.size == np.sum(mask) and s.numRings < g5.numRings
  assert all(abs(bruteDots(s, v)[s.find(v)[0]] - np.max(bruteDots(s, v))) < 1e-12 for v in vs[:50])


def test_subGrid_capAroundADirection(g5):
  s2 = g5.subGrid(vector3d.Z, 20 * degree)
  assert np.all(angle(s2, vector3d.Z) <= 20 * degree + 1e-10)


def test_polar_givesTheGridsTheta(g5):
  th, rh, _ = g5.polar()
  assert np.max(np.abs(th - g5.theta)) < 1e-12


def test_equispacedS2Grid_carriesItsFrameAndSlicesToVectors():
  gf = equispacedS2Grid(resolution=10 * degree, frame=cs)
  assert gf.frame is cs and gf[0].frame is cs and not isinstance(gf[0:5], S2Grid) and isinstance(gf[0:5], vector3d)


def test_S2Grid_rotatedOrAddedGivesVectors():
  gf = equispacedS2Grid(resolution=10 * degree, frame=cs)
  r = rotation.byAxisAngle(vector3d.X, 0.3)
  assert not isinstance(r * gf, S2Grid) and isinstance(r * gf, vector3d) and not isinstance(gf + gf, S2Grid)


def test_equispacedS2Grid_inAFundamentalSector(g5):
  sR = fundamentalSector(cs)
  gs = equispacedS2Grid(resolution=5 * degree, region=sR)
  assert gs.frame is cs and np.all(sR.checkInside(gs)) and abs(gs.size / (g5.size / 48) - 1) < 0.25


def test_S2Grid_reprCountsThePoints(g5):
  assert "points" in repr(g5) and str(g5) == f"{g5.size} points"


def test_regularS2Grid_byResolutionIsThePoleFigureLayout():
  R = regularS2Grid(resolution=10 * degree, antipodal=True)
  assert R.shape == (36, 10) and R.antipodal


def test_regularS2Grid_byThetaAndRho():
  R2 = regularS2Grid(theta=[0, 0.2, 0.4], rho=[0, 1, 2, 3])
  assert R2.shape == (4, 3) and abs(R2[1, 2].theta - 0.4) < 1e-12


def test_regularS2Grid_byPoints():
  assert regularS2Grid(points=(72, 19)).shape == (72, 19)


def test_S2Grid_equispacedHasUniformSteps():
  assert equispacedS2Grid(resolution=1 * degree).hasUniformSteps


def test_nearestStepwise_closeToTheSearchAndMostlyEqual():
  g = equispacedS2Grid(resolution=1 * degree)
  ws = vector3d.rand(2000, rng=np.random.default_rng(7))
  i, j = g.nearestStepwise(ws), g.find(ws)[0]
  assert np.all(angle(g[i], ws) < 2.5 * g.resolution) and np.mean(i == j) > 0.9


def test_nearestStepwise_findsTheNodesOfASectorGrid():
  sec = equispacedS2Grid(resolution=2 * degree, region=fundamentalSector(crystalFrame("m-3m")))
  for v in sec[::50]:
    assert sec.nearestStepwise(v) == sec.find(v)[0]


def test_equispacedS2Grid_ofAFrameCoversItsFundamentalSector():
  c = crystalFrame("432")
  full = equispacedS2Grid(resolution=3 * degree, frame=c)
  sector = equispacedS2Grid(c, resolution=3 * degree)
  assert sector.frame is c and isinstance(sector, S2Grid)
  assert abs(sector.size * c.numSym() / full.size - 1) < 0.1
  assert np.all(fundamentalSector(c).checkInside(sector))


def test_equispacedS2Grid_ofAFrameAntipodalIsTheLaueSector():
  c = crystalFrame("432")
  sector = equispacedS2Grid(c, resolution=3 * degree)
  axes = equispacedS2Grid(c, resolution=3 * degree, antipodal=True)
  laue = equispacedS2Grid(crystalFrame("m-3m"), resolution=3 * degree)
  assert axes.antipodal and axes.size == laue.size and abs(2 * axes.size / sector.size - 1) < 0.1


def test_equispaced_default_resolution():
  # MATLAB's default spacing of an equispaced grid is 2.5 degree
  assert np.isclose(equispacedS2Grid(antipodal=True).resolution, 2.5 * degree, rtol=0.05)


def test_polar_ofAGridWithoutPointsIsEmpty():
  g = equispacedS2Grid(resolution=10 * degree).subGrid(np.zeros(0, dtype=int))
  assert g.size == 0 and all(len(x) == 0 for x in g.polar())


def test_polar_givesEveryPointThePolarAngleOfItsRing():
  g = equispacedS2Grid(resolution=10 * degree)
  th, rh, _ = g.polar()
  assert np.allclose(vector3d.byPolar(th, rh).data, g.data)


def test_subGrid_dropsTheRingsAMaskEmpties():
  g = equispacedS2Grid(resolution=10 * degree)
  assert g.subGrid(np.arange(g.offset[2], g.offset[4])).numRings == 2


def test_thetaLadder_centredHoldsOneRingMoreThanItsCount():
  from mtex.geometry.grids.s2grid import _thetaLadder
  assert np.allclose(_thetaLadder(0.0, 2, 0.5, True), [0, 0.5, 1]) and np.allclose(_thetaLadder(0.0, 2, 0.5, False), [0.25, 0.75])


def test_ringCount_fitsTheRingsIntoThePolarRange():
  from mtex.geometry.grids.s2grid import _ringCount
  assert _ringCount(np.pi, np.pi / 4, True) == 4 and _ringCount(np.pi, np.pi / 4, False) == 4


def test_polarRange_ofTheUpperHemisphereEndsAtTheEquator():
  from mtex.geometry.grids.s2grid import _polarRange
  assert _polarRange(0.0, None, "upper", False) == (0.0, np.pi / 2) and _polarRange(0.0, None, "lower", False) == (np.pi / 2, np.pi)
