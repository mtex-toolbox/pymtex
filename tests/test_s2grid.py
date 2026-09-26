# the equispaced and regular sphere grids, their search and the grid of a frame
import numpy as np
import pytest

from mtex import S2Grid, equispacedS2Grid, regularS2Grid, crystalFrame, vector3d, rotation, degree, angle, fundamentalSector, symmetrise, Miller

rng = np.random.default_rng(1)
vs = vector3d.rand(200, rng=rng)


def bruteDots(g, v):
  d = g.data @ v.data
  return np.abs(d) if g.antipodal else d


@pytest.mark.parametrize("g", [equispacedS2Grid(resolution=5 * degree), equispacedS2Grid(resolution=5 * degree, antipodal=True),
                               equispacedS2Grid(resolution=4 * degree, hemisphere="upper", noCenter=True), equispacedS2Grid(points=1000)])
def test_search_against_brute_force(g):
  d = [bruteDots(g, v) for v in vs]
  ind, dist = g.find(vs)
  assert ind.shape == (200,) and all(abs(d[k][ind[k]] - np.max(d[k])) < 1e-12 for k in range(200))
  assert np.allclose(np.cos(dist), [d[k][ind[k]] for k in range(200)])
  eps = 1.7 * g.resolution
  assert all(g.neighbourIndices(vs[k], eps).tolist() == np.flatnonzero(d[k] >= np.cos(eps)).tolist() for k in range(200))
  I, D = g.find(vs, eps)
  assert I.shape == (200, g.size) and I.dtype == bool
  for k in range(200):
    cols = I[k].indices
    assert np.allclose(np.cos(D[k].data), d[k][cols])
  assert g.size == g.offset[-1] == sum(len(r) for r in g.rhoGrid)


def test_sector_grid_wraps_with_its_period():
  gq = equispacedS2Grid(resolution=6 * degree, minRho=0, maxRho=np.pi / 2, rhoPeriod=np.pi / 2)
  copies = lambda v: [rotation.byAxisAngle(vector3d.Z, k * np.pi / 2) * v for k in range(4)]
  dq = [np.max([gq.data @ c.data for c in copies(v)], axis=0) for v in vs]
  ind, _ = gq.find(vs)
  assert all(abs(dq[k][ind[k]] - np.max(dq[k])) < 1e-12 for k in range(200))
  assert all(gq.neighbourIndices(vs[k], 10 * degree).tolist() == np.flatnonzero(dq[k] >= np.cos(10 * degree)).tolist() for k in range(200))


def test_coverage_uniformity_subgrid_frame():
  g = equispacedS2Grid(resolution=5 * degree)
  assert 1600 < g.size < 1700 and abs(g.resolution - 5 * degree) < 1e-12 and isinstance(g, S2Grid)
  assert max(np.arccos(np.max(bruteDots(g, v))) for v in vs) < g.resolution
  I, D = g.find(g, 1.5 * g.resolution)
  nn = np.array([np.min(D[j].data[D[j].indices != j]) for j in range(g.size)])
  assert np.min(nn) > 0.9 * g.resolution and np.max(nn) < 1.1 * g.resolution
  ga = equispacedS2Grid(resolution=5 * degree, antipodal=True)
  assert ga.antipodal and abs(ga.size / (g.size / 2) - 1) < 0.02
  assert max(np.arccos(np.max(bruteDots(ga, v))) for v in vs) < ga.resolution
  mask = g.z > 0.2
  s = g.subGrid(mask)
  assert isinstance(s, S2Grid) and s.size == np.sum(mask) and s.nrings < g.nrings
  assert all(abs(bruteDots(s, v)[s.find(v)[0]] - np.max(bruteDots(s, v))) < 1e-12 for v in vs[:50])
  s2 = g.subGrid(vector3d.Z, 20 * degree)
  assert np.all(angle(s2, vector3d.Z) <= 20 * degree + 1e-10)
  th, rh, _ = g.polar()
  assert np.max(np.abs(th - g.theta)) < 1e-12
  cs = crystalFrame("m-3m")
  gf = equispacedS2Grid(resolution=10 * degree, frame=cs)
  assert gf.frame is cs and gf[0].frame is cs and not isinstance(gf[0:5], S2Grid) and isinstance(gf[0:5], vector3d)
  r = rotation.byAxisAngle(vector3d.X, 0.3)
  assert not isinstance(r * gf, S2Grid) and isinstance(r * gf, vector3d) and not isinstance(gf + gf, S2Grid)
  sR = fundamentalSector(cs)
  gs = equispacedS2Grid(resolution=5 * degree, region=sR)
  assert gs.frame is cs and np.all(sR.checkInside(gs)) and abs(gs.size / (g.size / 48) - 1) < 0.25
  assert "points" in repr(g) and str(g) == f"{g.size} points"
  # the regular grid is the pole figure layout
  R = regularS2Grid(resolution=10 * degree, antipodal=True)
  assert R.shape == (36, 10) and R.antipodal
  R2 = regularS2Grid(theta=[0, 0.2, 0.4], rho=[0, 1, 2, 3])
  assert R2.shape == (4, 3) and abs(R2[1, 2].theta - 0.4) < 1e-12
  assert regularS2Grid(points=(72, 19)).shape == (72, 19)


def test_the_index_without_a_search():
  g = equispacedS2Grid(resolution=1 * degree)
  assert g.hasUniformSteps
  ws = vector3d.rand(2000, rng=np.random.default_rng(7))
  i, j = g.nearestStepwise(ws), g.find(ws)[0]
  assert np.all(angle(g[i], ws) < 2.5 * g.resolution) and np.mean(i == j) > 0.9
  sec = equispacedS2Grid(resolution=2 * degree, region=fundamentalSector(crystalFrame("m-3m")))
  for v in sec[::50]:
    assert sec.nearestStepwise(v) == sec.find(v)[0]


def test_the_sphere_grid_of_a_frame():
  cs = crystalFrame("432")
  full = equispacedS2Grid(resolution=3 * degree, frame=cs)
  sector = equispacedS2Grid(cs, resolution=3 * degree)
  assert sector.frame is cs and isinstance(sector, S2Grid)
  assert abs(sector.size * cs.numSym() / full.size - 1) < 0.1
  assert np.all(fundamentalSector(cs).checkInside(sector))
  axes = equispacedS2Grid(cs, resolution=3 * degree, antipodal=True)
  laue = equispacedS2Grid(crystalFrame("m-3m"), resolution=3 * degree)
  assert axes.antipodal and axes.size == laue.size and abs(2 * axes.size / sector.size - 1) < 0.1


def test_equispaced_default_resolution():
  # MATLAB's default spacing of an equispaced grid is 2.5 degree
  assert np.isclose(equispacedS2Grid(antipodal=True).resolution, 2.5 * degree, rtol=0.05)
