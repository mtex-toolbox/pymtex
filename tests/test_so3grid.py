# the equispaced orientation grids and their coset-symmetrised search
import numpy as np
import pytest

from mtex import SO3Grid, equispacedSO3Grid, orientation, rotation, crystalFrame, specimenFrame, vector3d, degree, angle, dot

cs = crystalFrame("m-3m")
cs2 = crystalFrame("4/mmm", [3, 3, 4], mineral="other")

cases = [("-1", "1", 15), ("2/m", "1", 10), ("mmm", "1", 10), ("4/mmm", "1", 10), ("-3m", "1", 10),
         ("6/mmm", "1", 10), ("m-3", "1", 10), ("m-3m", "1", 7.5), ("m-3m", "222", 10), ("mmm", "mmm", 10), ("4/mmm", "mmm", 10)]


# a grid of each pair of groups, 60 random orientations and their dot products with every node
@pytest.fixture(scope="module", params=cases, ids=lambda c: f"{c[0]}|{c[1]}")
def grid(request):
  sym, ssym, res = request.param
  rng = np.random.default_rng(3)
  res = res * degree
  cs, ss = crystalFrame(sym), specimenFrame(ssym)
  g = equispacedSO3Grid(cs, ss, resolution=res)
  o = orientation.rand(60, cs, ss, rng=rng)
  d = dot(g[None, :], o[:, None])
  return dict(g=g, o=o, d=d, res=res, cs=cs, ss=ss)


def test_equispacedSO3Grid_hasTheResolutionAndFrames(grid):
  g, res = grid["g"], grid["res"]
  assert isinstance(g, SO3Grid) and abs(g.resolution / res - 1) < 0.1 and g.CS is grid["cs"] and g.SS is grid["ss"]


def test_find_nearestIsTheLargestDotProduct(grid):
  g, o, d = grid["g"], grid["o"], grid["d"]
  ind, dist = g.find(o)
  assert np.all(np.abs(np.max(d, axis=1) - d[np.arange(60), ind]) < 1e-10)
  assert np.allclose(dist, d[np.arange(60), ind])


def test_equispacedSO3Grid_coversUpToSymmetryBelowTheResolution(grid):
  d = grid["d"]
  assert np.max(2 * np.arccos(np.minimum(1, np.max(d, axis=1)))) < grid["res"]


def test_find_radiusGivesEveryNodeWithinIt(grid):
  g, o, d = grid["g"], grid["o"], grid["d"]
  eps = 1.5 * grid["res"]
  I, D = g.find(o, eps)
  assert I.shape == (g.size, 60)[::-1]
  for k in range(60):
    assert sorted(I[k].indices.tolist()) == np.flatnonzero(d[k] >= np.cos(eps / 2)).tolist()
    assert np.allclose(np.cos(D[k].data / 2), d[k][I[k].indices], atol=1e-10)


def test_equispacedSO3Grid_nodesAreHalfTheResolutionApart(grid):
  # no two nodes equivalent, and none closer than half the resolution
  g, res = grid["g"], grid["res"]
  In, Dn = g.find(g, res)
  for j in range(g.size):
    others = In[j].indices != j
    assert np.all(Dn[j].data[others] >= res / 2 - 1e-10)


# a grid whose left group is not abelian, the tetragonal crystal against the cubic specimen
@pytest.fixture(scope="module")
def nonAbelian():
  rng = np.random.default_rng(3)
  gm = equispacedSO3Grid(cs2, cs, resolution=10 * degree)
  om = orientation.rand(40, cs2, cs, rng=rng)
  return gm, om, dot(gm[None, :], om[:, None])


def test_find_nearestWithANonAbelianLeftGroup(nonAbelian):
  gm, om, dm = nonAbelian
  ind, _ = gm.find(om)
  assert np.all(np.abs(np.max(dm, axis=1) - dm[np.arange(40), ind]) < 1e-10)


def test_find_radiusWithANonAbelianLeftGroup(nonAbelian):
  gm, om, dm = nonAbelian
  Im, _ = gm.find(om, 15 * degree)
  for k in range(40):
    assert sorted(Im[k].indices.tolist()) == np.flatnonzero(dm[k] >= np.cos(7.5 * degree)).tolist()


def test_equispacedSO3Grid_byNumberOfPoints():
  gp = equispacedSO3Grid(cs, points=1000)
  assert 800 < gp.size < 1200


def test_equispacedSO3Grid_ballAroundACentre():
  o = orientation.rand(cs, rng=np.random.default_rng(3))
  gb = equispacedSO3Grid(cs, resolution=10 * degree, maxAngle=30 * degree, center=o)
  assert isinstance(gb, SO3Grid) and np.all(angle(gb, o) <= 30 * degree + 1e-10)


def test_subGrid_isAGridThatFindsItsNodes():
  g = equispacedSO3Grid(cs, resolution=10 * degree)
  s = g.subGrid(np.arange(g.size) < 200)
  assert isinstance(s, SO3Grid) and s.size == 200 and s.find(s[7])[0] == 7


def test_SO3Grid_sliceIsAPlainOrientation():
  g = equispacedSO3Grid(cs, resolution=10 * degree)
  assert not isinstance(g[0:10], SO3Grid) and isinstance(g[0:10], orientation)


def test_SO3Grid_rotationAboutZKeepsTheGrid():
  g = equispacedSO3Grid(cs, resolution=10 * degree)
  r = rotation.byAxisAngle(vector3d.Z, 0.3)
  assert isinstance(r * g, SO3Grid) and (r * g).find((r * g)[5])[0] == 5


def test_SO3Grid_rotationBreakingTheSpecimenSymmetryGivesOrientations():
  g2 = equispacedSO3Grid(cs, specimenFrame("222"), resolution=15 * degree)
  assert not isinstance(rotation.byAxisAngle(vector3d(1, 1, 0), 0.3) * g2, SO3Grid)


def test_equispacedSO3Grid_smallBallAroundTheIdentityIsNoGrid():
  assert not isinstance(equispacedSO3Grid(cs, resolution=5 * degree, maxAngle=10 * degree), SO3Grid)


def test_SO3Grid_reprShowsTheResolution():
  g = equispacedSO3Grid(cs, resolution=10 * degree)
  assert "resolution: 10°" in repr(g)
