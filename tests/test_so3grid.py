# the equispaced orientation grids and their coset-symmetrised search
import numpy as np
import pytest

from mtex import SO3Grid, equispacedSO3Grid, orientation, rotation, crystalFrame, specimenFrame, vector3d, degree, angle, dot

rng = np.random.default_rng(3)


@pytest.mark.parametrize("case", [("-1", "1", 15), ("2/m", "1", 10), ("mmm", "1", 10), ("4/mmm", "1", 10), ("-3m", "1", 10),
                                  ("6/mmm", "1", 10), ("m-3", "1", 10), ("m-3m", "1", 7.5), ("m-3m", "222", 10), ("mmm", "mmm", 10), ("4/mmm", "mmm", 10)])
def test_equispaced_orientation_grids(case):
  sym, ssym, res = case
  res = res * degree
  cs, ss = crystalFrame(sym), specimenFrame(ssym)
  g = equispacedSO3Grid(cs, ss, resolution=res)
  assert isinstance(g, SO3Grid) and abs(g.resolution / res - 1) < 0.1 and g.CS is cs and g.SS is ss
  o = orientation.rand(60, cs, ss, rng=rng)
  d = dot(g[None, :], o[:, None])
  ind, dist = g.find(o)
  assert np.all(np.abs(np.max(d, axis=1) - d[np.arange(60), ind]) < 1e-10)
  assert np.allclose(dist, d[np.arange(60), ind])
  # coverage up to symmetry, below the resolution
  assert np.max(2 * np.arccos(np.minimum(1, np.max(d, axis=1)))) < res
  eps = 1.5 * res
  I, D = g.find(o, eps)
  assert I.shape == (g.size, 60)[::-1]
  for k in range(60):
    assert sorted(I[k].indices.tolist()) == np.flatnonzero(d[k] >= np.cos(eps / 2)).tolist()
    assert np.allclose(np.cos(D[k].data / 2), d[k][I[k].indices], atol=1e-10)
  # no two nodes equivalent, and none closer than half the resolution
  In, Dn = g.find(g, res)
  for j in range(g.size):
    others = In[j].indices != j
    assert np.all(Dn[j].data[others] >= res / 2 - 1e-10)


def test_non_abelian_left_group_and_the_forms():
  cs, cs2 = crystalFrame("m-3m"), crystalFrame("4/mmm", [3, 3, 4], mineral="other")
  gm = equispacedSO3Grid(cs2, cs, resolution=10 * degree)
  om = orientation.rand(40, cs2, cs, rng=rng)
  dm = dot(gm[None, :], om[:, None])
  ind, _ = gm.find(om)
  assert np.all(np.abs(np.max(dm, axis=1) - dm[np.arange(40), ind]) < 1e-10)
  Im, _ = gm.find(om, 15 * degree)
  for k in range(40):
    assert sorted(Im[k].indices.tolist()) == np.flatnonzero(dm[k] >= np.cos(7.5 * degree)).tolist()
  g = equispacedSO3Grid(cs, resolution=10 * degree)
  gp = equispacedSO3Grid(cs, points=1000)
  assert 800 < gp.size < 1200
  o = orientation.rand(cs, rng=rng)
  gb = equispacedSO3Grid(cs, resolution=10 * degree, maxAngle=30 * degree, center=o)
  assert isinstance(gb, SO3Grid) and np.all(angle(gb, o) <= 30 * degree + 1e-10)
  s = g.subGrid(np.arange(g.size) < 200)
  assert isinstance(s, SO3Grid) and s.size == 200 and s.find(s[7])[0] == 7
  assert not isinstance(g[0:10], SO3Grid) and isinstance(g[0:10], orientation)
  r = rotation.byAxisAngle(vector3d.Z, 0.3)
  assert isinstance(r * g, SO3Grid) and (r * g).find((r * g)[5])[0] == 5
  g2 = equispacedSO3Grid(cs, specimenFrame("222"), resolution=15 * degree)
  assert not isinstance(rotation.byAxisAngle(vector3d(1, 1, 0), 0.3) * g2, SO3Grid)
  assert not isinstance(equispacedSO3Grid(cs, resolution=5 * degree, maxAngle=10 * degree), SO3Grid)
  assert "resolution: 10°" in repr(g)
