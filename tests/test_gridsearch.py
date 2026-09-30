# the compiled searches of the ring grids: the S2 nearest search against the Python search
# and a brute-force argmax, the SO(3) radius search against the quaternion k-d tree
import numpy as np
import pytest

import mtex
from mtex import equispacedS2Grid, setMTEXpref, vector3d

rng = np.random.default_rng(5)
pytest.importorskip("numba")


@pytest.fixture(autouse=True)
def numbaOn():
  setMTEXpref("kernels", "numba")
  yield
  setMTEXpref("kernels", "numba")


@pytest.mark.parametrize("res,antipodal", [(2, False), (3, True), (6, False)])
def test_s2_nearest_matches_python_and_brute(res, antipodal):
  g = equispacedS2Grid(resolution=res * mtex.degree, antipodal=antipodal)
  v = vector3d.rand(1000, rng=rng)
  setMTEXpref("kernels", "numpy")
  iN, dN = mtex.find(g, v)
  setMTEXpref("kernels", "numba")
  iK, dK = mtex.find(g, v)
  D = np.abs(v.data @ g.data.T) if antipodal else v.data @ g.data.T
  assert np.array_equal(iK, iN) and np.allclose(dK, dN, atol=1e-12)
  tie = np.take_along_axis(D, iK[:, None], axis=1)[:, 0]
  assert np.all(tie >= D.max(axis=1) - 1e-9)


def test_s2_nearest_shapes_and_single():
  g = equispacedS2Grid(resolution=5 * mtex.degree)
  v = vector3d.rand((7, 9), rng=rng)
  ind, d = mtex.find(g, v)
  assert ind.shape == (7, 9) and d.shape == (7, 9)
  i0, d0 = mtex.find(g, v[0, 0])
  assert np.ndim(i0) == 0 and ind[0, 0] == i0


# the incidence and cos(w/2) of the structured search and of the tree are the same
def sameAsTree(g, q, eps):
  from mtex.geometry.grids.orientationtree import OrientationTree
  I1, C1 = g.radius(q, eps)
  I0, C0 = OrientationTree(g).radius(q, eps)
  assert (I1 != I0).nnz == 0 and I1.nnz > 0
  assert abs(C1 - C0).max() < 1e-12 and I1.has_sorted_indices


@pytest.mark.parametrize("CS,SS,res,eps", [("m-3m", None, 5, 20), ("6/mmm", None, 5, 10), ("-1", None, 10, 15),
                                           ("mmm", "mmm", 5, 10), ("432", "222", 5, 10), ("-3m", None, 5, 12), ("4/m", None, 5, 30)])
def test_so3_radius_matches_the_tree(CS, SS, res, eps):
  g = mtex.equispacedSO3Grid(mtex.crystalFrame(CS), None if SS is None else mtex.specimenFrame(SS), resolution=res * mtex.degree)
  sameAsTree(g, mtex.orientation(mtex.rotation.rand(200, rng=rng), g.frameA, g.frameB), eps * mtex.degree)


def test_so3_radius_antipodal_turned_and_foreign_queries():
  cs = mtex.crystalFrame("m-3m")
  # grain exchange: the inverse of every query searched too
  g = mtex.equispacedSO3Grid(cs, cs, resolution=5 * mtex.degree)
  sameAsTree(g, mtex.orientation(mtex.rotation.rand(100, rng=rng), cs, cs, antipodal=True), 10 * mtex.degree)
  # a grid turned about its centre keeps its structure
  g = mtex.rotation.byAxisAngle(vector3d(1, 2, 3), 0.4) * mtex.equispacedSO3Grid(cs, resolution=5 * mtex.degree)
  assert isinstance(g, mtex.SO3Grid) and g.center.angle() > 0.3
  sameAsTree(g, mtex.orientation(mtex.rotation.rand(100, rng=rng), g.frameA, g.frameB), 10 * mtex.degree)
  # queries of other frames go to the tree, which symmetrises by their own groups
  q = mtex.orientation(mtex.rotation.rand(50, rng=rng), mtex.crystalFrame("432"))
  I, C = g.radius(q, 10 * mtex.degree)
  assert I.shape == (50, g.size)
