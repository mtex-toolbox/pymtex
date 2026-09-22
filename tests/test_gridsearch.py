# the compiled nearest search of the S2 ring grid agrees with the Python search and a
# brute-force argmax, over resolutions and the antipodal flag
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
  v = vector3d.rand(3000, rng=rng)
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
