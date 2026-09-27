import numpy as np

from mtex import fibonacciS2Grid, vector3d, degree


def test_fibonacci_grid():
  rng = np.random.default_rng(2)
  g = fibonacciS2Grid(3001)
  vs = vector3d.rand(200, rng=rng)
  d = [g.data @ v.data for v in vs]
  ind, dist = g.find(vs)
  assert g.size == 3001 and all(abs(d[k][ind[k]] - np.max(d[k])) < 1e-12 for k in range(200))
  eps = 6 * degree
  assert all(g.neighbourIndices(vs[k], eps).tolist() == np.flatnonzero(d[k] >= np.cos(eps)).tolist() for k in range(200))
  I, D = g.find(vs, eps)
  assert I.shape == (200, 3001) and np.allclose(np.cos(D[3].data), d[3][I[3].indices])
  assert max(np.arccos(np.max(d[k])) for k in range(200)) < g.resolution
  assert fibonacciS2Grid(resolution=5 * degree).size == 2 * ((int(np.round(np.exp(2.512) / (5 * degree) ** 2)) - 1) // 2) + 1
  assert not isinstance(g[:5], fibonacciS2Grid) and "points    : 3001" in repr(g)
