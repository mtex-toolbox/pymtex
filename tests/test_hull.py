# G7: the convex hull in the plane, the Julia H1 of shape.jl
import numpy as np

from mtex import convexHull, convexHullLoops, getMTEXpref, setMTEXpref
from mtex.maps.polygon import signedArea2


def test_H1_the_convex_hull():
  pts = np.array([[0.0, 0], [2, 0], [2, 2], [0, 2], [1, 1]])       # the last one is inside
  h = convexHull(pts)
  assert len(h) == 4 and 4 not in h and signedArea2(pts[h]) > 0    # anticlockwise
  # collinear points and repeated points do not become corners
  assert len(convexHull([[0.0, 0], [1, 0], [2, 0], [1, 1]])) == 3
  assert len(convexHull([[0.0, 0], [1, 0], [1, 0], [1, 1], [0, 1], [0, 0]])) == 4
  assert len(convexHull([[0.0, 0]])) == 1 and len(convexHull(np.zeros((0, 2)))) == 0
  # the hull of random points contains every point
  rng = np.random.default_rng(1)
  P = rng.normal(size=(300, 2))
  h = convexHull(P)
  H = P[h]
  e = np.roll(H, -1, axis=0) - H
  for q in P:
    assert np.all(e[:, 0] * (q[1] - H[:, 1]) - e[:, 1] * (q[0] - H[:, 0]) >= -1e-12)


def test_the_numba_and_the_numpy_chain_agree():
  rng = np.random.default_rng(2)
  before = getMTEXpref("kernels", "numba")
  Q = [rng.normal(size=(500, 2)), rng.integers(0, 5, (200, 2)).astype(float), np.array([[0.0, 0], [1, 0], [2, 0]])]
  try:
    out = {}
    for kernels in ("numba", "numpy"):
      setMTEXpref("kernels", kernels)
      out[kernels] = [convexHull(q) for q in Q]
  finally:
    setMTEXpref("kernels", before)
  assert all(np.array_equal(a, b) for a, b in zip(out["numba"], out["numpy"]))


def test_the_hulls_of_many_loops_are_the_hulls_of_each():
  rng = np.random.default_rng(5)
  V = rng.normal(size=(400, 2))
  V[100:150] = np.round(V[100:150] * 2) / 2                              # some repeated coordinates
  loops = [rng.choice(400, size=m, replace=False) for m in (1, 2, 3, 7, 40, 120)] + [np.arange(100, 150)]
  before = getMTEXpref("kernels", "numba")
  try:
    setMTEXpref("kernels", "numba")
    a = convexHullLoops(V, loops)
    setMTEXpref("kernels", "numpy")
    b = convexHullLoops(V, loops)
  finally:
    setMTEXpref("kernels", before)
  for l, x, y in zip(loops, a, b):
    assert np.array_equal(x, y) and np.array_equal(x, l[convexHull(V[l])])
  assert convexHullLoops(V, []) == []
