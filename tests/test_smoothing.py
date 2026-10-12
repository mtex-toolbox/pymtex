# M4: the staircase of a measured boundary turned into a smooth curve, the Julia SM1 to SM5
import numpy as np
import pytest

from mtex import EBSD, phaseList, crystalFrame, rotation, zvector, degree, calcGrains, smoothBoundary
from mtex import boundaryNetwork, laplaceFilter, taubinFilter, curvatureFilter, huberFilter, boundaryFilter
from conftest import close
from test_boundary import boundaryMap, SQ4


# a closed ring of n vertices, which is the network a filter's gain is stated on
def ringNetwork(n):
  nbr = np.array([[(i - 1) % n, (i + 1) % n] for i in range(n)]).reshape(-1)
  return boundaryNetwork.byArrays(np.arange(0, 2 * n + 1, 2), nbr, np.zeros(n, dtype=bool), np.arange(n))


def ringOf(n, R):
  t = 2 * np.pi * np.arange(n) / n
  return np.stack([R * np.cos(t), R * np.sin(t)], axis=1)


def meanRadius(V):
  return np.mean(np.linalg.norm(V, axis=1))


def test_SM1_the_network_is_what_may_move_and_what_may_not():
  g = calcGrains(boundaryMap(SQ4, 2, 10), angle=5 * degree)
  net = boundaryNetwork(g.boundary, g.innerBoundary)
  # every vertex of the boundary, and nothing else out of the pool of the map
  assert len(net) == len(np.unique(g.boundary.F)) and len(net) < len(g.boundary.V)
  assert np.all(net.degree >= 2) and all(net.numNeighbors(i) == len(net.neighbors(i)) for i in range(len(net)))
  # the border of the map is held, and so are the junctions; what is left to move is the
  # cross between the blocks, its four ends and its centre apart
  assert np.count_nonzero(~net.fixed) == 36
  rim = np.unique(g.boundary.F[g.boundary.grainId.min(axis=1) == 0])
  assert np.all(net.fixed[np.isin(net.used, rim)])
  free = boundaryNetwork(g.boundary, g.innerBoundary, moveOuterBoundary=True)
  assert np.count_nonzero(free.fixed) == 5           # four triple and one quadruple point
  assert not boundaryNetwork(g.boundary, g.innerBoundary, moveTriplePoints=True, moveOuterBoundary=True).fixed.any()
  # the Laplacian averages the neighbours and takes the vertex off
  L = net.laplacian()
  assert close(L.sum(axis=1), 0) and close(L.diagonal(), -1)
  assert "vertices" in repr(net)


def test_SM2_the_three_steps_leave_a_straight_boundary_where_it_is():
  g = calcGrains(boundaryMap(SQ4, 2, 10), angle=5 * degree)
  h = g.smoothBoundary(10)
  # the cross between the blocks is straight and its ends are junctions, so there is
  # nothing for the smoothing to move
  assert close(h.area, g.area) and close(h.area.sum(), 400) and close(h.perimeter, g.perimeter)
  # the fixed vertices are exactly where the first two steps left them
  h0 = smoothBoundary(g, 0)
  net = boundaryNetwork(h.boundary, h.innerBoundary)
  assert np.array_equal(h.boundary.V[net.used[net.fixed]], h0.boundary.V[net.used[net.fixed]])
  # simplifying and resampling are the first two steps and can be left out
  assert len(g.smoothBoundary(1, simplify=False, refine=False).boundary) == len(g.boundary)
  assert len(g.smoothBoundary(1, simplify=False, refine=False).boundary) == len(g.boundary)
  assert len(g.smoothBoundary(1).boundary) == 120     # simplified, then resampled at the pixel
  assert len(g.smoothBoundary(1, refine=0.5).boundary) == 240
  # the loops still index the one pool of vertices both boundaries share
  assert h.boundary.V is h.innerBoundary.V and np.array_equal(h.id, g.id)
  # a filter is the first argument as an iteration count is
  assert close(g.smoothBoundary(taubinFilter(3)).area.sum(), 400) and close(g.smoothBoundary(curvatureFilter()).area.sum(), 400)


def test_SM3_the_filters_have_the_gain_they_claim():
  n, R = 60, 10.0
  net, V = ringNetwork(n), ringOf(n, R)
  assert close(meanRadius(V), R)
  s2 = np.sin(np.pi / n) ** 2
  # a circle is the mode of the network at w = 2 pi / n, so every filter acts on it by its
  # gain there and nothing else happens
  for it in (1, 5, 25):
    assert close(meanRadius(laplaceFilter(it).smooth(V, net, 1)), R * (1 - 0.5 * s2) ** it, 1e-9)
    assert close(meanRadius(taubinFilter(it).smooth(V, net, 1)), R * ((1 - s2) * (1 + 1.06 * s2)) ** it, 1e-9)
  # the Laplacian shrinks without bound, Taubin gives the area back
  assert meanRadius(laplaceFilter(200).smooth(V, net, 1)) < 0.97 * R
  assert meanRadius(taubinFilter(200).smooth(V, net, 1)) > 0.999 * R
  # the variational filter is one solve, and its gain is 1 / (1 + 4 alpha sin^4(w/2))
  for alpha in (1.0, 100.0):
    assert close(meanRadius(curvatureFilter(alpha=alpha).smooth(V, net, 1)), R / (1 + 4 * alpha * s2 ** 2), 1e-9)
  assert np.array_equal(curvatureFilter(alpha=0).smooth(V, net, 1), V)
  with pytest.raises(ValueError):
    taubinFilter(mu=0.1).smooth(V, net, 1)
  with pytest.raises(NotImplementedError):
    boundaryFilter().smooth(V, net, 1)
  # a fixed vertex does not move
  net.fixed[0] = True
  assert np.array_equal(laplaceFilter(5).smooth(V, net, 1)[0], V[0]) and np.array_equal(curvatureFilter(5).smooth(V, net, 1)[0], V[0])


def test_SM4_a_smoothing_length_is_a_length_an_iteration_count_is_not():
  R = 10.0
  # the same circle sampled twice as densely: the wavelength damped to half amplitude is
  # stated in the units of the map, so the same length does the same thing to the same shape
  for n, m in ((60, 120), (100, 200)):
    h1, h2 = 2 * np.pi * R / n, 2 * np.pi * R / m
    V1, V2 = ringOf(n, R), ringOf(m, R)
    net1, net2 = ringNetwork(n), ringNetwork(m)
    # a wavelength of the whole circumference halves the circle itself
    F = curvatureFilter(2 * np.pi * R)
    assert close(F.weight(h1), 1 / (4 * np.sin(np.pi / n) ** 4), 1e-6)
    assert close(meanRadius(F.smooth(V1, net1, h1)), R / 2, 1e-9)
    assert close(meanRadius(F.smooth(V2, net2, h2)), R / 2, 1e-9)
    # a count of iterations at the same rate does not survive the resampling
    assert meanRadius(laplaceFilter(25).smooth(V1, net1, h1)) < meanRadius(laplaceFilter(25).smooth(V2, net2, h2)) - 0.05
  # and the smoothing length cannot be shorter than the sampling can represent
  assert curvatureFilter(0.1).weight(1.0) == curvatureFilter(2.0).weight(1.0)
  assert huberFilter(0.1).weight(1.0) == curvatureFilter(2.0).weight(1.0) and curvatureFilter(alpha=3).weight(1.0) == 3


def test_SM5_the_Huber_filter_leaves_a_facet_faceted():
  # a hexagon, each side sampled at ten vertices: the corners turn by 60 degrees and the
  # sides are straight
  R, m = 10.0, 10
  V = []
  for c in range(6):
    a = np.array([R * np.cos(np.pi * c / 3), R * np.sin(np.pi * c / 3)])
    b = np.array([R * np.cos(np.pi * (c + 1) / 3), R * np.sin(np.pi * (c + 1) / 3)])
    V += [a + (b - a) * k / m for k in range(m)]
  V = np.array(V)
  net = ringNetwork(6 * m)
  h = R / m
  # the quadratic penalty rounds the corner, the Huber one pays for it once and leaves it
  # where it is
  cv = curvatureFilter(4 * h).smooth(V, net, h)
  hb = huberFilter(4 * h).smooth(V, net, h)
  assert np.linalg.norm(cv[0]) < 0.985 * R and np.linalg.norm(hb[0]) > 0.995 * R
  # so the Huber result stays closer to the hexagon everywhere
  err = lambda W: np.sum(np.abs(np.linalg.norm(W, axis=1) - np.linalg.norm(V, axis=1)))
  assert err(hb) < err(cv) / 2
  # a threshold no corner reaches makes it the curvature filter again
  assert close(huberFilter(4 * h, threshold=np.pi).smooth(V, net, h), cv, 1e-8)
  assert "threshold" in repr(huberFilter()) and "mu" in repr(taubinFilter())


def test_SM6_a_subset_smooths_as_the_whole_does():
  # a two by two pixel grain at the junction of three neighbours; without the largest
  # neighbour the junction has degree two, and the chains meeting there have different
  # pairs of grains and so must not be welded into one walk
  n = 12
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  block = np.where((5 <= i) & (i <= 6) & (5 <= j) & (j <= 6), 4, np.where(i < 6, 1, np.where(j < 6, 2, 3)))
  pos = np.stack([i, j, np.zeros(n * n)], axis=-1).astype(float)
  e = EBSD(pos, rotation.byAxisAngle(zvector, block * 20 * degree), np.ones(n * n, dtype=np.int8), phaseList(crystalFrame("m-3m", mineral="t")))
  g = calcGrains(e, angle=5 * degree)
  assert len(g) == 4
  sub = g[g.id != g.id[np.argmax(g.numPixel)]]
  assert len(sub) == 3
  for sm in (sub.simplifyBoundary(), sub.refineBoundary(), sub.smoothBoundary(5)):
    assert all(len(ls) > 0 for ls in sm.loops) and np.all(sm.area > 0)
  sm = sub.smoothBoundary(5)
  tiny = int(np.argmin(sm.numPixel))
  assert sm.numPixel[tiny] == 4 and 0.5 < sm.area[tiny] / sub.area[tiny] < 1.5


def test_the_adjacency_lists_both_ends_of_every_edge_by_vertex():
  from mtex.maps.smoothing import _adjacency
  ptr, nbr = _adjacency(np.array([[0, 1], [1, 2], [2, 0], [1, 3]]), 4)
  assert np.array_equal(ptr, [0, 2, 5, 7, 8]) and np.array_equal(nbr, [1, 2, 2, 3, 0, 0, 1, 1])


def test_every_row_of_the_laplacian_sums_to_zero():
  net = boundaryNetwork.byArrays([0, 2, 4, 6], [1, 2, 0, 2, 0, 1], [False] * 3, [0, 1, 2])
  L = net.laplacian().toarray()
  assert np.allclose(L.sum(axis=1), 0) and np.allclose(np.diag(L), -1)
