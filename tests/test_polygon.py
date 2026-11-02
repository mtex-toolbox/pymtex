# the polygons of a map: the ragged loop indices and the point in polygon test over the loops
import numpy as np

from mtex import setMTEXpref, getMTEXpref
from mtex.maps import polygon
from mtex.maps.polygon import _cyclicNext, _insideLoop, _withinRun


class loopSet(polygon.polygonSet):
  # outlines given by their loops over one vertex pool, in the plane of the map
  def __init__(self, V, loops):
    self.V, self.loops, self.size = np.asarray(V, dtype=float), loops, len(loops)

  def _as2d(self, q):
    return np.asarray(q, dtype=float).reshape(-1, 2)


# a 10 x 10 square with a 4 x 4 hole, a 2 x 2 square inside the hole, and an outline without loops
def squares():
  V = [[0, 0], [10, 0], [10, 10], [0, 10], [3, 3], [3, 7], [7, 7], [7, 3], [4, 4], [6, 4], [6, 6], [4, 6]]
  return loopSet(V, [[np.arange(4), np.arange(4, 8)], [np.arange(8, 12)], []])


def test_the_place_within_a_run_counts_from_zero_in_every_run():
  assert np.array_equal(_withinRun([3, 0, 2]), [0, 1, 2, 0, 1])


def test_the_next_entry_of_a_cyclic_run_wraps_to_its_first():
  assert np.array_equal(_cyclicNext([3, 2]), [1, 2, 0, 4, 3])


def test_a_point_is_inside_the_outer_loop_and_outside_the_holes():
  inside = squares().checkInside([[1, 1], [3.5, 3.5], [5, 5], [11, 5]])
  assert np.array_equal(inside, [[True, False, False], [False, False, False], [False, True, False], [False, False, False]])


# the point in outline test of NumPy, the fallback and the oracle of the compiled one
def checkInsideNumpy(s, q):
  before = getMTEXpref("kernels")
  setMTEXpref("kernels", "numpy")
  try:
    return s.checkInside(q)
  finally:
    setMTEXpref("kernels", before)


def test_the_test_in_blocks_is_the_test_loop_by_loop(monkeypatch):
  s = squares()
  q = np.random.default_rng(1).uniform(-1, 11, (50, 2))
  monkeypatch.setattr(polygon, "INSIDE_BLOCK", 30)
  expected = np.stack([_insideLoop(s.V, s.loops[0][0], q) & ~_insideLoop(s.V, s.loops[0][1], q), _insideLoop(s.V, s.loops[1][0], q),
                       np.zeros(len(q), dtype=bool)], axis=1)
  assert np.array_equal(checkInsideNumpy(s, q), expected)


# many small squares in a row, every second one with a hole, and the big squares around them
def manySquares():
  V, loops = [], []
  for i in range(40):
    x = 3.0 * i
    V += [[x, 0], [x + 2, 0], [x + 2, 2], [x, 2], [x + 0.5, 0.5], [x + 0.5, 1.5], [x + 1.5, 1.5], [x + 1.5, 0.5]]
    b = 8 * i
    loops.append([np.arange(b, b + 4)] + ([np.arange(b + 4, b + 8)] if i % 2 else []))
  return loopSet(V, loops)


def test_the_compiled_test_through_the_buckets_is_the_numpy_test():
  rng = np.random.default_rng(2)
  for s, q in ((squares(), rng.uniform(-1, 11, (500, 2))), (manySquares(), rng.uniform(-1, 121, (2000, 2)) * [1, 0.025])):
    assert np.array_equal(s.checkInside(q), checkInsideNumpy(s, q))


def test_findByLocation_gives_the_first_outline_holding_a_point_or_minus_one():
  s = squares()
  assert np.array_equal(s.findByLocation([[1, 1], [5, 5], [3.5, 3.5], [11, 5], [np.nan, 1]]), [0, 1, -1, -1, -1])


def test_the_boundary_size_counts_the_vertices_of_all_loops():
  assert np.array_equal(squares().boundarySize, [8, 4, 0])
