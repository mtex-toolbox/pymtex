import numpy as np

from mtex.geometry.grids.s1grid import S1Grid


def test_periodic_and_bounded():
  # the periodic wrap of MATLAB check_mex: points on [0, 10) and a query near 0
  g = S1Grid([1, 2, 3, 4, 9, 9.8], min=0, max=10, periodic=True)
  assert g.nearest(0.1) == 5 and g.nearest(9.9) == 5 and g.nearest(4.4) == 3 and g.nearest(10.5) == 0
  assert g.nearest([0.1, 4.4]).tolist() == [5, 3]
  assert g.neighbours(0.5, 1.0).tolist() == [0, 5] and g.neighbours(2.5, 1.0).tolist() == [1, 2]
  assert g.neighbours(5, 100).tolist() == [0, 1, 2, 3, 4, 5]
  h = S1Grid([0.0, 0.5, 1.0], min=0, max=1)
  assert h.nearest(0.9) == 2 and h.neighbours(0.5, 0.6).tolist() == [0, 1, 2] and h.neighbours(-2, 0.5).size == 0
  assert len(g.subGrid([True, False, True, False, True, False])) == 3 and g.period == 10 and h.period == 0


def test_ladder_index():
  s = S1Grid(np.arange(0, 2.01, 0.25))
  assert s.step == 0.25 and s.ladderIndex(0.0) == 0 and s.ladderIndex(0.26) == 1 and s.ladderIndex(1.99) == 8
  assert s.ladderIndex(-5.0) == 0 and s.ladderIndex(5.0) == 8
  p = S1Grid(np.arange(0, 2 * np.pi - 1e-9, np.pi / 3), min=0, max=2 * np.pi, periodic=True)
  assert p.ladderIndex(0.0) == 0 and p.ladderIndex(2 * np.pi - 1e-9) == 0 and p.ladderIndex(np.pi) == 3
  u = S1Grid([0.0, 0.1, 0.7, 2.0])
  assert u.step == 0 and u.ladderIndex(0.69) == 2
  assert "points: 9" in repr(s)
