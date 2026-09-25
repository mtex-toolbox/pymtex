# the super-Fibonacci SO(3) grid covers one fundamental region at the asked resolution and
# every random orientation has a grid point within a small multiple of it
import numpy as np
import pytest

import mtex
from mtex import crystalFrame, orientation, superfibonacciSO3Grid


@pytest.mark.parametrize("name", ["m-3m", "222", "-3m"])
def test_covers_the_fundamental_region(name):
  cs = crystalFrame(name)
  g = superfibonacciSO3Grid(cs, resolution=5 * mtex.degree)
  # every random orientation is within about one resolution of a grid point (good covering)
  o = orientation.rand(5000, cs, rng=np.random.default_rng(0))
  _, c = mtex.find(g, o)
  d = 2 * np.arccos(np.minimum(c, 1))
  assert np.degrees(d.max()) < 12 and np.degrees(np.mean(d)) < 6


def test_points_option_sets_the_count():
  cs = crystalFrame("m-3m")
  g = superfibonacciSO3Grid(cs, points=3000)
  assert 2400 < g.size < 3600


def test_subgrid_keeps_the_type():
  cs = crystalFrame("m-3m")
  g = superfibonacciSO3Grid(cs, resolution=6 * mtex.degree)
  h = g.subGrid(g.angle(orientation.id(cs)) < 30 * mtex.degree)
  assert type(h) is type(g) and h.size < g.size
