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


def test_fibonacciSO3Grid_is_matlabs():
  # MATLAB's fibonacciSO3Grid (909ba8263): round(8 pi^2 / res^3) nodes over SO(3), trimmed to
  # the region; the counts and the first node are MATLAB develop's
  from mtex import crystalFrame, degree, fibonacciSO3Grid
  cs = crystalFrame("m-3m")
  assert fibonacciSO3Grid(cs).size == 4994
  g = fibonacciSO3Grid(cs, points=5000)
  assert g.size == 5034 and abs(g.resolution / degree - 4.9834) < 1e-4
  g10 = fibonacciSO3Grid(cs, resolution=10 * degree)
  assert g10.size == 619 and np.allclose(g10.data[0], [-0.86406675, -0.07456694, 0.34987243, 0.35414363])
  # the antipodal region holds half the nodes; MATLAB counts 2503, four more on the boundary
  ga = fibonacciSO3Grid(cs, antipodal=True)
  assert ga.antipodal and abs(ga.size - 2503) <= 5
