import numpy as np
import pytest

from mtex import mtexdata, plottingConvention


@pytest.fixture(scope="module")
def block():
  plottingConvention.default('y↓→x')
  return mtexdata('titanium')[9:16, 67:79]


# the cube coordinates of the titanium block are MATLAB's, ebsd(10:16,68:79).hex2cube
def test_hex2cube_matches_matlab(block):
  i, j = np.indices(block.shape)
  x, y, z = block.hex2cube(i, j)
  assert np.array_equal(x[:3, :4], [[0, 1, 2, 3], [-1, 0, 1, 2], [-1, 0, 1, 2]])
  assert np.array_equal(z[:3, :4], [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
  assert np.all(x + y + z == 0)


# cube2hex inverts hex2cube on the grid and gives NaN outside it
def test_cube2hex_inverts_hex2cube(block):
  i, j = np.indices(block.shape)
  row, col = block.cube2hex(*block.hex2cube(i, j))
  assert np.array_equal(row, i) and np.array_equal(col, j)
  row, col = block.cube2hex(-5, 5, 0)
  assert np.isnan(row) and np.isnan(col)
