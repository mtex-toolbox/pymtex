# the offset and cube coordinates of a hexagonal grid, MATLAB's EBSDhex/hex2cube and cube2hex
import numpy as np

from ..geometry.install import install
from .ebsd import EBSD


# which matrix index carries the stagger and which way its first line leans, read off the
# positions so that it holds at any rotation of the map
def _hexLayout(ebsd):
  if ebsd.ndim != 2 or ebsd.lattice is None or ebsd.lattice.shape != "hex":
    raise ValueError("hex2cube and cube2hex need a map gridded on a hexagonal lattice")
  p = np.asarray(ebsd.pos.data, dtype=float)
  if min(ebsd.shape) < 3:
    raise ValueError("the grid is too small to tell its staggered direction, at least 3 x 3 cells")
  rowStep1, rowStep2 = p[1, 0] - p[0, 0], p[2, 0] - p[1, 0]
  colStep = p[0, 1] - p[0, 0]
  isRowAlignment = np.linalg.norm(rowStep1 - rowStep2) > 1e-6 * ebsd.lattice.dxy
  dense, stag = (colStep, rowStep1) if isRowAlignment else (rowStep1, colStep)
  offset = np.sign(stag @ dense / np.linalg.norm(dense)) or 1.0
  return bool(isRowAlignment), float(offset)


def hex2cube(ebsd, row, col=None):
  """the cube coordinates x, y, z of cells of a hexagonal grid, given by their row and column
  (counted from 0) or by their flat index; x + y + z = 0, and the six neighbours of a cell
  differ by one step in two of the three; MATLAB's EBSDhex/hex2cube

    x, y, z = hex2cube(ebsd, row, col)            # by row and column
    x, y, z = ebsd.hex2cube([0, 100, 200])        # by flat index

  A cell of a hexagonal grid is addressed by its row and column, the offset coordinates; the
  cube coordinates make the neighbours one step apart in two of three, see
  redblobgames.com/grids/hexagons. Which index is staggered, and which way, is read off the
  positions, so it holds at any rotation of the map. The indices count from 0 and the flat
  index runs in the order of the grid, where MATLAB counts from 1 in column major order.

  Input
    ebsd - EBSD gridded on a hexagonal lattice, at least 3 x 3 cells
    row - the rows of the cells, or their flat indices when col is not given
  Output
    x, y, z - the cube coordinates
  Keywords
    col - the columns of the cells
  See also
    cube2hex"""
  if col is None:
    row, col = np.unravel_index(np.asarray(row), ebsd.shape)
  row, col = np.asarray(row, dtype=float), np.asarray(col, dtype=float)
  isRowAlignment, offset = _hexLayout(ebsd)
  if isRowAlignment:
    z = row
    x = col - (row - offset * (np.rint(row) % 2 == 1)) / 2
  else:
    x = col
    z = row - (col - offset * (np.rint(col) % 2 == 1)) / 2
  return x, -x - z, z


def cube2hex(ebsd, x, y, z):
  """the row and column (counted from 0) of the cells with cube coordinates x, y, z, NaN for a
  cell outside the grid

    row, col = cube2hex(ebsd, x, y, z)"""
  x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
  isRowAlignment, offset = _hexLayout(ebsd)
  if isRowAlignment:
    row = z
    col = x + (z - offset * (np.rint(z) % 2 == 1)) / 2
  else:
    col = x
    row = z + (x - offset * (np.rint(x) % 2 == 1)) / 2
  outside = (row < 0) | (col < 0) | (row >= ebsd.shape[0]) | (col >= ebsd.shape[1])
  return np.where(outside, np.nan, row), np.where(outside, np.nan, col)


install(EBSD, hex2cube)
install(EBSD, cube2hex)

__all__ = ["hex2cube", "cube2hex"]
