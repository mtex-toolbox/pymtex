"""Window filters for EBSD orientation maps."""
from .variational import EBSDFilter


class windowFilter(EBSDFilter):
  """the base of the filters over a window of neighbours, numNeighbours the rings of the lattice
  the window reaches: meanFilter, medianFilter, KuwaharaFilter"""

  def __init__(self, numNeighbours=1):
    self.numNeighbours = int(numNeighbours)

  def _items(self):
    return [("neighbours", self.numNeighbours)]


class meanFilter(windowFilter):
  """the mean of the neighbourhood, the fastest filter and the one that blurs a boundary most

  Input
    numNeighbours - number of lattice-neighbour rings in the window; default 1"""
  pass


class medianFilter(windowFilter):
  """the componentwise median of the neighbourhood, which takes a wild pixel out without
  pulling the boundary after it

  Input
    numNeighbours - number of lattice-neighbour rings in the window; default 1"""
  pass


class KuwaharaFilter(windowFilter):
  """the mean of the most homogeneous of the four quadrants of the window, which smooths within
  a grain and leaves a step where one is

  Input
    numNeighbours - number of lattice-neighbour rings in the window; default 1"""
  pass
