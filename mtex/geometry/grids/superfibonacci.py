# the super-Fibonacci SO(3) grid (Alexa, "Super-Fibonacci Spirals", CVPR 2022): a
# low-discrepancy spiral of unit quaternions covering SO(3), trimmed to one fundamental
# region of the frame pair. An alternative to the ring grid of equispacedSO3Grid: trivial to
# generate at any count with no ring structure, though the ring grid packs slightly more
# evenly. It carries no ladder structure, so its search is the quaternion k-d tree of
# orientationtree.py, the same one every SO(3) grid uses.
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from ..kernels import superfib
from ..orientation import orientation
from ..quaternion import rotation
from ..units import degree
from ..display import display, header

# the two irrational rates of the spiral, and the covering constant calibrated so that
# resolution is the mean nearest-neighbour spacing (119088 points give 3.62 deg on SO(3))
_PHI = np.sqrt(2.0)
_PSI = 1.533751168755204288118041
_COVER = 119088 * (3.62 * degree) ** 3


# n low-discrepancy unit quaternions on SO(3)
def _superfib(n):
  return superfib(int(n), _PHI, _PSI)


class SO3FibonacciGrid(orientation):
  """the super-Fibonacci grid of orientations, a low-discrepancy quaternion spiral trimmed to one
  fundamental region of the frame pair, searched through the quaternion k-d tree; built by
  superfibonacciSO3Grid

    S3G = superfibonacciSO3Grid(cs, ss, resolution=5*degree)"""

  def __init__(self, rot, frameA, frameB, resolution):
    orientation.__init__(self, rot, frameA, frameB)
    self.res = float(resolution)
    self._otree = None

  # anything derived from the grid is a plain orientation list
  def _new(self, data, **attrs):
    obj = object.__new__(orientation)
    obj.data = data
    for name in self._carried:
      setattr(obj, name, attrs.pop(name, getattr(self, name)))
    if attrs:
      raise TypeError(f"orientation carries no attribute {', '.join(attrs)}")
    return obj

  @property
  def resolution(self):
    """the mean nearest-neighbour spacing the grid was built for, in radians"""
    return self.res

  def subGrid(self, mask):
    """the grid of the orientations a mask or an index array selects"""
    return SO3FibonacciGrid(rotation._fromData(self.data[mask]), self.frameA, self.frameB, self.res)

  def find(self, o, epsOrK=None):
    """nearest (eps None), k nearest (int) or all within a radius (float), through the tree"""
    from .orientationtree import findInGrid
    return findInGrid(self, o, epsOrK)

  def __repr__(self):
    return display(header("SO3FibonacciGrid", self._frameSummary()), [("size", self.size), ("resolution", f"{self.res / degree:.4g}°")])

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    @property
    def frameA(self) -> Any:
      ...
    @frameA.setter
    def frameA(self, value) -> None:
      ...
    @property
    def frameB(self) -> Any:
      ...
    @frameB.setter
    def frameB(self, value) -> None:
      ...


def superfibonacciSO3Grid(CS, SS=None, resolution=5 * degree, points=None):
  """a super-Fibonacci grid of orientations covering one fundamental region of the frame pair,
  its resolution the mean spacing of nearest neighbours

    S3G = superfibonacciSO3Grid(cs)                            # about 5 degree spacing
    S3G = superfibonacciSO3Grid(cs, ss, points=1000)           # about 1000 nodes in the region
    S3G = superfibonacciSO3Grid(cs, resolution=2.5*degree)     # a finer grid

  The nodes are the super-Fibonacci spiral of Alexa, a low discrepancy sequence of
  quaternions, trimmed to the fundamental region. The length of the spiral is set from the
  resolution, taken as the mean distance to the nearest neighbour; with points it is set from
  the fraction of SO(3) the region fills, which is measured on a probe spiral, so that any
  pair of symmetries calibrates itself. fibonacciSO3Grid is MATLAB's form, one node per
  resolution^3 of the volume.

  Input
    CS - crystalFrame
    SS - specimenFrame, by default the default specimen frame
  Output
    S3G - SO3FibonacciGrid, orientations from CS to SS
  Keywords
    resolution - the mean distance to the nearest node, in radians
    points - the approximate number of nodes in the fundamental region, instead of the
        resolution
  References
    M. Alexa, Super-Fibonacci spirals: fast, low-discrepancy sampling of SO(3), CVPR 2022,
        8291-8300
  See also
    fibonacciSO3Grid, equispacedSO3Grid, localOrientationGrid, fibonacciS2Grid"""
  from ..frames import specimenFrame
  from ..orientationregion import fundamentalRegion
  SS = specimenFrame.default if SS is None else SS
  oR = fundamentalRegion(CS, SS)
  # the fraction of SO(3) the region fills, measured once, so any symmetry calibrates itself
  probe = _superfib(200000)
  accept = float(np.mean(oR.checkInside(probe)))
  if points is not None:
    nFull = int(np.round(points / max(accept, 1e-6)))
    resolution = (_COVER / nFull) ** (1 / 3)
  else:
    nFull = int(np.round(_COVER / float(resolution) ** 3))
  q = _superfib(max(nFull, 1))
  q = q[oR.checkInside(q)]
  res = (_COVER / max(nFull, 1)) ** (1 / 3)
  return SO3FibonacciGrid(rotation._fromData(q), CS, SS, res)


def fibonacciSO3Grid(CS, SS=None, resolution=5 * degree, points=None, antipodal=False):
  """a low discrepancy grid of orientations from a super-Fibonacci spiral, restricted to the
  fundamental region of the frame pair; MATLAB's fibonacciSO3Grid

    S3G = fibonacciSO3Grid(cs)                                 # 5 degree resolution
    S3G = fibonacciSO3Grid(cs, ss, points=10000)               # about 10000 nodes in the region
    S3G = fibonacciSO3Grid(cs, ss, resolution=2.5*degree)      # a finer grid
    S3G = fibonacciSO3Grid(cs, cs, antipodal=True)             # misorientations with antipodal symmetry

  The nodes are the super-Fibonacci spiral of Alexa with round(8 pi^2 / resolution^3) nodes
  over SO(3), one node per resolution^3 of its volume, trimmed to the fundamental region. Unlike
  a lattice the number of nodes is free; points sets the resolution from the volume fraction
  of the fundamental region. To cover a ball around one orientation take
  localOrientationGrid.

  Input
    CS - crystalFrame
    SS - specimenFrame or crystalFrame, by default the default specimen frame
  Output
    S3G - SO3FibonacciGrid, orientations from CS to SS
  Keywords
    resolution - the node spacing, in radians
    points - the approximate number of nodes in the fundamental region, instead of the
        resolution
    antipodal - the fundamental region with the antipodal symmetry of misorientations, half
        the region
  References
    M. Alexa, Super-Fibonacci spirals: fast, low-discrepancy sampling of SO(3), CVPR 2022,
        8291-8300
  See also
    superfibonacciSO3Grid, equispacedSO3Grid, localOrientationGrid, fibonacciS2Grid"""
  from ..frames import specimenFrame
  from ..orientationregion import fundamentalRegion
  SS = specimenFrame.default if SS is None else SS
  # the volume fraction of the fundamental region, as in find
  V = 1 / CS.properGroup().numSym() / SS.properGroup().numSym() / (1 + bool(antipodal))
  if points is not None:
    resolution = (8 * np.pi ** 2 * V / points) ** (1 / 3)
  q = _superfib(max(int(round(8 * np.pi ** 2 / float(resolution) ** 3)), 1))
  q = q[fundamentalRegion(CS, SS, antipodal=antipodal).checkInside(q)]
  g = SO3FibonacciGrid(rotation._fromData(q), CS, SS, resolution)
  g.antipodal = bool(antipodal)
  return g


__all__ = ["SO3FibonacciGrid", "superfibonacciSO3Grid", "fibonacciSO3Grid"]
