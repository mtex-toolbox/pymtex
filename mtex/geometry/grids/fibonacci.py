# the spherical Fibonacci grid (MATLAB fibonacciS2Grid): N = 2n + 1 points with z linear
# in the index and the azimuth advancing by the golden angle; an eps-band in z is an index
# range, so a search computes azimuths for the band only
from __future__ import annotations

import numpy as np

from ..units import degree
from ..vector3d import vector3d, asVector3d
from ..display import display


class fibonacciS2Grid(vector3d):
  """the spherical Fibonacci grid: N = 2n + 1 points along a spiral, z linear in the index and
  the azimuth advancing by the golden angle; by default the count fitting a resolution of 5 degree

    S2G = fibonacciS2Grid(1000)
    S2G = fibonacciS2Grid(resolution=2*degree)

  Input
    N - requested number of nodes, rounded to a nearby odd count
    resolution - target angular spacing in radians when N is omitted; default 5*degree"""

  def __init__(self, N=None, resolution=None):
    # the count MATLAB fits to the resolution, rounded once to the half count
    if N is None:
      resolution = 5 * degree if resolution is None else resolution
      N = np.exp(2.512) / resolution ** 2
    n = max(int(np.round((N - 1) / 2)), 0)
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n, -n - 1, -1)
    z = 2 * i / (2 * n + 1)
    rho = np.mod(2 * np.pi / golden * i, 2 * np.pi)
    r = np.sqrt(1 - z ** 2)
    vector3d.__init__(self, np.stack([r * np.cos(rho), r * np.sin(rho), z], axis=-1), isNormalized=True)
    self.n = n

  def _new(self, data, **attrs):
    obj = object.__new__(vector3d)
    obj.data = data
    for name in self._carried:
      setattr(obj, name, attrs.pop(name, getattr(self, name)))
    return obj

  @property
  def resolution(self):
    """the resolution of the grid, the spacing its number of points gives"""
    return np.sqrt(np.exp(2.512) / self.size)

  # the indices with |z - z_i| <= dz, a contiguous range since z_i falls with the index
  def _zBand(self, z, dz):
    N = 2 * self.n + 1
    lo = max(self.n - int(np.floor((z + dz) * N / 2)), 0)
    hi = min(self.n - int(np.ceil((z - dz) * N / 2)), N - 1)
    return np.arange(lo, hi + 1)

  def neighbourIndices(self, v, eps):
    """the indices of the grid points within the angle eps of the direction v"""
    d = asVector3d(v).normalize().data
    band = self._zBand(d[2], 2 * np.sin(eps / 2))
    return band[self.data[band] @ d >= np.cos(eps)]

  def find(self, v, eps=None):
    """the nearest grid point of every direction and its angle, or with eps the sparse matrices of
    the points within eps and their angles; MATLAB's fibonacciS2Grid.find

      ind, d = S2G.find(v)              # the nearest grid point and the angle to it
      I, d = S2G.find(v, 10*degree)     # every grid point within 10 degree

    The search runs through a k-d tree of the grid points. On an antipodal grid a direction and its antipode are the same.

    Input
      S2G - fibonacciS2Grid
      v - vector3d, the directions searched for
    Output
      ind - the index of the nearest grid point, of the shape of v
      d - its angle to v; with eps a sparse matrix (directions x grid) of the angles
      I - with eps a sparse boolean matrix (directions x grid) of the grid points within eps
    Keywords
      eps - the radius of the search; by default only the nearest point is returned
    See also
      fibonacciS2Grid, S2Grid.find, vector3d.find"""
    from .search import s2Tree
    v = asVector3d(v)
    xyz = v.data.reshape(-1, 3)
    if eps is None:
      ind, d = s2Tree(self).nearest(xyz)
      return ind.reshape(v.shape), d.reshape(v.shape)
    return s2Tree(self).radius(xyz, eps)

  def __repr__(self):
    return display("fibonacciS2Grid", [("points", self.size), ("resolution", f"{self.resolution / degree:.4g}°")])


__all__ = ["fibonacciS2Grid"]
