# a function on the sphere by its values at the vertices of a spherical triangulation,
# linear in between (MATLAB S2Fun/@S2FunTri)
#
#   sF = S2FunTri(v, values)           the values at the directions v
#   sF = S2FunTri(sT, values)          on a triangulation already made
#   sF = S2FunTri(fun)                 a callable or function sampled on a 1.5 degree grid
#   interp(v, values, 'linear')        the same, MATLAB's name for it
#
# The value at a direction is the barycentric mean of the values at the corners of the
# triangle it lies in, so the function reproduces its values at the vertices exactly.
from __future__ import annotations

import numpy as np

from ..geometry.s2triangulation import S2Triangulation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d


class S2FunTri:

  def __init__(self, nodes, values=None):
    if callable(nodes) and not isinstance(nodes, (vector3d, S2Triangulation)):
      from ..geometry.grids.s2grid import equispacedS2Grid
      fun, nodes = nodes, equispacedS2Grid(resolution=1.5 * degree)
      values = fun(nodes)
    if isinstance(nodes, S2Triangulation):
      self.tri = nodes
      values = np.asarray(values)
    else:
      # a direction given more than once takes the mean of its values
      V = nodes.flatten().normalize().data.reshape(-1, 3)
      values = np.asarray(values).reshape(len(V), -1)
      key, first, inv = np.unique(np.round(V, 12), axis=0, return_index=True, return_inverse=True)
      inv = inv.ravel()
      count = np.bincount(inv, minlength=len(key))
      values = np.stack([np.bincount(inv, weights=values[:, k], minlength=len(key)) / count for k in range(values.shape[1])], axis=-1)
      self.tri = S2Triangulation(vector3d(V[first], frame=nodes.frame))
    self.values = values.reshape(self.tri.vertices.size, -1)
    self.antipodal, self.isReal = False, not np.iscomplexobj(self.values)

  @property
  def vertices(self):
    return self.tri.vertices

  @property
  def frame(self):
    return self.tri.frame

  @property
  def CS(self):
    return self.frame

  def __len__(self):
    return self.values.shape[1]

  def eval(self, v):
    v = v if isinstance(v, vector3d) else vector3d(v)
    shape = v.shape
    W = v.data.reshape(-1, 3)
    ok = ~np.isnan(W).any(axis=1)
    f = np.full((len(W), self.values.shape[1]), np.nan, dtype=self.values.dtype if self.values.dtype.kind == "c" else float)
    if ok.any():
      f[ok] = self.tri.calcBario(vector3d(W[ok])) @ self.values
    return f.reshape(shape) if self.values.shape[1] == 1 else f.reshape(shape + (-1,))

  def __call__(self, v):
    return self.eval(v)

  # the mean by the area of the Voronoi cell of every vertex
  def mean(self):
    from .polefigure import _voronoiWeights
    w = _voronoiWeights(self.vertices)
    w = np.asarray(w, dtype=float).ravel()
    m = (w[:, None] * self.values).sum(axis=0) / w.sum()
    return m[0] if len(m) == 1 else m

  def _mapped(self, op):
    f = object.__new__(S2FunTri)
    f.tri, f.values, f.antipodal = self.tri, op(self.values), self.antipodal
    f.isReal = not np.iscomplexobj(f.values)
    return f

  def __add__(self, g):
    return self._mapped(lambda y: y + (g.eval(self.vertices).reshape(y.shape) if hasattr(g, "eval") else g))

  __radd__ = __add__

  def __mul__(self, g):
    return self._mapped(lambda y: y * (g.eval(self.vertices).reshape(y.shape) if hasattr(g, "eval") else g))

  __rmul__ = __mul__

  def __neg__(self):
    return self._mapped(lambda y: -y)

  def __sub__(self, g):
    return self + (-1 * g)

  def __rsub__(self, g):
    return (-1 * self) + g

  def __truediv__(self, g):
    return self._mapped(lambda y: y / (g.eval(self.vertices).reshape(y.shape) if hasattr(g, "eval") else g))

  def __pow__(self, p):
    return self._mapped(lambda y: y ** p)

  def __abs__(self):
    return self._mapped(np.abs)

  def abs(self):
    return abs(self)

  def sqrt(self):
    return self._mapped(np.sqrt)

  def __repr__(self):
    return f"S2FunTri\n  vertices: {self.vertices.size}" + (f"\n  size: {len(self)}" if len(self) > 1 else "")


__all__ = ["S2FunTri"]
