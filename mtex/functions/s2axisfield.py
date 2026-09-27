# axis fields on the sphere, unoriented axes at every direction stored as the six components
# of a a^T (MATLAB @S2AxisFieldHarmonic; docs/plans/functions.md F7)
#
#   aF = S2AxisFieldHarmonic(fhat, cs), S2AxisFieldHarmonic(fun, bandwidth=), S2AxisFieldHarmonic(v, axes), S2AxisFieldHarmonic(vF)
#   aF(v)  the axes at the directions, the leading eigenvector of the reconstructed matrix
#   dot(aF, x), dot(aF, bF), angle(aF, x)
from __future__ import annotations

from typing import TYPE_CHECKING
import math

import numpy as np

from ..geometry.frames import referenceFrame
from ..geometry.session import getMTEXpref
from ..geometry.vector3d import vector3d
from .funops import mapFun, smoothBandwidth
from .harmonics import mtexCoefficientsS2, _WignerD
from .quadrature import quadratureS2Grid
from .s2fun import S2FunHarmonic, realizeSpherical
from .s2vectorfield import S2VectorField, _adjointColumns, _seriesColumns, _values, _vectorsFor
from .so3vectorfield import _projectSphericalRows, equivariantProjection
from .transforms import sphericalSeriesAdjoint

AXIS_COMPONENTS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def outerComponents(a):
  """the six components of a a^T for axes as (n, 3)"""
  return np.stack([a[:, i] * a[:, j] for i, j in AXIS_COMPONENTS], axis=-1)


def symmetricMatrices(c):
  M = np.empty((*c.shape[:-1], 3, 3))
  for k, (i, j) in enumerate(AXIS_COMPONENTS):
    M[..., i, j] = M[..., j, i] = c[..., k]
  return M


def symmetricAction(M):
  """the action of an orthogonal matrix on the components, T -> M T M^T"""
  A = np.zeros((6, 6))
  for k, (i, j) in enumerate(AXIS_COMPONENTS):
    E = np.zeros((3, 3))
    E[i, j] = E[j, i] = 1
    B = M @ E @ M.T
    A[:, k] = [B[p, q] for p, q in AXIS_COMPONENTS]
  return A


class S2AxisField:

  @property
  def CS(self):
    return self.frame

  def __call__(self, x):
    return self.eval(x)

  def _framed(self, v):
    return v if self.frame is None else vector3d(v.data, frame=self.frame)

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..plotting.sphericalplot import plot, quiver
    from ..plotting.spaceplot import plot3d


class S2AxisFieldHandle(S2AxisField):

  def __init__(self, fun, frame=None, *, bandwidth=None):
    self.fun, self.frame = fun, frame
    self.bandwidth = getMTEXpref("maxS2Bandwidth") if bandwidth is None else int(bandwidth)

  def eval(self, x):
    v = _vectorsFor(self, x).normalize()
    return vector3d(_values(self.fun(self._framed(v))).reshape(*v.shape, 3), antipodal=True)

  def __repr__(self):
    info = "" if self.frame is None else f" ({self.frame.char()})"
    return f"S2AxisFieldHandle{info}\n  bandwidth hint: {self.bandwidth}"


class S2AxisFieldHarmonic(S2AxisField):

  def __init__(self, fhat, *args, bandwidth=None, weights=None, symmetrise=True):
    frames = [a for a in args if isinstance(a, referenceFrame)]
    frame = frames[0] if frames else None
    if isinstance(fhat, (S2VectorField, S2AxisField)):
      v = fhat
      frame = v.frame
      L = v.bandwidth if bandwidth is None else int(bandwidth)
      g = quadratureS2Grid(L)
      values = _values(v(self_framed(v, g.nodes())))
      fhat = _adjointColumns(g, outerComponents(values), L, g.weights)
    elif isinstance(fhat, vector3d) and args and not isinstance(args[0], referenceFrame):
      pts, values = fhat, args[0]
      frame = pts.frame if frame is None else frame
      L = getMTEXpref("maxS2Bandwidth") if bandwidth is None else int(bandwidth)
      d = vector3d(pts.data.reshape(-1, 3)).normalize()
      w = np.full(d.size, 4 * np.pi / d.size) if weights is None else np.asarray(weights, dtype=float).ravel()
      fhat = _adjointColumns(d, outerComponents(_values(values)), L, w)
    elif callable(fhat) and not isinstance(fhat, np.ndarray):
      L = getMTEXpref("maxS2Bandwidth") if bandwidth is None else int(bandwidth)
      g = quadratureS2Grid(L)
      v = g.nodes()
      values = _values(fhat(v if frame is None else vector3d(v, frame=frame)))
      fhat = _adjointColumns(g, outerComponents(values), L, g.weights)
    fhat = np.array(fhat, dtype=complex)
    if fhat.ndim != 2 or fhat.shape[1] != 6:
      raise ValueError("an axis field has six components")
    if math.isqrt(fhat.shape[0]) ** 2 != fhat.shape[0]:
      raise ValueError("a coefficient matrix has (L+1)^2 rows for its bandwidth L")
    for i in range(6):
      realizeSpherical(fhat[:, i])
    self._fhat, self.frame = fhat, frame
    if symmetrise:
      self._fhat = self.symmetrise()._fhat

  @classmethod
  def quadrature(cls, f, *args, bandwidth=None, weights=None):
    """S2AxisFieldHarmonic.quadrature(f, bandwidth=L), MATLAB's name for the constructor"""
    return cls(f, *args, bandwidth=bandwidth, weights=weights)

  @classmethod
  def interpolate(cls, nodes, values, *, bandwidth=None, weights=None, **kwargs):
    """the least squares fit to axes at directions: the six components of v v^T one by one"""
    from .density import interpolate
    d = vector3d(nodes.data.reshape(-1, 3)).normalize()
    comps = outerComponents(_values(values))
    fs = [interpolate(d, comps[:, i], bandwidth=bandwidth, weights=weights, **kwargs) for i in range(6)]
    L = max(f.bandwidth for f in fs)
    return cls(np.stack([f.truncate(L)._fhat for f in fs], axis=-1), nodes.frame)

  def _rebuild(self, fhat):
    a = object.__new__(S2AxisFieldHarmonic)
    a._fhat, a.frame = np.asarray(fhat), self.frame
    return a

  @property
  def fhat(self):
    """the coefficients of the components in MTEX's basis, one column per component"""
    return mtexCoefficientsS2(np.moveaxis(self._fhat, 0, -1)).swapaxes(-1, 0) if self._fhat.ndim > 1 else mtexCoefficientsS2(self._fhat)

  @property
  def bandwidth(self):
    return math.isqrt(self._fhat.shape[0]) - 1

  def eval(self, x):
    """the leading eigenvector of the reconstructed matrix"""
    v = _vectorsFor(self, x).normalize()
    M = symmetricMatrices(_seriesColumns(self._fhat, v))
    _, vecs = np.linalg.eigh(M)
    return vector3d(vecs[..., :, 2].reshape(*v.shape, 3), antipodal=True)

  def symmetrise(self):
    """the projection onto the equivariant fields, a(s xi) = +- s a(xi)"""
    if self.frame is None or self.frame.sym.numSym() == 1:
      return self
    L = self.bandwidth
    els = self.frame.sym.rot.flatten()
    actions = [symmetricAction(els[i].matrix()) for i in range(els.size)]
    Ds = []
    for i in range(els.size):
      D = _WignerD(els[i].rot, bandwidth=L)
      Ds.append([(-1 if els[i].isImproper and l % 2 else 1) * D[l] for l in range(L + 1)])
    return self._rebuild(_projectSphericalRows(self._fhat.copy(), equivariantProjection(actions, Ds)))

  def dot(self, x):
    """the cosine of the angle to a constant axis, sqrt(x^T (a a^T) x), and to another field"""
    if isinstance(x, (S2AxisField, S2VectorField)):
      if x.frame is not self.frame:
        raise ValueError("the fields live in different frames")
      L = min(smoothBandwidth(self, x), getMTEXpref("maxS2Bandwidth"))
      g = quadratureS2Grid(L)
      xs = self._framed(g.nodes())
      values = np.abs(np.sum(_values(self(xs)) * _values(x(xs)), axis=-1))
      return S2FunHarmonic(sphericalSeriesAdjoint(g, values, L), self.frame, real=True, symmetrise=False, _internal=True)
    x = x if isinstance(x, vector3d) else vector3d(x)
    if x.frame is not None and self.frame is not None and x.frame is not self.frame:
      raise ValueError("the axis lives in another frame than the field")
    y = x.data.reshape(3)
    q = self._fhat @ np.array([y[i] * y[j] * (1 if i == j else 2) for i, j in AXIS_COMPONENTS])
    f = S2FunHarmonic(q, None if self.frame is None else self.frame.stripSym(), real=True, symmetrise=False, _internal=True)
    return mapFun(lambda t: np.sqrt(np.maximum(t, 0)), f, bandwidth=smoothBandwidth(f))

  def angle(self, x):
    return mapFun(lambda t: np.arccos(np.clip(t, -1, 1)), self.dot(x))

  def __repr__(self):
    info = "" if self.frame is None else f" ({self.frame.char()})"
    return f"S2AxisFieldHarmonic{info}\n  bandwidth: {self.bandwidth}"


def self_framed(v, nodes):
  return nodes if v.frame is None else vector3d(nodes.data, frame=v.frame)


__all__ = ["S2AxisFieldHarmonic", "S2AxisFieldHandle"]
