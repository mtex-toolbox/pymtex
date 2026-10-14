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
from ..geometry.vector3d import vector3d, asVector3d
from .funops import mapFun, smoothBandwidth
from .harmonics import mtexCoefficientsS2
from .quadrature import quadratureS2Grid
from .s2fun import S2FunHarmonic, realizeSpherical
from .s2vectorfield import S2VectorField, _adjointColumns, _parityWignerD, _seriesColumns, _values, _vectorsFor
from .so3vectorfield import _projectSphericalRows, equivariantProjection
from .transforms import sphericalSeriesAdjoint
from ..geometry.display import aligned, frameLabel

AXIS_COMPONENTS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
# the rows and columns of the components, and the component at every entry of the matrix
_ROWS, _COLUMNS = np.array(AXIS_COMPONENTS).T
_COMPONENT = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])


def outerComponents(a):
  """the six components of a a^T for axes as (n, 3)"""
  return a[:, _ROWS] * a[:, _COLUMNS]


def symmetricMatrices(c):
  """the symmetric 3 x 3 matrices of the six components xx, xy, xz, yy, yz, zz, as (..., 3, 3)"""
  return c[..., _COMPONENT]


def symmetricAction(M):
  """the action of an orthogonal matrix on the components, T -> M T M^T"""
  A = np.zeros((6, 6))
  for k, (i, j) in enumerate(AXIS_COMPONENTS):
    E = np.zeros((3, 3))
    E[i, j] = E[j, i] = 1
    B = M @ E @ M.T
    A[:, k] = [B[p, q] for p, q in AXIS_COMPONENTS]
  return A


# directions in a frame, unchanged without one
def _inFrame(v, frame):
  return v if frame is None else vector3d(v.data, frame=frame)


class S2AxisField:
  """the axis fields on the sphere, an unoriented axis at every direction: S2AxisFieldHarmonic,
  S2AxisFieldHandle"""

  @property
  def CS(self):
    """the frame of the directions the field lives on, MATLAB's aF.CS"""
    return self.frame

  def __call__(self, x):
    return self.eval(x)

  def _framed(self, v):
    return _inFrame(v, self.frame)

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..plotting.sphericalplot import plot, quiver
    from ..plotting.spaceplot import plot3d


class S2AxisFieldHandle(S2AxisField):
  """an axis field on the sphere given by a function of the directions, with a bandwidth hint
  for its expansion

    aF = S2AxisFieldHandle(lambda v: ..., cs)"""

  def __init__(self, fun, frame=None, *, bandwidth=None):
    self.fun, self.frame = fun, frame
    self.bandwidth = getMTEXpref("defaultS2Bandwidth") if bandwidth is None else int(bandwidth)

  def eval(self, x):
    """the axes of the field at the directions, antipodal vectors"""
    v = _vectorsFor(self, x).normalize()
    return vector3d(_values(self.fun(self._framed(v))).reshape(*v.shape, 3), antipodal=True)

  def __repr__(self):
    info = "" if self.frame is None else f" ({frameLabel(self.frame)})"
    return aligned(f"S2AxisFieldHandle{info}\n  bandwidth hint: {self.bandwidth}")


class S2AxisFieldHarmonic(S2AxisField):
  """an axis field on the sphere by a harmonic series: an axis without sign at every direction,
  as the principal directions of a stress or the polarisations of a wave

    aF = S2AxisFieldHarmonic(lambda v: ..., cs, bandwidth=16)   # approximates a function of the directions
    aF = S2AxisFieldHarmonic(v, axes, bandwidth=16)             # fits axes given at the directions v
    aF = S2AxisFieldHarmonic(vF)                                # approximates a vector or axis field
    aF = S2AxisFieldHarmonic(fhat, cs)                          # from the coefficients of the six components

  Since an axis has no sign, not the axis a but the six independent entries xx, xy, yy, xz,
  yz, zz of the symmetric matrix a a^T are expanded in spherical harmonics; evaluation
  recovers the axis as the leading eigenvector of the matrix. A function or a field is
  approximated by quadrature on a grid of the bandwidth, axes at scattered directions by the
  quadrature sum with the weights given, equal ones by default (S2AxisFieldHarmonic.interpolate
  fits them by least squares). The field is symmetrised under the group of its frame.
  S2AxisField is the abstract class above it, S2AxisFieldHandle the field of a function
  evaluated where it is asked.

  The coefficient form takes a (L+1)^2 x 6 matrix, one column per component xx, xy, xz, yy,
  yz, zz, in MTEX's basis, the coefficients aF.fhat returns and S2FunHarmonic takes.

  Input
    fhat - a function of vector3d returning axes, S2VectorField or S2AxisField, vector3d of
        directions, or the coefficient matrix
    args - the axes at the directions, and the crystalFrame or specimenFrame of the field
  Output
    aF - S2AxisFieldHarmonic
  Keywords
    bandwidth - the largest harmonic degree; by default the preference
        'defaultS2Bandwidth', for a field the field's
    weights - the quadrature weights of axes at scattered directions
    symmetrise - symmetrise the field under the group of its frame
  See also
    S2VectorFieldHarmonic, S2FunHarmonic, S2AxisFieldHarmonic.interpolate, velocity,
    S2FunVectorField"""

  def __init__(self, fhat, *args, bandwidth=None, weights=None, symmetrise=True, _internal=False):
    # coefficients given are MTEX's, those the port passes itself its own
    fromUser = not _internal
    frames = [a for a in args if isinstance(a, referenceFrame)]
    frame = frames[0] if frames else None
    if isinstance(fhat, (S2VectorField, S2AxisField)):
      v = fhat
      frame = v.frame
      L = v.bandwidth if bandwidth is None else int(bandwidth)
      g = quadratureS2Grid(L)
      values = _values(v(v._framed(g.nodes())))
      fhat = _adjointColumns(g, outerComponents(values), L, g.weights)
      fromUser = False
    elif isinstance(fhat, vector3d) and args and not isinstance(args[0], referenceFrame):
      pts, values = fhat, args[0]
      frame = pts.frame if frame is None else frame
      L = getMTEXpref("defaultS2Bandwidth") if bandwidth is None else int(bandwidth)
      d = vector3d(pts.data.reshape(-1, 3)).normalize()
      w = np.full(d.size, 4 * np.pi / d.size) if weights is None else np.asarray(weights, dtype=float).ravel()
      fhat = _adjointColumns(d, outerComponents(_values(values)), L, w)
      fromUser = False
    elif callable(fhat) and not isinstance(fhat, np.ndarray):
      L = getMTEXpref("defaultS2Bandwidth") if bandwidth is None else int(bandwidth)
      g = quadratureS2Grid(L)
      values = _values(fhat(_inFrame(g.nodes(), frame)))
      fhat = _adjointColumns(g, outerComponents(values), L, g.weights)
      fromUser = False
    fhat = np.array(fhat, dtype=complex)
    if fhat.ndim != 2 or fhat.shape[1] != 6:
      raise ValueError("an axis field has six components")
    if math.isqrt(fhat.shape[0]) ** 2 != fhat.shape[0]:
      raise ValueError("a coefficient matrix has (L+1)^2 rows for its bandwidth L")
    if fromUser:
      fhat = mtexCoefficientsS2(fhat.T).T
    for i in range(6):
      realizeSpherical(fhat[:, i])
    self._fhat, self.frame = fhat, frame
    if symmetrise:
      self._fhat = self.symmetrise()._fhat

  @classmethod
  def quadrature(cls, f, *args, bandwidth=None, weights=None):
    """S2AxisFieldHarmonic.quadrature(f, bandwidth=L), MATLAB's name for the constructor

      aF = S2AxisFieldHarmonic.quadrature(f, bandwidth=8)               # of a function returning axes
      aF = S2AxisFieldHarmonic.quadrature(v, a, bandwidth=8, weights=w)  # of axes a at directions v

    A function is approximated by quadrature on a grid of the bandwidth, axes at scattered
    directions by the quadrature sum with the weights given; S2AxisFieldHarmonic explains the
    representation.

    Input
      f - a function of vector3d returning axes, S2VectorField or S2AxisField, or vector3d of
          directions
      args - the axes at the directions, and the frame of the field
    Output
      aF - S2AxisFieldHarmonic
    Keywords
      bandwidth - the largest harmonic degree; by default the preference
          'defaultS2Bandwidth', for a field the field's
      weights - the quadrature weights of axes at scattered directions, equal ones of sum
          4 pi by default
    See also
      S2AxisFieldHarmonic, S2AxisFieldHarmonic.interpolate"""
    return cls(f, *args, bandwidth=bandwidth, weights=weights)

  @classmethod
  def interpolate(cls, nodes, values, *, bandwidth=None, weights=None, **kwargs):
    """the least squares fit to axes at directions: the six components of v v^T one by one;
    MATLAB's S2AxisFieldHarmonic.interpolate

      aF = S2AxisFieldHarmonic.interpolate(v, a)                              # an axis field through axes a at directions v
      aF = S2AxisFieldHarmonic.interpolate(v, a, bandwidth=8, lambda_=1e-3)  # of a given degree, regularised

    Each of the six components of the matrices a a^T is fitted by interpolate, a regularised
    least squares fit, and the field evaluates to the leading eigenvector of the fitted matrix.
    Further keywords, lambda_, s, maxIter and tol, go to interpolate.

    Input
      nodes - vector3d, the directions
      values - vector3d, the axes at the directions
    Output
      aF - S2AxisFieldHarmonic
    Keywords
      bandwidth - the largest harmonic degree; by default it follows from the number of nodes
      weights - the weights of the nodes, as interpolate takes them
    See also
      interpolate, S2AxisFieldHarmonic.quadrature, S2VectorFieldHarmonic.interpolate"""
    from .density import interpolate
    d = vector3d(nodes.data.reshape(-1, 3)).normalize()
    comps = outerComponents(_values(values))
    fs = [interpolate(d, comps[:, i], bandwidth=bandwidth, weights=weights, **kwargs) for i in range(6)]
    L = max(f.bandwidth for f in fs)
    return cls(np.stack([f.truncate(L)._fhat for f in fs], axis=-1), nodes.frame, _internal=True)

  def _rebuild(self, fhat):
    a = object.__new__(S2AxisFieldHarmonic)
    a._fhat, a.frame = np.asarray(fhat), self.frame
    return a

  @property
  def fhat(self):
    """the coefficients of the components in MTEX's basis, one column per component"""
    return mtexCoefficientsS2(self._fhat.T).T

  @property
  def bandwidth(self):
    """the harmonic degree of the expansion"""
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
    return self._rebuild(_projectSphericalRows(self._fhat.copy(), equivariantProjection(actions, _parityWignerD(els, L))))

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
    x = asVector3d(x)
    if x.frame is not None and self.frame is not None and x.frame is not self.frame:
      raise ValueError("the axis lives in another frame than the field")
    y = x.data.reshape(3)
    q = self._fhat @ (outerComponents(y[None])[0] * [1, 2, 2, 1, 2, 1])
    f = S2FunHarmonic(q, None if self.frame is None else self.frame.stripSym(), real=True, symmetrise=False, _internal=True)
    return mapFun(lambda t: np.sqrt(np.maximum(t, 0)), f, bandwidth=smoothBandwidth(f))

  def angle(self, x):
    """the angle between the axes of the field and a constant axis, or another field, as a function
    on the sphere

      f = angle(sAF, v)
      f = angle(sAF1, sAF2)
      f = angle(sAF, sVF)  # to the directions of a vector field"""
    return mapFun(lambda t: np.arccos(np.clip(t, -1, 1)), self.dot(x))

  def __repr__(self):
    info = "" if self.frame is None else f" ({frameLabel(self.frame)})"
    return aligned(f"S2AxisFieldHarmonic{info}\n  bandwidth: {self.bandwidth}")


__all__ = ["S2AxisFieldHarmonic", "S2AxisFieldHandle"]
