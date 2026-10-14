# S2FunHarmonic: a function on the sphere as the series of H4, on directions or on axes
# (even degrees only), in a frame or none (MATLAB @S2FunHarmonic; docs/plans/functions.md F3)
#
#   sF = S2FunHarmonic(fhat), S2FunHarmonic(fhat, cs), S2FunHarmonic(psi), S2FunHarmonic(fun, bandwidth=L)
#   sF = S2FunHarmonic.quadrature(nodes, values, bandwidth=L, weights=w)
#   sF._fhat, sF.frame, sF.bandwidth, sF.isReal, sF.antipodal
#   sF(v), sF.eval(v), sF.symmetrise(), sF.conj(), sF.mean(), sF.norm(), sF.dot(t), the linear algebra
#
# A list of functions is one container, as in MATLAB: the coefficients sit on the last axis
# of fhat and the leading axes are the shape of the list, S2FunHarmonic(np.eye(10)) being
# ten functions; sF[1:3], sF.reshape(3, -1), sF.T and cat index and arrange them, sF(v) is
# v.shape + sF.shape, and the moments are one per function.
from __future__ import annotations

from typing import TYPE_CHECKING
import math

import numpy as np
from numpy.lib.array_utils import normalize_axis_index

from .harmonicarithmetic import _catParts, harmonicArithmetic
from .s2funclass import S2Fun
from ..geometry.frames import referenceFrame
from ..geometry.session import SESSION, getMTEXpref
from ..geometry.vector3d import vector3d, asVector3d
from .harmonics import _WignerD, _projectBlock, _s2Degrees, _s2PerDegree, invariantProjection, mtexCoefficientsS2
from .kernelfunctions import S2Kernel
from .quadrature import quadratureS2Grid
from .kernels import _onRows, realizeSpherical, _complexType
from .transforms import _perColumn, isRealSpherical, sphericalSeries, sphericalSeriesAdjoint


def quadratureBandwidth():
  """the bandwidth a callable or scattered values are expanded to when none is asked for, the
  preference defaultS2Bandwidth"""
  return int(getMTEXpref("defaultS2Bandwidth"))


def evenize(fhat):
  """an axis valued function has even degrees only"""
  fhat[..., _s2Degrees(math.isqrt(fhat.shape[-1]) - 1) % 2 == 1] = 0
  return fhat


# whether every function of a coefficient array is real valued
def _isReal(fhat):
  return all(isRealSpherical(r) for r in fhat.reshape(-1, fhat.shape[-1]))


def projectSpherical(fhat, pg):
  """the projection onto the functions invariant under a group, in place: an element s acts by
  f(s^-1 v) with its parity eps_s on the degree, so degree l projects by the mean of
  eps_s^l D_l(r_s). On the even degrees that is the projection P_K onto the rotation parts K
  of all elements; on the odd ones the proper elements H count +1 and the improper ones -1,
  which sums to 2|H|/|G| P_H - P_K. Both are the held projections of H5"""
  L = math.isqrt(fhat.shape[-1]) - 1
  dtype = fhat.real.dtype
  K = invariantProjection(pg, L, dtype)
  H = None if pg.isProper else invariantProjection(pg, L, dtype, proper=True)
  c = 2 * pg.numProper() / pg.numSym()

  # every degree of every row: the even ones by P_K, the odd ones by c P_H - P_K
  def projectRows(rows):
    for l in range(L + 1):
      B = rows[:, l * l:(l + 1) ** 2]
      b = _projectBlock(B.copy(), None, K, l)
      B[...] = b if H is None or l % 2 == 0 else c * _projectBlock(B.copy(), None, H, l) - b
  return _onRows(fhat, projectRows)


class S2FunHarmonic(S2Fun, harmonicArithmetic):
  """a function on the sphere by its spherical harmonic series, or a list of such functions;
  the default representation of a function on the sphere

    sF = S2FunHarmonic(fhat)                                 # from MTEX's coefficients
    sF = S2FunHarmonic(fhat, cs, antipodal=True)             # in a crystal frame, f(v) = f(-v)
    sF = S2FunHarmonic(lambda v: v.z**2, bandwidth=16)       # a callable expanded by quadrature
    sF = S2FunHarmonic(psi)                                  # the series of a kernel on the sphere
    sF = S2FunHarmonic(sF, bandwidth=8)                      # another function, truncated

  The harmonic series is the form on which convolution, rotation, products and integrals are
  cheap. The coefficients fhat are MTEX's, degree by degree, (L + 1)^2 of them for the
  bandwidth L; a vector ending inside a degree is padded with zeros. A list of functions
  carries its shape in front of the coefficients.

  The first argument may also be:

  - a callable of vector3d: sampled on a quadrature grid of the bandwidth, by default the
  preference defaultS2Bandwidth; S2FunHarmonic.quadrature takes values at nodes
  - an S2Kernel: its series
  - another S2Fun: its harmonic form, truncated to the bandwidth if given

  The function is made invariant under the group of its frame unless symmetrise=False, so a
  callable that is not invariant becomes its mean over the symmetric directions. A frame
  with an inversion makes the function antipodal.

  Input
    fhat - array of MTEX's coefficients, an S2Fun, an S2Kernel or a callable of vector3d
    frame - the crystalFrame or specimenFrame of the directions
  Output
    sF - S2FunHarmonic
  Keywords
    real - whether the function takes real values only; by default read off the
        coefficients or the sampled values
    antipodal - f(v) = f(-v)
    symmetrise - make the coefficients invariant under the group of the frame
    bandwidth - the largest degree of the series
    dtype - the complex type of the coefficients, complex128 unless given
  Properties
    fhat - spherical harmonic coefficients in MTEX's public basis
    bandwidth - largest spherical harmonic degree
    frame - reference frame carrying the function symmetry
    CS - alias for the spherical reference frame
    isReal - whether the function is real-valued
    antipodal - whether f(v)=f(-v)
    shape - shape of a list of functions, excluding the coefficient axis
  See also
    S2FunHarmonic.quadrature, S2FunHarmonic.interpolate, S2FunHandle, S2Kernel,
    SO3FunHarmonic, S2FunHarmonicRepresentation"""
  # NumPy leaves the arithmetic with a function to the function, not reading it as a sequence
  __array_ufunc__ = None

  def __init__(self, fhat, frame=None, *, real=None, antipodal=None, symmetrise=True, bandwidth=None, dtype=None,
               _internal=False):
    # coefficients given are MTEX's, those the port passes itself its own
    fromUser = not _internal
    # another representation converts itself to its harmonic form
    if not isinstance(fhat, S2FunHarmonic) and hasattr(fhat, "_harmonicS2"):
      fhat, bandwidth = fhat._harmonicS2(bandwidth), None
    if isinstance(fhat, S2FunHarmonic):
      f = fhat
      frame = f.frame if frame is None else frame
      real = f.isReal if real is None else real
      antipodal = f.antipodal if antipodal is None else antipodal
      fhat = f.truncate(bandwidth)._fhat if bandwidth is not None else f._fhat
      bandwidth, fromUser = None, False
    elif isinstance(fhat, S2Kernel):
      fhat = _kernelCoefficients(fhat, bandwidth)
      real, bandwidth, fromUser = True, None, False
    elif callable(fhat) and not isinstance(fhat, np.ndarray):
      fhat, real = _byQuadrature(fhat, frame, bandwidth, real)
      bandwidth, fromUser = None, False
    fhat = np.array(fhat, dtype=_complexType(np.result_type(np.asarray(fhat).dtype, np.float64) if dtype is None else dtype))
    fhat = fhat.reshape(1) if fhat.ndim == 0 else fhat
    # a count between two full degrees is padded with zeros to the next, as MATLAB does
    L = math.isqrt(fhat.shape[-1] - 1) if fhat.shape[-1] > 1 else 0
    if (L + 1) ** 2 != fhat.shape[-1]:
      fhat = _cut(fhat, L)
    if bandwidth is not None and bandwidth != L:
      fhat = _cut(fhat, bandwidth)
    if fromUser:
      fhat = mtexCoefficientsS2(fhat)
    real = _isReal(fhat) if real is None else bool(real)
    if real:
      realizeSpherical(fhat)
    # a function with a centrosymmetric frame is even whether or not it was asked to be
    antipodal = bool(antipodal) or bool(getattr(frame, "isLaue", False))
    if antipodal:
      evenize(fhat)
    self._fhat, self.frame, self.isReal, self.antipodal = fhat, frame, real, antipodal
    if symmetrise:
      self._fhat = self.symmetrise()._fhat

  def _rebuild(self, fhat, **changes):
    f = object.__new__(S2FunHarmonic)
    f._fhat, f.frame, f.isReal, f.antipodal = np.asarray(fhat), self.frame, self.isReal, self.antipodal
    for k, v in changes.items():
      setattr(f, k, v)
    return f

  @classmethod
  def quadrature(cls, nodes, values=None, frame=None, *, bandwidth=None, weights=None, real=None, antipodal=None):
    """the harmonic series of a function on the sphere by quadrature, from a callable or from
    values at nodes; MATLAB's approximate is the same

      sF = S2FunHarmonic.quadrature(lambda v: v.z**2)                     # a callable on a grid of the default bandwidth
      sF = S2FunHarmonic.quadrature(lambda v: v.z**2, cs, bandwidth=16)   # in the crystal frame cs
      sF = S2FunHarmonic.quadrature(v, values, bandwidth=8, weights=w)    # values at the nodes v
      sF = S2FunHarmonic.quadrature(g, values)                            # values on a quadratureS2Grid g

    The coefficients are the integrals of f against the spherical harmonics, approximated by
    the quadrature sum over the nodes. A callable is sampled on a quadrature grid exact up to
    the bandwidth, and a quadratureS2Grid brings its own weights and bandwidth.

    Values at scattered nodes take the weights given, which sum to 4 pi, equal ones by
    default. They give a good approximation only on nodes dense enough for the bandwidth;
    S2FunHarmonic.interpolate fits scattered values by least squares instead. Values for a
    list of functions have the nodes first.

    Input
      nodes - a callable of vector3d, vector3d, or a quadratureS2Grid
      values - the values at the nodes, the nodes along the first axis; for a callable the
          frame instead
      frame - the crystalFrame or specimenFrame of the result; by default that of the nodes
    Output
      sF - S2FunHarmonic
    Keywords
      bandwidth - the largest degree; by default the grid's for a quadrature grid, else the
          preference defaultS2Bandwidth
      weights - the quadrature weights of the nodes, summing to 4 pi
      real - whether the function is real; by default read off the values
      antipodal - f(v) = f(-v); by default that of the nodes
    See also
      S2FunHarmonic, S2FunHarmonic.interpolate, quadratureS2Grid, S2FunHarmonicRepresentation"""
    # quadrature(fun, cs): a callable takes the frame second
    if callable(nodes) and not isinstance(nodes, vector3d) and (values is None or isinstance(values, referenceFrame)):
      return cls(nodes, values if frame is None else frame, bandwidth=bandwidth, real=real, antipodal=bool(antipodal))
    values = np.asarray(values)
    if isinstance(nodes, quadratureS2Grid):
      L = nodes.bandwidth if bandwidth is None else int(bandwidth)
      frame = nodes.frame if frame is None else frame
      n, adjoint = len(nodes), lambda x: sphericalSeriesAdjoint(nodes, x, L, weights)
    else:
      L = quadratureBandwidth() if bandwidth is None else int(bandwidth)
      v = asVector3d(nodes)
      frame = v.frame if frame is None else frame
      antipodal = v.antipodal if antipodal is None else antipodal
      w = np.full(v.size, 4 * np.pi / max(v.size, 1)) if weights is None else np.asarray(weights).ravel()
      n, adjoint = v.size, lambda x: sphericalSeriesAdjoint(v, x, L, w)
    # values for a list of functions: the nodes first, then the shape of the list
    fhat = _perColumn(adjoint, values, n)
    return cls(fhat, frame, real=(values.dtype.kind != "c") if real is None else real, antipodal=bool(antipodal), _internal=True)

  approximate = quadrature

  @classmethod
  def adjointNFSFT(cls, nodes, values, *, bandwidth=None, weights=None):
    """S2FunHarmonic.adjointNFSFT(v, values, bandwidth=32): the adjoint transform, the
    coefficients sum_j w_j y_j conj(Y(v_j)) with unit weights unless given, a grid's own on a grid;
    MATLAB's adjoint and adjointNFSFT

      sF = S2FunHarmonic.adjoint(v, values, bandwidth=16)                # unit weights
      sF = S2FunHarmonic.adjoint(v, values, bandwidth=16, weights=w)     # with quadrature weights
      sF = S2FunHarmonic.adjoint(quadratureS2Grid(16), values)          # a quadrature grid: its bandwidth and weights

    With the weights of a quadrature rule exact up to twice the bandwidth the adjoint is the
    expansion of the function the values sample; with other weights it is the adjoint of the
    evaluation only, S2FunHarmonic.interpolate fits by least squares. A NaN value counts as 0.
    Several values per node, an array (n, ...), give one function per column. MATLAB's flags
    'nfsft' and 'directComputation' choose the algorithm and have no counterpart; its default
    bandwidth is 512.

    Input
      nodes - vector3d or quadratureS2Grid
      values - ndarray, one value or one row of values per node
    Output
      sF - S2FunHarmonic
    Keywords
      bandwidth - the largest harmonic degree; by default the bandwidth of the grid, else 128
      weights - the weights of the nodes, one number or one per node; by default those of a
          quadrature grid, else one
    See also
      S2FunHarmonic.quadrature, S2FunHarmonic.interpolate, sphericalSeriesAdjoint"""
    values = np.asarray(values)
    if isinstance(nodes, quadratureS2Grid):
      L, w, v = nodes.bandwidth, nodes.weights if weights is None else weights, nodes
    else:
      v = asVector3d(nodes).flatten()
      L = quadratureBandwidth() if bandwidth is None else int(bandwidth)
      w = np.ones(v.size) if weights is None else np.broadcast_to(np.asarray(weights, dtype=float).ravel(), (v.size,))
    fhat = _perColumn(lambda x: sphericalSeriesAdjoint(v, x, L, w), np.nan_to_num(values), len(v))
    return cls(fhat, v.frame, real=values.dtype.kind != "c", _internal=True)

  adjoint = adjointNFSFT

  @staticmethod
  def unimodal(*args, **kwargs):
    """S2FunHarmonic.unimodal(v, psi): MATLAB's other name of S2Fun.unimodal"""
    return S2Fun.unimodal(*args, **kwargs)

  @classmethod
  def interpolate(cls, nodes, values, **kwargs):
    """S2FunHarmonic.interpolate(v, y) is interpolate(v, y): the function of least squares through
    the values y at the directions v"""
    from .density import interpolate
    return interpolate(nodes, values, **kwargs)

  @property
  def CS(self):
    """the frame of the function, MATLAB's sF.CS"""
    return self.frame

  @property
  def how2plot(self):
    """the plotting convention of the frame, or the session's"""
    from ..geometry.plottingconvention import plottingConvention
    pC = getattr(self.frame, "how2plot", None)
    return plottingConvention(pC if pC is not None else plottingConvention.default)

  @property
  def fhat(self):
    """the coefficients in MTEX's basis, as MATLAB's sF.fhat reads and assigns them"""
    return mtexCoefficientsS2(self._fhat)

  @fhat.setter
  def fhat(self, value):
    """the coefficients in MTEX's basis, as MATLAB's sF.fhat reads and assigns them"""
    self._fhat = mtexCoefficientsS2(np.asarray(value, dtype=self._fhat.dtype))

  @property
  def bandwidth(self):
    """the bandwidth, the highest degree of the coefficients"""
    return math.isqrt(self._fhat.shape[-1]) - 1

  @bandwidth.setter
  def bandwidth(self, L):
    """sF.bandwidth = L cuts or pads the coefficients, as MATLAB's assignment does"""
    self._fhat = _cut(self._fhat, int(L))

  @property
  def shape(self):
    """the shape of a list of functions, () for one"""
    return self._fhat.shape[:-1]

  @property
  def size(self):
    """the number of functions in a list, 1 for one function"""
    return int(np.prod(self.shape))

  def __len__(self):
    return self.size

  def __getitem__(self, idx):
    if not self.shape:
      raise TypeError("a single function has no elements to index")
    fhat = self._fhat[idx]
    return self._rebuild(fhat if fhat.ndim else fhat.reshape(1))

  def reshape(self, *shape):
    """the list of functions in another shape

      sF = sF.reshape(2, 3)
      sF = sF.reshape((2, 3))"""
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return self._rebuild(self._fhat.reshape(tuple(shape) + (self._fhat.shape[-1],)))

  @property
  def T(self):
    """the list transposed, a row of n functions becoming a column, and MATLAB's ' as sF.conj().T"""
    if not self.shape:
      return self
    if len(self.shape) == 1:
      return self._rebuild(self._fhat[:, None, :])
    return self._rebuild(np.swapaxes(self._fhat, -2, -3))

  @classmethod
  def cat(cls, *fs, axis=0):
    """the functions joined into one list along an axis, padded to the largest bandwidth;
    MATLAB's cat, horzcat and vertcat

      sF = cat(sF1, sF2)                                   # a list of two functions
      sF = S2FunHarmonic.cat(sFa, sFb, axis=1)             # lists of shape (n, 1) side by side, an (n, 2) list

    A single function counts as a list of one, and a list lacking the axis gets it as a trailing
    axis of length one, as MATLAB's cat(2, sF1, sF2) gives a 1 x 2 list of two functions. The
    lists must agree in every other axis.

    Input
      fs - S2FunHarmonic, the functions or lists to join
    Output
      sF - S2FunHarmonic, a list of functions
    Keywords
      axis - the axis of the lists along which they are joined
    See also
      S2FunHarmonic.sum, S2FunHarmonic.mean"""
    L = max(f.bandwidth for f in fs)
    parts = [f.truncate(L)._fhat for f in fs]
    parts, axis = _catParts(parts, axis)
    f0 = fs[0]
    return f0._rebuild(np.concatenate(parts, axis=axis), isReal=all(f.isReal for f in fs), antipodal=all(f.antipodal for f in fs))

  @property
  def power(self):
    """the power of every degree, the norm of the coefficients of that degree, as an array of
    length bandwidth + 1 per function"""
    return np.sqrt(np.add.reduceat(np.abs(self._fhat) ** 2, np.arange(self.bandwidth + 1) ** 2, axis=-1))

  def truncate(self, L=None):
    """the coefficients up to degree L; without L the trailing degrees whose power, damped by
    (l+1)^2, stays below 1e-8 of the largest are dropped, MATLAB's truncate

      sF = truncate(sF)      # the negligible degrees dropped
      sF = truncate(sF, 8)   # cut, or padded with zeros, at degree 8

    The power of a degree is the norm of its coefficients divided by (l+1)^2, a bit of
    regularisation, as MATLAB's; of a list of functions the largest over the list counts.
    MATLAB's truncate takes no degree.

    Input
      sF - S2FunHarmonic
    Output
      sF - S2FunHarmonic
    Keywords
      L - the bandwidth of the result
    See also
      S2FunHarmonic.power, S2FunHarmonic.bandwidth"""
    if L is None:
      l = np.arange(self.bandwidth + 1)
      w = np.abs(self._fhat / _s2PerDegree((l + 1) ** 2)) ** 2
      per = np.sqrt(np.add.reduceat(w, l ** 2, axis=-1))
      per = per.reshape(-1, self.bandwidth + 1).max(axis=0)
      above = np.flatnonzero(per > per.max() * 1e-8)
      L = int(above[-1]) if above.size else 0
    return self if L == self.bandwidth else self._rebuild(_cut(self._fhat, L))

  # ------------------------------------------------------------------ evaluation

  # bare vectors, or framed ones in the frame of the function or one sharing its basis
  def _vectorsFor(self, x):
    v = asVector3d(x)
    if self.frame is not None and v.frame is not None and v.frame is not self.frame:
      try:
        SESSION.joinFrames(v.frame, self.frame)
      except ValueError:
        raise ValueError("the directions live in another frame than the function") from None
    return v

  def __call__(self, x):
    return self.eval(x)

  def eval(self, x):
    """the function at directions, or at the nodes of a quadrature grid; a list of functions
    adds its shape after that of the directions"""
    v = x if isinstance(x, quadratureS2Grid) else self._vectorsFor(x).normalize()
    if not self.shape:
      return sphericalSeries(self._fhat, v, real=self.isReal)
    # a list of functions: the values at the nodes first, then the shape of the list
    vals = [sphericalSeries(r, v, real=self.isReal) for r in self._fhat.reshape(-1, self._fhat.shape[-1])]
    return np.stack(vals, axis=-1).reshape(np.shape(vals[0]) + self.shape)

  # ------------------------------------------------------------------ symmetry

  def symmetrise(self, rots=None):
    """the projection onto the functions invariant under the group of the frame, or under a
    list of rotations forming a group the frame does not name; an axis valued function is even;
    MATLAB's symmetrise

      sF = symmetrise(sF)                     # under the group of the frame of sF
      sF = symmetrise(sF, crystalFrame('432'))  # under the group of another frame, the result in that frame
      sF = sF.symmetrise(rotation.byAxisAngle(zvector, np.arange(4)*90*degree))  # the mean over a group of rotations

    Each degree is averaged over the group, the mean of the Wigner matrices of its rotations
    applied to the coefficients, which keeps exactly the invariant part. A frame given puts the
    function in that frame, which names the group. MATLAB's symmetrise(sF, d) about a direction,
    returning a kernel, has no counterpart.

    Input
      sF - S2FunHarmonic
    Output
      sF - S2FunHarmonic
    Keywords
      rots - rotation of a group, or a crystalFrame or specimenFrame whose group is taken
    See also
      S2FunHarmonic.eval, Miller.symmetrise"""
    # symmetrise(sF, frame): the function averaged over the group of the frame, in that frame
    if isinstance(rots, referenceFrame):
      return S2FunHarmonic(self._fhat, rots, real=self.isReal, antipodal=self.antipodal, _internal=True)
    L = self.bandwidth
    fhat = self._fhat.copy()
    if rots is not None:
      rots = rots.flatten()
      D = _WignerD(rots, bandwidth=L, dtype=self._fhat.real.dtype)
      for l in range(L + 1):
        fhat[..., l * l:(l + 1) ** 2] = fhat[..., l * l:(l + 1) ** 2] @ np.mean(D[l], axis=0).T
    elif self.frame is not None:
      projectSpherical(fhat, self.frame.sym)
    if self.antipodal:
      evenize(fhat)
    return self._rebuild(fhat)

  def even(self):
    """return the even part (f(v)+f(-v))/2 of a spherical function.

      evenPart = f.even()

    Odd harmonic degrees are removed. The result is unchanged by reversal of a direction,
    which is appropriate when v and -v represent the same axis.

    See also
      S2FunHarmonic.radon, S2FunHarmonic.symmetrise"""
    return self._rebuild(evenize(self._fhat.copy()), antipodal=True)

  # ------------------------------------------------------------------ the algebra

  def conj(self):
    """the complex conjugate function"""
    L = self.bandwidth
    out = np.empty_like(self._fhat)
    for l in range(L + 1):
      m = np.arange(-l, l + 1)
      out[..., l * l:(l + 1) ** 2] = np.where(m % 2, -1.0, 1.0) * np.conj(self._fhat[..., l * l:(l + 1) ** 2][..., ::-1])
    return self._rebuild(out)

  def _sameFrames(self, g):
    if g.frame is not self.frame:
      raise ValueError("the functions live in different frames")

  def __add__(self, g):
    if hasattr(g, "_harmonicS2"):
      from .handles import S2FunSum
      return S2FunSum(self, g)
    if isinstance(g, S2FunHarmonic):
      self._sameFrames(g)
      L = max(self.bandwidth, g.bandwidth)
      return S2FunHarmonic(self.truncate(L)._fhat + g.truncate(L)._fhat, self.frame, real=self.isReal and g.isReal,
                           antipodal=self.antipodal and g.antipodal, symmetrise=False, _internal=True)
    if isinstance(g, (int, float, complex, np.number)):
      fhat = self._fhat.copy()
      fhat[..., 0] += g * math.sqrt(4 * math.pi)
      return S2FunHarmonic(fhat, self.frame, real=self.isReal and np.isreal(g), antipodal=self.antipodal, symmetrise=False, _internal=True)
    return NotImplemented

  __radd__ = __add__

  def __neg__(self):
    return self._rebuild(-self._fhat)

  def __mul__(self, a):
    if isinstance(a, S2FunHarmonic):
      from .funops import mapFun
      return mapFun(np.multiply, self, a)
    if isinstance(a, (int, float, complex, np.number)):
      return S2FunHarmonic(a * self._fhat, self.frame, real=self.isReal and np.isreal(a), antipodal=self.antipodal, symmetrise=False, _internal=True)
    return NotImplemented

  # |f| as a handle, evaluated when asked, as MATLAB's abs of a spherical function
  def __abs__(self):
    from .handles import S2FunHandle
    return S2FunHandle(lambda v: np.abs(self.eval(v)), self.frame, bandwidth=self.bandwidth, antipodal=self.antipodal,
                       symmetrise=False)

  def abs(self):
    """the absolute value |f| as a handle, evaluated when asked

      sF = abs(sF)"""
    return abs(self)

  def sqrt(self, bandwidth=None):
    """the square root of the function, expanded by quadrature, or at the bandwidth given

      sF = sqrt(sF)                # at twice the bandwidth, at least 32
      sF = sqrt(sF, bandwidth=64)  # at the bandwidth given

    The function is evaluated on a quadrature grid and expanded again (mapFun); the square root
    is no polynomial, so the expansion is cut at the bandwidth: by default twice that of the
    function, at least 32. A negative value gives NaN.

    Input
      sF - S2FunHarmonic
    Output
      sF - S2FunHarmonic
    Keywords
      bandwidth - the largest harmonic degree of the result
    See also
      mapFun, S2FunHarmonic.quadrature"""
    from .funops import mapFun, smoothBandwidth
    return mapFun(np.sqrt, self, bandwidth=smoothBandwidth(self) if bandwidth is None else bandwidth)

  def exp(self, bandwidth=None):
    """the exponential of the function, expanded by quadrature, or at the bandwidth given

      sF = exp(sF)                # at twice the bandwidth, at least 32
      sF = exp(sF, bandwidth=64)  # at the bandwidth given

    The function is evaluated on a quadrature grid and expanded again (mapFun); the exponential
    is no polynomial, so the expansion is cut at the bandwidth: by default twice that of the
    function, at least 32.

    Input
      sF - S2FunHarmonic
    Output
      sF - S2FunHarmonic
    Keywords
      bandwidth - the largest harmonic degree of the result
    See also
      mapFun, S2FunHarmonic.quadrature"""
    from .funops import mapFun, smoothBandwidth
    return mapFun(np.exp, self, bandwidth=smoothBandwidth(self) if bandwidth is None else bandwidth)

  def log(self, bandwidth=None):
    """the logarithm of the function, expanded by quadrature, or at the bandwidth given

      sF = log(sF)                # at twice the bandwidth, at least 32
      sF = log(sF, bandwidth=64)  # at the bandwidth given

    The function is evaluated on a quadrature grid and expanded again (mapFun); the logarithm
    is no polynomial, so the expansion is cut at the bandwidth: by default twice that of the
    function, at least 32.

    Input
      sF - S2FunHarmonic
    Output
      sF - S2FunHarmonic
    Keywords
      bandwidth - the largest harmonic degree of the result
    See also
      mapFun, S2FunHarmonic.quadrature"""
    from .funops import mapFun, smoothBandwidth
    return mapFun(np.log, self, bandwidth=smoothBandwidth(self) if bandwidth is None else bandwidth)

  # ------------------------------------------------------------------ the moments

  def mean(self, axis=None):
    """the mean over the sphere, int f dS / 4 pi, one per function; with an axis the mean
    function along that axis of a list; MATLAB's mean

      value = mean(sF)          # the mean value, one per function of a list
      sF = mean(sF, axis=0)     # the pointwise mean of a list along its first axis

    Input
      sF - S2FunHarmonic
    Output
      value - float, or an array of the shape of the list; with an axis S2FunHarmonic
    Keywords
      axis - the axis of the list along which the functions are averaged
    See also
      S2FunHarmonic.sum, S2FunHarmonic.norm"""
    if axis is not None:
      return self._rebuild(np.mean(self._fhat, axis=normalize_axis_index(axis, len(self.shape))))
    x = self._fhat[..., 0] / math.sqrt(4 * math.pi)
    x = x.real if self.isReal else x
    return x[()] if x.ndim == 0 else x

  def sum(self, axis=None):
    """the integral over the sphere, int f dS, MATLAB's sum; with an axis the sum of the
    functions along that axis of a list

      value = sum(sF)       # the integral, one per function of a list
      sF = sum(sF, axis=0)  # the pointwise sum along the first axis

    Input
      sF - S2FunHarmonic
    Output
      value - float, or an array of the shape of the list; with an axis S2FunHarmonic
    Keywords
      axis - the axis of the list along which the functions are summed
    See also
      S2FunHarmonic.mean, S2FunHarmonic.cat"""
    if axis is not None:
      return self._rebuild(np.sum(self._fhat, axis=normalize_axis_index(axis, len(self.shape))))
    return 4 * math.pi * self.mean()

  def norm(self, s=None):
    """the L2 norm for the normalised surface measure, or the Sobolev norm, one per function;
    MATLAB's norm

      t = norm(sF)       # the L2 norm
      t = norm(sF, s=1)  # the Sobolev norm of index s

    The L2 norm is (int |f|^2 dS / 4 pi)^(1/2); the Sobolev norm weights the coefficients of
    degree l by (2l+1)^(2s). MATLAB's norm has no Sobolev index.

    Input
      sF - S2FunHarmonic
    Output
      t - float, or an array of the shape of the list
    Keywords
      s - the Sobolev index; None for the L2 norm
    See also
      S2FunHarmonic.dot, S2FunHarmonic.mean"""
    if s is None:
      n = np.linalg.norm(self._fhat, axis=-1) / math.sqrt(4 * math.pi)
    else:
      w = _s2PerDegree((2.0 * np.arange(self.bandwidth + 1) + 1) ** (2 * s))
      n = np.sqrt(np.sum(w * np.abs(self._fhat) ** 2, axis=-1)) / math.sqrt(4 * math.pi)
    return float(n) if np.ndim(n) == 0 else n

  def dot(self, g):
    """the inner product with another function of the same frame, for the normalised surface
    measure: int conj(f) g dS / 4 pi"""
    self._sameFrames(g)
    L = max(self.bandwidth, g.bandwidth)
    x = np.sum(np.conj(self.truncate(L)._fhat) * g.truncate(L)._fhat, axis=-1) / (4 * math.pi)
    x = x.real if self.isReal and g.isReal else x
    return x[()] if np.ndim(x) == 0 else x

  # the pointwise minimum or maximum of the functions of a list along an axis, MATLAB's
  # min(sF, [], d), by quadrature; min(sF, axis=d) and max reach it through characteristics
  def _reduce(self, op, axis):
    ax = normalize_axis_index(axis, len(self.shape))
    def fun(v):
      return op(self.eval(v), axis=np.ndim(v.data) - 1 + ax)
    return S2FunHarmonic(fun, self.frame, bandwidth=self.bandwidth, real=self.isReal, antipodal=self.antipodal)

  def isclose(self, g, rtol=1e-8, atol=0.0):
    """whether the coefficients agree with those of g within the tolerances, relative to the norm of g

      tf = isclose(sF1, sF2)             # equal up to a relative 1e-8
      tf = isclose(sF1, sF2, rtol=1e-2)  # up to a relative 1e-2

    The functions are compared at the larger of their bandwidths: the norm of the difference of
    the coefficients must not exceed atol + rtol times the norm of those of g; a list counts as
    one array.

    Input
      sF1 - S2FunHarmonic
      g - S2FunHarmonic, the function to compare with
    Output
      tf - bool
    Keywords
      rtol - the tolerance relative to the norm of g
      atol - the absolute tolerance
    See also
      S2FunHarmonic.norm"""
    L = max(self.bandwidth, g.bandwidth)
    a, b = self.truncate(L)._fhat, g.truncate(L)._fhat
    return bool(np.linalg.norm(a - b) <= atol + rtol * np.linalg.norm(b))

  # ------------------------------------------------------------------ the operations of funops

  # rot * f is the function rotated, as MATLAB writes it
  def __rmul__(self, other):
    from ..geometry.quaternion import quaternion
    if isinstance(other, quaternion):
      return self.rotate(other)
    return self.__mul__(other)

  def rotate(self, rot):
    """the function rotated, f(rot^-1 v); rot * f is the same"""
    from .funops import _rotateS2
    return _rotateS2(self, rot)

  def conv(self, g):
    """convolve a spherical function with a kernel or another spherical function.

      smoothed = f.conv(psi)
      correlation = f.conv(g)

    A radial kernel multiplies each spherical harmonic degree by its kernel multiplier
    and returns an S2FunHarmonic. With another S2Fun, convolution compares their relative
    rotations and returns an SO3FunHarmonic. The result therefore depends on the second
    argument's domain.

    Input
      g - S2Kernel, SO3Kernel or spherical function
    See also
      SO3FunHarmonic.conv, S2Kernel.conv, S2FunConvolution"""
    from .funops import conv
    return conv(self, g)

  def laplace(self):
    """apply the spherical Laplace-Beltrami operator.

      laplacian = f.laplace()

    Coefficients of degree l are multiplied by `-l*(l+1)`; the constant mode vanishes.
    The result is another spherical harmonic function with the same frame and bandwidth.

    See also
      S2FunHarmonic.grad, S2FunHarmonic"""
    from .funops import laplace
    return laplace(self)

  def grad(self, x=None):
    """the surface gradient: a vector field, or its values at the directions x; MATLAB's grad

      sVF = grad(sF)       # the gradient as a vector field
      g = grad(sF, v)      # the gradient at the directions v, vector3d

    The gradient is tangent to the sphere, the derivative of the function along the sphere.

    Input
      sF - S2FunHarmonic
    Output
      sVF - S2VectorFieldHarmonic, or vector3d at x
    Keywords
      x - vector3d, the directions at which to evaluate the gradient
    See also
      S2VectorFieldHarmonic, S2FunHarmonic.laplace"""
    from .s2vectorfield import grad
    return grad(self, x)

  def radon(self, delta=None):
    """the Funk transform, the mean over the great circle perpendicular to every direction;
    radon(delta) over the small circle at pi/2 + delta; MATLAB's radon

      sF = radon(sF)              # the mean over the great circles
      sF = radon(sF, 10*degree)   # the mean over the small circles at 100 degree

    The transform is a convolution: the coefficients of degree n are multiplied by the Legendre
    polynomial P_n(cos(pi/2 + delta)), for the great circle (-1)^(n/2) (n-1)!!/n!! of an even n
    and zero of an odd one, so the odd part of a function is lost; invRadon inverts it on the
    even functions.

    Input
      sF - S2FunHarmonic
    Output
      sF - S2FunHarmonic
    Keywords
      delta - the angle by which the circle is moved from the great circle
    See also
      S2FunHarmonic.invRadon, calcPDF"""
    from .funops import radon
    return radon(self, delta)

  def invRadon(self):
    """invert the Funk transform on the even spherical harmonics.

      reconstructed = poleAverage.invRadon()

    The Funk transform averages over great circles and annihilates every odd degree.
    This operation divides the even degrees by their nonzero Funk multipliers and
    returns an even function. Odd information cannot be recovered from great-circle means.

    See also
      S2FunHarmonic.radon, S2Kernel.radon"""
    from .funops import invRadon
    return invRadon(self)

  def __repr__(self):
    from ..geometry.display import display, frameLabel, header, sizeString
    return display(header("S2FunHarmonic", frameLabel(self.frame)), [
      ("size", sizeString(self.shape) if self.shape else None), ("bandwidth", self.bandwidth),
      ("mean", None if self.shape else f"{self.mean():.4g}"), ("antipodal", "true" if self.antipodal else None),
      ("isReal", None if self.isReal else "false")])

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from .characteristics import max_ as max, min_ as min, entropy, textureIndex, discreteSample, volume
    from .discrepancy import discrepancy
    from ..analysis.optimalsample import optimalSample
    from ..plotting.sphericalplot import plot, contourf, contour, pcolor
    from ..plotting.spaceplot import plot3d
    from ..plotting.kernelplot import plotSpectrum


def _cut(fhat, L):
  out = np.zeros(fhat.shape[:-1] + ((L + 1) ** 2,), dtype=fhat.dtype)
  n = min(out.shape[-1], fhat.shape[-1])
  out[..., :n] = fhat[..., :n]
  return out


# a zonal kernel centred at z: fhat_l^0 = A_l sqrt(4 pi / (2l+1))
def _kernelCoefficients(psi, bandwidth):
  L = psi.bandwidth if bandwidth is None else int(bandwidth)
  A = psi.calcFourier(L)
  fhat = np.zeros((L + 1) ** 2, dtype=complex)
  l = np.arange(L + 1)
  fhat[l * l + l] = A * np.sqrt(4 * math.pi / (2 * l + 1))
  return fhat


# a callable by exact quadrature; the callable receives the directions of the rule in the frame
def _byQuadrature(fun, frame, bandwidth, real):
  L = quadratureBandwidth() if bandwidth is None else int(bandwidth)
  g = quadratureS2Grid(L)
  v = g.nodes()
  if frame is not None:
    v = vector3d(v, frame=frame)
  values = np.asarray(fun(v))
  return _perColumn(lambda x: sphericalSeriesAdjoint(g, x, L), values, len(g)), (values.dtype.kind != "c") if real is None else real


__all__ = ["S2FunHarmonic"]
