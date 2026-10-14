# functions given by a callable, and sums of functions of different representations
# (MATLAB @SO3FunHandle, @S2FunHandle, @S1FunHandle, @SO3FunComposition; docs/plans/functions.md F5)
#
#   SO3FunHandle(fun, CS, SS, bandwidth=, antipodal=, symmetrise=)   fun receives orientations in the frames
#   S2FunHandle(fun, frame, bandwidth=, antipodal=, symmetrise=)     fun receives directions
#   S1FunHandle(fun, bandwidth=)
#   SO3FunComposition(f, g, ...), S2FunSum(s, t, ...)                 what + returns when coefficients cannot be added
#
# Every operation the harmonic types offer reaches these through the harmonic form,
# SO3FunHarmonic(f, bandwidth=L), by quadrature of the values.
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from .so3funclass import SO3Fun
from .s2funclass import S2Fun
from ..geometry.orientation import orientation
from ..geometry.quaternion import orthogonalTransform, rotation
from ..geometry.session import SESSION, getMTEXpref
from ..geometry.symreduce import reductionGroups
from ..geometry.vector3d import vector3d, asVector3d
from .funops import conv
from .rbf import SO3FunCBF, SO3FunRBF, _framesAmong, _rotationsFor, uniformODF
from .s1fun import S1FunHarmonic, _localExtremaS1, _pointwiseExtremumS1
from ..geometry.units import degree
from .s2fun import S2FunHarmonic, quadratureBandwidth
from .so3fun import SO3FunHarmonic
from ..geometry.framepair import installFramePair
from ..geometry.install import install
from ..geometry.display import frameLabel, framesLabel
from ..geometry.display import aligned


class SO3FunGeneric(SO3Fun):
  """what every representation on SO(3) without an exact harmonic form shares: evaluation in its
  frames, the operations through the harmonic form, and sums as compositions"""
  isReal = True
  antipodal = False

  def __call__(self, x):
    return self.eval(x)

  def _harmonic(self, bandwidth=None):
    L = self.bandwidth if bandwidth is None else int(bandwidth)
    return SO3FunHarmonic(lambda o: self.eval(o), self.CS, self.SS, bandwidth=L, real=self.isReal, antipodal=self.antipodal)

  def mean(self, resolution=None):
    """the mean by the harmonic expansion, or with a resolution the average over an
    equispaced grid of it, as MATLAB's"""
    if resolution is None:
      return SO3FunHarmonic(self).mean()
    from ..geometry.grids.so3grid import equispacedSO3Grid
    nodes = equispacedSO3Grid(self.CS, self.SS, resolution=resolution)
    v = np.mean(np.asarray(self.eval(nodes.flatten())))
    return float(np.real(v)) if np.isclose(np.imag(v), 0) else v

  def norm(self):
    """the L2 norm, through the harmonic form"""
    return SO3FunHarmonic(self).norm()

  # the pointwise absolute value stays a handle, evaluated when asked, as MATLAB's
  def __abs__(self):
    return SO3FunHandle(lambda o: np.abs(self.eval(o)), self.CS, self.SS, bandwidth=self.bandwidth, antipodal=self.antipodal)

  def abs(self):
    """the pointwise absolute value, a handle evaluated when asked"""
    return abs(self)

  # the pole figure of one crystal direction, through the harmonic form of the bandwidth
  def _poleFigure(self, h, bandwidth=None):
    return SO3FunHarmonic(self, bandwidth=bandwidth)._poleFigure(h)

  # the inverse pole figure of one specimen direction, through the harmonic form of the bandwidth
  def _inversePoleFigure(self, r, bandwidth=None):
    return SO3FunHarmonic(self, bandwidth=bandwidth)._inversePoleFigure(r)

  def radon(self, h=None, r=None, superposition=None, bandwidth=None):
    """the Radon transform of a function on SO(3): its pole figures, inverse pole figures or the
    values of either, through the harmonic form of the bandwidth

      S2F = SO3F.radon(h)                                     # the pole figure of the crystal direction h
      S2F = SO3F.radon(r=r)                                   # the inverse pole figure of the specimen direction r
      v = SO3F.radon(h, r)                                    # the pole figure of h at the directions r
      S2F = SO3F.radon([h1, h2], superposition=[0.6, 0.4])    # a superposed pole figure

    The pole figure of h at r is the mean of the function over all orientations carrying h
    onto r. A function without coefficients, a composition, a handle, is expanded to an
    SO3FunHarmonic of the bandwidth first. MATLAB's pdf, ipdf and calcPDF are this function.

    Input
      SO3F - SO3Fun
      h - Miller or vector3d, the crystal direction, or a list of them to superpose
    Output
      S2F - S2FunHarmonic, the pole figure in the specimen frame or the inverse pole figure in
          the crystal frame
      v - the values at r
    Keywords
      r - vector3d, the specimen direction of the inverse pole figure, or the directions to
          evaluate the pole figure at
      superposition - the weights of several crystal directions, equal unless given
      bandwidth - the bandwidth of the harmonic form a function without coefficients is
          expanded to
    See also
      SO3FunHarmonic.radon, plotPF, plotIPF"""
    from .funops import radon
    return radon(self, h, r, superposition, bandwidth)

  def conv(self, psi):
    """the convolution with the kernel psi, through the harmonic form"""
    return conv(SO3FunHarmonic(self), psi)

  def grad(self, x=None, side="left"):
    """the gradient with respect to the tangent space basis, at the orientations x or as a
    field, left or right sided; MATLAB's grad

      G = SO3F.grad()                       # the gradient field
      g = G(ori)                            # its values at orientations
      G = SO3F.grad(side='right')           # the right sided gradient

    A function without a harmonic form is differentiated by central differences of 0.05
    degree along the three axes, a field evaluated when asked. Evaluated directly, as
    SO3F.grad(ori), it does not return at present (a recursion); evaluate the field instead.

    Input
      SO3F - SO3Fun
    Output
      G - SO3VectorFieldHandle, the gradient field
      g - the gradient at the orientations
    Keywords
      x - orientation, where to evaluate the gradient; the field when None
      side - 'left', the tangent vectors in specimen coordinates, or 'right', in crystal
          coordinates
    See also
      SO3FunHarmonic.grad, SO3VectorField, orientation.exp"""
    from .so3vectorfield import grad
    return grad(self, x, side)

  def __add__(self, g):
    if isinstance(g, (int, float, np.number)):
      return SO3FunComposition(self, uniformODF(self.CS, self.SS, antipodal=self.antipodal) * g)
    return SO3FunComposition(self, g)

  __radd__ = __add__

  def __sub__(self, g):
    return self + (-1) * g

  def __rsub__(self, g):
    return g + (-1) * self

  def __neg__(self):
    return -1 * self

  def __truediv__(self, a):
    return self * (1 / a)

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    from .characteristics import (max_ as max, min_ as min, entropy, textureIndex, discreteSample, volume, fibreVolume,
                                  calcComponents, calcMDF, calcAngleDistribution, calcAxisDistribution, calcMIndex,
                                  calcError, centerSpecimen)
    from .polefigure import calcPoleFigure
    from .discrepancy import discrepancy
    from ..analysis.texture import calcTensor, calcLankford, calcKearnsFactor
    from ..analysis.optimalsample import optimalSample
    from ..plotting.sphericalplot import plotPF, plotIPF
    from ..plotting.sectionplot import plotSection
    def sqrt(self) -> Any:
      """the square root of the function, pointwise, as a function handle

      SO3F = sqrt(odf)"""
    def exp(self) -> Any:
      """the exponential of the function, pointwise, as a function handle

      SO3F = exp(odf)"""
    def log(self) -> Any:
      """the logarithm of the function, pointwise, as a function handle

      SO3F = log(odf)"""
    def __pow__(self, p) -> Any:
      ...
    def sum(self, axis=None) -> Any:
      """the integral over orientation space, 8 pi^2 times the mean; with an axis the sum along it,
      MATLAB's sum(odf, d) with d = axis + 1, which for a single function is the function itself

        value = sum(odf)              # the integral, 8 pi^2 for an ODF
        odf = sum(odf, axis=0)        # along an axis of a list, the function itself for one

      Input
        odf - SO3Fun
      Output
        value - the integral, a number
        odf - the function summed along the axis
      Keywords
        axis - the axis to sum along, a non-negative integer; the integral when None
      See also
        SO3FunHarmonic.sum, SO3FunHarmonic.mean"""
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


# the callable averaged over the groups of the frames on every call, and the inverses under exchange
def _symmetrisedSO3(fun, CS, SS, antipodal):
  L, R = reductionGroups(orientation(rotation.id, CS, SS))
  L, R = rotation(L), rotation(R)

  def averaged(o):
    r = rotation(o.data.reshape(-1, 4))
    acc = np.zeros(r.size)
    for i in range(L.size):
      for j in range(R.size):
        acc = acc + np.asarray(fun(orientation._fromRotation((L[i] * r * R[j]).data, (CS, SS), antipodal))).ravel()
        if antipodal:
          acc = acc + np.asarray(fun(orientation._fromRotation((L[i] * r.inv() * R[j]).data, (CS, SS), antipodal))).ravel()
    return (acc / (L.size * R.size * (2 if antipodal else 1))).reshape(o.shape)

  return averaged


class SO3FunHandle(SO3FunGeneric):
  """a function on SO(3) given by a callable that receives orientations

    SO3F = SO3FunHandle(lambda o: np.cos(angle(o, r)), cs)         # a function of the orientations of cs
    SO3F = SO3FunHandle(fun, cs, ss, bandwidth=32)                  # both frames, the bandwidth of its harmonic form
    SO3F = SO3FunHandle(fun, cs, ss, symmetrise=True)               # averaged over the groups on every call
    SO3F = SO3FunHandle(fun, cs, cs, antipodal=True)                # with exchange symmetry, f(g) = f(inv(g))
    SO3F = SO3FunHandle(odf)                                        # the handle of another SO3Fun

  Nothing is precomputed: evaluating the function calls fun on orientations in the frames of
  the function, exactly. Every operation that needs coefficients, a pole figure, a
  convolution, a rotation, goes through SO3FunHarmonic(SO3F) of the bandwidth. The callable
  is trusted to be invariant under the groups unless symmetrise asks for the average.
  Exchange symmetry needs the same frame on both sides.

  Input
    fun - callable taking an orientation and returning its values, or an SO3Fun
    frames - the crystal frame and the specimen frame, one or both, positionally
  Output
    SO3F - SO3FunHandle
  Keywords
    bandwidth - the bandwidth of the harmonic form, by default the preference
        'defaultSO3Bandwidth'
    antipodal - exchange symmetry, f(g) = f(inv(g)), for misorientations
    real - the function takes only real values; the imaginary part of the values is dropped
    symmetrise - average the callable over the groups of both frames on every call
  See also
    SO3Fun, SO3FunHarmonic, S2FunHandle, SO3FunOperations"""

  def __init__(self, fun, *frames, bandwidth=None, antipodal=False, real=True, symmetrise=False):
    if isinstance(fun, (SO3FunHarmonic, SO3FunGeneric, SO3FunRBF, SO3FunCBF)):
      f = fun
      fun, frames = (lambda o: f(o)), (f.CS, f.SS)
    self.CS, self.SS = _framesAmong(frames)
    if antipodal and self.CS is not self.SS:
      raise ValueError("exchange symmetry needs the same frame on both sides")
    self.fun = _symmetrisedSO3(fun, self.CS, self.SS, antipodal) if symmetrise else fun
    self.bandwidth = getMTEXpref("defaultSO3Bandwidth") if bandwidth is None else int(bandwidth)
    self.antipodal, self.isReal = bool(antipodal), bool(real)

  def eval(self, x):
    """the callable receives orientations in the frames of the function"""
    r = _rotationsFor(self, x)
    o = r if isinstance(r, orientation) else orientation._fromRotation(r.data, (self.CS, self.SS), self.antipodal)
    v = np.asarray(self.fun(o))
    v = v.reshape(o.shape) if v.size == o.size else v
    return np.real(v) if self.isReal and v.dtype.kind == "c" else v

  def __mul__(self, a):
    if isinstance(a, (int, float, complex, np.number)):
      return SO3FunHandle(lambda o: a * self.fun(o), self.CS, self.SS, bandwidth=self.bandwidth, antipodal=self.antipodal, real=self.isReal and np.isreal(a))
    return SO3FunHarmonic(self) * a

  __rmul__ = __mul__

  def rotate(self, r, side="left"):
    """the function rotated by r from the left or the right, through the harmonic form;
    MATLAB's rotate

      SO3F = SO3F.rotate(r)                 # rotated in specimen coordinates
      SO3F = SO3F.rotate(r, side='right')   # rotated in crystal coordinates

    The handle is expanded into its harmonic form, which is rotated.

    Input
      SO3F - SO3FunHandle
      r - rotation
    Output
      SO3F - SO3FunHarmonic
    Keywords
      side - 'left', the specimen side, or 'right', the crystal side, as MATLAB's flag 'right'
    See also
      SO3FunHarmonic.rotate"""
    return SO3FunHarmonic(self).rotate(r, side)

  def __repr__(self):
    lines = [f"SO3FunHandle ({framesLabel(self.CS, self.SS)})", f"  bandwidth hint: {self.bandwidth}"]
    if self.antipodal:
      lines.append("  antipodal: true")
    return aligned("\n".join(lines))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from .polefigure import calcPoleFigure


class SO3FunComposition(SO3FunGeneric):
  """a sum of functions of different representations: evaluation adds the values, the harmonic
  form adds the harmonic forms"""

  def __init__(self, *terms):
    flat = []
    for t in terms:
      flat += t.components if isinstance(t, SO3FunComposition) else [t]
    f = flat[0]
    for g in flat[1:]:
      if g.CS is not f.CS or g.SS is not f.SS:
        raise ValueError("the functions live in different frames")
    self.components = flat
    self._CS, self._SS = f.CS, f.SS
    self.isReal = all(t.isReal for t in flat)
    self.antipodal = all(t.antipodal for t in flat)

  @property
  def bandwidth(self):
    """the largest bandwidth of the components"""
    return max(t.bandwidth for t in self.components)

  def eval(self, x):
    """the sum of the values of the components at the orientations x"""
    return sum(t(x) for t in self.components)

  def mean(self):
    """the sum of the means of the components"""
    return sum(t.mean() for t in self.components)

  def _harmonic(self, bandwidth=None):
    L = self.bandwidth if bandwidth is None else int(bandwidth)
    return sum(SO3FunHarmonic(t, bandwidth=min(L, t.bandwidth)) for t in self.components)

  def __mul__(self, a):
    if isinstance(a, (int, float, complex, np.number)):
      return SO3FunComposition(*(a * t for t in self.components))
    return SO3FunHarmonic(self) * a

  __rmul__ = __mul__

  def rotate(self, r, side="left"):
    """the composition of the components rotated by r from the left or the right; MATLAB's rotate

      SO3F = SO3F.rotate(r)                 # rotated in specimen coordinates
      SO3F = SO3F.rotate(r, side='right')   # rotated in crystal coordinates

    Every component is rotated in its own representation.

    Input
      SO3F - SO3FunComposition
      r - rotation
    Output
      SO3F - SO3FunComposition
    Keywords
      side - 'left', the specimen side, or 'right', the crystal side, as MATLAB's flag 'right'
    See also
      SO3FunHarmonic.rotate, SO3FunRBF.rotate"""
    return SO3FunComposition(*(t.rotate(r, side) for t in self.components))

  @property
  def CS(self):
    """the crystal frame the components share"""
    return self._CS

  @property
  def SS(self):
    """the specimen frame the components share"""
    return self._SS

  @SS.setter
  def SS(self, fr):
    """odf.SS = specimenFrame('222'): every component takes the symmetry"""
    for t in self.components:
      t.SS = fr
    self._SS = fr

  def __repr__(self):
    lines = [f"SO3FunComposition ({framesLabel(self.CS, self.SS)})", f"  components: {len(self.components)}"]
    for t in self.components:
      lines.append("  " + repr(t).replace("\n", "\n  "))
    return aligned("\n".join(lines))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from .polefigure import calcPoleFigure


# ------------------------------------------------------------------ the sphere


# the callable averaged over the elements of the frame and, for axes, the antipodes
def _symmetrisedS2(fun, frame, antipodal):
  els = frame.sym.rot.flatten() if frame is not None else orthogonalTransform(rotation.id)
  if antipodal and not np.any(els.isImproper & (els.rot.angle() < 1e-8)):
    els = orthogonalTransform.cat(els, orthogonalTransform(els.rot, ~els.isImproper))

  def averaged(v):
    d = vector3d(v.data.reshape(-1, 3))
    acc = np.zeros(d.size)
    for i in range(els.size):
      moved = els[i] * d
      acc = acc + np.asarray(fun(moved if frame is None else vector3d(moved, frame=frame))).ravel()
    return (acc / els.size).reshape(v.shape)

  return averaged


# what a function on the sphere evaluated directly shares, a handle or a sum of representations:
# the integral and the norm, sums with numbers and functions, the ratio, and the pointwise
# functions as handles evaluated when asked, as MATLAB's
class _S2Direct(S2Fun):

  def sum(self):
    """the surface integral, 4 pi times the mean, MATLAB's sum"""
    return 4 * np.pi * self.mean()

  def norm(self):
    """the L2 norm, through the harmonic form"""
    return S2FunHarmonic(self).norm()

  def __truediv__(self, g):
    if isinstance(g, (int, float, complex, np.number)):
      return self * (1 / g)
    return _directRatio(self, g)

  def __add__(self, g):
    if isinstance(g, (int, float, complex, np.number)):
      return S2FunSum(self, S2FunHarmonic([np.sqrt(4 * np.pi) * g], self.frame, antipodal=self.antipodal, _internal=True))
    return S2FunSum(self, g)

  __radd__ = __add__

  def __sub__(self, g):
    return self + (-1) * g

  def __rsub__(self, g):
    return g + (-1) * self

  def __neg__(self):
    return -1 * self

  def _mapped(self, op):
    return S2FunHandle(lambda v: op(self.eval(v)), self.frame, bandwidth=self.bandwidth, antipodal=self.antipodal,
                       real=self.isReal, symmetrise=False)

  def __abs__(self):
    return self._mapped(np.abs)

  def abs(self):
    """the pointwise absolute value, a handle evaluated when asked"""
    return abs(self)

  def sqrt(self):
    """the pointwise square root, a handle evaluated when asked"""
    return self._mapped(np.sqrt)

  def __pow__(self, p):
    return self._mapped(lambda x: x ** p)


class S2FunHandle(_S2Direct):
  """a function on the sphere given by a callable that receives directions

    sF = S2FunHandle(lambda v: v.z**2)                              # a function of the directions
    sF = S2FunHandle(fun, cs)                                       # in a crystal frame
    sF = S2FunHandle(fun, cs, antipodal=True)                       # averaged over v and -v and the group
    sF = S2FunHandle(fun, cs, symmetrise=True)                      # averaged over the group on every call
    sF = S2FunHandle(fun, cs, bandwidth=64)                         # the bandwidth of its harmonic form

  Nothing is precomputed: evaluating the function calls fun on the directions, normalised and
  in the frame of the function, exactly. Every operation that needs coefficients, a
  convolution, a quadrature, a smooth plot, goes through S2FunHarmonic(sF) of the bandwidth.
  An antipodal function is averaged over v and -v and the group of the frame, unless
  symmetrise=False says the callable is symmetric by construction. An S2FunHarmonic given as
  fun becomes the handle of its evaluation.

  Input
    fun - callable taking a vector3d and returning its values, or an S2Fun
    frame - crystalFrame or specimenFrame the directions are expressed in
  Output
    sF - S2FunHandle
  Keywords
    bandwidth - the bandwidth of the harmonic form, by default the quadrature bandwidth
    antipodal - the function fulfils f(v) = f(-v)
    real - the function takes only real values; the imaginary part of the values is dropped
    symmetrise - average the callable over the group of the frame (and over v and -v when
        antipodal) on every call; by default only when antipodal
  See also
    S2Fun, S2FunHarmonic, SO3FunHandle, S2FunOperations"""

  # an axis valued handle is averaged over the group of its frame and over v, -v unless
  # symmetrise=False says the callable is symmetric by construction
  def __init__(self, fun, frame=None, *, bandwidth=None, antipodal=False, real=True, symmetrise=None):
    if isinstance(fun, S2FunHarmonic) or hasattr(fun, "_harmonicS2"):
      f = fun
      fun, frame = (lambda v: f(v)), f.frame
      antipodal = antipodal or f.antipodal
      symmetrise = False if symmetrise is None else symmetrise
    self.frame = frame
    average = (antipodal or False) if symmetrise is None else symmetrise
    self.fun = _symmetrisedS2(fun, frame, antipodal) if average else fun
    self.bandwidth = quadratureBandwidth() if bandwidth is None else int(bandwidth)
    self.antipodal, self.isReal = bool(antipodal), bool(real)

  @property
  def CS(self):
    """the frame of the function, MATLAB's sF.CS"""
    return self.frame

  def _vectorsFor(self, x):
    v = asVector3d(x)
    if self.frame is not None and v.frame is not None and v.frame is not self.frame:
      try:
        SESSION.joinFrames(v.frame, self.frame)
      except ValueError:
        raise ValueError("the directions live in another frame than the function") from None
    return vector3d(v.normalize().data, frame=self.frame)

  def __call__(self, x):
    return self.eval(x)

  def eval(self, x):
    """the values of the callable at the directions x, taken into the frame of the function"""
    v = self._vectorsFor(x)
    out = np.asarray(self.fun(v))
    out = out.reshape(v.shape) if out.size == v.size else out
    return np.real(out) if self.isReal and out.dtype.kind == "c" else out

  def _harmonicS2(self, bandwidth=None):
    L = self.bandwidth if bandwidth is None else int(bandwidth)
    return S2FunHarmonic(lambda v: self.eval(v), self.frame, bandwidth=L, antipodal=self.antipodal, real=self.isReal)

  def mean(self):
    """the mean over the sphere, through the harmonic form"""
    return S2FunHarmonic(self).mean()

  def __mul__(self, a):
    if isinstance(a, (int, float, complex, np.number)):
      return S2FunHandle(lambda v: a * self.fun(v), self.frame, bandwidth=self.bandwidth, antipodal=self.antipodal, real=self.isReal and np.isreal(a), symmetrise=False)
    return S2FunHarmonic(self) * a

  __rmul__ = __mul__

  def __rtruediv__(self, a):
    return S2FunHandle(lambda v: a / self.eval(v), self.frame, bandwidth=self.bandwidth, antipodal=self.antipodal, real=self.isReal, symmetrise=False)

  def rotate(self, r):
    """the function rotated by r, through the harmonic form"""
    return S2FunHarmonic(self).rotate(r)

  def exp(self):
    """the pointwise exponential, a handle evaluated when asked"""
    return self._mapped(np.exp)

  def log(self):
    """the pointwise logarithm, a handle evaluated when asked"""
    return self._mapped(np.log)

  def __repr__(self):
    info = "" if self.frame is None else f" ({frameLabel(self.frame)})"
    lines = [f"S2FunHandle{info}", f"  bandwidth hint: {self.bandwidth}"]
    if self.antipodal:
      lines.append("  antipodal: true")
    return aligned("\n".join(lines))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from .characteristics import max_ as max, min_ as min, entropy, textureIndex, discreteSample
    from .discrepancy import discrepancy
    from ..analysis.optimalsample import optimalSample
    from ..plotting.sphericalplot import plot, contourf, contour, pcolor
    from ..plotting.spaceplot import plot3d


# a ratio of two functions on the sphere stays direct, its polarity untouched
def _directRatio(f, g):
  if f.frame is not g.frame:
    raise ValueError("the functions live in different frames")
  # the operands are symmetric already, so their ratio is
  return S2FunHandle(lambda v: f(v) / g(v), f.frame, bandwidth=max(f.bandwidth, g.bandwidth), antipodal=f.antipodal and g.antipodal, real=f.isReal and g.isReal, symmetrise=False)


class S2FunSum(_S2Direct):
  """a sum of functions on the sphere of different representations: evaluation adds the values,
  the harmonic form adds the harmonic forms; what + returns when coefficients cannot be added"""

  def __init__(self, *terms):
    flat = []
    for t in terms:
      flat += t.components if isinstance(t, S2FunSum) else [t]
    f = flat[0]
    for g in flat[1:]:
      if g.frame is not f.frame:
        raise ValueError("the functions live in different frames")
    self.components, self.frame = flat, f.frame
    self.isReal = all(t.isReal for t in flat)
    self.antipodal = all(t.antipodal for t in flat)

  @property
  def CS(self):
    """the frame the summands share, MATLAB's sF.CS"""
    return self.frame

  @property
  def bandwidth(self):
    """the largest bandwidth of the summands"""
    return max(t.bandwidth for t in self.components)

  def __call__(self, x):
    return self.eval(x)

  def eval(self, x):
    """the sum of the values of the summands at the directions x"""
    return sum(t(x) for t in self.components)

  def mean(self):
    """the sum of the means of the summands"""
    return sum(t.mean() for t in self.components)

  def _harmonicS2(self, bandwidth=None):
    L = self.bandwidth if bandwidth is None else int(bandwidth)
    return sum(S2FunHarmonic(t, bandwidth=min(L, t.bandwidth)) for t in self.components)

  def __mul__(self, a):
    if isinstance(a, (int, float, complex, np.number)):
      return S2FunSum(*(a * t for t in self.components))
    return S2FunHarmonic(self) * a

  __rmul__ = __mul__

  def __repr__(self):
    info = "" if self.frame is None else f" ({frameLabel(self.frame)})"
    lines = [f"S2FunSum{info}", f"  components: {len(self.components)}"]
    for t in self.components:
      lines.append("  " + repr(t).replace("\n", "\n  "))
    return aligned("\n".join(lines))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from .characteristics import max_ as max, min_ as min, entropy, textureIndex, discreteSample
    from .discrepancy import discrepancy
    from ..analysis.optimalsample import optimalSample
    from ..plotting.sphericalplot import plot, contourf, contour, pcolor
    from ..plotting.spaceplot import plot3d


class S1FunHandle:
  """a function on the circle given by a callable that receives angles

    S1FunHandle(fun)
    S1FunHandle(fun, bandwidth=L)     the bandwidth of its harmonic form"""

  def __init__(self, fun, *, bandwidth=None, real=True):
    self.fun = fun
    self.bandwidth = getMTEXpref("maxS1Bandwidth") if bandwidth is None else int(bandwidth)
    self.isReal = bool(real)

  def __call__(self, x):
    return self.eval(x)

  def eval(self, x):
    """the values of the callable at the angles x"""
    out = np.asarray(self.fun(np.asarray(x, dtype=float)))
    return np.real(out) if self.isReal and out.dtype.kind == "c" else out

  def _harmonicS1(self, bandwidth=None):
    L = self.bandwidth if bandwidth is None else int(bandwidth)
    return S1FunHarmonic(lambda x: self.eval(x), bandwidth=L, real=self.isReal)

  def mean(self):
    """the mean over the circle, through the harmonic form"""
    return S1FunHarmonic(self).mean()

  def max(self, *args, numLocal=1, resolution=0.25 * degree, bandwidth=None):
    """the global, local and pointwise maxima of a function on the circle given by a callable

      v, pos = max(sF)                              # the maximum and the angle it is taken at
      values, positions = max(sF, numLocal=3)       # the three largest local maxima
      sF = max(sF, 0.5)                             # the pointwise maximum with a constant
      sF = max(sF1, sF2, bandwidth=32)              # the pointwise maximum of two functions

    The maxima are searched on an equispaced raster of the resolution and refined by a ternary
    search about every peak; numLocal returns the largest local maxima, in descending order.
    With a number or a second function the result is the pointwise maximum, an S1FunHandle, or
    an S1FunHarmonic of the bandwidth when one of the two is harmonic; as S1FunHarmonic.max.

    Input
      sF, sF1, sF2 - S1Fun
    Output
      v, values - the largest value, or an array of the local maxima
      pos, positions - the angle in radians, or an array of them
      sF - S1FunHandle or S1FunHarmonic, the pointwise maximum
    Keywords
      numLocal - the number of local maxima to return
      resolution - the spacing of the raster the search starts from
      bandwidth - the bandwidth of the pointwise maximum, by default the larger of the two
    See also
      S1FunHandle.min, S1FunHarmonic.max, S1FunHandle"""
    if args:
      return _pointwiseExtremumS1(np.maximum, self, args[0], bandwidth)
    return _localExtremaS1(self, 1, numLocal, resolution)

  def min(self, *args, numLocal=1, resolution=0.25 * degree, bandwidth=None):
    """the global, local and pointwise minima of a function on the circle given by a callable

      v, pos = min(sF)                              # the minimum and the angle it is taken at
      values, positions = min(sF, numLocal=3)       # the three smallest local minima
      sF = min(sF, 0.5)                             # the pointwise minimum with a constant
      sF = min(sF1, sF2, bandwidth=32)              # the pointwise minimum of two functions

    The minima are searched on an equispaced raster of the resolution and refined by a ternary
    search about every peak; numLocal returns the smallest local minima, in ascending order.
    With a number or a second function the result is the pointwise minimum, an S1FunHandle, or
    an S1FunHarmonic of the bandwidth when one of the two is harmonic; as S1FunHarmonic.min.

    Input
      sF, sF1, sF2 - S1Fun
    Output
      v, values - the smallest value, or an array of the local minima
      pos, positions - the angle in radians, or an array of them
      sF - S1FunHandle or S1FunHarmonic, the pointwise minimum
    Keywords
      numLocal - the number of local minima to return
      resolution - the spacing of the raster the search starts from
      bandwidth - the bandwidth of the pointwise minimum, by default the larger of the two
    See also
      S1FunHandle.max, S1FunHarmonic.min, S1FunHandle"""
    if args:
      return _pointwiseExtremumS1(np.minimum, self, args[0], bandwidth)
    return _localExtremaS1(self, -1, numLocal, resolution)

  def __mul__(self, a):
    return S1FunHandle(lambda x: a * self.fun(x), bandwidth=self.bandwidth, real=self.isReal and np.isreal(a))

  __rmul__ = __mul__

  def __add__(self, g):
    return S1FunHarmonic(self) + g

  __radd__ = __add__

  def __repr__(self):
    return aligned(f"S1FunHandle\n  bandwidth hint: {self.bandwidth}")


# the pointwise functions of any function on SO(3) are handles, evaluated when asked, as
# MATLAB's abs, sqrt, exp, log and .^
def _pointwiseSO3(f, op):
  return SO3FunHandle(lambda o: op(f.eval(o)), f.CS, f.SS, bandwidth=f.bandwidth)


# a pointwise operation as a method returning a function handle
def _pointwiseMethod(op, what, name):
  def f(self):
    return _pointwiseSO3(self, op)
  f.__doc__ = f"the {what} of the function, pointwise, as a function handle\n\n  SO3F = {name}(odf)"
  return f


def _integralSO3(self, axis=None):
  """the integral over orientation space, 8 pi^2 times the mean; with an axis the sum along it,
  MATLAB's sum(odf, d) with d = axis + 1, which for a single function is the function itself

    value = sum(odf)              # the integral, 8 pi^2 for an ODF
    odf = sum(odf, axis=0)        # along an axis of a list, the function itself for one

  Input
    odf - SO3Fun
  Output
    value - the integral, a number
    odf - the function summed along the axis
  Keywords
    axis - the axis to sum along, a non-negative integer; the integral when None
  See also
    SO3FunHarmonic.sum, SO3FunHarmonic.mean"""
  if axis is None:
    return 8 * np.pi ** 2 * self.mean()
  # a single function is a list of one along every axis
  if not isinstance(axis, (int, np.integer)) or axis < 0:
    raise ValueError("the axis to sum along is a non-negative integer")
  return self


# the classes keep what they define themselves, the harmonic form its own products
_POINTWISE = {"__abs__": _pointwiseMethod(np.abs, "absolute value", "abs"), "abs": _pointwiseMethod(np.abs, "absolute value", "abs"),
              "sqrt": _pointwiseMethod(np.sqrt, "square root", "sqrt"), "exp": _pointwiseMethod(np.exp, "exponential", "exp"),
              "log": _pointwiseMethod(np.log, "logarithm", "log"), "__pow__": lambda self, p: _pointwiseSO3(self, lambda x: x ** p),
              "sum": _integralSO3}
for _cls in (SO3FunHarmonic, SO3FunRBF, SO3FunCBF, SO3FunGeneric):
  for _name, _f in _POINTWISE.items():
    if _name not in vars(_cls):
      install(_cls, _f, _name)
# the two frames under their names of the Julia port; assigning one relabels the function and
# leaves its values, symmetrise then projects
installFramePair((SO3FunHarmonic, SO3FunRBF, SO3FunCBF, SO3FunGeneric))

__all__ = ["SO3FunHandle", "SO3FunComposition", "S2FunHandle", "S2FunSum", "S1FunHandle"]
