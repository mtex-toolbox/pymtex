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

from ..geometry.frames import crystalFrame, referenceFrame, specimenFrame
from ..geometry.orientation import orientation
from ..geometry.quaternion import orthogonalTransform, quaternion, rotation
from ..geometry.session import SESSION, getMTEXpref
from ..geometry.symreduce import reductionGroups
from ..geometry.vector3d import vector3d
from .funops import conv
from .rbf import SO3FunCBF, SO3FunRBF, _readFrames, _rotationsFor, uniformODF
from .s1fun import S1FunHarmonic
from .s2fun import S2FunHarmonic, quadratureBandwidth
from .so3fun import SO3FunHarmonic
from ..geometry.install import install
from ..geometry.display import frameLabel, framesLabel
from ..geometry.display import aligned


class SO3FunGeneric:
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

  def pdf(self, h, bandwidth=None):
    """the pole figure of the crystal direction h, through the harmonic form of the bandwidth"""
    return SO3FunHarmonic(self, bandwidth=bandwidth).pdf(h)

  def ipdf(self, r, bandwidth=None):
    """the inverse pole figure of the specimen direction r, through the harmonic form of the bandwidth"""
    return SO3FunHarmonic(self, bandwidth=bandwidth).ipdf(r)

  def radon(self, h=None, r=None):
    """the Radon transform through the harmonic form: radon(h) the pole figure, radon(r=r) the
    inverse pole figure, radon(h, r) the values"""
    return SO3FunHarmonic(self).radon(h, r)

  def calcPDF(self, h, r=None, superposition=None):
    """the pole figure of superposed crystal directions, as a function or at specimen directions r"""
    from .density import calcPDF
    return calcPDF(self, h, r, superposition=superposition)

  def conv(self, psi):
    """the convolution with the kernel psi, through the harmonic form"""
    return conv(SO3FunHarmonic(self), psi)

  def grad(self, x=None, side="left"):
    """the gradient, at the orientations x or as a field, left or right sided"""
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
    from ..analysis.texture import calcTensor, calcLankford, calcKearnsFactor
    from ..analysis.optimalsample import optimalSample
    from ..plotting.sphericalplot import plotPDF, plotIPDF
    from ..plotting.sectionplot import plotSection
    def sqrt(self) -> Any:
      ...
    def exp(self) -> Any:
      ...
    def log(self) -> Any:
      ...
    def __pow__(self, p) -> Any:
      ...
    def sum(self) -> Any:
      ...
    @property
    def SRight(self) -> Any:
      ...
    @SRight.setter
    def SRight(self, value) -> None:
      ...
    @property
    def SLeft(self) -> Any:
      ...
    @SLeft.setter
    def SLeft(self, value) -> None:
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
  """a function on SO(3) given by a callable that receives orientations in the frames

    SO3FunHandle(fun, CS, SS)
    SO3FunHandle(fun, CS, SS, bandwidth=L)     the bandwidth of its harmonic form
    SO3FunHandle(fun, CS, SS, symmetrise=True)  averaged over the groups on every call
    SO3FunHandle(fun, CS, antipodal=True)       with exchange symmetry"""

  def __init__(self, fun, *frames, bandwidth=None, antipodal=False, real=True, symmetrise=False):
    if isinstance(fun, (SO3FunHarmonic, SO3FunGeneric, SO3FunRBF, SO3FunCBF)):
      f = fun
      fun, frames = (lambda o: f(o)), (f.CS, f.SS)
    self.CS, self.SS = _readFrames(frames)
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
    """the function rotated by r from the left or the right, through the harmonic form"""
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
    """the composition of the components rotated by r from the left or the right"""
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


class S2FunHandle:
  """a function on the sphere given by a callable that receives directions

    S2FunHandle(fun, frame)
    S2FunHandle(fun, frame, bandwidth=L)       the bandwidth of its harmonic form
    S2FunHandle(fun, frame, antipodal=True)    averaged over v and -v and the group
    S2FunHandle(fun, frame, symmetrise=True)   averaged over the group of the frame on every call"""

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
    v = x if isinstance(x, vector3d) else vector3d(x)
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

  def sum(self):
    """the surface integral, MATLAB's sum"""
    return 4 * np.pi * self.mean()

  def norm(self):
    """the L2 norm, through the harmonic form"""
    return S2FunHarmonic(self).norm()

  def __mul__(self, a):
    if isinstance(a, (int, float, complex, np.number)):
      return S2FunHandle(lambda v: a * self.fun(v), self.frame, bandwidth=self.bandwidth, antipodal=self.antipodal, real=self.isReal and np.isreal(a), symmetrise=False)
    return S2FunHarmonic(self) * a

  __rmul__ = __mul__

  def __truediv__(self, g):
    if isinstance(g, (int, float, complex, np.number)):
      return self * (1 / g)
    return _directRatio(self, g)

  def __rtruediv__(self, a):
    return S2FunHandle(lambda v: a / self.eval(v), self.frame, bandwidth=self.bandwidth, antipodal=self.antipodal, real=self.isReal, symmetrise=False)

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

  def rotate(self, r):
    """the function rotated by r, through the harmonic form"""
    return S2FunHarmonic(self).rotate(r)

  # the pointwise functions of a handle stay handles, evaluated when asked, as MATLAB's
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

  def exp(self):
    """the pointwise exponential, a handle evaluated when asked"""
    return self._mapped(np.exp)

  def log(self):
    """the pointwise logarithm, a handle evaluated when asked"""
    return self._mapped(np.log)

  def __pow__(self, p):
    return self._mapped(lambda x: x ** p)

  def __repr__(self):
    info = "" if self.frame is None else f" ({frameLabel(self.frame)})"
    lines = [f"S2FunHandle{info}", f"  bandwidth hint: {self.bandwidth}"]
    if self.antipodal:
      lines.append("  antipodal: true")
    return aligned("\n".join(lines))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from .characteristics import max_ as max, min_ as min, entropy, textureIndex, discreteSample
    from ..analysis.optimalsample import optimalSample
    from ..plotting.sphericalplot import plot, contourf, contour, pcolor
    from ..plotting.spaceplot import plot3d


# a ratio of two functions on the sphere stays direct, its polarity untouched
def _directRatio(f, g):
  if f.frame is not g.frame:
    raise ValueError("the functions live in different frames")
  # the operands are symmetric already, so their ratio is
  return S2FunHandle(lambda v: f(v) / g(v), f.frame, bandwidth=max(f.bandwidth, g.bandwidth), antipodal=f.antipodal and g.antipodal, real=f.isReal and g.isReal, symmetrise=False)


class S2FunSum:
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

  def sum(self):
    """the surface integral, 4 pi times the mean"""
    return 4 * np.pi * self.mean()

  def norm(self):
    """the L2 norm, through the harmonic form"""
    return S2FunHarmonic(self).norm()

  def _harmonicS2(self, bandwidth=None):
    L = self.bandwidth if bandwidth is None else int(bandwidth)
    return sum(S2FunHarmonic(t, bandwidth=min(L, t.bandwidth)) for t in self.components)

  def __mul__(self, a):
    if isinstance(a, (int, float, complex, np.number)):
      return S2FunSum(*(a * t for t in self.components))
    return S2FunHarmonic(self) * a

  __rmul__ = __mul__

  def __truediv__(self, g):
    return self * (1 / g) if isinstance(g, (int, float, complex, np.number)) else _directRatio(self, g)

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

  # the pointwise functions of a sum are handles, evaluated when asked
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

  def __repr__(self):
    info = "" if self.frame is None else f" ({frameLabel(self.frame)})"
    lines = [f"S2FunSum{info}", f"  components: {len(self.components)}"]
    for t in self.components:
      lines.append("  " + repr(t).replace("\n", "\n  "))
    return aligned("\n".join(lines))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from .characteristics import max_ as max, min_ as min, entropy, textureIndex, discreteSample
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


# the classes keep what they define themselves, the harmonic form its own products
_POINTWISE = {"__abs__": lambda self: _pointwiseSO3(self, np.abs), "abs": lambda self: _pointwiseSO3(self, np.abs),
              "sqrt": lambda self: _pointwiseSO3(self, np.sqrt), "exp": lambda self: _pointwiseSO3(self, np.exp),
              "log": lambda self: _pointwiseSO3(self, np.log), "__pow__": lambda self, p: _pointwiseSO3(self, lambda x: x ** p),
              "sum": lambda self: 8 * np.pi ** 2 * self.mean()}
for _cls in (SO3FunHarmonic, SO3FunRBF, SO3FunCBF, SO3FunGeneric):
  for _name, _f in _POINTWISE.items():
    if _name not in vars(_cls):
      install(_cls, _f, _name)
  # MATLAB's names of the crystal and the specimen frame; assigning one relabels the
  # function and leaves its values, symmetrise then projects
  install(_cls, property(lambda self: self.CS, lambda self, fr: setattr(self, "CS", fr)), "SRight")
  install(_cls, property(lambda self: self.SS, lambda self, fr: setattr(self, "SS", fr)), "SLeft")

__all__ = ["SO3FunHandle", "SO3FunComposition", "S2FunHandle", "S2FunSum", "S1FunHandle"]
