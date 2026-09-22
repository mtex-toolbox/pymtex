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
from .s2fun import S2FunHarmonic
from .so3fun import SO3FunHarmonic


# what every representation on SO(3) without an exact harmonic form shares: evaluation in its
# frames, the operations through the harmonic form, and sums as compositions
class SO3FunGeneric:
  isReal = True
  antipodal = False

  def __call__(self, x):
    return self.eval(x)

  def _harmonic(self, bandwidth=None):
    L = self.bandwidth if bandwidth is None else int(bandwidth)
    return SO3FunHarmonic(lambda o: self.eval(o), self.CS, self.SS, bandwidth=L, real=self.isReal, antipodal=self.antipodal)

  def mean(self):
    return SO3FunHarmonic(self).mean()

  def norm(self):
    return SO3FunHarmonic(self).norm()

  def pdf(self, h, bandwidth=None):
    return SO3FunHarmonic(self, bandwidth=bandwidth).pdf(h)

  def ipdf(self, r, bandwidth=None):
    return SO3FunHarmonic(self, bandwidth=bandwidth).ipdf(r)

  def radon(self, h=None, r=None):
    return SO3FunHarmonic(self).radon(h, r)

  def calcPDF(self, h, r=None, superposition=None):
    from .density import calcPDF
    return calcPDF(self, h, r, superposition=superposition)

  def conv(self, psi):
    return conv(SO3FunHarmonic(self), psi)

  def grad(self, x=None, side="left"):
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

  def __init__(self, fun, *frames, bandwidth=None, antipodal=False, real=True, symmetrise=False):
    if isinstance(fun, (SO3FunHarmonic, SO3FunGeneric, SO3FunRBF, SO3FunCBF)):
      f = fun
      fun, frames = (lambda o: f(o)), (f.CS, f.SS)
    self.CS, self.SS = _readFrames(frames)
    if antipodal and self.CS is not self.SS:
      raise ValueError("exchange symmetry needs the same frame on both sides")
    self.fun = _symmetrisedSO3(fun, self.CS, self.SS, antipodal) if symmetrise else fun
    self.bandwidth = getMTEXpref("maxSO3Bandwidth") if bandwidth is None else int(bandwidth)
    self.antipodal, self.isReal = bool(antipodal), bool(real)

  # the callable receives orientations in the frames of the function
  def eval(self, x):
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
    return SO3FunHarmonic(self).rotate(r, side)

  def __repr__(self):
    lines = [f"SO3FunHandle ({self.CS.char()} → {self.SS.char()})", f"  bandwidth hint: {self.bandwidth}"]
    if self.antipodal:
      lines.append("  antipodal: true")
    return "\n".join(lines)


# a sum of functions of different representations: evaluation adds the values, the harmonic
# form adds the harmonic forms
class SO3FunComposition(SO3FunGeneric):

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
    return max(t.bandwidth for t in self.components)

  def eval(self, x):
    return sum(t(x) for t in self.components)

  def mean(self):
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
    return SO3FunComposition(*(t.rotate(r, side) for t in self.components))

  @property
  def CS(self):
    return self._CS

  @property
  def SS(self):
    return self._SS

  # odf.SS = specimenFrame('222'): every component takes the symmetry
  @SS.setter
  def SS(self, fr):
    for t in self.components:
      t.SS = fr
    self._SS = fr

  def __repr__(self):
    lines = [f"SO3FunComposition ({self.CS.char()} → {self.SS.char()})", f"  components: {len(self.components)}"]
    for t in self.components:
      lines.append("  " + repr(t).replace("\n", "\n  "))
    return "\n".join(lines)


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
    self.bandwidth = getMTEXpref("maxS2Bandwidth") if bandwidth is None else int(bandwidth)
    self.antipodal, self.isReal = bool(antipodal), bool(real)

  @property
  def CS(self):
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
    v = self._vectorsFor(x)
    out = np.asarray(self.fun(v))
    out = out.reshape(v.shape) if out.size == v.size else out
    return np.real(out) if self.isReal and out.dtype.kind == "c" else out

  def _harmonicS2(self, bandwidth=None):
    L = self.bandwidth if bandwidth is None else int(bandwidth)
    return S2FunHarmonic(lambda v: self.eval(v), self.frame, bandwidth=L, antipodal=self.antipodal, real=self.isReal)

  def mean(self):
    return S2FunHarmonic(self).mean()

  def norm(self):
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
      return S2FunSum(self, S2FunHarmonic([np.sqrt(4 * np.pi) * g], self.frame, antipodal=self.antipodal))
    return S2FunSum(self, g)

  __radd__ = __add__

  def __sub__(self, g):
    return self + (-1) * g

  def __rsub__(self, g):
    return g + (-1) * self

  def __neg__(self):
    return -1 * self

  def rotate(self, r):
    return S2FunHarmonic(self).rotate(r)

  def __repr__(self):
    info = "" if self.frame is None else f" ({self.frame.char()})"
    lines = [f"S2FunHandle{info}", f"  bandwidth hint: {self.bandwidth}"]
    if self.antipodal:
      lines.append("  antipodal: true")
    return "\n".join(lines)


# a ratio of two functions on the sphere stays direct, its polarity untouched
def _directRatio(f, g):
  if f.frame is not g.frame:
    raise ValueError("the functions live in different frames")
  # the operands are symmetric already, so their ratio is
  return S2FunHandle(lambda v: f(v) / g(v), f.frame, bandwidth=max(f.bandwidth, g.bandwidth), antipodal=f.antipodal and g.antipodal, real=f.isReal and g.isReal, symmetrise=False)


class S2FunSum:

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
    return self.frame

  @property
  def bandwidth(self):
    return max(t.bandwidth for t in self.components)

  def __call__(self, x):
    return self.eval(x)

  def eval(self, x):
    return sum(t(x) for t in self.components)

  def mean(self):
    return sum(t.mean() for t in self.components)

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
      return S2FunSum(self, S2FunHarmonic([np.sqrt(4 * np.pi) * g], self.frame, antipodal=self.antipodal))
    return S2FunSum(self, g)

  __radd__ = __add__

  def __sub__(self, g):
    return self + (-1) * g

  def __neg__(self):
    return -1 * self

  def __repr__(self):
    info = "" if self.frame is None else f" ({self.frame.char()})"
    lines = [f"S2FunSum{info}", f"  components: {len(self.components)}"]
    for t in self.components:
      lines.append("  " + repr(t).replace("\n", "\n  "))
    return "\n".join(lines)


class S1FunHandle:

  def __init__(self, fun, *, bandwidth=None, real=True):
    self.fun = fun
    self.bandwidth = getMTEXpref("maxS1Bandwidth") if bandwidth is None else int(bandwidth)
    self.isReal = bool(real)

  def __call__(self, x):
    return self.eval(x)

  def eval(self, x):
    out = np.asarray(self.fun(np.asarray(x, dtype=float)))
    return np.real(out) if self.isReal and out.dtype.kind == "c" else out

  def _harmonicS1(self, bandwidth=None):
    L = self.bandwidth if bandwidth is None else int(bandwidth)
    return S1FunHarmonic(lambda x: self.eval(x), bandwidth=L, real=self.isReal)

  def mean(self):
    return S1FunHarmonic(self).mean()

  def __mul__(self, a):
    return S1FunHandle(lambda x: a * self.fun(x), bandwidth=self.bandwidth, real=self.isReal and np.isreal(a))

  __rmul__ = __mul__

  def __add__(self, g):
    return S1FunHarmonic(self) + g

  __radd__ = __add__

  def __repr__(self):
    return f"S1FunHandle\n  bandwidth hint: {self.bandwidth}"


__all__ = ["SO3FunHandle", "SO3FunComposition", "S2FunHandle", "S2FunSum", "S1FunHandle"]
