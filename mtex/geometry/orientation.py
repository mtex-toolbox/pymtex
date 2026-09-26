# orientations and misorientations (MATLAB @orientation)
#
#   orientation(rot, cs, ss)              maps coordinates in cs (frameA) to coordinates in ss (frameB)
#   orientation(a, b, c, d, cs, ss)
#   orientation.byEuler(phi1, Phi, phi2, cs, ss)
#   orientation.byEuler(alpha, beta, gamma, 'ZYZ', cs, ss)
#   orientation.byAxisAngle(v, omega, cs, ss)
#   orientation.byMatrix(M, cs, ss)
#   orientation.map(u1, v1, u2, v2, cs, ss)
#   orientation.rand(n, cs, ss), orientation.id(cs, ss), orientation.cube(cs)
#
# The two frames are named by position: CS is frameA and SS is frameB, so inv(ori).CS is
# the specimen side. A misorientation is one between two crystal frames, and only one
# within a single frame may claim grain exchange symmetry, ori.antipodal (Julia ADR 0008).
# Products, symmetrisation and distances are the business of G5.
from __future__ import annotations

import numpy as np

from ._array import MtexArray
from .display import sizeString
from .frames import crystalFrame, referenceFrame, specimenFrame
from .quaternion import _readCoefficients, orthogonalTransform, qrot, quaternion, rotation
from .homochoric import homochoric2quat
from .representations import axis2quat, euler2quat, eulerTriple, fitKernel, map1Kernel, map2Kernel, mat2quat, rodrigues2quat
from .session import SESSION
from .units import degree
from .vector3d import classproperty, vector3d


# the frames among the arguments, positionally: the first is CS, the second SS
def _splitFrames(args):
  args = list(args)
  frames = [a for a in args if isinstance(a, referenceFrame)]
  rest = [a for a in args if not isinstance(a, referenceFrame)]
  return rest, frames


# the symmetric equivalent of the largest rotation angle, the one MATLAB's 'max' names: a
# twin law is 86.3 degrees as a disorientation and 180 degrees about its twin axis
def _largestImage(o):
  from .symmetryops import symmetriseOrientation
  es = symmetriseOrientation(o).data
  k = np.argmin(np.abs(es[..., 0]), axis=0)
  best = np.take_along_axis(es, k[None, ..., None], axis=0)[0]
  return orientation._fromRotation(best, (o.CS, o.SS), o.antipodal)


class orientation(rotation):
  _carried = ("frameA", "frameB", "antipodal", "notIID")
  # orientations that are not independent draws, the pixels of a map: an automatic halfwidth
  # asks for their grains (docs/plans/halfwidth.md, section 10)
  notIID = False

  def __init__(self, *args, antipodal=False, dtype=None, points=1000):
    args, frames = _splitFrames(args)
    frameA = frames[0] if frames else None
    frameB = frames[1] if len(frames) > 1 else None
    # orientation(f, points=n) samples a fibre
    if args and hasattr(args[0], "isFull") and hasattr(args[0], "o1"):
      args = [args[0].orientation(points)] + list(args[1:])
    # orientation(e) projects an embedding back onto the orientations
    elif args and hasattr(args[0], "spec") and hasattr(args[0], "project"):
      args = [args[0].orientation()] + list(args[1:])
    notIID = False
    if args and isinstance(args[0], orientation):
      o = args[0]
      data = o.data.copy()
      antipodal = antipodal or o.antipodal
      notIID = o.notIID
      frameA = o.frameA if frameA is None else frameA
      frameB = o.frameB if frameB is None else frameB
    elif args and isinstance(args[0], orthogonalTransform):
      data = rotation(args[0]).data
    elif args and isinstance(args[0], quaternion):
      data = rotation(args[0]).data
    else:
      data, obj = _readCoefficients(orientation, args, 4)
      if obj is not None:
        raise TypeError(f"cannot build an orientation from a {type(obj).__name__}")
      if data.size:
        data = data / np.linalg.norm(data, axis=-1, keepdims=True)
    MtexArray.__init__(self, data, dtype)
    self.frameA = crystalFrame("1") if frameA is None else frameA
    self.frameB = specimenFrame.default if frameB is None else frameB
    self.antipodal = bool(antipodal)
    if notIID:
      self.notIID = True
    _checkExchange(self.antipodal, self.frameA, self.frameB)

  # ------------------------------------------------------------------ constructors

  @classmethod
  def _fromRotation(cls, data, frames, antipodal=False):
    frames = list(frames)
    frameA = frames[0] if frames else crystalFrame("1")
    frameB = frames[1] if len(frames) > 1 else specimenFrame.default
    _checkExchange(antipodal, frameA, frameB)
    obj = object.__new__(cls)
    obj.data = np.asarray(data, dtype=float)
    obj.frameA, obj.frameB, obj.antipodal = frameA, frameB, antipodal
    return obj

  @classmethod
  def byEuler(cls, a, b=None, c=None, *args, antipodal=False):
    a, b, c, args = eulerTriple(a, b, c, args)
    args, frames = _splitFrames(args)
    convention = args[0] if args else None
    return cls._fromRotation(euler2quat(a, b, c, convention), frames, antipodal)

  # orientation.load(fname, cs, columnNames=['phi1', 'Phi', 'phi2']): the orientations of a
  # text table; loadOrientation returns the other columns too
  @classmethod
  def load(cls, fname, *args, **kwargs):
    from ..io.orientations import loadOrientation
    return loadOrientation(fname, *args, **kwargs)[0]

  @classmethod
  def byAxisAngle(cls, v, omega, *frames, antipodal=False):
    frames = _splitFrames(frames)[1]
    # about a crystal direction without frames named: the rotation of that crystal frame into itself
    if not frames and isinstance(v, vector3d) and isinstance(v.frame, crystalFrame):
      frames = [v.frame, v.frame]
    return cls._fromRotation(axis2quat(_xyz(v), omega), frames, antipodal)

  @classmethod
  def byMatrix(cls, M, *frames, antipodal=False):
    M = np.asarray(M, dtype=float)
    if np.any(np.linalg.det(M) < 0):
      raise ValueError("the matrix has determinant -1; an orientation is proper")
    return cls._fromRotation(mat2quat(M), _splitFrames(frames)[1], antipodal)

  @classmethod
  def byRodrigues(cls, v, *frames, antipodal=False):
    return cls._fromRotation(rodrigues2quat(_xyz(v)), _splitFrames(frames)[1], antipodal)

  @classmethod
  def byHomochoric(cls, v, *frames, antipodal=False):
    return cls._fromRotation(homochoric2quat(_xyz(v)), _splitFrames(frames)[1], antipodal)

  # the smallest rotation taking u1 to v1, or the one taking u1 to v1 and u2 to v2; the frames
  # of framed inputs are adopted, CS from u1 and SS from v1
  @classmethod
  def map(cls, u1, v1, *args, antipodal=False):
    args, frames = _splitFrames(args)
    if not frames:
      frames = [f for f in (_frameOf(u1), _frameOf(v1)) if f is not None]
      if len(frames) == 1 and _frameOf(u1) is None:
        frames = [crystalFrame("1"), frames[0]]
    if args:
      return cls._fromRotation(map2Kernel(_xyz(u1), _xyz(v1), _xyz(args[0]), _xyz(args[1])), frames, antipodal)
    return cls._fromRotation(map1Kernel(_xyz(u1), _xyz(v1)), frames, antipodal)

  @classmethod
  def fit(cls, u, v, *frames, weights=None):
    return cls._fromRotation(fitKernel(_xyz(u), _xyz(v), weights), _splitFrames(frames)[1])

  @classmethod
  def rand(cls, *args, rng=None, antipodal=False, maxAngle=None):
    args, frames = _splitFrames(args)
    return cls._fromRotation(rotation.rand(*args, rng=rng, maxAngle=maxAngle).data, frames, antipodal)

  @classmethod
  def id(cls, *args):
    args, frames = _splitFrames(args)
    shape = tuple(args[0]) if args and isinstance(args[0], (tuple, list)) else tuple(args)
    return cls._fromRotation(np.broadcast_to([1.0, 0.0, 0.0, 0.0], shape + (4,)).copy(), frames)

  # orientation.eye(cs): MATLAB's other name of the identity
  @classmethod
  def eye(cls, *args):
    return cls.id(*args)

  @classmethod
  def nan(cls, *args):
    args, frames = _splitFrames(args)
    shape = tuple(args[0]) if args and isinstance(args[0], (tuple, list)) else tuple(args)
    return cls._fromRotation(np.full(shape + (4,), np.nan), frames)

  # the standard orientations, by their Euler angles
  cube = classmethod(lambda cls, *f: cls.byEuler(0, 0, 0, *f))
  cubeND22 = classmethod(lambda cls, *f: cls.byEuler(22 * degree, 0, 0, *f))
  cubeND45 = classmethod(lambda cls, *f: cls.byEuler(45 * degree, 0, 0, *f))
  cubeRD = classmethod(lambda cls, *f: cls.byEuler(0, 22 * degree, 0, *f))
  goss = classmethod(lambda cls, *f: cls.byEuler(0, 45 * degree, 0, *f))
  invGoss = classmethod(lambda cls, *f: cls.byEuler(90 * degree, 45 * degree, 0, *f))
  brass = classmethod(lambda cls, *f: cls.byEuler(35 * degree, 45 * degree, 0, *f))
  brass2 = classmethod(lambda cls, *f: cls.byEuler(325 * degree, 45 * degree, 0, *f))
  SR = classmethod(lambda cls, *f: cls.byEuler(53 * degree, 35 * degree, 63 * degree, *f))
  SR2 = classmethod(lambda cls, *f: cls.byEuler(233 * degree, 35 * degree, 63 * degree, *f))
  SR3 = classmethod(lambda cls, *f: cls.byEuler(307 * degree, 35 * degree, 27 * degree, *f))
  SR4 = classmethod(lambda cls, *f: cls.byEuler(127 * degree, 35 * degree, 27 * degree, *f))
  PLage = classmethod(lambda cls, *f: cls.byEuler(65 * degree, 45 * degree, 0, *f))
  PLage2 = classmethod(lambda cls, *f: cls.byEuler(245 * degree, 45 * degree, 0, *f))
  QLage = classmethod(lambda cls, *f: cls.byEuler(65 * degree, 20 * degree, 0, *f))
  QLage2 = classmethod(lambda cls, *f: cls.byEuler(245 * degree, 20 * degree, 0, *f))
  QLage3 = classmethod(lambda cls, *f: cls.byEuler(115 * degree, 160 * degree, 0, *f))
  QLage4 = classmethod(lambda cls, *f: cls.byEuler(295 * degree, 160 * degree, 0, *f))

  # ------------------------------------------------------------------ the frames

  @property
  def CS(self):
    return self.frameA

  @CS.setter
  def CS(self, fr):
    self.frameA = fr

  @property
  def SS(self):
    return self.frameB

  @SS.setter
  def SS(self, fr):
    self.frameB = fr

  @property
  def isMisorientation(self):
    return isinstance(self.frameA, crystalFrame) and isinstance(self.frameB, crystalFrame)

  # the frames of a list are joined by the ADR 0009 rule on both sides
  def _join(self, other):
    return {"frameA": SESSION.joinFrames(self.frameA, other.frameA), "frameB": SESSION.joinFrames(self.frameB, other.frameB),
            "antipodal": self.antipodal and other.antipodal, "notIID": self.notIID or getattr(other, "notIID", False)}

  def _coerce(self, other):
    if isinstance(other, orientation):
      return other
    if isinstance(other, (quaternion, orthogonalTransform)):
      return other
    raise TypeError(f"expected an orientation, got {type(other).__name__}")

  # the inverse maps the other way: the frames swap
  def inv(self):
    return self._new(self.conj().data, frameA=self.frameB, frameB=self.frameA)

  # a plain rotation of the coefficients, the frames kept; products between frames are the
  # business of products.py
  def __mul__(self, other):
    from .products import mtimes
    return mtimes(self, other)

  def __rmul__(self, other):
    from .products import mtimes
    return mtimes(other, self)

  def __neg__(self):
    raise TypeError("an orientation is proper; there is no improper orientation")

  # ------------------------------------------------------------------ symmetry-aware operations

  # the symmetric equivalents l * o * r, shaped (numL * numR,) + shape
  def symmetrise(self, unique=False):
    from .symmetryops import symmetriseOrientation
    return symmetriseOrientation(self, unique=unique)

  # the reduced dot product and the disorientation angle, over the groups of both frames
  def dot(self, other, noSymmetry=False, noAntipodal=False):
    from .symmetryops import orientationDot
    if noSymmetry or not isinstance(other, orientation):
      return rotation.dot(self, other, noAntipodal=noAntipodal)
    return orientationDot(self, other)

  def angle(self, other=None, noSymmetry=False):
    from .symmetryops import orientationAngle, orientationSelfAngle
    # angle(mori, 'max'): the angle of the equivalent that turns the most, MATLAB's option
    if isinstance(other, str):
      if other.lower() != "max":
        raise ValueError("the only representative angle takes is 'max'")
      return _largestImage(self).angle(noSymmetry=True)
    if other is None:
      return rotation.angle(self) if noSymmetry else orientationSelfAngle(self)
    if hasattr(other, "o1"):
      return other.angle(self, noSymmetry=noSymmetry)
    if noSymmetry or not isinstance(other, orientation):
      return rotation.angle(self, other)
    return orientationAngle(self, other)

  # the rotational axis, of the representative in the fundamental region; with a second
  # orientation the axis of their disorientation in the specimen frame
  def axis(self, other=None, noSymmetry=False, antipodal=None):
    from .symmetryops import orientationAxis, pairAxis
    # axis(mori, 'max'): the axis of the equivalent that turns the most, MATLAB's option
    if isinstance(other, str):
      if other.lower() != "max":
        raise ValueError("the only representative axis takes is 'max'")
      v = _largestImage(self).axis(noSymmetry=True)
      v.frame = self.frameA
      return v
    if other is not None:
      v = pairAxis(self, other)
      if antipodal is not None:
        v.antipodal = bool(antipodal)
      return v
    v = orientationAxis(self, noSymmetry=noSymmetry)
    if antipodal is not None:
      v.antipodal = bool(antipodal)
    return v

  # equal up to symmetry and a tolerance in the angle, between the same frames
  def __eq__(self, other, tolerance=5e-4):
    if not isinstance(other, orientation):
      return rotation.__eq__(self, other, tolerance)
    if self.frameA is not other.frameA or self.frameB is not other.frameB or self.antipodal != other.antipodal:
      return np.zeros(np.broadcast_shapes(self.shape, other.shape), dtype=bool)
    return self.dot(other) > np.cos(tolerance)

  def __ne__(self, other):
    return ~(self == other)

  def unique(self, tolerance=1e-3, noSymmetry=False, stable=False, return_index=False, return_inverse=False):
    from .symmetryops import orientationUnique
    return orientationUnique(self, tolerance, noSymmetry, return_index, return_inverse)

  def mean(self, weights=None, robust=False, iterations=3):
    from .symmetryops import orientationMean
    return orientationMean(self, weights=weights, robust=robust, iterations=iterations)

  def project2FundamentalRegion(self, ref=None, antipodal=False):
    from .symmetryops import project2FundamentalRegionOrientation
    # antipodal adds grain exchange to the reduction, MATLAB's 'antipodal'
    o = orientation._fromRotation(self.data, (self.CS, self.SS), True) if antipodal and not self.antipodal else self
    return project2FundamentalRegionOrientation(o, ref)

  # the tangent vector at the reference pointing to this orientation, the pair reduced by
  # symmetry first so that its norm is angle(ori, ref)
  def log(self, ref=None, side="left", noSymmetry=False):
    from .tangent import logOrientation
    return logOrientation(self, ref, side, noSymmetry)

  def fundamentalRegion(self):
    from .orientationregion import fundamentalRegion
    return fundamentalRegion(self)

  # the Bunge Euler angles of the representative inside the Euler plotting box
  def project2EulerFR(self):
    from .orientationregion import project2EulerFR
    return project2EulerFR(self)

  # the same orientation with another crystal frame, both frames of a misorientation
  def transformReferenceFrame(self, to, toB=None, rule=None, tolerance=1e-1):
    from .frametransform import transformOrientation
    if isinstance(toB, str) or hasattr(toB, "frameA"):
      rule, toB = toB, None
    return transformOrientation(self, to, toB, rule, tolerance)

  def find(self, w, epsOrK=None):
    from .grids.search import find
    return find(self, w, epsOrK)

  def variants(self, second=None, ids=None, parent=False):
    from .standardorientations import variants
    return variants(self, second, ids, parent)

  # the distinct relationships c p2c over the child group c: a child times them is every
  # parent it can come from, MATLAB's parents
  def parents(self):
    from .standardorientations import variants
    return variants(self, parent=True)

  # ------------------------------------------------------------------ orientations from frames and directions

  @classmethod
  def byMiller(cls, hkl, uvw, *frames):
    from .standardorientations import byMiller
    return byMiller(hkl, uvw, *frames)

  @classmethod
  def align(cls, frA, frB, tolerance=1e-1):
    from .frametransform import align
    return align(frA, frB, tolerance)

  @classmethod
  def byScreenAlignment(cls, frA, frB):
    from .frametransform import byScreenAlignment
    return byScreenAlignment(frA, frB)

  # the named components of rolling textures and the orientation relationships, orientation.brass(cs)
  for _name in ("copper", "copper2", "cube", "cubeND22", "cubeND45", "cubeRD", "goss", "SR", "SR2", "SR3", "SR4", "brass", "brass2",
                "PLage", "PLage2", "QLage", "QLage2", "QLage3", "QLage4", "invGoss",
                "Bain", "KurdjumovSachs", "NishiyamaWassermann", "Pitsch", "GreningerTrojano", "Bagaryatsky", "Burgers",
                "PitschSchrader", "Potter", "ShojiNishiyama"):
    locals()[_name] = classmethod((lambda n: lambda cls, *f: getattr(__import__("mtex.geometry.standardorientations", fromlist=[n]), n)(*f))(_name))
  del _name

  # ------------------------------------------------------------------ display

  def _frameSummary(self):
    return f"{self.frameA.char()} → {self.frameB.char()}"

  # the parallel planes and directions a single misorientation is made of, when low indices
  # (up to 5) give it exactly, MATLAB's char of a misorientation; None otherwise
  def _relation(self):
    if not self.isMisorientation or self.size != 1:
      return None
    from .parent2child import round2Miller
    n1, n2, d1, d2 = round2Miller(self.reshape(()), maxIndex=5)
    exact = orientation.map(n1, n2, d1, d2)
    if exact.angle(self.reshape(()), noSymmetry=False) >= 1e-3 * degree:
      return None
    return f"{n1.char()} || {n2.char()}   {d1.char()} || {d2.char()}"

  def char(self):
    rel = self._relation()
    return rel if rel is not None else self._eulerLine() if self.size == 1 else self._eulerTable()

  def __repr__(self):
    lines = [("misorientation" if self.isMisorientation else "orientation") + f" ({self._frameSummary()})"]
    if self.size != 1:
      lines.append(f"  size: {sizeString(self.shape)}")
    if self.antipodal:
      lines.append("  antipodal: true")
    rel = self._relation()
    if rel is not None:
      lines.append(f" {rel}")
    elif 0 < self.size < 20:
      lines.append(self._eulerTable())
    return "\n".join(lines)


# grain exchange needs the same frame handle on both sides (Julia ADR 0008)
def _checkExchange(antipodal, frameA, frameB):
  if antipodal and frameA is not frameB:
    raise ValueError(f"grain exchange symmetry needs a misorientation within one frame, not between {frameA.char()} and {frameB.char()}")


def _xyz(v):
  return v.data if isinstance(v, vector3d) else np.asarray(v, dtype=float)


def _frameOf(v):
  return v.frame if isinstance(v, vector3d) else None


__all__ = ["orientation"]
