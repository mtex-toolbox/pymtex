# three dimensional vectors: a point cloud in one object, with polar coordinates, axes and
# the comparisons that read an axis up to sign
#
#   v = vector3d(x, y, z)
#   v = vector3d(x, y, z, antipodal=True)   an axis, v == -v
#   v = vector3d(xyz)                        an (..., 3) array, one vector per row
#   v = vector3d.byPolar(theta, rho)
#   vector3d.X, vector3d.rand(n)
#
# antipodal and isNormalized are the MATLAB fields with the Julia semantics behind them
# (docs/plans/geometry.md 3): a comparison with one antipodal operand is antipodal,
# arithmetic clears both flags, rotation keeps them. The frame and the index basis
# join in G4.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar
import numpy as np

from ._array import MtexArray
from .display import display, header, sizeString, table, takeRows
from .representations import isnull, orthKernel


_T = TypeVar("_T")


class classproperty(Generic[_T]):
  """a property of the class itself, vector3d.X"""
  def __init__(self, f: Callable[[Any], _T]):
    self.f = f

  def __get__(self, obj, cls) -> _T:
    return self.f(cls)


def _scalar(x):
  return np.asarray(x, dtype=float)[..., None]


class vector3d(MtexArray):
  """three dimensional vectors, a container over (..., 3) coordinates carrying antipodal,
  isNormalized, the reference frame and the display style; a vector in a crystal frame is a
  crystal direction with Miller indices

    vector3d(x, y, z)
    vector3d(xyz, frame=cs)             coordinates of shape (..., 3) or (3, N)
    vector3d.byPolar(theta, rho)
    vector3d.X, vector3d.Y, vector3d.Z"""
  _ncoord = 3
  _carried = ("antipodal", "isNormalized", "_frame", "dispStyle")

  def __init__(self, *args, antipodal=False, isNormalized=False, frame=None, dispStyle=None, dtype=None):
    # a reference frame or a plotting convention may be given positionally, as in MATLAB
    args = list(args)
    for k, a in enumerate(args):
      if _isFrame(a) or _isConvention(a):
        frame = args.pop(k)
        break
    if len(args) == 3:
      data = np.stack(np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in args)), axis=-1)
    elif len(args) == 1:
      a = args[0]
      if isinstance(a, vector3d):
        data = a.data.copy()
        antipodal = antipodal or a.antipodal
        isNormalized = isNormalized or a.isNormalized
        frame = a._frame if frame is None else frame
        dispStyle = a.dispStyle if dispStyle is None else dispStyle
      # vector3d(S): the axial vector of a spin tensor, in its frame
      elif hasattr(a, "M") and getattr(a, "rank", None) == 2:
        M = np.asarray(a.M, dtype=float)
        A = 0.5 * (M - np.swapaxes(M, -1, -2))
        data = np.stack([A[..., 2, 1], A[..., 0, 2], A[..., 1, 0]], axis=-1)
        frame = a.frame if frame is None else frame
      # vector3d(T): the components of a rank one tensor, in its frame
      elif hasattr(a, "M") and getattr(a, "rank", None) == 1:
        data = np.asarray(a.M, dtype=float).copy()
        frame = a.frame if frame is None else frame
      else:
        a = np.asarray(a, dtype=float)
        if a.ndim >= 1 and a.shape[-1] == 3:
          data = a
        elif a.ndim == 2 and a.shape[0] == 3:
          data = a.T
        else:
          raise ValueError(f"vector3d needs coordinates of shape (..., 3) or (3, N), got {a.shape}")
    elif len(args) == 0:
      data = np.zeros((0, 3))
    else:
      raise TypeError("vector3d takes x, y, z or one array of coordinates")
    super().__init__(data, dtype)
    self.antipodal = bool(antipodal)
    self.isNormalized = bool(isNormalized)
    self._frame = None
    self.dispStyle = "xyz" if dispStyle is None else dispStyle
    self.frame = frame

  # ------------------------------------------------------------------ the frame

  @property
  def frame(self):
    """the reference frame this vector is expressed in, None for frame-free data; assigning a
    frame keeps the components, assigning CS keeps the indices"""
    return self._frame

  @frame.setter
  def frame(self, fr):
    """the reference frame the vector is expressed in, None for frame-free data; assigning a frame
    keeps the components"""
    if _isConvention(fr):
      from .frames import specimenFrame
      fr = specimenFrame.frameFor(fr)
    if fr is not None and not _isFrame(fr):
      raise TypeError("the frame of a vector3d has to be a referenceFrame or None")
    self._frame = fr
    if not self.isCrystalDirection:
      self.dispStyle = "xyz"
    elif self.dispStyle == "xyz":
      self.dispStyle = "hkil" if fr.lattice.isTriHex else "hkl"

  @property
  def CS(self):
    """the crystal frame of a crystal direction, None for any other vector"""
    return self._frame if self.isCrystalDirection else None

  @CS.setter
  def CS(self, cs):
    """assigning a crystal frame to a crystal direction keeps its indices, otherwise its components"""
    if self.isCrystalDirection and self._frame is not cs and self.dispStyle != "xyz":
      style, coord = self.dispStyle, self.coordinates
      self._frame = cs
      self.coordinates = coord
      self.dispStyle = style
    else:
      self.frame = cs

  @property
  def how2plot(self):
    """the plotting convention of the vector, that of its frame or the default"""
    from .plottingconvention import plottingConvention
    return self._frame.how2plot if self._frame is not None else plottingConvention.default

  @property
  def isCrystalDirection(self):
    """whether the vector is given in a crystal frame"""
    return _isCrystalFrame(self._frame)

  @property
  def lattice(self):
    """the lattice type of the crystal frame, latticeType.none for a vector in no crystal frame"""
    from .latticetype import latticeType
    return self._frame.lattice if self.isCrystalDirection else latticeType.none

  # ------------------------------------------------------------------ indices

  @property
  def convention(self):
    """the convention the vector is written in, its display style: 'xyz', 'hkl', 'hkil', 'uvw' or 'UVTW'"""
    return self.dispStyle

  @convention.setter
  def convention(self, s):
    """assigning the convention sets the display style"""
    self.dispStyle = s

  @property
  def coordinates(self):
    """the indices in the convention the vector is written in"""
    return self.data if self.dispStyle == "xyz" else getattr(self, self.dispStyle)

  @coordinates.setter
  def coordinates(self, c):
    """the indices in the convention the vector is written in, the components for 'xyz'"""
    if self.dispStyle == "xyz":
      self.data[...] = c
    else:
      setattr(self, self.dispStyle, c)

  def _needsCrystal(self):
    if not self.isCrystalDirection:
      raise ValueError("Miller indices need a direction given in a crystal frame")

  @property
  def hkl(self):
    """the Miller indices of a plane normal, the coordinates in the dual basis of the crystal frame"""
    self._needsCrystal()
    return self.data @ np.linalg.inv(self._frame.basisDual().data)

  @hkl.setter
  def hkl(self, hkl):
    """assigning Miller indices, three or four, sets the vector and writes it as a plane normal"""
    self._needsCrystal()
    hkl = np.asarray(hkl, dtype=float)
    hkl = hkl[..., [0, 1, -1]]
    self.data = hkl @ self._frame.basisDual().data
    self.dispStyle = "hkil" if self.lattice.isTriHex else "hkl"
    self.isNormalized = False

  @property
  def hkil(self):
    """the four Miller-Bravais indices of a plane normal, i = -(h + k)"""
    hkl = self.hkl
    return np.concatenate([hkl[..., :2], -hkl[..., :1] - hkl[..., 1:2], hkl[..., 2:]], axis=-1)

  @hkil.setter
  def hkil(self, hkil):
    """assigning four Miller-Bravais indices sets the vector, as assigning hkl"""
    self.hkl = hkil

  @property
  def uvw(self):
    """the indices of a lattice direction, the coordinates in the basis of the crystal frame"""
    self._needsCrystal()
    return self.data @ np.linalg.inv(self._frame.basis.data)

  @uvw.setter
  def uvw(self, uvw):
    """assigning three direction indices sets the vector and writes it as a direction; four are UVTW"""
    self._needsCrystal()
    uvw = np.asarray(uvw, dtype=float)
    if uvw.shape[-1] == 4:
      raise ValueError("use UVTW to set four Miller indices")
    self.data = uvw @ self._frame.basis.data
    self.dispStyle = "uvw"
    self.isNormalized = False

  @property
  def UVTW(self):
    """the Weber indices: U = 2u - v, V = 2v - u, T = -(u + v), W = 3w"""
    u, v, w = np.moveaxis(self.uvw, -1, 0)
    return np.stack([2 * u - v, 2 * v - u, -u - v, 3 * w], axis=-1)

  @UVTW.setter
  def UVTW(self, UVTW):
    """assigning Weber indices, four or three, sets the vector and writes it in UVTW"""
    UVTW = np.asarray(UVTW, dtype=float)
    if UVTW.shape[-1] == 4:
      self.uvw = np.stack([UVTW[..., 0] - UVTW[..., 2], UVTW[..., 1] - UVTW[..., 2], UVTW[..., 3]], axis=-1) / 3
    else:
      self.uvw = np.stack([2 * UVTW[..., 0] + UVTW[..., 1], 2 * UVTW[..., 1] + UVTW[..., 0], UVTW[..., 2]], axis=-1) / 3
    self.dispStyle = "UVTW"

  h = property(lambda self: self.hkl[..., 0])
  k = property(lambda self: self.hkl[..., 1])
  i = property(lambda self: self.hkil[..., 2])
  l = property(lambda self: self.hkl[..., 2])
  u = property(lambda self: self.uvw[..., 0])
  v = property(lambda self: self.uvw[..., 1])
  w = property(lambda self: self.uvw[..., 2])
  U = property(lambda self: self.UVTW[..., 0])
  V = property(lambda self: self.UVTW[..., 1])
  T = property(lambda self: self.UVTW[..., 2])
  W = property(lambda self: self.UVTW[..., 3])

  def dspacing(self):
    """the distance between the lattice planes with this normal, in the units of the cell"""
    self._needsCrystal()
    return 1 / self.norm()

  def round(self, style=None, maxHKL=12):
    """a crystal direction rounded to integer indices, vector3d/round"""
    from .miller import roundIndices
    return roundIndices(self, style, maxHKL)

  def char(self, style=None, utf8=None, family=None):
    """the indices in brackets, {111} or [uvw], Miller/char"""
    from .miller import indexChar
    return indexChar(self, style, utf8, family)

  def __str__(self):
    if self.size > 4:
      return f"{self.size} points"
    if self.isCrystalDirection:
      return self.char()
    d = np.round(self.data.reshape(-1, 3), 10)
    return " ".join("(" + ",".join(f"{x:g}" for x in row) + ")" for row in d)

  # ------------------------------------------------------------------ constructors

  @classmethod
  def load(cls, fname, *args, **kwargs):
    """vector3d.load('vectors.txt', columnNames=[...]): the directions a text table holds, read
    by the interfaces layer above this one; loadVector3d returns the other columns too"""
    from ..io.vectors import loadVector3d
    return loadVector3d(fname, *args, **kwargs)[0]

  def calcDensity(self, *args, **kwargs):
    """v.calcDensity(): the kernel density estimate of a list of directions, by the function
    spaces above this layer, vector3d/calcDensity"""
    from ..functions.density import calcDensity
    return calcDensity(self, *args, **kwargs)

  def interp(self, y, *args, **kwargs):
    """interp(v, y, 'linear' | 'harmonic' | 'nearest' | 'inverseDistance'): a function on the
    sphere through the values y at the directions v, by the function spaces above this
    layer, vector3d/interp; interp(v, y, vi, ...) its values at vi"""
    from ..functions.density import interpS2
    return interpS2(self, y, *args, **kwargs)

  def export(self, fname, *args, **kwargs):
    """v.export('vectors.txt'): the directions written as a text table"""
    from ..io.vectors import exportVector3d
    return exportVector3d(self, fname, *args, **kwargs)

  @classmethod
  def byPolar(cls, theta, rho, antipodal=False):
    """the vectors of polar angle theta to z and azimuth rho from x towards y, unit vectors

      vector3d.byPolar(theta, rho)
      vector3d.byPolar(theta, rho, antipodal=True)      axes"""
    theta, rho = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(rho, dtype=float))
    v = cls(np.sin(theta) * np.cos(rho), np.sin(theta) * np.sin(rho), np.cos(theta), antipodal=antipodal)
    v.isNormalized = True
    return v

  @classmethod
  def byXYZ(cls, xyz):
    """an N x 3 or N x 2 matrix, z zero in the second case"""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.size == 0 and xyz.ndim < 2:
      xyz = np.zeros((0, 3))
    if xyz.ndim != 2 or xyz.shape[1] not in (2, 3):
      raise ValueError(f"byXYZ needs an N x 3 or N x 2 matrix, got {xyz.shape}")
    if xyz.shape[1] == 2:
      xyz = np.concatenate([xyz, np.zeros((xyz.shape[0], 1))], axis=-1)
    return cls(xyz)

  @classmethod
  def rand(cls, *shape, antipodal=False, rng=None):
    """uniformly distributed unit vectors or axes"""
    rng = np.random.default_rng() if rng is None else rng
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    theta = np.arccos(2 * rng.random(shape) - 1)
    rho = 2 * np.pi * rng.random(shape)
    return cls.byPolar(theta, rho, antipodal=antipodal)

  @classmethod
  def zeros(cls, *shape):
    """the zero vectors of a shape"""
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return cls(np.zeros(tuple(shape) + (3,)))

  @classmethod
  def nan(cls, *shape):
    """the vectors of a shape with NaN coordinates"""
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return cls(np.full(tuple(shape) + (3,), np.nan))

  @classproperty
  def X(cls):
    """the unit vector along x, minted anew on every access"""
    return cls(1.0, 0.0, 0.0, isNormalized=True)

  @classproperty
  def Y(cls):
    """the unit vector along y, minted anew on every access"""
    return cls(0.0, 1.0, 0.0, isNormalized=True)

  @classproperty
  def Z(cls):
    """the unit vector along z, minted anew on every access"""
    return cls(0.0, 0.0, 1.0, isNormalized=True)

  # ------------------------------------------------------------------ coordinates

  @property
  def x(self):
    """the x coordinates"""
    return self.data[..., 0]

  @x.setter
  def x(self, value):
    """assigning the x coordinates"""
    self.data[..., 0] = value
    self.isNormalized = False

  @property
  def y(self):
    """the y coordinates"""
    return self.data[..., 1]

  @y.setter
  def y(self, value):
    """assigning the y coordinates"""
    self.data[..., 1] = value
    self.isNormalized = False

  @property
  def z(self):
    """the z coordinates"""
    return self.data[..., 2]

  @z.setter
  def z(self, value):
    """assigning the z coordinates"""
    self.data[..., 2] = value
    self.isNormalized = False

  @property
  def xyz(self):
    """the coordinates, an array of shape (..., 3)"""
    return self.data

  def polar(self):
    """polar angle to z in [0, pi] and azimuth from x towards y in [0, 2pi), vector3d/polar"""
    r = np.ones(self.shape) if self.isNormalized else self.norm()
    rho = np.arctan2(self.y, self.x)
    rho = rho + (rho < 0) * 2 * np.pi
    theta = np.arccos(np.clip(self.z / np.where(r == 0, 1.0, r), -1, 1))
    return theta, rho, r

  @property
  def theta(self):
    """the polar angle to z, in [0, pi]"""
    return self.polar()[0]

  @property
  def rho(self):
    """the azimuth from x towards y, in [0, 2 pi)"""
    return self.polar()[1]

  @property
  def resolution(self):
    """the mean distance between the points (MATLAB vector3d.resolution): the whole circle
    for up to four points, the square root of the median Voronoi cell area up to fifty
    thousand, the equal area estimate beyond; a grid knows its own"""
    n = self.size
    if n <= 4:
      return 2 * np.pi
    if n > 50000:
      return float(np.sqrt(40000 / n / (1 + self.antipodal)) * np.pi / 180)
    from .hull import _cellAreas
    xyz = self.flatten().normalize().data
    if self.antipodal:
      xyz = np.concatenate([xyz, -xyz])
    u = np.unique(np.round(xyz / 1e-6) * 1e-6, axis=0)
    try:
      area = _cellAreas(u)
    except Exception:
      return 2 * np.pi
    return float(np.sqrt(np.median(area)))

  # a list holds normalised vectors only when every part does; axes and vectors do not mix;
  # frames join by the ADR 0009 rule and plane normals do not mix with lattice directions
  def _join(self, other):
    if self.antipodal != other.antipodal:
      raise ValueError("cannot combine axes with vectors")
    return {"antipodal": self.antipodal, "isNormalized": self.isNormalized and other.isNormalized,
            "_frame": _joinFrames(self._frame, other._frame), "dispStyle": _joinStyle(self.dispStyle, other.dispStyle)}

  # the attributes of the result of a binary operation: the joined frame, no flags
  def _joined(self, other, **flags):
    attrs = {"antipodal": False, "isNormalized": False, "_frame": self._frame, "dispStyle": self.dispStyle}
    if other is not None:
      attrs["_frame"] = _joinFrames(self._frame, other._frame)
      attrs["dispStyle"] = _joinStyle(self.dispStyle, other.dispStyle)
    attrs.update(flags)
    return attrs

  # ------------------------------------------------------------------ arithmetic

  # any vector3d, a grid included, is an operand
  def _coerce(self, other):
    if isinstance(other, vector3d):
      return other
    raise TypeError(f"expected a vector3d, got {type(other).__name__}")

  # a numeric operand scales every coordinate; the flags do not survive arithmetic; an
  # object with its own rotate, a crystal shape, takes the operation itself
  def _other(self, other):
    if isinstance(other, vector3d):
      return other.data, other
    if isinstance(other, MtexArray) or hasattr(other, "rotate"):
      return NotImplemented, None
    return _scalar(other), None

  def __add__(self, other):
    d, o = self._other(other)
    if d is NotImplemented:
      return NotImplemented
    return self._new(self.data + d, **self._joined(o))

  __radd__ = __add__

  def __sub__(self, other):
    d, o = self._other(other)
    if d is NotImplemented:
      return NotImplemented
    return self._new(self.data - d, **self._joined(o))

  def __rsub__(self, other):
    return self._new(_scalar(other) - self.data, **self._joined(None))

  def __neg__(self):
    return self._new(-self.data)

  def __pos__(self):
    return self

  # v * s scales, v1 * v2 is componentwise; a rotation on the left acts through its own class
  def __mul__(self, other):
    d, o = self._other(other)
    if d is NotImplemented:
      return NotImplemented
    return self._new(self.data * d, **self._joined(o, antipodal=self.antipodal or (o is not None and o.antipodal)))

  def __rmul__(self, other):
    if isinstance(other, MtexArray):
      return NotImplemented
    return self._new(self.data * _scalar(other), isNormalized=False)

  def __truediv__(self, other):
    if isinstance(other, MtexArray):
      return NotImplemented
    return self._new(self.data / _scalar(other), isNormalized=False)

  # ------------------------------------------------------------------ products and angles

  def dot(self, other, antipodal=None, symmetry=True, min=False, all=False, outer=False):
    """the inner product; an axis on either side reads it up to sign, antipodal=True always,
    antipodal=False never; a crystal direction reduces over the group of its frame unless
    symmetry=False says otherwise; outer=True pairs
    every entry of self with every entry of other, shape self.shape + other.shape (MATLAB's dot_outer)"""
    other = self._coerce(other)
    if outer:
      a, b = self._outerPair(other)
      return a.dot(b, antipodal=antipodal, symmetry=symmetry, min=min, all=all)
    _joinFrames(self._frame, other._frame)
    if symmetry:
      from .symmetryops import symDot
      d = symDot(self, other, min=min, all=all, antipodal=antipodal)
      if d is not None:
        return d
    d = np.sum(self.data * other.data, axis=-1)
    if (self.antipodal or other.antipodal) if antipodal is None else antipodal:
      d = np.abs(d)
    return d

  def cross(self, other):
    """the cross product of two plane normals is a lattice direction and the one of two
    directions a plane normal, so the indices switch sides"""
    other = self._coerce(other)
    attrs = self._joined(other, antipodal=self.antipodal or other.antipodal)
    attrs["dispStyle"] = _otherStyle(attrs["dispStyle"])
    return self._new(np.cross(self.data, other.data), **attrs)

  def norm(self):
    """the length of the vectors"""
    return np.linalg.norm(self.data, axis=-1)

  def angle(self, other, N=None, antipodal=None, symmetry=True, outer=False):
    """the angle between two vectors, through atan2 for full accuracy near 0 and pi; an axis
    on either side reads it up to sign, antipodal=True always, antipodal=False never; with a
    normal N the angle in that plane, counterclockwise from self to other; outer=True pairs
    every entry of self with every entry of other, shape self.shape + other.shape (MATLAB's angle_outer)"""
    other = self._coerce(other)
    if outer:
      a, b = self._outerPair(other)
      return a.angle(b, N=N, antipodal=antipodal, symmetry=symmetry)
    _joinFrames(self._frame, other._frame)
    if symmetry and N is None:
      from .symmetryops import symAngle
      a = symAngle(self, other, antipodal=antipodal)
      if a is not None:
        return a
    isAntipodal = (self.antipodal or other.antipodal) if antipodal is None else antipodal
    if N is not None:
      n = N.normalize()
      w1 = (self - n * self.dot(n, antipodal=False)).normalize()
      w2 = (other - n * other.dot(n, antipodal=False)).normalize()
      a = w1.angle(w2, symmetry=False)
      a = np.where(n.dot(w1.cross(w2), antipodal=False) < 0, 2 * np.pi - a, a)
      return np.mod(a, np.pi) if isAntipodal else a
    d = np.sum(self.data * other.data, axis=-1)
    c = np.linalg.norm(np.cross(self.data, other.data), axis=-1)
    return np.arctan2(c, np.abs(d) if isAntipodal else d)

  # self and other broadcast against each other: self's axes first, then other's
  def _outerPair(self, other):
    return self.reshape(self.shape + (1,) * other.ndim), other.reshape((1,) * self.ndim + other.shape)

  def orthProj(self, N):
    """the component orthogonal to a normal N, vector3d/orthProj"""
    n = self._coerce(N).normalize()
    return self - n * self.dot(n, antipodal=False, symmetry=False)

  # two vectors are equal when they point the same way, up to the group of a crystal
  # frame, vector3d/eq
  def __eq__(self, other):
    if not isinstance(other, vector3d):
      return NotImplemented
    if (self._frame is not other._frame) and self.isCrystalDirection and other.isCrystalDirection:
      from .frames import _sameBasis
      if not _sameBasis(self._frame, other._frame):
        return np.zeros(np.broadcast_shapes(self.shape, other.shape), dtype=bool)
    d = self.dot(other)
    return isnull(d - self.norm() * other.norm())

  def __ne__(self, other):
    return ~(self == other)

  def isnan(self):
    """whether a vector has a NaN coordinate"""
    return np.any(np.isnan(self.data), axis=-1)

  # ------------------------------------------------------------------ derived vectors

  def normalize(self, zeroToZero=False):
    """the vectors scaled to unit length; with zeroToZero=True a zero vector stays zero"""
    if self.isNormalized:
      return self
    n = self.norm()
    if zeroToZero:
      n = np.where(n > np.finfo(float).eps, n, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
      return self._new(self.data / n[..., None], isNormalized=True)

  def orth(self):
    """an arbitrary orthogonal direction, vector3d/orth"""
    return self._new(orthKernel(self.data), isNormalized=True)

  def perp(self, robust=False):
    """the axis best orthogonal to a list of directions, the smallest principal axis"""
    v = self[~self.isnan()] if np.any(self.isnan()) else self
    xyz = v.data.reshape(-1, 3)
    w, V = np.linalg.eigh(xyz.T @ xyz)
    N = vector3d(V[:, 0], antipodal=True, isNormalized=True, frame=self._frame, dispStyle=_otherStyle(self.dispStyle))
    if robust and v.size > 4:
      delta = np.pi / 2 - N.angle(v)
      keep = delta < np.quantile(delta, 0.8) * (1 + 1e-5)
      if np.any(keep):
        N = v[keep].perp()
    return N

  def eig(self):
    """the principal axes of the moment matrix and its eigenvalues, vector3d/eig"""
    xyz = self.data.reshape(-1, 3)
    w, V = np.linalg.eigh(xyz.T @ xyz)
    return w, vector3d(V.T, isNormalized=True, frame=self._frame)

  def sum(self, axis=None):
    """the sum of the vectors, over all axes or the given ones"""
    axis = tuple(range(self.ndim)) if axis is None else axis
    return self._new(np.sum(self.data, axis=axis), **self._joined(None))

  def mean(self, weights=None, antipodal=False, robust=False, axis=None, symmetry=True):
    """the mean vector: the arithmetic mean, the normalised weighted sum, or for axes the
    largest principal axis of the moment matrix"""
    if symmetry and self.isCrystalDirection and self._frame.numSym() > 1 and self.size > 1:
      from .symmetryops import symMean
      return symMean(self, weights=weights)
    if robust and self.size > 4:
      m = self.mean(weights=weights, antipodal=antipodal)
      omega = m.angle(self)
      keep = omega < np.quantile(omega, 0.8) * (1 + 1e-5)
      return self[keep].mean(weights=None if weights is None else np.asarray(weights)[keep], antipodal=antipodal)
    axis = tuple(range(self.ndim)) if axis is None else axis
    if antipodal or self.antipodal:
      w = np.ones(self.shape) if weights is None else np.asarray(weights, dtype=float)
      M = np.mean(w[..., None, None] * self.data[..., :, None] * self.data[..., None, :], axis=axis)
      vals, vecs = np.linalg.eigh(M)
      return self._new(np.take(vecs, -1, axis=-1), **self._joined(None, antipodal=True, isNormalized=True))
    if weights is not None:
      w = np.asarray(weights, dtype=float)
      return self._new(np.sum(self.data * w[..., None], axis=axis), **self._joined(None)).normalize()
    return self._new(np.nanmean(self.data, axis=axis), **self._joined(None))

  def unique(self, tol=1e-6, antipodal=None, symmetry=True, stable=False, return_index=False, return_inverse=False):
    """the disjoint list of vectors up to a tolerance, vector3d/unique, up to sign for axes,
    antipodal=True always, antipodal=False never; under a group the directions are projected
    to the fundamental sector first"""
    if symmetry and self.isCrystalDirection and self._frame.numSym() > 1:
      from .symmetryops import symUnique
      return symUnique(self, tol, return_index, return_inverse)
    xyz = self.data.reshape(-1, 3)
    if self.antipodal if antipodal is None else antipodal:
      x, y, z = xyz.T
      key = np.stack([x * x, y * y, z * z, x * y, x * z, y * z], axis=-1)
    else:
      key = xyz
    iv, iu = _uniqueRows(key, tol, stable)
    u = self._new(xyz[iv])
    out = (u,) + ((iv,) if return_index else ()) + ((iu,) if return_inverse else ())
    return out if len(out) > 1 else u

  def isclose(self, other, antipodal=False, tol=None):
    """isclose(v, w), isclose(v, w, antipodal=True, tol=): equal up to symmetry within the
    tolerance, and up to sign when antipodal (MATLAB's eq)"""
    if antipodal:
      a = self._new(self.data, antipodal=True)
      return a.__eq__(other) if tol is None else a.__eq__(other, tol)
    return self.__eq__(other) if tol is None else self.__eq__(other, tol)

  def rotate(self, rot):
    """the vectors turned by a rotation, rot * v; a number is an angle about z"""
    if not isinstance(rot, MtexArray):
      from .quaternion import rotation
      rot = rotation.byAxisAngle(vector3d.Z, rot)
    return rot * self

  # the vector turned by the quaternion coefficients q; a tangent vector moves its reference too
  def _rotated(self, q):
    from .kernels import qrot
    return self._new(qrot(q, self.data))

  def exp(self, ref=None, side=None):
    """the rotation vector to its rotation: at the identity, or at a reference as a tangent vector"""
    from .tangent import expVector
    return expVector(self, ref, side)

  def symmetrise(self, unique=False, antipodal=None):
    """the symmetrically equivalent directions, the number of distinct ones, and the
    representative in the fundamental sector of the frame"""
    from .symmetryops import symmetrise
    return symmetrise(self, unique=unique, antipodal=antipodal)

  def multiplicity(self, antipodal=None):
    """the number of distinct directions symmetrically equivalent to each vector, of axes under
    antipodal"""
    from .symmetryops import multiplicity
    return multiplicity(self, antipodal=antipodal)

  def projectIntoFundamentalRegion(self, center=None, antipodal=False):
    """the directions folded into the fundamental sector of their frame, with antipodal=True into
    that of the Laue group; given a centre, the equivalent nearest it instead
    (MATLAB's vector3d.project2FundamentalRegion, Miller.project2FundamentalRegion)

      v.projectIntoFundamentalRegion()
      v.projectIntoFundamentalRegion(center)"""
    from .symmetryops import projectIntoFundamentalRegion
    v = self
    if antipodal and self.isCrystalDirection:
      v = self._new(self.data, _frame=self._frame.Laue())
    return projectIntoFundamentalRegion(v, center)

  def find(self, w, epsOrK=None, antipodal=False):
    """the closest points of this list to others, the k closest, or all within a radius"""
    from .grids.search import find
    return find(self, w, epsOrK, antipodal)

  def transformReferenceFrame(self, to, rule=None, tol=1e-1):
    """the same direction written in another frame"""
    from .frametransform import transformVector
    return transformVector(self, to, rule, tol)

  def det(self, b=None, c=None):
    """the triple product, vector3d/det
    det(a, b, c) the triple product; det(v) of three vectors the determinant of their matrix"""
    if b is None:
      return np.linalg.det(self.data.reshape(3, 3))
    return self.dot(b.cross(c), antipodal=False)

  # ------------------------------------------------------------------ display

  def __repr__(self):
    body = None
    if self.size:
      take, pos = takeRows(self.size)
      shown = self.reshape(-1)[take]
      if self.isCrystalDirection and self.dispStyle != "xyz":
        c = np.round(shown.coordinates.reshape(len(take), -1), 4)
        body = table(list(self.dispStyle), c.T, decimals=4, tiny=1e-8, rows=pos)
      else:
        body = table(["x", "y", "z"], shown.data.reshape(-1, 3).T, rows=pos)
    return display(header(type(self).__name__, self._headerChar()),
                   [("size", sizeString(self.shape) if self.size != 1 else None), ("antipodal", "true" if self.antipodal else None)], body)

  # a crystal direction is headed by its frame, any other by the convention it is drawn in
  def _headerChar(self):
    if self.isCrystalDirection:
      return self._frame.char()
    from .frames import specimenFrame
    fr = self._frame if self._frame is not None else specimenFrame.default
    return fr.conventionChar(self.how2plot)

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..functions.polefigure import calcQuadratureWeights, refine
    from ..functions.discrepancy import discrepancy
    from ..analysis.cluster import calcCluster
    from ..plotting.sphericalplot import plot, scatter, text, contourf, contour, pcolor, quiver, pcolor as smooth
    from ..plotting.spaceplot import plot3d, scatter3d, arrow3d
    from ..plotting.grainplots import histogram


# rows equal up to a tolerance: the first index of each and the inverse map
def _uniqueRows(key, tol, stable):
  rounded = np.round(key / tol)
  _, iv, iu = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
  iu = iu.reshape(-1)
  if stable:
    order = np.argsort(iv)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    iv, iu = iv[order], rank[iu]
  return iv, iu


# a reference frame and a plotting convention, recognised without importing their files,
# which import this one
def _isFrame(x):
  return hasattr(x, "sym") and hasattr(x, "how2plot") and hasattr(x, "basis")


def _isCrystalFrame(x):
  return _isFrame(x) and hasattr(x, "basisDual")


def _isConvention(x):
  return type(x).__name__ == "plottingConvention"


def _joinFrames(a, b):
  from .session import SESSION
  return SESSION.joinFrames(a, b)


# the index basis two crystal directions share: the first, since the vectors are the same
def _joinStyle(a, b):
  if a == b or b == "xyz":
    return a
  if a == "xyz":
    return b
  # plane normals and lattice directions are the same vectors, shown in the style of the first
  return a


def _isReciprocal(style):
  return style in ("hkl", "hkil")


def _otherStyle(style):
  return {"hkl": "uvw", "hkil": "UVTW", "uvw": "hkl", "UVTW": "hkil", "xyz": "xyz"}[style]


# xvector, yvector and zvector are fresh on every access, as MATLAB's functions are, so
# that giving one a frame or a flag cannot reach the next use
_CONSTANTS = {"xvector": vector3d.X.data, "yvector": vector3d.Y.data, "zvector": vector3d.Z.data}


def __getattr__(name):
  if name in _CONSTANTS:
    return vector3d(_CONSTANTS[name].copy(), isNormalized=True)
  raise AttributeError(name)


# what an editor reads for them
if TYPE_CHECKING:
  xvector: vector3d
  yvector: vector3d
  zvector: vector3d



__all__ = ["vector3d", "isCrystalDirection"]
__lazy__ = ["xvector", "yvector", "zvector"]


def isCrystalDirection(v):
  """whether v is a vector given in a crystal frame"""
  return isinstance(v, vector3d) and v.isCrystalDirection
