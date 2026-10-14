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

from ._array import MtexArray, keptDims, numpyReduction
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

    v = vector3d(x, y, z)                       # from the Cartesian coordinates
    v = vector3d(xyz)                           # coordinates of shape (..., 3) or (3, N)
    v = vector3d(x, y, z, antipodal=True)       # axes, v and -v the same
    m = vector3d(xyz, frame=cs)                 # a crystal direction from its Euclidean coordinates
    v = vector3d.byPolar(theta, rho)            # from the polar angles
    v = vector3d.X                              # the axes of the frame, also Y and Z
    m = Miller(1, 1, 1, cs)                     # a lattice plane normal by its indices hkl
    m = Miller(1, 0, -1, 0, csHex)              # four indices hkil of a hexagonal lattice
    m = Miller(1, 1, 0, cs, 'uvw')              # a lattice direction, 'UVTW' with four indices
    m = Miller('{111}', cs)                     # (hkl) one plane, {hkl} the family, [uvw] and <uvw> directions
    m = Miller(v, cs)                           # a vector restated in the crystal frame

  A crystal direction is a vector3d whose frame is a crystal frame: it has the Miller
  indices m.hkl and m.uvw, the four index forms m.hkil and m.UVTW of a trigonal or hexagonal
  lattice, a d-spacing, and where the frame carries a point group it stands for its
  symmetrically equivalent set, which dot, angle, mean and unique take into account. It is
  stored in Euclidean coordinates; MATLAB's class Miller is a vector3d here, built by the
  function Miller from its indices.

  Input
    x, y, z - arrays of the Cartesian coordinates, broadcast against each other
    xyz - array of shape (..., 3) or (3, N), a vector3d, or a rank one tensor
    cs - crystalFrame, also given positionally to vector3d
  Output
    v - vector3d
    m - vector3d, a crystal direction
  Keywords
    antipodal - the vectors are axes: v and -v are the same
    isNormalized - the coordinates are known to be of length one
    frame - the referenceFrame the coordinates are given in, a crystal frame making the
        vectors crystal directions; also given as a positional argument
    dispStyle - the indices the display shows, 'hkl', 'hkil', 'uvw', 'UVTW' or 'xyz'
    dtype - the floating point type of the coordinates
  Properties
    xyz - Cartesian coordinates as a numeric array of shape v.shape + (3,)
    frame - referenceFrame of the coordinates, or None for frame-free data
    CS - crystalFrame for Miller directions, otherwise None
    antipodal - whether opposite directions represent the same axis
    isNormalized - whether the coordinates are known to have unit length
    dispStyle - display convention: 'xyz', 'hkl', 'hkil', 'uvw' or 'UVTW'
    hkl - three reciprocal-lattice indices for plane normals in a crystal frame
    uvw - three direct-lattice indices for lattice directions in a crystal frame
    shape - array shape excluding the trailing Cartesian coordinate axis
  See also
    Miller, vector3d.byPolar, crystalFrame, VectorDefinition, CrystalDirections"""
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

  h = property(lambda self: self.hkl[..., 0], doc="the Miller index h of a plane normal")
  k = property(lambda self: self.hkl[..., 1], doc="the Miller index k of a plane normal")
  i = property(lambda self: self.hkil[..., 2], doc="the third Miller-Bravais index i = -(h + k) of a plane normal")
  l = property(lambda self: self.hkl[..., 2], doc="the Miller index l of a plane normal")
  u = property(lambda self: self.uvw[..., 0], doc="the index u of a lattice direction")
  v = property(lambda self: self.uvw[..., 1], doc="the index v of a lattice direction")
  w = property(lambda self: self.uvw[..., 2], doc="the index w of a lattice direction")
  U = property(lambda self: self.UVTW[..., 0], doc="the Miller-Bravais index U of a lattice direction")
  V = property(lambda self: self.UVTW[..., 1], doc="the Miller-Bravais index V of a lattice direction")
  T = property(lambda self: self.UVTW[..., 2], doc="the Miller-Bravais index T = -(U + V) of a lattice direction")
  W = property(lambda self: self.UVTW[..., 3], doc="the Miller-Bravais index W of a lattice direction")

  def dspacing(self):
    """return the spacing of the lattice planes represented by each Miller normal.

      d = h.dspacing()

    The spacing is 1/norm(h), in the same length units as the crystal cell. The vector
    must carry a crystal frame. Scaling its Miller indices changes the spacing; normalising
    the Cartesian vector therefore loses the plane spacing.

    See also
      Miller, crystalFrame.basisDual, vector3d.hkl"""
    self._needsCrystal()
    return 1 / self.norm()

  def round(self, style=None, maxHKL=12, maxXYZ=12, accuracy=np.pi / 180, return_error=False):
    """the direction with its indices, or its coordinates, rounded to small integers

      m = round(m)                              # the indices of the display rounded
      m = round(m, maxHKL=5)                    # no index larger than 5
      m = round(m, 'uvw')                       # the direct indices integer
      v = round(v, maxXYZ=5)                    # the Cartesian coordinates of a vector
      v = round(v, accuracy=0.1*degree)         # rounded only where it turns by less
      m, err = round(m, return_error=True)      # also the angle the rounding turned it by

    The direction is scaled so that its largest index is 1, 2, ..., maxHKL, and the scale
    that brings the indices nearest to integers is kept. For a trigonal or hexagonal lattice the three and the four index forms
    are not integer at the same time, UVTW = (1, 3, -4, 8) is uvw = (5, 7, 8)/3, so style says
    which is rounded. A vector without a lattice has its Cartesian coordinates rounded and
    keeps them where the rounding would turn it by more than the accuracy.

    Input
      m - vector3d in a crystal frame
      v - vector3d
    Output
      m, v - vector3d, the rounded directions
      err - array, the angle between the given and the rounded direction, without symmetric
          equivalents, with return_error=True
    Keywords
      style - the indices rounded, 'hkl', 'hkil', 'uvw' or 'UVTW'; by default those of the
          display
      maxHKL - the largest index of a crystal direction
      maxXYZ - the largest coordinate of a vector without a lattice
      accuracy - the largest angle the rounding may turn a vector without a lattice by, in
          radians
      return_error - also return the angle the rounding turned the directions by
    See also
      vector3d.char, Miller, CrystalDirections"""
    from .miller import roundIndices
    r = roundIndices(self, style, maxHKL if self.isCrystalDirection else maxXYZ)
    # a vector rounding would turn too far keeps its coordinates
    if not self.isCrystalDirection:
      far = np.asarray(vector3d.angle(self, r, symmetry=False)) > accuracy
      if np.any(far):
        data = r.data.copy()
        data[far] = self.data[far]
        r = r._new(data)
    if not return_error:
      return r
    return r, vector3d.angle(self, r, symmetry=False)

  def char(self, style=None, utf8=None, family=None):
    """the Miller indices of a crystal direction in brackets, as text

      s = m.char()                       # {111}, <uvw> where the frame has symmetry, else (111), [uvw]
      s = m.char('uvw')                  # in another convention: 'hkl', 'hkil', 'uvw', 'UVTW'
      s = m.char(family=False)           # one plane or direction, (hkl) or [uvw]
      s = m.char('LaTex')                # as mathematics, a bar over the negative indices
      s = m.char(utf8=False)             # a minus sign instead of the bar

    A point group makes the indices stand for the whole symmetrically equivalent set, so they
    are written in the family brackets {hkl} and <uvw>, while the trivial group leaves one plane
    or one direction, (hkl) and [uvw]. A direction outside a crystal frame is written in its
    coordinates, (1,2,3). A list of directions is written one after another. MATLAB's flags
    no_scopes and commasep are not taken.

    Input
      m - vector3d, in a crystal frame for the indices
    Output
      s - str
    Keywords
      style - the convention, 'hkl', 'hkil', 'uvw', 'UVTW' or 'xyz', or 'LaTex'; by default
          the display's
      utf8 - a negative index as a digit with a bar over it rather than a minus sign; by
          default the preference 'UTF8Output'
      family - the family brackets {hkl} and <uvw>; by default where the frame has symmetry
    See also
      vector3d.round, Miller, CrystalDirections"""
    from .miller import indexChar
    # a direction outside a crystal frame is written in its coordinates
    if not self.isCrystalDirection and style in (None, "xyz"):
      d = np.round(self.data.reshape(-1, 3), 10)
      return " ".join("(" + ",".join(f"{x:g}" for x in row) + ")" for row in d)
    return indexChar(self, style, utf8, family)

  def __str__(self):
    if self.size > 4:
      return f"{self.size} points"
    return self.char()

  # ------------------------------------------------------------------ constructors

  @classmethod
  def load(cls, fname, *args, **kwargs):
    """read Cartesian directions from a text file.

      v = vector3d.load('vectors.txt', columnNames=['x', 'y', 'z'])

    The first result of `loadVector3d` is returned as a vector3d. File-format and column
    options are passed to that reader. Use loadVector3d directly to retain additional columns.

    Input
      fname - path to the input file
    See also
      loadVector3d, vector3d.export"""
    from ..io.vectors import loadVector3d
    return loadVector3d(fname, *args, **kwargs)[0]

  def calcDensity(self, *args, **kwargs):
    """estimate a smooth density of directions or axes on the sphere.

      f = v.calcDensity()
      f = v.calcDensity(halfwidth=10*degree)
      f = v.calcDensity(weights=w, antipodal=True)

    The result is an S2FunHarmonic with mean one: a value of two represents twice the
    uniform directional density. The input's crystal symmetry and antipodal state are
    respected. Without a kernel or halfwidth, the smoothing width is selected from the data.
    Evaluate the density with f(queryDirections). All options are those of `calcDensity`.

    See also
      calcDensity, calcKernel, vector3d.interp"""
    from ..functions.density import calcDensity
    return calcDensity(self, *args, **kwargs)

  def interp(self, y, *args, **kwargs):
    """interpolate values sampled at directions on the sphere.

      f = v.interp(values, 'linear')
      f = v.interp(values, 'harmonic', bandwidth=16)
      result = v.interp(values, queryDirections, 'nearest')

    Supply one value per direction. The methods are 'linear', 'harmonic', 'nearest' and
    'inverseDistance'. Without query directions, return a callable spherical function;
    with query directions, return its values there. Additional options are passed to
    the selected interpolation method.

    Input
      y - values sampled at the directions in v
    See also
      interpolate, S2FunTri, S2FunHarmonic.interpolate"""
    from ..functions.density import interpS2
    return interpS2(self, y, *args, **kwargs)

  def export(self, fname, *args, **kwargs):
    """write vector coordinates to a text table.

      v.export('vectors.txt')

    The file-writing options are passed to `exportVector3d`. The input vectors are unchanged.

    Input
      fname - destination file path
    See also
      exportVector3d, vector3d.load"""
    from ..io.vectors import exportVector3d
    return exportVector3d(self, fname, *args, **kwargs)

  @classmethod
  def byPolar(cls, theta, rho, antipodal=False):
    """the vectors of polar angle theta to z and azimuth rho from x towards y, unit vectors

      v = vector3d.byPolar(theta, rho)                    # directions
      v = vector3d.byPolar(theta, rho, antipodal=True)    # axes

    theta and rho broadcast against each other.

    Input
      theta - the polar angles, 0 at z
      rho - the azimuths, 0 at x
    Output
      v - vector3d of unit vectors
    Keywords
      antipodal - the vectors as axes, v and -v one
    See also
      vector3d.theta, vector3d.rho, vector3d.byXYZ"""
    theta, rho = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(rho, dtype=float))
    v = cls(np.sin(theta) * np.cos(rho), np.sin(theta) * np.sin(rho), np.cos(theta), antipodal=antipodal)
    v.isNormalized = True
    return v

  @classmethod
  def byXYZ(cls, xyz):
    """construct a list of vectors from rows of Cartesian coordinates.

      v = vector3d.byXYZ([[1, 0, 0], [0, 1, 0]])
      v = vector3d.byXYZ([[1, 2], [3, 4]])  # append a zero z coordinate

    Input
      xyz - numeric array of shape (n, 3) or (n, 2)
    Output
      v - vector3d of shape (n,); coordinates are not normalised
    See also
      vector3d, vector3d.byPolar"""
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
    """uniformly distributed unit vectors or axes, on the sphere or in a sphericalRegion given
    last; rng= a NumPy random generator

      v = vector3d.rand()                     # one random unit vector
      v = vector3d.rand(m, n)                 # an m x n array of them
      v = vector3d.rand(100, antipodal=True)  # random axes
      v = vector3d.rand(100, sR)              # uniform in the region sR
      v = vector3d.rand(100, rng=np.random.default_rng(0))  # reproducible

    In a region the uniform vectors outside it are rejected until enough are left.

    Input
      shape - the sizes of the array, one tuple or several integers, and a sphericalRegion
          as the last argument
    Output
      v - vector3d of unit vectors
    Keywords
      antipodal - the vectors as axes
      rng - the NumPy random generator, a fresh one by default
    See also
      rotation.rand, equispacedS2Grid"""
    rng = np.random.default_rng() if rng is None else rng
    sR = shape[-1] if shape and hasattr(shape[-1], "checkInside") else None
    shape = shape[:-1] if sR is not None else shape
    shape = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else tuple(shape)
    if sR is None:
      theta = np.arccos(2 * rng.random(shape) - 1)
      rho = 2 * np.pi * rng.random(shape)
      return cls.byPolar(theta, rho, antipodal=antipodal)
    # uniform on the sphere, the vectors outside the region rejected until enough are left
    n, kept, drawn, inside = int(np.prod(shape)), [], 0, 0
    while inside < n:
      m = max(100, int(1.5 * (n - inside) * (drawn + 1) / (inside + 1)))
      x = rng.standard_normal((m, 3))
      x = x / np.linalg.norm(x, axis=1, keepdims=True)
      x = x[sR.checkInside(x, tol=0)]
      kept.append(x)
      drawn, inside = drawn + m, inside + len(x)
    xyz = np.concatenate(kept)[:n].reshape(shape + (3,))
    return cls(xyz, isNormalized=True, antipodal=antipodal)

  @classmethod
  def zeros(cls, *shape):
    """the zero vectors of a shape

      v = vector3d.zeros()        # one zero vector
      v = vector3d.zeros(m, n)    # an m x n array of them"""
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return cls(np.zeros(tuple(shape) + (3,)))

  @classmethod
  def nan(cls, *shape):
    """construct vectors with missing Cartesian coordinates.

      v = vector3d.nan(10)
      v = vector3d.nan(2, 3)

    Every coordinate is NaN. Sizes may be supplied as separate integers or one tuple;
    without a size, return one scalar vector. Use `vector3d.isnan` to identify missing vectors."""
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
    """Cartesian x coordinate of each vector; assignment changes the coordinates and clears isNormalized."""
    return self.data[..., 0]

  @x.setter
  def x(self, value):
    """assigning the x coordinates"""
    self.data[..., 0] = value
    self.isNormalized = False

  @property
  def y(self):
    """Cartesian y coordinate of each vector; assignment changes the coordinates and clears isNormalized."""
    return self.data[..., 1]

  @y.setter
  def y(self, value):
    """assigning the y coordinates"""
    self.data[..., 1] = value
    self.isNormalized = False

  @property
  def z(self):
    """Cartesian z coordinate of each vector; assignment changes the coordinates and clears isNormalized."""
    return self.data[..., 2]

  @z.setter
  def z(self, value):
    """assigning the z coordinates"""
    self.data[..., 2] = value
    self.isNormalized = False

  @property
  def xyz(self):
    """Cartesian coordinates as a numeric array of shape v.shape + (3,); the array shares storage with v."""
    return self.data

  def polar(self):
    """return polar angle, azimuth and length for every vector.

      theta, rho, radius = v.polar()

    The polar angle theta is measured from +z towards the xy plane and lies in [0, pi].
    The azimuth rho is measured from +x towards +y and lies in [0, `2*pi`). Both angles
    are in radians. Radius has the units of the Cartesian coordinates. The three arrays
    have the shape of v; for a zero vector the implementation returns theta=pi/2 and rho=0.

    See also
      vector3d.byPolar, vector3d.theta, vector3d.rho"""
    r = np.ones(self.shape) if self.isNormalized else self.norm()
    rho = np.arctan2(self.y, self.x)
    rho = rho + (rho < 0) * 2 * np.pi
    theta = np.arccos(np.clip(self.z / np.where(r == 0, 1.0, r), -1, 1))
    return theta, rho, r

  @property
  def theta(self):
    """polar angle from +z in radians, in [0, pi], with one value per vector."""
    return self.polar()[0]

  @property
  def rho(self):
    """azimuth from +x towards +y in radians, in [0, `2*pi`), with one value per vector."""
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
    # Qhull refuses fewer than four points and points in a plane
    try:
      area = _cellAreas(u)
    except (RuntimeError, ValueError):
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
    """the inner product of two vectors, pointwise; of crystal directions the largest over their
    symmetric equivalents

      d = dot(v1, v2)                       # pointwise, broadcast against each other
      d = dot(v1, v2, antipodal=True)       # up to sign: the absolute value
      d = dot(m1, m2, symmetry=False)       # crystal directions as they are given
      d = dot(m1, m2, min=True)             # the smallest over the symmetric equivalents
      d = dot(m1, m2, all=True)             # every symmetric equivalent
      d = dot(v1, v2, outer=True)           # every pair, MATLAB's dot_outer

    An axis on either side reads the product up to sign; antipodal=True does so always,
    antipodal=False never. Crystal directions in a frame with symmetry give the largest product
    over their symmetric equivalents.

    Input
      v1, v2 - vector3d
      m1, m2 - vector3d in one crystal frame
    Output
      d - array, of the broadcast shape; with outer=True of shape v1.shape + v2.shape
    Keywords
      antipodal - read the vectors as axes: True always, False never; by default where one
          of them is an axis
      symmetry - take the symmetric equivalents of crystal directions into account
      min - the smallest product over the symmetric equivalents instead of the largest
      all - the products with every symmetric equivalent, along a new first axis
      outer - every entry of v1 with every entry of v2
    See also
      vector3d.angle, vector3d.cross, vector3d.norm"""
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
    """the length of the vectors

      n = norm(v)"""
    return np.linalg.norm(self.data, axis=-1)

  def angle(self, other, N=None, antipodal=None, symmetry=True, outer=False):
    """the angle between two vectors; of crystal directions the smallest over their symmetric
    equivalents

      omega = angle(v1, v2)                   # pointwise, broadcast against each other
      omega = angle(v1, v2, antipodal=True)   # between axes, at most 90 degree
      omega = angle(h1, h2, symmetry=False)   # crystal directions as they are given
      omega = angle(v1, v2, N)                # in the plane normal to N, from 0 to 2 pi
      omega = angle(v1, v2, outer=True)       # every pair, MATLAB's angle_outer

    The angle is taken through atan2 of the cross and the inner product, accurate near 0 and
    pi. An axis on either side reads it up to sign; antipodal=True does so always,
    antipodal=False never. With a normal N both vectors are projected to the plane normal to N
    and the angle is counted counterclockwise about N from v1 to v2.

    Input
      v1, v2 - vector3d
      h1, h2 - vector3d in one crystal frame
      N - vector3d, the normal of the plane the angle is measured in
    Output
      omega - array, in radians; with outer=True of shape v1.shape + v2.shape
    Keywords
      antipodal - read the vectors as axes: True always, False never; by default where one
          of them is an axis
      symmetry - take the symmetric equivalents of crystal directions into account
      outer - every entry of v1 with every entry of v2
    See also
      vector3d.dot, vector3d.cross, VectorsAxes, CrystalOperations"""
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
    """project each vector onto the plane perpendicular to a normal.

      inPlane = v.orthProj(normal)

    Compute `v - dot(v, n)*n` with n the unit normal. This uses Cartesian dot products,
    without crystal symmetry or identification of opposite directions. Inputs broadcast.

    Input
      N - nonzero vector3d normal to the projection plane
    Output
      inPlane - vector3d in the frame of v, with its original length component in the plane
    See also
      vector3d.orth, vector3d.dot"""
    n = self._coerce(N).normalize()
    return self - n * self.dot(n, antipodal=False, symmetry=False)

  # two vectors are equal when their coordinates are, in frames of one basis; isclose compares
  # their directions up to symmetry
  def __eq__(self, other):
    if not isinstance(other, vector3d):
      return NotImplemented
    if not self._sameBasisAs(other):
      return np.zeros(np.broadcast_shapes(self.shape, other.shape), dtype=bool)
    same = np.all(self.data == other.data, axis=-1)
    # an axis is the same axis as its negative, as a rotation is its quaternion up to sign
    if self.antipodal or other.antipodal:
      same = same | np.all(self.data == -other.data, axis=-1)
    return same

  def _sameBasisAs(self, other):
    if (self._frame is not other._frame) and self.isCrystalDirection and other.isCrystalDirection:
      from .frames import _sameBasis
      return _sameBasis(self._frame, other._frame)
    return True

  def __ne__(self, other):
    return ~(self == other)

  def isnan(self):
    """whether a vector has a NaN coordinate"""
    return np.any(np.isnan(self.data), axis=-1)

  # ------------------------------------------------------------------ derived vectors

  def normalize(self, zeroToZero=False):
    """the vectors scaled to unit length; with zeroToZero=True a zero vector stays zero

      v = v.normalize()                             # a zero vector becomes NaN
      v = v.normalize(zeroToZero=True)              # a zero vector stays zero

    Input
      v - vector3d
    Output
      v - vector3d of unit length, isNormalized set
    Keywords
      zeroToZero - keep vectors of length zero at zero instead of NaN
    See also
      vector3d.norm"""
    if self.isNormalized:
      return self
    n = self.norm()
    if zeroToZero:
      n = np.where(n > np.finfo(float).eps, n, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
      return self._new(self.data / n[..., None], isNormalized=True)

  def orth(self):
    """choose a unit vector perpendicular to each input vector.

      normal = v.orth()

    The choice is deterministic but arbitrary: it specifies no preferred direction in
    the perpendicular plane. Each result keeps the input frame. Use `vector3d.perp`
    to fit one axis perpendicular to a collection of directions.

    See also
      vector3d.cross, vector3d.orthProj, vector3d.perp"""
    return self._new(orthKernel(self.data), isNormalized=True)

  def perp(self, robust=False):
    """the axis best orthogonal to a list of directions, the smallest principal axis

      n = v.perp()                                  # e.g. the zone axis of plane normals
      n = v.perp(robust=True)                       # outliers left out

    The eigenvector of the smallest eigenvalue of sum v v^T; NaN vectors are left out. A
    robust fit drops the 20 percent of the directions furthest from orthogonal and fits
    again, when more than four are given. The result is an axis in the frame of v, in the
    other display style, a zone axis of planes.

    Input
      v - vector3d or Miller
    Output
      n - vector3d or Miller, antipodal
    Keywords
      robust - fit again without the directions furthest from orthogonal
    See also
      vector3d.eig, vector3d.cross"""
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
    """return the eigenvalues and principal axes of the vector moment matrix.

      values, axes = v.eig()

    Flatten the list and form `sum(v_i*v_i.T)`, without normalising vector lengths or
    reducing crystal symmetry. The three eigenvalues are in ascending order; axes[j]
    is the corresponding unit eigenvector in the input frame. The first axis is the
    least represented direction, and the last is the most represented.

    See also
      vector3d.perp, vector3d.mean"""
    xyz = self.data.reshape(-1, 3)
    w, V = np.linalg.eigh(xyz.T @ xyz)
    return w, vector3d(V.T, isNormalized=True, frame=self._frame)

  def sum(self, axis=None, dtype=None, out=None, keepdims=False, **kwargs):
    """the sum of the vectors, over all axes or the given ones

      s = sum(v)                              # over all vectors
      s = sum(v, axis=0)                      # along the first axis
      s = np.sum(v, axis=1, keepdims=True)    # NumPy's form, the summed axis kept

    The result keeps the frame and the display style of the vectors. MATLAB sums along the
    first axis that is not a singleton; the sum here is over all axes unless one is given.

    Input
      v - vector3d
    Output
      s - vector3d
    Keywords
      axis - an axis or a tuple of axes to sum over; by default all
      dtype - NumPy's argument, refused: a container has no dtype to choose
      out - NumPy's argument, refused: a container has no output buffer
      keepdims - keep the summed axes with length one
    See also
      vector3d.mean, vector3d.norm"""
    numpyReduction("sum", dtype, out, kwargs, "vectors")
    s = self._new(np.sum(self.data, axis=tuple(range(self.ndim)) if axis is None else axis), **self._joined(None))
    return keptDims(s, self.shape, axis) if keepdims else s

  def mean(self, weights=None, antipodal=False, robust=False, axis=None, symmetry=True, return_projected=False,
           dtype=None, out=None, keepdims=False, **kwargs):
    """the mean direction of a list of vectors, of axes, or of crystal directions

      m = mean(v)                                     # the arithmetic mean over all vectors
      m = mean(v, axis=0)                             # along the first axis
      m = mean(v, weights=w)                          # the normalised weighted sum
      m = mean(v, antipodal=True)                     # the mean axis
      m = mean(v, robust=True)                        # without the farthest fifth
      m, hProjected = mean(h, return_projected=True)  # also the equivalents averaged, MATLAB's mSym

    Vectors give their arithmetic mean, with weights the normalised weighted sum. Axes give the
    eigenvector of the largest eigenvalue of the moment matrix. Crystal directions in a frame
    with symmetry are first moved to the equivalents closest to the mean, which
    return_projected=True returns too. robust=True drops the vectors farther from a first mean
    than 80 percent of them and averages again.

    Input
      v - vector3d
      h - vector3d in a crystal frame
    Output
      m - vector3d
      hProjected - vector3d, the equivalents of the directions the mean was taken over
    Keywords
      weights - array, a weight per vector
      antipodal - average the vectors as axes
      robust - leave out the outliers, the vectors farthest from the mean
      axis - an axis or a tuple of axes to average over; by default all
      symmetry - take the symmetric equivalents of crystal directions into account
      return_projected - also return the equivalents the mean was taken over
      dtype - NumPy's argument, refused: a container has no dtype to choose
      out - NumPy's argument, refused: a container has no output buffer
      keepdims - keep the averaged axes with length one
    See also
      vector3d.sum, vector3d.unique, VectorsOperations"""
    numpyReduction("mean", dtype, out, kwargs, "vectors")
    if keepdims:
      m = self.mean(weights=weights, antipodal=antipodal, robust=robust, axis=axis, symmetry=symmetry)
      return keptDims(m, self.shape, axis)
    if symmetry and self.isCrystalDirection and self._frame.numSym() > 1 and self.size > 1:
      from .symmetryops import symMean
      return symMean(self, weights=weights, return_projected=return_projected)
    if return_projected:
      return self.mean(weights=weights, antipodal=antipodal, robust=robust, axis=axis, symmetry=symmetry), self
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
    """the disjoint list of vectors, equal ones within a tolerance taken once

      u = unique(v)                                            # sorted, each vector once
      u = unique(v, tol=0.01)                                  # a larger tolerance
      u = unique(v, antipodal=True)                            # v and -v the same
      u = unique(v, stable=True)                               # in the order of their first appearance
      u, iv, iu = unique(v, return_index=True, return_inverse=True)  # u = v[iv], v = u[iu]

    Axes are equal up to sign; antipodal=True takes them so always, antipodal=False never.
    Crystal directions in a frame with symmetry are equal when they are symmetrically
    equivalent: they are projected to the fundamental sector first.

    Input
      v - vector3d
    Output
      u - vector3d, one dimensional
      iv - array, the indices with u = v.flat[iv], with return_index=True
      iu - array, the indices with v.flat = u[iu], with return_inverse=True
    Keywords
      tol - the tolerance of the coordinates within which vectors are equal
      antipodal - take the vectors as axes: True always, False never; by default where they
          are axes
      symmetry - take the symmetric equivalents of crystal directions into account
      stable - keep the order of the first appearances rather than sorting
      return_index - also return the indices iv of the kept vectors
      return_inverse - also return the indices iu reconstructing the list
    See also
      vector3d.symmetrise, vector3d.mean"""
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
    return _uniqueOutput(self._new(xyz[iv]), iv, iu, return_index, return_inverse)

  def isclose(self, other, antipodal=False, tol=1e-10):
    """isclose(v, w), isclose(v, w, antipodal=True, tol=): pointing the same way up to the group
    of a crystal frame within the tolerance tol of the cosine, and up to sign when antipodal
    (MATLAB's eq)

      b = v.isclose(w)                              # the same direction
      b = v.isclose(w, antipodal=True)              # the same axis
      b = v.isclose(w, tol=1e-6)                    # a coarser tolerance

    True where <v, w> differs from |v| |w| by less than tol, the product reduced over the
    group of a crystal frame; vectors of different frames are never close. The operator ==
    compares exactly.

    Input
      v, w - vector3d or Miller
    Output
      b - boolean array
    Keywords
      antipodal - up to sign, also when neither vector is antipodal
      tol - the tolerance of the cosine times the lengths
    See also
      vector3d.angle, vector3d.dot"""
    other = self._coerce(other)
    a = self._new(self.data, antipodal=True) if antipodal else self
    if not a._sameBasisAs(other):
      return np.zeros(np.broadcast_shapes(a.shape, other.shape), dtype=bool)
    return isnull(a.dot(other) - a.norm() * other.norm(), tol)

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
    """the rotation vector to its rotation: at the identity, or at a reference as a tangent vector

      rot = exp(v)                                  # the rotation of axis v and angle |v|
      mori = exp(h)                                 # a crystal direction: a misorientation of its frame
      ori = exp(h, oriRef)                          # at a reference orientation

    A vector in a crystal frame gives the misorientation of that frame with itself, and is
    taken on the right of a reference; a specimen vector on the left.

    Input
      v - vector3d or Miller, the rotation vector: the axis times the angle
      ref - rotation or orientation, the reference
    Output
      rot - rotation, or orientation for a crystal direction or a reference orientation
    Keywords
      side - 'left' or 'right', the side of the reference; by default right for a crystal
          direction, left otherwise
    See also
      quaternion.log, quaternion.exp, SO3TangentVector"""
    from .tangent import expVector
    return expVector(self, ref, side)

  def symmetrise(self, unique=False, antipodal=None):
    """the symmetrically equivalent directions, numSym x shape, or the distinct ones of one
    direction with unique=True; MATLAB's second output, the multiplicity, is v.multiplicity(),
    its third the rotations of the frame, v.frame.rotations()

      hs = h.symmetrise()                           # every image, numSym x shape
      hs = h.symmetrise(unique=True)                # the distinct ones of one direction
      hs = h.symmetrise(antipodal=True)             # the negatives too

    With antipodal=True the negatives are added, unless the group holds the inversion or h
    is an axis; with unique=True an axis, or antipodal=True, takes v and -v as one, and
    antipodal=False keeps them apart. A vector in no crystal frame is returned with a leading
    axis of one.

    Input
      h - Miller
    Output
      hs - Miller
    Keywords
      unique - the distinct images of a single direction, a flat list
      antipodal - treat the directions as axes; by default as h says
    See also
      Miller.multiplicity, symmetry.rotations"""
    from .symmetryops import symmetrise
    return symmetrise(self, unique=unique, antipodal=antipodal)

  def multiplicity(self, antipodal=None):
    """the number of distinct directions symmetrically equivalent to each vector, of axes under
    antipodal

      n = h.multiplicity()                          # 12 for <110> in 432
      n = h.multiplicity(antipodal=True)            # 6, the axes

    Input
      h - Miller
    Output
      n - an integer per vector, in the shape of h
    Keywords
      antipodal - count the axes, v and -v one; by default as h says
    See also
      Miller.symmetrise"""
    from .symmetryops import multiplicity
    return multiplicity(self, antipodal=antipodal)

  def projectIntoFundamentalRegion(self, center=None, antipodal=False):
    """the directions folded into the fundamental sector of their frame, with antipodal=True into
    that of the Laue group; given a centre, the equivalent nearest it instead
    (MATLAB's vector3d.project2FundamentalRegion, Miller.project2FundamentalRegion)

      h = h.projectIntoFundamentalRegion()          # into the sector of an inverse pole figure
      h = h.projectIntoFundamentalRegion(antipodal=True)  # into the sector of the Laue group
      h = h.projectIntoFundamentalRegion(center)    # the equivalent nearest a centre

    The sector is the conventional one, fundamentalSector of the frame. Vectors in no
    crystal frame are returned as they are.

    Input
      h - Miller
      center - Miller, the centre the equivalent nearest to it is taken for
    Output
      h - Miller, in the shape of h
    Keywords
      antipodal - fold the directions as axes, into the sector of the Laue group
    See also
      fundamentalSector, projectIntoRegion, Miller.symmetrise"""
    from .symmetryops import projectIntoFundamentalRegion
    v = self
    if antipodal and self.isCrystalDirection:
      v = self._new(self.data, _frame=self._frame.Laue())
    return projectIntoFundamentalRegion(v, center)

  def find(self, w, epsOrK=None, antipodal=None):
    """the closest points of this list to others, the k closest, or all within a radius; up to sign
    where either is antipodal, antipodal=True always, False never

      ind, d = v.find(w)                            # the nearest of v to every w, and its angle
      ind, d = v.find(w, 3)                         # the 3 nearest, an integer
      I, D = v.find(w, 5*degree)                    # all within 5 degree, a float

    The angles are reduced over the group of a crystal frame. Within a radius the result is
    two sparse matrices of len(w) x len(v), the hits and the angles; large lists of specimen
    directions are searched by a tree.

    Input
      v - vector3d, the list searched
      w - vector3d, the directions searched for
      epsOrK - the number of neighbours, an integer, or the radius, a float
    Output
      ind - the indices into v, in the shape of w; (len(w), k) for k neighbours
      d - the angles to them
      I, D - sparse matrices of the hits and of the angles
    Keywords
      antipodal - read the directions up to sign; by default where v or w is antipodal
    See also
      vector3d.angle, S2Grid.find"""
    from .grids.search import find
    return find(self, w, epsOrK, antipodal)

  def transformReferenceFrame(self, to, rule=None, tol=1e-1):
    """the same direction written in another frame

      h2 = h.transformReferenceFrame(cs2)           # by the axes of the two frames
      h2 = h.transformReferenceFrame(cs2, rule='byScreenAlignment')  # as the two frames are drawn
      h2 = h.transformReferenceFrame(cs2, rule=ori) # by a given orientation from one to the other

    The direction keeps its meaning and gets the coordinates of the other frame, in hkl or
    uvw as it was written; data of two settings of one crystal, 'X||a' and 'X||a*', are
    combined this way.

    Input
      h - vector3d or Miller
      to - the frame to write it in
    Output
      h2 - vector3d or Miller in the frame to
    Keywords
      rule - 'byAxes' or None, the transition of the two bases; 'byScreenAlignment', the one
          of the plotting conventions; or an orientation from the frame of h to to
      tol - how far the transition of the bases may be from a rotation, the norm of
          I - M M'; a larger deviation is refused
    See also
      align, orientation.byScreenAlignment, orientation.transformReferenceFrame"""
    from .frametransform import transformVector
    return transformVector(self, to, rule, tol)

  def det(self, b=None, c=None):
    """the triple product, vector3d/det
    det(a, b, c) the triple product; det(v) of three vectors the determinant of their matrix

      d = det(a, b, c)                              # <a, b x c>, element by element
      d = det(v)                                    # three vectors as the rows of a matrix

    Input
      a - vector3d
      b, c - vector3d, the second and third factor
    Output
      d - the triple products
    See also
      vector3d.cross, vector3d.dot"""
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


# the indices of the kept elements and the index of every element into them, from the mask of
# the kept ones and the kept one each element went to
def _keptAndInverse(keep, first):
  iv = np.flatnonzero(keep)
  rank = np.empty(len(keep), dtype=int)
  rank[iv] = np.arange(len(iv))
  return iv, rank[first]


# what unique returns: the distinct elements, with the indices of the kept and the inverse map
# when asked
def _uniqueOutput(u, iv, iu, return_index, return_inverse):
  out = (u,) + ((iv,) if return_index else ()) + ((iu,) if return_inverse else ())
  return out if len(out) > 1 else u


def asVector3d(v):
  """v itself if it is a vector3d (a Miller included), else the vector3d of its coordinates"""
  return v if isinstance(v, vector3d) else vector3d(v)


# the coordinates of a vector3d, or a plain triple or array of them as floats
def _xyz(v):
  return v.data if isinstance(v, vector3d) else np.asarray(v, dtype=float)


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
  # plane normals and lattice directions are the same vectors, shown in the style of the first
  return b if a == "xyz" else a


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



__lazy__ = ["xvector", "yvector", "zvector"]


def isCrystalDirection(v):
  """whether v is a vector given in a crystal frame"""
  return isinstance(v, vector3d) and v.isCrystalDirection


__all__ = ["vector3d", "isCrystalDirection"]
