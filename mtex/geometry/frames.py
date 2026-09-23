# reference frames (MATLAB @referenceFrame, @crystalFrame, @specimenFrame, notIndexedFrame;
# Julia ADR 0004)
#
# A reference frame answers "what coordinate system is this data expressed in": an
# identity (a mineral, or 'rolling'), a basis, the point group the data is invariant
# under, and the default convention it is drawn with. A frame is a handle: == is
# identity, and the session register hands out one handle per key, so two imports of
# forsterite are one frame while ferrite and martensite with one cell are two.
#
#   crystalFrame('m-3m', [4.05, 4.05, 4.05], mineral='Aluminium')
#   crystalFrame('2/m', [8.6, 13, 7.2], [90, 116, 90]*degree, 'X||a')
#   specimenFrame('222')                  the session frame in that sample symmetry
#   specimenFrame.rolling, specimenFrame.rolling('mmm')    the named specimen frames
#
# A registered frame is sealed: basis, axes names and group cannot change under the
# datasets holding it. Name, colour and the plotting convention stay open, as in MATLAB.
from __future__ import annotations

import warnings

import numpy as np

from .crystalaxes import alignment as _alignment, crystalAxes, isAlignment, reciprocalAxes
from .data.pointgroupnames import LAUE_CLASSES, POINT_GROUP_NAMES, findPointGroup, isPointGroupName, monoclinicSetting
from .latticetype import latticeType
from .plottingconvention import ARROWS_ASCII, ARROWS_UTF8, plottingConvention
from .quaternion import orthogonalTransform, quaternion, rotation
from .session import SESSION, getMTEXpref
from .symmetry import spaceGroupToPointGroup, symmetry
from .vector3d import classproperty, vector3d, xvector, yvector, zvector

CANONICAL_BASIS = np.eye(3)


class referenceFrame:
  tolAligned = 5e-2      # the same frame: eqTol
  tolCompatible = 1e-1   # a transformable frame: isCompatible

  def __init__(self, basis=None, name="", axesNames=("X", "Y", "Z"), how2plot=None, sym=None, color=None):
    self._sealed = False
    self._basis = CANONICAL_BASIS.copy() if basis is None else _basisArray(basis)
    self._axesNames = tuple(str(n) for n in axesNames)
    self._sym = symmetry() if sym is None else sym
    self.name = str(name)
    self.color = color
    self.opt = {}
    self.how2plot = how2plot

  # ------------------------------------------------------------------ identity

  # frames compare by identity; the same frame in another group is a sibling
  def __eq__(self, other):
    return self is other

  def __ne__(self, other):
    return self is not other

  def __hash__(self):
    return id(self)

  # a registered frame is sealed: what its identity is keyed on can no longer be written
  def _assertOpen(self, what):
    if self._sealed:
      raise AttributeError(f"the {what} of a registered reference frame cannot be changed - every dataset holding it would change with it; construct the frame you want and let the register answer")

  @property
  def basis(self):
    return vector3d(self._basis.copy())

  @basis.setter
  def basis(self, v):
    self._assertOpen("basis")
    self._basis = _basisArray(v)

  @property
  def axesNames(self):
    return self._axesNames

  @axesNames.setter
  def axesNames(self, n):
    self._assertOpen("axes names")
    self._axesNames = tuple(str(x) for x in n)

  @property
  def sym(self):
    return self._sym

  @sym.setter
  def sym(self, s):
    self._assertOpen("point group")
    self._sym = s

  @property
  def how2plot(self):
    return self._how2plot

  @how2plot.setter
  def how2plot(self, pC):
    self._how2plot = plottingConvention(pC) if isinstance(pC, str) else pC

  @property
  def mineral(self):
    return self.name

  @mineral.setter
  def mineral(self, v):
    self.name = str(v)

  @property
  def isIndexed(self):
    return not isinstance(self, notIndexedFrame)

  # the group questions a frame answers, read through the frame
  @property
  def id(self):
    return self.sym.id

  @property
  def rot(self):
    return self.sym.rot

  @property
  def pointGroup(self):
    return self.sym.pointGroup

  @property
  def lattice(self):
    return self.sym.lattice

  @property
  def multiplicityZ(self):
    return self.sym.multiplicityZ

  @property
  def multiplicityPerpZ(self):
    return self.sym.multiplicityPerpZ

  @property
  def isLaue(self):
    return self.sym.isLaue

  @property
  def isProper(self):
    return self.sym.isProper

  @property
  def Schoenflies(self):
    return self.sym.Schoenflies

  def numSym(self):
    return self.sym.numSym()

  def numProper(self):
    return self.sym.numProper()

  def nfold(self, axis=None):
    return self.sym.nfold(axis)

  def elements(self, multiplicity=None):
    return self.sym.elements(multiplicity)

  def LaueName(self):
    return self.sym.LaueName()

  def rotations(self):
    return self.sym.rotations()

  def __len__(self):
    return self.sym.numSym()

  def __contains__(self, e):
    return e in self.sym

  # ------------------------------------------------------------------ siblings

  # this frame with another group in it, from the register
  def sibling(self, s):
    if s == self.sym:
      return self
    fr = self.clone()
    fr._sym = s
    return referenceFrame.intern(fr)

  def clone(self):
    fr = object.__new__(type(self))
    fr.__dict__.update(self.__dict__)
    fr._sealed = False
    fr.opt = dict(self.opt)
    return fr

  def stripSym(self):
    return self.sibling(symmetry())

  def Laue(self):
    return self.sibling(self.sym.Laue())

  def properGroup(self):
    return self.sibling(self.sym.properGroup())

  def properSubGroup(self):
    return self.sibling(self.sym.properSubGroup())

  # the largest group registered over this basis
  def fullSym(self):
    best = self
    for r in SESSION.frames:
      if _sameBasis(r, self) and r.numSym() > best.numSym():
        best = r
    return best

  # the same frame with the group generated by a rotation added, or with another frame's
  # group joined; the intersection of the groups
  def union(self, other):
    if isinstance(other, referenceFrame):
      if other.sym == self.sym:
        return self
      if self >= other:
        return self.sibling(self.sym)
      if other >= self:
        return self.sibling(other.sym)
      return self.sibling(self.sym.union(other.sym))
    return self.sibling(symmetry(orthogonalTransform.cat(self.rot, orthogonalTransform(other)), self._basis))

  def disjoint(self, other):
    if isinstance(other, referenceFrame):
      if self.sym == other.sym:
        return other if isinstance(other, specimenFrame) else self
      d = self.sym.disjoint(other.sym)
    else:
      d = self.sym.disjoint(symmetry(other, self._basis))
    return self.sibling(d)

  def factor(self, other):
    return self.sym.factor(other.sym)

  def __le__(self, other):
    return self.sym <= other.sym

  def __ge__(self, other):
    return other.sym <= self.sym

  def __lt__(self, other):
    return self.sym < other.sym

  def __gt__(self, other):
    return other.sym < self.sym

  # a frame called with a group name is that frame in that group: specimenFrame.rolling('mmm')
  def __call__(self, *group):
    return withGroup(self, group)

  # ------------------------------------------------------------------ regions

  def fundamentalSector(self, center=None, antipodal=False, angle=None):
    from .sphericalregion import fundamentalSector
    return fundamentalSector(self, center=center, antipodal=antipodal, angle=angle)

  def conventionalSector(self, up=None, east=None):
    from .sphericalregion import conventionalSector
    return conventionalSector(self, up=up, east=east)

  def fundamentalRegion(self, other=None, antipodal=False):
    from .orientationregion import fundamentalRegion
    return fundamentalRegion(self, other, antipodal=antipodal)

  # ------------------------------------------------------------------ comparing frames by value

  # the same basis, up to a relative tolerance on the basis vectors
  def isAligned(self, other, tol=None):
    tol = referenceFrame.tolAligned if tol is None else tol
    return bool(np.all(np.linalg.norm(self._basis - other._basis, axis=1) / np.linalg.norm(self._basis, axis=1) < tol))

  # the transition matrix taking coordinates in this frame to coordinates in the other,
  # on normalised bases: with the unit axes as rows, inv(B) A is the rotation between the
  # two settings of one lattice
  def transformationMatrix(self, other):
    A = _unitRows(self._basis)
    B = _unitRows(other._basis)
    return np.linalg.inv(B) @ A

  # whether a transition between the frames exists: the matrix is close to orthogonal
  def isCompatible(self, other, tol=None):
    tol = referenceFrame.tolCompatible if tol is None else tol
    M = self.transformationMatrix(other)
    MM = M @ M.T
    return bool(np.linalg.norm(MM - np.eye(3)) / np.linalg.norm(MM) < tol), M

  # whether two frames may be treated as one: the same class of group and aligned bases,
  # and for crystal frames the same mineral
  def eqTol(self, other):
    if self is other:
      return True
    if type(self) is not type(other):
      return False
    return self._eqTolPair(other)

  # whether two frames describe the same symmetry, without requiring aligned bases
  def sim(self, other):
    if self is other:
      return True
    if type(self) is not type(other):
      return False
    return self._simPair(other)

  # a named group is compared by its Laue class, an unnamed one as the group it is
  def _sameGroup(self, other):
    if self.id == 0:
      return self.sym == other.sym
    return self.Laue().id == other.Laue().id

  # a pickled frame comes back as the session's handle of the same entity, so that identity
  # survives a round trip through a file
  def __reduce__(self):
    state = {k: v for k, v in self.__dict__.items() if k != "_registered"}
    return (_rebuildFrame, (type(self), state))

  # ------------------------------------------------------------------ the register

  # the session instance of a frame: the registered one when it agrees by key, otherwise
  # this one, registered and sealed from now on
  @staticmethod
  def intern(fr):
    with SESSION.lock:
      for r in SESSION.frames:
        if _sameEntity(r, fr):
          return r
      fr._sealed = True
      SESSION.frames.append(fr)
      return fr

  @staticmethod
  def byName(name, fr=None):
    if fr is not None:
      SESSION.namedFrames[name] = fr
    return SESSION.namedFrames.get(name)

  @staticmethod
  def reset():
    SESSION.frames.clear()
    SESSION.namedFrames.clear()
    SESSION.defaultFrame = None

  # ------------------------------------------------------------------ display

  # the convention in the frame's own axes names, e.g. RD↑→TD, or the pictogram in x, y, z
  def conventionChar(self, pC=None):
    pC = self.how2plot if pC is None else pC
    if pC is None:
      return ""
    if self._axesNames == ("X", "Y", "Z"):
      return pC.char()
    dirs = vector3d(_unitRows(self._basis))
    iN, ud = _matchSigned(pC.north, dirs)
    iE, lr = _matchSigned(pC.east, dirs)
    if iN is None or iE is None:
      return pC.char()
    a = ARROWS_UTF8 if getMTEXpref("UTF8Output") else ARROWS_ASCII
    northPart = self._axesNames[iN] + a[2 if ud else 3]
    return northPart + a[1] + self._axesNames[iE] if lr else self._axesNames[iE] + a[0] + northPart

  def char(self):
    return self.name if self.name else type(self).__name__

  def __str__(self):
    return self.char()

  def __repr__(self):
    info = [x for x in (self.name, self.conventionChar()) if x]
    lines = [type(self).__name__ + (f" ({', '.join(info)})" if info else "")]
    if self.numSym() > 1:
      lines += [f"  symmetry: {self.sym.char()}", f"  elements: {self.numSym()}"]
    if np.max(np.abs(self._basis - CANONICAL_BASIS)) > 1e-10:
      for n, b in zip(self._axesNames, self._basis):
        lines.append(f"  {n}: (" + ",".join(f"{x:.3g}" for x in b) + ")")
    return "\n".join(lines)


# ------------------------------------------------------------------ helpers

def _basisArray(v):
  if isinstance(v, vector3d):
    return v.data.reshape(3, 3).copy()
  return np.asarray(v, dtype=float).reshape(3, 3).copy()


def _unitRows(M):
  return M / np.linalg.norm(M, axis=1, keepdims=True)


# which of the directions a vector is, and with which sign
def _matchSigned(v, dirs):
  for sign, ud in ((1, True), (-1, False)):
    hit = np.flatnonzero(v == sign * dirs)
    if hit.size:
      return int(hit[0]), ud
  return None, None


def _rebuildFrame(cls, state):
  fr = object.__new__(cls)
  fr.__dict__.update(state)
  fr.__dict__["_sealed"] = False
  fr.__dict__["_registered"] = None
  r = referenceFrame.intern(fr)
  if r is fr:
    fr.__dict__["_registered"] = fr
  return r


# two frames are the same entity, the register key of ADR 0008: kind, name, group, cell
# shape, alignment and plotting convention; colour is not in it
def _sameEntity(a, b):
  return a.sym == b.sym and _sameBasis(a, b)


def _sameBasis(a, b):
  if type(a) is not type(b) or a.name != b.name:
    return False
  if (a.how2plot is None) != (b.how2plot is None):
    return False
  if a.how2plot is not None and a.how2plot != b.how2plot:
    return False
  if isinstance(a, notIndexedFrame):
    return True
  if isinstance(a, crystalFrame):
    tol = getMTEXpref("frameShapeTolerance", 1e-2)
    return (bool(np.all(np.abs(a.abc - b.abc) / a.abc < tol)) and bool(np.all(np.abs(a.abg - b.abg) < tol))
            and a.alignment() == b.alignment())
  return a.isAligned(b)


# ------------------------------------------------------------------ crystal frames

class crystalFrame(referenceFrame):
  """the reference frame glued to the lattice basis of a phase

    crystalFrame('m-3m', [4.05, 4.05, 4.05], mineral='Aluminium', color='blue')
    crystalFrame('2/m', [8.6, 13, 7.2], [90, 116, 90]*degree, 'X||a', mineral='orthoclase')
    crystalFrame([1, 2, 3], [70, 80, 120]*degree, 'Z||a*')     by the cell alone, no group
    crystalFrame(axes), crystalFrame(axes, name='Forsterite')  by the axes themselves
    crystalFrame(rot)                                           the group generated by rotations
    crystalFrame(otherFrame)                                    the trivial group carrying it

  Positional strings are alignments, 'X||a', 'Z||c*', 'EDAX'; everything else is a keyword.
  """

  def __init__(self, *args, mineral=None, name=None, color=None, alignment=(), rotAxes=None, pointId=None, LaueId=None,
               laueId=None, spaceId=None, how2plot=None, noIntern=False, density=None, axesNames=("a", "b", "c"),
               check=True):
    args = list(args)
    # crystalFrame(LaueId=7): the Laue class by its number
    if LaueId is not None:
      laueId = LaueId
    adopt = None
    alignment = [str(s) for s in ([alignment] if isinstance(alignment, str) else alignment)]
    alignment += [a for a in args if isinstance(a, str) and isAlignment(a)]
    args = [a for a in args if not (isinstance(a, str) and isAlignment(a))]
    sym, axes, rot = None, None, None
    if args and isinstance(args[0], crystalFrame):
      adopt = args.pop(0)
    elif args and isinstance(args[0], referenceFrame):
      raise ValueError(f"a crystal frame is written in a lattice, not in a {type(args[0]).__name__}")
    elif args and isinstance(args[0], (quaternion, orthogonalTransform)):
      rot = args.pop(0)
      axes = _basisArray(args.pop(0)) if args and isinstance(args[0], (vector3d, np.ndarray, list)) and np.size(args[0]) == 9 else CANONICAL_BASIS.copy()
      sym = symmetry(rot, axes) if pointId is None else symmetry(pointId=pointId, axes=axes)
    elif args and isinstance(args[0], vector3d):
      axes = _basisArray(args.pop(0))
      sym = symmetry(pointId=1 if pointId is None else pointId, axes=axes)
    else:
      # the point group by name or id, the cell, the angles, the alignment
      symbol = None
      if args and isinstance(args[0], str):
        symbol = args.pop(0)
      elif pointId is not None:
        symbol = POINT_GROUP_NAMES[pointId - 1][0]
      elif laueId is not None:
        symbol = LAUE_CLASSES[laueId - 1]
      elif spaceId is not None:
        symbol = spaceGroupToPointGroup(spaceId)
      abc = np.asarray(args.pop(0), dtype=float).reshape(-1) if args and not isinstance(args[0], str) else np.array([1.0, 1.0, 1.0])
      abg = np.asarray(args.pop(0), dtype=float).reshape(-1) if args and not isinstance(args[0], str) else None
      if abg is not None and np.any(abg > 2 * np.pi):
        abg = abg * np.pi / 180
      if symbol is None:
        symbol, check = "1", False
      # '2', 'm', '2/m' choose their unique axis from the angles
      symbol = findPointGroup(symbol if abg is None else monoclinicSetting(symbol, abg))
      lat = symmetry(symbol).lattice
      axes = crystalAxes(lat, abc, abg, alignment=alignment, rotate=rotAxes, uniqueAxis=_uniqueMonoclinicAxis(symbol), check=check)
      sym = symmetry(symbol, axes)
    if args:
      raise TypeError(f"crystalFrame does not understand the argument {args[0]!r}")
    if adopt is not None:
      fr = adopt.stripSym()
      self.__dict__.update(fr.__dict__)
      self._sealed = False
      self.opt = dict(fr.opt)
    else:
      super().__init__(axes, "", axesNames, how2plot, sym, None)
    mineral = name if mineral is None else mineral
    if mineral is not None:
      self.name = str(mineral).strip()
    self.color = color
    if density is not None:
      self.opt["density"] = density
    if self.how2plot is None:
      self.how2plot = _crystalConvention(self)
    self._registered = None if noIntern else _registerKeepingColor(self, color)

  # the constructor answers with the registered instance
  # crystalFrame.load('Al-Aluminum.cif'): the frame a CIF file describes, read by the
  # interfaces layer above this one
  @classmethod
  def load(cls, fname):
    from ..io.cif import loadCIF
    return loadCIF(fname)

  def __new__(cls, *args, **kwargs):
    return object.__new__(cls)

  @classmethod
  def byElements(cls, rot, **kwargs):
    return crystalFrame(rot, **kwargs)

  # ------------------------------------------------------------------ the lattice

  @property
  def axes(self):
    return self.basis

  # the reciprocal axes a*, b*, c*, MATLAB's axesDual
  @property
  def axesDual(self):
    from .crystalaxes import reciprocalAxes
    return vector3d(reciprocalAxes(self._basis))

  @property
  def abc(self):
    return np.linalg.norm(self._basis, axis=1)

  @property
  def abg(self):
    b = vector3d(self._basis)
    return np.array([b[1].angle(b[2]), b[2].angle(b[0]), b[0].angle(b[1])])

  @property
  def alpha(self):
    return self.abg[0]

  @property
  def beta(self):
    return self.abg[1]

  @property
  def gamma(self):
    return self.abg[2]

  # a lattice axis as a Miller direction in this frame: [100] or (100) of the frame
  def _axisMiller(self, xyz, style):
    from .miller import Miller
    return Miller(vector3d(xyz), self, dispStyle=style)

  @property
  def aAxis(self):
    return self._axisMiller(self._basis[0], 'uvw')

  @property
  def bAxis(self):
    return self._axisMiller(self._basis[1], 'uvw')

  @property
  def cAxis(self):
    return self._axisMiller(self._basis[2], 'uvw')

  @property
  def aAxisRec(self):
    return self._axisMiller(reciprocalAxes(self._basis)[0], 'hkl')

  @property
  def bAxisRec(self):
    return self._axisMiller(reciprocalAxes(self._basis)[1], 'hkl')

  @property
  def cAxisRec(self):
    return self._axisMiller(reciprocalAxes(self._basis)[2], 'hkl')

  def basisDual(self):
    return vector3d(reciprocalAxes(self._basis))

  def alignment(self):
    return _alignment(self._basis, self._axesNames)

  # between crystal frames the axes are first re-paired by nearest length when the cells
  # disagree, which is how the monoclinic settings 121 and 112 relate
  def transformationMatrix(self, other):
    A = _unitRows(self._basis)
    B = _unitRows(other._basis)
    if isinstance(other, crystalFrame):
      la, lb = self.abc, other.abc
      if not np.all(np.abs(la - lb) / np.sum(la) < 0.01):
        A = A[[int(np.argmin(np.abs(la - lb[k]))) for k in range(3)]]
    return np.linalg.inv(B) @ A

  def _eqTolPair(self, other):
    return self.name.lower() == other.name.lower() and self._sameGroup(other) and self.isAligned(other)

  def _simPair(self, other):
    return (self.name.lower() == other.name.lower() and self._sameGroup(other)
            and bool(np.all(np.abs(self.abc - other.abc) / np.max(self.abc) < 1e-2)) and bool(np.all(np.abs(self.abg - other.abg) < 1e-2)))

  # the convention in crystal directions, e.g. ⊙c→a, with the reciprocal axes named too
  def conventionChar(self, pC=None):
    pC = self.how2plot if pC is None else pC
    if pC is None:
      return ""
    dirs = vector3d(np.concatenate([_unitRows(self._basis), _unitRows(reciprocalAxes(self._basis))]))
    names = list(self._axesNames) + [n + "*" for n in self._axesNames]
    out, east = _matchAxis(pC.outOfScreen, dirs, names), _matchAxis(pC.east, dirs, names)
    if not out or not east:
      return pC.char()
    a = ARROWS_UTF8 if getMTEXpref("UTF8Output") else ARROWS_ASCII
    return a[5] + out + a[1] + east

  def char(self):
    return self.name if self.name else self.sym.char()

  def __repr__(self):
    info = [x for x in (self.name, self.conventionChar()) if x]
    lines = [f"crystalFrame ({', '.join(info)})" if info else "crystalFrame"]
    if self.name:
      lines.append(f"  mineral: {self.name}")
    if self.color is not None:
      lines.append(f"  color: {self.color}")
    lines += [f"  symmetry: {self.sym.char()}", f"  elements: {self.numSym()}",
              "  a, b, c: " + ", ".join(f"{x:.4g}" for x in self.abc)]
    if self.lattice is not latticeType.cubic:
      lines.append("  alpha, beta, gamma: " + ", ".join(f"{x * 180 / np.pi:.4g}°" for x in self.abg))
    al = self.alignment()
    if al:
      lines.append("  reference frame: " + ", ".join(al))
    return "\n".join(lines)


# the first of the named directions a vector lies along, the direct axis before its dual
def _matchAxis(v, dirs, names):
  a = v.angle(dirs)
  hits = np.flatnonzero(a < 1e-4)
  if hits.size:
    return names[hits[0]]
  hits = np.flatnonzero(np.pi - a < 1e-4)
  if hits.size:
    return "-" + names[hits[0]]
  return ""


# the unique axis of a monoclinic setting, b unless the symbol says otherwise
def _uniqueMonoclinicAxis(symbol):
  return 0 if symbol in ("211", "m11", "2/m11") else 2 if symbol in ("112", "11m", "112/m") else 1


# the convention of a crystal frame: c* out of the screen, a to the east (b for the
# monoclinic settings with unique axis b, as MTEX draws them)
def _crystalConvention(fr):
  east = fr.bAxis if 1 <= fr.id <= 11 else fr.aAxis
  return plottingConvention(fr.cAxisRec, east)


# colour is not in the register key: the registered instance keeps the one it was first
# given, unless a colour was asked for explicitly
def _registerKeepingColor(fr, askedColor):
  registered = referenceFrame.intern(fr)
  if registered is not fr and askedColor is not None:
    registered.color = askedColor
  return registered


# ------------------------------------------------------------------ specimen frames

class specimenFrame(referenceFrame):
  """the reference frame a sample is expressed in

    specimenFrame()                 the session default frame
    specimenFrame('mmm')            the session frame in that sample symmetry
    specimenFrame(pC)               the session frame drawn that way
    specimenFrame(otherFrame)       that frame, claiming no symmetry
    specimenFrame('rolling', axesNames=('RD', 'TD', 'ND'), how2plot='y←↑x')   a fresh named frame
    specimenFrame.specimen, .measurement, .rolling, .geological, .default   the named session frames

  A specimen frame has no lattice; a point group on it is a claim about a function, never
  about a direction (MATLAB ADR 0008 rule 10).
  """

  def __init__(self, name=None, *args, axesNames=("X", "Y", "Z"), how2plot=None, noIntern=False, basis=None):
    if isinstance(name, str) and how2plot is None and args and isinstance(args[0], plottingConvention):
      how2plot = args[0]
    if isinstance(name, str) and not isPointGroupName(name):
      super().__init__(basis, name, axesNames, how2plot, None, None)
      self._registered = None if noIntern else self
      return
    super().__init__(basis, "", axesNames, how2plot, None, None)
    self._registered = self
    if name is None:
      self._registered = specimenFrame.default
    elif isinstance(name, str):
      # the group on the frame that carries the convention asked for
      self._registered = withGroup(specimenFrame.default if how2plot is None else specimenFrame.frameFor(how2plot), name)
    elif isinstance(name, plottingConvention):
      self._registered = specimenFrame.frameFor(name)
    elif isinstance(name, specimenFrame):
      self._registered = name.stripSym()
    elif isinstance(name, referenceFrame):
      raise ValueError(f"a specimen frame carries no lattice, so it cannot stand in for a {type(name).__name__}")
    elif isinstance(name, (rotation, orthogonalTransform)):
      self._registered = specimenFrame.default.sibling(symmetry(name))
    else:
      raise TypeError(f"specimenFrame does not understand {name!r}")

  # ------------------------------------------------------------------ the named frames

  # the interned session instance of a named frame, constructed on first use with its seed
  # convention and kept in the session's name map
  @staticmethod
  def named(name, axesNames, seed=None):
    with SESSION.lock:
      fr = referenceFrame.byName(name)
      if fr is None:
        fr = referenceFrame.intern(specimenFrame(name, axesNames=axesNames, how2plot=seed))
        referenceFrame.byName(name, fr)
      return fr

  # the generic specimen frame X, Y, Z - where an import lands and the session default
  @classproperty
  def specimen(cls):
    return specimenFrame.named("specimen", ("X", "Y", "Z"))

  # the sample surface frame of an Oxford instrument, drawn as its SEM images are
  @classproperty
  def measurement(cls):
    return specimenFrame.named("measurement", ("X1", "Y1", "Z1"), plottingConvention.ij)

  # rolling, transverse and normal direction, RD north and TD west
  @classproperty
  def rolling(cls):
    return specimenFrame.named("rolling", ("RD", "TD", "ND"), plottingConvention("y←↑x"))

  # north, east, down of structural geology
  @classproperty
  def geological(cls):
    return specimenFrame.named("geological", ("N", "E", "D"))

  # the frame that supplies the session default convention
  @classproperty
  def default(cls):
    if SESSION.defaultFrame is None:
      SESSION.defaultFrame = specimenFrame.specimen
    if SESSION.defaultFrame.how2plot is None:
      SESSION.defaultFrame.how2plot = plottingConvention.ij
    return SESSION.defaultFrame

  def makeDefault(self):
    SESSION.defaultFrame = self

  # the session frame carrying a convention: the default when it already does, otherwise a
  # fork of it with that convention, registered
  @staticmethod
  def frameFor(pC):
    if isinstance(pC, str):
      pC = plottingConvention(pC)
    if pC is None:
      return None
    fr = specimenFrame.default
    if pC == fr.how2plot:
      return fr
    fork = fr.clone()
    fork._how2plot = plottingConvention(pC)
    return referenceFrame.intern(fork)

  def _eqTolPair(self, other):
    return self.Laue().id == other.Laue().id and self.isAligned(other)

  def _simPair(self, other):
    return self.Laue().id == other.Laue().id

  # the name or the convention, the point group after it when there is one: y↑→x (mmm)
  def char(self):
    base = self.name if self.name else self.conventionChar()
    g = self.sym.char()
    return base if g in ("1", "") else f"{base} ({g})"


# a frame in a group: its sibling, which the register owns; a specimen symmetry above
# orthorhombic is not a sample symmetry
def withGroup(fr, group):
  group = group[0] if isinstance(group, (tuple, list)) and len(group) == 1 else group
  if isinstance(group, (tuple, list)) and len(group) == 0:
    return fr
  s = group if isinstance(group, symmetry) else symmetry(group)
  if s.id > 16:
    warnings.warn(f"{s.pointGroup} is not a suitable specimen symmetry")
  return fr.sibling(s)


# the constructors answer with the registered instance: the object built is discarded
def _answerRegistered(cls):
  new = cls.__new__

  def __new__(klass, *args, **kwargs):
    obj = object.__new__(klass)
    obj.__init__(*args, **kwargs)
    return obj._registered if obj._registered is not None else obj

  cls.__new__ = staticmethod(__new__)
  cls.__init__ = _once(cls.__init__)
  return cls


# an __init__ that runs only when __new__ has not already run it
def _once(init):
  def __init__(self, *args, **kwargs):
    if getattr(self, "_registered", "unset") == "unset":
      init(self, *args, **kwargs)
  return __init__


crystalFrame = _answerRegistered(crystalFrame)
specimenFrame = _answerRegistered(specimenFrame)


# ------------------------------------------------------------------ not indexed

class notIndexedFrame(referenceFrame):

  def __init__(self, name="notIndexed", color=None):
    super().__init__(None, name, ("X", "Y", "Z"), None, None, np.array([np.nan, np.nan, np.nan]) if color is None else color)

  def _eqTolPair(self, other):
    return self.name.lower() == other.name.lower()

  _simPair = _eqTolPair

  def sibling(self, s):
    return self

  def __repr__(self):
    return f"notIndexedFrame ({self.name})"


SESSION.mintDefaultFrame = lambda: specimenFrame.specimen


# ------------------------------------------------------------------ the join rule (ADR 0009)

# the frame two framed operands share: the same handle, the other side when one is
# frame-free, the sibling carrying the intersection of the groups when they differ only in
# their group, and an error otherwise
def joinFrames(a, b):
  if a is b or b is None:
    return a
  if a is None:
    return b
  if _sameBasis(a, b):
    return a.sibling(a.sym.disjoint(b.sym))
  raise ValueError(f"cannot combine data in the frames {a.char()} and {b.char()}; use transformReferenceFrame")


SESSION.joinFrames = joinFrames


# ------------------------------------------------------------------ the alignment setups

# the twelve X||_, Z||_ pairs that construct on every lattice and between them reach every
# distinct frame a lattice has; measured over the 36 pairs, not guessed
ALIGNMENT_SETUPS = (("X||a", "Z||b*"), ("X||a", "Z||c*"), ("X||b", "Z||a*"), ("X||b", "Z||c*"), ("X||c", "Z||a*"),
                    ("X||c", "Z||b*"), ("X||a*", "Z||b"), ("X||a*", "Z||c"), ("X||b*", "Z||a"), ("X||b*", "Z||c"),
                    ("X||c*", "Z||a"), ("X||c*", "Z||b"))


def alignmentSetups():
  return [list(s) for s in ALIGNMENT_SETUPS]


# the setup a crystal frame is reported in: the frame rebuilt in each of the twelve and the
# nearest kept, since the axes it states say nothing where they coincide, X||a on an
# orthogonal lattice or Z||c on a hexagonal one being every setup at once
_CLOSEST_ALIGNMENT = {}


def closestAlignment(cs):
  key = (cs.pointGroup, np.round(cs._basis, 9).tobytes())
  if key in _CLOSEST_ALIGNMENT:
    return list(_CLOSEST_ALIGNMENT[key])
  n = _unitRows(cs._basis)
  best, at = None, None
  for setup in ALIGNMENT_SETUPS:
    try:
      fr = crystalFrame(cs.pointGroup, cs.abc, cs.abg, *setup, noIntern=True, check=False)
    except Exception:
      continue
    d = float(np.max(np.abs(_unitRows(fr._basis) - n)))
    if best is None or d < best - 1e-9:
      best, at = d, list(setup)
  _CLOSEST_ALIGNMENT[key] = tuple(at) if at is not None else None
  return at


__all__ = ["referenceFrame", "crystalFrame", "specimenFrame", "notIndexedFrame", "alignmentSetups", "closestAlignment"]
