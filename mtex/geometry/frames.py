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

from typing import TYPE_CHECKING, Any
import functools
import warnings

import numpy as np

from .representations import unitRows
from .crystalaxes import alignment as _alignment, crystalAxes, isAlignment, reciprocalAxes
from .data.pointgroupnames import LAUE_CLASSES, POINT_GROUP_NAMES, findPointGroup, hm2point, isPointGroupName, monoclinicSetting
from .display import display, header, numString
from .plottingconvention import ARROWS_ASCII, ARROWS_UTF8, plottingConvention
from .quaternion import orthogonalTransform, quaternion, rotation
from .session import SESSION, getMTEXpref
from .symmetry import spaceGroupToPointGroup, symmetry
from .vector3d import classproperty, vector3d, xvector, yvector, zvector
from .display import aligned

if TYPE_CHECKING:
  from .vocabulary import AlignmentName, ColorName, ConventionName, PointGroupName

CANONICAL_BASIS = np.eye(3)


class referenceFrame:
  """a reference frame, the coordinate system data is expressed in: an identity (a name, a
  mineral), a basis, the point group the data is invariant under and the default plotting
  convention; a handle compared by identity, registered and sealed by the session"""
  tolAligned = 5e-2      # the same frame: isclose
  tolCompatible = 1e-1   # a transformable frame: isCompatible

  def __init__(self, basis=None, name="", axesNames=("X", "Y", "Z"), how2plot: ConventionName | Any = None, sym=None,
               color: ColorName | Any = None):
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
    """the basis vectors of the frame in the canonical Euclidean coordinates, one per row"""
    return vector3d(self._basis.copy())

  @basis.setter
  def basis(self, v):
    """sets the basis; refused on a registered frame"""
    self._assertOpen("basis")
    self._basis = _basisArray(v)

  @property
  def axesNames(self):
    """the names of the three axes, ('X', 'Y', 'Z'), ('a', 'b', 'c') or ('RD', 'TD', 'ND')"""
    return self._axesNames

  @axesNames.setter
  def axesNames(self, n):
    """sets the axes names; refused on a registered frame"""
    self._assertOpen("axes names")
    self._axesNames = tuple(str(x) for x in n)

  @property
  def sym(self):
    """the point group the frame carries, a symmetry"""
    return self._sym

  @sym.setter
  def sym(self, s):
    """sets the point group; refused on a registered frame, whose group changes through sibling"""
    self._assertOpen("point group")
    self._sym = s

  @property
  def how2plot(self):
    """the default plotting convention the frame is drawn with, a plottingConvention or None"""
    return self._how2plot

  @how2plot.setter
  def how2plot(self, pC):
    """sets the plotting convention, from a plottingConvention or its string"""
    self._how2plot = plottingConvention(pC) if isinstance(pC, str) else pC

  @property
  def mineral(self):
    """the mineral of the frame, its name"""
    return self.name

  @mineral.setter
  def mineral(self, v):
    """sets the mineral, the name of the frame"""
    self.name = str(v)

  @property
  def isIndexed(self):
    """whether the frame is an indexed phase, False for the notIndexedFrame"""
    return not isinstance(self, notIndexedFrame)

  @property
  def id(self):
    """the group questions a frame answers, read through the frame"""
    return self.sym.id

  @property
  def rot(self):
    """the elements of the point group, an orthogonalTransform"""
    return self.sym.rot

  @property
  def pointGroup(self):
    """the Hermann-Mauguin symbol of the point group"""
    return self.sym.pointGroup

  @property
  def lattice(self):
    """the crystal system of the point group, a latticeType"""
    return self.sym.lattice

  @property
  def multiplicityZ(self):
    """the highest order of a rotation of the group about the z axis"""
    return self.sym.multiplicityZ

  @property
  def multiplicityPerpZ(self):
    """the highest order of a rotation of the group about an axis perpendicular to z"""
    return self.sym.multiplicityPerpZ

  @property
  def isLaue(self):
    """whether the point group is a Laue group, i.e. contains the inversion"""
    return self.sym.isLaue

  @property
  def isProper(self):
    """whether the point group has proper rotations only"""
    return self.sym.isProper

  @property
  def Schoenflies(self):
    """the Schoenflies symbol of the point group"""
    return self.sym.Schoenflies

  def numSym(self):
    """the number of elements of the point group"""
    return self.sym.numSym()

  def numProper(self):
    """the number of proper rotations in the point group"""
    return self.sym.numProper()

  def nfold(self, axis=None):
    """the highest order of a symmetry axis of the group; with axes, the order about each of them"""
    return self.sym.nfold(axis)

  def elements(self, multiplicity=None):
    """the rotational axes of the group, both signs, and their multiplicities; with a multiplicity
    only the axes of that order

      axes, mult = cs.elements()
      axes = cs.elements(3)"""
    return self.sym.elements(multiplicity)

  def LaueName(self):
    """the international symbol of the Laue class of the group"""
    return self.sym.LaueName()

  def rotations(self):
    """the proper rotations of the point group"""
    return self.sym.rotations()

  def __len__(self):
    return self.sym.numSym()

  def __contains__(self, e):
    return e in self.sym

  # ------------------------------------------------------------------ siblings

  def sibling(self, s):
    """this frame with another group in it, from the register"""
    if s == self.sym:
      return self
    # a named group is the same group by its symbol, so the register answers once for it and a
    # registered frame
    key = (self, s._symbol) if s._symbol and self._sealed else None
    hit = SESSION.siblings.get(key)
    if hit is not None:
      return hit
    fr = self.clone()
    fr._sym = s
    fr = referenceFrame.intern(fr)
    if key is not None:
      SESSION.siblings[key] = fr
    return fr

  def clone(self):
    """an unsealed copy of the frame, not registered"""
    fr = object.__new__(type(self))
    fr.__dict__.update(self.__dict__)
    fr._sealed = False
    fr.opt = dict(self.opt)
    return fr

  def stripSym(self):
    """the sibling frame carrying the trivial group"""
    return self.sibling(_trivialGroup())

  def Laue(self):
    """the sibling frame carrying the Laue group, the group with the inversion added"""
    return self.sibling(self.sym.Laue())

  def properGroup(self):
    """the sibling frame carrying the proper point group, the elements with their parity dropped"""
    return self.sibling(self.sym.properGroup())

  def properSubGroup(self):
    """the sibling frame carrying only the proper rotations of this group"""
    return self.sibling(self.sym.properSubGroup())

  def fullSym(self):
    """the largest group registered over this basis"""
    best = self
    for r in SESSION.frames:
      if _sameBasis(r, self) and r.numSym() > best.numSym():
        best = r
    return best

  def union(self, other):
    """the same frame with the group generated by a rotation added, or with another frame's
    group joined; the intersection of the groups"""
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
    """the sibling frame carrying the intersection of the two groups; the other frame when it is a
    specimen frame of the same group"""
    if isinstance(other, referenceFrame):
      if self.sym == other.sym:
        return other if isinstance(other, specimenFrame) else self
      d = self.sym.disjoint(other.sym)
    else:
      d = self.sym.disjoint(symmetry(other, self._basis))
    return self.sibling(d)

  def factor(self, other):
    """the factorisation of the two groups: self = rot1 * d and other = d * rot2 with d the common
    elements

      rot1, d, rot2 = cs1.factor(cs2)"""
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
    """the fundamental sector of the group on the sphere, the conventional one of the inverse pole
    figure where it is a fundamental domain; with a centre that centre's Dirichlet domain

      sR = cs.fundamentalSector()
      sR = cs.fundamentalSector(antipodal=True)
      sR = cs.fundamentalSector(angle=omega)       for a misorientation angle"""
    from .sphericalregion import fundamentalSector
    return fundamentalSector(self, center=center, antipodal=antipodal, angle=angle)

  def conventionalSector(self, up=None, east=None):
    """the conventional sector of the group, the one MTEX draws, oriented by the up and east
    directions of the frame's plotting convention"""
    from .sphericalregion import conventionalSector
    return conventionalSector(self, up=up, east=east)

  def fundamentalRegion(self, other=None, *, antipodal=False):
    """the fundamental region in orientation space of this group, or of the pair with another frame

      oR = cs.fundamentalRegion()
      oR = cs1.fundamentalRegion(cs2, antipodal=True)"""
    from .orientationregion import fundamentalRegion
    return fundamentalRegion(self, other, antipodal=antipodal)

  # ------------------------------------------------------------------ comparing frames by value

  def isAligned(self, other, tol=None):
    """the same basis, up to a relative tolerance on the basis vectors"""
    tol = referenceFrame.tolAligned if tol is None else tol
    return bool(np.all(np.linalg.norm(self._basis - other._basis, axis=1) / np.linalg.norm(self._basis, axis=1) < tol))

  def transformationMatrix(self, other):
    """the transition matrix taking coordinates in this frame to coordinates in the other,
    on normalised bases: with the unit axes as rows, inv(B) A is the rotation between the
    two settings of one lattice"""
    A = unitRows(self._basis)
    B = unitRows(other._basis)
    return np.linalg.inv(B) @ A

  def isCompatible(self, other, tol=None):
    """whether a transition between the frames exists: the matrix is close to orthogonal"""
    tol = referenceFrame.tolCompatible if tol is None else tol
    M = self.transformationMatrix(other)
    MM = M @ M.T
    return bool(np.linalg.norm(MM - np.eye(3)) / np.linalg.norm(MM) < tol), M

  def isclose(self, other):
    """whether two frames may be treated as one: the same class of group and aligned bases,
    and for crystal frames the same mineral (MATLAB's eqTol)"""
    if self is other:
      return True
    if type(self) is not type(other):
      return False
    return self._isclosePair(other)

  def sim(self, other):
    """whether two frames describe the same symmetry, without requiring aligned bases"""
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

  @staticmethod
  def intern(fr):
    """the session instance of a frame: the registered one when it agrees by key, otherwise
    this one, registered and sealed from now on"""
    with SESSION.lock:
      for r in SESSION.frames:
        if _sameEntity(r, fr):
          # a centring stated now is what a frame registered without one lacked
          if fr.__dict__.get("_centering") and not r.__dict__.get("_centering"):
            for sib in SESSION.frames:
              if _sameBasis(sib, r):
                sib.__dict__["_centering"] = fr.__dict__["_centering"]
          return r
      fr._sealed = True
      SESSION.frames.append(fr)
      return fr

  @staticmethod
  def byName(name, fr=None):
    """the specimen frame registered under a name; with a frame, registers it under that name"""
    if fr is not None:
      SESSION.namedFrames[name] = fr
    return SESSION.namedFrames.get(name)

  @staticmethod
  def reset():
    """empties the frame register, the named frames and the default frame of the session"""
    SESSION.frames.clear()
    SESSION.namedFrames.clear()
    SESSION.siblings.clear()
    SESSION.defaultFrame = None

  # ------------------------------------------------------------------ display

  def conventionChar(self, pC=None):
    """the convention in the frame's own axes names, e.g. RD↑→TD, or the pictogram in x, y, z"""
    pC = self.how2plot if pC is None else pC
    if pC is None:
      return ""
    if self._axesNames == ("X", "Y", "Z"):
      return pC.char()
    dirs = vector3d(unitRows(self._basis))
    iN, ud = _matchSigned(pC.north, dirs)
    iE, lr = _matchSigned(pC.east, dirs)
    if iN is None or iE is None:
      return pC.char()
    a = ARROWS_UTF8 if getMTEXpref("UTF8Output") else ARROWS_ASCII
    northPart = self._axesNames[iN] + a[2 if ud else 3]
    return northPart + a[1] + self._axesNames[iE] if lr else self._axesNames[iE] + a[0] + northPart

  def char(self):
    """the name of the frame, or its class name"""
    return self.name if self.name else type(self).__name__

  def __str__(self):
    return self.char()

  def __repr__(self):
    pairs = []
    if self.numSym() > 1:
      pairs += [("symmetry", self.sym.char()), ("elements", self.numSym())]
    if np.max(np.abs(self._basis - CANONICAL_BASIS)) > 1e-10:
      pairs += [(n, "(" + ", ".join(numString(x) for x in b) + ")") for n, b in zip(self._axesNames, self._basis)]
    return display(header(type(self).__name__, self.name, self.conventionChar()), pairs)

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..plotting.sphericalplot import plotSymmetry as plot, plotHKL


# ------------------------------------------------------------------ helpers

def _basisArray(v):
  if isinstance(v, vector3d):
    return v.data.reshape(3, 3).copy()
  return np.asarray(v, dtype=float).reshape(3, 3).copy()


# which of the directions a vector is, and with which sign
def _matchSigned(v, dirs):
  for sign, ud in ((1, True), (-1, False)):
    hit = np.flatnonzero(v.isclose(sign * dirs))
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


# the trivial group, built once: a group is not changed once built
_TRIVIAL = []


def _trivialGroup():
  if not _TRIVIAL:
    _TRIVIAL.append(symmetry())
  return _TRIVIAL[0]


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
            and a.alignment() == b.alignment() and not _centeringsDisagree(a, b))
  return a.isAligned(b)


# ------------------------------------------------------------------ crystal frames

# the centrings of a lattice and a primitive cell of each, its rows in the conventional axes
CENTERINGS = ("P", "A", "B", "C", "I", "F", "R")
CENTERING_BASES = {
  "P": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
  "A": ((1, 0, 0), (0, 1 / 2, 1 / 2), (0, -1 / 2, 1 / 2)),
  "B": ((1 / 2, 0, 1 / 2), (0, 1, 0), (-1 / 2, 0, 1 / 2)),
  "C": ((1 / 2, 1 / 2, 0), (-1 / 2, 1 / 2, 0), (0, 0, 1)),
  "I": ((-1 / 2, 1 / 2, 1 / 2), (1 / 2, -1 / 2, 1 / 2), (1 / 2, 1 / 2, -1 / 2)),
  "F": ((0, 1 / 2, 1 / 2), (1 / 2, 0, 1 / 2), (1 / 2, 1 / 2, 0)),
  "R": ((2 / 3, 1 / 3, 1 / 3), (-1 / 3, 1 / 3, 1 / 3), (-1 / 3, -2 / 3, 1 / 3)),
}


# two frames whose centrings are both stated and differ are two lattices; an unstated one agrees
def _centeringsDisagree(a, b):
  ca, cb = a.__dict__.get("_centering"), b.__dict__.get("_centering")
  return bool(ca and cb and ca != cb)


# whether a string is a Hermann-Mauguin space group symbol, 'Fm-3m', 'P6_3/mmc', 'I 4/m m m'
def _isSpaceGroupSymbol(symbol):
  try:
    hm2point(symbol.strip())
  except ValueError:
    return False
  return True

class crystalFrame(referenceFrame):
  """the reference frame glued to the lattice basis of a phase

    crystalFrame('m-3m', [4.05, 4.05, 4.05], mineral='Aluminium', color='blue')
    crystalFrame('2/m', [8.6, 13, 7.2], [90, 116, 90]*degree, 'X||a', mineral='orthoclase')
    crystalFrame([1, 2, 3], [70, 80, 120]*degree, 'Z||a*')     by the cell alone, no group
    crystalFrame(axes), crystalFrame(axes, name='Forsterite')  by the axes themselves
    crystalFrame(rot)                                           the group generated by rotations
    crystalFrame(otherFrame)                                    the trivial group carrying it
    crystalFrame('I4/mmm', [3.3, 3.3, 5.8])                     by the space group, the centring kept

  Positional strings are alignments, 'X||a', 'Z||c*', 'EDAX'; everything else is a keyword.
  A space group symbol gives the point group and the centring of the lattice, P, A, B, C, I, F
  or R, which the keyword centering= also sets.
  """

  def __init__(self, *args: PointGroupName | AlignmentName | Any, mineral=None, name=None, color: ColorName | Any = None,
               alignment: AlignmentName | Any = (), rotAxes=None, pointId=None, LaueId=None, laueId=None, spaceId=None,
               how2plot: ConventionName | Any = None, intern=True, density=None, axesNames=("a", "b", "c"),
               check=True, centering=None):
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
        # a space group symbol names the centring of the lattice by its first letter
        if centering is None and not isPointGroupName(symbol) and _isSpaceGroupSymbol(symbol):
          centering = symbol.strip()[0]
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
    if centering is not None:
      if centering not in CENTERINGS:
        raise ValueError(f"the centring of a lattice is one of {', '.join(CENTERINGS)}, not {centering!r}")
      self._centering = centering
    if self.how2plot is None:
      self.how2plot = _crystalConvention(self)
    self._registered = None if not intern else _registerKeepingColor(self, color)

  @classmethod
  def load(cls, fname):
    """the constructor answers with the registered instance
    crystalFrame.load('Al-Aluminum.cif'): the frame a CIF file describes, read by the
    interfaces layer above this one"""
    from ..io.cif import loadCIF
    return loadCIF(fname)

  def __new__(cls, *args, **kwargs):
    return object.__new__(cls)

  @classmethod
  def byElements(cls, rot, **kwargs):
    """the crystal frame of the group generated by the rotations

      cs = crystalFrame.byElements(rot, mineral='...')"""
    return crystalFrame(rot, **kwargs)

  # ------------------------------------------------------------------ the lattice

  @property
  def centering(self):
    """the centring of the lattice: P primitive, A, B, C one face, I the body, F every face, R
    rhombohedral in the hexagonal cell; P where neither a space group nor centering= stated it"""
    return self.__dict__.get("_centering") or "P"

  def primitiveAxes(self):
    """the axes of a primitive cell of the lattice, the conventional axes where it is primitive;
    rhombohedral axes are primitive already"""
    T = CENTERING_BASES[self.centering]
    if self.centering == "R" and abs(self.abg[2] - 2 * np.pi / 3) > 1e-6:
      T = CENTERING_BASES["P"]
    return vector3d(np.asarray(T) @ _basisArray(self.axes))

  @property
  def axes(self):
    """the crystal axes a, b, c in Euclidean coordinates, the basis"""
    return self.basis

  @property
  def axesDual(self):
    """the reciprocal axes a*, b*, c*, MATLAB's axesDual"""
    from .crystalaxes import reciprocalAxes
    return vector3d(reciprocalAxes(self._basis))

  @property
  def abc(self):
    """the axis lengths a, b, c"""
    return np.linalg.norm(self._basis, axis=1)

  @property
  def abg(self):
    """the axis angles alpha, beta, gamma in radians"""
    b = vector3d(self._basis)
    return np.array([b[1].angle(b[2]), b[2].angle(b[0]), b[0].angle(b[1])])

  @property
  def alpha(self):
    """the angle alpha between the b and c axes, in radians"""
    return self.abg[0]

  @property
  def beta(self):
    """the angle beta between the c and a axes, in radians"""
    return self.abg[1]

  @property
  def gamma(self):
    """the angle gamma between the a and b axes, in radians"""
    return self.abg[2]

  # a lattice axis as a Miller direction in this frame: [100] or (100) of the frame
  def _axisMiller(self, xyz, style):
    from .miller import Miller
    return Miller(vector3d(xyz), self, dispStyle=style)

  @property
  def aAxis(self):
    """the a axis as the Miller direction [100] of the frame"""
    return self._axisMiller(self._basis[0], 'uvw')

  @property
  def bAxis(self):
    """the b axis as the Miller direction [010] of the frame"""
    return self._axisMiller(self._basis[1], 'uvw')

  @property
  def cAxis(self):
    """the c axis as the Miller direction [001] of the frame"""
    return self._axisMiller(self._basis[2], 'uvw')

  @property
  def aAxisRec(self):
    """the reciprocal axis a* as the Miller plane normal (100) of the frame"""
    return self._axisMiller(reciprocalAxes(self._basis)[0], 'hkl')

  @property
  def bAxisRec(self):
    """the reciprocal axis b* as the Miller plane normal (010) of the frame"""
    return self._axisMiller(reciprocalAxes(self._basis)[1], 'hkl')

  @property
  def cAxisRec(self):
    """the reciprocal axis c* as the Miller plane normal (001) of the frame"""
    return self._axisMiller(reciprocalAxes(self._basis)[2], 'hkl')

  def basisDual(self):
    """the reciprocal axes a*, b*, c* as vectors"""
    return vector3d(reciprocalAxes(self._basis))

  def alignment(self):
    """the alignment of the Euclidean axes to the crystal axes as strings, e.g. ['X||a', 'Z||c*'];
    empty for an orthogonal lattice"""
    return _alignment(self._basis, self._axesNames)

  def transformationMatrix(self, other):
    """between crystal frames the axes are first re-paired by nearest length when the cells
    disagree, which is how the monoclinic settings 121 and 112 relate"""
    A = unitRows(self._basis)
    B = unitRows(other._basis)
    if isinstance(other, crystalFrame):
      la, lb = self.abc, other.abc
      if not np.all(np.abs(la - lb) / np.sum(la) < 0.01):
        A = A[[int(np.argmin(np.abs(la - lb[k]))) for k in range(3)]]
    return np.linalg.inv(B) @ A

  def _isclosePair(self, other):
    return self.name.lower() == other.name.lower() and self._sameGroup(other) and self.isAligned(other)

  def _simPair(self, other):
    return (self.name.lower() == other.name.lower() and self._sameGroup(other)
            and bool(np.all(np.abs(self.abc - other.abc) / np.max(self.abc) < 1e-2)) and bool(np.all(np.abs(self.abg - other.abg) < 1e-2)))

  def conventionChar(self, pC=None):
    """the convention in crystal directions, e.g. ⊙c→a, with the reciprocal axes named too"""
    pC = self.how2plot if pC is None else pC
    if pC is None:
      return ""
    dirs = vector3d(np.concatenate([unitRows(self._basis), unitRows(reciprocalAxes(self._basis))]))
    names = list(self._axesNames) + [n + "*" for n in self._axesNames]
    out, east = _matchAxis(pC.outOfScreen, dirs, names), _matchAxis(pC.east, dirs, names)
    if not out or not east:
      return pC.char()
    a = ARROWS_UTF8 if getMTEXpref("UTF8Output") else ARROWS_ASCII
    return a[5] + out + a[1] + east

  def char(self):
    """the mineral name of the frame, or the symbol of its point group"""
    return self.name if self.name else self.sym.char()

  def __repr__(self):
    # the angles only where the lattice does not imply them, as MATLAB shows them
    angles = None
    if np.max(np.abs(np.asarray(self.abg) - self.lattice.defaultAngles)) > 1e-6:
      angles = ", ".join(f"{x * 180 / np.pi:.4g}°" for x in self.abg)
    al = self.alignment()
    return display(header("crystalFrame", self.conventionChar()), [
      ("mineral", self.name or None), ("color", self.color), ("symmetry", self.sym.char()), ("elements", self.numSym()),
      ("centering", None if self.centering == "P" else self.centering),
      ("a, b, c", ", ".join(f"{x:.4g}" for x in self.abc)), ("alpha, beta, gamma", angles),
      ("reference frame", ", ".join(al) if al else None)])


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

  def __init__(self, name: PointGroupName | Any = None, *args, axesNames=("X", "Y", "Z"), how2plot: ConventionName | Any = None,
               intern=True, basis=None):
    if isinstance(name, str) and how2plot is None and args and isinstance(args[0], plottingConvention):
      how2plot = args[0]
    if isinstance(name, str) and not isPointGroupName(name):
      super().__init__(basis, name, axesNames, how2plot, None, None)
      self._registered = None if not intern else self
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

  @staticmethod
  def named(name, axesNames, seed=None):
    """the interned session instance of a named frame, constructed on first use with its seed
    convention and kept in the session's name map"""
    with SESSION.lock:
      fr = referenceFrame.byName(name)
      if fr is None:
        fr = referenceFrame.intern(specimenFrame(name, axesNames=axesNames, how2plot=seed))
        referenceFrame.byName(name, fr)
      return fr

  @classproperty
  def specimen(cls):
    """the generic specimen frame X, Y, Z - where an import lands and the session default"""
    return specimenFrame.named("specimen", ("X", "Y", "Z"))

  @classproperty
  def measurement(cls):
    """the sample surface frame of an Oxford instrument, drawn as its SEM images are"""
    return specimenFrame.named("measurement", ("X1", "Y1", "Z1"), plottingConvention.ij)

  @classproperty
  def rolling(cls):
    """rolling, transverse and normal direction, RD north and TD west"""
    return specimenFrame.named("rolling", ("RD", "TD", "ND"), plottingConvention("y←↑x"))

  @classproperty
  def geological(cls):
    """north, east, down of structural geology"""
    return specimenFrame.named("geological", ("N", "E", "D"))

  @classproperty
  def default(cls):
    """the frame that supplies the session default convention"""
    if SESSION.defaultFrame is None:
      SESSION.defaultFrame = specimenFrame.specimen
    if SESSION.defaultFrame.how2plot is None:
      SESSION.defaultFrame.how2plot = plottingConvention.ij
    return SESSION.defaultFrame

  def makeDefault(self):
    """makes this frame the session's default frame, the one supplying the default plotting
    convention"""
    SESSION.defaultFrame = self

  @staticmethod
  def frameFor(pC):
    """the session frame carrying a convention: the default when it already does, otherwise a
    fork of it with that convention, registered"""
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

  def _isclosePair(self, other):
    return self.Laue().id == other.Laue().id and self.isAligned(other)

  def _simPair(self, other):
    return self.Laue().id == other.Laue().id

  def char(self):
    """the name or the convention, the point group after it when there is one: y↑→x (mmm)"""
    base = self.name if self.name else self.conventionChar()
    g = self.sym.char()
    return base if g in ("1", "") else f"{base} ({g})"


def withGroup(fr, group):
  """a frame in a group: its sibling, which the register owns; a specimen symmetry above
  orthorhombic is not a sample symmetry"""
  group = group[0] if isinstance(group, (tuple, list)) and len(group) == 1 else group
  if isinstance(group, (tuple, list)) and len(group) == 0:
    return fr
  s = group if isinstance(group, symmetry) else symmetry(group)
  if s.id > 16:
    warnings.warn(f"{s.pointGroup} is not a suitable specimen symmetry")
  return fr.sibling(s)


# the constructors answer with the registered instance: the object built is discarded
def _answerRegistered(cls):
  # the signature help() and the completion read is that of __init__
  @functools.wraps(cls.__init__, assigned=())
  def __new__(klass, *args, **kwargs):
    obj = object.__new__(klass)
    obj.__init__(*args, **kwargs)
    return obj._registered if obj._registered is not None else obj

  cls.__new__ = staticmethod(__new__)
  cls.__init__ = _once(cls.__init__)
  return cls


# an __init__ that runs only when __new__ has not already run it, with the signature and help of
# the one it wraps
def _once(init):
  @functools.wraps(init)
  def __init__(self, *args, **kwargs):
    if getattr(self, "_registered", "unset") == "unset":
      init(self, *args, **kwargs)
  return __init__


crystalFrame = _answerRegistered(crystalFrame)
specimenFrame = _answerRegistered(specimenFrame)


# ------------------------------------------------------------------ not indexed

class notIndexedFrame(referenceFrame):
  """the frame of the measurements that could not be indexed, a phase of a map without basis,
  group or colour"""

  def __init__(self, name="notIndexed", color: ColorName | Any = None):
    super().__init__(None, name, ("X", "Y", "Z"), None, None, np.array([np.nan, np.nan, np.nan]) if color is None else color)

  def _isclosePair(self, other):
    return self.name.lower() == other.name.lower()

  _simPair = _isclosePair

  def sibling(self, s):
    """the frame itself: a not indexed frame has no group to change"""
    return self

  def __repr__(self):
    return aligned(header("notIndexedFrame", self.name))


SESSION.mintDefaultFrame = lambda: specimenFrame.specimen


# ------------------------------------------------------------------ the join rule (ADR 0009)

def joinFrames(a, b):
  """the frame two framed operands share: the same handle, the other side when one is
  frame-free, the sibling carrying the intersection of the groups when they differ only in
  their group, and an error otherwise"""
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
  """the twelve alignment pairs ['X||a', 'Z||b*'], ... that construct on every lattice and between
  them reach every distinct frame of a lattice"""
  return [list(s) for s in ALIGNMENT_SETUPS]


# the setup a crystal frame is reported in: the frame rebuilt in each of the twelve and the
# nearest kept, since the axes it states say nothing where they coincide, X||a on an
# orthogonal lattice or Z||c on a hexagonal one being every setup at once
_CLOSEST_ALIGNMENT = {}


def closestAlignment(CS):
  """the alignment setup a crystal frame is reported in: of the twelve setups the one whose frame
  lies nearest its basis"""
  key = (CS.pointGroup, np.round(CS._basis, 9).tobytes())
  if key in _CLOSEST_ALIGNMENT:
    kept = _CLOSEST_ALIGNMENT[key]
    return None if kept is None else list(kept)
  n = unitRows(CS._basis)
  best, at = None, None
  for setup in ALIGNMENT_SETUPS:
    try:
      fr = crystalFrame(CS.pointGroup, CS.abc, CS.abg, *setup, intern=False, check=False)
    except ValueError:
      continue
    d = float(np.max(np.abs(unitRows(fr._basis) - n)))
    if best is None or d < best - 1e-9:
      best, at = d, list(setup)
  _CLOSEST_ALIGNMENT[key] = tuple(at) if at is not None else None
  return at


__all__ = ["referenceFrame", "crystalFrame", "specimenFrame", "notIndexedFrame", "alignmentSetups", "closestAlignment"]
