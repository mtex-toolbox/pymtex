# point groups, defined by their Hermann-Mauguin symbol (MATLAB @symmetry)
#
# A point group is its Hermann-Mauguin symbol read against the symmetry directions of its
# lattice system, as the International Tables define it: each position of the symbol names
# what sits on one direction - an n-fold rotation `n`, an n-fold rotoinversion `-n`, a
# mirror `m` perpendicular to it, or both `n/m` - and the group is the closure of those
# generators. Everything else - crystal system, Laue group, proper subgroup, orders and
# axes - is derived from the elements. The name table only names.
#
# A group says nothing about where it acts: the axes its elements are written in belong to
# the reference frame that carries the group (MATLAB ADR 0008).
#
#   symmetry('m-3m')                 in the canonical axes x, y, z
#   symmetry('321', axes)            in the axes a, b, c of a lattice
#   symmetry(rot)                    the closure of a list of elements
#   symmetry(spaceId=225), symmetry(laueId=7), symmetry(pointId=45)
#
# The element order is the closure order, not MTEX's: nothing may index elements by
# position, see docs/plans/geometry.md.
from __future__ import annotations

from functools import cached_property
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from .data.pointgroupnames import (LAUE_CLASSES, POINT_GROUP_NAMES, SPACE_GROUPS, findPointGroup, hm2point,
                                   isPointGroupName, monoclinicSetting, pointGroupId)
from .latticetype import latticeType
from .quaternion import orthogonalTransform, reflection, rotation
from .session import getMTEXpref
from .symreduce import _firstOfClusters
from .units import degree
from .vector3d import classproperty, vector3d, xvector, yvector, zvector, asVector3d

if TYPE_CHECKING:
  from .vocabulary import PointGroupName

CANONICAL_AXES = np.eye(3)


# ------------------------------------------------------------------ elements up to tolerance

# the elements of a list that are not repeats of an earlier one, up to an angle and with
# the same parity; pairwise, since a group is small and rounding keys can split a pair
def _uniqueElements(o, tol=1e-6):
  e = o.data.reshape(-1, 5)
  return orthogonalTransform(e[_firstOfClusters(_sameElement(e, e, tol))])


# the pairs of elements, rows of quaternion and parity, that agree up to an angle and in parity
def _sameElement(ea, eb, tol):
  return (np.abs(ea[:, :4] @ eb[:, :4].T) > 1 - tol) & (ea[:, 4, None] == eb[None, :, 4])


# whether each element of a is in the list b
def _isin(a, b, tol=1e-6):
  return np.any(_sameElement(a.data.reshape(-1, 5), b.data.reshape(-1, 5), tol), axis=1)


# representatives r of the cosets r * d (left) or d * r of the subgroup d that cover the group
# G, the identity first and then the first element not yet reached
def _cosetRepresentatives(G, d, left):
  reps = orthogonalTransform.id.reshape(1)
  while True:
    products = reps[:, None] * d[None, :] if left else d[:, None] * reps[None, :]
    reached = _isin(G, products.flatten())
    if np.all(reached):
      return reps
    reps = orthogonalTransform.cat(reps, G[np.argmin(reached)])


def _sameElements(a, b, tol=1e-6):
  return a.size == b.size and bool(np.all(_isin(a, b, tol)))


def closure(gens, tol=1e-6):
  """the closure of a list of elements under multiplication: the set grows by its products
  with the generators until it stops, which for a finite group is the group"""
  if not isinstance(gens, orthogonalTransform):
    gens = orthogonalTransform.cat(*gens) if gens else orthogonalTransform()
  gens = _uniqueElements(orthogonalTransform.cat(orthogonalTransform.id, gens), tol)
  els = gens
  while True:
    grown = _uniqueElements(orthogonalTransform.cat(els, (els[:, None] * gens[None, :]).flatten()), tol)
    if grown.size == els.size:
      return grown
    if grown.size >= 200:
      raise ValueError("no finite symmetry group could be built")
    els = grown


# ------------------------------------------------------------------ the symbol as a recipe

def _symbolPositions(symbol):
  return re.findall(r"-?\d(?:/m)?|m", symbol)


def _positionOrder(p):
  return 2 if p == "m" else int(re.search(r"\d", p).group())


# the crystal system a symbol belongs to, which fixes the symmetry directions
def _symbolSystem(positions):
  orders = [_positionOrder(p) for p in positions]
  if len(positions) >= 2 and orders[1] == 3:
    return latticeType.cubic
  if orders[0] == 6:
    return latticeType.hexagonal
  if orders[0] == 3:
    return latticeType.trigonal
  if orders[0] == 4:
    return latticeType.tetragonal
  n = sum(p != "1" for p in positions)
  return latticeType.orthorhombic if n >= 3 else latticeType.monoclinic if n == 1 else latticeType.triclinic


# the symmetry directions of a lattice system by position of the symbol: the tertiary
# direction of the tetragonal and hexagonal systems is the secondary one turned about the
# primary, 45 degree for a square and -30 degree for a hexagonal net (a - b in hexagonal
# axes), so it is right in any axes
def _symmetryDirections(system, a, b, c):
  def turned(theta):
    return rotation.byAxisAngle(c, theta) * a
  if system is latticeType.cubic:
    return (a, a + b + c, a + b)
  if system is latticeType.tetragonal:
    return (c, a, turned(np.pi / 4))
  if system.isTriHex:
    return (c, a, turned(-np.pi / 6))
  return (a, b, c)


# the generators one position puts on its direction
def _positionGenerators(p, d):
  if p == "1":
    return []
  if p == "-1":
    return [rotation.inversion]
  if p == "m":
    return [reflection(d)]
  n = _positionOrder(p)
  rot = orthogonalTransform(rotation.byAxisAngle(d, 2 * np.pi / n))
  if p.startswith("-"):
    return [rotation.inversion * rot]
  return [rot, reflection(d)] if p.endswith("/m") else [rot]


def _generators(symbol, axes):
  a, b, c = (vector3d(axes[k]) for k in range(3))
  if symbol in ("532", "-5-32/m"):
    # no lattice has a five-fold axis: MTEX's icosahedron, one five-fold and three two-folds
    gens = [orthogonalTransform(rotation.byAxisAngle(c + 2 / (1 + np.sqrt(5)) * a, 2 * np.pi / 5))]
    gens += [orthogonalTransform(rotation.byAxisAngle(v, np.pi)) for v in (a, b, c)]
    return gens + ([rotation.inversion] if symbol != "532" else [])
  positions = _symbolPositions(symbol)
  dirs = _symmetryDirections(_symbolSystem(positions), a, b, c)
  return [g for p, d in zip(positions, dirs) for g in _positionGenerators(p, d)]


def _asAxes(axes):
  if axes is None:
    return CANONICAL_AXES.copy()
  if isinstance(axes, vector3d):
    return axes.data.reshape(3, 3).copy()
  return np.asarray(axes, dtype=float).reshape(3, 3).copy()


# ------------------------------------------------------------------ the class

class symmetry:
  """a point group: the closure of the generators its Hermann-Mauguin symbol names, written in
  given axes

    pg = symmetry('m-3m')                              # by its international symbol
    pg = symmetry('D2')                                # by its Schoenflies symbol or an alias
    pg = symmetry(spaceId=225)                         # the point group of a space group
    pg = symmetry(pointId=45)                          # by its number in symmetry.pointGroups
    pg = symmetry('2', abg=[90*degree, 100*degree, 90*degree])  # the monoclinic setting from the lattice angles
    pg = symmetry(rotation.byAxisAngle(zvector, 90*degree))  # the group the elements generate

  A group says nothing about where it acts: whether it describes a crystal or a specimen, and
  in which coordinate system, is the business of the frame carrying it, crystalFrame or
  specimenFrame, which builds its group from the same symbol. The elements are built from the
  symbol in the order of its generators, as orthogonal transforms, proper rotations and
  rotoinversions; nothing refers to an element by its position.

  A symbol of a monoclinic group, '2', 'm' or '2/m', names no unique axis; with the lattice
  angles abg the unique axis is the one whose angle is not 90 degree, as MATLAB.

  Input
    name - str, the Hermann-Mauguin or Schoenflies symbol, a lattice name or an alias such as
        'orthotropic'; or rotation or orthogonalTransform, elements that generate the group;
        or a symmetry to copy; the triclinic group '1' when omitted
    axes - vector3d or an array (3, 3), the axes a, b, c the elements are written along;
        x, y, z when omitted
  Output
    pg - symmetry, with pg.pointGroup its symbol, pg.rot its elements, pg.id its number
  Keywords
    pointId - the number of the group in symmetry.pointGroups, 1 to 47
    laueId - the number of a Laue class, 1 to 11
    spaceId - the number of a space group, 1 to 230, which gives its point group
    abg - the lattice angles alpha, beta, gamma in radians, which choose the unique axis of
        a monoclinic group
  See also
    crystalFrame, specimenFrame, symmetry.fundamentalSector, symmetry.fundamentalRegion,
    CrystalSymmetries"""

  def __init__(self, name: PointGroupName | Any = None, axes=None, *, pointId=None, laueId=None, spaceId=None, abg=None):
    if isinstance(name, symmetry):
      self._symbol, self.rot, self.axes = name._symbol, name.rot, name.axes
      return
    if isinstance(name, (rotation, orthogonalTransform)):
      self.axes = _asAxes(axes)
      self.rot = closure(orthogonalTransform(name), tol=1e-2)
      self._symbol = identifyGroup(self.rot, self.axes)
      return
    if pointId is not None:
      name = POINT_GROUP_NAMES[pointId - 1][0]
    elif laueId is not None:
      name = LAUE_CLASSES[laueId - 1]
    elif spaceId is not None:
      name = spaceGroupToPointGroup(spaceId)
    elif name is None:
      name = "1"
    if abg is not None:
      name = monoclinicSetting(name, abg)
    self._symbol = findPointGroup(name)
    self.axes = _asAxes(axes)
    self.rot = closure(_generators(self._symbol, self.axes))

  @classmethod
  def _fromElements(cls, els, axes, symbol=None):
    obj = object.__new__(cls)
    obj.rot, obj.axes = els, axes
    obj._symbol = identifyGroup(els, axes) if symbol is None else symbol
    return obj

  @classproperty
  def pointGroups(cls):
    """the table of the 32 point groups, the Hermann-Mauguin symbols with their Schoenflies names"""
    return POINT_GROUP_NAMES

  # ------------------------------------------------------------------ names

  @property
  def pointGroup(self):
    """the Hermann-Mauguin symbol of the group, 'unknown' when its elements name no group"""
    return self._symbol if self._symbol else "unknown"

  @property
  def name(self):
    """the Hermann-Mauguin symbol of the group, as pointGroup"""
    return self.pointGroup

  @property
  def id(self):
    """the number of the group in the table of the point groups"""
    return pointGroupId(self._symbol)

  @property
  def Schoenflies(self):
    """the Schoenflies symbol of the group, 'unknown' when its elements name no group"""
    k = self.id
    return POINT_GROUP_NAMES[k - 1][1] if k else "unknown"

  def char(self, utf8=None):
    """the symbol with a bar over every rotoinversion axis under utf8 output

      s = cs.sym.char()                             # '3̅m1' or '-3m1', as the preference says
      s = cs.sym.char(utf8=False)                   # '-3m1'

    Input
      pg - symmetry
    Output
      s - str, the Hermann-Mauguin symbol
    Keywords
      utf8 - a combining bar instead of the minus sign; by default the preference
          'UTF8Output'
    See also
      symmetry.pointGroup, symmetry.Schoenflies"""
    utf8 = getMTEXpref("UTF8Output") if utf8 is None else utf8
    return re.sub(r"-(\d)", "\\1\u0305", self.pointGroup) if utf8 else self.pointGroup

  # ------------------------------------------------------------------ the elements

  def __len__(self):
    return self.rot.size

  def numSym(self):
    """the number of elements of the group"""
    return self.rot.size

  def numProper(self):
    """the number of proper rotations of the group"""
    return int(np.sum(~self.rot.isImproper))

  # a group is not changed once built, so what it is is read once, on first use
  @property
  def isProper(self):
    """whether the group holds proper rotations only"""
    if "_isProper" not in self.__dict__:
      self._isProper = not np.any(self.rot.isImproper)
    return self._isProper

  @property
  def isLaue(self):
    """whether the group contains the inversion"""
    if "_isLaue" not in self.__dict__:
      self._isLaue = rotation.inversion in self
    return self._isLaue

  # an element is in the group when the group holds it up to a small angle with its parity
  def __contains__(self, e):
    e = orthogonalTransform(e) if isinstance(e, rotation) else e
    return bool(np.all(_isin(e, self.rot)))

  def rotations(self):
    """return the proper rotations already contained in the point group.

      rotations = cs.sym.rotations()

    The result is a rotation container. Operations with determinant -1 are omitted;
    this method returns no axis or multiplicity arrays. For those use elements.

    See also
      symmetry.elements, symmetry.properSubGroup"""
    return self.rot[~self.rot.isImproper].rot

  # the order of the rotation part of every element, 1 for the identity
  def _orders(self):
    omega = self.rot.angle()
    return np.where(omega < 1e-4, 1, np.round(2 * np.pi / np.where(omega < 1e-4, 1, omega))).astype(int)

  def elements(self, multiplicity=None):
    """the rotational axes and their multiplicities, both signs, symmetry/elements

      axes, mult = cs.sym.elements()                # every axis with its order
      axes = cs.sym.elements(multiplicity=2)        # the two-fold axes alone

    Every axis of a rotation or rotoinversion of order above one, with the highest order
    about it, and its antipode with the same order, as MATLAB lists them.

    Input
      pg - symmetry
    Output
      axes - vector3d, antipodal
      mult - the order of every axis
    Keywords
      multiplicity - return the axes of this order alone
    See also
      symmetry.nfold, symmetry.rotations"""
    order = self._orders()
    sel = order > 1
    axes = self.rot[sel].axis()
    mult = order[sel]
    if axes.size == 0:
      return vector3d(), np.zeros(0, dtype=int)
    axes.antipodal = True
    u, iu = axes.unique(tol=1e-3, return_inverse=True)
    mult = np.array([np.max(mult[iu == k]) for k in range(u.size)])
    axes = vector3d.cat(u, -u)
    mult = np.concatenate([mult, mult])
    if multiplicity is not None:
      return axes[mult == multiplicity]
    return axes, mult

  # the highest order of a rotation axis satisfying the predicate, 1 if there is none
  def _maxOrder(self, pred=None):
    order = self._orders()
    sel = order > 1 if pred is None else (order > 1) & pred(self.rot)
    return int(np.max(order[sel])) if np.any(sel) else 1

  def nfold(self, axis=None):
    """the maximal n-fold of the group, and about given axes, symmetry/nfold

      n = cs.sym.nfold()                            # the highest order of any axis
      n = cs.sym.nfold(vector3d(0, 0, 1))           # about z, 1 when it is no axis

    Input
      pg - symmetry
      axis - vector3d, the axes asked about
    Output
      n - an integer, or one per axis
    See also
      symmetry.elements, symmetry.multiplicityZ"""
    if axis is None:
      return self._maxOrder()
    axis = asVector3d(axis)
    return np.array([self._maxOrder(lambda r: r.axis().angle(v, antipodal=True) < 1e-5) for v in axis.flatten()]).reshape(axis.shape)

  @property
  def multiplicityZ(self):
    """the highest order of a rotation axis of the group along z, 1 if there is none"""
    return self._maxOrder(lambda r: np.abs(r.axis().dot(zvector)) > 1 - 1e-8)

  @property
  def multiplicityPerpZ(self):
    """the highest order of a rotation axis of the group perpendicular to z, 1 if there is none"""
    return self._maxOrder(lambda r: np.abs(r.axis().dot(zvector)) < 1e-4)

  @cached_property
  def lattice(self):
    """the crystal system, from the axes the group has"""
    axes, mult = self.elements()
    mult = mult[:len(mult) // 2]
    if 5 in mult:
      return latticeType.icosahedral
    if np.sum(mult == 3) >= 4:
      return latticeType.cubic
    if 6 in mult:
      return latticeType.hexagonal
    if 3 in mult:
      return latticeType.trigonal
    if 4 in mult:
      return latticeType.tetragonal
    n = np.sum(mult == 2)
    return latticeType.orthorhombic if n >= 2 else latticeType.monoclinic if n == 1 else latticeType.triclinic

  # ------------------------------------------------------------------ derived groups

  def _withParity(self, improper):
    return orthogonalTransform(self.rot.rot, improper)

  def Laue(self):
    """the smallest Laue group containing this one: every element once proper and once improper"""
    if self.isLaue:
      return self
    return symmetry._fromElements(orthogonalTransform.cat(self._withParity(False), self._withParity(True)), self.axes)

  def LaueName(self):
    """the Hermann-Mauguin symbol of the Laue group, the smallest one containing this group"""
    return self.Laue().pointGroup

  # a group is not changed once built, so the derived groups are built once, on first use
  def properGroup(self):
    """form a proper group from the rotational parts of all symmetry operations.

      rotations = group.properGroup()

    Parity is discarded and duplicate rotations are removed. The result is a symmetry
    object. This is the proper rotation group of the associated Laue class, which can
    be larger than the proper subgroup of the original group.

    See also
      symmetry.properSubGroup, symmetry.Laue"""
    if self.isProper:
      return self
    if self.__dict__.get("_properGroup") is None:
      self._properGroup = symmetry._fromElements(_uniqueElements(self._withParity(False)), self.axes)
    return self._properGroup

  def properSubGroup(self):
    """retain the symmetry operations whose determinant is +1.

      subgroup = group.properSubGroup()

    The result is a symmetry object. Improper operations are removed rather than
    converted to their rotational parts.

    See also
      symmetry.properGroup, symmetry.rotations"""
    if self.isProper:
      return self
    if self.__dict__.get("_properSubGroup") is None:
      self._properSubGroup = symmetry._fromElements(self.rot[~self.rot.isImproper], self.axes)
    return self._properSubGroup

  def disjoint(self, other):
    """return the intersection of two point groups in their current coordinate axes.

      common = group1.disjoint(group2)

    Only operations present in both groups are retained. The inputs are not changed.
    To generate a group containing both inputs, use union.

    See also
      symmetry.union, symmetry.factor"""
    if self == other:
      return self
    keep = _isin(self.rot, other.rot, 1e-4)
    if np.all(keep):
      return self
    if np.sum(keep) == other.rot.size:
      return other
    return symmetry._fromElements(self.rot[keep], self.axes)

  def union(self, other):
    """generate the smallest point group containing the operations of both inputs.

      combined = group1.union(group2)

    The operation list is closed under multiplication. This is a group union, rather
    than a concatenation of two lists; additional products can therefore appear.

    See also
      symmetry.disjoint, symmetry"""
    if other <= self:
      return self
    if self <= other:
      return other
    return symmetry(orthogonalTransform.cat(self.rot, other.rot), self.axes)

  def factor(self, other):
    """factor two groups using their common symmetry operations.

      left, common, right = group1.factor(group2)

    The outputs are orthogonalTransform containers satisfying `group1 = left*common`
    and `group2 = common*right` as sets of products. The outer factors contain coset
    representatives; common contains the operations shared by both groups.

    See also
      symmetry.disjoint, symmetry.union"""
    d = self.rot[_isin(self.rot, other.rot)]
    if d.size == 1:
      return self.rot, d, other.rot
    return _cosetRepresentatives(self.rot, d, left=True), d, _cosetRepresentatives(other.rot, d, left=False)

  # ------------------------------------------------------------------ regions

  def fundamentalSector(self, center=None, antipodal=False, angle=None):
    """the fundamental sector of the group on the sphere, the region of the directions one
    symmetric copy of each falls into

      sR = pg.fundamentalSector()                        # the conventional sector
      sR = pg.fundamentalSector(antipodal=True)          # with the inversion added
      sR = pg.fundamentalSector(center=vector3d(1, 2, 3))  # the Dirichlet domain about a centre
      sR = pg.fundamentalSector(angle=55*degree)         # the rotation axes allowed at one rotation angle

    The sector is the conventional one, the sector an inverse pole figure shows, wherever it is
    a fundamental domain of the group, which it is for every crystal symmetry; for a group
    turned against its axes it is a Dirichlet domain. With a centre it is
    the Dirichlet domain of that centre, the directions nearer to it than to any of its
    symmetric copies.

    With an angle it is the sector of rotation axes the fundamental region of orientations
    allows at that rotation angle: small circles about the symmetry axes, of radius set by the
    angle and the fold of the axis; MATLAB intersects them with the conventional sector, here
    they stand alone.

    A frame's fundamentalSector places the sector by the frame's plotting convention; a group
    has no convention and places it on x, y, z.

    Input
      pg - symmetry
    Output
      sR - sphericalRegion
    Keywords
      center - vector3d, the centre of a Dirichlet domain
      antipodal - add the inversion to the group, the sector of axes
      angle - the rotation angle of misorientations, in radians, whose allowed axes are sought
    See also
      sphericalRegion, symmetry.fundamentalRegion, conventionalSector, dirichletSector, CrystalSymmetries"""
    from .sphericalregion import fundamentalSector
    return fundamentalSector(self, center=center, antipodal=antipodal, angle=angle)

  def conventionalSector(self, up=None, east=None):
    """the conventional sector of the group, the one MTEX draws, with up and east choosing its
    placement

      sR = cs.sym.conventionalSector()              # towards z
      sR = cs.sym.conventionalSector(up=-zvector)   # on the lower hemisphere

    The hemisphere towards up when some element sends up below it, a wedge about the c axis
    starting on a mirror plane, else on a two-fold axis, else at a, and the standard
    triangles of the cubic groups. east matters only where the sector has a choice of side:
    the monoclinic groups and the orthorhombic ones with a mirror. Called on the group it
    has no frame; frame.fundamentalSector() takes up and east from the frame's plotting
    convention.

    Input
      pg - symmetry
    Output
      sR - sphericalRegion
    Keywords
      up - vector3d, the direction out of the screen, z by default
      east - vector3d, the direction to the right, by default the a axis
    See also
      fundamentalSector, dirichletSector, tilesSphere"""
    from .sphericalregion import conventionalSector
    return conventionalSector(self, up=up, east=east)

  def fundamentalRegion(self, other=None, *, antipodal=False):
    """the fundamental region of orientations of this group and another one, the Dirichlet domain
    of the identity

      oR = cs.sym.fundamentalRegion()               # of orientations
      oR = cs.sym.fundamentalRegion(cs.sym)         # of misorientations
      oR = cs.sym.fundamentalRegion(cs.sym, antipodal=True)  # a misorientation and its inverse as one

    The orientations nearer the identity than any of their equivalents. The frames are
    built from the groups, so cs.fundamentalRegion() on the frame is the usual call.

    Input
      pg - symmetry
      other - symmetry, the second group, by default that of the default specimen frame
    Output
      oR - orientationRegion
    Keywords
      antipodal - identify every misorientation with its inverse
    See also
      fundamentalRegion, orientationRegion, maxAngle"""
    from .orientationregion import fundamentalRegion
    return fundamentalRegion(self, other, antipodal=antipodal)

  # ------------------------------------------------------------------ comparison

  # the same group: by symbol when both are named, by elements otherwise
  def __eq__(self, other):
    if not isinstance(other, symmetry):
      return NotImplemented
    if self._symbol and other._symbol:
      return self._symbol == other._symbol
    return _sameElements(self.rot, other.rot)

  def __ne__(self, other):
    return not self == other

  __hash__ = None

  # a subgroup
  def __le__(self, other):
    return bool(np.all(_isin(self.rot, other.rot)))

  def __ge__(self, other):
    return other <= self

  def __lt__(self, other):
    return self <= other and not other <= self

  def __gt__(self, other):
    return other < self

  # ------------------------------------------------------------------ display

  def __repr__(self):
    from .display import display, header
    return display(header("symmetry", self.char()), [("elements", self.numSym()), ("lattice", self.lattice),
                                                     ("Laue class", self.Laue().char())])

  def __str__(self):
    return self.char()


# ------------------------------------------------------------------ identification

# the (angle, parity) multiset of the elements: the same for every setting of a group
def _signature(els):
  return tuple(sorted(zip(np.round(els.angle() / degree).astype(int).tolist(), els.isImproper.tolist())))


_GROUPS_BY_SIGNATURE = {}


def _groupsBySignature():
  if not _GROUPS_BY_SIGNATURE:
    for row in POINT_GROUP_NAMES:
      _GROUPS_BY_SIGNATURE.setdefault(_signature(closure(_generators(row[0], CANONICAL_AXES))), []).append(row[0])
  return _GROUPS_BY_SIGNATURE


def identifyGroup(els, axes=None):
  """the symbol of a list of elements written in the given axes, '' when it names no group

    s = identifyGroup(cs.sym.rot, axes=cs.sym.axes)  # '-3m1' for a trigonal crystal
    s = identifyGroup(els)                        # in orthonormal axes x, y, z

  The groups whose elements have the same angles and parities are tried in turn; the first
  whose closure in the axes is the list names it. A group in a setting no symbol spells
  takes the type its angles say.

  Input
    els - orthogonalTransform, the elements of the group
  Output
    s - str, the Hermann-Mauguin symbol
  Keywords
    axes - the crystal axes a, b, c as a vector3d or a 3 x 3 array, rows, by default the
        orthonormal ones; the secondary symbols depend on them, '-3m1' and '-31m' differing
        by the place of a
  See also
    symmetry, findPointGroup"""
  axes = _asAxes(axes)
  candidates = _groupsBySignature().get(_signature(els), [])
  for symbol in candidates:
    if _sameElements(els, closure(_generators(symbol, axes))):
      return symbol
  # a group in a setting the symbols do not spell, the proper subgroup 222 of -4m2 with its
  # twofold axes along the diagonals: the type is still what the signature says
  return candidates[0] if candidates else ""


def spaceGroupToPointGroup(n):
  """return the crystallographic point-group symbol for a space-group number.

    symbol = spaceGroupToPointGroup(225)

  Input
    number - international space-group number from 1 through 230
  Output
    symbol - Hermann-Mauguin symbol of its point group
  See also
    symmetry, crystalFrame"""
  if not 1 <= n <= 230:
    raise ValueError("I know only 230 space groups")
  return findPointGroup(next(sg[1] for sg in SPACE_GROUPS if n <= sg[0]))


__all__ = ["symmetry", "findPointGroup", "isPointGroupName", "hm2point", "monoclinicSetting", "spaceGroupToPointGroup", "identifyGroup"]
