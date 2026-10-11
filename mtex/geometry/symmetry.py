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

import re
from typing import TYPE_CHECKING, Any

import numpy as np

from .data.pointgroupnames import (LAUE_CLASSES, POINT_GROUP_NAMES, SPACE_GROUPS, findPointGroup, hm2point,
                                   isPointGroupName, monoclinicSetting, pointGroupId)
from .latticetype import latticeType
from .quaternion import orthogonalTransform, reflection, rotation
from .session import getMTEXpref
from .units import degree
from .vector3d import classproperty, vector3d, xvector, yvector, zvector

if TYPE_CHECKING:
  from .vocabulary import PointGroupName

CANONICAL_AXES = np.eye(3)


# ------------------------------------------------------------------ elements up to tolerance

# the elements of a list that are not repeats of an earlier one, up to an angle and with
# the same parity; pairwise, since a group is small and rounding keys can split a pair
def _uniqueElements(o, tol=1e-6):
  q, p = o.data.reshape(-1, 5)[:, :4], o.data.reshape(-1, 5)[:, 4]
  same = (np.abs(q @ q.T) > 1 - tol) & (p[:, None] == p[None, :])
  keep = np.ones(len(q), dtype=bool)
  for k in range(len(q)):
    if keep[k]:
      keep[k + 1:] &= ~same[k, k + 1:]
  return orthogonalTransform(o.data.reshape(-1, 5)[keep])


# whether each element of a is in the list b
def _isin(a, b, tol=1e-6):
  qa, pa = a.data.reshape(-1, 5)[:, :4], a.data.reshape(-1, 5)[:, 4]
  qb, pb = b.data.reshape(-1, 5)[:, :4], b.data.reshape(-1, 5)[:, 4]
  return np.any((np.abs(qa @ qb.T) > 1 - tol) & (pa[:, None] == pb[None, :]), axis=1)


def _sameElements(a, b, tol=1e-6):
  return a.size == b.size and bool(np.all(_isin(a, b, tol)))


def closure(gens, tol=1e-6):
  """the closure of a list of elements under multiplication: the set grows by its products
  with the generators until it stops, which for a finite group is the group"""
  gens = gens if isinstance(gens, orthogonalTransform) else orthogonalTransform.cat(*gens) if gens else orthogonalTransform()
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
  turned = lambda theta: rotation.byAxisAngle(c, theta) * a
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
  the given axes; also from its elements or its number among the point groups, the Laue classes
  or the space groups

    pg = symmetry('m-3m')
    pg = symmetry(pointId=32)
    pg = symmetry(spaceId=225)"""

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
    """the symbol with a bar over every rotoinversion axis under utf8 output"""
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
    """the proper rotations, and the rotational axes with the highest order about each"""
    return self.rot[~self.rot.isImproper].rot

  # the order of the rotation part of every element, 1 for the identity
  def _orders(self):
    omega = self.rot.angle()
    return np.where(omega < 1e-4, 1, np.round(2 * np.pi / np.where(omega < 1e-4, 1, omega))).astype(int)

  def elements(self, multiplicity=None):
    """the rotational axes and their multiplicities, both signs, symmetry/elements"""
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
    """the maximal n-fold of the group, and about given axes, symmetry/nfold"""
    if axis is None:
      return self._maxOrder()
    axis = axis if isinstance(axis, vector3d) else vector3d(axis)
    return np.array([self._maxOrder(lambda r: r.axis().angle(v, antipodal=True) < 1e-5) for v in axis.flatten()]).reshape(axis.shape)

  @property
  def multiplicityZ(self):
    """the highest order of a rotation axis of the group along z, 1 if there is none"""
    return self._maxOrder(lambda r: np.abs(r.axis().dot(zvector)) > 1 - 1e-8)

  @property
  def multiplicityPerpZ(self):
    """the highest order of a rotation axis of the group perpendicular to z, 1 if there is none"""
    return self._maxOrder(lambda r: np.abs(r.axis().dot(zvector)) < 1e-4)

  @property
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
    """the proper rotation group of the Laue group: the rotation parts of all elements"""
    if self.isProper:
      return self
    if self.__dict__.get("_properGroup") is None:
      self._properGroup = symmetry._fromElements(_uniqueElements(self._withParity(False)), self.axes)
    return self._properGroup

  def properSubGroup(self):
    """the subgroup of proper rotations"""
    if self.isProper:
      return self
    if self.__dict__.get("_properSubGroup") is None:
      self._properSubGroup = symmetry._fromElements(self.rot[~self.rot.isImproper], self.axes)
    return self._properSubGroup

  def disjoint(self, other):
    """the intersection of two groups, symmetry/disjoint"""
    if self == other:
      return self
    keep = _isin(self.rot, other.rot, 1e-4)
    if np.all(keep):
      return self
    if np.sum(keep) == other.rot.size:
      return other
    return symmetry._fromElements(self.rot[keep], self.axes)

  def union(self, other):
    """the group generated by both, symmetry/union"""
    if other <= self:
      return self
    if self <= other:
      return other
    return symmetry(orthogonalTransform.cat(self.rot, other.rot), self.axes)

  def factor(self, other):
    """factor self = rot1 * d and other = d * rot2 with d the common elements, symmetry/factor"""
    d = self.rot[_isin(self.rot, other.rot)]
    if d.size == 1:
      return self.rot, d, other.rot
    rot1 = orthogonalTransform.id.reshape(1)
    while True:
      reached = _isin(self.rot, (rot1[:, None] * d[None, :]).flatten())
      if np.all(reached):
        break
      rot1 = orthogonalTransform.cat(rot1, self.rot[np.argmin(reached)])
    rot2 = orthogonalTransform.id.reshape(1)
    while True:
      reached = _isin(other.rot, (d[:, None] * rot2[None, :]).flatten())
      if np.all(reached):
        break
      rot2 = orthogonalTransform.cat(rot2, other.rot[np.argmin(reached)])
    return rot1, d, rot2

  # ------------------------------------------------------------------ regions

  def fundamentalSector(self, center=None, antipodal=False, angle=None):
    """the fundamental sector of the group on the sphere: the conventional one where it is a
    fundamental domain, else the Dirichlet domain; with center that centre's Dirichlet domain"""
    from .sphericalregion import fundamentalSector
    return fundamentalSector(self, center=center, antipodal=antipodal, angle=angle)

  def conventionalSector(self, up=None, east=None):
    """the conventional sector of the group, the one MTEX draws, with up and east choosing its
    placement"""
    from .sphericalregion import conventionalSector
    return conventionalSector(self, up=up, east=east)

  def fundamentalRegion(self, other=None, *, antipodal=False):
    """the fundamental region of orientations of this group and another one, the Dirichlet domain
    of the identity"""
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
  """the symbol of a list of elements written in the given axes, '' when it names no group"""
  axes = _asAxes(axes)
  candidates = _groupsBySignature().get(_signature(els), [])
  for symbol in candidates:
    if _sameElements(els, closure(_generators(symbol, axes))):
      return symbol
  # a group in a setting the symbols do not spell, the proper subgroup 222 of -4m2 with its
  # twofold axes along the diagonals: the type is still what the signature says
  return candidates[0] if candidates else ""


def spaceGroupToPointGroup(n):
  """the Hermann-Mauguin symbol of the point group of a space group given by its number, 1 to 230"""
  if not 1 <= n <= 230:
    raise ValueError("I know only 230 space groups")
  return findPointGroup(next(sg[1] for sg in SPACE_GROUPS if n <= sg[0]))


__all__ = ["symmetry", "findPointGroup", "isPointGroupName", "hm2point", "monoclinicSetting", "spaceGroupToPointGroup", "identifyGroup"]
