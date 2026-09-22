# the Bravais lattice systems (MATLAB geometry/latticeType.m)
from __future__ import annotations

import enum

import numpy as np


class latticeType(enum.Enum):
  triclinic = 1
  monoclinic = 2
  orthorhombic = 3
  trigonal = 4
  tetragonal = 5
  hexagonal = 6
  cubic = 7
  none = 8
  icosahedral = 9

  def __str__(self):
    return self.name

  # the axis angles the lattice implies
  @property
  def defaultAngles(self):
    return (np.pi / 2, np.pi / 2, 2 * np.pi / 3) if self.isTriHex else (np.pi / 2, np.pi / 2, np.pi / 2)

  @property
  def isTriHex(self):
    return self in (latticeType.trigonal, latticeType.hexagonal)

  # a lattice whose axes are mutually orthogonal
  @property
  def isOrthogonal(self):
    return self in (latticeType.orthorhombic, latticeType.tetragonal, latticeType.cubic)

  # which of a, b and c the lattice leaves independent
  @property
  def freeLengths(self):
    return (True, self in (latticeType.triclinic, latticeType.monoclinic, latticeType.orthorhombic),
            self not in (latticeType.cubic, latticeType.icosahedral))

  # which of alpha, beta and gamma the lattice leaves independent
  def freeAngles(self, uniqueAxis=1):
    if self is latticeType.triclinic:
      return (True, True, True)
    if self is latticeType.monoclinic:
      return tuple(k == uniqueAxis for k in range(3))
    return (False, False, False)

  # three or four Miller indices
  @property
  def hklForm(self):
    return "hkil" if self.isTriHex else "hkl"


__all__ = ["latticeType"]
