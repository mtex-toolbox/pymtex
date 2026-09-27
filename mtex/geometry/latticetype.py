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

  @property
  def defaultAngles(self):
    """the axis angles the lattice implies"""
    return (np.pi / 2, np.pi / 2, 2 * np.pi / 3) if self.isTriHex else (np.pi / 2, np.pi / 2, np.pi / 2)

  @property
  def isTriHex(self):
    return self in (latticeType.trigonal, latticeType.hexagonal)

  @property
  def isOrthogonal(self):
    """a lattice whose axes are mutually orthogonal"""
    return self in (latticeType.orthorhombic, latticeType.tetragonal, latticeType.cubic)

  @property
  def freeLengths(self):
    """which of a, b and c the lattice leaves independent"""
    return (True, self in (latticeType.triclinic, latticeType.monoclinic, latticeType.orthorhombic),
            self not in (latticeType.cubic, latticeType.icosahedral))

  def freeAngles(self, uniqueAxis=1):
    """which of alpha, beta and gamma the lattice leaves independent"""
    if self is latticeType.triclinic:
      return (True, True, True)
    if self is latticeType.monoclinic:
      return tuple(k == uniqueAxis for k in range(3))
    return (False, False, False)

  @property
  def hklForm(self):
    """three or four Miller indices"""
    return "hkil" if self.isTriHex else "hkl"


__all__ = ["latticeType"]
