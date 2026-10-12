# products of orientations with rotations, orientations and vectors (S19 to S22)
#
#   o1 * o2      o2.frameB must fit o1.frameA; the result maps o2.frameA -> o1.frameB
#   rot * o      acts in the specimen frame: the specimen group is dropped unless rot is an element
#   o * rot      acts in the crystal frame: the crystal group is dropped unless rot is an element
#   o * cs       the equivalents o * s over the proper elements of the crystal side
#   ss * o       the equivalents s * o over the proper elements of the specimen side
#   o * v        a direction in o.frameA, expressed in o.frameB
#   inv(o) * r   a specimen direction in crystal coordinates
#
# What has to fit between two factors is the frame the inner coordinates are written in,
# never the groups (MATLAB ADR 0003): identical frames pass, compatible crystal frames have
# their transition absorbed, a crystal frame never fits a specimen one. The symmetry-aware
# refinements (fitFrame with the transition, dropGroup) are completed in G5.
from __future__ import annotations

import numpy as np

from ._array import MtexArray
from .frames import crystalFrame, referenceFrame, specimenFrame
from .orientation import orientation
from .quaternion import orthogonalTransform, qmul, qrot, quaternion, rotation
from .vector3d import vector3d


def fitFrame(fr, side):
  """whether a frame fits the side an orientation acts on: frame-free data passes, the same
  handle passes, aligned crystal frames pass, compatible ones give the rotation that absorbs
  the transition, anything else is an error"""
  if fr is None or fr is side:
    return None
  if isinstance(fr, crystalFrame) != isinstance(side, crystalFrame):
    raise ValueError(f"the frame {fr.char()} does not fit the side {side.char()}: a crystal frame never fits a specimen one")
  if isinstance(fr, specimenFrame):
    return None
  if fr.isAligned(side):
    return None
  ok, M = fr.isCompatible(side)
  if not ok:
    raise ValueError(f"the frames {fr.char()} and {side.char()} are not compatible; use transformReferenceFrame")
  return rotation.byMatrix(M)


def dropGroup(rot, fr):
  """a rotation that is not an element of the group on a side strips the group from that side's
  frame, silently, and keeps the frame"""
  if fr.sym.Laue().numSym() <= 2:
    return fr
  if rot in fr:
    return fr
  return fr.stripSym()


def sideGroup(o, fr, right):
  """the proper elements an orientation is symmetrised by on one side, read off the frame the
  factor names; the side has to be the one the orientation carries"""
  from .symreduce import reductionGroups
  side = o.frameA if right else o.frameB
  if fr is not side and not (type(fr) is type(side) and fr.isAligned(side)):
    raise ValueError(f"{fr.char()} is not the {'crystal' if right else 'specimen'} frame of the orientation, {side.char()}")
  L, R = reductionGroups(o)
  return R if right else L


def mtimes(a, b):
  """the product of orientations with orientations, rotations, frames and vectors, what ori * x
  computes: the frames fitted between the factors, a frame on either side listing the symmetric
  equivalents along a new axis, a vector mapped from the crystal frame into the specimen frame

    o1 * o2        o2.SS must fit o1.CS
    ori * v        a crystal direction in specimen coordinates
    ori * cs       the equivalents over the proper elements of the crystal side"""
  # a frame on either side lists the symmetric equivalents, the group along a new axis
  if isinstance(a, orientation) and isinstance(b, referenceFrame):
    R = sideGroup(a, b, True)
    return a._new(qmul(a.data[..., None, :], R), antipodal=False)
  if isinstance(b, orientation) and isinstance(a, referenceFrame):
    L = sideGroup(b, a, False)
    return b._new(qmul(L.reshape((-1,) + (1,) * b.ndim + (4,)), b.data), antipodal=False)
  if isinstance(a, orientation) and isinstance(b, orientation):
    t = fitFrame(b.frameB, a.frameA)
    q = b.data if t is None else qmul(t.data, b.data)
    return a._new(qmul(a.data, q), frameA=b.frameA, frameB=a.frameB, antipodal=False)
  if isinstance(a, orientation):
    if isinstance(b, vector3d):
      t = fitFrame(b.frame, a.frameA)
      xyz = b.data if t is None else qrot(t.data, b.data)
      out = b._new(qrot(a.data, xyz), _frame=a.frameB, dispStyle="xyz")
      if isinstance(a.frameB, crystalFrame):
        out.dispStyle = "hkil" if a.frameB.lattice.isTriHex else "hkl"
      return out
    if isinstance(b, orthogonalTransform):
      raise TypeError("orthogonalTransform * orientation is not defined; orientations are proper")
    if isinstance(b, quaternion):
      return a._new(qmul(a.data, b.data), frameA=dropGroup(b, a.frameA), antipodal=False)
    if isinstance(b, MtexArray) or hasattr(b, "rotate"):
      return NotImplemented
    return quaternion(a.data * np.asarray(b, dtype=float)[..., None])
  if isinstance(b, orientation):
    if isinstance(a, orthogonalTransform):
      raise TypeError("orthogonalTransform * orientation is not defined; orientations are proper")
    if isinstance(a, quaternion):
      return b._new(qmul(a.data, b.data), frameB=dropGroup(a, b.frameB), antipodal=False)
    if isinstance(a, MtexArray):
      return NotImplemented
    return quaternion(b.data * np.asarray(a, dtype=float)[..., None])
  return NotImplemented


__all__ = []
