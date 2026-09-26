# the boundary normal distribution of 3D grains (MATLAB grain3Boundary/calcGBND, ../master)
#
#   gbnd = calcGBND(gB3)                     the normals in specimen coordinates, weighted by area
#   gbnd = calcGBND(gB3, grains)             the normals in the crystal frames of both grains
#   gbcd = calcGBND(gB3, grains, moriRef)    the crystal normals of the faces whose misorientation
#                                            is near moriRef, the grain boundary character
#
# The normals of a 3D boundary are measured, so every form is a density of directions: the
# specimen form of the face normals, the crystal form of the normals seen from each grain, and
# the character of the normals seen from the first grain of every face, the misorientation
# brought to its equivalent closest to moriRef and weighted by the kernel of its distance to
# it; the character is averaged over the symmetries that keep moriRef as it is.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import qmul
from ..geometry.units import degree
from ..geometry.vector3d import vector3d


def calcGBND(gB, *args, **kwargs):
  from ..maps3d.grain3boundary import grain3Boundary
  from .gbnd import calcGBND as calcGBND2d
  if not isinstance(gB, grain3Boundary):
    return calcGBND2d(gB, *args, **kwargs)
  return _calcGBND3d(gB, *args, **kwargs)


def _calcGBND3d(gB, grains=None, moriRef=None, halfwidth=10 * degree, noSymmetry=False, total=False, **kwargs):
  from ..functions.density import calcDensity
  from ..geometry.miller import Miller
  # a face of no area has no normal and no weight
  gB = gB[gB.area > 0]
  if grains is None:
    return calcDensity(vector3d(gB.N.data, frame=gB.frame, antipodal=True), weights=gB.area, halfwidth=halfwidth, **kwargs)
  if moriRef is None:
    p = [q for q in np.unique(grains.phaseId) if grains.CSList[int(q)].isIndexed]
    if len(p) != 1:
      raise ValueError("the grains should be of one indexed phase")
    p = int(p[0])
    side = gB.phaseId == p
    f, s = np.nonzero(side)
    q = grains.meanRotation.data.reshape(-1, 4)[grains.id2ind(gB.grainId[f, s])]
    N = _inverseRotate(q, gB.N.data[f])
    cs = grains.CSList[p]
    return calcDensity(Miller(vector3d(N, antipodal=True), cs), weights=gB.area[f], halfwidth=halfwidth, antipodal=True, **kwargs)
  return _gbcd(gB, grains, moriRef, halfwidth, noSymmetry, total, **kwargs)


# the vectors v turned by the inverses of the rotations q, row by row
def _inverseRotate(q, v):
  from ..geometry.quaternion import rotation
  return (rotation._fromData(q * np.array([1, -1, -1, -1])) * vector3d(v)).data.reshape(-1, 3)


def _gbcd(gB, grains, moriRef, hw, noSymmetry, total, **kwargs):
  from ..functions.density import calcDensity
  from ..functions.kernelfunctions import SO3DeLaValleePoussinKernel
  from ..geometry.miller import Miller
  from ..geometry.orientation import orientation
  from ..geometry.quaternion import rotation
  psi = SO3DeLaValleePoussinKernel(halfwidth=hw)
  cs1, cs2 = moriRef.CS, moriRef.SS
  p1, p2 = grains.CSList.phaseId(cs1), grains.CSList.phaseId(cs2)
  sel = np.all(gB.phaseId == [p1, p2], axis=1)
  gB = gB[np.flatnonzero(sel)]
  qg = grains.meanRotation.data.reshape(-1, 4)
  o1, o2 = qg[grains.id2ind(gB.grainId[:, 0])], qg[grains.id2ind(gB.grainId[:, 1])]
  m = qmul(o1 * np.array([1, -1, -1, -1]), o2)
  omega = np.asarray(orientation(rotation._fromData(m), cs1, cs2).angle(moriRef)).reshape(-1)
  inc = omega < 2 * hw
  area = gB.area[inc]
  w = area * np.asarray(psi.eval(np.cos(omega[inc]))).reshape(-1)
  # the pair of symmetries that brings every misorientation closest to moriRef
  G1 = cs1.sym.properGroup().rotations().data.reshape(-1, 4)
  G2 = cs2.sym.properGroup().rotations().data.reshape(-1, 4)
  r = moriRef.data.reshape(4)
  mi = m[inc]
  best = np.full(len(mi), -1.0)
  s1 = np.zeros((len(mi), 4))
  s2 = np.zeros((len(mi), 4))
  for a in G1:
    left = qmul(np.broadcast_to(a * np.array([1, -1, -1, -1]), mi.shape), mi)
    for b in G2:
      c = np.abs(qmul(left, np.broadcast_to(b, mi.shape)) @ r)
      better = c > best
      best[better] = c[better]
      s1[better], s2[better] = a, b
  N = _inverseRotate(qmul(o1[inc], s1), gB.N.data[inc])
  if moriRef.antipodal:
    N = np.concatenate([N, _inverseRotate(qmul(o2[inc], s2), gB.N.data[inc])])
    w = np.concatenate([w, w])
  f = calcDensity(Miller(vector3d(N, antipodal=True), cs1.stripSym()), weights=w, halfwidth=hw, antipodal=True, **kwargs)
  if not noSymmetry:
    f = f.symmetrise(rotation._fromData(_stabiliser(G1, G2, r)))
  if total:
    f = f * (w.sum() / gB.area.sum() / len(w))
  return f


# the symmetries a of the first frame that keep the reference misorientation: a^-1 r b = r for
# some b of the second
def _stabiliser(G1, G2, r):
  keep = []
  for a in G1:
    left = qmul(a * np.array([1, -1, -1, -1]), r)
    if np.any(np.abs(qmul(np.broadcast_to(left, G2.shape), G2) @ r) > 1 - 1e-9):
      keep.append(a)
  return np.array(keep)


__all__ = ["calcGBND"]
