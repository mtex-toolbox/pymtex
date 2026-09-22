# symmetrisation and the comparisons that reduce over the group (MATLAB
# @vector3d/symmetrise, dot, angle, unique, multiplicity; @orientation/symmetrise, dot,
# angle, unique, mean, axis, project2FundamentalRegion)
#
# A direction in a frame carrying a group stands for its whole orbit, an orientation for
# its coset l * o * r; comparison reduces over the group, representation does not (MATLAB
# ADR 0008 rule 10). noSymmetry=True is the opt-out at the call site. Only a crystal
# frame's group enters vector comparison: a specimen frame's group is a claim about
# functions, never about directions.
from __future__ import annotations

import numpy as np

from ._array import MtexArray

_ID = np.array([1.0, 0.0, 0.0, 0.0])

from .quaternion import qmul, qrot
from .kernels import closestImage, projectDirections, projectToRef
from .symreduce import (_conj, bestDot, chordAngle, closestToRef, referenceImages, reductionGroups, reductionGroupsPair,
                        symReduce, uniqueRotations)
from .vector3d import _isCrystalFrame, vector3d


def reducesOverGroup(v):
  return v.isCrystalDirection and v.frame.numSym() > 1


# ------------------------------------------------------------------ directions

# the images e * x of the elements of a group, shaped x.shape + (|G|, 3): the group axis
# sits next to the coordinates so that two lists of any shapes broadcast on the leading axes
def _images(frame, xyz):
  o = frame.rot.data.reshape(-1, 5)
  return qrot(o[:, :4], xyz[..., None, :]) * o[:, 4:5]


# the orbit of a crystal direction under the group of its frame, every element once
# (shape numSym x size); antipodal adds the negatives unless the group holds the inversion
# or the vector is an axis; unique keeps the distinct images of one direction only
def symmetrise(v, unique=False, antipodal=None, noAntipodal=False, tolerance=1e-6):
  if not v.isCrystalDirection:
    return v.reshape((1,) + v.shape)
  fr = v.frame
  imgs = np.moveaxis(_images(fr, v.data), -2, 0)
  if antipodal and not v.antipodal and not fr.isLaue:
    imgs = np.concatenate([imgs, -imgs])
  out = v._new(imgs)
  if not unique:
    return out
  if v.ndim > 0 and v.size > 1:
    raise ValueError("unique symmetrisation gives orbits of different sizes; symmetrise the elements one by one")
  xyz = imgs.reshape(-1, 3)
  asAxes = (v.antipodal or antipodal) and not noAntipodal
  n = np.linalg.norm(xyz, axis=1)
  d = xyz @ xyz.T / np.outer(n, n)
  same = (np.abs(d) if asAxes else d) > 1 - tolerance
  keep = np.ones(len(xyz), dtype=bool)
  for k in range(len(xyz)):
    if keep[k]:
      keep[k + 1:] &= ~same[k, k + 1:]
  return v._new(xyz[keep])


# the number of distinct directions equivalent to each element, of axes under antipodal
def multiplicity(v, antipodal=None, noAntipodal=False):
  return np.array([symmetrise(x, unique=True, antipodal=antipodal, noAntipodal=noAntipodal).size for x in v.flatten()]).reshape(v.shape)


# the side that reduces, and the other one's coordinates
def _symSide(a, b):
  if reducesOverGroup(a):
    return a, b
  if isinstance(b, vector3d) and reducesOverGroup(b):
    return b, a
  return None, None


# the inner product reduced over the group of the side that carries one: the maximum over
# the orbit, or the minimum, or all of them
def symDot(a, b, min=False, all=False, antipodal=False):
  sym, other = _symSide(a, b)
  if sym is None:
    return None
  ds = np.sum(_images(sym.frame, sym.data) * other.data[..., None, :], axis=-1)
  if antipodal or a.antipodal or b.antipodal:
    ds = np.abs(ds)
  return np.moveaxis(ds, -1, 0) if all else ds.min(axis=-1) if min else ds.max(axis=-1)


# the angle to the closest symmetric equivalent, through atan2 for accuracy
def symAngle(a, b, antipodal=False):
  sym, other = _symSide(a, b)
  if sym is None:
    return None
  imgs = _images(sym.frame, sym.data)
  d = np.sum(imgs * other.data[..., None, :], axis=-1)
  c = np.linalg.norm(np.cross(imgs, other.data[..., None, :]), axis=-1)
  if antipodal or a.antipodal or b.antipodal:
    d = np.abs(d)
  return np.arctan2(c, d).min(axis=-1)


# the centre of the Dirichlet domain of a frame, and every direction of a list moved into
# it: the element whose image of the centre is closest to the direction, applied inversely
def project2FundamentalRegion(v, center=None):
  if not reducesOverGroup(v):
    return v
  from .sphericalregion import sectorCenter
  fr = v.frame
  o = fr.rot.data.reshape(-1, 5)
  c = sectorCenter(fr) if center is None else center
  c = c.data if isinstance(c, vector3d) else np.asarray(c, dtype=float)
  imgs = qrot(o[:, :4], c[None]) * o[:, 4:5]
  return v._new(projectDirections(v.data, imgs, o[:, :4], o[:, 4]))


# the distinct directions of a list up to the group: each is projected to the fundamental
# sector of its frame first, so equivalent ones become the same vector
def symUnique(v, tolerance=1e-6, return_index=False, return_inverse=False):
  rep = project2FundamentalRegion(v).data.reshape(-1, 3)
  n = np.linalg.norm(rep, axis=1)
  d = rep @ rep.T / np.outer(n, n)
  same = d > 1 - tolerance
  keep = np.ones(len(rep), dtype=bool)
  first = np.arange(len(rep))
  for k in range(len(rep)):
    if keep[k]:
      later = np.flatnonzero(same[k, k + 1:]) + k + 1
      keep[later] = False
      first[later] = k
  iv = np.flatnonzero(keep)
  rank = np.empty(len(rep), dtype=int)
  rank[iv] = np.arange(len(iv))
  iu = rank[first]
  u = v._new(v.data.reshape(-1, 3)[iv])
  out = (u,) + ((iv,) if return_index else ()) + ((iu,) if return_inverse else ())
  return out if len(out) > 1 else u

# the mean of crystal directions: every element is moved to the equivalent closest to the
# running mean before the vector mean is taken
def symMean(v, weights=None, iterations=3):
  m = v.flatten()[0]
  xs = v.flatten()
  for _ in range(iterations):
    proj = project2FundamentalRegion(xs, center=m.normalize())
    m = vector3d.mean(proj, weights=weights, noSymmetry=True)
  return v._new(m.data)


# ------------------------------------------------------------------ orientations

def _exchange(a, b=None):
  return a.antipodal or (b is not None and getattr(b, "antipodal", False))


# the symmetric equivalents l * o * r, and under grain exchange their inverses; shaped
# (numL * numR,) + o.shape, or the distinct ones as a flat list
def symmetriseOrientation(o, unique=False):
  L, R = reductionGroups(o)
  q = o.data
  es = qmul(qmul(L[:, None, None, :], q[None, None, ...]), R[None, :, None, :])   # (|L|, |R|, n, 4)
  es = es.reshape(-1, *q.shape)
  if o.antipodal:
    ei = qmul(qmul(L[:, None, None, :], _conj(q)[None, None, ...]), R[None, :, None, :]).reshape(-1, *q.shape)
    es = np.concatenate([es, ei])
  if unique:
    return o._new(uniqueRotations(es.reshape(-1, 4)))
  return o._new(es)


# two orientations can be compared when their target frames fit; two different crystal
# frames on the source side are a phase boundary
def fitOrientations(a, b):
  from .frames import _sameBasis, crystalFrame
  x, y = a.frameB, b.frameB
  if not (x is y or _sameBasis(x, y) or (isinstance(x, crystalFrame) and isinstance(y, crystalFrame) and x.sim(y))):
    raise ValueError(f"cannot compare orientations into {x.char()} with orientations into {y.char()}")
  if isinstance(a.frameA, crystalFrame) != isinstance(b.frameA, crystalFrame):
    raise ValueError("cannot compare an orientation with the inverse of one")
  # crystal frames of one group aligned differently: the first is re-expressed in the
  # second, said aloud, as MATLAB's ensureCS does
  x, y = a.frameA, b.frameA
  if isinstance(x, crystalFrame) and x is not y and not _sameBasis(x, y) and x.pointGroup == y.pointGroup:
    ok, M = x.isCompatible(y)
    if ok and np.linalg.norm(M - np.eye(3)) > 1e-1:
      print("\n  The involved symmetries have different reference systems")
      print(f"  1: {x.char()}, {', '.join(x.alignment())}")
      print(f"  2: {y.char()}, {', '.join(y.alignment())}")
      print("  I'm going to transform the data from the first one to the second one\n")
      a = a.transformReferenceFrame(y)
  return a, b


# the reduced dot product of pairs, broadcast
def orientationDot(a, b):
  a, b = fitOrientations(a, b)
  L, R = reductionGroupsPair(a, b)
  if b.ndim == 0:
    return symReduce(a.data, b.data, L, R, _exchange(a, b))[0] if a.ndim == 0 else bestDot(a.data, referenceImages(b.data, L, R, _exchange(a, b))[0])[0]
  return symReduce(a.data, b.data, L, R, _exchange(a, b))[0]


# the disorientation angle of pairs, through the closest equivalent for accuracy near zero
def orientationAngle(a, b):
  a, b = fitOrientations(a, b)
  L, R = reductionGroupsPair(a, b)
  q1, q2 = np.broadcast_arrays(a.data, b.data)
  return chordAngle(q1, closestImage(q1, q2, L, R, _exchange(a, b))[1])


# the rotational angle of an orientation reduced over its own groups: the smallest angle
# among its equivalents, the distance to the identity
def orientationSelfAngle(o):
  L, R = reductionGroups(o)
  return chordAngle(_ID, closestImage(_ID, o.data, L, R, o.antipodal)[1])


# every orientation moved to its equivalent closest to a reference - the identity when none
# is given - which is the representative in the fundamental region; around the identity the
# conjugates tie and the axis sector of the shared rotations decides. A list of references,
# one per element, is the pair search of dot without the images of a single reference
def project2FundamentalRegionOrientation(o, ref=None):
  from .orientationregion import commonAxisCenter, commonRotations
  L, R = reductionGroups(o)
  refq = np.array([1.0, 0, 0, 0]) if ref is None else (ref.data if isinstance(ref, MtexArray) else np.asarray(ref, dtype=float))
  if refq.ndim > 1:
    return o._new(closestImage(refq, o.data, L, R, o.antipodal)[1])
  es, frm = referenceImages(refq, L, R, o.antipodal)
  common, cimgs = None, None
  if abs(refq[0]) > 1 - 1e-12:
    common = commonRotations(L, R)
    if len(common) > 1:
      cimgs = qrot(common, commonAxisCenter(common)[None])
  return o._new(projectToRef(o.data, es, frm, L, R, refq, common, cimgs))


# the distinct orientations of a list up to symmetry: representatives in the fundamental
# region, then unique on those
def orientationUnique(o, tolerance=1e-3, noSymmetry=False, return_index=False, return_inverse=False):
  rep = o.data.reshape(-1, 4) if noSymmetry else project2FundamentalRegionOrientation(o.flatten()).data
  c = np.cos(tolerance / 2)
  same = np.abs(rep @ rep.T) > c
  keep = np.ones(len(rep), dtype=bool)
  first = np.arange(len(rep))
  for k in range(len(rep)):
    if keep[k]:
      later = np.flatnonzero(same[k, k + 1:]) + k + 1
      keep[later] = False
      first[later] = k
  iv = np.flatnonzero(keep)
  rank = np.empty(len(rep), dtype=int)
  rank[iv] = np.arange(len(iv))
  iu = rank[first]
  u = o._new(o.data.reshape(-1, 4)[iv])
  out = (u,) + ((iv,) if return_index else ()) + ((iu,) if return_inverse else ())
  return out if len(out) > 1 else u


# the mean orientation: every element is moved to the equivalent closest to the running mean
# before the quaternion mean is taken (orientation/mean)
def orientationMean(o, weights=None, robust=False, iterations=3):
  from .quaternion import quaternion
  q = o.data.reshape(-1, 4)
  # a site without an orientation does not enter
  finite = np.all(np.isfinite(q), axis=-1)
  if not np.all(finite):
    q = q[finite]
    weights = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)[finite]
  if len(q) == 0:
    raise ValueError("the mean of no orientations")
  L, R = reductionGroups(o)
  m = q[0]
  for _ in range(iterations):
    moved = closestToRef(q, m, L, R, o.antipodal)
    m = quaternion.mean(quaternion(moved), weights=weights, robust=robust).data
  return o._new(m)


# the rotational axis of a misorientation: that of the representative in the fundamental
# region; only the rotations both frames share leave it invariant, and grain exchange takes
# away its sign, which for a crystal direction is the Laue sibling of the frame
def axisFrame(o):
  g = o.frameA.sym.disjoint(o.frameB.sym).properGroup()
  if o.antipodal:
    g = g.Laue()
  return o.frameB.sibling(g)


def orientationAxis(o, noSymmetry=False):
  from .representations import quatAxis
  q = o.data if noSymmetry else project2FundamentalRegionOrientation(o).data
  return vector3d(quatAxis(q), isNormalized=True, frame=axisFrame(o))


# the axis of the disorientation between two orientations, in the frame they share, the
# specimen: over the equivalents o1 g inv(o2) of the crystal group the one of the smallest
# angle, whose axis is a direction of the specimen; its two ends are one axis
def pairAxis(o1, o2):
  from .products import fitFrame
  from .representations import quatAxis
  if o1.frameB is not o2.frameB:
    raise ValueError("the two lists are not in the same specimen frame")
  fit = fitFrame(o2.frameA, o1.frameA)
  b = o2.data if fit is None else qmul(o2.data, _conj(fit.data))
  G = o1.frameA.sym.properGroup().rotations().data.reshape(-1, 4)
  a, b = np.broadcast_arrays(o1.data, b)
  cand = qmul(qmul(a[..., None, :], G), _conj(b)[..., None, :])      # (..., |G|, 4)
  k = np.argmax(np.abs(cand[..., 0]), axis=-1)
  best = np.take_along_axis(cand, k[..., None, None], axis=-2)[..., 0, :]
  return vector3d(quatAxis(best), isNormalized=True, antipodal=True, frame=o1.frameB)


__all__ = []
