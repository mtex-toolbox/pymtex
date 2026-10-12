# symmetrisation and the comparisons that reduce over the group (MATLAB
# @vector3d/symmetrise, dot, angle, unique, multiplicity; @orientation/symmetrise, dot,
# angle, unique, mean, axis, projectIntoFundamentalRegion)
#
# A direction in a frame carrying a group stands for its whole orbit, an orientation for
# its coset l * o * r; comparison reduces over the group, representation does not (MATLAB
# ADR 0008 rule 10). symmetry=False is the opt-out at the call site. Only a crystal
# frame's group enters vector comparison: a specimen frame's group is a claim about
# functions, never about directions.
from __future__ import annotations

import numpy as np

from ._array import MtexArray
from .kernels import closestAngle, closestImage, nearestAngle, projectDirections, projectToRef, qconj, threads
from .quaternion import qmul, qrot
from .symreduce import (_firstOfClusters, bestDot, closestToRef, referenceImages, reductionGroups, reductionGroupsPair,
                        symReduce, uniqueRotations)
from .vector3d import _isCrystalFrame, _keptAndInverse, _uniqueOutput, _xyz, vector3d

_ID = np.array([1.0, 0.0, 0.0, 0.0])


def reducesOverGroup(v):
  """whether an operation on the vectors reduces over a group: crystal directions in a frame of
  more than one element"""
  return v.isCrystalDirection and v.frame.numSym() > 1


# ------------------------------------------------------------------ directions

# the images e * x of the elements of a group, shaped x.shape + (|G|, 3): the group axis
# sits next to the coordinates so that two lists of any shapes broadcast on the leading axes
def _images(frame, xyz):
  o = frame.rot.data.reshape(-1, 5)
  return qrot(o[:, :4], xyz[..., None, :]) * o[:, 4:5]


def symmetrise(v, unique=False, antipodal=None, tol=1e-6):
  """the orbit of a crystal direction under the group of its frame, every element once
  (shape numSym x size); antipodal=True adds the negatives unless the group holds the
  inversion or the vector is an axis; unique keeps the distinct images of one direction only,
  up to sign for an axis unless antipodal=False"""
  # symmetrise(v, cs): the coordinates of v read in the frame given, as MATLAB's
  from .frames import referenceFrame
  if isinstance(unique, referenceFrame):
    v, unique = v._new(v.data, _frame=unique), False
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
  asAxes = v.antipodal if antipodal is None else antipodal
  n = np.linalg.norm(xyz, axis=1)
  d = xyz @ xyz.T / np.outer(n, n)
  return v._new(xyz[_firstOfClusters((np.abs(d) if asAxes else d) > 1 - tol)])


def multiplicity(v, antipodal=None):
  """the number of distinct directions equivalent to each element, of axes under antipodal"""
  return np.array([symmetrise(x, unique=True, antipodal=antipodal).size for x in v.flatten()]).reshape(v.shape)


# the side that reduces, and the other one's coordinates
def _symSide(a, b):
  if reducesOverGroup(a):
    return a, b
  if isinstance(b, vector3d) and reducesOverGroup(b):
    return b, a
  return None, None


def symDot(a, b, min=False, all=False, antipodal=None):
  """the inner product reduced over the group of the side that carries one: the maximum over
  the orbit, or the minimum, or all of them"""
  sym, other = _symSide(a, b)
  if sym is None:
    return None
  ds = np.sum(_images(sym.frame, sym.data) * other.data[..., None, :], axis=-1)
  if (a.antipodal or b.antipodal) if antipodal is None else antipodal:
    ds = np.abs(ds)
  if all:
    return np.moveaxis(ds, -1, 0)
  return ds.min(axis=-1) if min else ds.max(axis=-1)


def symAngle(a, b, antipodal=None):
  """the angle to the closest symmetric equivalent, through atan2 for accuracy"""
  sym, other = _symSide(a, b)
  if sym is None:
    return None
  imgs = _images(sym.frame, sym.data)
  d = np.sum(imgs * other.data[..., None, :], axis=-1)
  c = np.linalg.norm(np.cross(imgs, other.data[..., None, :]), axis=-1)
  if (a.antipodal or b.antipodal) if antipodal is None else antipodal:
    d = np.abs(d)
  return np.arctan2(c, d).min(axis=-1)


def projectIntoFundamentalRegion(v, center=None):
  """every direction of a list folded into the fundamental sector of its frame, the one an
  inverse pole figure shows; an axis into the sector of the Laue group. Given a centre, a
  direction goes to the equivalent nearest it instead: the element whose image of the
  centre is closest to the direction, applied inversely"""
  if not reducesOverGroup(v):
    return v
  fr = v.frame
  if center is None:
    from .sphericalregion import fundamentalSector, projectIntoRegion
    return projectIntoRegion(v, fundamentalSector(fr, antipodal=v.antipodal), fr.sym.Laue() if v.antipodal else fr.sym)
  o = fr.rot.data.reshape(-1, 5)
  c = _xyz(center)
  imgs = qrot(o[:, :4], c[None]) * o[:, 4:5]
  return v._new(projectDirections(v.data, imgs, o[:, :4], o[:, 4]))


def symUnique(v, tol=1e-6, return_index=False, return_inverse=False):
  """the distinct directions of a list up to the group: each is moved to its equivalent nearest
  the centre first, so equivalent ones become the same vector even on the rim of a sector,
  where a fold into it could keep two of them"""
  from .sphericalregion import sectorCenter
  rep = projectIntoFundamentalRegion(v, center=sectorCenter(v.frame)).data.reshape(-1, 3) if reducesOverGroup(v) else v.data.reshape(-1, 3)
  n = np.linalg.norm(rep, axis=1)
  d = rep @ rep.T / np.outer(n, n)
  same = d > 1 - tol
  keep = np.ones(len(rep), dtype=bool)
  first = np.arange(len(rep))
  for k in range(len(rep)):
    if keep[k]:
      later = np.flatnonzero(same[k, k + 1:]) + k + 1
      keep[later] = False
      first[later] = k
  iv, iu = _keptAndInverse(keep, first)
  return _uniqueOutput(v._new(v.data.reshape(-1, 3)[iv]), iv, iu, return_index, return_inverse)


def symMean(v, weights=None, iterations=3):
  """the mean of crystal directions: every element is moved to the equivalent closest to the
  running mean before the vector mean is taken"""
  m = v.flatten()[0]
  xs = v.flatten()
  for _ in range(iterations):
    proj = projectIntoFundamentalRegion(xs, center=m.normalize())
    m = vector3d.mean(proj, weights=weights, symmetry=False)
  return v._new(m.data)


# ------------------------------------------------------------------ orientations

def _exchange(a, b=None):
  return a.antipodal or (b is not None and getattr(b, "antipodal", False))


def symmetriseOrientation(o, unique=False):
  """the symmetric equivalents l * o * r, and under grain exchange their inverses; shaped
  (numL * numR,) + o.shape, or the distinct ones as a flat list"""
  L, R = reductionGroups(o)
  q = o.data
  es = qmul(qmul(L[:, None, None, :], q[None, None, ...]), R[None, :, None, :])   # (|L|, |R|, n, 4)
  es = es.reshape(-1, *q.shape)
  if o.antipodal:
    ei = qmul(qmul(L[:, None, None, :], qconj(q)[None, None, ...]), R[None, :, None, :]).reshape(-1, *q.shape)
    es = np.concatenate([es, ei])
  if unique:
    return o._new(uniqueRotations(es.reshape(-1, 4)))
  return o._new(es)


def fitOrientations(a, b):
  """two orientations can be compared when their target frames fit; two different crystal
  frames on the source side are a phase boundary"""
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


def orientationDot(a, b):
  """the reduced dot product of pairs, broadcast"""
  a, b = fitOrientations(a, b)
  L, R = reductionGroupsPair(a, b)
  exchange = _exchange(a, b)
  # a list against one orientation goes through the images of that one
  if b.ndim == 0 and a.ndim > 0:
    return bestDot(a.data, referenceImages(b.data, L, R, exchange)[0])[0]
  return symReduce(a.data, b.data, L, R, exchange)[0]


def orientationAngle(a, b):
  """the disorientation angle of pairs, through the closest equivalent for accuracy near zero"""
  a, b = fitOrientations(a, b)
  L, R = reductionGroupsPair(a, b)
  return closestAngle(a.data, b.data, L, R, _exchange(a, b))


def orientationSelfAngle(o):
  """the rotational angle of an orientation reduced over its own groups: the smallest angle
  among its equivalents, the distance to the identity"""
  # the images l r of the identity are the group's elements, few where the pairs are many; the
  # chord to the nearest of them is the angle, unchanged by turning both by l and r
  L, R = reductionGroups(o)
  es, _ = referenceImages(_ID, L, R, o.antipodal)
  return nearestAngle(o.data, es)


def projectIntoFundamentalRegionOrientation(o, ref=None):
  """every orientation moved to its equivalent closest to a reference - the identity when none
  is given - which is the representative in the fundamental region; around the identity the
  conjugates tie and the axis sector of the shared rotations decides. A list of references,
  one per element, is the pair search of dot without the images of a single reference"""
  from .orientationregion import commonAxisCenter, commonRotations
  L, R = reductionGroups(o)
  if ref is None:
    refq = _ID
  else:
    refq = ref.data if isinstance(ref, MtexArray) else np.asarray(ref, dtype=float)
  if refq.ndim > 1:
    return o._new(closestImage(refq, o.data, L, R, o.antipodal)[1])
  es, frm = referenceImages(refq, L, R, o.antipodal)
  common, cimgs = None, None
  if abs(refq[0]) > 1 - 1e-12:
    common = commonRotations(L, R)
    if len(common) > 1:
      cimgs = qrot(common, commonAxisCenter(common)[None])
  p = projectToRef(o.data, es, frm, L, R, refq, common, cimgs)
  if o.antipodal and abs(refq[0]) > 1 - 1e-12:
    p = _exchangeIntoSector(p, commonRotations(L, R))
  return o._new(p)


# with grain exchange the inverse, the same angle about the opposite axis, is a further
# representative: among the conjugates of p and of its inverse by the shared rotations the
# first whose axis lies in the sector of the shared rotations with inversion, else the one
# whose axis is nearest its centre
def _exchangeIntoSector(p, common):
  from .orientationregion import commonAxisSector
  sector, center = commonAxisSector(common, True)
  q = p.reshape(-1, 4)
  cands = []
  for c in (common if len(common) else _ID[None]):
    ci = qconj(c)
    for x in (q, qconj(q)):
      cands.append(qmul(qmul(ci, x), c))
  cands = np.stack(cands, axis=1)
  cands *= np.where(cands[..., :1] < 0, -1.0, 1.0)
  axes = cands[..., 1:]
  inside = np.all(axes @ sector.N.T >= -1e-9 * np.linalg.norm(axes, axis=-1, keepdims=True), axis=-1)
  score = np.where(inside, 2.0, axes @ center / np.maximum(np.linalg.norm(axes, axis=-1), 1e-300))
  best = cands[np.arange(len(q)), np.argmax(score, axis=1)]
  # a rotation about no axis stays as it is
  best = np.where((np.linalg.norm(q[:, 1:], axis=1) < 1e-12)[:, None], q, best)
  return best.reshape(p.shape)


# the distinct orientations of a list up to symmetry: representatives in the fundamental
# region, then unique on those
# the pairs of orientations within the tolerance, over all their symmetric images, as a
# sparse matrix from the radius search of a quaternion tree
def _neighbours(o, tol, symmetry):
  from scipy import sparse
  n = o.size
  if not symmetry:
    from scipy.spatial import cKDTree
    q = o.data.reshape(-1, 4)
    hits = cKDTree(np.vstack([q, -q])).query_ball_point(q, 2 * np.sin(tol / 4), workers=threads())
    rows = np.repeat(np.arange(n), [len(h) for h in hits])
    cols = np.concatenate([np.asarray(h, dtype=int) for h in hits]) % n
    return sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))
  from .grids.orientationtree import OrientationTree
  flat = o.flatten()
  return OrientationTree(flat).radius(flat, tol)[0].tocsr()


def orientationUnique(o, tol=1e-3, symmetry=True, return_index=False, return_inverse=False):
  """the orientations with the duplicates within the tolerance removed, symmetric equivalents
  counting as one unless symmetry=False; the first of each kept, in order, with the indices
  of the kept and the index of every orientation into them when asked

    u = orientationUnique(ori, tol=1*degree)
    u, iv, iu = orientationUnique(ori, return_index=True, return_inverse=True)"""
  n = o.size
  same = _neighbours(o, tol, symmetry)
  keep = np.ones(n, dtype=bool)
  first = np.arange(n)
  # every orientation not yet taken absorbs the later ones within the tolerance
  for k in range(n):
    if keep[k]:
      nb = same.indices[same.indptr[k]:same.indptr[k + 1]]
      later = nb[nb > k]
      later = later[keep[later]]
      keep[later] = False
      first[later] = k
  iv, iu = _keptAndInverse(keep, first)
  return _uniqueOutput(o._new(o.data.reshape(-1, 4)[iv]), iv, iu, return_index, return_inverse)


def orientationMean(o, weights=None, robust=False, iterations=3):
  """the mean orientation: every element is moved to the equivalent closest to the running mean
  before the quaternion mean is taken (orientation/mean)"""
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


def axisFrame(o):
  """the rotational axis of a misorientation: that of the representative in the fundamental
  region; only the rotations both frames share leave it invariant, and grain exchange takes
  away its sign, which for a crystal direction is the Laue sibling of the frame
  the frame an axis is expressed in: that of the first crystal of a misorientation, as MATLAB
  labels it, the specimen for an orientation; the axis is fixed by the rotation, so its
  components are the same in both frames"""
  g = o.frameA.sym.disjoint(o.frameB.sym).properGroup()
  if o.antipodal:
    g = g.Laue()
  return (o.frameA if o.isMisorientation else o.frameB).sibling(g)


def orientationAxis(o, symmetry=True):
  """the rotational axes of orientations or misorientations, those of the representatives in the
  fundamental region, or of the rotations as given with symmetry=False, in the frame axisFrame
  names"""
  from .representations import quatAxis
  q = o.data if not symmetry else projectIntoFundamentalRegionOrientation(o).data
  return vector3d(quatAxis(q), isNormalized=True, frame=axisFrame(o))


def pairAxis(o1, o2):
  """the axis of the disorientation between two orientations, in the frame they share, the
  specimen: over the equivalents o1 g inv(o2) of the crystal group the one of the smallest
  angle, whose axis is a direction of the specimen; its two ends are one axis"""
  from .products import fitFrame
  from .representations import quatAxis
  if o1.frameB is not o2.frameB:
    raise ValueError("the two lists are not in the same specimen frame")
  fit = fitFrame(o2.frameA, o1.frameA)
  b = o2.data if fit is None else qmul(o2.data, qconj(fit.data))
  G = o1.frameA.sym.properGroup().rotations().data.reshape(-1, 4)
  a, b = np.broadcast_arrays(o1.data, b)
  cand = qmul(qmul(a[..., None, :], G), qconj(b)[..., None, :])      # (..., |G|, 4)
  k = np.argmax(np.abs(cand[..., 0]), axis=-1)
  best = np.take_along_axis(cand, k[..., None, None], axis=-2)[..., 0, :]
  # q and -q are one rotation; the axis is that of the one with a positive scalar part
  best = best * np.where(best[..., :1] < 0, -1.0, 1.0)
  return vector3d(quatAxis(best), isNormalized=True, antipodal=True, frame=o1.frameB)


__all__ = []
