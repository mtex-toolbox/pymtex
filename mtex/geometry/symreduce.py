# the one kernel behind every symmetry reduced comparison of orientations
#
# Two orientations describe the same physical orientation when o2 = l * o1 * r with l in
# the specimen group and r in the crystal group (S22), and under grain exchange also when
# inv(o2) does. Every comparison - dot, angle, ==, unique, mean, the projection into the
# fundamental region - is the same maximisation of |<q1, l q2 r>| over the proper elements
# (Julia ADR 0003), and this file computes it once, together with the pair (l, r) that
# attains it; the two searches themselves are the loops of kernels.py.
from __future__ import annotations

import numpy as np

from .kernels import bestDot, closestAngle, closestImage, qconj, symReduce
from .quaternion import qmul
from .representations import chordAngle, quatAngle, quatAxis


def reductionGroups(o):
  """the proper rotations each side reduces over: the rotation parts of the Laue group when
  either side is centrosymmetric, the proper subgroup otherwise, as symmetry-reduced
  distances in MTEX have always done it"""
  a, b = o.frameA.sym, o.frameB.sym
  laue = a.isLaue or b.isLaue
  R = (a.properGroup() if laue else a.properSubGroup()).rotations().data.reshape(-1, 4)   # acts on the right, crystal side
  L = (b.properGroup() if laue else b.properSubGroup()).rotations().data.reshape(-1, 4)   # acts on the left, specimen side
  return L, R


def functionGroups(CS, SS):
  """the rotations a function on SO(3) is invariant under, on the specimen side and on the
  crystal side: those of the Laue groups of both frames, as MATLAB's SO3Fun and the harmonic
  projections take them; they exceed reductionGroups for a group with mirrors but without the
  inversion (mm2, -43m, ...), where diffraction cannot tell a crystal from its inverse"""
  L = SS.sym.properGroup().rotations().data.reshape(-1, 4)
  R = CS.sym.properGroup().rotations().data.reshape(-1, 4)
  return L, R


def reductionGroupsPair(a, b):
  """the groups two orientations are compared over: the specimen sides have to fit; two
  crystal frames of different phases reduce over the products r2 * inv(r1) of both crystal
  groups (orientation/dot)"""
  La, Ra = reductionGroups(a)
  if a.frameA is b.frameA:
    return La, Ra
  Lb, Rb = reductionGroups(b)
  prod = qmul(Rb[:, None, :], qconj(Ra)[None, :, :]).reshape(-1, 4)
  return La, uniqueRotations(prod)


# the first of every cluster of a pairwise relation, as a mask: an element stays unless an
# earlier one that stays is the same
def _firstOfClusters(same):
  keep = np.ones(len(same), dtype=bool)
  for k in range(len(same)):
    if keep[k]:
      keep[k + 1:] &= ~same[k, k + 1:]
  return keep


# the distinct rows of unit vectors, the first of every set whose dots exceed 1 - tol
def _distinctUnitRows(X, tol):
  X = np.asarray(X, dtype=float)
  X = X.reshape(-1, X.shape[-1]) if X.size else X.reshape(0, 3)
  return X[_firstOfClusters(X @ X.T > 1 - tol)]


# whether rotations are about z, the identity included
def _isAboutZ(q):
  return (quatAngle(q) < 1e-8) | (np.abs(quatAxis(q)[..., 2]) > 1 - 1e-8)


def uniqueRotations(q, tol=1e-6):
  """distinct rotations up to sign and a tolerance, pairwise, the first of each kept"""
  return q[_firstOfClusters(np.abs(q @ q.T) > 1 - tol)]


def referenceImages(ref, L, R, exchange=False):
  """the images l ref r of a reference, those of its inverse appended under grain exchange,
  each with the (l, r, usedInverse) it came from; for the identity the products l r
  coincide in whole cosets, so only the distinct images are kept"""
  refs = [ref] + ([qconj(ref)] if exchange else [])
  es, frm = [], []
  for useInv, q in enumerate(refs):
    e = qmul(qmul(L[:, None, :], q[None, None, :]), R[None, :, :])        # (|L|, |R|, 4)
    li, rj = np.meshgrid(np.arange(len(L)), np.arange(len(R)), indexing="ij")
    es.append(e.reshape(-1, 4))
    frm.append(np.stack([li.ravel(), rj.ravel(), np.full(li.size, useInv)], axis=-1))
  es, frm = np.concatenate(es), np.concatenate(frm)
  if abs(ref[0]) > 1 - 1e-12:
    keep = _firstOfClusters(np.abs(es @ es.T) > 1 - 1e-10)
    es, frm = es[keep], frm[keep]
  return es, frm


def symDots(qs, ref, L, R, exchange=False):
  """the reduced dot products and angles of many rotations against one reference"""
  es, _ = referenceImages(ref, L, R, exchange)
  return bestDot(qs, es)[0]


def symAngles(qs, ref, L, R, exchange=False):
  """the reduced angles of many rotations to one reference under the two groups L and R, through
  the closest symmetric image of the reference"""
  es, _ = referenceImages(ref, L, R, exchange)
  k = bestDot(qs, es)[1]
  return chordAngle(qs, es[k])


def symPairAngles(q1, q2, L, R, exchange=False):
  """the disorientation angle of each pair, through the closest image for accuracy near zero"""
  return closestAngle(q1, q2, L, R, exchange)


def closestToRef(qs, ref, L, R, exchange=False):
  """every rotation of a list moved to the equivalent closest to one reference, or to its own
  reference of a list, the step the orientation mean iterates"""
  return closestImage(ref, qs, L, R, exchange)[1]


__all__ = []
