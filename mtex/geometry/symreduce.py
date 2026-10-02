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

from .kernels import bestDot, closestAngle, closestImage, symReduce
from .quaternion import qmul

_ID = np.array([1.0, 0.0, 0.0, 0.0])


def _conj(q):
  return q * [1, -1, -1, -1]


def reductionGroups(o):
  """the proper rotations each side reduces over: the rotation parts of the Laue group when
  either side is centrosymmetric, the proper subgroup otherwise, as symmetry-reduced
  distances in MTEX have always done it"""
  a, b = o.frameA.sym, o.frameB.sym
  laue = a.isLaue or b.isLaue
  R = (a.properGroup() if laue else a.properSubGroup()).rotations().data.reshape(-1, 4)   # acts on the right, crystal side
  L = (b.properGroup() if laue else b.properSubGroup()).rotations().data.reshape(-1, 4)   # acts on the left, specimen side
  return L, R


def reductionGroupsPair(a, b):
  """the groups two orientations are compared over: the specimen sides have to fit; two
  crystal frames of different phases reduce over the products r2 * inv(r1) of both crystal
  groups (orientation/dot)"""
  La, Ra = reductionGroups(a)
  if a.frameA is b.frameA:
    return La, Ra
  Lb, Rb = reductionGroups(b)
  prod = qmul(Rb[:, None, :], _conj(Ra)[None, :, :]).reshape(-1, 4)
  return La, uniqueRotations(prod)


def uniqueRotations(q, tol=1e-6):
  """distinct rotations up to sign and a tolerance, pairwise, the first of each kept"""
  same = np.abs(q @ q.T) > 1 - tol
  keep = np.ones(len(q), dtype=bool)
  for k in range(len(q)):
    if keep[k]:
      keep[k + 1:] &= ~same[k, k + 1:]
  return q[keep]


def chordAngle(p, q):
  """the chord angle |p -/+ q| = 2 sin(w/4) between rotations, accurate over the whole range"""
  s = np.where(np.sum(p * q, axis=-1) < 0, -1.0, 1.0)[..., None]
  h = np.linalg.norm(p - s * q, axis=-1)
  return 4 * np.arcsin(np.minimum(h / 2, 1.0))


def referenceImages(ref, L, R, exchange=False):
  """the images l ref r of a reference, those of its inverse appended under grain exchange,
  each with the (l, r, usedInverse) it came from; for the identity the products l r
  coincide in whole cosets, so only the distinct images are kept"""
  refs = [ref] + ([_conj(ref)] if exchange else [])
  es, frm = [], []
  for useInv, q in enumerate(refs):
    e = qmul(qmul(L[:, None, :], q[None, None, :]), R[None, :, :])        # (|L|, |R|, 4)
    li, rj = np.meshgrid(np.arange(len(L)), np.arange(len(R)), indexing="ij")
    es.append(e.reshape(-1, 4))
    frm.append(np.stack([li.ravel(), rj.ravel(), np.full(li.size, useInv)], axis=-1))
  es, frm = np.concatenate(es), np.concatenate(frm)
  if abs(ref[0]) > 1 - 1e-12:
    same = np.abs(es @ es.T) > 1 - 1e-10
    keep = np.ones(len(es), dtype=bool)
    for k in range(len(es)):
      if keep[k]:
        keep[k + 1:] &= ~same[k, k + 1:]
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
