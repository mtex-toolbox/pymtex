# coincidence site lattices of any crystal lattice
#
# A rotation R is a coincidence rotation of the lattice L when L and RL share a sublattice of
# finite index Sigma. Then Sigma L lies in L and RL, so R takes Sigma times every basis vector
# onto a lattice vector of the same length: the columns of X = Sigma A^-1 R A are integer. The
# search runs over those lattice vectors: the shell of length Sigma |a| for the first basis
# vector, one per orbit of the point group, the shell of Sigma |b| for the second at the angle
# of a and b, and the third follows. Sigma itself is the index of L n RL, read off the Smith
# invariants of the integer matrix X, so a rotation of a smaller Sigma dividing it drops out.
# The lattice is the Bravais lattice: a centred one is searched in a primitive cell of it, and
# Sigma is its index there, so a cubic F or I lattice has the coincidences of the primitive
# cubic one while a centred lattice of lower symmetry has its own.
# Nothing assumes a cubic lattice: a metric with rational ratios has exact coincidences, and any
# other is replaced by the nearest rational metric within a misfit delta, whose coincidences are
# the near coincidences of the real one (the constrained coincidence site lattices of Bonnet,
# Cousineau and Warrington 1981).
from __future__ import annotations

from fractions import Fraction
from math import gcd

import numpy as np

from .frames import CENTERING_BASES
from .orientation import orientation
from .quaternion import rotation
from .symmetryops import projectIntoFundamentalRegionOrientation


# the lattice vectors, as integer coordinates, whose length is r within a relative misfit: for
# every first and third coordinate the second solves a quadratic, so the shell costs its area
def _shell(A, r, delta):
  lo, hi = r * (1 - delta), r * (1 + delta)
  b = np.ceil(hi * np.linalg.norm(np.linalg.inv(A), axis=1) + 1e-9).astype(int)
  i, k = np.meshgrid(np.arange(-b[0], b[0] + 1), np.arange(-b[2], b[2] + 1), indexing="ij")
  i, k = i.ravel(), k.ravel()
  p = np.outer(i, A[:, 0]) + np.outer(k, A[:, 2])
  qa, qb, qc = A[:, 1] @ A[:, 1], p @ A[:, 1], np.sum(p * p, axis=1)
  # the roots of |p + j a2| = R
  def roots(R):
    d = qb * qb - qa * (qc - R * R)
    s = np.sqrt(np.maximum(d, 0))
    return d >= 0, (-qb - s) / qa, (-qb + s) / qa
  outer, o1, o2 = roots(hi)
  inner, i1, i2 = roots(lo)
  eps = 1e-9
  # the j between the outer roots and outside the inner ones: two ranges, or one without inner roots
  starts = [np.ceil(o1 - eps), np.where(inner, np.ceil(i2 - eps), np.inf)]
  stops = [np.where(inner, np.floor(i1 + eps), np.floor(o2 + eps)), np.floor(o2 + eps)]
  out = []
  for st, sp in zip(starts, stops):
    n = np.where(outer, np.maximum(sp - st + 1, 0), 0).astype(int)
    m = n > 0
    rows = np.repeat(np.arange(len(i)), n)
    offset = np.arange(n.sum()) - np.repeat(np.cumsum(n[m]) - n[m], n[m])
    out.append(np.column_stack([i[rows], np.repeat(st[m], n[m]).astype(int) + offset, k[rows]]))
  y = np.unique(np.concatenate(out), axis=0)
  return y[np.abs(np.linalg.norm(y @ A.T, axis=1) - r) <= delta * r + 1e-9]


# one lattice vector per orbit of the point group, given as integer matrices on the coordinates:
# the largest image, read as one integer, names the orbit
def _orbitRepresentatives(y, ops):
  images = np.einsum("gij,nj->ngi", ops, y)
  w = 2 * int(np.abs(images).max()) + 1
  key = ((images[..., 0] * w + images[..., 1]) * w + images[..., 2]).max(axis=1)
  _, first = np.unique(key, return_index=True)
  return y[np.sort(first)]


# orthonormal frames spanned by the pairs (u, v): columns u, the part of v normal to u, their cross
def _frames(u, v):
  e1 = u / np.linalg.norm(u, axis=-1, keepdims=True)
  e2 = v - np.sum(v * e1, axis=-1, keepdims=True) * e1
  e2 /= np.linalg.norm(e2, axis=-1, keepdims=True)
  return np.stack([e1, e2, np.cross(e1, e2)], axis=-1)


# the index of L n (X / sigma) L in L: x lies in it when sigma adj(X) x = 0 mod det X, and the
# image of an integer matrix modulo q has the size prod q / gcd(d_i, q) over its Smith invariants
def _index(X, sigma):
  X = [[int(v) for v in row] for row in X]
  q = (X[0][0] * (X[1][1] * X[2][2] - X[1][2] * X[2][1]) - X[0][1] * (X[1][0] * X[2][2] - X[1][2] * X[2][0])
       + X[0][2] * (X[1][0] * X[2][1] - X[1][1] * X[2][0]))
  if q <= 0:
    return 0
  K = [[sigma * (X[(j + 1) % 3][(i + 1) % 3] * X[(j + 2) % 3][(i + 2) % 3]
                 - X[(j + 1) % 3][(i + 2) % 3] * X[(j + 2) % 3][(i + 1) % 3]) for j in range(3)] for i in range(3)]
  d1 = 0
  for row in K:
    for v in row:
      d1 = gcd(d1, v)
  m2 = 0
  for r in ((0, 1), (0, 2), (1, 2)):
    for c in ((0, 1), (0, 2), (1, 2)):
      m2 = gcd(m2, K[r[0]][c[0]] * K[r[1]][c[1]] - K[r[0]][c[1]] * K[r[1]][c[0]])
  d3 = sigma ** 3 * q * q
  inv = [d1, m2 // d1 if d1 else 0, d3 // m2 if m2 else 0]
  n = 1
  for d in inv:
    n *= q // gcd(d, q)
  return n


# the basis nearest A whose metric has the ratios of small integers within the relative misfit
# delta: A G^-1/2 G'^1/2 has the metric G' and differs from A by a stretch alone
def _rationalBasis(A, delta):
  G = A.T @ A
  g = G / G[0, 0]
  for den in range(1, 1001):
    f = np.array([[float(Fraction(x).limit_denominator(den)) for x in row] for row in g])
    if np.all(np.abs(f - g) <= delta * np.abs(g) + 1e-12):
      break
  else:
    raise ValueError(f"no metric of ratios with denominators up to 1000 lies within {delta} of the lattice")
  def sqrtm(S, p):
    w, V = np.linalg.eigh(S)
    return (V * w ** p) @ V.T
  return A @ sqrtm(G, -0.5) @ sqrtm(f * G[0, 0], 0.5)


# the coincidence rotations of one sigma, as rotation matrices in the crystal's cartesian axes
def _coincidences(A, ops, sigma, delta=1e-9):
  lengths = np.linalg.norm(A, axis=0)
  y1 = _orbitRepresentatives(_shell(A, sigma * lengths[0], delta), ops)
  y2 = _shell(A, sigma * lengths[1], delta)
  u, v = A @ y1.T, A @ y2.T
  # the pairs at the angle of the first two basis vectors
  target = sigma ** 2 * (A[:, 0] @ A[:, 1])
  i, j = np.nonzero(np.abs(u.T @ v - target) <= 3 * delta * sigma ** 2 * lengths[0] * lengths[1])
  if not len(i):
    return np.zeros((0, 3, 3)), np.zeros((0, 3, 3), dtype=int)
  R = _frames(u.T[i], v.T[j]) @ _frames(A[:, 0], A[:, 1]).T
  # the third basis vector lands on a lattice vector
  w = R @ (sigma * A[:, 2])
  y3 = np.linalg.solve(A, w.T).T
  y3r = np.rint(y3)
  ok = np.linalg.norm((y3r - y3) @ A.T, axis=1) <= delta * sigma * lengths[2]
  X = np.stack([y1[i[ok]], y2[j[ok]], y3r[ok].astype(int)], axis=-1)
  return R[ok], X


def CSL(sigma, CS, *args, delta=0):
  """the coincidence site lattice misorientations of order sigma of the lattice of the crystal
  frame cs, MATLAB's CSL

    mori = CSL(3, cs)                # the twin of a cubic lattice
    mori = CSL([3, 5, 7], cs)        # several orders at once
    mori = CSL(5, cs, 'I')           # of the lattice centred I
    mori = CSL(7, cs, delta=0.01)    # near coincidences within a misfit of the metric

  The misorientations are from cs to cs with grain exchange, one per coincidence class,
  ascending in sigma and then in angle: CSL(3, cs) is the twin of a cubic lattice. An order
  without coincidences, such as an even one of a cubic lattice, contributes none.

  Any lattice is searched for the rotations that take sigma times every lattice vector onto a
  lattice vector, where MATLAB reads a table of the cubic ones. The lattice is the Bravais
  lattice of cs, centred as cs.centering says or as a letter given; the cubic P, I and F
  lattices share their coincidences.

  A metric with rational ratios, a cubic one or a hexagonal one with c^2/a^2 rational, has
  exact coincidences. With a misfit delta > 0 the metric is replaced by the one of the
  smallest denominators within the relative misfit delta, and its coincidences are the near
  ones of the real lattice, the constrained coincidence site lattices of a hexagonal metal:
  c^2/a^2 = 2.52 of titanium is 5/2 within delta = 0.01.

  MATLAB's delta, a search radius about the angle or axis of its table, is not this one.

  Input
    sigma - the order, an integer or a list of them
    CS - crystalFrame, whose lattice is searched
    args - a letter P, I, F, A, B, C or R, the centring of the lattice instead of cs.centering
  Output
    mori - orientation from cs to cs, antipodal, one per coincidence class
  Keywords
    delta - the relative misfit of the metric within which rational ratios replace it; 0 keeps
        the metric as it is
  References
    H. Grimmer, W. Bollmann, D. H. Warrington, Coincidence-site lattices and complete
        pattern-shift lattices in cubic crystals, Acta Cryst. A 30 (1974) 197-207
    H. Grimmer, Acta Cryst. A 40 (1984), the coincidence site lattices of any lattice
  See also
    orientation.byAxisAngle, calcMisorientation"""
  sigmas = [int(s) for s in np.atleast_1d(sigma)]
  if any(s < 1 for s in sigmas):
    raise ValueError(f"Sigma is a positive integer, not {sigma}")
  centering = next((a for a in args if isinstance(a, str)), CS.centering)
  if centering not in CENTERING_BASES:
    raise ValueError(f"the centring of a lattice is one of {', '.join(CENTERING_BASES)}, not {centering!r}")
  T = np.asarray(CENTERING_BASES["P" if centering == "R" and abs(CS.abg[2] - 2 * np.pi / 3) > 1e-6 else centering])
  A = (T @ CS.axes.data).T
  if delta > 0:
    A = _rationalBasis(A, delta)
  # the point group acts on the primitive cell by integer matrices, or the centring does not fit it
  ops = np.linalg.solve(A, CS.properGroup().rot.matrix() @ A)
  if not np.allclose(ops, np.rint(ops), atol=1e-6):
    raise ValueError(f"the point group {CS.sym.char()} does not keep a lattice centred {centering} in these axes")
  ops = np.rint(ops).astype(int)
  found, order = [], []
  for s in sorted(set(sigmas)):
    R, X = _coincidences(A, ops, s)
    if not len(R):
      continue
    o = projectIntoFundamentalRegionOrientation(orientation(rotation.byMatrix(R), CS, CS, antipodal=True))
    # one per class: the symmetric equivalents left after the projection are dropped
    keep, left = [], np.arange(o.size)
    while len(left):
      k = left[0]
      same = np.asarray(o[left].angle(o[k])) < 1e-6
      if _index(X[k], s) == s:
        keep.append(k)
      left = left[~same]
    for k in keep:
      found.append(o[k])
      order.append((s, float(o[k].angle())))
  if not found:
    return orientation(rotation(np.zeros((0, 4))), CS, CS, antipodal=True)
  return orientation.cat(*[found[k] for k in sorted(range(len(found)), key=order.__getitem__)])


__all__ = ["CSL"]
