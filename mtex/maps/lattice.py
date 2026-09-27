# the lattice an EBSD map is measured on (MATLAB ebsd.lattice, latticeBasis; Julia lattice.jl, latticefit.jl)
#
#   mapPlane(pts)               an orthonormal frame of the measurement plane
#   lattice(pos)                the basis, the plane and the integer index of every measurement, inferred
#   lattice(pos, basis=(A, shape))  the same with a known basis
#   latticeBasis(unitCell)      the basis, the stencil and the spacing a cell polygon describes
#
# The cell shape is 'square' (four neighbours, a possibly rotated and sheared rectangle)
# or 'hex' (six). The basis holds the two shortest translations, so the stencil follows
# from the shape alone. The basis is inferred by direction, the shape is the number of
# edges of the Voronoi cell, and the index is one checked rounding, grown over the
# neighbour graph only where the check fails.
from __future__ import annotations

import numpy as np

from .polygon import signedArea2, voronoiCell
from ..geometry.display import display, header, numString

SHAPES = ("square", "hex")
STENCIL = {"square": np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int32),
           "hex": np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]], dtype=np.int32)}
# a quarter cell: half of what it takes to round onto the neighbouring site
INDEX_TOL = 0.25


def _xyz(pts):
  from ..geometry.vector3d import vector3d
  p = pts.data if isinstance(pts, vector3d) else np.asarray(pts)
  if not np.issubdtype(p.dtype, np.floating):
    p = p.astype(np.float64)
  return p.reshape(-1, 3)


# ------------------------------------------------------------------ the measurement plane

class mapPlane:
  """the plane of a set of positions with an orthonormal frame: the normal fitted and pointing up,
  e1 the given direction projected into the plane, the origin the specimen origin's foot on it

    plane = mapPlane(pos)
    plane.to2d(xyz), plane.from2d(xy)"""

  # the plane of a set of positions, with e1 along the given direction projected into it
  #
  # The normal is fitted on a sample, since a plane has three degrees of freedom. It
  # points up, towards z where the plane is not vertical, and the origin is the specimen
  # origin's foot on the plane, so the coordinates in the plane are the specimen's own
  # where the map lies in its xy plane.
  def __init__(self, pts, e1=(1.0, 0.0, 0.0)):
    P = _xyz(pts)
    T = P.dtype
    if len(P) >= 3:
      S = P[np.round(np.linspace(0, len(P) - 1, min(len(P), 10000))).astype(int)]
      c = S.mean(axis=0)
      n = np.linalg.svd(S - c, full_matrices=False)[2][-1]
    else:
      n = np.array([0.0, 0.0, 1.0])
    up = n[2] if n[2] != 0 else (n[1] if n[1] != 0 else n[0])
    if up < 0:
      n = -n
    e1 = np.asarray(e1, dtype=float)
    t = e1 - (e1 @ n) * n
    if np.linalg.norm(t) < np.finfo(T).eps ** (2 / 3):
      t = np.cross(n, [1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.cross(n, [0.0, 1.0, 0.0])
    self.e1 = (t / np.linalg.norm(t)).astype(T)
    self.e2 = np.cross(n, self.e1).astype(T)
    self.o = ((P[0] @ n) * n).astype(T)

  @property
  def normal(self):
    """the unit normal of the plane, e1 x e2"""
    return np.cross(self.e1, self.e2)

  @property
  def dtype(self):
    """the floating point type the plane computes in"""
    return self.e1.dtype

  def to2d(self, xyz):
    """the coordinates in the plane of positions of shape (..., 3), as (..., 2)"""
    d = np.asarray(xyz) - self.o
    return np.stack([d @ self.e1, d @ self.e2], axis=-1)

  def from2d(self, xy):
    """the positions of plane coordinates of shape (..., 2)"""
    xy = np.asarray(xy)
    return self.o + xy[..., :1] * self.e1 + xy[..., 1:] * self.e2

  def __repr__(self):
    return header("mapPlane", "normal (" + ", ".join(numString(x, digits=4) for x in self.normal) + ")")


# ------------------------------------------------------------------ the lattice

class lattice:
  """the lattice an EBSD map is measured on: the basis of the two shortest translations, the cell
  shape 'square' or 'hex', the plane and the integer index of every position, inferred from the
  positions or fitted to a known basis

    l = lattice(pos)
    l = lattice(pos, basis=(A, 'hex'))"""

  # the lattice of measured positions: the basis from the differences of neighbouring
  # points, the index by fitting the lattice to all of them
  def __init__(self, pos, basis=None, plane=None):
    P3 = _xyz(pos)
    self.plane = mapPlane(P3) if plane is None else plane
    P = self.plane.to2d(P3)
    if basis is None:
      A0, shape = inferBasis(P)
    else:
      A0, shape = np.asarray(basis[0], dtype=P.dtype), basis[1]
    ij, A, p0 = assignIndex(P, A0)
    lo = ij.min(axis=0)
    self.ij = (ij - lo).astype(np.int32)
    self.A = A.astype(P.dtype)
    self.origin = (p0 + A @ lo).astype(P.dtype)
    self.shape = shape

  @classmethod
  def _build(cls, plane, origin, A, ij, shape):
    l = object.__new__(cls)
    l.plane, l.origin, l.A, l.ij, l.shape = plane, np.asarray(origin), np.asarray(A), np.asarray(ij, dtype=np.int32).reshape(-1, 2), shape
    return l

  @property
  def stencil(self):
    """the index offsets of the neighbours of a cell, four for a square lattice and six for a
    hexagonal one"""
    return STENCIL[self.shape]

  def __len__(self):
    return len(self.ij)

  @property
  def dxy(self):
    """the representative spacing: the mean length of the two basis vectors"""
    return float((np.linalg.norm(self.A[:, 0]) + np.linalg.norm(self.A[:, 1])) / 2)

  def position(self, ij):
    """the physical position of indices of shape (..., 2), exactly as the ideal lattice places them"""
    ij = np.asarray(ij, dtype=self.A.dtype)
    return self.plane.from2d(self.origin + ij @ self.A.T)

  @property
  def unitCell(self):
    """the corners of the cell around a site, relative to it, anticlockwise, shape (k, 3)"""
    return self.plane.from2d(cellCorners(self.A, self.shape)) - self.plane.o

  def subset(self, ids):
    """the same lattice on a subset of its sites, the index counted from the least one kept"""
    ij = self.ij[ids]
    lo = ij.min(axis=0)
    return self._build(self.plane, self.origin + self.A @ lo, self.A, ij - lo, self.shape)

  def __repr__(self):
    tuple_ = lambda v: "(" + ", ".join(numString(x, digits=4, tiny=1e-10 * max(self.dxy, 1)) for x in v) + ")"
    return display(header("lattice", self.shape), [("sites", len(self)), ("spacing", f"{self.dxy:.4g}"),
                                                   ("basis", ", ".join(tuple_(self.A[:, j]) for j in range(2))),
                                                   ("normal", tuple_(self.plane.normal))])


def cellCorners(A, shape):
  """the corners of the cell around a site in the plane, anticlockwise: the parallelogram of
  the basis for a square lattice, the Voronoi cell for a hexagonal one"""
  A = np.asarray(A, dtype=float)
  if shape == "square":
    a1, a2 = A[:, 0] / 2, A[:, 1] / 2
    return np.array([a1 + a2, a2 - a1, -a1 - a2, a1 - a2])
  nb = STENCIL["hex"] @ A.T
  return voronoiCell(np.zeros(2), nb, np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))


def latticeBasis(cell):
  """the basis, the stencil and the spacing a cell polygon of four or six corners describes"""
  V3 = _xyz(cell)
  n = len(V3)
  if n not in (4, 6):
    raise ValueError(f"a unit cell has four or six corners, not {n}")
  pl = mapPlane(V3)
  V = pl.to2d(V3)
  c = V.mean(axis=0)
  # the cell to cell translations are twice the vectors to the edge midpoints
  trans = V + np.roll(V, -1, axis=0) - 2 * c
  A, shape = reduceBasis(trans, "hex" if n == 6 else "square")
  return A, STENCIL[shape], float((np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1])) / 2)


# ------------------------------------------------------------------ inference from positions

def centrePatch(P, m):
  """about m points from a square around the centre of the map"""
  n = len(P)
  if n <= m:
    return P
  lo, hi = P.min(axis=0), P.max(axis=0)
  c = (lo + hi) / 2
  span = np.maximum(hi - lo, np.finfo(P.dtype).eps)
  r = np.sqrt(span[0] * span[1] * m / n) / 2
  for _ in range(8):
    inside = np.flatnonzero((np.abs(P[:, 0] - c[0]) <= r) & (np.abs(P[:, 1] - c[1]) <= r))
    if len(inside) >= m:
      return P[inside[:m]]
    r *= 1.6
  return P[:m]


def inferBasis(P):
  """the shortest two translations between neighbouring points

  The difference vectors to the nearest neighbours cluster on the lattice translations.
  They are clustered by direction and each cluster represented by its median length, never
  by a tolerance on the vector: under a smooth distortion one translation is measured with
  lengths spread over tens of percent, and a direction survives what a tolerance splits."""
  if len(P) < 3:
    raise ValueError("a lattice needs at least three positions")
  Q = centrePatch(P, 2000)
  lo, hi = Q.min(axis=0), Q.max(axis=0)
  span = np.maximum(hi - lo, np.finfo(Q.dtype).eps)
  d0 = max(np.sqrt(span[0] * span[1] / len(Q)), np.finfo(Q.dtype).eps)
  from scipy.spatial import cKDTree
  pairs = cKDTree(Q).query_pairs(1.9 * d0, output_type="ndarray")
  v = Q[pairs[:, 1]] - Q[pairs[:, 0]] if len(pairs) else np.zeros((0, 2), dtype=Q.dtype)
  v = v[np.linalg.norm(v, axis=1) >= 1e-9 * d0]
  if len(v) == 0:
    raise ValueError("the positions have no neighbours within the estimated spacing")
  # every translation occurs as a ± pair: fold into the upper half plane
  flip = (v[:, 1] < 0) | ((v[:, 1] == 0) & (v[:, 0] < 0))
  v[flip] = -v[flip]
  # bin by direction over the half circle, three degrees per bin; a cluster is a bin
  # with its two neighbours, represented by the median of its first shell
  nBin = 60
  b = (np.floor(np.mod(np.arctan2(v[:, 1], v[:, 0]), np.pi) / np.pi * nBin).astype(int)) % nBin
  counts = np.bincount(b, minlength=nBin)
  best = counts.max()
  cand = []
  for k in range(nBin):
    if counts[k] < best / 4 or counts[k] < counts[(k - 1) % nBin] or counts[k] < counts[(k + 1) % nBin]:
      continue
    vs = v[(b == k) | (b == (k - 1) % nBin) | (b == (k + 1) % nBin)]
    ls = np.sort(np.linalg.norm(vs, axis=1))
    shell = 1.5 * ls[max(0, (len(ls) + 9) // 10 - 1)]
    first = vs[np.linalg.norm(vs, axis=1) <= shell]
    ms = np.sort(np.linalg.norm(first, axis=1))
    med = ms[(len(ms) + 1) // 2 - 1]
    cand.append(first[np.argmin(np.abs(np.linalg.norm(first, axis=1) - med))])
  if len(cand) < 2:
    raise ValueError("the positions lie on a line, not on a lattice")
  return reduceBasis(np.array(cand), None)


def _cross2(a, b):
  return a[0] * b[1] - a[1] * b[0]


def reduceBasis(vs, shape):
  """Gauss reduction of a two dimensional basis, and the shape it turns out to have"""
  vs = np.asarray(vs, dtype=float)
  order = np.argsort(np.linalg.norm(vs, axis=1))
  cand = vs[order]
  a1 = cand[0]
  i2 = next((i for i in range(len(cand)) if abs(_cross2(a1, cand[i])) > 0.3 * np.linalg.norm(a1) * np.linalg.norm(cand[i])), None)
  if i2 is None:
    raise ValueError("the positions lie on a line, not on a lattice")
  a2 = cand[i2]
  # replace the longer vector by its shortest translate, until neither shortens
  while True:
    mu = np.round(a1 @ a2 / (a1 @ a1))
    a2 = a2 - mu * a1
    if np.linalg.norm(a2) >= np.linalg.norm(a1):
      break
    a1, a2 = a2, a1
  if a1 @ a2 < 0:
    a2 = -a2
  S = cellShapeOf(a1, a2) if shape is None else shape
  return canonicalBasis(a1, a2, S), S


def canonicalBasis(a1, a2, shape):
  """one basis for one lattice, whichever pair the reduction returned: a1 is the shortest
  translation pointing most along the first axis of the plane, a2 the shortest one turning
  anticlockwise from it by the least angle, so that a1 - a2 is short on a hexagonal lattice"""
  cands = [a1, -a1, a2, -a2]
  if shape == "hex":
    cands += [a1 - a2, a2 - a1]
  cands = np.array(cands)
  lengths = np.linalg.norm(cands, axis=1)
  shortest = cands[lengths <= lengths.min() * 1.1]
  key = np.round(shortest / np.linalg.norm(shortest, axis=1, keepdims=True), 9)
  b1 = shortest[np.lexsort((key[:, 1], key[:, 0]))[-1]]
  cross = cands[:, 0] * b1[1] - cands[:, 1] * b1[0]
  left = cands[-cross > 0.3 * lengths * np.linalg.norm(b1)]
  cosines = np.round(left @ b1 / (np.linalg.norm(left, axis=1) * np.linalg.norm(b1)), 9)
  b2 = left[np.lexsort((-np.linalg.norm(left, axis=1), cosines))[-1]]
  return np.stack([b1, b2], axis=1)


def cellShapeOf(a1, a2):
  """the shape is how many edges the Voronoi cell of the lattice has, four or six, which is
  how many neighbours a measurement has, not whether the translations have equal length.
  A square lattice is exactly degenerate, its four cocircular corners split into slivers
  under any shear, so an edge counts only above a fraction of the spacing"""
  a1, a2 = np.asarray(a1, dtype=float), np.asarray(a2, dtype=float)
  nb = [a1, -a1, a2, -a2, a1 - a2, a2 - a1, a1 + a2, -a1 - a2]
  vs = voronoiCell(np.zeros(2), nb, 2 * max(np.linalg.norm(a1), np.linalg.norm(a2)))
  tol = 0.15 * (np.linalg.norm(a1) + np.linalg.norm(a2)) / 2
  n = int(np.sum(np.linalg.norm(vs - np.roll(vs, -1, axis=0), axis=1) > tol))
  return "hex" if n >= 6 else "square"


# ------------------------------------------------------------------ the integer index

def assignIndex(P, A0):
  """the integer index of every position, and the lattice that was fitted to reach it

  For a rigid lattice the index is one rounding of inv(A) (p - p0). A smooth distortion
  can put a far point more than half a cell from where an affine model places it, so the
  model is fitted to every measurement first and judged by the largest residual: below a
  quarter cell no rounding can have flipped. Only above it is the index grown over the
  neighbour graph, which costs a neighbour query per measurement."""
  fit = affineIndex(P, A0)
  if fit is not None:
    return fit
  return growIndex(P, A0), np.asarray(A0, dtype=P.dtype), P[0]


def affineIndex(P, A0):
  """the index from one affine model, refined and verified, or None where it does not hold"""
  ij, worst = roundOnto(P, P[0], np.linalg.inv(A0))
  if worst < 0:
    return None
  fitted = fitLattice(P, ij)
  if fitted is None:
    return None
  p0, A = fitted
  ij, worst = roundOnto(P, p0, np.linalg.inv(A))
  if not 0 <= worst <= INDEX_TOL:
    return None
  if hasCollision(ij):
    return None
  return ij, A, p0


def roundOnto(P, p0, Ainv):
  """the index of every position under one affine model and how far the worst of them missed
  its site; negative where the model sends a position out of an index's range"""
  f = (P - p0) @ Ainv.T
  if not np.all(np.abs(f) < 1e8):
    return None, -1.0
  ij = np.rint(f)
  return ij.astype(np.int64), float(np.max(np.abs(f - ij))) if len(f) else 0.0


def fitLattice(P, ij):
  """the lattice that fits the positions best for the indices they were given: the least
  squares solution of p = p0 + i a1 + j a2 over all of them"""
  X = np.column_stack([np.ones(len(P), dtype=P.dtype), ij.astype(P.dtype)])
  M, B = X.T @ X, X.T @ P
  # a map that is a single scan line fixes only one direction: a singular normal matrix
  # is that map saying so
  try:
    C = np.linalg.solve(M, B)
  except np.linalg.LinAlgError:
    return None
  if not np.all(np.isfinite(C)) or np.linalg.cond(M) > 1e12:
    return None
  return C[0], C[1:].T


def hasCollision(ij):
  """whether two positions were given the same site; a range far larger than the map is a fit
  that has run away and counts as a collision"""
  n = len(ij)
  if n == 0:
    return False
  lo, hi = ij.min(axis=0), ij.max(axis=0)
  d = hi - lo + 1
  if d[0] * d[1] > 16 * n:
    return True
  # a mark per cell of the bounded box, which is one pass; np.unique hashes and sorts
  lin = (ij[:, 0] - lo[0]) + (ij[:, 1] - lo[1]) * d[0]
  seen = np.zeros(int(d[0] * d[1]), dtype=bool)
  seen[lin] = True
  return int(seen.sum()) < n


def growIndex(P, A):
  """the index of every position grown over the graph of nearest neighbours, level by level

  A single cell step never carries more than a fraction of a smooth distortion, so
  rounding inv(A) dp is exact for it however far the map drifts overall"""
  from scipy.spatial import cKDTree
  n = len(P)
  Ainv = np.linalg.inv(A)
  tree = cKDTree(P)
  r = 1.4 * max(np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1]))
  ij = np.zeros((n, 2), dtype=np.int64)
  seen = np.zeros(n, dtype=bool)
  nComp = 0
  for s in range(n):
    if seen[s]:
      continue
    nComp += 1
    # a further component has no path to the first: its seed is placed by the affine map
    if nComp > 1:
      ij[s] = np.rint(Ainv @ (P[s] - P[0]))
    seen[s] = True
    frontier = np.array([s])
    while len(frontier):
      hits = tree.query_ball_point(P[frontier], r)
      nxt = []
      for k, js in zip(frontier, hits):
        js = np.asarray(js, dtype=np.int64)
        js = js[~seen[js]]
        if len(js) == 0:
          continue
        step = np.rint((P[js] - P[k]) @ Ainv.T).astype(np.int64)
        keep = np.any(step != 0, axis=1)          # a duplicate position stays where it is
        js, step = js[keep], step[keep]
        seen[js] = True
        ij[js] = ij[k] + step
        nxt.append(js)
      frontier = np.concatenate(nxt) if nxt else np.zeros(0, dtype=np.int64)
  return ij


__all__ = ["lattice", "mapPlane", "latticeBasis"]
