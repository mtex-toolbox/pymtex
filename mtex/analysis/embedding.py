# the isometric embedding of the orientation space (MATLAB geometry/@embedding;
# docs/plans/functions.md P4f)
#
#   e = embedding(ori)                 one embedding per orientation, in one matrix of components
#   double(e), setDouble(e, d)         the Euclidean coordinates and back
#   mean(e), e.norm(), e1 - e2         the algebra of the images
#   e.orientation(), project(e)        the orientation whose image is closest
#
# An orientation is mapped to a tuple of symmetric tensors u_i(g) = w_i g (mean over the
# proper group of (s l_i)^(x alpha_i) - its mean over SO(3)), so that symmetrically
# equivalent orientations have the same image and the Euclidean distance of the images is
# a metric on the quotient (Hielscher and Lippert, arXiv 2007.09664). The vectors l_i, the
# powers alpha_i and the weights w_i are that paper's table restated as a rule on the
# group: the power is the order of the principal axis, the vectors its axis and a two fold
# axis across it where the group has one. The coordinates are taken in an orthonormal
# basis of the invariant subspace the rotation generators build from the identity image,
# computed from the group, so they are isometric and invertible by construction.
from __future__ import annotations

import itertools
import math

import numpy as np

from ..geometry.display import sizeString
from ..geometry.frames import crystalFrame, specimenFrame
from ..geometry.latticetype import latticeType
from ..geometry.orientation import orientation
from ..geometry.quaternion import quaternion, rotation
from ..geometry.vector3d import vector3d
from ..tensors.kernels import turnBlocks

_SPECS = {}


# ------------------------------------------------------------------ the recipe of a crystal frame

# the vectors and powers of the embedding, by the structure of the group
def _terms(cs):
  pg = cs.Laue().properGroup()
  axes, mult = pg.elements()
  axes = axes.data.reshape(-1, 3) if axes.size else np.zeros((0, 3))
  mult = np.asarray(mult)
  z = lambda m: axes[np.flatnonzero(mult == m)[0]] / np.linalg.norm(axes[np.flatnonzero(mult == m)[0]])
  if pg.lattice == latticeType.cubic:
    return ([z(4)], [4], [3 / (2 * math.sqrt(2))]) if 4 in mult else ([z(3)], [3], [3 / (2 * math.sqrt(2))])
  n = int(np.max(mult)) if mult.size else 1
  if n == 1:                                   # no rotation at all: three directions
    return [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])], [1, 1, 1], [1 / math.sqrt(2)] * 3
  if n == 2:
    twofold = axes[mult == 2]
    if len(twofold) >= 6:                      # 222: the three two fold axes (both signs listed)
      u = _distinctAxes(twofold)[:3]
      return [a / np.linalg.norm(a) for a in u], [2, 2, 2], [0.5] * 3
    a = z(2)
    b = vector3d(a).orth().normalize().data.reshape(3)
    return [a, b, np.cross(a, b)], [1, 2, 2], [1 / math.sqrt(2), 0.5, 0.5]
  # a principal axis of order 3, 4 or 6, with a two fold axis across it when the group has one
  c = z(n)
  perp = [k for k in range(len(mult)) if mult[k] == 2 and abs(axes[k] @ c) < 1e-6 * np.linalg.norm(axes[k])]
  dihedral = len(perp) > 0
  if dihedral:
    l1 = axes[perp[0]] / np.linalg.norm(axes[perp[0]])
  else:
    a = cs.aAxis.data.reshape(3)
    l1 = a - (a @ c) * c
    l1 = l1 / np.linalg.norm(l1)
  return [l1, c], [n, 2 if dihedral else 1], _weights(n, dihedral)


def _distinctAxes(a, tol=1e-6):
  out = []
  for v in a:
    u = v / np.linalg.norm(v)
    if not any(abs(u @ w) > 1 - tol for w in out):
      out.append(u)
  return out


# the weights of the paper's table, by the order of the principal axis
def _weights(n, dihedral):
  if n == 3:
    return [math.sqrt(8) / (math.sqrt(2) * 3), math.sqrt(5 / 12) if dihedral else math.sqrt(5 / 6)]
  if n == 4:
    return [1 / math.sqrt(2), 0.5 if dihedral else 1 / math.sqrt(2)]
  if n == 6:
    return [4 / (math.sqrt(2) * 3), 1 / math.sqrt(24) if dihedral else 1 / math.sqrt(12)]
  raise ValueError(f"no embedding is tabulated for a {n} fold principal axis")


# v^(x alpha), flattened
def _powerTensor(v, alpha):
  A = np.array(1.0)
  for _ in range(alpha):
    A = np.multiply.outer(A, v)
  return A.ravel()


# the symmetric tensor delta^(x alpha/2), the only tensor of even rank every rotation fixes
def _isotropicTensor(alpha):
  if alpha % 2:
    return None
  A = np.zeros((3,) * alpha)
  perms = list(itertools.permutations(range(alpha)))
  for idx in np.ndindex(*A.shape):
    A[idx] = sum(all(idx[p[2 * k]] == idx[p[2 * k + 1]] for k in range(alpha // 2)) for p in perms) / len(perms)
  return A.ravel()


# Q on every index of the flattened rank alpha tensors, the rows of X: Q on the first index
# and that index moved last, alpha times, so that every step is one batched product; Q one
# matrix or one per row
def _turn(X, alpha, Q):
  n = len(X)
  for _ in range(alpha):
    X = (Q @ X.reshape(n, 3, -1)).transpose(0, 2, 1).reshape(n, -1)
  return X


# the derivative of the rotation action along a generator: the spin matrix S on one index
# after the other, summed, the terms cycled along with the tensor
def _derive(X, alpha, S):
  n = len(X)
  Y = np.zeros_like(X)
  for _ in range(alpha):
    Y = Y + (S @ X.reshape(n, 3, -1)).reshape(n, -1)
    X = X.reshape(n, 3, -1).transpose(0, 2, 1).reshape(n, -1)
    Y = Y.reshape(n, 3, -1).transpose(0, 2, 1).reshape(n, -1)
  return Y


# f on every term of the images, the rows of U, in chunks that stay in the cache
def _blocks(f, U, alpha, offsets, Q, chunk=256):
  out = np.empty(U.shape)
  for i in range(0, len(U), chunk):
    q = Q if np.ndim(Q) == 2 else Q[i:i + chunk]
    for k, a in enumerate(alpha):
      r = slice(offsets[k], offsets[k + 1])
      out[i:i + chunk, r] = f(U[i:i + chunk, r], a, q)
  return out


# the rows of U, one or n, turned by Q, one matrix or n: the compiled loop, the chunked
# products its fallback and oracle
def _turnRows(U, alpha, offsets, Q):
  out = turnBlocks(U, alpha, offsets, Q)
  if out is not None:
    return out
  n = max(len(U), len(Q) if np.ndim(Q) == 3 else 1)
  return _blocks(_turn, np.broadcast_to(U, (n, U.shape[1])), alpha, offsets, Q)


# the identity image turned by a list of rotations, one row each
def _rotateBatch(u0, alpha, offsets, Q):
  return _turnRows(u0[None], alpha, offsets, Q)


_GENERATORS = [np.array([[0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]), np.array([[0, 0, 1.0], [0, 0, 0], [-1.0, 0, 0]]),
               np.array([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 0]])]


# an orthonormal basis of the invariant subspace the generators build from v
def _orbitBasis(v, alpha, offsets, tol=1e-8):
  B, queue = [], []

  def add(w):
    for b in B:
      w = w - (b @ w) * b
    nw = np.linalg.norm(w)
    if nw <= tol * math.sqrt(len(w)):
      return
    B.append(w / nw)
    queue.append(B[-1])

  add(v.copy())
  while queue:
    w = queue.pop(0)
    for S in _GENERATORS:
      add(_blocks(_derive, w[None], alpha, offsets, S)[0])
  return np.stack(B, axis=1)


class embeddingSpec:
  """the recipe of the embedding of a crystal frame: the identity image and the orthonormal
  basis of the space it moves in, built once per frame"""

  def __init__(self, cs):
    ls, alpha, weights = _terms(cs)
    rots = rotation(cs.Laue().properGroup().rotations().data.reshape(-1, 4)).matrix()
    terms = []
    for l, a, w in zip(ls, alpha, weights):
      u = sum(_powerTensor(Q @ l, a) for Q in rots) / len(rots)
      z = _isotropicTensor(a)
      if z is not None:
        u = u - (z @ u) / (z @ z) * z                      # the mean over SO(3)
      terms.append(w * u)
    self.frame, self.alpha, self.weights = cs, list(alpha), list(weights)
    self.offsets = np.concatenate([[0], np.cumsum([len(t) for t in terms])]).astype(int)
    self.u0 = np.concatenate(terms)
    self.basis = _orbitBasis(self.u0, self.alpha, self.offsets)
    self._seeds = None

  @property
  def dim(self):
    return self.basis.shape[1]

  # the seeds of a projection and their images, built on first use
  def seeds(self):
    if self._seeds is None:
      qs = rotation(_seedRotations(self.frame))
      self._seeds = qs, _rotateBatch(self.u0, self.alpha, self.offsets, qs.matrix())
    return self._seeds


def _spec(cs):
  if not isinstance(cs, crystalFrame):
    raise ValueError("the embedding is defined for a crystal frame")
  key = id(cs)
  if key not in _SPECS:
    _SPECS[key] = embeddingSpec(cs)
  return _SPECS[key]


# ------------------------------------------------------------------ the embedding

class embedding:
  """the isometric embedding of orientations into a Euclidean space: one matrix of
  components with a column per orientation, so that norm(e1 - e2) is a metric on the
  orientations modulo the crystal symmetry

    e = embedding(ori)                 the images of a list, e[i] the image of one
    double(e)                          the coordinates, one row per orientation
    mean(e), e.orientation()           the mean image and the orientation closest to it
  """

  def __init__(self, o=None, *, spec=None, u=None):
    if spec is not None:
      self.spec, self.u = spec, np.asarray(u, dtype=float).reshape(spec.u0.size, -1)
      self.shape = () if self.u.shape[1] == 1 and not isinstance(o, tuple) else (self.u.shape[1],) if o is None else o
      return
    if isinstance(o, crystalFrame):
      self.spec = _spec(o)
      self.u, self.shape = self.spec.u0.reshape(-1, 1), ()
      return
    if not isinstance(o, orientation):
      raise TypeError("the embedding is of orientations")
    self.spec = _spec(o.CS)
    Q = rotation(o.data.reshape(-1, 4)).matrix()
    self.u = _rotateBatch(self.spec.u0, self.spec.alpha, self.spec.offsets, Q).T
    self.shape = o.shape

  def _new(self, u, shape=None):
    e = object.__new__(embedding)
    e.spec, e.u = self.spec, np.asarray(u, dtype=float).reshape(self.spec.u0.size, -1)
    e.shape = (() if e.u.shape[1] == 1 else (e.u.shape[1],)) if shape is None else shape
    return e

  @property
  def CS(self):
    return self.spec.frame

  @property
  def dim(self):
    return self.spec.dim

  @property
  def size(self):
    return self.u.shape[1]

  def __len__(self):
    return self.size

  def __getitem__(self, i):
    idx = np.arange(self.size).reshape(self.shape or (1,))[i]
    return self._new(self.u[:, np.atleast_1d(idx).ravel()], np.shape(idx))

  def _same(self, other):
    if not isinstance(other, embedding) or other.spec is not self.spec:
      raise ValueError("the embeddings belong to different crystal frames")

  def __add__(self, other):
    self._same(other)
    return self._new(self.u + other.u)

  def __sub__(self, other):
    self._same(other)
    return self._new(self.u - other.u)

  def __neg__(self):
    return self._new(-self.u)

  def __mul__(self, s):
    return self._new(self.u * np.asarray(s, dtype=float))

  __rmul__ = __mul__

  def __truediv__(self, s):
    return self._new(self.u / np.asarray(s, dtype=float))

  def dot(self, other):
    self._same(other)
    return np.einsum("ij,ij->j", self.u, other.u).reshape(self.shape)

  # with normalized the norm over the norm every orientation's image has; the sum of squares
  # without a temporary, which is four times faster than np.linalg.norm on a large list
  def norm(self, normalized=False):
    n = np.sqrt(np.einsum("ij,ij->j", self.u, self.u)).reshape(self.shape)
    return n / np.linalg.norm(self.spec.u0) if normalized else n

  def isclose(self, other, atol=1e-10):
    return isinstance(other, embedding) and other.spec is self.spec and self.u.shape == other.u.shape and bool(np.allclose(self.u, other.u, atol=atol))

  def mean(self, weights=None):
    # the mean of nothing is not a number, as the mean of an empty orientation list
    if self.size == 0:
      return self._new(np.full(self.spec.u0.size, np.nan), ())
    w = np.full(self.size, 1 / self.size) if weights is None else np.asarray(weights, dtype=float).ravel() / np.sum(weights)
    return self._new(self.u @ w, ())

  # the images turned by a rotation
  def rotate(self, r):
    Q = rotation(r.data).matrix()
    return self._new(_turnRows(self.u.T, self.spec.alpha, self.spec.offsets, Q).T, self.shape)

  # the images of the seeds of a projection, the batch route
  @staticmethod
  def _images(spec, Q):
    return _rotateBatch(spec.u0, spec.alpha, spec.offsets, Q)

  def __rmul__(self, r):
    if isinstance(r, quaternion):
      return self.rotate(r)
    return self._new(self.u * np.asarray(r, dtype=float))

  # the Euclidean coordinates in the orthonormal basis, one row per orientation; with full
  # the tensor components, repeated ones included
  def double(self, full=False):
    d = self.u if full else self.spec.basis.T @ self.u
    return d[:, 0] if self.shape == () else d.T.reshape(self.shape + (d.shape[0],))

  def setDouble(self, d):
    d = np.asarray(d, dtype=float).reshape(-1, self.dim)
    return self._new(self.spec.basis @ d.T, () if d.shape[0] == 1 else (d.shape[0],))

  # the orientation whose image is closest, by steepest ascent from the best of a few seeds,
  # all images at once; the gradient along a generator G is <G u, t> = -<u, G t>, the action
  # being skew, so G t is formed once per target and every step is one batch of images
  def project(self, seeds=None, maxIter=200, tol=1e-10, chunk=2048):
    sp = self.spec
    if seeds is None:
      qs, seedImages = sp.seeds()
    else:
      qs = rotation(seeds)
      seedImages = _rotateBatch(sp.u0, sp.alpha, sp.offsets, qs.matrix())
    U = self.u.T
    n = np.linalg.norm(U, axis=1)
    out = np.empty((len(U), 4))
    for i in range(0, len(U), chunk):
      Ui, ni = U[i:i + chunk], n[i:i + chunk]
      T = np.where(ni[:, None] > 0, (np.linalg.norm(sp.u0) / np.where(ni > 0, ni, 1))[:, None] * Ui, sp.u0)
      H = np.stack([_blocks(_derive, T, sp.alpha, sp.offsets, S) for S in _GENERATORS], axis=1)
      q = qs.data.reshape(-1, 4)[np.argmax(seedImages @ T.T, axis=0)]
      active = np.arange(len(T))
      for _ in range(maxIter):
        u = _rotateBatch(sp.u0, sp.alpha, sp.offsets, rotation(q[active]).matrix())
        g = -np.einsum("nd,njd->nj", u, H[active])
        moving = np.linalg.norm(g, axis=1) >= tol
        active, g = active[moving], g[moving]
        if not len(active):
          break
        q[active] = (vector3d(g).exp() * rotation(q[active])).data.reshape(-1, 4)
      out[i:i + chunk] = q
    return rotation(out.reshape(self.shape + (4,)) if self.shape != () else out[0])

  def orientation(self, SS=None, **kwargs):
    return orientation(self.project(**kwargs), self.spec.frame, specimenFrame.default if SS is None else SS)

  def __repr__(self):
    ranks = " ".join(str(a) for a in self.spec.alpha)
    return f"embedding\n symmetry: {self.spec.frame.char()}\n ranks: {ranks}\n dim: {self.dim}\n size: {sizeString(self.shape)}"


# a handful of orientations to start from, the coarsest grid the group allows
def _seedRotations(cs):
  from ..geometry.grids.so3grid import equispacedSO3Grid
  return equispacedSO3Grid(cs, points=20).data.reshape(-1, 4)


def double(e, full=False):
  return e.double(full)


def setDouble(e, d):
  return e.setDouble(d)


def project(e, **kwargs):
  return e.project(**kwargs)


__all__ = ["embedding", "embeddingSpec", "double", "setDouble"]
