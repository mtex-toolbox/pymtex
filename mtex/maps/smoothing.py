# smoothing a grain boundary (MATLAB @grain2d/smoothBoundary and BoundarySmoothing/;
# Julia smoothing.jl)
#
#   grains.smoothBoundary(), grains.smoothBoundary(iter), grains.smoothBoundary(taubinFilter())
#   grains.smoothBoundary(curvatureFilter(3)), grains.smoothBoundary(huberFilter(3, threshold=3 * degree))
#   grains.smoothBoundary(iter, noSimplify=True, noRefine=True, moveTriplePoints=True, moveOuterBoundary=True)
#   F.smooth(V, net, h), boundaryNetwork(gB, ib)
#
# A measured boundary is the staircase of the pixel grid: every segment runs along one of
# the lattice directions whatever direction the boundary has. It is repaired in three steps:
# the staircase is simplified away, what is left is resampled at equal arc length, and only
# then is the boundary smoothed. Neither of the first two is cosmetic: smoothing the
# staircase itself leaves the directions scattered around the grid axes, and smoothing a
# handful of long chords cuts the corners off the polygon they form. Which algorithm does
# the last step is a boundaryFilter: the iterative ones (Laplace, Taubin) smooth by how
# densely the boundary is sampled; the variational ones (curvature, Huber) state a length
# and mean the same thing at any step size of the map.
from __future__ import annotations

import numpy as np

from ..geometry.units import degree
from .boundaryresampling import _coarsenGrains, _refineGrains, simplifyBoundary
from .grains import grain2d


# ------------------------------------------------------------------ the network

# the network the smoothing runs on: only the vertices the boundaries use, their neighbours
# in compressed form, and which of them may not move. The pool of vertices a boundary
# indexes into is the whole cell complex, most of which no segment touches; compressing
# first is what keeps the sparse system the size of the boundary and not of the map
class boundaryNetwork:

  # the network of one or more boundaries over their shared pool of vertices: a vertex
  # where other than two segments meet is a junction and stays where it is, and so does one
  # on the border of the map, which is not a boundary between grains but the edge of what
  # was measured
  def __init__(self, *gBs, moveTriplePoints=False, moveOuterBoundary=False):
    nV = len(gBs[0].V)
    F = np.concatenate([gB.F for gB in gBs]).reshape(-1, 2)
    deg = np.bincount(F.reshape(-1), minlength=nV)
    used = np.flatnonzero(deg > 0).astype(np.int32)
    at = np.zeros(nV, dtype=np.int32)
    at[used] = np.arange(len(used), dtype=np.int32)
    # the neighbour lists, both ends of every segment sorted by the vertex they belong to
    a, b = at[F[:, 0]], at[F[:, 1]]
    frm, to = np.concatenate([a, b]), np.concatenate([b, a])
    order = np.argsort(frm, kind="stable")
    fixed = np.zeros(len(used), dtype=bool) if moveTriplePoints else deg[used] != 2
    if not moveOuterBoundary:
      for gB in gBs:
        rim = (gB.grainId[:, 0] == 0) | (gB.grainId[:, 1] == 0)
        fixed[at[gB.F[rim]]] = True
    self.ptr = np.concatenate([[0], np.cumsum(deg[used])]).astype(np.int32)
    self.nbr, self.fixed, self.used = to[order].astype(np.int32), fixed, used

  # a network given by its arrays, the neighbours of vertex i being nbr[ptr[i]:ptr[i + 1]]
  @classmethod
  def byArrays(cls, ptr, nbr, fixed, used):
    net = cls.__new__(cls)
    net.ptr, net.nbr = np.asarray(ptr, dtype=np.int32), np.asarray(nbr, dtype=np.int32)
    net.fixed, net.used = np.asarray(fixed, dtype=bool), np.asarray(used, dtype=np.int32)
    return net

  def __len__(self):
    return len(self.ptr) - 1

  @property
  def degree(self):
    return np.diff(self.ptr)

  def nNeighbors(self, i):
    return int(self.ptr[i + 1] - self.ptr[i])

  def neighbors(self, i):
    return self.nbr[self.ptr[i]:self.ptr[i + 1]]

  # the normalised Laplacian D^-1 A - I of the network, as a sparse matrix
  def laplacian(self):
    from scipy.sparse import csr_matrix
    n = len(self)
    deg = self.degree
    rows = np.repeat(np.arange(n), deg)
    w = np.repeat(1.0 / np.maximum(deg, 1), deg)
    A = csr_matrix((w, (rows, self.nbr)), shape=(n, n))
    A.setdiag(A.diagonal() - 1)
    return A.tocsr()

  def __repr__(self):
    return f"boundaryNetwork\n  vertices: {len(self)}, fixed: {int(np.count_nonzero(self.fixed))}"


# ------------------------------------------------------------------ the filters

class boundaryFilter:

  # V are the vertices (n, 2) of the network, h the spacing they are sampled at
  def smooth(self, V, net, h):
    raise NotImplementedError

  # one step of the normalised Laplacian: every free vertex moves by the given rate towards
  # the mean of its neighbours
  @staticmethod
  def laplaceStep(V, L, free, rate):
    out = V.copy()
    out[free] += rate * (L @ V)[free]
    return out

  # solve M V = V0 for the vertices that may move: the fixed ones are eliminated rather
  # than penalised, so they stay exactly where they are and enter through the right hand side
  @staticmethod
  def solveDirichlet(M, V, fixed):
    from scipy.sparse.linalg import spsolve
    free = ~fixed
    if not free.any():
      return V.copy()
    M = M.tocsr()
    rhs = V[free] - M[free][:, fixed] @ V[fixed]
    out = V.copy()
    out[free] = np.asarray(spsolve(M[free][:, free].tocsc(), rhs)).reshape(-1, V.shape[1])
    return out

  def _items(self):
    return []

  def __repr__(self):
    return f"{type(self).__name__}\n" + "\n".join(f"  {k}: {v}" for k, v in self._items())


# repeated local averaging, the default. A Laplacian is a low pass filter with gain 1 - lk,
# below one everywhere but at k = 0, so it shrinks every convex region, without bound; and
# the count of iterations is not a physical quantity, the same call on the same sample
# measured at a finer step does something else. The averaging counts the vertex itself with
# the weight of its degree, (2V + Vl + Vr) / 4 for a degree of two, so lambda acts on half
# the Laplacian and the default of 0.5 is in truth a rate of 0.25
class laplaceFilter(boundaryFilter):

  def __init__(self, iter=1, lambda_=0.5):
    self.iter, self.lambda_ = int(iter), float(lambda_)

  def smooth(self, V, net, h):
    L, free = net.laplacian(), ~net.fixed
    out = np.array(V, dtype=float)
    for _ in range(self.iter):
      out = self.laplaceStep(out, L, free, self.lambda_ / 2)
    return out

  def _items(self):
    return [("iterations", self.iter), ("lambda", self.lambda_)]


# smoothing without the shrinking: every smoothing pass is followed by a slightly larger
# unshrinking one with a negative step, so the gain over the pair, (1 - lk)(1 - mk), is
# about one over the low frequencies and the area is given back. This is approximate; the
# areas move, only not systematically one way (Taubin, SIGGRAPH 1995)
class taubinFilter(boundaryFilter):

  def __init__(self, iter=5, lambda_=0.5, mu=-0.53):
    self.iter, self.lambda_, self.mu = int(iter), float(lambda_), float(mu)

  def smooth(self, V, net, h):
    if self.mu >= 0:
      raise ValueError("mu is the unshrinking step and has to be negative")
    L, free = net.laplacian(), ~net.fixed
    out = np.array(V, dtype=float)
    for _ in range(self.iter):
      out = self.laplaceStep(out, L, free, self.lambda_)
      out = self.laplaceStep(out, L, free, self.mu)
    return out

  def _items(self):
    return [("iterations", self.iter), ("lambda", self.lambda_), ("mu", self.mu)]


# the regularisation weight of a smoothing length, by inverting the gain
# 1 / (1 + 4 alpha sin(w/2)^4) = 1/2 at w = 2 pi h / Lambda; a wavelength shorter than
# twice the vertex spacing is not something the sampling can represent
def _weight(F, h):
  if F.alpha is not None:
    return float(F.alpha)
  Lambda = 4 * h if F.smoothingLength is None else F.smoothingLength
  return 1 / (4 * np.sin(np.pi * h / max(Lambda, 2 * h)) ** 4)


# the smooth boundary as the minimiser of |V - V0|^2 + alpha |L V|^2, where L V is how far
# a vertex sits off the line through its neighbours. There is no iteration count: the
# minimiser is the solution of one sparse system, so the result depends on alpha alone.
# alpha is stated as a length, the wavelength damped to half amplitude, detail finer than
# it goes, detail coarser than it stays, which means the same thing at any step size
class curvatureFilter(boundaryFilter):

  def __init__(self, smoothingLength=None, alpha=None):
    self.smoothingLength = None if smoothingLength is None else float(smoothingLength)
    self.alpha = None if alpha is None else float(alpha)

  def weight(self, h):
    return _weight(self, h)

  def smooth(self, V, net, h):
    from scipy.sparse import identity
    alpha = self.weight(h)
    V = np.array(V, dtype=float)
    if alpha <= 0:
      return V
    L = net.laplacian()
    return self.solveDirichlet(identity(len(net), format="csr") + alpha * (L.T @ L), V, net.fixed)

  def _items(self):
    return [("smoothing length", self.smoothingLength), ("alpha", self.alpha)]


# the same, with an l1/l2 penalty that leaves a faceted boundary faceted: the quadratic
# penalty of the curvature filter rounds every corner; the Huber one is quadratic below a
# threshold and linear above it, so a vertex that turns by more than the threshold is paid
# for once rather than by its square. It is solved by reweighting the same system until
# nothing moves; iterMax is a safety limit, not a tuning knob
class huberFilter(boundaryFilter):

  def __init__(self, smoothingLength=None, threshold=5 * degree, alpha=None, iterMax=50, tol=1e-4):
    self.smoothingLength = None if smoothingLength is None else float(smoothingLength)
    self.threshold, self.alpha = float(threshold), None if alpha is None else float(alpha)
    self.iterMax, self.tol = int(iterMax), float(tol)

  def weight(self, h):
    return _weight(self, h)

  def smooth(self, V, net, h):
    from scipy.sparse import identity, diags
    alpha = self.weight(h)
    V = np.array(V, dtype=float)
    if alpha <= 0:
      return V
    n = len(net)
    L = net.laplacian()
    Id = identity(n, format="csr")
    # a vertex turning by theta sits about h theta / 2 off the line through its neighbours,
    # which is what L V measures
    delta = h * self.threshold / 2
    w = np.ones(n)
    out = V
    for k in range(self.iterMax):
      new = self.solveDirichlet(Id + alpha * (L.T @ diags(w) @ L), V, net.fixed)
      step = float(np.max(np.linalg.norm(new - out, axis=1), initial=0.0))
      out = new
      if k > 0 and step < self.tol * h:
        break
      r = np.linalg.norm(L @ out, axis=1)
      w = np.minimum(1, delta / np.maximum(r, np.finfo(float).eps))
    return out

  def _items(self):
    return [("smoothing length", self.smoothingLength), ("threshold", f"{self.threshold / degree:g}°"), ("alpha", self.alpha),
            ("iterations", self.iterMax)]


# ------------------------------------------------------------------ the three steps

# turn the staircase of a measured boundary into a smooth curve. The simplification and the
# resampling both take their length from the median segment of the boundary as it came out
# of the segmentation, which is the pixel spacing; afterwards the median is the length of
# the straightened runs and means something else. Both change how many segments there are,
# and a resampled segment no longer runs between one pair of pixels, so switch them off
# where ebsdId is read per segment. Smoothing happens in the plane of the map, where the
# vertices already are
def smoothBoundary(grains, iter=1, simplify=None, refine=None, noSimplify=False, noRefine=False, moveTriplePoints=False,
                   moveOuterBoundary=False):
  F = iter if isinstance(iter, boundaryFilter) else laplaceFilter(iter)
  if grains.boundary is None or grains.boundary.isempty:
    return grains
  d0 = float(np.median(grains.boundary.segLength))
  if not (np.isfinite(d0) and d0 > 0):
    return grains
  if not noSimplify and simplify is not False:
    grains = _coarsenGrains(simplifyBoundary, grains, d0 / np.sqrt(2) if simplify is None else simplify)
  if not noRefine and refine is not False:
    grains = _refineGrains(grains, d0 if refine is None else refine)
  net = boundaryNetwork(grains.boundary, grains.innerBoundary, moveTriplePoints=moveTriplePoints, moveOuterBoundary=moveOuterBoundary)
  h = float(np.median(grains.boundary.segLength))
  V = np.array(grains.boundary.V, dtype=float)
  V[net.used] = F.smooth(V[net.used], net, h)
  V = V.astype(grains.boundary.V.dtype, copy=False)
  return grain2d(grains.id, grains.phaseId, grains.CSList, grains.numPixel, grains.meanRotation, grains.GOS, grains.frame, grains.prop,
                 boundary=grains.boundary._with(V=V), innerBoundary=grains.innerBoundary._with(V=V), loops=grains.loops, phaseMap=grains.phaseMap)


grain2d.smoothBoundary = smoothBoundary

__all__ = ["boundaryFilter", "laplaceFilter", "taubinFilter", "curvatureFilter", "huberFilter", "boundaryNetwork"]
