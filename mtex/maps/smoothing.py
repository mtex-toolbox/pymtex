# smoothing a grain boundary (MATLAB @grain2d/smoothBoundary and BoundarySmoothing/;
# Julia smoothing.jl)
#
#   grains.smoothBoundary(), grains.smoothBoundary(iter), grains.smoothBoundary(taubinFilter())
#   grains.smoothBoundary(curvatureFilter(3)), grains.smoothBoundary(huberFilter(3, threshold=3 * degree))
#   grains.smoothBoundary(iter, simplify=False, refine=False, moveTriplePoints=True, moveOuterBoundary=True)
#   F.smooth(V, net, h), boundaryNetwork(gB, ib)
#
# A measured boundary is the staircase of the pixel grid: every segment runs along one of
# the lattice directions whatever direction the boundary has. It is repaired in three steps:
# the staircase is simplified away, what is left is split into pieces of equal length, and only
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
from ..geometry.install import install
from ..geometry.display import aligned


# ------------------------------------------------------------------ the network

# the neighbour lists of the edges E (m, 2) between n vertices in compressed form, the neighbours
# of vertex i being nbr[ptr[i]:ptr[i + 1]]: both ends of every edge sorted by the vertex they
# belong to
def _adjacency(E, n):
  frm, to = np.concatenate([E[:, 0], E[:, 1]]), np.concatenate([E[:, 1], E[:, 0]])
  order = np.argsort(frm, kind="stable")
  ptr = np.concatenate([[0], np.cumsum(np.bincount(frm, minlength=n))]).astype(np.int32)
  return ptr, to[order].astype(np.int32)


class boundaryNetwork:
  """the network the smoothing runs on: only the vertices the boundaries use, their neighbours
  in compressed form, and which of them may not move. The pool of vertices a boundary
  indexes into is the whole cell complex, most of which no segment touches; compressing
  first is what keeps the sparse system the size of the boundary and not of the map"""

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
    fixed = np.zeros(len(used), dtype=bool) if moveTriplePoints else deg[used] != 2
    if not moveOuterBoundary:
      for gB in gBs:
        rim = (gB.grainId[:, 0] == 0) | (gB.grainId[:, 1] == 0)
        fixed[at[gB.F[rim]]] = True
    self.ptr, self.nbr = _adjacency(at[F], len(used))
    self.fixed, self.used = fixed, used

  @classmethod
  def byArrays(cls, ptr, nbr, fixed, used):
    """a network given by its arrays, the neighbours of vertex i being nbr[ptr[i]:ptr[i + 1]]"""
    net = cls.__new__(cls)
    net.ptr, net.nbr = np.asarray(ptr, dtype=np.int32), np.asarray(nbr, dtype=np.int32)
    net.fixed, net.used = np.asarray(fixed, dtype=bool), np.asarray(used, dtype=np.int32)
    return net

  def __len__(self):
    return len(self.ptr) - 1

  @property
  def degree(self):
    """the number of neighbours of every vertex of the network"""
    return np.diff(self.ptr)

  def numNeighbors(self, i):
    """the number of neighbours of vertex i"""
    return int(self.ptr[i + 1] - self.ptr[i])

  def neighbors(self, i):
    """the neighbours of vertex i, indices into the network's vertices"""
    return self.nbr[self.ptr[i]:self.ptr[i + 1]]

  def laplacian(self):
    """the normalised Laplacian D^-1 A - I of the network, as a sparse matrix"""
    from scipy.sparse import csr_matrix, identity
    n = len(self)
    deg = self.degree
    rows = np.repeat(np.arange(n), deg)
    w = np.repeat(1.0 / np.maximum(deg, 1), deg)
    return (csr_matrix((w, (rows, self.nbr)), shape=(n, n)) - identity(n, format="csr")).tocsr()

  def __repr__(self):
    return aligned(f"boundaryNetwork\n  vertices: {len(self)}, fixed: {int(np.count_nonzero(self.fixed))}")


# ------------------------------------------------------------------ the filters

class boundaryFilter:
  """the base of the boundary smoothing filters: smooth(V, net, h) moves the free vertices V of a
  boundary network sampled at the spacing h, the fixed ones stay"""

  def smooth(self, V, net, h):
    """V are the vertices (n, 2) of the network, h the spacing they are sampled at"""
    raise NotImplementedError

  @staticmethod
  def laplaceStep(V, L, free, rate):
    """one step of the normalised Laplacian: every free vertex moves by the given rate towards
    the mean of its neighbours"""
    out = V.copy()
    out[free] += rate * (L @ V)[free]
    return out

  @staticmethod
  def solveDirichlet(M, V, fixed):
    """solve M V = V0 for the vertices that may move: the fixed ones are eliminated rather
    than penalised, so they stay exactly where they are and enter through the right hand side"""
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
    return aligned(f"{type(self).__name__}\n" + "\n".join(f"  {k}: {v}" for k, v in self._items()))


class laplaceFilter(boundaryFilter):
  """repeated local averaging, the default. A Laplacian is a low pass filter with gain 1 - lk,
  below one everywhere but at k = 0, so it shrinks every convex region, without bound; and
  the count of iterations is not a physical quantity, the same call on the same sample
  measured at a finer step does something else. The averaging counts the vertex itself with
  the weight of its degree, (2V + Vl + Vr) / 4 for a degree of two, so lambda acts on half
  the Laplacian and the default of 0.5 is in truth a rate of 0.25"""

  def __init__(self, iter=1, lambda_=0.5):
    self.iter, self.lambda_ = int(iter), float(lambda_)

  def smooth(self, V, net, h):
    """the vertices after iter steps of local averaging at the rate lambda_ / 2, the fixed ones kept"""
    L, free = net.laplacian(), ~net.fixed
    out = np.array(V, dtype=float)
    for _ in range(self.iter):
      out = self.laplaceStep(out, L, free, self.lambda_ / 2)
    return out

  def _items(self):
    return [("iterations", self.iter), ("lambda", self.lambda_)]


class taubinFilter(boundaryFilter):
  """smoothing without the shrinking: every smoothing pass is followed by a slightly larger
  unshrinking one with a negative step, so the gain over the pair, (1 - lk)(1 - mk), is
  about one over the low frequencies and the area is given back. This is approximate; the
  areas move, only not systematically one way (Taubin, SIGGRAPH 1995)"""

  def __init__(self, iter=5, lambda_=0.5, mu=-0.53):
    self.iter, self.lambda_, self.mu = int(iter), float(lambda_), float(mu)

  def smooth(self, V, net, h):
    """the vertices after iter pairs of a smoothing step lambda_ and an unshrinking step mu, the fixed
    ones kept; mu has to be negative"""
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


# the base of the variational filters, whose weight alpha is given or stated as a smoothing length
class _variationalFilter(boundaryFilter):

  def weight(self, h):
    """the regularisation weight alpha: the one given, or the one that damps the smoothing length,
    four times the spacing h by default, to half amplitude"""
    if self.alpha is not None:
      return float(self.alpha)
    # the gain 1 / (1 + 4 alpha sin(w/2)^4) inverted at 1/2 for w = 2 pi h / Lambda; a wavelength
    # shorter than twice the vertex spacing is not something the sampling can represent
    Lambda = 4 * h if self.smoothingLength is None else self.smoothingLength
    return 1 / (4 * np.sin(np.pi * h / max(Lambda, 2 * h)) ** 4)


class curvatureFilter(_variationalFilter):
  """the smooth boundary as the minimiser of |V - V0|^2 + alpha |L V|^2, where L V is how far
  a vertex sits off the line through its neighbours. There is no iteration count: the
  minimiser is the solution of one sparse system, so the result depends on alpha alone.
  alpha is stated as a length, the wavelength damped to half amplitude, detail finer than
  it goes, detail coarser than it stays, which means the same thing at any step size"""

  def __init__(self, smoothingLength=None, alpha=None):
    self.smoothingLength = None if smoothingLength is None else float(smoothingLength)
    self.alpha = None if alpha is None else float(alpha)

  def smooth(self, V, net, h):
    """the minimiser of |V - V0|^2 + alpha |L V|^2 over the free vertices, one sparse system"""
    from scipy.sparse import identity
    alpha = self.weight(h)
    V = np.array(V, dtype=float)
    if alpha <= 0:
      return V
    L = net.laplacian()
    return self.solveDirichlet(identity(len(net), format="csr") + alpha * (L.T @ L), V, net.fixed)

  def _items(self):
    return [("smoothing length", self.smoothingLength), ("alpha", self.alpha)]


class huberFilter(_variationalFilter):
  """the same, with an l1/l2 penalty that leaves a faceted boundary faceted: the quadratic
  penalty of the curvature filter rounds every corner; the Huber one is quadratic below a
  threshold and linear above it, so a vertex that turns by more than the threshold is paid
  for once rather than by its square. It is solved by reweighting the same system until
  nothing moves; maxIter is a safety limit, not a tuning knob"""

  def __init__(self, smoothingLength=None, threshold=5 * degree, alpha=None, maxIter=50, tol=1e-4):
    self.smoothingLength = None if smoothingLength is None else float(smoothingLength)
    self.threshold, self.alpha = float(threshold), None if alpha is None else float(alpha)
    self.maxIter, self.tol = int(maxIter), float(tol)

  def smooth(self, V, net, h):
    """the minimiser of |V - V0|^2 plus alpha times the Huber penalty of the curvature L V, by
    reweighting the sparse system until nothing moves, the fixed vertices kept"""
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
    for k in range(self.maxIter):
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
            ("iterations", self.maxIter)]


# ------------------------------------------------------------------ the three steps

def smoothBoundary(grains, iter=1, simplify=None, refine=None, moveTriplePoints=False,
                   moveOuterBoundary=False):
  """the grains with the pixel staircase of their boundaries turned into smooth curves; MATLAB's
  grain2d.smoothBoundary

    grains = smoothBoundary(grains)                                   # one smoothing step
    grains = smoothBoundary(grains, 5, moveTriplePoints=True)         # five steps, the junctions moving too
    grains = smoothBoundary(grains, 3, simplify=False, refine=False)  # the segments kept, smoothing only
    grains = smoothBoundary(grains, 3, simplify=0.5, refine=1)        # own tolerance and resampling length
    grains = smoothBoundary(grains, taubinFilter())                   # smoothing without shrinking
    grains = smoothBoundary(grains, curvatureFilter(smoothingLength=3))  # a smoothing length

  calcGrains gives a boundary as a staircase of pixel edges, every segment along a grid axis.
  The smoothing repairs it in three steps:

  - simplify: drops every vertex whose removal moves the boundary by less than a tolerance,
  by default d / sqrt(2) for the pixel spacing d, which removes the staircase and nothing else.
  - refine: resamples what is left at equal arc length, by default d, so that the vertices are
  evenly spaced and no longer tied to the grid.
  - the boundaryFilter smooths, by default laplaceFilter(iter), iter steps of local averaging.

  The first two steps matter: simplifying alone leaves a curve as a few long chords whose
  corners the smoothing cuts off, and smoothing alone leaves straight boundaries scattered
  around the grid axes. They change the segments, though, and a resampled segment no longer
  lies between one pair of pixels: where gB.ebsdId is read per segment, as calcGBND does,
  switch them off with simplify=False, refine=False.

  The junctions (triple and quadruple points, and the ends of the inner boundary on the outer
  one) stay fixed unless moveTriplePoints=True, the outer boundary of the map unless
  moveOuterBoundary=True. MATLAB's second output, the filter used, is the one passed or
  laplaceFilter(iter).

  Input
    grains - grain2d
    iter - the number of smoothing steps of the default laplaceFilter, or the boundaryFilter
        to smooth with: laplaceFilter, taubinFilter, curvatureFilter or huberFilter
  Output
    grains - grain2d with the smoothed boundary and inner boundary
  Keywords
    simplify - the tolerance of the simplification in the units of the map, by default
        d / sqrt(2) for the pixel spacing d; False leaves it out, MATLAB's 'noSimplify'
    refine - the resampling length in the units of the map, by default d; False leaves it
        out, MATLAB's 'noRefine'
    moveTriplePoints - smooth the junctions as well
    moveOuterBoundary - smooth the outer boundary of the map as well
  See also
    grain2d.simplifyBoundary, grain2d.refineBoundary, grain2d.reduceBoundary, boundaryFilter,
    laplaceFilter, taubinFilter, curvatureFilter, huberFilter, GrainSmoothing"""
  F = iter if isinstance(iter, boundaryFilter) else laplaceFilter(iter)
  if grains.boundary is None or grains.boundary.isempty:
    return grains
  d0 = float(np.median(grains.boundary.segLength))
  if not (np.isfinite(d0) and d0 > 0):
    return grains
  if simplify is not False:
    grains = _coarsenGrains(simplifyBoundary, grains, d0 / np.sqrt(2) if simplify is None else simplify)
  if refine is not False:
    grains = _refineGrains(grains, d0 if refine is None else refine)
  net = boundaryNetwork(grains.boundary, grains.innerBoundary, moveTriplePoints=moveTriplePoints, moveOuterBoundary=moveOuterBoundary)
  h = float(np.median(grains.boundary.segLength))
  V = np.array(grains.boundary.V, dtype=float)
  V[net.used] = F.smooth(V[net.used], net, h)
  V = V.astype(grains.boundary.V.dtype, copy=False)
  return grain2d(grains.id, grains.phaseId, grains.CSList, grains.numPixel, grains.meanRotation, grains.GOS, grains.frame, grains.prop,
                 boundary=grains.boundary._with(V=V), innerBoundary=grains.innerBoundary._with(V=V), loops=grains.loops, phaseMap=grains.phaseMap)


install(grain2d, smoothBoundary)

__all__ = ["boundaryFilter", "laplaceFilter", "taubinFilter", "curvatureFilter", "huberFilter", "boundaryNetwork"]
