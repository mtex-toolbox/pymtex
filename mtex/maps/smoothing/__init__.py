"""smoothing a grain boundary (MATLAB @grain2d/smoothBoundary and BoundarySmoothing/;
Julia smoothing.jl)

  grains.smoothBoundary(), grains.smoothBoundary(iter), grains.smoothBoundary(taubinFilter(40))
  grains.smoothBoundary(curvatureFilter(3)), grains.smoothBoundary(huberFilter(3, threshold=3 * degree))
  grains.smoothBoundary(iter, simplify=False, moveOuterBoundary=True)
  F.smooth(V, net, h, project), boundaryNetwork(gB, ib)

A measured boundary is the staircase of the pixel grid: every segment runs along one of
the lattice directions whatever direction the boundary has. It is repaired in two steps: the
vertices are smoothed, every vertex of a pixel edge held in the cell of the pixel centres
around it, and the smooth curve is then simplified. Which algorithm smooths is a
boundaryFilter: the iterative ones (Taubin, Laplace) smooth by how densely the boundary is
sampled; the variational ones (curvature, Huber) state a length and mean the same thing at
any step size of the map."""
from __future__ import annotations

import numpy as np

from ...geometry.hull import convexHull
from ...geometry.units import degree
from ..boundaryresampling import _coarsenGrains, simplifyBoundary
from ..grains import grain2d
from ...geometry.install import install
from ...geometry.display import aligned


# ------------------------------------------------------------------ the network

# no projection: every position is allowed
def _keep(V):
  return V


# the neighbour lists of the edges E (m, 2) between n vertices in compressed form, the neighbours
# of vertex i being nbr[ptr[i]:ptr[i + 1]]: both ends of every edge sorted by the vertex they
# belong to
def _adjacency(E, n):
  frm, to = np.concatenate([E[:, 0], E[:, 1]]), np.concatenate([E[:, 1], E[:, 0]])
  order = np.argsort(frm, kind="stable")
  ptr = np.concatenate([[0], np.cumsum(np.bincount(frm, minlength=n))]).astype(np.int32)
  return ptr, to[order].astype(np.int32)


# the points on the boundary of their convex hull, the ones along its edges included
def _onHull(P):
  h = P[convexHull(P)]
  tol = 1e-9 * max(float(np.abs(P).max(initial=0)), 1.0)
  on = np.zeros(len(P), dtype=bool)
  for a, b in zip(h, np.roll(h, -1, axis=0)):
    on |= np.abs((P[:, 0] - a[0]) * (b[1] - a[1]) - (P[:, 1] - a[1]) * (b[0] - a[0])) <= tol * np.hypot(*(b - a))
  return on


class boundaryNetwork:
  """the network the smoothing runs on: only the vertices the boundaries use, their neighbours
  in compressed form, and which of them may not move. The pool of vertices a boundary
  indexes into is the whole cell complex, most of which no segment touches; compressing
  first is what keeps the sparse system the size of the boundary and not of the map

  Input
    gBs - grainBoundary networks to smooth together
    moveOuterBoundary - let the junctions of the outer boundary and the vertices on the convex
        hull of the network move; default False"""

  # the network of one or more boundaries over their shared pool of vertices: a loose end stays
  # where it is, and so do the junctions of the outer boundary and the vertices on the convex
  # hull, so that the map keeps its extent while a ragged outer boundary is smoothed
  def __init__(self, *gBs, moveOuterBoundary=False):
    nV = len(gBs[0].V)
    F = np.concatenate([gB.F for gB in gBs]).reshape(-1, 2)
    deg = np.bincount(F.reshape(-1), minlength=nV)
    used = np.flatnonzero(deg > 0).astype(np.int32)
    at = np.zeros(nV, dtype=np.int32)
    at[used] = np.arange(len(used), dtype=np.int32)
    fixed = deg[used] < 2
    if not moveOuterBoundary:
      for gB in gBs:
        rim = gB.F[(gB.grainId == 0).any(axis=1)].reshape(-1)
        fixed[at[rim[deg[rim] != 2]]] = True
      fixed |= _onHull(np.asarray(gBs[0].V, dtype=float)[used, :2])
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
  """the base of the boundary smoothing filters: smooth(V, net, h, project) moves the free
  vertices V of a boundary network sampled at the spacing h, the fixed ones stay; project(V)
  moves the vertices back where they may be, after every step of an iterative filter and after
  the solve of a variational one"""

  def smooth(self, V, net, h, project=_keep):
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
  """repeated local averaging. A Laplacian is a low pass filter with gain 1 - lk,
  below one everywhere but at k = 0, so it shrinks every convex region, without bound; and
  the count of iterations is not a physical quantity, the same call on the same sample
  measured at a finer step does something else. The averaging counts the vertex itself with
  the weight of its degree, (2V + Vl + Vr) / 4 for a degree of two, so lambda acts on half
  the Laplacian and the default of 0.5 is in truth a rate of 0.25

  Input
    iter - number of smoothing steps; default 1
    lambda_ - smoothing parameter, with each step moving by lambda_/2 towards the neighbour mean; default 0.5"""

  def __init__(self, iter=1, lambda_=0.5):
    self.iter, self.lambda_ = int(iter), float(lambda_)

  def smooth(self, V, net, h, project=_keep):
    """the vertices after iter steps of local averaging at the rate lambda_ / 2, the fixed ones kept"""
    L, free = net.laplacian(), ~net.fixed
    out = np.array(V, dtype=float)
    for _ in range(self.iter):
      out = project(self.laplaceStep(out, L, free, self.lambda_ / 2))
    return out

  def _items(self):
    return [("iterations", self.iter), ("lambda", self.lambda_)]


class taubinFilter(boundaryFilter):
  """smoothing without the shrinking: every smoothing pass is followed by a slightly larger
  unshrinking one with a negative step, so the gain over the pair, (1 - lk)(1 - mk), is
  about one over the low frequencies and the area is given back. This is approximate; the
  areas move, only not systematically one way (Taubin, SIGGRAPH 1995)

  Input
    iter - number of smoothing and unshrinking step pairs; default 5
    lambda_ - positive smoothing parameter; default 0.5
    mu - negative unshrinking parameter; default -0.53"""

  def __init__(self, iter=5, lambda_=0.5, mu=-0.53):
    self.iter, self.lambda_, self.mu = int(iter), float(lambda_), float(mu)

  def smooth(self, V, net, h, project=_keep):
    """the vertices after iter pairs of a smoothing step lambda_ and an unshrinking step mu, the fixed
    ones kept; mu has to be negative"""
    if self.mu >= 0:
      raise ValueError("mu is the unshrinking step and has to be negative")
    L, free = net.laplacian(), ~net.fixed
    out = np.array(V, dtype=float)
    for _ in range(self.iter):
      out = project(self.laplaceStep(out, L, free, self.lambda_))
      out = project(self.laplaceStep(out, L, free, self.mu))
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
  smoothingLength specifies the spatial wavelength damped to half amplitude. The weight
  alpha is derived from this length and the vertex spacing, so the smoothing scale
  remains comparable between samplings. Supplying alpha overrides that conversion.

  Input
    smoothingLength - spatial wavelength damped to half amplitude, in the units of the boundary
    alpha - dimensionless regularisation weight, overriding smoothingLength"""

  def __init__(self, smoothingLength=None, alpha=None):
    self.smoothingLength = None if smoothingLength is None else float(smoothingLength)
    self.alpha = None if alpha is None else float(alpha)

  def smooth(self, V, net, h, project=_keep):
    """the minimiser of |V - V0|^2 + alpha |L V|^2 over the free vertices, one sparse system"""
    from scipy.sparse import identity
    alpha = self.weight(h)
    V = np.array(V, dtype=float)
    if alpha <= 0:
      return V
    L = net.laplacian()
    return project(self.solveDirichlet(identity(len(net), format="csr") + alpha * (L.T @ L), V, net.fixed))

  def _items(self):
    return [("smoothing length", self.smoothingLength), ("alpha", self.alpha)]


class huberFilter(_variationalFilter):
  """smooth grain boundaries with a Huber curvature penalty that preserves sharp corners

    F = huberFilter(smoothingLength=10, threshold=5*degree)
    smoothed = grains.smooth(F)

  The penalty is quadratic for small boundary turns and linear for turns above threshold.
  This reduces small fluctuations while penalising large corners less strongly than
  curvatureFilter. Reweighted sparse solves minimise the smoothing energy. Fixed network
  vertices remain fixed; stopping uses tol or the maximum iteration count.

  Input
    smoothingLength - spatial wavelength controlling smoothing, in boundary coordinate units;
        defaults to four vertex spacings when alpha is omitted
    threshold - angular transition threshold in radians; default 5*degree
    alpha - dimensionless regularisation weight, overriding smoothingLength
    maxIter - maximum reweighting iterations; default 50
    tol - maximum vertex movement below which iteration stops, in spatial units; default 1e-4
  Output
    F - huberFilter, used by grain and boundary smoothing methods
  See also
    curvatureFilter, laplaceFilter, taubinFilter, grain2d.smooth"""

  def __init__(self, smoothingLength=None, threshold=5 * degree, alpha=None, maxIter=50, tol=1e-4):
    self.smoothingLength = None if smoothingLength is None else float(smoothingLength)
    self.threshold, self.alpha = float(threshold), None if alpha is None else float(alpha)
    self.maxIter, self.tol = int(maxIter), float(tol)

  def smooth(self, V, net, h, project=_keep):
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
    return project(out)

  def _items(self):
    return [("smoothing length", self.smoothingLength), ("threshold", f"{self.threshold / degree:g}°"), ("alpha", self.alpha),
            ("iterations", self.maxIter)]


# ------------------------------------------------------------------ the cells

class pixelCells:
  """where the vertices of a boundary may go while it is smoothed: every vertex of a pixel edge
  stays in its cell, the polygon of the pixel centres around it, a square or on a hexagonal grid
  a triangle, shrunk by margin. Its faces lie at half the edge length, one facing along the
  edge, so the boundary passes every pixel it was measured next to on the side it was
  measured on. A junction without a cell gets one of inradius d/2 - margin, so it stays within
  half a step

    cells = pixelCells(V, E, pixelDist, junctions, d, margin)
    V = cells.project(V)

  Input
    V - the vertices (n, 2) where the cells are centred
    E - the pixel edges (m, 2), indices into V
    pixelDist - the distance of the two pixel centres of every pixel edge
    junctions - the indices of the vertices that get a cell of their own unless they are
        on a pixel edge
    d - the pixel spacing
    margin - how far the cells are shrunk"""

  def __init__(self, V, E, pixelDist, junctions, d, margin):
    t = V[E[:, 1]] - V[E[:, 0]]
    L = np.linalg.norm(t, axis=1)
    # a pixel edge of a square grid is as long as the step, of a hexagonal one sqrt(3) shorter
    self.K = 3 if len(E) and np.median(pixelDist / L) > 1.5 else 4
    ids, first = np.unique(np.concatenate([E[:, 0], E[:, 1]]), return_index=True)
    phi = np.arctan2(t[:, 1], t[:, 0])
    j = np.setdiff1d(junctions, ids)
    self.id = np.concatenate([ids, j]).astype(np.int64)
    self.V0 = np.asarray(V, dtype=float)[self.id]
    self.h = np.concatenate([np.concatenate([L, L])[first] / 2 - margin, np.full(len(j), d / 2 - margin)])
    # the directions of the faces, the first along the edge
    a = np.concatenate([np.concatenate([phi, phi + np.pi])[first], np.zeros(len(j))])[:, None] + 2 * np.pi * np.arange(self.K) / self.K
    self.cos, self.sin = np.cos(a), np.sin(a)

  def project(self, V):
    """the vertices with those outside their cell moved onto its nearest face, as far along it as
    its corners allow"""
    x = V[self.id] - self.V0
    o = x[:, :1] * self.cos + x[:, 1:] * self.sin - self.h[:, None]
    k = np.argmax(o, axis=1)
    r = np.arange(len(k))
    c, s, out = self.cos[r, k], self.sin[r, k], o[r, k] > 0
    w = self.h * np.tan(np.pi / self.K)
    w = np.clip(x[:, 1] * c - x[:, 0] * s, -w, w)
    x[out] = (self.h * np.stack([c, s]) + w * np.stack([-s, c])).T[out]
    V = V.copy()
    V[self.id] = x + self.V0
    return V


# ------------------------------------------------------------------ the two steps

def smoothBoundary(grains, iter=20, simplify=None, moveOuterBoundary=False):
  """the grains with the pixel staircase of their boundaries turned into smooth curves; MATLAB's
  grain2d.smoothBoundary

    grains = smoothBoundary(grains)                                   # 20 Taubin pairs, then simplified
    grains = smoothBoundary(grains, 40)                               # 40 pairs
    grains = smoothBoundary(grains, simplify=False)                   # every segment kept
    grains = smoothBoundary(grains, simplify=0.2)                     # own tolerance of the simplification
    grains = smoothBoundary(grains, laplaceFilter(5))                 # another smoothing
    grains = smoothBoundary(grains, curvatureFilter(smoothingLength=3))  # a smoothing length

  calcGrains gives a boundary as a staircase of pixel edges, every segment along a grid axis.
  The smoothing repairs it in two steps:

  - the vertices are smoothed, by default by iter pairs of taubinFilter steps. After every
  step each vertex of a pixel edge is moved back into its cell, the square of the four pixel
  centres around it, the triangle of three on a hexagonal grid, shrunk by d/20, d being the
  pixel spacing. So the boundary passes every pixel it was measured next to on the side it
  was measured on, and a triple point stays within half a step of where it was. A loose end
  stays put.
  - simplifyBoundary drops every vertex whose removal moves the boundary by less than a
  tolerance, d/10 by default, unless that would bring one of those pixel centres closer than
  d/20 to the boundary or put it on the other side.

  The pixel edges are those with a nonzero gB.pixelDist, which calcGrains records. Grains built
  otherwise have none; then only the junctions are held within half a step. The outer boundary
  is smoothed as well, its junctions and the vertices on the convex hull of the map staying
  fixed unless moveOuterBoundary=True, so that the map keeps its extent.

  The simplification changes the number of boundary segments, and a merged segment no longer
  lies between one pair of pixels: where gB.ebsdId is read per segment, as calcGBND does,
  switch it off with simplify=False. MATLAB's second output, the filter used, is the one passed
  or taubinFilter(iter).

  Input
    grains - grain2d
    iter - the number of Taubin pairs of the default taubinFilter, or the boundaryFilter to
        smooth with: taubinFilter, laplaceFilter, curvatureFilter or huberFilter
  Output
    grains - grain2d with the smoothed boundary and inner boundary
  Keywords
    simplify - the tolerance of the simplification in the units of the map, by default d/10
        for the pixel spacing d; False leaves it out, MATLAB's 'noSimplify'
    moveOuterBoundary - let the junctions of the outer boundary and the vertices on the convex
        hull of the map move as well

  Example
  Compare reconstructed and smoothed boundaries on the small Forsterite sample.
  The sample file is downloaded and cached on first use.

    import matplotlib.pyplot as plt
    from mtex import mtexdata, calcGrains, smoothBoundary, degree, plot, hold

    ebsd = mtexdata('small', verbose=False)['Forsterite']
    grains = calcGrains(ebsd, angle=10*degree, minPixel=5)
    smoothed = smoothBoundary(grains)
    plot(grains.boundary, lineColor='black', lineWidth=2, displayName='Reconstructed')
    hold(True)
    plot(smoothed.boundary, lineColor='red', lineWidth=1, displayName='Smoothed')
    hold(False)
    print('Grains before and after:', grains.size, smoothed.size)
    plt.show()

  Black shows the pixel-derived outlines and red the smoothed geometry. Smoothing changes
  the curves, while the pixel grain ids and the grain membership remain those of the reconstruction.

  See also
    grain2d.simplifyBoundary, grain2d.refineBoundary, grain2d.reduceBoundary, boundaryFilter,
    laplaceFilter, taubinFilter, curvatureFilter, huberFilter, pixelCells, GrainSmoothing"""
  F = iter if isinstance(iter, boundaryFilter) else taubinFilter(iter)
  b, ib = grains.boundary, grains.innerBoundary
  if b is None or b.isempty:
    return grains
  net = boundaryNetwork(b, ib, moveOuterBoundary=moveOuterBoundary)
  V0 = np.asarray(b.V, dtype=float)
  # the pixel edges and the pixel spacing d
  E, pd = np.concatenate([b.F, ib.F]), np.concatenate([b.pixelDist, ib.pixelDist])
  E, pd = E[pd > 0], pd[pd > 0]
  d = float(np.median(pd)) if len(pd) else float(np.median(b.segLength))
  at = np.zeros(len(V0), dtype=np.int64)
  at[net.used] = np.arange(len(net.used))
  cells = pixelCells(V0[net.used], at[E], pd, np.flatnonzero((net.degree > 2) & ~net.fixed), d, d / 20)
  V = V0.copy()
  V[net.used] = F.smooth(V0[net.used], net, float(np.median(b.segLength)), cells.project)
  V = V.astype(b.V.dtype, copy=False)
  grains = grain2d(grains.id, grains.phaseId, grains.CSList, grains.numPixel, grains.meanRotation, grains.GOS, grains.frame, grains.prop,
                   boundary=b._with(V=V), innerBoundary=ib._with(V=V), loops=grains.loops, phaseMap=grains.phaseMap)
  if simplify is False:
    return grains
  return _coarsenGrains(simplifyBoundary, grains, d / 10 if simplify is None else simplify, margin=d / 20, measuredV=V0)


install(grain2d, smoothBoundary)

__all__ = ["boundaryFilter", "laplaceFilter", "taubinFilter", "curvatureFilter", "huberFilter", "boundaryNetwork", "pixelCells"]
