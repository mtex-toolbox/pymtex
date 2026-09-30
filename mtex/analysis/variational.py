# the variational filters: one energy over the graph of the map, quadratic or absolute in
# each term (MATLAB EBSDSmoothing/splineFilter, halfQuadraticFilter, l1TVFilter,
# infimalConvolutionFilter; Julia variational.jl)
#
#   splineFilter(smoothingLength), halfQuadraticFilter(smoothingLength, l1DataFit=, l1TV=, threshold=, eps=)
#   l1TVFilter(smoothingLength), infimalConvolutionFilter(smoothingLength, alpha=, beta=)
#
# The model is min_u sum_p w_p phi_d(|u_p - f_p|) + alpha sum_e phi_r(|(K u)_e|), each penalty
# quadratic or an eps relaxed absolute value, and K the difference along an edge or the
# Laplacian of the graph. MTEX has a class per combination; here it is one solver, called
# with what it minimises. The smoothing is stated as a length, the wavelength damped to half
# amplitude, four steps of the map by default, and alpha follows from it and from the
# lattice, so the same filter means the same thing on a coarse and on a fine map.
from __future__ import annotations

import numpy as np

from ..geometry.units import degree
from ..geometry.display import aligned


class EBSDFilter:
  """the base of the denoising filters of a gridded EBSD map, which replace the orientation of
  every pixel by one computed from its neighbours: smooth(ebsd, F)"""

  def _items(self):
    return []

  def __repr__(self):
    return aligned(f"{type(self).__name__}\n" + "\n".join(f"  {k}: {v}" for k, v in self._items()))


# ------------------------------------------------------------------ the filters

class variationalFilter(EBSDFilter):
  """the base of the variational filters, splineFilter, halfQuadraticFilter, l1TVFilter and
  infimalConvolutionFilter: one energy over the graph of the map, solved by one solver"""
  pass


class splineFilter(variationalFilter):
  """the smoothing spline: the field of least |u - f|^2 + alpha |L u|^2, with L the Laplacian
  of the neighbour graph; one linear system, no iteration"""

  def __init__(self, smoothingLength=None, alpha=None):
    self.smoothingLength = None if smoothingLength is None else float(smoothingLength)
    self.alpha = None if alpha is None else float(alpha)

  def _items(self):
    return [("smoothing length", self.smoothingLength), ("alpha", self.alpha)]


class halfQuadraticFilter(variationalFilter):
  """the filter of Bergmann, Chan, Hielscher, Persch and Steidl: the data term and the
  regularisation each quadratic or an absolute value, and an absolute value keeps a step
  where a square blurs it. eps is the difference at which the penalty turns from a square
  into a line, threshold the one above which two neighbours are not smoothed together at
  all, a subgrain boundary the filter is to leave alone"""

  def __init__(self, smoothingLength=None, alpha=None, l1DataFit=True, l1TV=True, threshold=15 * degree, eps=1e-3, maxIter=100,
               tol=0.001 * degree):
    self.smoothingLength = None if smoothingLength is None else float(smoothingLength)
    self.alpha = None if alpha is None else float(alpha)
    self.l1DataFit, self.l1TV = bool(l1DataFit), bool(l1TV)
    self.threshold, self.eps, self.maxIter, self.tol = float(threshold), float(eps), int(maxIter), float(tol)

  def _items(self):
    return [("smoothing length", self.smoothingLength), ("alpha", self.alpha), ("threshold", f"{self.threshold / degree:g}°"),
            ("data term", "l1" if self.l1DataFit else "l2"), ("TV term", "l1" if self.l1TV else "l2")]


class l1TVFilter(halfQuadraticFilter):
  """the same filter with both norms absolute, the default of the maps layer"""

  def __init__(self, smoothingLength=None, alpha=None, threshold=15 * degree, eps=1e-3, maxIter=100, tol=0.001 * degree):
    super().__init__(smoothingLength, alpha, True, True, threshold, eps, maxIter, tol)


class infimalConvolutionFilter(variationalFilter):
  """the field split into two, u = u1 + u2, one penalised in its differences and one in its
  Laplacian, which keeps both a step and a slope: the infimal convolution of first and
  second order total variation"""

  def __init__(self, smoothingLength=None, alpha=None, beta=None, threshold=15 * degree, eps=1e-3, maxIter=20, tol=0.001 * degree):
    self.smoothingLength = None if smoothingLength is None else float(smoothingLength)
    self.alpha = None if alpha is None else float(alpha)
    self.beta = None if beta is None else float(beta)
    self.threshold, self.eps, self.maxIter, self.tol = float(threshold), float(eps), int(maxIter), float(tol)

  def _items(self):
    return [("smoothing length", self.smoothingLength), ("alpha", self.alpha), ("beta", self.beta), ("threshold", f"{self.threshold / degree:g}°"),
            ("iterations", self.maxIter)]


# ------------------------------------------------------------------ the weight of a length

def graphEigenvalue(l, Lambda):
  """the eigenvalue of the graph Laplacian at a plane wave of wavelength Lambda along the first
  lattice direction, sum over the offsets of 1 - cos(2 pi a1 . A o / Lambda); nothing
  shorter than two steps is sampled at all"""
  A = np.asarray(l.A, dtype=float)
  a1 = A[:, 0] / np.linalg.norm(A[:, 0])
  lam = max(float(Lambda), 2 * np.linalg.norm(A[:, 0]))
  return float(sum(1 - np.cos(2 * np.pi * (a1 @ (A @ np.asarray(o, dtype=float))) / lam) for o in l.stencil))


def regularisation(F, l, order, smoothingLength=None, alpha=None):
  """the weight of one filter on one lattice: what it was given, or what its smoothing length
  comes to, the gain 1 / (1 + alpha lambda^p) being one half at alpha = lambda^-p"""
  alpha = F.alpha if alpha is None else alpha
  if alpha is not None:
    return float(alpha)
  Lambda = F.smoothingLength if smoothingLength is None else smoothingLength
  if Lambda is None:
    Lambda = 4 * np.linalg.norm(np.asarray(l.A[:, 0], dtype=float))
  return 1 / graphEigenvalue(l, Lambda) ** order


# ------------------------------------------------------------------ the solver

class graphCSR:
  """the graph in the form the solver reads: the edges at every pixel in CSR form, with the
  neighbour and the edge, both directions of every edge"""

  def __init__(self, edges, n):
    a, b = edges[:, 0], edges[:, 1]
    frm = np.concatenate([a, b])
    to = np.concatenate([b, a])
    eid = np.tile(np.arange(len(edges)), 2)
    order = np.argsort(frm, kind="stable")
    self.n, self.edges = n, edges
    self.ptr = np.concatenate([[0], np.cumsum(np.bincount(frm, minlength=n))]).astype(np.int64)
    self.adj, self.eid = to[order].astype(np.int64), eid[order].astype(np.int64)

  def difference(self):
    """the difference along the edges and the Laplacian, as sparse matrices"""
    from scipy.sparse import csr_matrix
    ne = len(self.edges)
    rows = np.repeat(np.arange(ne), 2)
    cols = self.edges.reshape(-1)
    return csr_matrix((np.tile([1.0, -1.0], ne), (rows, cols)), shape=(ne, self.n))

  def laplacian(self):
    """the Laplacian of the graph, D.T @ D with D the difference along the edges, a sparse matrix"""
    D = self.difference()
    return (D.T @ D).tocsr()


def irls(G, f, alpha, order, l1Data, l1Reg, threshold, eps, maxIter, tol, isData, pin, comp):
  """iteratively reweighted least squares for the model over the whole graph: an absolute
  value is minimised through the quadratic that touches it, and the surrogate is solved
  exactly at every step, which is what half quadratic minimisation is. The weight is
  eps / sqrt(t^2 + eps^2), one at a difference of zero and eps / |t| above, Huber's
  penalty, so one alpha means one thing whichever of the two penalties is absolute. Where
  the field is pinned the measured pixels are not unknowns at all; where it is not they
  carry the data term; a pixel with no measurement is free and has none. The surrogate is
  solved by the conjugate gradient kernel over the graph, or by SuperLU without numba"""
  n = G.n
  x = f.astype(float).copy()
  # a component stops on its own once nothing in it moves by more than tol, as it would
  # were it solved alone; its pixels are frozen and cost the remaining steps nothing
  comp = np.asarray(comp, dtype=np.int64)
  nc = int(comp.max(initial=0))
  activeComp = np.ones(nc + 1, dtype=bool)
  activeComp[0] = False
  active = activeComp[comp]
  freeBase = active & (~isData if pin else True)
  free = freeBase.copy()
  if not free.any():
    return x
  cg = _cgKernel()
  K = None if cg is not None else (G.difference() if order == 1 else G.laplacian())
  norms, weights = (None, None) if cg is None else (_kernel("residualNorm"), _kernel("irlsWeights"))
  edges = np.ascontiguousarray(G.edges, dtype=np.int64)
  isData = np.ascontiguousarray(isData)
  if cg is not None:
    work = np.zeros((5,) + x.shape)
    t = np.zeros(len(edges) if order == 1 else n)
    wr, wd, b = np.zeros(len(t)), np.zeros(n), np.zeros(x.shape)
  for _ in range(maxIter):
    idx = np.flatnonzero(free)
    prev = x[idx]
    if cg is not None:
      norms(x, int(order), edges, G.ptr, G.adj, active, t)
      weights(t, x, f, isData, bool(pin), bool(l1Data), bool(l1Reg), float(threshold), float(eps), edges, active, wr, wd, b)
      cg(x, b, wd, wr, float(alpha), int(order), G.ptr, G.adj, G.eid, free, active, 1e-7, 2000, work)
    else:
      tt = np.linalg.norm(np.asarray(K @ x), axis=1)
      wr = np.where(tt > threshold, 0.0, eps / np.sqrt(tt ** 2 + eps ** 2) if l1Reg else 1.0)
      wd = np.zeros(n)
      if not pin:
        td = np.linalg.norm(x - f, axis=1)
        wd[isData] = (eps / np.sqrt(td ** 2 + eps ** 2) if l1Data else np.ones(n))[isData]
      y = _solveDirect(K, wd, wr, alpha, wd[:, None] * f, f, x, free, G)
      x[idx] = y[idx]
    if not l1Data and not l1Reg:
      break                                          # a quadratic model is one solve
    moved = np.zeros(nc + 1)
    np.maximum.at(moved, comp[idx], np.max(np.abs(x[idx] - prev), axis=1))
    activeComp &= moved >= tol
    active = activeComp[comp]
    free = freeBase & active
    if not free.any():
      break
  return x


def _kernel(name):
  from ..geometry.kernels import backend
  if backend() != "numba":
    return None
  from ..maps.kernels import pixelKernel
  return pixelKernel(name)


def _cgKernel():
  return _kernel("cg")


# the same surrogate by a direct sparse solve: the free unknowns with the fixed ones on the
# right hand side. A pixel the weights isolate from every measurement has only its
# differences determined; such a component is regularised at one pixel and then shifted so
# that its mean is the one of the previous iterate, which is where the conjugate gradient
# leaves it as well
def _solveDirect(K, wd, wr, alpha, b, f, x, free, G):
  from scipy.sparse import diags
  from scipy.sparse.csgraph import connected_components
  from scipy.sparse.linalg import spsolve
  n = len(wd)
  A = (diags(wd) + alpha * (K.T @ diags(wr) @ K)).tocsr()
  # the components of the graph the positive weights leave, and whether a measurement
  # anchors them
  on = (alpha * wr > 0) if K.shape[0] == len(G.edges) else np.ones(len(G.edges), dtype=bool)
  E = G.edges[on]
  from scipy.sparse import coo_matrix
  nc, lab = connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)), directed=False)
  anchored = np.zeros(nc, dtype=bool)
  np.logical_or.at(anchored, lab, (wd > 0) | ~free)
  loose = np.flatnonzero(~anchored)
  A = A.tolil() if len(loose) else A
  rhs = b.copy()
  firstOf = np.full(nc, -1)
  firstOf[lab[::-1]] = np.arange(n)[::-1]
  for c in loose:
    i = firstOf[c]
    A[i, i] += 1.0
    rhs[i] += x[i]
  A = A.tocsr()
  fixed = ~free
  y = f.astype(float).copy()
  if fixed.any():
    y[free] = np.asarray(spsolve(A[free][:, free].tocsc(), rhs[free] - A[free][:, fixed] @ f[fixed])).reshape(-1, f.shape[1])
  else:
    y = np.asarray(spsolve(A.tocsc(), rhs)).reshape(f.shape)
  for c in loose:
    m = lab == c
    y[m] += x[m].mean(axis=0) - y[m].mean(axis=0)
  return y


def solveField(F, l, edges, f, isData, pin, comp):
  """the field over the whole graph under one filter: the edges of the map, the values f
  (n, d), which pixels carry a measurement, whether they are pinned, and the component of
  every pixel, zero for none"""
  G = graphCSR(np.asarray(edges, dtype=np.int64).reshape(-1, 2), len(f))
  if isinstance(F, splineFilter):
    return irls(G, f, regularisation(F, l, 2), 2, False, False, np.inf, 1.0, 1, 0.0, isData, pin, comp)
  if isinstance(F, halfQuadraticFilter):
    return irls(G, f, regularisation(F, l, 1), 1, F.l1DataFit, F.l1TV, F.threshold, F.eps, F.maxIter, F.tol, isData, pin, comp)
  if isinstance(F, infimalConvolutionFilter):
    # the two fields minimised in turn, one over the differences and one over the Laplacian
    alpha = regularisation(F, l, 1)
    beta = regularisation(F, l, 2, alpha=F.beta)
    u1, u2 = np.zeros(f.shape), np.zeros(f.shape)
    for _ in range(F.maxIter):
      p = irls(G, f - u2, alpha, 1, False, True, F.threshold, F.eps, 20, F.tol, isData, pin, comp)
      q = irls(G, f - p, beta, 2, False, True, F.threshold, F.eps, 20, F.tol, isData, pin, comp)
      delta = max(np.max(np.abs(p - u1)), np.max(np.abs(q - u2)))
      u1, u2 = p, q
      if delta < F.tol:
        break
    return u1 + u2
  raise TypeError(f"{type(F).__name__} is not a variational filter")


__all__ = ["EBSDFilter", "splineFilter", "halfQuadraticFilter", "l1TVFilter", "infimalConvolutionFilter"]
