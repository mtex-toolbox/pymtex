# the surfaces of 3D grains smoothed, coarsened and refined (MATLAB grain3d/smoothBoundary,
# reduceBoundary and refineBoundary, ../master)
#
#   grains = smoothBoundary(grains, iter)               a laplaceFilter(iter) on the vertices
#   grains = smoothBoundary(grains, taubinFilter(5))    any boundary filter of mtex.maps
#   grains = reduceBoundary(grains, n)                  the vertices within cells of n edges merged
#   grains = refineBoundary(grains, n)                  every triangle split into four, n times
#
# The filters are the 2D ones, run on the graph of the edges of the triangles. By default the
# smoothing is hierarchical, as MATLAB's: the triple lines are smoothed first along themselves,
# then held while the faces are smoothed; the vertices where four or more grains meet stay,
# and so do the vertices on the surface of the volume unless moveOuterBoundary.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import columnMax, columnMin
from ..maps.smoothing import boundaryFilter, boundaryNetwork, laplaceFilter


# the network of edges over a subset of the vertices, the fixed ones given over all vertices
def _network(E, verts, fixed):
  at = np.full(int(max(E.max(initial=0), verts.max(initial=0))) + 1, -1, dtype=np.int64)
  at[verts] = np.arange(len(verts))
  a, b = at[E[:, 0]], at[E[:, 1]]
  frm, to = np.concatenate([a, b]), np.concatenate([b, a])
  order = np.argsort(frm, kind="stable")
  deg = np.bincount(frm, minlength=len(verts))
  ptr = np.concatenate([[0], np.cumsum(deg)])
  return boundaryNetwork.byArrays(ptr, to[order], fixed[verts], verts)


def _smoothBoundary3d(grains, iter=1, F=None, moveOuterBoundary=False, fixTripleLines=False, scheme="hierarchical",
                   maxDisplacement=np.inf):
  if isinstance(iter, boundaryFilter):
    F, iter = iter, 1
  F = laplaceFilter(iter) if F is None else F
  gB = grains.boundary
  V0 = gB.allV
  V = V0.copy()
  E, F2E = gB.edges()
  t = gB.nodeType()
  isTL = np.zeros(len(V), dtype=bool)
  isTL[E[np.bincount(F2E.reshape(-1), minlength=len(E)) >= 3].reshape(-1)] = True
  fixed = (t % 10) >= 4
  if not moveOuterBoundary:
    fixed |= t >= 10
  h = float(np.median(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)))
  if scheme == "hierarchical":
    if not fixTripleLines:
      tl = np.flatnonzero(isTL)
      Etl = E[np.all(isTL[E], axis=1)]
      if len(tl):
        V[tl] = F.smooth(V[tl], _network(Etl, tl, fixed), h)
    fixed = fixed | isTL
  used = np.unique(E)
  V[used] = F.smooth(V[used], _network(E, used, fixed), h)
  V = V0 + np.clip(V - V0, -maxDisplacement, maxDisplacement)
  out = gB[np.arange(len(gB))]
  out.allV = V
  return grains._new(np.arange(len(grains)), grains.I_GF, out)


# the vertices merged within the cells of a grid of n shortest edges, a vertex joining only
# the vertices of the same grains and the same sides of the volume, a vertex of more than
# four grains staying; the faces that collapse go, and of faces that come to lie on each other
# the net orientation is kept
def _reduceBoundary3d(grains, n=2, cell=None, quadric=False):
  from scipy.sparse import csr_matrix
  gB = grains.boundary
  V = gB.allV
  E, _ = gB.edges()
  h = n * float(np.min(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1))) if cell is None else float(cell)
  lo, hi = columnMin(V), columnMax(V)
  onLo, onHi = np.abs(V - lo) < 1e-6 * h, np.abs(V - hi) < 1e-6 * h
  I_VG = gB.I_VG.tocsr()
  I_VG.sort_indices()
  deg = np.diff(I_VG.indptr)
  gTuple = np.zeros((len(V), 4), dtype=np.int64)
  k = np.arange(len(I_VG.indices)) - np.repeat(I_VG.indptr[:-1], deg)
  rows = np.repeat(np.arange(len(V)), deg)
  first4 = k < 4
  gTuple[rows[first4], k[first4]] = I_VG.indices[first4] + 1
  key = np.column_stack([np.floor((V - lo) / h), gTuple, onLo @ [1, 2, 4] + onHi @ [8, 16, 32]])
  many = deg > 4
  key[many] = 0
  key[many, 0] = -np.arange(1, int(many.sum()) + 1)
  _, c = np.unique(key, axis=0, return_inverse=True)
  c = c.reshape(-1)
  nC = c.max() + 1
  cnt = np.bincount(c, minlength=nC)
  Vc = np.stack([np.bincount(c, weights=V[:, j], minlength=nC) for j in range(3)], axis=1) / cnt[:, None]
  if quadric:
    Vc = _quadricPlacement(Vc, c, gB)
  for j in range(3):
    Vc[np.bincount(c, weights=onLo[:, j], minlength=nC) > 0, j] = lo[j]
    Vc[np.bincount(c, weights=onHi[:, j], minlength=nC) > 0, j] = hi[j]
  F = c[gB.F]
  ok = np.flatnonzero(np.all(np.diff(np.sort(F, axis=1), axis=1) != 0, axis=1))
  S = np.sort(F[ok], axis=1)
  f0, f1 = F[ok, 0], F[ok, 1]
  even = ((f0 == S[:, 0]) & (f1 == S[:, 1])) | ((f0 == S[:, 1]) & (f1 == S[:, 2])) | ((f0 == S[:, 2]) & (f1 == S[:, 0]))
  g = gB.grainId[ok]
  sgn = (2 * even - 1) * (1 - 2 * (g[:, 0] > g[:, 1]))
  _, first, gid = np.unique(np.column_stack([S, np.sort(g, axis=1)]), axis=0, return_index=True, return_inverse=True)
  net = np.bincount(gid.reshape(-1), weights=sgn)
  nz = net != 0
  keep = ok[first[nz]]
  doFlip = np.sign(net[nz]) != sgn[first[nz]]
  out = gB[keep]
  Fk = F[keep]
  Fk[doFlip] = Fk[doFlip][:, ::-1]
  out._idx = Fk.reshape(-1)
  out = out.flip(np.flatnonzero(doFlip))
  out.id = np.arange(1, len(keep) + 1)
  out.allV = Vc
  I = grains.I_GF[:, keep].tocsc()
  sign = np.where(doFlip, -1.0, 1.0)
  I = csr_matrix(I @ csr_matrix((sign, (np.arange(len(keep)), np.arange(len(keep)))), shape=(len(keep), len(keep))))
  return grains._new(np.arange(len(grains)), I, out)


# the vertex of every cluster placed where the planes of its faces meet in the least squares
# sense, drawn towards the mean by a small regularisation
def _quadricPlacement(Vc, c, gB):
  # a face of no area has no plane
  N = np.nan_to_num(gB.N.data)
  d = -np.sum(N * gB.centroid.data, axis=1)
  cF = c[gB.F].reshape(-1)
  f = np.repeat(np.arange(len(gB)), 3)
  nC = len(Vc)
  A = np.zeros((nC, 3, 3))
  b = np.zeros((nC, 3))
  for i in range(3):
    b[:, i] = np.bincount(cF, weights=-d[f] * N[f, i], minlength=nC)
    for j in range(i, 3):
      A[:, i, j] = A[:, j, i] = np.bincount(cF, weights=N[f, i] * N[f, j], minlength=nC)
  # a cluster no face touches, of a vertex a selection left unused, stays at its mean
  reg = 1e-3 * np.trace(A, axis1=1, axis2=2)
  reg[reg == 0] = 1
  A += reg[:, None, None] * np.eye(3)
  b += reg[:, None] * Vc
  return np.linalg.solve(A, b[:, :, None])[:, :, 0]


# every triangle split into four at the midpoints of its edges, n times
def _refineBoundary3d(grains, n=1):
  for _ in range(n):
    gB = grains.boundary
    E, F2E = gB.edges()
    nV, nF = len(gB.allV), len(gB)
    T = gB.F
    M = nV + F2E
    Fn = np.stack([np.column_stack([T[:, 0], M[:, 0], M[:, 2]]), np.column_stack([M[:, 0], T[:, 1], M[:, 1]]),
                   np.column_stack([M[:, 2], M[:, 1], T[:, 2]]), M], axis=1).reshape(-1, 3)
    rep = np.repeat(np.arange(nF), 4)
    out = gB[rep]
    out._idx = Fn.reshape(-1)
    out.id = np.arange(1, 4 * nF + 1)
    out.allV = np.concatenate([gB.allV, (gB.allV[E[:, 0]] + gB.allV[E[:, 1]]) / 2])
    grains = grains._new(np.arange(len(grains)), grains.I_GF[:, rep], out)
  return grains


# the flat smoothBoundary, reduceBoundary and refineBoundary reach these through the methods
# of grain3d
__all__ = []
