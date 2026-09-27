# segmentation: which cells of the decomposition form a grain (Julia segment.jl)
#
#   segment(c, ebsd, crit)            the criterion on every face, and the grains it leaves joined
#   faceConnectivity(c, ebsd, crit)   the criterion on every face of the complex
#   segmentFrom(c, ebsd, connect)     the segmentation of a complex whose criterion is evaluated
#   components(n, a, b)               the connected components of a graph, by union-find
#
# The criterion answers per face of the complex; the faces it does not separate are the
# edges of a graph over the measurements, and the grains are its connected components.
# The complex is an explicit edge list, so this is union-find over the pixels and nothing
# else. The grain ids count from 1, and 0 is a measurement without a grain, as in MATLAB.
from __future__ import annotations

import numpy as np

from ..geometry.session import getMTEXpref
from .criteria import connectivity
from .decomposition import NO_PIXEL


def components(n, a, b, backend=None):
  """the component of every node of a graph on n nodes with edges (a, b), labelled by a root"""
  backend = getMTEXpref("kernels", "numba") if backend is None else backend
  if backend == "numba":
    from .kernels import unionFind
    out = unionFind(n, a, b)
    if out is not None:
      return out
  from scipy.sparse import coo_matrix
  from scipy.sparse.csgraph import connected_components
  a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
  g = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
  return connected_components(g, directed=False)[1].astype(np.int32)


class segmentation:

  def __init__(self, grainId, nGrains, connect, isInner, isOuter):
    self.grainId, self.nGrains, self.connect, self.isInner, self.isOuter = grainId, nGrains, connect, isInner, isOuter

  def __repr__(self):
    return f"segmentation\n  grains: {self.nGrains}\n  faces: {len(self.connect)}\n  inner faces: {int(self.isInner.sum())}\n  outer faces: {int(self.isOuter.sum())}"


def faceConnectivity(c, ebsd, crit):
  """the criterion on every face of the complex, zero where there is no pair to judge: a face
  against a site that carries no measurement is a border of the map"""
  p1, p2 = c.site2pixel[c.cells[:, 0]], c.site2pixel[c.cells[:, 1]]
  inner = (p1 >= 0) & (p2 >= 0)
  connect = np.zeros(len(c), dtype=float)
  if inner.any():
    connect[inner] = connectivity(crit, ebsd, p1[inner], p2[inner])
  return connect


def segmentFrom(c, ebsd, connect, backend=None, mcl=None):
  """the connected components of the faces the criterion joins, and the flags that say what
  every face became; the grains are numbered over the measurements that have a cell, in
  order of first appearance, so an absorbed pixel does not open a grain of its own. With
  mcl = (p, maxIter) the links go through Markov clustering first (MATLAB's 'mcl'): the
  grains are the components of what it keeps, and a pair it keeps no link for is a boundary"""
  n = ebsd.size
  p1, p2 = c.site2pixel[c.cells[:, 0]], c.site2pixel[c.cells[:, 1]]
  connect = np.asarray(connect, dtype=float)
  join = (p1 >= 0) & (p2 >= 0) & (connect > 0)
  if mcl is not None:
    a, b, cut = _mclLinks(n, p1, p2, connect, join, mcl)
    raw = components(n, a, b, backend)
  else:
    raw = components(n, p1[join], p2[join], backend)
  hasCell = np.zeros(n, dtype=bool)
  hasCell[c.site2pixel[c.site2pixel >= 0]] = True
  grainId, nGrains = relabel(raw, hasCell, backend)
  g1 = np.where(p1 >= 0, grainId[np.maximum(p1, 0)], 0)
  g2 = np.where(p2 >= 0, grainId[np.maximum(p2, 0)], 0)
  outside = (p1 < 0) | (p2 < 0)
  isOuter = (outside & ((p1 >= 0) | (p2 >= 0))) | (~outside & (g1 != g2))
  isInner = ~outside & (g1 == g2) & (cut if mcl is not None else connect <= 0.5)
  return segmentation(grainId, nGrains, connect, isInner, isOuter)


# the links Markov clustering keeps of the criterion's weighted graph over the neighbour
# pairs, each pair once from the lower to the higher measurement as MATLAB enters it, and
# for every face whether its pair lost its link
def _mclLinks(n, p1, p2, connect, join, mcl):
  from scipy.sparse import coo_matrix
  from .variantgraph import mclComponents
  p, maxIter = (float(mcl), 4) if np.ndim(mcl) == 0 else (float(mcl[0]), mcl[1] if len(mcl) > 1 else 4)
  lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
  A = coo_matrix((connect[join], (lo[join], hi[join])), shape=(n, n)).tocsr()
  A.sum_duplicates()
  M = mclComponents(A, p, maxIter).tocoo()
  inside = (p1 >= 0) & (p2 >= 0)
  linked = np.zeros(len(p1), dtype=bool)
  if inside.any():
    Mc = M.tocsr()
    linked[inside] = np.asarray(Mc[lo[inside], hi[inside]]).reshape(-1) != 0
  keep = M.data != 0
  return M.row[keep], M.col[keep], ~linked


def relabel(raw, hasCell, backend=None):
  """the grains numbered from 1 in order of the first measurement of their component, over the
  measurements that have a cell"""
  n = len(raw)
  if (getMTEXpref("kernels", "numba") if backend is None else backend) == "numba":
    from .kernels import grainKernel
    loop = grainKernel("relabel")
    if loop is not None:
      grainId = np.zeros(n, dtype=np.int32)
      nGrains = loop(np.ascontiguousarray(raw, dtype=np.int64), hasCell, np.zeros(n, dtype=np.int64), grainId)
      return grainId, int(nGrains)
  grainId = np.zeros(n, dtype=np.int32)
  _, first, inverse = np.unique(raw[hasCell], return_index=True, return_inverse=True)
  grainId[hasCell] = np.argsort(np.argsort(first))[inverse] + 1
  return grainId, len(first)


def segment(c, ebsd, crit, backend=None, mcl=None):
  return segmentFrom(c, ebsd, faceConnectivity(c, ebsd, crit), backend, mcl)


__all__ = []
