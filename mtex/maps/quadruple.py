# quadruple points: the four way contacts a square grid invents (Julia quadruple.jl)
#
#   quadruplePoints(c, seg)                 the vertices where four boundary faces meet
#   splitQuadruplePoints(c, seg, ebsd, crit)   each split into two triple points, the map segmented again
#
# On a square lattice four cells meet at every interior vertex, so a checkerboard of two
# labels puts four boundary segments on one point. Which diagonal is connected is not in
# the data, so removing a quadruple point means deciding it: the vertex is split into two
# triple points, one diagonal pair of cells becomes adjacent, and the criterion judges
# that pair like any other. The pairing is chosen by the measurements alone: the
# criterion first, then the smaller misorientation angle, then where the two pixels lie.
# The new adjacency is a face like any other and the segmentation runs once more over it.
# The two vertices are pulled apart by sqrt(eps) of the shortest segment meeting them, so
# the added segment has a direction.
from __future__ import annotations

import numpy as np

from .criteria import connectivity, _bestDot
from .decomposition import cellComplex
from .segment import segmentFrom


# the one site both faces bound, the cell in the wedge between them, or -1
def _commonSite(cells, fa, fb):
  a, b = cells[fa], cells[fb]
  common = [x for x in a if x in b]
  return int(common[0]) if len(common) == 1 else -1


def quadruplePoints(c, seg):
  """the vertices where four faces meet, all four of them a grain boundary, with their faces
  in counterclockwise order"""
  nV = len(c.V)
  deg = np.bincount(c.F.reshape(-1), minlength=nV)
  degOuter = np.bincount(c.F[seg.isOuter].reshape(-1), minlength=nV)
  verts = np.flatnonzero((deg == 4) & (degOuter == 4)).astype(np.int32)
  slot = np.full(nV, -1, dtype=np.int64)
  slot[verts] = np.arange(len(verts))
  faces = np.zeros((len(verts), 4), dtype=np.int32)
  fill = np.zeros(len(verts), dtype=np.int64)
  for col in range(2):
    r = slot[c.F[:, col]]
    for k in np.flatnonzero(r >= 0):
      faces[r[k], fill[r[k]]] = k
      fill[r[k]] += 1
  for r, v in enumerate(verts):
    p = c.V[v]
    far = np.where(c.F[faces[r], 0] == v, c.F[faces[r], 1], c.F[faces[r], 0])
    d = c.V[far] - p
    faces[r] = faces[r][np.argsort(np.arctan2(d[:, 1], d[:, 0]), kind="stable")]
  return verts, faces


# the misorientation angle of every pair, inf where none is defined, so such a pair never wins a tie
def _pairAngle(ebsd, i, j):
  rot = ebsd.rotations.data.reshape(-1, 4)
  ph = ebsd.phaseId.reshape(-1)
  out = np.full(len(i), np.inf, dtype=rot.dtype)
  for p in range(len(ebsd.CSList)):
    fr = ebsd.CSList[p]
    sel = (ph[i] == p) & (ph[j] == p)
    if fr.isIndexed and sel.any():
      out[sel] = 2 * np.arccos(np.minimum(_bestDot(fr, rot[i[sel]], rot[j[sel]]), 1.0))
  return out


# the last tie break: the ordinate of the lexicographically first of the two positions
def _posKey(a, b):
  first = (b[:, 0] < a[:, 0]) | ((b[:, 0] == a[:, 0]) & (b[:, 1] < a[:, 1]))
  return np.where(first, b[:, 1], a[:, 1])


def splitQuadruplePoints(c, seg, ebsd, crit):
  """the cell complex with every quadruple point of a square grid split into two triple points and the
  map segmented again: the diagonal pair of cells joined is the one the criterion connects more,
  then the one of smaller misorientation, then by position"""
  verts, faces = quadruplePoints(c, seg)
  sites = c.sites
  n0 = len(verts)
  wedge = np.zeros((n0, 4), dtype=np.int32)
  ok = np.zeros(n0, dtype=bool)
  for r in range(n0):
    for m in range(4):
      wedge[r, m] = _commonSite(c.cells, faces[r, m], faces[r, (m + 1) % 4])
    ok[r] = np.all(wedge[r] >= 0) and len(np.unique(wedge[r])) == 4 and np.all(sites.pixel[wedge[r]] >= 0)
  keep = np.flatnonzero(ok)
  if len(keep) == 0:
    return c, seg
  verts, faces, wedge = verts[keep], faces[keep], wedge[keep]
  n = len(verts)
  pix = sites.pixel
  # the two candidate pairings: A joins the cells between faces 1,2 and 3,0 and detaches
  # faces 0,1 from the vertex; B joins those between 0,1 and 2,3
  aL, aR = wedge[:, 1], wedge[:, 3]
  bL, bR = wedge[:, 0], wedge[:, 2]
  cA, cB = connectivity(crit, ebsd, pix[aL], pix[aR]), connectivity(crit, ebsd, pix[bL], pix[bR])
  wA, wB = _pairAngle(ebsd, pix[aL], pix[aR]), _pairAngle(ebsd, pix[bL], pix[bR])
  kA, kB = _posKey(sites.pos[aL], sites.pos[aR]), _posKey(sites.pos[bL], sites.pos[bR])
  useB = (cB > cA) | ((cB == cA) & ((wB < wA) | ((wB == wA) & (kB < kA))))
  T = c.V.dtype
  V = np.concatenate([c.V, np.zeros((n, 2), dtype=T)])
  F = c.F.copy()
  newF = np.zeros((n, 2), dtype=np.int32)
  newCells = np.zeros((n, 2), dtype=np.int32)
  eps = np.sqrt(np.finfo(T).eps)
  for r in range(n):
    v = verts[r]
    m = 3 if useB[r] else 0                    # the first of the two faces that move
    move = (faces[r, m], faces[r, (m + 1) % 4])
    wMove = wedge[r, m]                        # the cell between the two that move
    wStay = wedge[r, (m + 2) % 4]              # the one between the two that stay
    seg4 = np.linalg.norm(c.V[c.F[faces[r], 1]] - c.V[c.F[faces[r], 0]], axis=1).min()
    delta = eps * seg4
    p = c.V[v]
    dMove, dStay = sites.pos[wMove] - p, sites.pos[wStay] - p
    new = len(c.V) + r
    V[new] = p + (delta / 2) * dMove / np.linalg.norm(dMove)
    V[v] = p + (delta / 2) * dStay / np.linalg.norm(dStay)
    for f in move:
      F[f, 0 if F[f, 0] == v else 1] = new
    newF[r] = (v, new)
    newCells[r] = (wedge[r, (m + 1) % 4], wedge[r, (m + 3) % 4])
  c2 = cellComplex(V, np.concatenate([F, newF]), np.concatenate([c.cells, newCells]), sites, c.A, c.plane, c.shape)
  # the criterion has never seen the new pairs, and it alone decides whether the two
  # cells a split joins are one grain
  connect = np.concatenate([seg.connect, connectivity(crit, ebsd, pix[newCells[:, 0]], pix[newCells[:, 1]])])
  return c2, segmentFrom(c2, ebsd, connect)


__all__ = []
