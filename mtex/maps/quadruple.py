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

from .criteria import connectivity, _misorientationAngle
from .decomposition import cellComplex
from .segment import segmentFrom


# the one site both of two faces bound, the cell in the wedge between them, or -1, for faces of
# shape (..., 2) of sites
def _commonSite(a, b):
  match = a[..., :, None] == b[..., None, :]
  return np.where(match.sum(axis=(-2, -1)) == 1, np.where(match[..., 0, :].any(axis=-1), a[..., 0], a[..., 1]), -1)


def quadruplePoints(c, seg):
  """the vertices where four faces meet, all four of them a grain boundary, with their faces
  in counterclockwise order"""
  nV = len(c.V)
  deg = np.bincount(c.F.reshape(-1), minlength=nV)
  degOuter = np.bincount(c.F[seg.isOuter].reshape(-1), minlength=nV)
  verts = np.flatnonzero((deg == 4) & (degOuter == 4)).astype(np.int32)
  slot = np.full(nV, -1, dtype=np.int64)
  slot[verts] = np.arange(len(verts))
  # the four faces of every vertex, those it begins before those it ends, each in face order
  r = slot[c.F.T].reshape(-1)
  at = np.flatnonzero(r >= 0)
  faces = np.tile(np.arange(len(c.F), dtype=np.int32), 2)[at[np.argsort(r[at], kind="stable")]].reshape(-1, 4)
  # turned counterclockwise about the vertex
  v = verts[:, None]
  far = np.where(c.F[faces, 0] == v, c.F[faces, 1], c.F[faces, 0])
  d = c.V[far] - c.V[v]
  return verts, np.take_along_axis(faces, np.argsort(np.arctan2(d[..., 1], d[..., 0]), axis=1, kind="stable"), axis=1)


# the misorientation angle of every pair, inf where none is defined, so such a pair never wins a tie
def _pairAngle(ebsd, i, j):
  rot = ebsd.rotations.data.reshape(-1, 4)
  ph = ebsd.phaseId.reshape(-1)
  out = np.full(len(i), np.inf, dtype=rot.dtype)
  for p, fr in enumerate(ebsd.CSList):
    sel = (ph[i] == p) & (ph[j] == p)
    if fr.isIndexed and sel.any():
      out[sel] = _misorientationAngle(fr, rot[i[sel]], rot[j[sel]])
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
  # the cell in every wedge between two neighbouring faces: four distinct measured cells
  cells = c.cells[faces]
  wedge = _commonSite(cells, np.roll(cells, -1, axis=1)).astype(np.int32)
  s = np.sort(wedge, axis=1)
  ok = np.all(wedge >= 0, axis=1) & np.all(s[:, 1:] != s[:, :-1], axis=1) & np.all(sites.pixel[wedge] >= 0, axis=1)
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
  rows = np.arange(n)
  # the first of the two faces that move, the cell between them and the one between the two that stay
  m = np.where(useB, 3, 0)
  wMove, wStay = wedge[rows, m], wedge[rows, (m + 2) % 4]
  # the vertex and its copy pulled apart by sqrt(eps) of the shortest of its four segments
  F4 = c.F[faces]
  delta = np.sqrt(np.finfo(T).eps) * np.linalg.norm(c.V[F4[..., 1]] - c.V[F4[..., 0]], axis=2).min(axis=1)
  p = c.V[verts]
  dMove, dStay = sites.pos[wMove] - p, sites.pos[wStay] - p
  new = (len(c.V) + rows).astype(np.int32)
  V[new] = p + (delta / 2)[:, None] * dMove / np.linalg.norm(dMove, axis=1)[:, None]
  V[verts] = p + (delta / 2)[:, None] * dStay / np.linalg.norm(dStay, axis=1)[:, None]
  # the two faces that move end at the copy instead
  for f in (faces[rows, m], faces[rows, (m + 1) % 4]):
    F[f, np.where(F[f, 0] == verts, 0, 1)] = new
  newF = np.stack([verts, new], axis=1).astype(np.int32)
  newCells = np.stack([wedge[rows, (m + 1) % 4], wedge[rows, (m + 3) % 4]], axis=1)
  c2 = cellComplex(V, np.concatenate([F, newF]), np.concatenate([c.cells, newCells]), sites, c.A, c.plane, c.shape)
  # the criterion has never seen the new pairs, and it alone decides whether the two
  # cells a split joins are one grain
  connect = np.concatenate([seg.connect, connectivity(crit, ebsd, pix[newCells[:, 0]], pix[newCells[:, 1]])])
  return c2, segmentFrom(c2, ebsd, connect)


__all__ = []
