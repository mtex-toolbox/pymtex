# quadruple points: the four way contacts a square grid invents (Julia quadruple.jl)
#
#   quadruplePoints(c, seg)                    the vertices where four boundary faces meet
#   splitQuadruplePoints(c, seg, ebsd, crit)   each split into two triple points, the grains merged across
#
# On a square lattice four cells meet at every interior vertex, so a checkerboard of two
# labels puts four boundary segments on one point; in the filled cells, where the Voronoi
# cells of one grain meet at the same point, further faces of that grain may end there.
# Which diagonal is connected is not in the data, so removing a quadruple point means
# deciding it: the vertex is split into two triple points, one diagonal pair of cells
# becomes adjacent, and the criterion judges that pair like any other. A pairing is
# possible where each of the two wedges it joins holds one measured cell. The pairing is
# chosen by the measurements alone: first the diagonal the criterion connects whose pixels
# lie in two grains, so that joining it merges them, then the criterion, then the smaller
# misorientation angle, then where the two pixels lie. A point whose chosen diagonal fails
# the criterion stays a quadruple point, so every split joins its pair and no added face is
# a boundary. The two vertices move an eighth of a pixel apart, each into the wedge between
# its two boundary edges, so the joined cells meet along a neck of a quarter pixel and the
# added face has a direction.
from __future__ import annotations

import numpy as np

from .criteria import connectivity, _misorientationAngle
from .decomposition import cellComplex
from .segment import mergeAcross


# the one site both of two faces bound, the cell in the wedge between them, or -1, for faces of
# shape (..., 2) of sites
def _commonSite(a, b):
  match = a[..., :, None] == b[..., None, :]
  return np.where(match.sum(axis=(-2, -1)) == 1, np.where(match[..., 0, :].any(axis=-1), a[..., 0], a[..., 1]), -1)


def quadruplePoints(c, seg):
  """the vertices where exactly four grain boundary faces meet, whatever other faces end there,
  with those four faces in counterclockwise order"""
  nV = len(c.V)
  outer = np.flatnonzero(seg.isOuter).astype(np.int32)
  degOuter = np.bincount(c.F[outer].reshape(-1), minlength=nV)
  verts = np.flatnonzero(degOuter == 4).astype(np.int32)
  return verts, _incidentFaces(c, verts, outer)[0].reshape(-1, 4)


# the faces at every vertex of verts, among the candidates or all of them: the face, the direction
# of its other end from the vertex on the lattice, before the straightening moved them, which vertex
# it belongs to and at which end; a vertex's faces are consecutive, ordered counterclockwise
def _incidentFaces(c, verts, candidates=None):
  F = c.F if candidates is None else c.F[candidates]
  slot = np.full(len(c.V), -1, dtype=np.int32)
  slot[verts] = np.arange(len(verts), dtype=np.int32)
  r = slot[F]
  f, side = np.nonzero(r >= 0)
  owner = r[f, side]
  d = c.V0[F[f, 1 - side]] - c.V0[verts[owner]]
  angle = np.arctan2(d[:, 1], d[:, 0])
  order = np.lexsort((angle, owner))
  faces = f if candidates is None else candidates[f]
  return faces[order], angle[order], owner[order], side[order]


# the other end of every face of the vertices verts, both of the same shape
def _farVertex(c, verts, faces):
  return np.where(c.F[faces, 0] == verts, c.F[faces, 1], c.F[faces, 0])


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


# the keys that rank the pairing joining the sites a and b, larger first: whether the criterion
# connects their pixels and they lie in two grains, the criterion, the smaller misorientation, the position
def _rank(ebsd, crit, grainId, sites, a, b):
  i, j = sites.pixel[a], sites.pixel[b]
  c = connectivity(crit, ebsd, i, j)
  return (c > 0) & (grainId[i] != grainId[j]), c, -_pairAngle(ebsd, i, j), -_posKey(sites.pos[a], sites.pos[b])


# whether B beats A on the first key the two differ in
def _beats(KB, KA):
  win, tied = np.zeros(len(KA[0]), dtype=bool), np.ones(len(KA[0]), dtype=bool)
  for a, b in zip(KA, KB):
    win |= tied & (b > a)
    tied &= a == b
  return win


# the unit bisector of two unit directions, zero where they are opposite
def _bisector(u, v):
  b = u + v
  L = np.linalg.norm(b, axis=1)
  return np.where(L[:, None] > 0, b / np.where(L > 0, L, 1)[:, None], 0.0)


def splitQuadruplePoints(c, seg, ebsd, crit, backend=None):
  """the cell complex with every quadruple point of a square grid split into two triple points a
  quarter pixel apart and the grains merged across the diagonal pair joined: the pair that merges
  two grains the criterion connects, else the one it connects more, then of smaller
  misorientation, then by position; a point whose chosen pair the criterion separates stays"""
  verts, faces = quadruplePoints(c, seg)
  if len(verts) == 0:
    return c, seg
  sites = c.sites
  # the one cell in every wedge between two neighbouring boundary faces, -1 where there are more
  cells = c.cells[faces]
  wedge = _commonSite(cells, np.roll(cells, -1, axis=1)).astype(np.int32)
  single = (wedge >= 0) & (sites.pixel[np.maximum(wedge, 0)] >= 0)
  # the two candidate pairings: A joins the cells between faces 1,2 and 3,0 and detaches
  # faces 0,1 from the vertex; B joins those between 0,1 and 2,3
  okA = single[:, 1] & single[:, 3] & (wedge[:, 1] != wedge[:, 3])
  okB = single[:, 0] & single[:, 2] & (wedge[:, 0] != wedge[:, 2])
  w = np.maximum(wedge, 0)
  KA = _rank(ebsd, crit, seg.grainId, sites, w[:, 1], w[:, 3])
  KB = _rank(ebsd, crit, seg.grainId, sites, w[:, 0], w[:, 2])
  useB = (okB & ~okA) | (okA & okB & _beats(KB, KA))
  keep = np.flatnonzero((okA | okB) & (np.where(useB, KB[1], KA[1]) > 0))
  if len(keep) == 0:
    return c, seg
  verts, faces, wedge, useB = verts[keep], faces[keep], wedge[keep], useB[keep]
  n = len(verts)
  rows = np.arange(n)
  # the first of the two boundary faces that move; the cells joined lie between m+1, m+2 and m+3, m
  m = np.where(useB, 3, 0)
  at = lambda k: (m + k) % 4
  # the vertex and its copy an eighth of a pixel apart, each along the bisector of its two faces
  T = c.V.dtype
  p = c.V[verts]
  d = c.V[_farVertex(c, verts[:, None], faces)] - p[:, None, :]
  u = d / np.linalg.norm(d, axis=2)[..., None]
  p = p.astype(float)
  step = c.dxy / 8
  V = np.concatenate([c.V, (p + step * _bisector(u[rows, at(0)], u[rows, at(1)])).astype(T)])
  V[verts] = (p + step * _bisector(u[rows, at(2)], u[rows, at(3)])).astype(T)
  new = (len(c.V) + rows).astype(np.int32)
  # every face from the first moving face counterclockwise to the second ends at the copy instead
  F = c.F.copy()
  incident, angle, owner, side = _incidentFaces(c, verts)
  d0 = c.V0[_farVertex(c, verts[:, None], faces)] - c.V0[verts][:, None, :]
  a0, a1 = (np.arctan2(d0[rows, at(k), 1], d0[rows, at(k), 0]) for k in (0, 1))
  move = np.mod(angle - a0[owner], 2 * np.pi) <= np.mod(a1[owner] - a0[owner], 2 * np.pi)
  F[incident[move], side[move]] = new[owner[move]]
  newF = np.stack([verts, new], axis=1).astype(np.int32)
  newCells = np.stack([wedge[rows, at(1)], wedge[rows, at(3)]], axis=1)
  c2 = cellComplex(V, np.concatenate([F, newF]), np.concatenate([c.cells, newCells]), sites, c.A, c.plane, c.shape,
                   np.concatenate([c.V0, c.V0[verts]]))
  i, j = sites.pixel[newCells[:, 0]], sites.pixel[newCells[:, 1]]
  return c2, mergeAcross(seg, c2, i, j, connectivity(crit, ebsd, i, j), backend)


__all__ = []
