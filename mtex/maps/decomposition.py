# the spatial decomposition of a map: one cell per measurement (Julia decomposition.jl)
#
#   decompose(ebsd, alpha=3.1)    the closing, the sites and their cells, as a cellComplex
#   c.V, c.F, c.cells             the vertices, the vertex pair and the site pair of every face
#   c.siteAt(q, ij)               the site whose cell covers a point
#
# calcGrains needs a partition of the plane into one cell per measurement, so that two
# cells sharing a face is the adjacency the criterion runs on and the faces are the
# boundary geometry. On a lattice that partition is local: the alpha closing leaves no
# site-less region wider than 2 alpha cells, so every cell is the box around its site
# clipped against the sites within a bounded reach. The sites are the lattice cells the
# closing keeps: a measurement, indexed or not; a hole, an enclosed region the closing did
# not fill, which becomes a notIndexed grain; a filled region has no site and the
# neighbours grow in; and the band and dummy rings outside the map bound its outer cells.
from __future__ import annotations

import warnings

import numpy as np

from .kernels import localVoronoiNumba, weldNumba
from .lattice import STENCIL, cellCorners
from .polygon import clipHalfPlane, seedBox, voronoiCell
from ..geometry.display import aligned
from ..geometry.hull import convexHull

NO_PIXEL = -1


# the entries of values at the indices, fill where an index is negative, NO_PIXEL say
def _gather(values, ind, fill):
  return np.where(ind >= 0, values[np.maximum(ind, 0)], fill)


class siteSet:
  """the sites of the decomposition: their positions, lattice indices, measurements, which are
  holes, the lookup from lattice cell to site and its offset"""

  def __init__(self, pos, ij, pixel, isHole, lookup, offset):
    self.pos, self.ij, self.pixel, self.isHole, self.lookup, self.offset = pos, ij, pixel, isHole, lookup, offset

  def __len__(self):
    return len(self.pixel)


class cellComplex:
  """the spatial decomposition of a map, one cell per site: the vertices V, the vertex pair F and
  the site pair of every face, the sites and the lattice

  Input
    V - planar vertex coordinates
    F - pairs of vertex indices describing faces between cells
    cells - pairs of adjacent site indices for every face
    sites - measurement sites represented by the cells
    A - lattice basis
    plane - mapPlane of the geometry
    shape - cell or map shape metadata"""

  def __init__(self, V, F, cells, sites, A, plane, shape):
    self.V, self.F, self.cells, self.sites, self.A, self.plane, self.shape = V, F, cells, sites, A, plane, shape

  @property
  def site2pixel(self):
    """the measurement of every site, NO_PIXEL where it has none, and the index of every site"""
    return self.sites.pixel

  @property
  def siteIJ(self):
    """the lattice indices of the sites"""
    return self.sites.ij

  @property
  def nSites(self):
    """the number of sites, the measurements, the holes and the outer rings"""
    return len(self.sites)

  def __len__(self):
    return len(self.F)

  def siteAt(self, q, ij, reach=5):
    """the site whose cell covers a point: its nearest site, since a cell is the set of points
    nearer to its site than to any other, searched over the lattice cells around the point

      c = decompose(ebsd)                        # from mtex.maps.decomposition
      s = c.siteAt(q, ij)                        # the site of the point q near the lattice cell ij
      s = c.siteAt(q, ij, reach=2)               # searched over fewer lattice cells

    The candidates are the sites of the lattice cells within reach of ij in either index; the
    nearest of them to q is returned, NO_PIXEL when there is none. calcGrains asks this for the
    corners of a not indexed pixel to find the grain that absorbs it.

    Input
      c - cellComplex
      q - the point (x, y) in the plane of the map
      ij - the lattice indices of the cell the point lies in or near
    Output
      s - the index of the site, a row of c.siteIJ, or NO_PIXEL
    Keywords
      reach - how many lattice cells either way are searched
    See also
      calcGrains"""
    s = self.sites
    a0, b0 = int(ij[0]) - s.offset[0], int(ij[1]) - s.offset[1]
    n1, n2 = s.lookup.shape
    block = s.lookup[max(0, a0 - reach):min(n1, a0 + reach + 1), max(0, b0 - reach):min(n2, b0 + reach + 1)].reshape(-1)
    cand = block[block >= 0]
    if len(cand) == 0:
      return NO_PIXEL
    d = np.sum((s.pos[cand] - np.asarray(q)) ** 2, axis=1)
    return int(cand[np.argmin(d)])

  def unitCell2d(self):
    """the corners of one lattice cell in the plane's coordinates"""
    return cellCorners(self.A, self.shape)

  def __repr__(self):
    return aligned(f"cellComplex ({self.shape})\n  sites: {self.nSites}\n  faces: {len(self)}\n  vertices: {len(self.V)}")


# ------------------------------------------------------------------ the alpha closing

def diskOffsets(A, r, stencil):
  """the structuring element of the closing, shape (m, 2): the lattice offsets within a physical
  radius r, and beside the extreme of the disk along every stencil direction the offsets one
  step to the side, so that a one cell recess at a straight edge is covered from its
  neighbours wherever r falls between the lattice points"""
  wi = int(np.ceil(r / np.linalg.norm(A[:, 0]))) + 1
  wj = int(np.ceil(r / np.linalg.norm(A[:, 1]))) + 1
  di, dj = np.meshgrid(np.arange(-wi, wi + 1), np.arange(-wj, wj + 1), indexing="ij")
  offs = np.stack([di.reshape(-1), dj.reshape(-1)], axis=1)
  M = np.floor(r / np.linalg.norm(stencil @ A.T, axis=1)).astype(int)
  side = [M[k] * s - t for k, s in enumerate(stencil) if M[k] >= 1 for t in stencil if not np.all((t == s) | (t == -s))]
  return np.unique(np.concatenate([offs[np.linalg.norm(offs @ A.T, axis=1) <= r], np.array(side, dtype=int).reshape(-1, 2)]), axis=0)


def dilate(m, offs):
  """binary dilation of a raster mask by a set of index offsets, with zero padding"""
  m = np.asarray(m, dtype=bool)
  offs = np.asarray(offs, dtype=int).reshape(-1, 2)
  out = np.zeros_like(m)
  n1, n2 = m.shape
  for di, dj in offs:
    r0, r1 = max(0, di), min(n1, n1 + di)
    c0, c1 = max(0, dj), min(n2, n2 + dj)
    if r0 < r1 and c0 < c1:
      out[r0:r1, c0:c1] |= m[r0 - di:r1 - di, c0 - dj:c1 - dj]
  return out


def alphaClose(present, A, r, stencil, outside=None):
  """the closing of a set by a disk of radius r: the cells no empty disk of that radius
  covers, as the De Morgan pair of dilations; the cells outside count as present while it
  erodes and are clipped away after"""
  offs = diskOffsets(np.asarray(A, dtype=float), r, np.asarray(stencil))
  present = np.asarray(present, dtype=bool)
  outside = np.zeros_like(present) if outside is None else outside
  return ~dilate(~(dilate(present, offs) | outside), offs) & ~outside | present


def hullCells(cells, dims):
  """the cells of a raster inside the convex hull of the cells given, all of them when none is
  given: each row meets the hull in one interval, bounded by the edges of the hull crossing it"""
  if len(cells) == 0:
    return np.ones(dims, dtype=bool)
  h = cells[convexHull(cells)].astype(float)
  R, C = [h[:, 0]], [h[:, 1]]
  for (r0, c0), (r1, c1) in zip(h, np.roll(h, -1, axis=0)):
    if r0 != r1:
      R.append(np.arange(min(r0, r1), max(r0, r1) + 1))
      C.append(c0 + (R[-1] - r0) * (c1 - c0) / (r1 - r0))
  R, C = np.concatenate(R).astype(int), np.concatenate(C)
  lo, hi = np.full(dims[0], np.inf), np.full(dims[0], -np.inf)
  np.minimum.at(lo, R, C)
  np.maximum.at(hi, R, C)
  j = np.arange(dims[1])
  return (j >= lo[:, None] - 1e-9) & (j <= hi[:, None] + 1e-9)


def steps(seed, mask, stencil, n):
  """the number of stencil steps from the seed to every cell of the mask, through the mask,
  infinite beyond n"""
  d = np.full(seed.shape, np.inf)
  reached = front = seed
  for k in range(1, n + 1):
    front = dilate(front, stencil) & mask & ~reached
    d[front] = k
    reached = reached | front
  return d


def recesses(material, A, r, stencil):
  """the cells with material within the reach of the closing on both sides along a stencil
  direction"""
  def reach(s):
    return dilate(material, -np.outer(np.arange(1, np.floor(r / np.linalg.norm(A @ s)) + 1), s))
  return np.any([reach(s) & reach(-s) for s in stencil], axis=0)


def floodBorder(bg, stencil):
  """the background cells connected to the border of the raster"""
  from scipy import ndimage
  bg = np.asarray(bg, dtype=bool)
  structure = np.zeros((3, 3), dtype=bool)
  structure[1, 1] = True
  for di, dj in stencil:
    structure[1 + di, 1 + dj] = True
  labels, n = ndimage.label(bg, structure=structure)
  border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
  ext = np.unique(border[border > 0])
  return np.isin(labels, ext)


# ------------------------------------------------------------------ the sites

def buildSites(l, P, present, alpha=3.1, measured=None):
  """classify every lattice cell and place the sites; a cell of the padding of a grid is not
  measured and lies outside the map as a cell beyond the lattice does"""
  ij = l.ij
  n = len(ij)
  r = alpha * l.dxy
  st = STENCIL[l.shape]
  pad = int(np.ceil(2 * alpha)) + 4
  lo = (int(ij[:, 0].min()) - pad, int(ij[:, 1].min()) - pad)
  dims = (int(ij[:, 0].max()) - lo[0] + pad + 1, int(ij[:, 1].max()) - lo[1] + pad + 1)
  a, b = ij[:, 0] - lo[0], ij[:, 1] - lo[1]
  m = np.ones(n, dtype=bool) if measured is None else measured
  hasPixel = np.zeros(dims, dtype=bool)
  isPresent = np.zeros(dims, dtype=bool)
  pixelOf = np.full(dims, NO_PIXEL, dtype=np.int32)
  hasPixel[a[m], b[m]] = True
  pixelOf[a[m], b[m]] = np.arange(n, dtype=np.int32)[m]
  isPresent[a[present], b[present]] = True
  # the edge of the map is not the edge of a hole: beyond the convex hull of the indexed cells
  # the material counts as present while the closing erodes, so a gap along the edge closes
  # as one further in would
  closed = alphaClose(isPresent, l.A, r, st, ~hullCells(np.stack([a[present], b[present]], axis=1), dims))
  # a cell with material on more than half of its stencil is a one cell recess, which the
  # alpha shape fills; filling the mouth of a slit exposes the next cell
  while alpha >= 1:
    add = ~closed & (sum(dilate(closed, [s]).astype(int) for s in st) > len(st) // 2)
    if not add.any():
      break
    closed |= add
  ext = floodBorder(~closed, st)
  # a hole joined to the outside by a filled path of at most 2 ceil(alpha) steps is part of the
  # outside, and so is the path
  k = 2 * int(np.ceil(alpha))
  path = closed & ~isPresent
  path &= steps(ext, path, st, k) + steps(~closed & ~ext, path, st, k) <= k
  if path.any():
    closed &= ~path
    ext = floodBorder(~closed, st)
  material = ~ext
  filled = closed & ~isPresent                 # the neighbours grow into these
  # the band, one ring outside, closes the map's cells; a band cell touching only filled cells,
  # or in a recess too wide for the closing, makes way for one further out, so that the outer
  # boundary runs straight across the filled cells
  band = dilate(material, st) & ext
  vacated = band & (~dilate(material & ~filled, st) | recesses(material, l.A, r, st))
  band = (band | dilate(vacated, st) & ext) & ~vacated
  dummy = dilate(material | band, st) & ext & ~band & ~vacated
  isSite = (material & ~filled) | band | dummy
  flat = np.flatnonzero(isSite.reshape(-1))
  ns = len(flat)
  lookup = np.full(dims, NO_PIXEL, dtype=np.int32)
  lookup.reshape(-1)[flat] = np.arange(ns, dtype=np.int32)
  sa, sb = np.unravel_index(flat, dims)
  sij = np.stack([sa + lo[0], sb + lo[1]], axis=1).astype(np.int32)
  # a cell of the outer rings carries no measurement even where one was made there: the
  # rings only bound the map's cells, and a ring cell that took its pixel would segment
  # into a grain of its own
  onMap = material.reshape(-1)[flat]
  pixel = np.where(onMap, pixelOf.reshape(-1)[flat], NO_PIXEL).astype(np.int32)
  # a cell with a measurement sits where it was measured, one without it one lattice step
  # per index from the nearest measured cell, so that the rings and the holes follow a
  # smooth distortion of the map as the measurements do and not the affine fit of the whole
  from scipy import ndimage
  near = ndimage.distance_transform_edt(~hasPixel, return_distances=False, return_indices=True)
  nearest = pixelOf.reshape(-1)[np.ravel_multi_index((near[0].reshape(-1)[flat], near[1].reshape(-1)[flat]), dims)]
  step = (sij - ij.take(nearest, axis=0)).astype(P.dtype)
  pos = (P.take(nearest, axis=0) + step @ l.A.T).astype(P.dtype)
  has = np.flatnonzero(pixel >= 0)
  pos[has] = P.take(pixel.take(has), axis=0)
  isHole = (pixel < 0) & onMap
  return siteSet(pos, sij, pixel, isHole, lookup, lo)


# ------------------------------------------------------------------ the local Voronoi

def candidateOffsets(A, kmax):
  """the lattice offsets within kmax steps, sorted by their ideal physical distance"""
  di, dj = np.meshgrid(np.arange(-kmax, kmax + 1), np.arange(-kmax, kmax + 1), indexing="ij")
  offs = np.stack([di.reshape(-1), dj.reshape(-1)], axis=1)
  offs = offs[np.any(offs != 0, axis=1)]
  d = np.linalg.norm(offs @ A.T, axis=1)
  p = np.argsort(d, kind="stable")
  return offs[p].astype(np.int64), d[p]


def candidateLevels(A, alpha, dtype):
  """the three candidate lists of increasing reach: the closing bounds the cell of a site
  with material all round it, and that is what the first covers; a site on the rim of a
  bay wider than the closing has no such bound, so the search grows for it"""
  dxy = (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1])) / 2
  kmax = int(np.ceil(2 * alpha)) + 2
  levels = []
  for k in (kmax, 4 * kmax, 16 * kmax):
    offs, d = candidateOffsets(A, k)
    levels.append((offs, d.astype(dtype), dtype.type(1.5 * k * dxy)))
  return levels


# clip the cell of one site against one candidate list; True when an edge of the seed box
# survived, so the cell is not bounded by sites yet
def _cellOf(sites, s, offs, dIdeal, W, distortTol):
  p = sites.pos[s]
  a0, b0 = sites.ij[s, 0] - sites.offset[0], sites.ij[s, 1] - sites.offset[1]
  n1, n2 = sites.lookup.shape
  vs, gs = seedBox(p, W)
  rmax = W * np.sqrt(2)
  stillOpen = True
  for k in range(len(offs)):
    # a site further than twice the cell radius cannot cut it, but only once the cell is
    # closed: while an edge of the seed box survives the radius is the box's
    if not stillOpen and dIdeal[k] * (1 - distortTol) > 2 * rmax:
      break
    a, b = a0 + offs[k, 0], b0 + offs[k, 1]
    if not (0 <= a < n1 and 0 <= b < n2):
      continue
    s2 = sites.lookup[a, b]
    if s2 < 0:
      continue
    vs, gs = clipHalfPlane(vs, gs, p, sites.pos[s2], s2 + 1)
    rmax = np.sqrt(np.max(np.sum((np.array(vs) - p) ** 2, axis=1)))
    stillOpen = 0 in gs
  return vs, gs, stillOpen


def localVoronoiNumpy(sites, levels, distortTol=0.05):
  """the faces of every cell as (site, neighbour, v1, v2) with site < neighbour, the Python
  form of the compiled loop and its oracle; returns the faces and how many cells of the
  map stayed open"""
  faces = []
  unbounded = 0
  for s in range(len(sites)):
    isMap = sites.pixel[s] >= 0 or sites.isHole[s]
    vs, gs, stillOpen = [], [], True
    for lv, (offs, dIdeal, W) in enumerate(levels):
      if not (stillOpen and (lv == 0 or isMap)):
        break
      vs, gs, stillOpen = _cellOf(sites, s, offs, dIdeal, W, distortTol)
    m = len(vs)
    for i in range(m):
      g = gs[i] - 1
      if g < 0:
        unbounded += isMap
        continue
      if g > s:
        faces.append((s, g, vs[i][0], vs[i][1], vs[(i + 1) % m][0], vs[(i + 1) % m][1]))
  if not faces:
    return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 4), dtype=sites.pos.dtype), unbounded
  f = np.array(faces, dtype=np.float64)
  return f[:, :2].astype(np.int32), f[:, 2:].astype(sites.pos.dtype), unbounded


def localVoronoi(sites, A, alpha=3.1, distortTol=0.05, backend=None):
  """the faces of the Voronoi cell of every site, clipped against the lattice neighbours within
  the reach alpha leaves, as (site, neighbour) pairs and their vertices, with how many cells of
  the map stayed open; compiled with a NumPy fallback"""
  levels = candidateLevels(np.asarray(A, dtype=sites.pos.dtype), alpha, sites.pos.dtype)
  out = localVoronoiNumba(sites, levels, distortTol, backend=backend)
  return out if out is not None else localVoronoiNumpy(sites, levels, distortTol)


# ------------------------------------------------------------------ the complex

def meshVertices(cells, verts, tol, backend=None):
  """the vertices of the face list welded: a Voronoi vertex is computed once per cell that
  meets there, three or four on a lattice, and the copies are welded. The copies of one
  vertex differ by rounding, so a fine grid welds them, and the few that lie within
  rounding of a boundary of that grid are welded again on the half-shifted grid, which is
  where a single grid misses. The tolerance grid then merges what lies within a hundredth
  of the spacing, and a face whose two vertices welded together is dropped"""
  n = len(cells)
  xy = verts.reshape(-1, 2)
  # column by column, which NumPy reduces ten times faster than along the first axis
  lo = np.array([xy[:, 0].min(), xy[:, 1].min()], dtype=xy.dtype) if n else np.zeros(2)
  # the groups through one hashed pass over the points, the sorts below its fallback and oracle
  if n and xy.dtype == np.float64:
    out = weldNumba(xy, lo, tol * 1e-7, 50 * tol, backend=backend)
    if out is not None:
      group, first = out
      return _weldedMesh(cells, xy, tol, group, first)
  u = (xy - lo) / (tol * 1e-7)
  _, _, group = np.unique(_gridKey(u), return_index=True, return_inverse=True)
  group = group.reshape(-1)
  nG = int(group.max(initial=-1)) + 1
  frac = u - np.floor(u)
  cand = np.flatnonzero(np.any((frac < 1e-3) | (frac > 1 - 1e-3), axis=1))
  if len(cand):
    from .segment import components
    _, f2, inv2 = np.unique(_gridKey(u[cand] + 0.5), return_index=True, return_inverse=True)
    inv2 = inv2.reshape(-1)
    gc = group[cand]
    group = np.unique(components(nG, gc[f2[inv2]], gc), return_inverse=True)[1].reshape(-1)[group]
    nG = int(group.max(initial=-1)) + 1
  # the first point of every group, without a sort
  first = np.empty(nG, dtype=np.int64)
  first[group[::-1]] = np.arange(len(group))[::-1]
  return _weldedMesh(cells, xy, tol, group, first)


# the tolerance grid over the first point of every group, and the faces on the merged vertices;
# the groups taken in the order of their first points, so that where two share a cell of the
# tolerance grid the vertex is the earlier one whatever the groups were numbered. The grid is
# rounded from the origin of the map, whose measurements lie at multiples of the spacing from
# it, so its cells are centred on the vertices of a square lattice: a grid floored from the
# lowest vertex puts the vertices on cell edges wherever that vertex falls on the lattice, and
# two vertices a rounding apart - the ends of the vanishing face of a near four way tie - part
def _weldedMesh(cells, xy, tol, group, first):
  n = len(cells)
  if np.any(np.diff(first) < 0):
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    first, group = first[order], rank[group]
  reps = xy[first]
  _, firstV, inverse = np.unique(_gridKey(reps / tol + 0.5), return_index=True, return_inverse=True)
  V = reps[firstV]
  F = inverse.reshape(-1).astype(np.int32)[group].reshape(n, 2)
  # the rows kept by index, which NumPy copies several times faster than by a mask
  keep = np.flatnonzero(F[:, 0] != F[:, 1])
  return V, F.take(keep, axis=0), cells.take(keep, axis=0)


# one integer per grid cell of points given in grid units
def _gridKey(u):
  key = np.floor(u).astype(np.int64)
  return key[:, 0] * (key[:, 1].max(initial=0) - key[:, 1].min(initial=0) + 1) + key[:, 1]


def decompose(ebsd, alpha=3.1, tol=None, backend=None):
  """the spatial decomposition of a map into one cell around every pixel, the cell complex
  calcGrains builds the grains from; MATLAB's spatialDecomposition

    cells = decompose(ebsd)              # the cells of the map
    cells = decompose(ebsd, alpha=5)     # wider holes filled

  The cell of a pixel is its Voronoi cell among the lattice neighbours within reach, so a
  cell of a square or hexagonal grid is its unit cell. The map must lie on a lattice; a map
  without one is refused.

  alpha decides what happens to regions of not indexed pixels. A region narrower than
  2 alpha pixel spacings is filled, the cells of its indexed neighbours growing into it. A
  wider one keeps its pixels as holes, which calcGrains makes a notIndexed grain, unless a filled
  path of at most 2 alpha steps joins it to the outside, of which it is then a part.

  The vertices that the cells of neighbouring pixels compute each for themselves are welded
  into one when they lie within tol.

  Input
    ebsd - EBSD on a square or hexagonal lattice
  Output
    cells - cellComplex: the vertices V, the vertex pair F and the site pair of every face,
        the sites with the pixel each stands for (site2pixel), and the lattice
  Keywords
    alpha - the closing radius in pixel spacings: a not indexed region narrower than 2 alpha
        is filled and its neighbours grow in, a wider one is kept
    tol - the distance within which two vertices are welded into one, by default a
        hundredth of the pixel spacing
    backend - 'numba' or 'numpy', the loops of the clipping and the welding; by default the
        preference 'kernels'
  See also
    calcGrains, EBSD.gridify, GrainReconstruction"""
  l = ebsd.lattice
  if l is None:
    raise ValueError("the map is not on a lattice, so it cannot be decomposed")
  P = l.plane.to2d(ebsd.pos.data.reshape(-1, 3))
  present = ebsd.isIndexed.reshape(-1)
  sites = buildSites(l, P, present, alpha, None if ebsd._pad is None else ~ebsd._pad.reshape(-1))
  cells, verts, unbounded = localVoronoi(sites, l.A, alpha, backend=backend)
  if unbounded:
    warnings.warn(f"the decomposition left {unbounded} cells of the map open; its sites reach beyond the closing")
  V, F, cells = meshVertices(cells, verts, l.dxy / 100 if tol is None else tol, backend=backend)
  return cellComplex(V, F, cells, sites, l.A, l.plane, l.shape)


__all__ = ["decompose", "cellComplex"]
