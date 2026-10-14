# denoising and repair: the filters that replace an orientation by one computed from its
# neighbours, and the repairs that give a pixel one at all (MATLAB EBSDSmoothing/,
# @EBSD/smooth, fill, fillByGrainId, reduce, interp; Julia denoising.jl)
#
#   ebsd.smooth(), ebsd.smooth(F), ebsd.smooth(F, fill=True, extrapolate=True, props=['ci'])
#   ebsd.fill(), ebsd.fill(F, extrapolate=True), ebsd.fillByGrainId(grains), ebsd.reduce(n)
#   ebsd.interp(pos), ebsd.interp(x, y)
#   meanFilter(numNeighbours=1), medianFilter(numNeighbours=1), KuwaharaFilter(numNeighbours=1)
#
# Every filter works on the same object: the field of tangent vectors of the component a
# pixel belongs to, its grain where the map knows them and its phase otherwise, over the
# graph of neighbouring pixels. That makes the window filters ordinary image filters and
# the variational ones an energy over a graph, and one implementation serves a square, a
# hexagonal, a rotated and a sheared grid. Filling is the same model with the data term
# switched off where there is no data: a pixel without a measurement is a free unknown,
# and what fills it is the continuation of the field around it. So smooth and fill are one
# solver under two boundary conditions, the measurements moving in the first and pinned in
# the second, and interp is the filled field read at positions of one's own choosing.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import backend, closestImage, qconj, qmul
from ..geometry.quaternion import rotation
from ..geometry.tangent import rotationOf, rotationVector
from ..geometry.vector3d import vector3d
from ..maps.ebsd import EBSD
from ..maps.grains import grain2d
from ..maps.intragranular import IDENTITY, latticeLookup, mapLattice, mapPairs, neighbourAt, neighbourBall
from ..maps.lattice import lattice
from .variational import EBSDFilter, l1TVFilter, solveField, variationalFilter
from ..geometry.install import install



# ------------------------------------------------------------------ the window filters

class windowFilter(EBSDFilter):
  """the base of the filters over a window of neighbours, numNeighbours the rings of the lattice
  the window reaches: meanFilter, medianFilter, KuwaharaFilter"""

  def __init__(self, numNeighbours=1):
    self.numNeighbours = int(numNeighbours)

  def _items(self):
    return [("neighbours", self.numNeighbours)]


class meanFilter(windowFilter):
  """the mean of the neighbourhood, the fastest filter and the one that blurs a boundary most"""
  pass


class medianFilter(windowFilter):
  """the componentwise median of the neighbourhood, which takes a wild pixel out without
  pulling the boundary after it"""
  pass


class KuwaharaFilter(windowFilter):
  """the mean of the most homogeneous of the four quadrants of the window, which smooths within
  a grain and leaves a step where one is"""
  pass


# ------------------------------------------------------------------ the field

def smoothingGraph(ebsd, fill=False, extrapolate=False):
  """the components a filter works within, the neighbour pairs inside them, which pixels carry
  a measurement and, for the ones that do not, the measured pixel nearest to them. The
  components are the grains where the map knows them and the phases where it does not. A
  pixel with no measurement joins the component that reaches it first over the lattice,
  which is all the flood fill decides, and one the outside of the map reaches without
  crossing a measurement joins none: it was never inside the scan"""
  l = mapLattice(ebsd)
  lk, m = latticeLookup(l), mapPairs(ebsd)
  n = len(l.ij)
  isData = m.indexed.copy()
  comp = (m.phase.astype(np.int64) if m.grain is None else m.grain.astype(np.int64)).copy()
  # a pixel calcGrains absorbed into a grain (its phase, a NaN rotation) is filled within that grain
  absorbed = ~isData & (m.grain is not None) & (fill is not False) & (fill is not None)
  if m.grain is not None:
    absorbed &= (comp > 0) & ebsd.isIndexed.reshape(-1)
  comp[(~isData & ~absorbed) | (comp < 0)] = 0
  source = np.where(comp > 0, np.arange(n), -1)
  offs = l.stencil.astype(np.int64)
  # fill=grains: an empty pixel is filled only inside the grain whose polygon holds it, a
  # cell of padding too, which a grain's polygon puts inside the scan
  host = None
  if isinstance(fill, grain2d):
    empty = np.flatnonzero(~isData)
    host = np.zeros(n, dtype=np.int64)
    if len(empty):
      hit = fill.checkInside(vector3d(ebsd.pos.data.reshape(-1, 3)[empty], frame=ebsd.frame))
      has = hit.any(axis=1)
      host[empty[has]] = fill.id[np.argmax(hit[has], axis=1)]
  if fill is not False and fill is not None:
    # the padding of a grid lies outside the scan, whatever extrapolate asks, unless a grain holds it
    if host is not None:
      outside = host == 0
    else:
      outside = np.zeros(n, dtype=bool) if extrapolate else outsideMask(l, lk, isData)
      outside |= ebsd.isPadding.reshape(-1)
    front = np.flatnonzero(comp > 0)
    while len(front):
      nxt = []
      for o in offs:
        nb = neighbourAt(lk, l.ij[front], o)[0]
        ok = (nb >= 0)
        nbb = np.maximum(nb, 0)
        ok &= (comp[nbb] == 0) & ~isData[nbb] & ~outside[nbb]
        if host is not None:
          ok &= host[nbb] == comp[front]
        # a site reached from two pixels of the front takes the first of them
        nb, src = nb[ok], front[ok]
        _, first = np.unique(nb, return_index=True)
        nb, src = nb[first], src[first]
        fresh = comp[nb] == 0
        nb, src = nb[fresh], src[fresh]
        comp[nb] = comp[src]
        source[nb] = source[src]
        nxt.append(nb)
      front = np.concatenate(nxt) if nxt else np.zeros(0, dtype=np.int64)
  a, b = [], []
  k = np.arange(n)
  for o in offs:
    nb = neighbourAt(lk, l.ij, o)[0]
    ok = (nb > k) & (comp > 0)
    ok[ok] &= comp[nb[ok]] == comp[ok]
    a.append(k[ok])
    b.append(nb[ok])
  edges = np.stack([np.concatenate(a), np.concatenate(b)], axis=1) if a else np.zeros((0, 2), dtype=np.int64)
  return comp, edges, isData, source


def outsideMask(l, lk, isData):
  """the pixels no measurement encloses: those the outside of the map reaches over the lattice
  without crossing a measured pixel"""
  ij = l.ij
  border = (ij[:, 0] == 0) | (ij[:, 1] == 0) | (ij[:, 0] == lk.shape[0] - 1) | (ij[:, 1] == lk.shape[1] - 1)
  out = ~isData & border
  offs = l.stencil.astype(np.int64)
  front = np.flatnonzero(out)
  while len(front):
    nxt = []
    for o in offs:
      nb = neighbourAt(lk, ij[front], o)[0]
      nb = nb[nb >= 0]
      nb = nb[~out[nb] & ~isData[nb]]
      out[nb] = True
      nxt.append(nb)
    front = np.unique(np.concatenate(nxt)) if nxt else np.zeros(0, dtype=np.int64)
  return out


def tangentField(ebsd, comp, isData):
  """every pixel as a tangent vector at the mean orientation of its own component, so that a
  difference within a component is a difference of vectors; the mean is taken over the
  measured pixels alone, and a pixel without a measurement starts at it and is solved for"""
  m = mapPairs(ebsd)
  n = len(comp)
  nc = int(comp.max(initial=0))
  ref = np.tile(IDENTITY[0], (nc + 1, 1))
  u = np.zeros((n, 3))
  data = (comp > 0) & isData
  if not data.any():
    return u, ref
  # the first measured member of every component is the seed the others are gathered at
  idx = np.flatnonzero(data)
  order = np.lexsort((idx, comp[idx]))
  cd = comp[idx][order]
  first = idx[order][np.concatenate([[True], cd[1:] != cd[:-1]])]
  seedOf = np.zeros(nc + 1, dtype=np.int64)
  seedOf[comp[first]] = first
  q = _closestImages(m, data, m.rot[seedOf[comp]])[0]
  q *= np.where(np.sum(q * m.rot[seedOf[comp]], axis=1) < 0, -1.0, 1.0)[:, None]
  acc = np.stack([np.bincount(comp[data], q[data, c], nc + 1) for c in range(4)], axis=1)
  nrm = np.linalg.norm(acc, axis=1)
  ok = nrm > 0
  ref[ok] = acc[ok] / nrm[ok, None]
  # every measurement as the left tangent vector at its component's mean
  u = _closestImages(m, data, ref[comp])[1]
  return u, ref


# the image of the measurement of every pixel closest to a reference of its own, and the
# tangent vector from the reference to it, for the pixels selected
def _closestImages(m, sel, refs):
  n = len(sel)
  if backend() == "numba":
    from ..maps.kernels import pixelKernel
    loop = pixelKernel("tangent")
    if loop is not None:
      img, u = np.empty((n, 4)), np.empty((n, 3))
      indexed, phase, _, _ = m.arrays()
      Gall, nG = m.groupTable()
      loop(m.rot, np.ascontiguousarray(refs), np.ascontiguousarray(indexed & sel), phase, Gall, nG, img, u)
      return img, u
  img, u = np.full((n, 4), np.nan), np.zeros((n, 3))
  idx = np.flatnonzero(sel)
  for p, G in m.G.items():
    at = idx[m.phase[idx] == p]
    if len(at):
      img[at] = closestImage(refs[at], m.rot[at], IDENTITY, G)[1]
  u[idx] = rotationVector(qmul(qconj(refs[idx]), img[idx]))
  return img, u


# the rotations ref exp(v) of the pixels selected
def _expAt(ref, v, sel):
  out = np.empty((len(v), 4))
  if backend() == "numba":
    from ..maps.kernels import pixelKernel
    loop = pixelKernel("exp")
    if loop is not None:
      loop(np.ascontiguousarray(ref), np.ascontiguousarray(v), out)
      return out[sel]
  return qmul(ref[sel], rotationOf(v[sel]))


# the window filters over the lattice, never across a component; a window filter has no
# variational form, so it has nothing to say about a pixel that carries no measurement and
# leaves it where it is. The compiled loop, or the NumPy gather of the windows without it
def _applyWindow(F, ebsd, comp, isData, u):
  l = mapLattice(ebsd)
  lk = latticeLookup(l)
  offs = np.concatenate([[[0, 0]], neighbourBall(l, F.numNeighbours)]).astype(np.int64)
  # the four quadrants of the window over the lattice index, so a hexagonal grid is divided
  # the way its own axes divide it; every quadrant holds the pixel itself
  quad = np.array([np.isin(np.sign(offs[:, 0]), (0, si)) & np.isin(np.sign(offs[:, 1]), (0, sj)) for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1))])
  mode = 0 if isinstance(F, meanFilter) else 1 if isinstance(F, medianFilter) else 2
  if backend() == "numba":
    from ..maps.kernels import pixelKernel
    loop = pixelKernel("window")
    if loop is not None:
      out = np.empty_like(u)
      loop(np.ascontiguousarray(u), np.ascontiguousarray(comp), np.ascontiguousarray(isData), lk, np.ascontiguousarray(l.ij, dtype=np.int64), offs, quad,
           mode, out)
      return out
  act = (comp > 0) & isData
  W = np.full((len(u), len(offs), 3), np.nan)
  for c, o in enumerate(offs):
    nb = neighbourAt(lk, l.ij, o)[0]
    ok = nb >= 0
    ok &= (comp[np.maximum(nb, 0)] == comp) & isData[np.maximum(nb, 0)]
    W[ok, c] = u[nb[ok]]
  W = W[act]
  out = u.copy()
  if mode == 0:
    val = np.nanmean(W, axis=1)
  elif mode == 1:
    val = np.nanmedian(W, axis=1)
  else:
    best, bestVar = np.zeros((len(W), 3)), np.full(len(W), np.inf)
    for q in quad:
      Wq = W[:, q]
      cnt = np.sum(~np.isnan(Wq[..., 0]), axis=1)
      mu = np.nanmean(Wq, axis=1)
      var = np.nansum(np.sum((Wq - mu[:, None, :]) ** 2, axis=2), axis=1) / cnt
      better = var < bestVar
      bestVar[better], best[better] = var[better], mu[better]
    val = best
  out[act] = val
  return out


def _applyFilter(F, ebsd, comp, edges, isData, u, pin):
  if isinstance(F, variationalFilter):
    return _applyVariational(F, ebsd, comp, edges, isData, u, pin)
  if isinstance(F, windowFilter):
    return _applyWindow(F, ebsd, comp, isData, u)
  raise TypeError(f"{type(F).__name__} is not an EBSD filter")


# the variational filters over the whole map at once: the edges never cross a component,
# so the one system is block diagonal in them, and a pixel of no component has no edge and
# no data term and stays where it is
def _applyVariational(F, ebsd, comp, edges, isData, f, pin):
  out = f.copy()
  if len(edges) == 0:
    return out
  x = solveField(F, mapLattice(ebsd), edges, f.astype(float), isData & (comp > 0), pin, comp)
  w = comp > 0
  out[w] = x[w]
  return out


# ------------------------------------------------------------------ smooth and fill

def smooth(ebsd, F=None, fill=False, extrapolate=False, props=()):
  """the map with its orientations denoised, every one replaced by one computed from its
  neighbours, within the grains where the map knows them and within the phases otherwise

    ebsd_s = smooth(ebsd)                                # the default filter, total variation
    ebsd_s = smooth(ebsd, halfQuadraticFilter())         # another filter
    ebsd_s = smooth(ebsd, F, fill=True)                  # the pixels without a measurement solved for as well
    ebsd_s = smooth(ebsd, F, fill=grains)                # only those inside a grain
    ebsd_s = smooth(ebsd, F, props=('bc',))              # a numeric property smoothed along

  The orientations are turned into a field of tangent vectors per connected component of
  the map (a grain, if calcGrains has written the grain ids, else a phase) and the filter
  acts on that field: a variational filter (splineFilter, halfQuadraticFilter, l1TVFilter,
  infimalConvolutionFilter) solves for the smoothed field, a window filter (meanFilter,
  medianFilter, KuwaharaFilter) replaces each pixel by its neighbourhood. Nothing is mixed
  across a grain boundary or a phase boundary.

  With fill the pixels that carry no measurement are solved for too (this is not the same as
  notIndexed); fill=grains fills a pixel only from the grain whose polygon holds it, so
  pixels between grains stay empty. A window filter solves for nothing, so with it the holes
  are closed by fill first. The map passed is not changed; MATLAB's second output, the
  filter, is the F passed.

  Input
    ebsd - EBSD, the map
    F - EBSDFilter, the filter; by default l1TVFilter()
  Output
    ebsd_s - EBSD, the smoothed map
  Keywords
    fill - True to fill the pixels without a measurement, or grain2d to fill only those
        inside a grain
    extrapolate - fill up to the outer boundary of the map, the pixels the outside reaches
        without crossing a measurement included
    props - the names of numeric properties of the map smoothed with the orientations, by a
        variational filter
  See also
    fill, interp, halfQuadraticFilter, splineFilter, meanFilter, EBSDDenoising, EBSDFilling"""
  F = l1TVFilter() if F is None else F
  window = isinstance(F, windowFilter)
  out = ebsd.fill(grains=fill if isinstance(fill, grain2d) else None, extrapolate=extrapolate) if fill and window else ebsd.copy()
  comp, edges, isData, source = smoothingGraph(out, fill=False if window else fill, extrapolate=extrapolate)
  u, ref = tangentField(out, comp, isData)
  v = _applyFilter(F, out, comp, edges, isData, u, False)
  _writeBack(out, comp, ref, v, isData, source, np.ones(len(comp), dtype=bool))
  for key in props:
    _solveProperty(out, key, F, comp, edges, isData, False)
  return out


def fill(ebsd, F=None, grains=None, extrapolate=False):
  """the map with every pixel that carries no measurement filled, the measured ones left as
  they are

    ebsd_f = fill(ebsd)                                  # the holes filled
    ebsd_f = fill(ebsd, grains)                          # filled within the grains only
    ebsd_f = fill(ebsd, halfQuadraticFilter(), extrapolate=True)  # another filter, up to the outer boundary

  A hole is filled with the smooth continuation of the orientations around it, by a
  variational filter, l1TVFilter by default; a window filter is refused. Nothing is filled
  across a grain boundary, where the map has grain ids, or a phase boundary. MATLAB fills by
  the nearest neighbour instead.

  The numeric properties are filled by the same solve, the labels (phase, grain id, ...)
  copied from the nearest measured pixel. A pixel the outside reaches without crossing a
  measurement stays empty, unless extrapolate=True. Only pixels on the lattice are filled: a
  gridded map, or the padding gridify adds.

  With grains, MATLAB's fill(ebsd, grains), a pixel is filled only from the grain whose
  polygon holds it, which needs the grain ids calcGrains writes into the map:

  - a pixel inside no grain becomes notIndexed
  - a pixel its grain's measurements do not reach over the lattice takes the grain's mean
  orientation
  - padding a polygon holds becomes a cell of the map

  A pixel counts as measured when it is indexed and has an orientation: the notIndexed pixels
  calcGrains absorbs into a grain, which carry its phase and a NaN orientation, are filled
  from within that grain. The map passed is not changed.

  Input
    ebsd - EBSD, the map
    F - EBSDFilter, a variational filter, or grain2d in place of grains
  Output
    ebsd_f - EBSD, the filled map
  Keywords
    grains - grain2d of the map: fill a pixel only from the grain whose polygon holds it
    extrapolate - fill up to the outer boundary of the map
  See also
    smooth, interp, gridify, calcGrains, EBSDFilling"""
  if isinstance(F, grain2d):
    F, grains = None, F
  F = l1TVFilter() if F is None else F
  if isinstance(F, windowFilter):
    raise TypeError(f"{type(F).__name__} has no variational form and cannot fill; use a spline or half quadratic filter")
  if grains is not None and "grainId" not in ebsd.prop:
    raise ValueError("filling within grains needs the grain ids of the map; reconstruct them with calcGrains(ebsd)")
  out = ebsd.copy()
  # a map whose every pixel but the padding is measured has nothing to fill, and with grains only
  # the padding their polygons hold, so nothing where there is none
  pad = out.isPadding.reshape(-1)
  measured = np.asarray(out.isIndexed).reshape(-1) & ~np.isnan(out.rotations.data.reshape(-1, 4)[:, 0])
  if np.all(measured | pad) and (grains is None or not pad.any()):
    return out
  comp, edges, isData, source = smoothingGraph(out, fill=True if grains is None else grains, extrapolate=extrapolate)
  if np.any((comp > 0) & ~isData):
    u, ref = tangentField(out, comp, isData)
    v = _applyFilter(F, out, comp, edges, isData, u, True)
    _writeBack(out, comp, ref, v, isData, source, ~isData)
    for key, val in out.prop.items():
      if np.issubdtype(np.asarray(val).dtype, np.floating):
        _solveProperty(out, key, F, comp, edges, isData, True)
  if grains is not None:
    _fillFromGrains(out, grains, comp == 0)
  return out


# the empty pixels the flood did not reach: the grain whose polygon holds one gives it its
# phase, id and mean orientation, and one in no grain is notIndexed
def _fillFromGrains(ebsd, grains, unreached):
  empty = np.flatnonzero(unreached & ~ebsd.isPadding.reshape(-1))
  if not len(empty):
    return
  hit = grains.checkInside(vector3d(ebsd.pos.data.reshape(-1, 3)[empty], frame=ebsd.frame))
  has = hit.any(axis=1)
  k = np.argmax(hit[has], axis=1)
  ph, gid = ebsd.phaseId.reshape(-1), ebsd.prop["grainId"].reshape(-1)
  ph[empty[has]] = grains.phaseId[k]
  gid[empty[has]] = grains.id[k]
  ebsd.rotations.data.reshape(-1, 4)[empty[has]] = grains.meanRotation.data.reshape(-1, 4)[k]
  _unpad(ebsd, empty[has])
  # a measured cell in no grain is notIndexed, a cell of padding stays padding
  out = empty[~has & ~ebsd.isPadding.reshape(-1)[empty]]
  ph[out] = 0
  gid[out] = 0


# cells of padding that now hold a value are cells of the map
def _unpad(ebsd, sel):
  if ebsd._pad is not None:
    # a copy of a map shares the flags with its source
    pad = ebsd._pad.copy()
    pad.reshape(-1)[sel] = False
    ebsd._pad = pad if pad.any() else None


# the orientations of the pixels written back from the field, and for the ones that were
# solved for the phase and every label of the pixel they were solved from
def _writeBack(ebsd, comp, ref, v, isData, source, sel):
  rot = ebsd.rotations.data.reshape(-1, 4)
  ph = ebsd.phaseId.reshape(-1)
  w = sel & (comp > 0)
  rot[w] = _expAt(ref[comp], v, w).astype(rot.dtype)
  filled = w & ~isData & (source >= 0)
  ph[filled] = ph[source[filled]]
  _unpad(ebsd, filled)
  for key, val in ebsd.prop.items():
    if not np.issubdtype(np.asarray(val).dtype, np.floating):
      flat = val.reshape(-1, *val.shape[ebsd.ndim:])
      flat[filled] = flat[source[filled]]


# one numeric property over the same graph, by the same solver, pinned where the
# orientations were
def _solveProperty(ebsd, key, F, comp, edges, isData, pin):
  if key not in ebsd.prop:
    raise ValueError(f"the map carries no property {key!r}")
  val = ebsd.prop[key]
  if not np.issubdtype(val.dtype, np.floating):
    raise ValueError(f"the property {key!r} is not numeric, so it is copied and not solved")
  if not isinstance(F, variationalFilter):
    raise ValueError(f"a property is smoothed by a variational filter, not by {type(F).__name__}")
  flat = val.reshape(len(comp), -1)
  f = flat.astype(float)
  # a pixel with no measurement starts at the mean of the ones of its component that have it
  nc = int(comp.max(initial=0))
  data = (comp > 0) & isData & ~np.any(np.isnan(f), axis=1)
  cnt = np.bincount(comp[data], minlength=nc + 1)
  for c in range(f.shape[1]):
    mean = np.bincount(comp[data], f[data, c], nc + 1) / np.maximum(cnt, 1)
    f[~data & (comp > 0), c] = mean[comp[~data & (comp > 0)]]
  x = _applyVariational(F, ebsd, comp, edges, isData, f, pin)
  w = (comp > 0) & (~isData if pin else np.ones(len(comp), dtype=bool))
  flat[w] = x[w].astype(val.dtype)


# ------------------------------------------------------------------ repair

def fillByGrainId(ebsd, grains):
  """the map with every pixel of a grain given the mean orientation of that grain, which is the
  map the grains describe rather than the one they were built from"""
  if "grainId" not in ebsd.prop:
    raise ValueError("the map carries no grain id; reconstruct it with calcGrains(ebsd)")
  out = ebsd.copy()
  byId = np.full(int(grains.id.max(initial=0)) + 1, -1, dtype=np.int64)
  byId[grains.id] = np.arange(len(grains))
  gid = out.prop["grainId"].reshape(-1).astype(np.int64)
  pos = np.where((gid > 0) & (gid < len(byId)), byId[np.clip(gid, 0, len(byId) - 1)], -1)
  ok = pos >= 0
  out.rotations.data.reshape(-1, 4)[ok] = grains.meanRotation.data.reshape(-1, 4)[pos[ok]]
  out.phaseId.reshape(-1)[ok] = grains.phaseId[pos[ok]]
  return out


def reduce(ebsd, factor=2):
  """every n-th measurement along each lattice direction, which is the map at a coarser step;
  MATLAB's reduce

    ebsd = reduce(ebsd)        # every second pixel along both lattice directions
    ebsd = reduce(ebsd, 3)     # every third pixel

  The measurements are selected by their lattice index, so a square, a hexagonal and a
  rotated lattice are all reduced alike; the kept sites form the same kind of lattice factor
  times coarser, which the result carries. A gridded map gives a grid, a list of
  measurements a list.

  Input
    ebsd - EBSD with a lattice
  Output
    ebsd - EBSD, the kept measurements
  Keywords
    factor - the step of the reduction, an integer of at least one
  See also
    gridify, EBSD.lattice"""
  if factor < 1:
    raise ValueError("the factor of a reduction is at least one")
  l = mapLattice(ebsd)
  out = ebsd[np.flatnonzero((l.ij[:, 0] % factor == 0) & (l.ij[:, 1] % factor == 0))]
  # the sublattice is the same kind of lattice factor times coarser
  l2 = out.lattice
  if l2 is not None:
    out.lattice = lattice._build(l2.plane, l2.origin, l2.A * factor, l2.ij // factor, l2.shape)
  # a grid stays a grid, a list a list
  return out.gridify() if ebsd.isGridded else out


def interp(ebsd, pos, y=None, F=None, fill=True, extrapolate=True):
  """the map read at positions of one's own choosing, a new map with these positions

    ebsd_new = interp(ebsd, x, y)                        # at the coordinates x, y
    ebsd_new = interp(ebsd, pos)                         # at the positions, a vector3d
    ebsd_new = interp(ebsd, x, y, fill=False, extrapolate=False)  # empty outside the measurements

  The orientation at a position is interpolated linearly from the corners of the lattice cell
  it falls in, over the corners that share the grain (or, without grain ids, the phase) of
  the nearest measured corner only, so nothing is averaged across a grain boundary. The
  phase and the other properties are those of the nearest measured corner. MATLAB takes the
  nearest measurement within its pixel instead.

  With fill=True the holes of the map are first filled as fill(ebsd, F) fills them. A
  position no measured corner surrounds takes the nearest measurement with
  extrapolate=True, and is notIndexed otherwise.

  Input
    ebsd - EBSD, a map on a lattice
    pos - vector3d, or an array of rows (x, y) or (x, y, z), the positions; or the x
        coordinates, with y
    y - the y coordinates, with pos the x coordinates
  Output
    ebsd_new - EBSD, the map at the positions, in the order given
  Keywords
    F - EBSDFilter, the variational filter the holes are filled by; by default l1TVFilter()
    fill - fill the holes of the map before interpolating
    extrapolate - a position outside the measured cells takes the nearest measurement
        instead of staying empty
  See also
    fill, smooth, gridify, EBSDGrid"""
  if y is not None:
    pos = np.stack([np.asarray(pos, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)], axis=1)
  src = ebsd.fill(F) if fill else ebsd
  l = mapLattice(src)
  lk = latticeLookup(l)
  comp, _, isData, _ = smoothingGraph(src)
  u, ref = tangentField(src, comp, isData)
  P = l.plane.to2d(src.pos.data.reshape(-1, 3))
  # the positions as rows of their two or three coordinates
  if isinstance(pos, vector3d):
    Q = np.asarray(pos.data, dtype=float).reshape(-1, 3)
  else:
    Q = np.asarray(pos, dtype=float).reshape(-1, np.shape(pos)[-1])
  Q3 = Q if Q.shape[1] == 3 else l.plane.from2d(Q)
  Q = l.plane.to2d(Q3) if Q.shape[1] == 3 else Q
  n = len(Q)
  ij = (Q - l.origin) @ np.linalg.inv(l.A).T
  i0, j0 = np.floor(ij).astype(np.int64).T
  s, t = ij[:, 0] - i0, ij[:, 1] - j0
  corners = [(i0, j0, (1 - s) * (1 - t)), (i0 + 1, j0, s * (1 - t)), (i0, j0 + 1, (1 - s) * t), (i0 + 1, j0 + 1, s * t)]
  # the nearest measured corner of every cell
  best, bd = np.full(n, -1, dtype=np.int64), np.full(n, np.inf)
  ks = []
  for i, j, _ in corners:
    inside = (i >= 0) & (j >= 0) & (i < lk.shape[0]) & (j < lk.shape[1])
    k = np.full(n, -1, dtype=np.int64)
    k[inside] = lk[i[inside], j[inside]]
    ks.append(k)
    ok = (k >= 0) & isData[np.maximum(k, 0)]
    d = np.where(ok, np.linalg.norm(P[np.maximum(k, 0)] - Q, axis=1), np.inf)
    closer = d < bd
    bd[closer], best[closer] = d[closer], k[closer]
  none = best < 0
  if none.any() and extrapolate:
    from scipy.spatial import cKDTree
    data = np.flatnonzero(isData)
    best[none] = data[cKDTree(P[data]).query(Q[none])[1]]
  have = best >= 0
  c = np.where(have, comp[np.maximum(best, 0)], 0)
  acc, w = np.zeros((n, 3)), np.zeros(n)
  for k, (_, _, wij) in zip(ks, corners):
    ok = have & (k >= 0) & (wij > 0)
    ok[ok] &= comp[k[ok]] == c[ok]
    acc[ok] += wij[ok, None] * u[k[ok]]
    w[ok] += wij[ok]
  T = src.rotations.data.dtype
  rot = np.full((n, 4), np.nan, dtype=T)
  solved = have & (c > 0) & (w > 0)
  rot[solved] = qmul(ref[c[solved]], rotationOf(acc[solved] / w[solved, None]))
  nearest = have & ~solved
  rot[nearest] = src.rotations.data.reshape(-1, 4)[best[nearest]]
  empty = [p for p in range(len(src.CSList)) if not src.CSList.isIndexed(p)]
  ph = np.full(n, empty[0] if empty else 0, dtype=np.int8)
  ph[have] = src.phaseId.reshape(-1)[best[have]]
  prop = {}
  for key, val in src.prop.items():
    flat = val.reshape(len(comp), *val.shape[src.ndim:])
    out = np.full((n,) + flat.shape[1:], np.nan if np.issubdtype(flat.dtype, np.floating) else 0, dtype=flat.dtype)
    out[have] = flat[best[have]]
    prop[key] = out
  return EBSD(Q3.astype(T), rotation(rot), ph, src.CSList, prop, scanUnit=src.scanUnit, frame=src.frame, meta=src.meta)


for _name, _f in (("smooth", smooth), ("fill", fill), ("fillByGrainId", fillByGrainId), ("reduce", reduce), ("subsample", reduce), ("interp", interp)):
  install(EBSD, _f, _name)

__all__ = ["meanFilter", "medianFilter", "KuwaharaFilter"]
