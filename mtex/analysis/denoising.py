# denoising and repair: the filters that replace an orientation by one computed from its
# neighbours, and the repairs that give a pixel one at all (MATLAB EBSDSmoothing/,
# @EBSD/smooth, fill, fillByGrainId, reduce, interp; Julia denoising.jl)
#
#   ebsd.smooth(), ebsd.smooth(F), ebsd.smooth(F, grains), ebsd.smooth(F, props=['ci'])
#   ebsd.fill(), ebsd.fill(grains), ebsd.fill(mask), ebsd.fillByGrainId(grains), ebsd.reduce(n)
#   ebsd.interp(pos), ebsd.interp(x, y)
#   meanFilter(numNeighbours=1), medianFilter(numNeighbours=1), KuwaharaFilter(numNeighbours=1)
#
# Every filter works on the same object: the field of tangent vectors of the component a
# pixel belongs to, its grain where the map knows them and its phase otherwise, over the
# graph of neighbouring pixels. That makes the window filters ordinary image filters and
# the variational ones an energy over a graph, and one implementation serves a square, a
# hexagonal, a rotated and a sheared grid. Filling, as MATLAB's, repeats the nearest
# measurement of the pixel's own grain; smooth fills first and then solves for the filled
# pixels without a data term, so what they end up with is the continuation of the field
# around them. interp is the filled field read at positions of one's own choosing.
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
from .filters.variational import EBSDFilter, l1TVFilter, solveField, variationalFilter
from .filters.window import windowFilter, meanFilter, medianFilter, KuwaharaFilter
from ..geometry.install import install



# ------------------------------------------------------------------ the window filters

# ------------------------------------------------------------------ the field

def smoothingGraph(ebsd):
  """the components a filter works within, the neighbour pairs inside them, and which pixels
  carry a measurement. The components are the grains where the map knows them and the phases
  where it does not, of the pixels with an orientation"""
  l = mapLattice(ebsd)
  lk, m = latticeLookup(l), mapPairs(ebsd)
  n = len(l.ij)
  isData = m.indexed.copy()
  comp = (m.phase.astype(np.int64) if m.grain is None else m.grain.astype(np.int64)).copy()
  comp[~isData | (comp < 0)] = 0
  a, b = [], []
  k = np.arange(n)
  for o in l.stencil.astype(np.int64):
    nb = neighbourAt(lk, l.ij, o)[0]
    ok = (nb > k) & (comp > 0)
    ok[ok] &= comp[nb[ok]] == comp[ok]
    a.append(k[ok])
    b.append(nb[ok])
  edges = np.stack([np.concatenate(a), np.concatenate(b)], axis=1) if a else np.zeros((0, 2), dtype=np.int64)
  return comp, edges, isData


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


# the window filters over the lattice, never across a component: every pixel of a component
# takes the filter of the measured pixels of its window, a filled one too, and a pixel whose
# window holds no measurement keeps its value, as a fixed one does. The compiled loop, or the
# NumPy gather of the windows without it
def _applyWindow(F, ebsd, comp, isData, u, fixed):
  out = _windowed(F, ebsd, comp, isData, u)
  if fixed is not None:
    out[fixed] = u[fixed]
  return out


def _windowed(F, ebsd, comp, isData, u):
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
  act = comp > 0
  W = np.full((len(u), len(offs), 3), np.nan)
  for c, o in enumerate(offs):
    nb = neighbourAt(lk, l.ij, o)[0]
    ok = nb >= 0
    ok &= (comp[np.maximum(nb, 0)] == comp) & isData[np.maximum(nb, 0)]
    W[ok, c] = u[nb[ok]]
  has = ~np.all(np.isnan(W[..., 0]), axis=1)
  act &= has
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


def _applyFilter(F, ebsd, comp, edges, isData, u, fixed):
  if isinstance(F, variationalFilter):
    return _applyVariational(F, ebsd, comp, edges, isData, u, fixed)
  if isinstance(F, windowFilter):
    return _applyWindow(F, ebsd, comp, isData, u, fixed)
  raise TypeError(f"{type(F).__name__} is not an EBSD filter")


# the variational filters over the whole map at once: the edges never cross a component,
# so the one system is block diagonal in them, and a pixel of no component has no edge and
# no data term and stays where it is, as a fixed one (the mask fixed, or None) does
def _applyVariational(F, ebsd, comp, edges, isData, f, fixed):
  out = f.copy()
  if len(edges) == 0:
    return out
  x = solveField(F, mapLattice(ebsd), edges, f.astype(float), isData & (comp > 0), fixed, comp)
  w = comp > 0
  out[w] = x[w]
  return out


# ------------------------------------------------------------------ smooth and fill

def smooth(ebsd, F=None, grains=None, fill=True, fixed=None, props=()):
  """the map with its orientations denoised, every one replaced by one computed from its
  neighbours, within the grains where the map knows them and within the phases otherwise

    ebsd_s = smooth(ebsd)                                # the default filter, total variation
    ebsd_s = smooth(ebsd, halfQuadraticFilter())         # another filter
    ebsd_s = smooth(ebsd, F, grains)                     # the grains of the empty sites from the outlines
    ebsd_s = smooth(ebsd, F, fill=mask)                  # only the pixels of the mask filled
    ebsd_s = smooth(ebsd, F, fixed=ebsd.ci > 0.9)        # the pixels of the mask kept as they are
    ebsd_s = smooth(ebsd, F, props=('bc',))              # a numeric property smoothed along

  The orientations are turned into a field of tangent vectors per connected component of
  the map (a grain, if calcGrains has written the grain ids, else a phase) and the filter
  acts on that field: a variational filter (splineFilter, halfQuadraticFilter, l1TVFilter,
  infimalConvolutionFilter) solves for the smoothed field, a window filter (meanFilter,
  medianFilter, KuwaharaFilter) replaces each pixel by its neighbourhood. Nothing is mixed
  across a grain boundary or a phase boundary.

  The pixels without an orientation, notIndexed or never measured, are filled as
  fill(ebsd, grains) fills them and smoothed along: a variational filter solves for them
  without a data term, a window filter takes the filter of the measured pixels of their
  window. Grain ids and phases stay as fill leaves them. The map passed is not changed;
  MATLAB's second output, the filter, is the F passed.

  A fixed pixel keeps its orientation, and the properties smoothed along keep their values
  there; it is no unknown of a variational filter but a value its neighbours are smoothed
  towards, and a window filter counts it among the neighbours of the others. A fixed pixel
  without an orientation keeps the one fill gives it.

  Input
    ebsd - EBSD, the map
    F - EBSDFilter, the filter; by default l1TVFilter()
    grains - grain2d of the map, the outlines that decide the grain of an empty site
  Output
    ebsd_s - EBSD, the smoothed map
  Keywords
    fill - True to fill the pixels without an orientation, False to leave them as they are,
        or a boolean mask of the pixels of the map filling is restricted to
    fixed - a boolean mask of the pixels of the map that keep their orientation
    props - the names of numeric properties of the map smoothed with the orientations, by a
        variational filter

  Example
  Denoise a synthetic map containing two grains and reproducible orientation noise.
  Reconstruct the grains first so the filter respects their boundary.

    import numpy as np
    import matplotlib.pyplot as plt
    from mtex import (EBSD, crystalFrame, phaseList, rotation, vector3d, degree,
                      calcGrains, smooth, meanFilter, KAM, plot, mtexColorbar, mtexTitle, newMtexFigure)

    x, y = np.meshgrid(np.arange(20), np.arange(12))
    base = np.where(x < 10, 0.0, 20.0)
    noise = np.random.default_rng(7).normal(0, 0.4, x.shape)
    rot = rotation.byAxisAngle(vector3d.Z, (base + noise).ravel()*degree)
    cs = crystalFrame('m-3m')
    ebsd = EBSD(np.c_[x.ravel(), y.ravel()], rot, np.ones(x.size, dtype=int), phaseList(cs))
    grains = calcGrains(ebsd, angle=10*degree)
    filtered = smooth(ebsd, meanFilter(numNeighbours=1))
    for data, title in ((ebsd, 'Before smoothing'), (filtered, 'After smoothing')):
        newMtexFigure()
        kam = KAM(data)/degree
        print(title, 'median KAM (degrees):', round(float(np.nanmedian(kam)), 3))
        plot(data, kam, colorRange=(0, 1.5), micronbar=False)
        mtexColorbar(title='KAM (degrees)')
        mtexTitle(title)
    print('Grain ids unchanged:', np.array_equal(filtered.grainId, ebsd.grainId))
    plt.show()

  The returned map has new orientations and retains the original segmentation. Lower KAM
  shows reduced local variation in this synthetic example; it is not a general measure of
  denoising quality. Choose a filter and length scale appropriate to the features to retain.

  See also
    fill, interp, halfQuadraticFilter, splineFilter, meanFilter, EBSDDenoising, EBSDFilling"""
  F = l1TVFilter() if F is None else F
  if isinstance(fill, (bool, np.bool_)):
    mask, fill = None, bool(fill)
  elif np.ndim(fill) == 0:
    raise TypeError(f"fill is True, False or a mask of the pixels, not {fill!r}")
  else:
    mask = _asMask(ebsd, fill, "fill")
  fixed = _asMask(ebsd, fixed, "fixed")
  # the empty sites exist on the grid, the masks go along as properties
  out = ebsd.copy()
  for key, m in (("_fillMask", mask), ("_fixedMask", fixed)):
    if m is not None:
      out.prop[key] = m.reshape(ebsd.shape).astype(float)
  if fill is not False and not out.isGridded:
    out = out.gridify()
  estimate = np.full(out.size, fill is not False) if mask is None else out.prop.pop("_fillMask").reshape(-1) == 1
  if fixed is not None:
    fixed = out.prop.pop("_fixedMask").reshape(-1) == 1
  measured = smoothingGraph(out)[2]
  rot0 = out.rotations.data.reshape(-1, 4).copy()
  if fill is not False:
    out = out.fill(grains, None if mask is None else estimate)
  comp, edges, isData = smoothingGraph(out)
  isData &= measured
  u, ref = tangentField(out, comp, isData)
  v = _applyFilter(F, out, comp, edges, isData, u, fixed)
  rot = out.rotations.data.reshape(-1, 4)
  w = comp > 0 if fixed is None else (comp > 0) & ~fixed
  rot[w] = _expAt(ref[comp], v, w).astype(rot.dtype)
  # the pixels without an orientation that are not to be filled, as they were
  keep = ~measured & ~estimate
  rot[keep] = rot0[keep]
  for key in props:
    _solveProperty(out, key, F, comp, edges, isData, fixed)
  return out if ebsd.isGridded or not out.isGridded else _asList(ebsd, out)


def fill(ebsd, grains=None, mask=None):
  """the map with every pixel without an orientation given the one of the nearest measured
  pixel of its own grain; MATLAB's fill

    ebsd_f = fill(ebsd)                                  # every pixel without an orientation in the scan
    ebsd_f = fill(ebsd, grains)                          # their grains from the outlines
    ebsd_f = fill(ebsd, mask)                            # restricted to the pixels of the mask
    ebsd_f = fill(ebsd, grains, mask)

  A pixel that was not measured and a notIndexed pixel are the same to fill: every pixel
  without an orientation inside the convex hull of the indexed measurements is filled, an
  empty site of the lattice as well. A mask restricts this to its pixels; it names only
  pixels of the map, so the empty sites of a list need gridify first.

  A filled pixel takes the orientation and the phase of the nearest measured pixel of its
  grain. With grain ids every pixel to fill needs a grain: an empty site takes the grain
  whose outline holds its whole cell if grains are given, else the grain id of the nearest
  pixel. A pixel on a grain boundary, grain id 0, stays or becomes notIndexed, and so do the
  pixels of notIndexed grains. Without grain ids the nearest measured pixel decides. The other
  properties keep what the pixel had. The map passed is not changed.

  Input
    ebsd - EBSD, the map
    grains - grain2d of the map, the outlines that decide the grain of an empty site; a mask
        in its place is taken as the mask
    mask - boolean mask of the pixels of the map filling is restricted to
  Output
    ebsd_f - EBSD, the filled map

  Example
  Fill a three-by-three hole in a synthetic orientation gradient. The missing cells
  exist on the grid and have phase zero and NaN rotations; no data file is needed.

    import numpy as np
    from mtex import EBSD, crystalFrame, phaseList, rotation, vector3d, degree, gridify, fill

    x, y = np.meshgrid(np.arange(11), np.arange(11))
    rot = rotation.byAxisAngle(vector3d.Z, (0.3*x + 0.2*y).ravel()*degree)
    hole = ((x >= 4) & (x <= 6) & (y >= 4) & (y <= 6)).ravel()
    rot.data[hole] = np.nan
    phase = np.where(hole, 0, 1)
    ebsd = gridify(EBSD(np.c_[x.ravel(), y.ravel()], rot, phase, phaseList(crystalFrame('m-3m'))))
    filled = fill(ebsd)
    print('Indexed cells before and after:', np.count_nonzero(ebsd.isIndexed), np.count_nonzero(filled.isIndexed))

  Every hole cell repeats a measured orientation of its neighbourhood; smooth(ebsd, F)
  turns such a fill into a continuous field.

  See also
    smooth, interp, gridify, calcGrains, EBSDFilling"""
  if grains is not None and not isinstance(grains, grain2d):
    if mask is not None:
      raise TypeError(f"grains is a grain2d, not {type(grains).__name__}")
    grains, mask = None, grains
  if isinstance(mask, EBSDFilter):
    raise TypeError("fill copies the nearest measurement and takes no filter; smooth(ebsd, F) fills and filters")
  mask = _asMask(ebsd, mask, "mask")
  hasGrains = "grainId" in ebsd.prop
  if grains is not None and not hasGrains:
    raise ValueError("filling within grains needs the grain ids of the map; reconstruct them with calcGrains(ebsd)")
  # the empty sites of a list exist on its grid; a mask names the pixels as they are
  out = (ebsd.gridify() if mask is None and not ebsd.isGridded else ebsd).copy()
  sel = np.ones(out.size, dtype=bool) if mask is None else mask
  rot = out.rotations.data.reshape(-1, 4)
  ph = out.phaseId.reshape(-1)
  gid = out.prop["grainId"].reshape(-1) if hasGrains else None
  pos = np.asarray(out.pos.data, dtype=float).reshape(-1, 3)
  pad = out.isPadding.reshape(-1).copy()
  measured = smoothingGraph(out)[2] & ~pad
  inScan = _inHull(out, measured)
  if hasGrains and grains is not None:
    # an empty site whose whole cell lies in a grain of the map takes its id and phase, the
    # corners pulled slightly inward as calcGrains does; any other site of the scan is
    # notIndexed as a measurement a grain boundary passes through
    held = grains[np.isin(grains.id, gid)]
    empty = np.flatnonzero(pad & sel)
    corners = 0.95 * np.asarray(out.unitCell.data, dtype=float).reshape(-1, 3)
    at = [held.findByLocation(vector3d(pos[empty] + c, frame=out.frame)) for c in corners]
    has = (at[0] >= 0) & np.logical_and.reduce([a == at[0] for a in at])
    k = at[0][has]
    ph[empty[has]] = held.phaseId[k]
    gid[empty[has]] = held.id[k]
    inScan[empty[has]] = True
    rest = empty[~has & inScan[empty]]
    ph[rest] = 0
    gid[rest] = 0
    pad[empty[has]] = False
    pad[rest] = False
    _unpad(out, np.concatenate([empty[has], rest]))
  elif hasGrains:
    # without grains an empty site of the scan takes the grain and phase of its nearest pixel
    empty = np.flatnonzero(pad & sel & inScan)
    near = np.flatnonzero(~pad)
    if len(empty) and len(near):
      from scipy.spatial import cKDTree
      k = near[cKDTree(pos[near]).query(pos[empty])[1]]
      ph[empty], gid[empty] = ph[k], gid[k]
      pad[empty] = False
      _unpad(out, empty)
  else:
    pad &= ~inScan
  target = np.flatnonzero(~measured & ~pad & inScan & sel)
  # the nearest measured pixel with the grains moved apart by more than the map's diameter
  group = gid.astype(float) if hasGrains else np.zeros(len(ph))
  group[pad] = -1
  L = 2 * np.linalg.norm(np.ptp(pos[measured], axis=0)) + 1 if measured.any() else 1.0
  _copyNearest(out, rot, ph, gid, np.c_[pos, L * group], np.flatnonzero(measured), target, L / 2)
  return out if ebsd.isGridded or mask is not None else _asList(ebsd, out)


# the cells of a grid made from a list that are not padding, the pixels of the list first in
# their order and the sites added after them
def _asList(ebsd, out):
  out = out[np.flatnonzero(~out.isPadding.reshape(-1))]
  ids, orig = np.asarray(out.id).reshape(-1), np.asarray(ebsd.id).reshape(-1)
  sorter = np.argsort(ids)
  first = sorter[np.searchsorted(ids, orig, sorter=sorter)]
  return out[np.concatenate([first, np.setdiff1d(np.arange(len(ids)), first)])]


# which pixels lie inside the convex hull of the selected ones, the scanned area: on the inner
# side of every facet of the hull, up to rounding
def _inHull(ebsd, sel):
  from scipy.spatial import ConvexHull, QhullError
  P = mapLattice(ebsd).plane.to2d(np.asarray(ebsd.pos.data, dtype=float).reshape(-1, 3))
  try:
    E = ConvexHull(P[sel]).equations
  except (QhullError, ValueError):
    return sel.copy()
  tol = 1e-9 * max(float(np.abs(P[sel]).max()), 1.0)
  return np.all(P @ E[:, :2].T + E[:, 2] <= tol, axis=1)


# a mask of the pixels of a map, the argument name, as a flat boolean array; None for none
def _asMask(ebsd, mask, name):
  if mask is None:
    return None
  mask = np.asarray(mask)
  if mask.dtype.kind not in "bui":
    raise TypeError(f"{name} is a boolean mask of the pixels, not {mask.dtype}")
  if mask.size != ebsd.size:
    raise ValueError("a mask has one entry per pixel of the map")
  return mask.astype(bool).reshape(-1)


# orientation, phase and grain id of the nearest source for every target closer than maxDist
def _copyNearest(ebsd, rot, ph, gid, X, src, tgt, maxDist):
  if not len(src) or not len(tgt):
    return
  from scipy.spatial import cKDTree
  d, k = cKDTree(X[src]).query(X[tgt])
  ok = d < maxDist
  s, t = src[k[ok]], tgt[ok]
  rot[t] = rot[s]
  ph[t] = ph[s]
  if gid is not None:
    gid[t] = gid[s]
  _unpad(ebsd, t)


# cells of padding that now hold a value are cells of the map
def _unpad(ebsd, sel):
  if ebsd._pad is not None:
    # a copy of a map shares the flags with its source
    pad = ebsd._pad.copy()
    pad.reshape(-1)[sel] = False
    ebsd._pad = pad if pad.any() else None


# one numeric property over the same graph, by the same solver, fixed where the
# orientations are
def _solveProperty(ebsd, key, F, comp, edges, isData, fixed):
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
  x = _applyVariational(F, ebsd, comp, edges, isData, f, fixed)
  w = (comp > 0) if fixed is None else (comp > 0) & ~fixed
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


def interp(ebsd, pos, y=None, fill=True, extrapolate=True):
  """the map read at positions of one's own choosing, a new map with these positions

    ebsd_new = interp(ebsd, x, y)                        # at the coordinates x, y
    ebsd_new = interp(ebsd, pos)                         # at the positions, a vector3d
    ebsd_new = interp(ebsd, x, y, fill=False, extrapolate=False)  # empty outside the measurements

  The orientation at a position is interpolated linearly from the corners of the lattice cell
  it falls in, over the corners that share the grain (or, without grain ids, the phase) of
  the nearest measured corner only, so nothing is averaged across a grain boundary. The
  phase and the other properties are those of the nearest measured corner. MATLAB takes the
  nearest measurement within its pixel instead.

  With fill=True the holes of the map are first filled as fill(ebsd) fills them. A
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
    fill - fill the holes of the map before interpolating
    extrapolate - a position outside the measured cells takes the nearest measurement
        instead of staying empty
  See also
    fill, smooth, gridify, EBSDGrid"""
  if y is not None:
    pos = np.stack([np.asarray(pos, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)], axis=1)
  src = ebsd.fill() if fill else ebsd
  l = mapLattice(src)
  lk = latticeLookup(l)
  comp, _, isData = smoothingGraph(src)
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
