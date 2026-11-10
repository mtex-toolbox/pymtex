# grain reconstruction (MATLAB EBSD/calcGrains, Julia calcgrains.jl)
#
#   grains = calcGrains(ebsd)                       15 degrees, as MATLAB
#   grains = calcGrains(ebsd, angle=10 * degree)
#   grains = calcGrains(ebsd, angle=[10 * degree, 2 * degree], minPixel=5, alpha=3.1)
#   grains = calcGrains(ebsd, criterion=gbcSoft(10 * degree))
#   grains = calcGrains(ebsd, fmc=3, minPixel=10, verbose=True)     fast multiscale clustering, gbcFMC
#   grains = calcGrains(ebsd, alphaCrack=15)                        cracks up to 30 pixels wide bridged
#   kinds, parts = notIndexedKinds(ebsd, alpha=3.1)                 the kind of every notIndexed pixel
#
# The map is written into: every measurement gets its grain id in ebsd.grainId, and the
# cull marks the pixels of the grains it drops as not indexed, so the map that comes back
# is the one the grains were built from. Keep the original with calcGrains(ebsd.copy()).
# The pipeline is the decomposition (M2), the criterion on its faces and the components
# of what it joins (M3), and the grain table over the components.
from __future__ import annotations

from functools import partial

import numpy as np

from ..geometry.symreduce import chordAngle, closestImage
from .boundary import buildBoundary
from .criteria import DEFAULT_ANGLE, connectivity, gbcAngle, gbcEqual, grainBoundaryCriterion
from .decomposition import closeLattice, decompose, splitPinches
from .criteria.fmc import gbcFMC
from .grains import grain2d
from .kernels import absorbNumba, grainKernel
from .kinds import INDEXED, KINDS, SPECK, VOID, kindCodes, kindOfParts
from .lattice import STENCIL
from .quadruple import splitQuadruplePoints
from .segment import segment
from .straighten import straightenInFill


def calcGrains(ebsd, angle=DEFAULT_ANGLE, threshold=None, criterion=None, minPixel=1, alpha=3.1, alphaPore=np.inf, alphaPocket=None,
               alphaCrack=0, kinds=None, removeQuadruplePoints=False, mcl=None, fmc=None, verbose=False, backend=None, grainId=False,
               straighten=True):
  """the grains of a map, the pixels joined across every boundary the criterion keeps: a
  misorientation angle below angle (or threshold), or a grainBoundaryCriterion; minPixel
  culls smaller grains, alpha bounds the decomposition, alphaPore, alphaPocket and alphaCrack
  close the notIndexed regions of that kind (notIndexedKinds) in place of alpha, by default inf,
  alpha and 0, so a pore is part of its grain and a crack stays open; kinds hands the kind of
  every pixel back, edited or not; straighten draws the boundaries through
  the filled cells straight (False: along the bisectors between the pixels around them), mcl and
  fmc select Markov clustering and fast multiscale clustering, grainId=True joins the pixels of
  one grain id stored with the map, ebsd.prop['grainId']. The map keeps the grain ids"""
  if isinstance(angle, grainBoundaryCriterion):
    criterion, angle = angle, None
  if grainId:
    stored = ebsd.prop.get("grainId")
    if stored is None:
      raise ValueError("calcGrains(ebsd, grainId=True) needs the grain ids stored with the map, ebsd.prop['grainId']")
    stored = np.asarray(stored).reshape(ebsd.shape).copy()
    # a stored id of 0 is no grain, as it is in the output
    ebsd.phaseId[stored == 0] = 0
    criterion = gbcEqual(stored)
  if criterion is None and fmc is not None:
    criterion = gbcFMC(fmc, verbose=verbose)
  crit = criterionOf(angle, threshold, criterion)
  # a criterion that absorbs the small regions itself takes minPixel instead of the cull
  if crit.handlesMinPixel:
    crit = crit.setMinPixel(minPixel) if minPixel > 1 else crit
    minPixel = 1
  releaseAbsorbed(ebsd)
  decomposed = partial(decompose, ebsd, alpha, backend=backend, alphaPore=alphaPore, alphaPocket=alphaPocket, alphaCrack=alphaCrack,
                       kinds=kinds, criterion=crit)
  c = decomposed()
  # the cull changes the map the closing runs on, so it is a pass over the whole reconstruction;
  # the first sizes the grains before the pinches are split, as MATLAB's minPixelMask, and the
  # decomposition repeats until no grain re-fragments below minPixel
  if minPixel > 1 and markUndersized(ebsd, segment(c, ebsd, crit, backend, mcl), minPixel):
    c = decomposed()
  while True:
    c = splitPinches(c)
    seg = segment(c, ebsd, crit, backend, mcl)
    if minPixel <= 1 or not markUndersized(ebsd, seg, minPixel):
      break
    c = decomposed()
  if straighten:
    c, seg = straightenInFill(c, seg)
  # the diagonal ambiguity is decided on the map the grains are built from, so the split
  # runs after the cull and before anything reads the boundary
  if removeQuadruplePoints:
    c, seg = splitQuadruplePoints(c, seg, ebsd, crit)
  grains = grainTable(seg, ebsd, buildBoundary(c, seg, ebsd), buildBoundary(c, seg, ebsd, inner=True))
  checkClosed(grains.boundary)
  ebsd.grainId = absorbInteriorPixels(c, ebsd, seg.grainId, grains.boundary, backend=backend).reshape(ebsd.shape)
  absorbNotIndexed(ebsd, grains)
  return grains


def notIndexedKinds(ebsd, angle=DEFAULT_ANGLE, threshold=None, criterion=None, alpha=3.1, kinds=None):
  """the kinds of the notIndexed regions of a map, as calcGrains sees them; MATLAB's notIndexedKinds

    kinds, parts = notIndexedKinds(ebsd, alpha=3.1)

    # plot the cracks and reconstruct with a kind corrected by hand
    plot(ebsd[kinds == 'crack'])
    kinds[(kinds == 'crack') & (ebsd.x < 1000)] = 'pocket'
    grains = calcGrains(ebsd, alphaCrack=15, kinds=kinds)

  The closing with alpha fills every notIndexed region narrower than about 2 alpha pixels. What
  it leaves open are the thick parts, and each of them is a pore, a pocket, a crack or a void.
  calcGrains closes a part with the radius of its kind, alphaPore, alphaPocket or alphaCrack. By
  default it fills the pores whole and leaves the cracks entirely open.

  The kinds are decided in this order: a part with Feret^2 / area >= 5 whose boundary runs along
  grains for at least half of its length is a crack; one touching the edge of the scan is a void;
  one whose rim is held to 85 percent by pixels the criterion connects is a pore; the rest are
  pockets. What lies between two parts of a crack, joined to both by notIndexed pixels, is part
  of the crack.

  Input
    ebsd - EBSD on a lattice
  Output
    kinds - the kind of every pixel, an array of the map's shape: 'pore', 'pocket', 'crack',
        'void', 'speck' (a thin region, filled by alpha) or 'indexed'
    parts - the thick parts as a dict of columns: id, kind, area in pixels, feret, the Feret
        diameter, elong, Feret^2 / area, border, gap, the boundary cells against notIndexed cells,
        rim, those against grains, and share, the rim share of the largest orientation
  Keywords
    alpha - the closing radius in pixels
    angle - the misorientation angle of calcGrains, since the rim test uses its criterion
    threshold - the same as angle, which it replaces when given
    criterion - a grainBoundaryCriterion instead of the angle
    kinds - a kind per pixel replacing the computed kind of every part it covers
  See also
    calcGrains, GrainReconstructionCracks"""
  crit = criterionOf(angle, threshold, criterion)
  l = ebsd.lattice
  if l is None:
    raise ValueError("the map is not on a lattice, so its notIndexed regions have no kinds")
  # an indexed pixel without rotation is notIndexed, as in calcGrains
  present = ebsd.isIndexed.reshape(-1) & ~np.isnan(ebsd.rotations.data.reshape(-1, 4)[:, 0])
  closed, isPresent, outside, pixelOf, lo = closeLattice(l, present, alpha, None if ebsd._pad is None else ~ebsd._pad.reshape(-1))

  # the criterion on pairs of pixels
  def connects(i, j):
    return np.asarray(connectivity(crit, ebsd, i, j)) > 0

  k, parts = kindOfParts(closed, isPresent, outside, pixelOf, connects, kindCodes(kinds), STENCIL[l.shape], l.A)
  k = np.where(isPresent, INDEXED, np.where(k > 0, k, np.where(outside, VOID, SPECK)))
  return KINDS[k[l.ij[:, 0] - lo[0], l.ij[:, 1] - lo[1]] - 1].reshape(ebsd.shape), parts


def criterionOf(angle, threshold, criterion):
  """the criterion of the options of calcGrains: a grainBoundaryCriterion, given as such or in
  place of the angle, or else the angle criterion with the angle or the threshold"""
  if isinstance(angle, grainBoundaryCriterion):
    return angle
  return criterion if criterion is not None else gbcAngle(angle if threshold is None else threshold)


def checkClosed(boundary):
  """every grain's boundary closes: within a grain, every vertex is met by an even number of
  segments; the first grain whose boundary does not is named in the error"""
  G, F = np.asarray(boundary.grainId), np.asarray(boundary.F)
  g, v = np.concatenate([G[:, 0], G[:, 1], G[:, 0], G[:, 1]]), np.concatenate([F[:, 0], F[:, 0], F[:, 1], F[:, 1]])
  nV = len(boundary.V)
  u, c = np.unique(v[g > 0] + (g[g > 0] - 1) * nV, return_counts=True)
  if (c % 2).any():
    raise ValueError(f"the boundary of grain {u[c % 2 == 1][0] // nV + 1} does not close")


def absorbNotIndexed(ebsd, grains):
  """a not indexed pixel inside a grain takes the phase of the grain and no rotation, as
  MATLAB's [grains, ebsd] = calcGrains returns the map; the padding of a grid stays empty"""
  gid = ebsd.grainId.reshape(-1)
  absorbed = ~ebsd.isIndexed.reshape(-1) & (gid > 0) & ~ebsd.isPadding.reshape(-1)
  if not absorbed.any():
    return
  ebsd.phaseId.reshape(-1)[absorbed] = np.asarray(grains.phaseId).reshape(-1)[gid[absorbed] - 1]
  ebsd.rotations.data.reshape(-1, 4)[absorbed] = np.nan


def releaseAbsorbed(ebsd):
  """a pixel a previous reconstruction absorbed has a phase but no rotation; it is segmented
  as the not indexed pixel it was, so that calcGrains on its own result gives the same grains"""
  blank = ebsd.isIndexed.reshape(-1) & np.isnan(ebsd.rotations.data.reshape(-1, 4)[:, 0])
  ebsd.phaseId.reshape(-1)[blank] = 0


def undersized(grainId, nGrains, isIndexed, minPixel):
  """the indexed pixels of the grains, numbered from 1, that came out below minPixel"""
  n = np.bincount(grainId, minlength=nGrains + 1)
  return (grainId > 0) & isIndexed & (n[grainId] < minPixel)


def markUndersized(ebsd, seg, minPixel):
  """mark the indexed pixels of the grains below minPixel as not indexed where they lie, and
  say whether there were any"""
  under = undersized(seg.grainId, seg.nGrains, ebsd.isIndexed.reshape(-1), minPixel)
  ebsd.phaseId.reshape(-1)[under] = 0
  return bool(under.any())


def absorbInteriorPixels(c, ebsd, grainId, boundary, nudge=0.95, backend=None):
  """give a grain to the pixels the closing left without a cell of their own

  A pixel at 0 whose unit cell, pulled towards its centre by nudge, no boundary segment meets
  takes the grain of a neighbour, sharing an edge or a corner, whose centre no segment separates
  from its own, and passes it on the same way; zero stays for a pixel a boundary runs through.
  Every segment is tested only against the pixels of the lattice cells it passes, so nothing is
  searched. MATLAB's absorbInteriorPixels."""
  out = grainId.copy()
  l = ebsd.lattice
  z = np.flatnonzero((out == 0) & ~ebsd.isPadding.reshape(-1))
  if l is None or len(z) == 0:
    return out
  # the raster of the lattice slots with a margin of one cell, -1 where the map has no pixel
  ij = l.ij.astype(np.int64) + 1
  sz = np.array([ij[:, 0].max() + 2, ij[:, 1].max() + 2])
  slot = ij[:, 0] * sz[1] + ij[:, 1]
  P = np.full(sz[0] * sz[1], -1, dtype=np.int64)
  P[slot] = np.arange(len(slot))
  xy = np.ascontiguousarray(l.plane.to2d(ebsd.pos.data.reshape(-1, 3))[:, :2], dtype=float)
  # the slot offsets of the neighbours sharing an edge before those sharing a corner
  o = np.array([(di, dj) for dj in (-1, 0, 1) for di in (-1, 0, 1)])
  d = np.linalg.norm(o @ l.A.T, axis=1)
  o = o[np.argsort(d, kind="stable")][np.sort(d) < 1.5 * l.dxy][1:]
  off = o[:, 0] * sz[1] + o[:, 1]
  # the segments, the cells they may meet found by the ideal lattice, their boxes widened by how far the
  # cell reaches past 1/2, the nudged cell about a centre with its outward edge normals and offsets
  V, F = np.asarray(boundary.V, dtype=float)[:, :2], np.asarray(boundary.F)
  a, b = np.ascontiguousarray(V[F[:, 0]]), np.ascontiguousarray(V[F[:, 1]])
  Ainv = np.ascontiguousarray(np.linalg.inv(l.A), dtype=float)
  origin = np.asarray(l.origin, dtype=float)
  corners = np.asarray(c.unitCell2d(), dtype=float)[:, :2]
  mg = np.abs(corners @ Ainv.T).max(axis=0) - 0.5 + 1e-6
  uc = nudge * corners
  e = np.roll(uc, -1, axis=0) - uc
  nrm = np.column_stack([e[:, 1], -e[:, 0]]) * np.sign(np.sum(e[:, 0] * np.roll(e[:, 1], -1) - e[:, 1] * np.roll(e[:, 0], -1)))
  h = np.sum(nrm * uc, axis=1)
  # the compiled loops, or their NumPy form below as fallback and oracle
  if not absorbNumba(z, slot, P, off, xy, uc, nrm, h, a, b, origin, Ainv, mg, sz, out, backend):
    _absorbPython(z, slot, P, off, xy, uc, nrm, h, a, b, origin, Ainv, mg, sz, out)
  return out


def _absorbPython(z, slot, P, off, xy, uc, nrm, h, a, b, origin, Ainv, mg, sz, out):
  # the neighbours of the pixels at 0, -1 for an empty slot, and the cells of interest
  nb = P[slot[z][:, None] + off]
  isI = np.zeros(len(P), dtype=bool)
  isI[slot[z]] = True
  isI[slot[nb[nb >= 0]]] = True
  # the segments of every cell of interest, as the cells' starts into the sorted list
  u1, u2 = (a - origin) @ Ainv.T + 1, (b - origin) @ Ainv.T + 1
  lo = np.maximum(np.floor(np.minimum(u1, u2) - mg + 0.5), 0).astype(np.int64)
  hi = np.minimum(np.floor(np.maximum(u1, u2) + mg + 0.5), sz - 1).astype(np.int64)
  w = hi[:, 0] - lo[:, 0] + 1
  n = w * (hi[:, 1] - lo[:, 1] + 1)
  s = np.repeat(np.arange(len(n)), n)
  r = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
  cell = (lo[s, 0] + r % w[s]) * sz[1] + lo[s, 1] + r // w[s]
  s, cell = s[isI[cell]], cell[isI[cell]]
  order = np.argsort(cell, kind="stable")
  s, cell = s[order], cell[order]
  first = np.concatenate([[0], np.cumsum(np.bincount(cell, minlength=len(isI)))])

  def segsOf(cells):
    m = first[cells + 1] - first[cells]
    i = np.repeat(np.arange(len(cells)), m)
    return i, s[np.repeat(first[cells], m) + np.arange(m.sum()) - np.repeat(np.cumsum(m) - m, m)]

  # a pixel is crossed where a segment and its nudged cell have no separating axis
  i, e = segsOf(slot[z])
  cen, A, B = xy[z[i]], a[e], b[e]
  d = B - A
  pr = (cen[:, :1] - A[:, :1] + uc[:, 0]) * d[:, 1:] - (cen[:, 1:] - A[:, 1:] + uc[:, 1]) * d[:, :1]
  pa = (A[:, :1] - cen[:, :1]) * nrm[:, 0] + (A[:, 1:] - cen[:, 1:]) * nrm[:, 1]
  pb = (B[:, :1] - cen[:, :1]) * nrm[:, 0] + (B[:, 1:] - cen[:, 1:]) * nrm[:, 1]
  sep = np.all(pr > 0, axis=1) | np.all(pr < 0, axis=1) | np.any((pa > h) & (pb > h), axis=1)
  crossed = np.zeros(len(z), dtype=bool)
  crossed[i[~sep]] = True
  # the grain passes along every link no segment of its two cells cuts
  u = np.flatnonzero(~crossed)
  tested = np.zeros((len(u), nb.shape[1]), dtype=bool)
  while True:
    nNew = 0
    for k in range(nb.shape[1]):
      j = np.flatnonzero((out[z[u]] == 0) & ~tested[:, k] & (nb[u, k] >= 0))
      j = j[out[nb[u[j], k]] > 0]
      tested[j, k] = True
      p, q = z[u[j]], nb[u[j], k]
      i, e = segsOf(np.concatenate([slot[p], slot[q]]))
      i = i % len(p)
      cut = np.zeros(len(p), dtype=bool)
      cut[i[_cutsNumpy(xy[p[i]], xy[q[i]], a[e], b[e])]] = True
      out[p[~cut]] = out[q[~cut]]
      nNew += int((~cut).sum())
    if nNew == 0:
      break


# whether the segments p1-p2 and a-b meet, touching included; a segment on the line of the other
# cuts it only where they overlap
def _cutsNumpy(p1, p2, a, b):
  cr = lambda u, v: u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
  o1, o2 = cr(b - a, p1 - a), cr(b - a, p2 - a)
  o3, o4 = cr(p2 - p1, a - p1), cr(p2 - p1, b - p1)
  t = np.column_stack([np.sum((a - p1) * (p2 - p1), axis=1), np.sum((b - p1) * (p2 - p1), axis=1)])
  apart = (o1 == 0) & (o2 == 0) & ((t.max(axis=1) < 0) | (t.min(axis=1) > np.sum((p2 - p1) ** 2, axis=1)))
  return (o1 * o2 <= 0) & (o3 * o4 <= 0) & ~apart


def grainTable(seg, ebsd, boundary=None, innerBoundary=None, backend=None):
  """the grain table: phase, size, mean orientation and orientation spread

  The mean is the quaternion mean of the pixels of a grain, each first brought to the
  equivalent closest to the first one so that the symmetry does not tear the average
  apart; the spread is the mean angle to that mean. Both run as compiled passes over all
  the pixels, the NumPy form phase by phase their fallback and oracle."""
  n = seg.nGrains
  gid = seg.grainId
  ph = ebsd.phaseId.reshape(-1)
  rot = ebsd.rotations.data.reshape(-1, 4)
  numPixel = np.bincount(gid, minlength=n + 1)[1:].astype(np.int32)
  phaseId, meanRot, gos = _grainTable(gid, ph, rot, n, numPixel, ebsd.CSList, backend)
  return grain2d(np.arange(1, n + 1), phaseId, ebsd.CSList, numPixel, meanRot, gos, ebsd.frame, boundary=boundary, innerBoundary=innerBoundary, phaseMap=ebsd.phaseMap)


# the phase, the mean rotation and the spread of every grain of the pixels' grain ids, numbered
# from 1: the compiled passes, or their NumPy form without them
def _grainTable(gid, ph, rot, n, numPixel, CSList, backend=None):
  table = _grainTableNumba(gid, ph, rot, n, numPixel, CSList, backend)
  return table if table is not None else _grainTableNumpy(gid, ph, rot, n, numPixel, CSList)


# the rows scaled to unit length, a zero row kept
def _normalizeRows(m):
  norm = np.linalg.norm(m, axis=1, keepdims=True)
  return np.where(norm > 0, m / np.where(norm > 0, norm, 1), m)


# the grain table in compiled passes over all pixels at once: the images at the first pixel
# of every grain, their sums per grain in the order of the pixels, the normalised mean, and
# the chord angles to it; None without the compiled loops
def _grainTableNumba(gid, ph, rot, n, numPixel, CSList, backend=None):
  from .criteria import phaseGroups
  loops = [grainKernel(k, backend) for k in ("firstPixel", "toReference", "sumByGrain")]
  if any(f is None for f in loops):
    return None
  firstPixel, toReference, sumByGrain = loops
  T = rot.dtype
  gid64, ph64 = np.ascontiguousarray(gid, dtype=np.int64), np.ascontiguousarray(ph, dtype=np.int64)
  first = np.full(n + 1, -1, dtype=np.int64)
  firstPixel(gid64, first)
  phaseId = ph[first[1:]].astype(np.int8)
  Gall, nG, indexed = phaseGroups(CSList)
  r64 = np.ascontiguousarray(rot, dtype=np.float64)
  refs = r64[np.maximum(first, 0)]
  q = np.zeros((len(gid), 4))
  dev = np.zeros(len(gid))
  toReference(r64, ph64, gid64, refs, Gall, nG, indexed, False, q, dev)
  acc = np.zeros((n + 1, 4), dtype=T)
  sumByGrain(gid64, ph64, indexed, q.astype(T, copy=False), acc)
  m = _normalizeRows(acc[1:])
  hasMean = indexed[phaseId]
  meanRot = np.full((n, 4), np.nan, dtype=T)
  meanRot[hasMean] = m[hasMean]
  refs = np.vstack([np.zeros((1, 4)), m.astype(np.float64)])
  toReference(r64, ph64, gid64, refs, Gall, nG, indexed, True, q, dev)
  spread = np.zeros((n + 1, 1), dtype=T)
  sumByGrain(gid64, ph64, indexed, dev.astype(T, copy=False).reshape(-1, 1), spread)
  gos = np.zeros(n, dtype=T)
  gos[hasMean] = spread[1:, 0][hasMean] / numPixel[hasMean]
  return phaseId, meanRot, gos


def _grainTableNumpy(gid, ph, rot, n, numPixel, CSList):
  T = rot.dtype
  first = np.full(n + 1, -1, dtype=np.int64)
  order = np.argsort(gid, kind="stable")
  ids, at = np.unique(gid[order], return_index=True)
  first[ids] = order[at]
  phaseId = ph[first[1:]].astype(np.int8)
  meanRot = np.full((n, 4), np.nan, dtype=T)
  gos = np.zeros(n, dtype=T)
  L = np.array([[1.0, 0, 0, 0]], dtype=T)
  for p, fr in enumerate(CSList):
    pix = np.flatnonzero((ph == p) & (gid > 0))
    if not fr.isIndexed or len(pix) == 0:
      continue
    G = fr.sym.properGroup().rotations().data.reshape(-1, 4).astype(T)
    ref = rot[first[gid[pix]]]
    q = closestImage(ref, rot[pix], L, G, False)[1]
    q = np.where(np.sum(q * ref, axis=1)[:, None] < 0, -q, q)
    acc = np.zeros((n + 1, 4), dtype=T)
    np.add.at(acc, gid[pix], q)
    m = _normalizeRows(acc[1:])
    grainsOfPhase = np.unique(gid[pix])
    meanRot[grainsOfPhase - 1] = m[grainsOfPhase - 1]
    dev = chordAngle(m[gid[pix] - 1], closestImage(m[gid[pix] - 1], rot[pix], L, G, False)[1])
    spread = np.zeros(n + 1, dtype=T)
    np.add.at(spread, gid[pix], dev)
    gos[grainsOfPhase - 1] = spread[grainsOfPhase] / numPixel[grainsOfPhase - 1]
  return phaseId, meanRot, gos


__all__ = ["calcGrains", "notIndexedKinds"]
