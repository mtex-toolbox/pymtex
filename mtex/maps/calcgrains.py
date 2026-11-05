# grain reconstruction (MATLAB EBSD/calcGrains, Julia calcgrains.jl)
#
#   grains = calcGrains(ebsd)                       15 degrees, as MATLAB
#   grains = calcGrains(ebsd, angle=10 * degree)
#   grains = calcGrains(ebsd, angle=[10 * degree, 2 * degree], minPixel=5, alpha=3.1)
#   grains = calcGrains(ebsd, criterion=gbcSoft(10 * degree))
#   grains = calcGrains(ebsd, fmc=3, minPixel=10, verbose=True)     fast multiscale clustering, gbcFMC
#
# The map is written into: every measurement gets its grain id in ebsd.grainId, and the
# cull marks the pixels of the grains it drops as not indexed, so the map that comes back
# is the one the grains were built from. Keep the original with calcGrains(ebsd.copy()).
# The pipeline is the decomposition (M2), the criterion on its faces and the components
# of what it joins (M3), and the grain table over the components.
from __future__ import annotations

import numpy as np

from ..geometry.symreduce import chordAngle, closestImage
from .boundary import buildBoundary
from .criteria import DEFAULT_ANGLE, gbcAngle, gbcEqual, grainBoundaryCriterion
from .decomposition import decompose, splitPinches
from .criteria.fmc import gbcFMC
from .grains import grain2d
from .kernels import grainKernel, pixelKernel
from .quadruple import splitQuadruplePoints
from .segment import segment
from .straighten import straightenInFill


def calcGrains(ebsd, angle=DEFAULT_ANGLE, threshold=None, criterion=None, minPixel=1, alpha=3.1, removeQuadruplePoints=False,
               mcl=None, fmc=None, verbose=False, backend=None, grainId=False, straighten=True):
  """the grains of a map, the pixels joined across every boundary the criterion keeps: a
  misorientation angle below angle (or threshold), or a grainBoundaryCriterion; minPixel
  culls smaller grains, alpha bounds the decomposition, straighten draws the boundaries through
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
  crit = criterion if criterion is not None else gbcAngle(angle if threshold is None else threshold)
  # a criterion that absorbs the small regions itself takes minPixel instead of the cull
  if crit.handlesMinPixel:
    crit = crit.setMinPixel(minPixel) if minPixel > 1 else crit
    minPixel = 1
  releaseAbsorbed(ebsd)
  c = decompose(ebsd, alpha, backend=backend)
  # the cull changes the map the closing runs on, so it is a pass over the whole reconstruction;
  # the first sizes the grains before the pinches are split, as MATLAB's minPixelMask, and the
  # decomposition repeats until no grain re-fragments below minPixel
  if minPixel > 1 and markUndersized(ebsd, segment(c, ebsd, crit, backend, mcl), minPixel):
    c = decompose(ebsd, alpha, backend=backend)
  while True:
    c = splitPinches(c)
    seg = segment(c, ebsd, crit, backend, mcl)
    if minPixel <= 1 or not markUndersized(ebsd, seg, minPixel):
      break
    c = decompose(ebsd, alpha, backend=backend)
  if straighten:
    c, seg = straightenInFill(c, seg)
  # the diagonal ambiguity is decided on the map the grains are built from, so the split
  # runs after the cull and before anything reads the boundary
  if removeQuadruplePoints:
    c, seg = splitQuadruplePoints(c, seg, ebsd, crit)
  grains = grainTable(seg, ebsd, buildBoundary(c, seg, ebsd), buildBoundary(c, seg, ebsd, inner=True))
  ebsd.grainId = absorbInteriorPixels(c, ebsd, seg.grainId, grains.boundary, backend=backend, reach=int(np.ceil(alpha)) + 2).reshape(ebsd.shape)
  absorbNotIndexed(ebsd, grains)
  return grains


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


def absorbInteriorPixels(c, ebsd, grainId, boundary, nudge=0.95, backend=None, reach=5):
  """give a grain to the pixels the closing left without a cell of their own

  A filled hole belongs to the cells that grew into it, so a pixel there lies in one grain,
  in several, or on the boundary between them. The candidates are the grains of the sites
  nearest to the pixel, and the pixel joins the one whose polygon holds every corner of its
  footprint, nudged inwards, off its boundary. Zero stays for a pixel a boundary runs through."""
  out = grainId.copy()
  l = ebsd.lattice
  if l is None:
    return out
  corners = np.ascontiguousarray(nudge * np.asarray(c.unitCell2d(), dtype=float)[:, :2])
  P = np.ascontiguousarray(l.plane.to2d(ebsd.pos.data.reshape(-1, 3))[:, :2], dtype=float)
  siteGrain = np.zeros(c.nSites, dtype=np.int32)
  has = c.site2pixel >= 0
  siteGrain[has] = grainId[c.site2pixel[has]]
  todo = np.flatnonzero((out == 0) & ~ebsd.isPadding.reshape(-1))
  # the segments of every grain, each listed for both its sides
  g = boundary.grainId.reshape(-1)
  k = np.repeat(np.arange(len(boundary.F)), 2)
  order = np.argsort(g, kind="stable")
  order = order[g[order] > 0]
  V = np.asarray(boundary.V, dtype=float)[:, :2]
  seg = np.ascontiguousarray(np.column_stack([V[boundary.F[k[order], 0]], V[boundary.F[k[order], 1]]]))
  segStart = np.searchsorted(g[order], np.arange(int(max(grainId.max(initial=0), g.max(initial=0))) + 2)).astype(np.int64)
  s = c.sites
  args = (todo.astype(np.int64), P, np.ascontiguousarray(l.ij, dtype=np.int64), corners, np.ascontiguousarray(s.lookup, dtype=np.int64),
          int(s.offset[0]), int(s.offset[1]), np.ascontiguousarray(s.pos[:, :2], dtype=float), siteGrain, reach, segStart, seg,
          1e-9 * c.dxy)
  # all pixels in one compiled loop, which tests only the pixels near a boundary against the
  # polygons; its Python form below as fallback
  loop = pixelKernel("absorb", backend)
  if loop is None:
    _absorbPython(*args, out)
    return out
  # the cells a boundary segment touches, in lattice indices of the ideal lattice; a scale per unit
  # length that bounds the index distance, widened by how far a pixel lies from its lattice point
  Minv = np.linalg.inv(np.asarray(l.A, dtype=float).T)
  p0 = np.asarray(l.origin, dtype=float)
  scale = float(np.linalg.norm(Minv))
  dev = float(np.abs((P - p0) @ Minv - l.ij).max(initial=0))
  mark = np.zeros(s.lookup.shape, dtype=bool)
  pixelKernel("mark", backend)(seg, p0, Minv, int(s.offset[0]), int(s.offset[1]), mark)
  loop(*args, mark, scale, float(np.linalg.norm(corners, axis=1).max()) + dev / max(scale, 1e-300), out)
  return out


def _absorbPython(todo, P, ij, corners, lookup, off0, off1, sitePos, siteGrain, reach, segStart, seg, eps, out):
  for k in todo:
    a0, b0 = ij[k, 0] - off0, ij[k, 1] - off1
    block = lookup[max(0, a0 - reach):a0 + reach + 1, max(0, b0 - reach):b0 + reach + 1].reshape(-1)
    cand = block[block >= 0]
    cand = cand[siteGrain[cand] > 0]
    if len(cand) == 0:
      continue
    d = np.sum((sitePos[cand] - P[k]) ** 2, axis=1)
    for g in siteGrain[cand[d <= d.min() * (1 + 1e-9)]]:
      S = seg[segStart[g]:segStart[g + 1]]
      q = P[k] + corners
      cross = ((S[:, 1] > q[:, 1, None]) != (S[:, 3] > q[:, 1, None])) & \
        (q[:, 0, None] < S[:, 0] + (q[:, 1, None] - S[:, 1]) * (S[:, 2] - S[:, 0]) / np.where(S[:, 3] == S[:, 1], 1, S[:, 3] - S[:, 1]))
      d, t = S[:, 2:] - S[:, :2], q[:, None, :] - S[:, :2]
      u = np.clip(np.sum(t * d, axis=2) / np.maximum(np.sum(d * d, axis=1), 1e-300), 0, 1)
      near = np.linalg.norm(t - u[..., None] * d, axis=2).min(axis=1) <= eps
      if np.all((cross.sum(axis=1) % 2 == 1) & ~near):
        out[k] = g
        break


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


__all__ = ["calcGrains"]
