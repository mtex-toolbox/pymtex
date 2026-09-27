# grain reconstruction (MATLAB EBSD/calcGrains, Julia calcgrains.jl)
#
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

from ..geometry.kernels import backend as kernelBackend
from ..geometry.symreduce import chordAngle, closestImage
from ..geometry.units import degree
from .boundary import buildBoundary
from .criteria import gbcAngle, grainBoundaryCriterion
from .decomposition import NO_PIXEL, decompose
from .fmc import gbcFMC
from .grains import grain2d
from .quadruple import splitQuadruplePoints
from .segment import segment


def calcGrains(ebsd, angle=10 * degree, threshold=None, criterion=None, minPixel=1, alpha=3.1, removeQuadruplePoints=False,
               mcl=None, fmc=None, verbose=False, backend=None):
  """the grains of a map, the pixels joined across every boundary the criterion keeps: a
  misorientation angle below angle (or threshold), or a grainBoundaryCriterion; minPixel
  culls smaller grains, alpha bounds the decomposition, mcl and fmc select Markov
  clustering and fast multiscale clustering. The map keeps the grain ids"""
  if isinstance(angle, grainBoundaryCriterion):
    criterion, angle = angle, None
  if criterion is None and fmc is not None:
    criterion = gbcFMC(fmc, verbose=verbose)
  crit = criterion if criterion is not None else gbcAngle(angle if threshold is None else threshold)
  # a criterion that absorbs the small regions itself takes minPixel instead of the cull
  if crit.handlesMinPixel:
    crit = crit.setMinPixel(minPixel) if minPixel > 1 else crit
    minPixel = 1
  releaseAbsorbed(ebsd)
  # the cull changes the map the closing runs on, so it is a second pass over the whole
  # reconstruction and not a filter on the grains that came out
  for pas in (1, 2):
    c = decompose(ebsd, alpha, backend=backend)
    seg = segment(c, ebsd, crit, backend, mcl)
    if minPixel <= 1 or pas == 2:
      break
    under = undersized(seg, ebsd, minPixel)
    if not under.any():
      break
    markNotIndexed(ebsd, under)
  # the diagonal ambiguity is decided on the map the grains are built from, so the split
  # runs after the cull and before anything reads the boundary
  if removeQuadruplePoints:
    c, seg = splitQuadruplePoints(c, seg, ebsd, crit)
  grains = grainTable(seg, ebsd, buildBoundary(c, seg, ebsd), buildBoundary(c, seg, ebsd, inner=True))
  ebsd.grainId = absorbInteriorPixels(c, ebsd, seg.grainId, backend=backend).reshape(ebsd.shape)
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


def undersized(seg, ebsd, minPixel):
  """the pixels of indexed grains that came out below minPixel"""
  n = np.bincount(seg.grainId, minlength=seg.nGrains + 1)
  return (seg.grainId > 0) & ebsd.isIndexed.reshape(-1) & (n[seg.grainId] < minPixel)


def markNotIndexed(ebsd, mask):
  """the pixels of the grains the cull drops, marked as not indexed where they lie"""
  ebsd.phaseId.reshape(-1)[mask] = 0


def absorbInteriorPixels(c, ebsd, grainId, nudge=0.95, backend=None, reach=5):
  """give a grain to the pixels the closing left without a cell of their own

  A filled hole belongs to the cells that grew into it, so a pixel there lies in one grain,
  in several, or on the boundary between them. Which cell covers a point is which site is
  nearest to it, so the footprint of the pixel is sampled at its corners and the pixel
  joins the grain only when every corner lands in it. Zero stays for a pixel a boundary
  runs through."""
  out = grainId.copy()
  l = ebsd.lattice
  if l is None:
    return out
  corners = c.unitCell2d()
  P = l.plane.to2d(ebsd.pos.data.reshape(-1, 3))
  siteGrain = np.zeros(c.nSites, dtype=np.int32)
  has = c.site2pixel >= 0
  siteGrain[has] = grainId[c.site2pixel[has]]
  todo = np.flatnonzero(out == 0)
  # the corners of all pixels in one compiled loop, the Python loop below its fallback
  if (backend or kernelBackend()) == "numba":
    from .kernels import pixelKernel
    loop = pixelKernel("absorb")
    if loop is not None:
      s = c.sites
      loop(todo.astype(np.int64), np.ascontiguousarray(P[:, :2], dtype=float), np.ascontiguousarray(l.ij, dtype=np.int64),
           np.ascontiguousarray(nudge * np.asarray(corners, dtype=float)[:, :2]), np.ascontiguousarray(s.lookup, dtype=np.int64),
           int(s.offset[0]), int(s.offset[1]), np.ascontiguousarray(s.pos[:, :2], dtype=float), siteGrain, reach, out)
      return out
  for k in todo:
    g = 0
    ok = True
    for v in corners:
      s = c.siteAt(P[k] + nudge * v, l.ij[k], reach)
      gs = 0 if s == NO_PIXEL else siteGrain[s]
      if gs == 0 or (g != 0 and gs != g):
        ok = False
        break
      g = gs
    if ok:
      out[k] = g
  return out


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
  table = _grainTableNumba(gid, ph, rot, n, numPixel, ebsd.CSList) if (backend or kernelBackend()) == "numba" else None
  phaseId, meanRot, gos = table if table is not None else _grainTableNumpy(gid, ph, rot, n, numPixel, ebsd.CSList)
  return grain2d(np.arange(1, n + 1), phaseId, ebsd.CSList, numPixel, meanRot, gos, ebsd.frame, boundary=boundary, innerBoundary=innerBoundary, phaseMap=ebsd.phaseMap)


# the grain table in compiled passes over all pixels at once: the images at the first pixel
# of every grain, their sums per grain in the order of the pixels, the normalised mean, and
# the chord angles to it; None without numba
def _grainTableNumba(gid, ph, rot, n, numPixel, CSList):
  from .criteria import phaseGroups
  from .kernels import grainKernel
  loops = [grainKernel(k) for k in ("firstPixel", "toReference", "sumByGrain")]
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
  m = acc[1:]
  norm = np.linalg.norm(m, axis=1, keepdims=True)
  m = np.where(norm > 0, m / np.where(norm > 0, norm, 1), m)
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
  phaseId = np.zeros(n, dtype=np.int8)
  first = np.full(n + 1, -1, dtype=np.int64)
  order = np.argsort(gid, kind="stable")
  firstOf = np.unique(gid[order], return_index=True)
  first[firstOf[0]] = order[firstOf[1]]
  phaseId[:] = ph[first[1:]]
  meanRot = np.full((n, 4), np.nan, dtype=T)
  gos = np.zeros(n, dtype=T)
  for p in range(len(CSList)):
    fr = CSList[p]
    if not fr.isIndexed:
      continue
    pix = np.flatnonzero((ph == p) & (gid > 0))
    if len(pix) == 0:
      continue
    G = fr.sym.properGroup().rotations().data.reshape(-1, 4).astype(T)
    L = np.array([[1.0, 0, 0, 0]], dtype=T)
    ref = rot[first[gid[pix]]]
    q = closestImage(ref, rot[pix], L, G, False)[1]
    q = np.where(np.sum(q * ref, axis=1)[:, None] < 0, -q, q)
    acc = np.zeros((n + 1, 4), dtype=T)
    np.add.at(acc, gid[pix], q)
    m = acc[1:]
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    m = np.where(norm > 0, m / np.where(norm > 0, norm, 1), m)
    grainsOfPhase = np.unique(gid[pix])
    meanRot[grainsOfPhase - 1] = m[grainsOfPhase - 1]
    dev = chordAngle(m[gid[pix] - 1], closestImage(m[gid[pix] - 1], rot[pix], L, G, False)[1])
    spread = np.zeros(n + 1, dtype=T)
    np.add.at(spread, gid[pix], dev)
    gos[grainsOfPhase - 1] = spread[grainsOfPhase] / numPixel[grainsOfPhase - 1]
  return phaseId, meanRot, gos


__all__ = ["calcGrains"]
