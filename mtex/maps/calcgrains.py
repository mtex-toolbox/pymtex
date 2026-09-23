# grain reconstruction (MATLAB EBSD/calcGrains, Julia calcgrains.jl)
#
#   grains = calcGrains(ebsd, angle=10 * degree)
#   grains = calcGrains(ebsd, angle=[10 * degree, 2 * degree], minPixel=5, alpha=3.1)
#   grains = calcGrains(ebsd, criterion=gbcSoft(10 * degree))
#
# The map is written into: every measurement gets its grain id in ebsd.grainId, and the
# cull marks the pixels of the grains it drops as not indexed, so the map that comes back
# is the one the grains were built from. Keep the original with calcGrains(ebsd.copy()).
# The pipeline is the decomposition (M2), the criterion on its faces and the components
# of what it joins (M3), and the grain table over the components.
from __future__ import annotations

import numpy as np

from ..geometry.symreduce import chordAngle, closestImage
from ..geometry.units import degree
from .boundary import buildBoundary
from .criteria import gbcAngle, grainBoundaryCriterion
from .decomposition import NO_PIXEL, decompose
from .grains import grain2d
from .quadruple import splitQuadruplePoints
from .segment import segment


def calcGrains(ebsd, angle=10 * degree, threshold=None, criterion=None, minPixel=1, alpha=3.1, removeQuadruplePoints=False,
               backend=None):
  if isinstance(angle, grainBoundaryCriterion):
    criterion, angle = angle, None
  crit = criterion if criterion is not None else gbcAngle(angle if threshold is None else threshold)
  releaseAbsorbed(ebsd)
  # the cull changes the map the closing runs on, so it is a second pass over the whole
  # reconstruction and not a filter on the grains that came out
  for pas in (1, 2):
    c = decompose(ebsd, alpha, backend=backend)
    seg = segment(c, ebsd, crit, backend)
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
  ebsd.grainId = absorbInteriorPixels(c, ebsd, seg.grainId).reshape(ebsd.shape)
  absorbNotIndexed(ebsd, grains)
  return grains


# a not indexed pixel inside a grain takes the phase of the grain and no rotation, as
# MATLAB's [grains, ebsd] = calcGrains returns the map; the padding of a grid stays empty
def absorbNotIndexed(ebsd, grains):
  gid = ebsd.grainId.reshape(-1)
  absorbed = ~ebsd.isIndexed.reshape(-1) & (gid > 0) & ~ebsd.isPadding.reshape(-1)
  if not absorbed.any():
    return
  ebsd.phaseId.reshape(-1)[absorbed] = np.asarray(grains.phaseId).reshape(-1)[gid[absorbed] - 1]
  ebsd.rotations.data.reshape(-1, 4)[absorbed] = np.nan


# a pixel a previous reconstruction absorbed has a phase but no rotation; it is segmented
# as the not indexed pixel it was, so that calcGrains on its own result gives the same grains
def releaseAbsorbed(ebsd):
  blank = ebsd.isIndexed.reshape(-1) & np.isnan(ebsd.rotations.data.reshape(-1, 4)[:, 0])
  ebsd.phaseId.reshape(-1)[blank] = 0


# the pixels of indexed grains that came out below minPixel
def undersized(seg, ebsd, minPixel):
  n = np.bincount(seg.grainId, minlength=seg.nGrains + 1)
  return (seg.grainId > 0) & ebsd.isIndexed.reshape(-1) & (n[seg.grainId] < minPixel)


# the pixels of the grains the cull drops, marked as not indexed where they lie
def markNotIndexed(ebsd, mask):
  ebsd.phaseId.reshape(-1)[mask] = 0


# give a grain to the pixels the closing left without a cell of their own
#
# A filled hole belongs to the cells that grew into it, so a pixel there lies in one grain,
# in several, or on the boundary between them. Which cell covers a point is which site is
# nearest to it, so the footprint of the pixel is sampled at its corners and the pixel
# joins the grain only when every corner lands in it. Zero stays for a pixel a boundary
# runs through.
def absorbInteriorPixels(c, ebsd, grainId, nudge=0.95):
  out = grainId.copy()
  l = ebsd.lattice
  if l is None:
    return out
  corners = c.unitCell2d()
  P = l.plane.to2d(ebsd.pos.data.reshape(-1, 3))
  siteGrain = np.zeros(c.nSites, dtype=np.int32)
  has = c.site2pixel >= 0
  siteGrain[has] = grainId[c.site2pixel[has]]
  for k in np.flatnonzero(out == 0):
    g = 0
    ok = True
    for v in corners:
      s = c.siteAt(P[k] + nudge * v, l.ij[k])
      gs = 0 if s == NO_PIXEL else siteGrain[s]
      if gs == 0 or (g != 0 and gs != g):
        ok = False
        break
      g = gs
    if ok:
      out[k] = g
  return out


# the grain table: phase, size, mean orientation and orientation spread
#
# The mean is the quaternion mean of the pixels of a grain, each first brought to the
# equivalent closest to the first one so that the symmetry does not tear the average
# apart; the spread is the mean angle to that mean. Both run vectorised over all the
# pixels of a phase at once, with the reference of every pixel that of its grain.
def grainTable(seg, ebsd, boundary=None, innerBoundary=None):
  n = seg.nGrains
  gid = seg.grainId
  ph = ebsd.phaseId.reshape(-1)
  rot = ebsd.rotations.data.reshape(-1, 4)
  T = rot.dtype
  numPixel = np.bincount(gid, minlength=n + 1)[1:].astype(np.int32)
  phaseId = np.zeros(n, dtype=np.int8)
  first = np.full(n + 1, -1, dtype=np.int64)
  order = np.argsort(gid, kind="stable")
  firstOf = np.unique(gid[order], return_index=True)
  first[firstOf[0]] = order[firstOf[1]]
  phaseId[:] = ph[first[1:]]
  meanRot = np.full((n, 4), np.nan, dtype=T)
  gos = np.zeros(n, dtype=T)
  for p in range(len(ebsd.CSList)):
    fr = ebsd.CSList[p]
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
  return grain2d(np.arange(1, n + 1), phaseId, ebsd.CSList, numPixel, meanRot, gos, ebsd.frame, boundary=boundary, innerBoundary=innerBoundary, phaseMap=ebsd.phaseMap)


__all__ = ["calcGrains"]
