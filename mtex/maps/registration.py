# how far each part of one image moved against another (MATLAB tools/registration_tools:
# xcfShift, @pairShifts)
#
#   u, peak, pos = xcfShift(A, B)                               two arrays, in pixels
#   u, peak, pos = xcfShift(mgA, mgB, ROISize=64, numROI=(24, 18))  two mapImage, in specimen units
#   T = spatialTransformShift.fit(pos, pos + u, weights=peak)
#   ps = pairShifts(pos, u, peak, roiSize); ps.meanShift(px)
#
# The region both images are finite in is cut into a grid of tiles, and every tile is phase
# correlated against the same tile of the other image: zero mean and unit deviation, a cosine
# window, a band pass, the peak of the correlation on a half pixel grid and then refined by the
# matrix multiply DFT of Guizar-Sicairos et al. (Optics Letters 33, 2008), in two passes,
# first to 1/48 pixel and then to 1/XCFMesh. The feature at pos in A is at pos + u in B, and
# the peak height is the weight of the tile in a fit. A tile with a non finite value, or a
# flat one, is NaN in all three outputs. The tile arithmetic is MATLAB's, counted from one,
# so that the tiles are MATLAB's tile for tile.
from __future__ import annotations

import numpy as np

from ..geometry.vector3d import vector3d
from .mapimage import mapImage


# MATLAB's round, half away from zero
def _round(x):
  return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _fft():
  import scipy.fft
  from ..geometry.kernels import threads
  return scipy.fft, threads()


def xcfROIGrid(roiSize, numROI, region):
  """the centres of a grid of tiles inside a region [left top width height], counted from one,
  the outermost half a tile in from its edge; a tile that rounding pushes over it is dropped"""
  h = roiSize / 2
  if region[2] < roiSize or region[3] < roiSize:
    raise ValueError(f"a tile of {roiSize} pixels does not fit inside the {region[2]:g} x {region[3]:g} region the two "
                     "images share; use a smaller ROISize, or check that they overlap")
  x = _round(np.linspace(np.ceil(h + region[0]), np.floor(region[0] + region[2] - h), int(numROI[0])))
  y = _round(np.linspace(np.ceil(h + region[1]), np.floor(region[1] + region[3] - h), int(numROI[1])))
  while len(x) and x[-1] + h > region[0] + region[2]:
    x = x[:-1]
  while len(y) and y[-1] + h > region[1] + region[3]:
    y = y[:-1]
  if not len(x) or not len(y):
    raise ValueError(f"no tile of {roiSize} pixels fits inside the {region[2]:g} x {region[3]:g} region the two images share")
  cx, cy = np.meshgrid(x, y)
  return cx.astype(np.int64), cy.astype(np.int64)


def xcfFilters(roiSize, fPass):
  """the band pass, in FFT order, and the raised cosine window of a tile; fPass is
  [highCutOff highWidth lowCutOff lowWidth]"""
  lCut, lWidth = fPass[2], fPass[3] / 2
  hCut, hWidth = fPass[0], fPass[1] / 2
  if lCut < hCut:
    raise ValueError(f"the low pass cut off {lCut:g} is below the high pass cut off {hCut:g}")
  u, v = np.meshgrid(np.arange(1, roiSize + 1), np.arange(1, roiSize + 1), indexing="ij")
  v = v - roiSize / 2 - 0.5
  u = u - roiSize / 2 - 0.5
  window = np.cos(np.pi * u / roiSize) * np.cos(np.pi * v / roiSize)
  d = np.hypot(u, v)
  with np.errstate(divide="ignore", invalid="ignore"):
    low = np.exp(-((d - lCut) / (np.sqrt(2) * lWidth / 2)) ** 2)
    high = np.exp(-((hCut - d) / (np.sqrt(2) * hWidth / 2)) ** 2)
  low[d > lCut + 2 * lWidth] = 0
  low[d < lCut] = 1
  high[d < hCut - 2 * hWidth] = 0
  high[d > hCut] = 1
  return np.fft.fftshift(np.nan_to_num(high * low)), window


# the part of the upsampled DFT no tile changes: the transform sampled at
# (u - fix(w/2)) / mesh pixels, u = 0 .. w-1
def _dftKernels(f, w, mesh, rs):
  u = np.arange(w) - w // 2
  pf = -2j * np.pi / (rs * mesh)
  return np.exp(pf * np.outer(u, f)), np.exp(pf * np.outer(f, u))


# the peak on a grid of spacing 1/mesh centred on the estimate, the phase ramp of each tile
# applied to its spectrum rather than to a pair of kernels of its own
def _refine(kern, f, rowShift, colShift, Kr, Kc, mesh, rs):
  w = Kr.shape[0]
  rowShift = _round(rowShift * mesh) / mesh
  colShift = _round(colShift * mesh) / mesh
  pr = np.exp((-2j * np.pi / rs) * (f[None, :, None] * rowShift[:, None, None]))
  pc = np.exp((-2j * np.pi / rs) * (f[None, None, :] * colShift[:, None, None]))
  # Kr X_t Kc for every tile t as two products over all tiles at once
  T, nb = kern.shape[0], kern.shape[1]
  X = (kern * pr * pc).transpose(1, 0, 2).reshape(nb, T * nb)
  Y = (Kr @ X).reshape(w, T, nb).transpose(1, 0, 2).reshape(T * w, nb)
  CC = (Y @ Kc).reshape(T, w, w)
  # MATLAB's max of complex numbers is by their modulus, the first in column order
  a = np.abs(CC).transpose(0, 2, 1).reshape(T, -1)
  k = np.argmax(a, axis=1)
  peak = a[np.arange(T), k]
  d = w // 2
  return rowShift + (k % w - d) / mesh, colShift + (k // w - d) / mesh, peak


# zero mean, unit deviation, window, transform, band pass; a tile with a non finite value or
# none of either is bad, and its spectrum zero
def _tileSpectrum(tile, window, band, bandFilt, fft, workers):
  mu = tile.mean(axis=(1, 2), keepdims=True)
  sd = tile.std(axis=(1, 2), ddof=1, keepdims=True)
  with np.errstate(invalid="ignore", divide="ignore"):
    tile = (tile - mu) / sd
  bad = ~np.isfinite(sd[:, 0, 0]) | (sd[:, 0, 0] == 0) | ~np.isfinite(tile).all(axis=(1, 2))
  tile = np.where(bad[:, None, None], 0.0, tile)
  spec = fft.fft2(tile * window, workers=workers)
  spec = bandFilt * np.take(np.take(spec, band, axis=1), band, axis=2)
  spec[bad] = 0
  return spec, bad


def xcfCorrelate(A, B, cx, cy, roiSize, fPass, mesh, window, bandPass):
  """the sub pixel shift of the tiles about (cx, cy) of B against A, in pixels, with the height
  of their peaks; batched, the coarse correlation held near 128 MB"""
  fft, workers = _fft()
  coarse = min(48, mesh)
  rs = int(roiSize)
  h = rs // 2
  nR = cx.size
  batch = max(1, (1 << 21) // max(rs * rs, 1))
  nKeep = int(fPass[2] + fPass[3])
  band = np.concatenate([np.arange(nKeep), np.arange(rs - nKeep, rs)])
  bandFilt = bandPass[np.ix_(band, band)]
  fIdx = np.fft.ifftshift(np.arange(rs)) - rs // 2
  f = fIdx[band].astype(float)
  nb = len(band)
  twoPass = coarse < mesh
  KrC, KcC = _dftKernels(f, int(np.ceil(coarse * 1.5)), coarse, rs)
  if twoPass:
    KrF, KcF = _dftKernels(f, 2 * int(np.ceil(mesh / coarse)) + 1, mesh, rs)
  cenX, cenY = cx.reshape(-1, order="F"), cy.reshape(-1, order="F")
  dx, dy, peak = np.zeros(nR), np.zeros(nR), np.zeros(nR)
  off = np.arange(-h, h)
  # where the band lands once it is shifted to the centre of a 2 rs array and back
  lo = rs - nb // 2
  at = (lo + (np.arange(nb) + nb // 2) % nb + rs) % (2 * rs)
  for b0 in range(0, nR, batch):
    b = np.arange(b0, min(b0 + batch, nR))
    rows = cenY[b][:, None] + off[None, :] - 1
    cols = cenX[b][:, None] + off[None, :] - 1
    tA = A[rows[:, :, None], cols[:, None, :]]
    tB = B[rows[:, :, None], cols[:, None, :]]
    specA, badA = _tileSpectrum(tA, window, band, bandFilt, fft, workers)
    specB, badB = _tileSpectrum(tB, window, band, bandFilt, fft, workers)
    bad = badA | badB
    # the kept band embedded in a 2 rs array, so that the inverse lands the peak on a half
    # pixel grid over the whole range
    CC = np.zeros((len(b), 2 * rs, 2 * rs), dtype=complex)
    CC[:, at[:, None], at[None, :]] = specA * np.conj(specB)
    CC = fft.ifft2(CC, workers=workers)
    pk = np.argmax(np.abs(CC).transpose(0, 2, 1).reshape(len(b), -1), axis=1)
    rloc, cloc = pk % (2 * rs) + 1, pk // (2 * rs) + 1
    rowShift = np.where(rloc > rs, rloc - 2 * rs - 1, rloc - 1) / 2
    colShift = np.where(cloc > rs, cloc - 2 * rs - 1, cloc - 1) / 2
    kern = specB * np.conj(specA)
    rowShift, colShift, pkH = _refine(kern, f, rowShift, colShift, KrC, KcC, coarse, rs)
    if twoPass:
      rowShift, colShift, pkH = _refine(kern, f, rowShift, colShift, KrF, KcF, mesh, rs)
    rowShift[bad], colShift[bad], pkH[bad] = np.nan, np.nan, np.nan
    dy[b], dx[b], peak[b] = rowShift, colShift, pkH
  return dx, dy, peak


def xcfShift(A, B, ROISize=None, numROI=24, XCFMesh=250):
  """u, peak, pos = xcfShift(A, B, ROISize=64, numROI=(24, 18), XCFMesh=250)

  The displacement from A to B of every tile of a grid over the region both images are finite
  in, to sub pixel accuracy, with the height of its correlation peak, the weight of the tile in
  a fit, and its centre in A. Two mapImage on one grid answer in specimen units, two arrays in
  pixels, pixel (i, j) at (j, i). ROISize is the tile width in pixels, by default
  2^ceil(log2(rows/4)); numROI the tiles across, a scalar scaled by the aspect ratio for those
  down; XCFMesh the upsampling of the peak. The outputs run over the tiles down the columns of
  the grid first, as MATLAB's.
  """
  if not isinstance(A, mapImage):
    A = mapImage(A)
  if not isinstance(B, mapImage):
    B = mapImage(B)
  if A.gridSize() != B.gridSize():
    raise ValueError(f"the two images have to be on one grid before they can be correlated; they are {A.gridSize()} and {B.gridSize()}")
  mg = A
  a = A.img if A.img.ndim == 2 else A.img.mean(axis=2)
  b = B.img if B.img.ndim == 2 else B.img.mean(axis=2)
  nRows, nCols = a.shape
  roiSize = 2 ** np.ceil(np.log2(nRows / 4)) if ROISize is None else ROISize
  roiSize = int(2 * _round(roiSize / 2))
  if roiSize < 8:
    raise ValueError(f"a correlation tile of {roiSize} pixels is too small to filter and correlate; ROISize is in pixels and has to be at least 8")
  n = np.atleast_1d(numROI)
  n = (int(n[0]), max(1, int(_round(n[0] * nRows / nCols)))) if n.size == 1 else (int(n[0]), int(n[1]))
  valid = np.isfinite(a) & np.isfinite(b)
  if not valid.any():
    raise ValueError("the two images have no pixel where both are finite")
  rows, cols = np.flatnonzero(valid.any(axis=1)) + 1, np.flatnonzero(valid.any(axis=0)) + 1
  region = (cols[0] + 1, rows[0] + 1, cols[-1] - cols[0] - 1, rows[-1] - rows[0] - 1)
  cx, cy = xcfROIGrid(roiSize, n, region)
  fPass = (0, 0, _round(roiSize / 2), _round(roiSize / 4))
  bandPass, window = xcfFilters(roiSize, fPass)
  dx, dy, peak = xcfCorrelate(a, b, cx, cy, roiSize, fPass, XCFMesh, window, bandPass)
  # the correlation finds A(p) = B(p - d); u is the displacement from A to B
  dx, dy = -dx, -dy
  u = vector3d(dx[:, None] * mg._d2 + dy[:, None] * mg._d1)
  ccy, ccx = cy.reshape(-1, order="F") - 1, cx.reshape(-1, order="F") - 1
  pos = vector3d(mg._origin + ccy[:, None] * mg._d1 + ccx[:, None] * mg._d2)
  return u, peak, pos


class pairShifts:
  """the shifts one cross correlation pass measured between two images

    ps = pairShifts(pos, u, peak, roiSize)    the tile centres, their shifts, peaks and width
    v = ps.meanShift(px)                      [mean length, mean x, mean y] in pixels of size px
  """

  def __init__(self, pos=None, u=None, peak=None, roiSize=None):
    self.pos = vector3d(np.zeros((0, 3))) if pos is None else pos.flatten()
    self.u = vector3d(np.zeros((0, 3))) if u is None else u.flatten()
    self.peak = np.zeros(0) if peak is None else np.asarray(peak, dtype=float).reshape(-1)
    self.roiSize = roiSize

  @property
  def isempty(self):
    return self.u.size == 0

  def meanShift(self, px):
    x, y = np.asarray(self.u.x).reshape(-1) / px, np.asarray(self.u.y).reshape(-1) / px
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
      return np.zeros(3)
    return np.array([np.mean(np.hypot(x[ok], y[ok])), np.mean(x[ok]), np.mean(y[ok])])

  def __repr__(self):
    ok = np.isfinite(self.peak)
    return f"pairShifts\n  tiles: {self.peak.size} ({int(ok.sum())} measured)\n  ROI: {self.roiSize} px"


__all__ = ["xcfShift", "pairShifts"]
