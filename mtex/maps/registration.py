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
from ..geometry.display import aligned


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
  x, y = x[x + h <= region[0] + region[2]], y[y + h <= region[1] + region[3]]
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
def _dftKernel(f, w, mesh, rs):
  u = np.arange(w) - w // 2
  return np.exp((-2j * np.pi / (rs * mesh)) * np.outer(u, f))


# the peak on a grid of spacing 1/mesh centred on the estimate, CC_t = K diag(pr_t) Q_t diag(pc_t) K^T:
# the right factor of all tiles as one product, the row ramp applied to it, and CC_t computed
# transposed, so that MATLAB's column order is the memory order its maximum is taken in
def _refine(Q, f, rowShift, colShift, K, mesh, rs):
  w, nf = K.shape
  T = Q.shape[0]
  rowShift = _round(rowShift * mesh) / mesh
  colShift = _round(colShift * mesh) / mesh
  pr = np.exp((-2j * np.pi / rs) * (rowShift[:, None] * f[None, :]))
  pc = np.exp((-2j * np.pi / rs) * (colShift[:, None] * f[None, :]))
  Y = ((Q * pc[:, None, :]).reshape(T * nf, nf) @ K.T).reshape(T, nf, w)
  Y *= pr[:, :, None]
  CCt = (Y.transpose(0, 2, 1).reshape(T * w, nf) @ K.T).reshape(T, w, w)
  # MATLAB's max of complex numbers is by their modulus, the first in column order
  a = np.abs(CCt).reshape(T, -1)
  k = np.argmax(a, axis=1)
  peak = a[np.arange(T), k]
  d = w // 2
  return rowShift + (k % w - d) / mesh, colShift + (k // w - d) / mesh, peak


# zero mean, unit deviation, window, transform, band pass; a tile with a non finite value or
# none of either is bad, and its spectrum zero
def _tileSpectrum(tile, window, bandPass, fft, workers):
  mu = tile.mean(axis=(1, 2), keepdims=True)
  sd = tile.std(axis=(1, 2), ddof=1, keepdims=True)
  with np.errstate(invalid="ignore", divide="ignore"):
    tile = (tile - mu) / sd
  bad = ~np.isfinite(sd[:, 0, 0]) | (sd[:, 0, 0] == 0) | ~np.isfinite(tile).all(axis=(1, 2))
  tile[bad] = 0
  tile *= window
  spec = fft.fft2(tile, workers=workers, overwrite_x=True)
  spec *= bandPass
  spec[bad] = 0
  return spec, bad


def xcfCorrelate(A, B, cx, cy, roiSize, fPass, mesh, window, bandPass):
  """the sub pixel shift of the tiles about (cx, cy) of B against A, in pixels, with the height
  of their peaks; batched, the coarse correlation held near 128 MB"""
  from numpy.lib.stride_tricks import sliding_window_view
  fft, workers = _fft()
  coarse = min(48, mesh)
  rs = int(roiSize)
  h = rs // 2
  nR = cx.size
  batch = max(1, (1 << 21) // max(rs * rs, 1))
  # MATLAB's band, the lowest nKeep frequencies of either sign in FFT order: its two halves
  # overlap, and the refinement weights a frequency held twice by its multiplicity
  nKeep = int(fPass[2] + fPass[3])
  mult = np.zeros(rs)
  mult[:nKeep] += 1
  mult[rs - nKeep:] += 1
  used = np.flatnonzero(mult)
  f = (np.fft.ifftshift(np.arange(rs)) - rs // 2)[used].astype(float)
  M = (mult[:, None] * mult[None, :])[np.ix_(used, used)]
  twoPass = coarse < mesh
  KC = _dftKernel(f, int(np.ceil(coarse * 1.5)), coarse, rs)
  if twoPass:
    KF = _dftKernel(f, 2 * int(np.ceil(mesh / coarse)) + 1, mesh, rs)
  cenX, cenY = cx.reshape(-1, order="F"), cy.reshape(-1, order="F")
  dx, dy, peak = np.zeros(nR), np.zeros(nR), np.zeros(nR)
  winA, winB = sliding_window_view(A, (rs, rs)), sliding_window_view(B, (rs, rs))
  # the band embedded in a 2 rs array, its low half at the start and its high half at the end,
  # so that the inverse lands the peak on a half pixel grid over the whole range
  runs = ((slice(0, nKeep), slice(0, nKeep)), (slice(2 * rs - nKeep, 2 * rs), slice(rs - nKeep, rs)))
  for b0 in range(0, nR, batch):
    b = np.arange(b0, min(b0 + batch, nR))
    r0, c0 = cenY[b] - h - 1, cenX[b] - h - 1
    specA, badA = _tileSpectrum(winA[r0, c0], window, bandPass, fft, workers)
    specB, badB = _tileSpectrum(winB[r0, c0], window, bandPass, fft, workers)
    bad = badA | badB
    P = specA * np.conj(specB)
    # transposed, so that the column order MATLAB takes its maximum in is the memory order
    Pt = P.transpose(0, 2, 1)
    CC = np.zeros((len(b), 2 * rs, 2 * rs), dtype=complex)
    for to1, of1 in runs:
      for to2, of2 in runs:
        CC[:, to1, to2] = Pt[:, of1, of2]
    CC = fft.ifft2(CC, workers=workers, overwrite_x=True)
    pk = np.argmax(np.abs(CC).reshape(len(b), -1), axis=1)
    rloc, cloc = pk % (2 * rs) + 1, pk // (2 * rs) + 1
    rowShift = np.where(rloc > rs, rloc - 2 * rs - 1, rloc - 1) / 2
    colShift = np.where(cloc > rs, cloc - 2 * rs - 1, cloc - 1) / 2
    Q = np.conj(P)
    if len(used) < rs:
      Q = Q[:, used][:, :, used]
    Q *= M
    rowShift, colShift, pkH = _refine(Q, f, rowShift, colShift, KC, coarse, rs)
    if twoPass:
      rowShift, colShift, pkH = _refine(Q, f, rowShift, colShift, KF, mesh, rs)
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
    """whether no shift was measured"""
    return self.u.size == 0

  def meanShift(self, px):
    """the mean shift in pixels of size px over the tiles measured, [mean length, mean x, mean y]"""
    x, y = np.asarray(self.u.x).reshape(-1) / px, np.asarray(self.u.y).reshape(-1) / px
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
      return np.zeros(3)
    return np.array([np.mean(np.hypot(x[ok], y[ok])), np.mean(x[ok]), np.mean(y[ok])])

  def __repr__(self):
    ok = np.isfinite(self.peak)
    return aligned(f"pairShifts\n  tiles: {self.peak.size} ({int(ok.sum())} measured)\n  ROI: {self.roiSize} px")


__all__ = ["xcfShift", "pairShifts"]
