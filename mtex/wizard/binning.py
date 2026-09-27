# the pole figure preview's density, estimated from a binned map (Julia W4)
#
#   odf = binnedDensity(ori, cell=4*degree)
#   odf = previewDensity(ebsd, phaseId)        of at most PREVIEW_POINTS measurements
#
# A preview density is a convolution, so what it costs is the number of centres; for a
# large map the orientations are merged onto a regular grid of their Euler angles, one
# representative per cell carrying how many measurements fell into it as its weight. Nothing
# is discarded, only merged, and a kernel several times wider than the cell cannot tell the
# difference. The cells are indexed by arithmetic, not hashed.
from __future__ import annotations

import numpy as np

from ..geometry.units import degree

BIN_ABOVE = 500
PREVIEW_POINTS = 100_000   # the measurements a preview density is estimated from


def binnedDensity(ori, cell=4 * degree, halfwidth=None, **kwargs):
  from ..functions.density import calcDensity
  # binned counts are no sample to select a halfwidth from: MATLAB's 10 degrees
  halfwidth = 10 * degree if halfwidth is None else halfwidth
  q = ori.data.reshape(-1, 4)
  keep = ~np.any(np.isnan(q), axis=1)
  ori = ori.reshape(-1)[keep] if not keep.all() else ori.reshape(-1)
  if ori.size <= BIN_ABOVE:
    return calcDensity(ori, halfwidth=halfwidth, harmonic=True, **kwargs)
  reps, weights = binOrientations(ori, cell)
  return calcDensity(reps, weights=weights, halfwidth=halfwidth, harmonic=True, **kwargs)


# the density of one phase of a map for the preview: a larger map is subsampled first, drawn
# at random but the same draw every time, since a pole figure at the kernel's width looks the
# same from a hundred thousand measurements as from millions, and the selection and the
# binning of millions cost a second
def previewDensity(ebsd, phaseId, maxPoints=PREVIEW_POINTS):
  idx = np.flatnonzero((np.asarray(ebsd.phaseId).reshape(-1) == phaseId) & np.asarray(ebsd.isIndexed).reshape(-1))
  if idx.size > maxPoints:
    idx = np.sort(np.random.default_rng(0).choice(idx, maxPoints, replace=False))
  return binnedDensity(ebsd[idx].orientations)


# one representative per occupied cell of the Euler grid, and the count of the cell
def binOrientations(ori, cell=4 * degree):
  phi1, Phi, phi2 = (np.asarray(a, dtype=float).reshape(-1) for a in ori.Euler("Bunge"))
  n1, n2, n3 = (int(np.ceil(2 * np.pi / cell)), int(np.ceil(np.pi / cell)) + 1, int(np.ceil(2 * np.pi / cell)))
  i1 = np.clip(np.floor(np.mod(phi1, 2 * np.pi) / cell).astype(np.int64), 0, n1 - 1)
  i2 = np.clip(np.floor(Phi / cell).astype(np.int64), 0, n2 - 1)
  i3 = np.clip(np.floor(np.mod(phi2, 2 * np.pi) / cell).astype(np.int64), 0, n3 - 1)
  key = (i1 * n2 + i2) * n3 + i3
  _, first, counts = np.unique(key, return_index=True, return_counts=True)
  return ori[first], counts.astype(float)


__all__ = ["binnedDensity", "binOrientations", "previewDensity"]
