# M8: the filters that replace an orientation by one of its neighbourhood, the Julia D1 to D9
import numpy as np
import pytest

from mtex import (EBSD, phaseList, crystalFrame, rotation, zvector, degree, calcGrains, smooth, fill, fillByGrainId, reduce, subsample, interp, setMTEXpref, getMTEXpref,
                  meanFilter, medianFilter, KuwaharaFilter, splineFilter, halfQuadraticFilter, l1TVFilter, infimalConvolutionFilter)
from conftest import close

CSD = crystalFrame("m-3m", mineral="test")
HEX = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])


# a square map whose pixel (i, j) carries a rotation about z of omega(i, j) degrees, the
# pixels missingAt names without a measurement
def angleMapD(omega, n, A=np.eye(2), dtype=np.float64, missingAt=None):
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  pos = np.stack([A[0, 0] * i + A[0, 1] * j, A[1, 0] * i + A[1, 1] * j, np.zeros(n * n)], axis=-1).astype(dtype)
  w = np.asarray(omega(i, j), dtype=float) * degree
  q = rotation.byAxisAngle(zvector, w).data.astype(dtype)
  ph = np.ones(n * n, dtype=np.int8)
  if missingAt is not None:
    gone = np.asarray(missingAt(i, j), dtype=bool)
    q[gone] = np.nan
    ph[gone] = 0
  return EBSD(pos, rotation(q), ph, phaseList(CSD))


# the rotation angle about z of every pixel, in degrees
def angles(e):
  q = e.rotations.data.reshape(-1, 4)
  return np.degrees(2 * np.arctan2(q[:, 3], q[:, 0]))


def maxAngle(a, b):
  return float(np.nanmax(a.rotations.angle(b.rotations)))


# a linear field with one hole in its middle, at pixel k
def holedLinearMap():
  return angleMapD(lambda i, j: 0.5 * i + 0.25 * j, 9, missingAt=lambda i, j: (i == 4) & (j == 4)), 4 + 4 * 9


# a map with a hole, a numeric property ci and a label
def holedMapWithProperties():
  e = angleMapD(lambda i, j: 0.5 * i, 9, missingAt=lambda i, j: (i == 4) & (j == 4))
  i = np.arange(81) % 9
  e.ci = i.astype(float)
  e.label = np.where(i < 4, 1, 2).astype(np.int32)
  return e, 4 + 4 * 9


@pytest.mark.parametrize("F", [meanFilter(), medianFilter(), KuwaharaFilter(), splineFilter(), halfQuadraticFilter(), l1TVFilter(),
                               infimalConvolutionFilter(), None], ids=lambda F: type(F).__name__)
def test_smooth_keepsAConstantFieldFixed(F):
  e = angleMapD(lambda i, j: np.full(len(i), 10.0), 7)
  s = smooth(e) if F is None else e.smooth(F)
  assert maxAngle(e, s) < 1e-9 and s.shape == e.shape


def test_repr_namesTheParametersOfAFilter():
  assert "smoothing length" in repr(splineFilter()) and "neighbours" in repr(meanFilter(2)) and "TV term" in repr(l1TVFilter())


@pytest.mark.parametrize("F", [meanFilter(), medianFilter(), meanFilter(numNeighbours=2)], ids=["mean", "median", "mean2"])
def test_smooth_windowFiltersKeepALinearField(F):
  # the mean and the median of a window symmetric about its centre are the value at the
  # centre, whatever the slope
  e = angleMapD(lambda i, j: 0.5 * i + 0.25 * j, 9)
  inner = np.array([i + j * 9 for i in range(2, 7) for j in range(2, 7)])
  s = e.smooth(F)
  assert np.max(np.abs(angles(s)[inner] - angles(e)[inner])) < 1e-9


def test_medianFilter_putsOneWildPixelBackAndTheMeanSpreadsIt():
  # a constant field with one pixel turned by 5 degrees: the median puts it back exactly,
  # the mean spreads it over the five pixels of the window
  e = angleMapD(lambda i, j: np.where((i == 4) & (j == 4), 5.0, 0.0), 9)
  k = 4 + 4 * 9
  assert abs(angles(e.smooth(medianFilter()))[k]) < 1e-9
  m = angles(e.smooth(meanFilter()))
  assert close(m[k], 1.0, 1e-9) and close(m[k + 1], 1.0, 1e-9)


def stepMap():
  return angleMapD(lambda i, j: np.where(i < 3, 0.0, 20.0), 6), 2 + 3 * 6, 3 + 3 * 6


def test_meanFilter_blursAStepByOneFifth():
  e, left, right = stepMap()
  m = angles(e.smooth(meanFilter()))
  assert close(m[left], 4.0, 1e-6) and close(m[right], 16.0, 1e-6)          # one fifth of the step


@pytest.mark.parametrize("F", [l1TVFilter(threshold=15 * degree), KuwaharaFilter()], ids=["l1TV", "Kuwahara"])
def test_smooth_edgePreservingFiltersKeepAStep(F):
  # the l1 filter does not smooth across a difference above its threshold, and what is left
  # is constant on either side, so nothing moves at all
  e, _, _ = stepMap()
  assert np.max(np.abs(angles(e.smooth(F)) - angles(e))) < 1e-6


def test_l1TVFilter_smoothsAcrossAStepBelowItsThreshold():
  # with the threshold above the step the same filter does smooth across it
  e, left, _ = stepMap()
  assert angles(e.smooth(l1TVFilter(threshold=25 * degree, alpha=10.0)))[left] > 0.5


def noisyMap():
  rng = np.random.default_rng(5)
  nu = 0.7 * rng.normal(size=15 * 15)
  return angleMapD(lambda i, j: nu[i + j * 15], 15)


@pytest.mark.parametrize("F", [splineFilter(), meanFilter(), halfQuadraticFilter(alpha=5.0)], ids=["spline", "mean", "halfQuadratic"])
def test_smooth_takesOutNoise(F):
  e = noisyMap()
  assert np.std(angles(e.smooth(F))) < 0.6 * np.std(angles(e))


def test_splineFilter_keepsTheMeanOfANoisyField():
  # and the mean of the field stays where it was
  e = noisyMap()
  assert abs(np.mean(angles(e.smooth(splineFilter()))) - np.mean(angles(e))) < 0.05


def test_fill_repeatsANeighbourInAHole():
  # a hole takes the orientation of its nearest measurement, as MATLAB's fill
  e, k = holedLinearMap()
  assert e.rotations.isnan()[k] and not e.isIndexed[k]
  f = e.fill()
  assert f.isIndexed[k] and min(abs(angles(f)[k] - angles(e)[[k - 1, k + 1, k - 9, k + 9]])) < 1e-9
  assert e.rotations.isnan()[k]                                              # the original is left alone


# the notIndexed pixels calcGrains absorbs into a grain keep its phase and a NaN rotation;
# smooth fills them and solves for them without a data term, so they continue a linear field
def test_smooth_fillsThePixelsCalcGrainsAbsorbed():
  holes = lambda i, j: ((i == 4) & (j == 4)) | ((i == 2) & (j == 6)) | ((i == 6) & (j == 3))
  e = angleMapD(lambda i, j: 0.5 * i + 0.25 * j, 9, missingAt=holes)
  calcGrains(e, angle=5 * degree)
  k = np.flatnonzero(e.rotations.isnan().reshape(-1))
  assert len(k) == 3 and np.all(e.isIndexed.reshape(-1)[k])
  i, j = k % 9, k // 9
  s = e.smooth(splineFilter())
  assert not s.rotations.isnan().any() and close(angles(s)[k], 0.5 * i + 0.25 * j, 0.1)


def test_fill_keepsTheMeasurements():
  e, k = holedLinearMap()
  others = np.arange(81) != k
  assert np.array_equal(e.fill().rotations.data[others], e.rotations.data[others])


def test_fill_withoutGrainIdsFillsTheNotIndexedPixelsInsideTheScan():
  # a notIndexed pixel inside the hull of the indexed ones is filled, one at the corner of the map is not
  b = angleMapD(lambda i, j: np.full(len(i), 3.0), 7, missingAt=lambda i, j: ((i == 0) & (j == 0)) | ((i == 3) & (j == 3)))
  f = b.fill()
  assert f.isIndexed[3 + 3 * 7] and not f.isIndexed[0]


def test_smooth_fillsHoles():
  # smooth fills before it filters, with a variational filter and a window filter alike
  e, k = holedLinearMap()
  assert not e.smooth(splineFilter()).rotations.isnan()[k] and not e.smooth(meanFilter()).rotations.isnan()[k]


def test_smooth_aWindowFilterComputesAFilledPixelFromItsMeasuredNeighbours():
  # the hole is the mean of its measured neighbours, exact on a linear field, not the copy fill made
  e, k = holedLinearMap()
  assert close(angles(e.smooth(meanFilter()))[k], 3.0, 1e-9) and abs(angles(e.fill())[k] - 3.0) > 0.1


def test_fill_withoutGrainIdsFillsThePaddingInsideTheMeasurements():
  # a site missing inside the hull of the measurements is a hole, the padding outside it stays empty
  e = angleMapD(lambda i, j: 0.5 * i, 7)
  g = e[np.arange(49) != 3 + 3 * 7].gridify()
  f = g.fill()
  assert g.isPadding.sum() == 1 and f.isPadding.sum() == 0 and not f.rotations.isnan().any()


def test_fill_keepsTheGrainIdsAndFillsWithinTheGrain():
  # a hole takes an orientation of its own grain, never one across the boundary
  e = angleMapD(lambda i, j: np.where(i < 4, 0.0, 30.0), 8, missingAt=lambda i, j: (i == 6) & (j == 3))
  calcGrains(e, angle=10 * degree)
  f = e.fill()
  k = 6 + 3 * 8
  assert np.array_equal(f.grainId, e.grainId) and abs(angles(f)[k] - 30.0) < 1e-9


def test_fill_isRestrictedToTheMask():
  # only the pixels without an orientation that are masked are filled, the others stay as they are
  holes = lambda i, j: ((i == 2) & (j == 2)) | ((i == 5) & (j == 5))
  e = angleMapD(lambda i, j: np.where(i < 4, 0.2 * j, 20.0 + 0.1 * j), 8, missingAt=holes)
  calcGrains(e, angle=10 * degree)
  mask = np.zeros(64, dtype=bool)
  mask[[2 + 2 * 8, 9]] = True
  f = e.fill(mask)
  nan = f.rotations.isnan()
  assert not nan[2 + 2 * 8] and nan[5 + 5 * 8] and np.array_equal(f.rotations.data[9], e.rotations.data[9])


def test_fill_refusesAMaskOfTheWrongSize():
  e, _ = holedLinearMap()
  with pytest.raises(ValueError):
    e.fill(np.ones(5, dtype=bool))


def test_fill_leavesThePropertiesAlone():
  # only orientation, phase and grain id are filled, as MATLAB's fill
  e, k = holedMapWithProperties()
  f = e.fill()
  assert np.array_equal(f.ci, e.ci) and np.array_equal(f.label, e.label)


def test_smooth_leavesThePropertiesAloneUnlessAsked():
  e, _ = holedMapWithProperties()
  assert np.array_equal(e.smooth(splineFilter()).ci, e.ci) and not np.array_equal(e.smooth(splineFilter(), props=["ci"]).ci, e.ci)


def test_smooth_refusesPropertiesItCannotSmooth():
  e, _ = holedMapWithProperties()
  with pytest.raises(ValueError):
    e.smooth(splineFilter(), props=["label"])
  with pytest.raises(ValueError):
    e.smooth(meanFilter(), props=["ci"])


def test_fillByGrainId_givesEveryPixelTheMeanOfItsGrain():
  e = angleMapD(lambda i, j: np.where(i < 4, 0.2 * j, 20.0), 8)
  g = calcGrains(e, angle=5 * degree)
  f = e.fillByGrainId(g)
  assert len(np.unique(np.round(angles(f), 9))) == len(g)
  # every pixel of a grain now carries its mean
  assert np.all(f.rotations.angle(g.meanRotation[e.grainId - 1]) < 1e-9) and fillByGrainId(e, g).shape == e.shape
  with pytest.raises(ValueError):
    angleMapD(lambda i, j: 0 * i, 4).fillByGrainId(g)


def test_reduce_keepsEveryNthPixelOnACoarserLattice():
  e = angleMapD(lambda i, j: 0.5 * i, 9)
  c = e.reduce(3)
  assert len(c) == 9 and close(np.sort(np.unique(np.round(angles(c), 6))), [0.0, 1.5, 3.0]) and len(reduce(e, 3)) == 9 == len(subsample(e, 3))
  assert len(e.reduce()) == 25 and c.lattice.shape == "square" and close(c.lattice.dxy, 3)
  # a grid stays a grid, its every n-th row and column
  g = e.gridify()
  r = g.reduce(3)
  assert r.isGridded and r.shape == (3, 3) and np.array_equal(r.pos.data, g.pos.data[::3, ::3])
  with pytest.raises(ValueError):
    e.reduce(0)


def test_interp_isExactOnALinearField():
  # the field between two measurements, which for a linear one is exact
  e = angleMapD(lambda i, j: 0.5 * i, 9)
  p = e.interp(np.array([[2.1, 3.0], [0.0, 0.0], [2.5, 1.5]]))
  assert len(p) == 3 and close(angles(p), [1.05, 0.0, 1.25], 1e-9) and p.frame is e.frame
  assert close(angles(interp(e, [2.1, 0.0], [3.0, 0.0])), [1.05, 0.0], 1e-9)


def test_interp_takesThePositionsAsVectors():
  # positions given as vectors in the plane of the map are read as the coordinates they hold
  from mtex import vector3d
  e = angleMapD(lambda i, j: 0.5 * i, 9)
  p = e.interp(vector3d(np.array([[2.1, 3.0, 0.0], [0.0, 0.0, 0.0], [2.5, 1.5, 0.0], [4.0, 1.0, 0.0], [6.5, 7.5, 0.0]])))
  assert close(angles(p), [1.05, 0.0, 1.25, 2.0, 3.25], 1e-9)


def test_interp_doesNotAverageAcrossAGrainBoundary():
  # and it does not average across a grain boundary
  two = angleMapD(lambda i, j: np.where(i < 4, 0.0, 20.0), 9)
  calcGrains(two, angle=5 * degree)
  q = two.interp(np.array([[3.5, 4.0]]))
  assert min(abs(angles(q)[0] - 0.0), abs(angles(q)[0] - 20.0)) < 1e-9


def test_interp_beyondTheScanTakesTheNearestOrNothing():
  # a position beyond the scan takes the nearest measurement, or nothing
  e = angleMapD(lambda i, j: 0.5 * i, 9)
  far = e.interp(np.array([[30.0, 30.0]]))
  assert not far.rotations.isnan()[0] and e.interp(np.array([[30.0, 30.0]]), extrapolate=False).rotations.isnan()[0]


def test_smooth_onAHexagonalMapUsesSixNeighbours():
  e = angleMapD(lambda i, j: np.where((i == 3) & (j == 3), 5.0, 0.0), 8, A=HEX)
  k = 3 + 3 * 8
  assert abs(angles(e.smooth(medianFilter()))[k]) < 1e-9 and close(angles(e.smooth(meanFilter()))[k], 5 / 7, 1e-9)   # six neighbours


def test_smooth_inSinglePrecisionStaysSingleAndNearlyFixesALinearField():
  e32 = angleMapD(lambda i, j: 0.5 * i + 0.25 * j, 8, dtype=np.float32)
  s = e32.smooth(splineFilter())
  assert s.rotations.data.dtype == np.float32
  # a linear field is very nearly a fixed point of the spline: the Laplacian of it is zero
  # inside, and only the border, where a pixel has fewer neighbours, moves at all
  assert maxAngle(e32, s) < 0.3 * degree


def test_the_compiled_and_the_numpy_filters_agree():
  # grains, a hole, a border gap and a hexagonal lattice, so every branch of the graph and
  # of the windows is met; the direct solve and the conjugate gradient meet to their tolerance
  rng = np.random.default_rng(9)
  nu = 0.4 * rng.normal(size=14 * 14)
  e = angleMapD(lambda i, j: np.where(i >= 7, 20.0, 0.0) + nu[i + j * 14], 14, A=HEX, missingAt=lambda i, j: ((i == 3) & (j == 4)) | ((i == 13) & (j == 0)))
  calcGrains(e, angle=5 * degree)
  before = getMTEXpref("kernels", "numba")
  out = {}
  try:
    for kernels in ("numba", "numpy"):
      setMTEXpref("kernels", kernels)
      out[kernels] = [angles(e.smooth(F)) for F in (meanFilter(), medianFilter(2), KuwaharaFilter(), splineFilter(), l1TVFilter(), halfQuadraticFilter(l1TV=False))]
      out[kernels] += [angles(e.smooth(splineFilter())), angles(e.smooth(l1TVFilter(threshold=0.01))), angles(e.smooth(infimalConvolutionFilter(alpha=1.0, beta=1.0)))]
  finally:
    setMTEXpref("kernels", before)
  for a, b in zip(out["numba"], out["numpy"]):
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) < 1e-6


def test_the_padding_of_a_grid_is_never_filled():
  from test_gridify import turnedMap
  g = turnedMap().gridify()
  pad = g.isPadding.reshape(-1)
  for f in (g.fill(), g.smooth(halfQuadraticFilter())):
    assert np.all(f.isPadding.reshape(-1) == pad) and f.rotations.isnan().reshape(-1)[pad].all() and np.all(f.phaseId.reshape(-1)[pad] == 0)
  # the one measured notIndexed pixel is filled, since the map has no grain ids
  assert len(g.fill()["notIndexed"]) == 0


def test_fill_within_the_grains_fills_their_padding_too():
  # a sparse map whose dropped measurements are padding of its grid, MATLAB's
  # fill(ebsd, grains, 'gridify')
  from mtex.io.sampledata import testGrainsMap as grainsMap
  e = grainsMap()
  sub = e[np.random.default_rng(1).random(e.size) > 0.6]
  grains = calcGrains(sub, 5 * degree)
  g = sub.gridify()
  pad = g.isPadding.reshape(-1).copy()
  f = fill(g, grains)
  assert np.array_equal(g.isPadding.reshape(-1), pad)
  measured = ~g.rotations.isnan().reshape(-1)
  assert np.array_equal(f.rotations.data.reshape(-1, 4)[measured], g.rotations.data.reshape(-1, 4)[measured])
  # what is left empty is padding outside every grain or crossed by a boundary, or a measured site in none
  empty = f.rotations.isnan().reshape(-1)
  assert pad.sum() > 1000 and f.isPadding.sum() < pad.sum() / 5
  assert np.all(f.isPadding.reshape(-1)[empty] | (f.phaseId.reshape(-1)[empty] == 0))
  # a filled site lies in the grain it was filled from
  k = np.flatnonzero(pad & ~empty)[:200]
  from mtex import vector3d
  hit = grains.checkInside(vector3d(f.pos.data.reshape(-1, 3)[k], frame=f.frame))
  assert np.all(hit[np.arange(len(k)), grains.id2ind(f.grainId.reshape(-1)[k])])
  # smooth fills the same sites before it filters
  assert np.array_equal(smooth(g, splineFilter(), grains).grainId, f.grainId)


# the sparse map of the grains test and its grains, every third measurement dropped
def sparseGrainsMap():
  from mtex.io.sampledata import testGrainsMap as grainsMap
  e = grainsMap()
  sub = e[np.random.default_rng(1).random(e.size) > 0.6]
  return sub, calcGrains(sub, 5 * degree)


def test_fill_turnsTheEmptySitesOnABoundaryIntoNotIndexedCells():
  # with grains every empty site of the scan is a cell afterwards, in a grain or notIndexed on a boundary
  sub, grains = sparseGrainsMap()
  f = fill(sub.gridify(), grains)
  added = ~np.isin(np.asarray(f.id).reshape(-1), np.asarray(sub.id).reshape(-1)) & ~f.isPadding.reshape(-1)
  gid = f.grainId.reshape(-1)[added]
  assert np.any(gid == 0) and np.all(f.phaseId.reshape(-1)[added][gid == 0] == 0)


def test_fill_withGrainIdsOnlyGivesAnEmptySiteTheGrainOfItsNearestPixel():
  sub, _ = sparseGrainsMap()
  f = fill(sub)
  from scipy.spatial import cKDTree
  new = ~np.isin(np.asarray(f.id), np.asarray(sub.id))
  P, Q = np.asarray(sub.pos.data).reshape(-1, 3), np.asarray(f.pos.data).reshape(-1, 3)[new]
  assert new.sum() > 1000 and np.array_equal(f.grainId[new], sub.grainId[cKDTree(P).query(Q)[1]])


def test_fill_keepsTheOrderOfAList():
  sub, _ = sparseGrainsMap()
  assert np.array_equal(np.asarray(fill(sub).id)[:sub.size], np.asarray(sub.id))


def test_smooth_withoutFillLeavesThePixelsWithoutOrientationAsTheyWere():
  e, k = holedLinearMap()
  s = e.smooth(splineFilter(), fill=False)
  assert s.rotations.isnan()[k] and not s.isIndexed[k] and s.size == e.size


def test_smooth_fillsOnlyTheMask():
  holes = lambda i, j: ((i == 2) & (j == 2)) | ((i == 6) & (j == 6))
  e = angleMapD(lambda i, j: 0.5 * i + 0.25 * j, 9, missingAt=holes)
  mask = np.zeros(81, dtype=bool)
  mask[2 + 2 * 9] = True
  s = e.smooth(splineFilter(), fill=mask)
  assert s.isIndexed[2 + 2 * 9] and s.rotations.isnan()[6 + 6 * 9]


# a map read at scattered positions is a list of them, read on a grid it has the grid's lattice;
# a map with no pixel to fill is returned as it is
def test_interp_atScatteredPositionsHasNoLatticeAndFillOfACompleteMapIsTheMap():
  e, _, _ = stepMap()
  rng = np.random.default_rng(9)
  assert e.interp(rng.random(500) * 5, rng.random(500) * 5).lattice is None
  i, j = np.meshgrid(np.arange(0, 5, 0.5), np.arange(0, 5, 0.5))
  assert e.interp(i.ravel(), j.ravel()).lattice is not None
  f = e.fill()
  assert f is not e and np.array_equal(f.rotations.data, e.rotations.data)
