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


def test_fill_continuesALinearFieldIntoAHole():
  # a hole in a linear field is filled with the continuation of it, not with a copy of a
  # neighbour: the field is harmonic, so the answer is exact
  e, k = holedLinearMap()
  assert e.rotations.isnan()[k] and not e.isIndexed[k]
  f = e.fill(splineFilter())
  assert not f.rotations.isnan()[k] and close(angles(f)[k], 0.5 * 4 + 0.25 * 4, 1e-6) and f.isIndexed[k]
  assert e.rotations.isnan()[k]                                              # the original is left alone


# the notIndexed pixels calcGrains absorbs into a grain keep its phase and a NaN rotation; fill
# continues the field into them as into any hole
def test_fill_fillsThePixelsCalcGrainsAbsorbed():
  holes = lambda i, j: ((i == 4) & (j == 4)) | ((i == 2) & (j == 6)) | ((i == 6) & (j == 3))
  e = angleMapD(lambda i, j: 0.5 * i + 0.25 * j, 9, missingAt=holes)
  calcGrains(e, angle=5 * degree)
  k = np.flatnonzero(e.rotations.isnan().reshape(-1))
  assert len(k) == 3 and np.all(e.isIndexed.reshape(-1)[k])
  i, j = k % 9, k // 9
  f = e.fill(splineFilter())
  assert not f.rotations.isnan().any() and close(angles(f)[k], 0.5 * i + 0.25 * j, 1e-6)
  assert not e.smooth(splineFilter(), fill=True).rotations.isnan().any() and e.smooth(splineFilter()).rotations.isnan().sum() == 3


def test_fill_pinsTheMeasurements():
  # the measurements themselves do not move, which is what pinning them means
  e, k = holedLinearMap()
  f = e.fill(splineFilter())
  others = np.arange(81) != k
  assert np.max(np.abs(angles(f)[others] - angles(e)[others])) < 1e-9


def test_fill_edgePreservingFillsALittleShort():
  # the edge preserving filter fills too, a little short of the continuation
  e, k = holedLinearMap()
  assert abs(angles(fill(e, l1TVFilter()))[k] - 3.0) < 0.2 and abs(angles(e.fill())[k] - 3.0) < 0.2


def test_fill_leavesAHoleAtTheBorderUnlessExtrapolating():
  # a hole at the border is outside the measured region, so it stays empty
  b = angleMapD(lambda i, j: np.full(len(i), 3.0), 7, missingAt=lambda i, j: (i == 0) & (j == 0))
  assert b.fill().rotations.isnan()[0] and not b.fill(extrapolate=True).rotations.isnan()[0]


def test_smooth_fillsHolesOnlyWhenAsked():
  # smoothing solves for the holes as well when it is asked to, and a window filter, which
  # can solve for nothing, has them closed first instead
  e, k = holedLinearMap()
  assert not e.smooth(splineFilter(), fill=True).rotations.isnan()[k] and e.smooth(splineFilter()).rotations.isnan()[k]
  assert not e.smooth(meanFilter(), fill=True).rotations.isnan()[k] and e.smooth(meanFilter()).rotations.isnan()[k]


def test_fill_refusesAWindowFilter():
  e, _ = holedLinearMap()
  with pytest.raises(TypeError):
    e.fill(meanFilter())


def test_fill_solvesForANumericProperty():
  # a numeric property is filled by the same solve, so a linear one is exact
  e, k = holedMapWithProperties()
  f = e.fill(splineFilter())
  assert close(f.ci[k], 4.0, 1e-6) and np.max(np.abs(f.ci - e.ci)) < 1e-6


def test_fill_copiesALabelFromTheNearestMeasurement():
  # a label is copied from the nearest measurement, since continuing it is meaningless
  e, k = holedMapWithProperties()
  f = e.fill(splineFilter())
  assert f.label[k] in (1, 2) and f.label.dtype == np.int32


def test_smooth_leavesThePropertiesAloneUnlessAsked():
  # smoothing leaves the properties alone unless it is asked
  e, _ = holedMapWithProperties()
  n = e.smooth(splineFilter(), fill=True)
  assert np.array_equal(n.ci, e.ci) and not np.array_equal(e.smooth(splineFilter(), fill=True, props=["ci"]).ci, e.ci)


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
      out[kernels] += [angles(e.fill(splineFilter())), angles(e.fill(l1TVFilter(threshold=0.01))), angles(e.smooth(infimalConvolutionFilter(alpha=1.0, beta=1.0), fill=True))]
  finally:
    setMTEXpref("kernels", before)
  for a, b in zip(out["numba"], out["numpy"]):
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) < 1e-6


def test_the_padding_of_a_grid_is_never_filled():
  from test_gridify import turnedMap
  g = turnedMap().gridify()
  pad = g.isPadding.reshape(-1)
  for f in (g.fill(), g.fill(extrapolate=True), g.smooth(halfQuadraticFilter(), fill=True, extrapolate=True)):
    assert np.all(f.isPadding.reshape(-1) == pad) and f.rotations.isnan().reshape(-1)[pad].all() and np.all(f.phaseId.reshape(-1)[pad] == 0)
  # the one measured notIndexed pixel is filled, since it is inside the scan
  assert len(g.fill(extrapolate=True)["notIndexed"]) == 0


def test_fill_within_the_grains_fills_their_padding_too():
  # a sparse map whose dropped measurements are padding of its grid, MATLAB's
  # fill(ebsd, grains, 'gridify')
  from mtex.io.sampledata import testGrainsMap as grainsMap
  e = grainsMap()
  sub = e[np.random.default_rng(1).random(e.size) > 0.6]
  grains = calcGrains(sub, 5 * degree)
  g = sub.gridify()
  pad = g.isPadding.reshape(-1).copy()
  f = fill(g, grains, splineFilter())
  assert np.array_equal(g.isPadding.reshape(-1), pad)
  measured = ~g.rotations.isnan().reshape(-1)
  assert np.array_equal(f.rotations.data.reshape(-1, 4)[measured], g.rotations.data.reshape(-1, 4)[measured])
  # what is left empty is padding outside every grain or a measured site in none
  empty = f.rotations.isnan().reshape(-1)
  assert pad.sum() > 1000 and f.isPadding.sum() < pad.sum() / 10
  assert np.all(f.isPadding.reshape(-1)[empty] | (f.phaseId.reshape(-1)[empty] == 0))
  # a filled site lies in the grain it was filled from
  k = np.flatnonzero(pad & ~empty)[:200]
  from mtex import vector3d
  hit = grains.checkInside(vector3d(f.pos.data.reshape(-1, 3)[k], frame=f.frame))
  assert np.all(hit[np.arange(len(k)), grains.id2ind(f.grainId.reshape(-1)[k])])
  # the grains are the second argument or the keyword, and the filter may come first
  assert np.array_equal(fill(g, splineFilter(), grains=grains).grainId, f.grainId)


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
