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


def test_D1_a_constant_field_is_a_fixed_point():
  e = angleMapD(lambda i, j: np.full(len(i), 10.0), 7)
  for F in (meanFilter(), medianFilter(), KuwaharaFilter(), splineFilter(), halfQuadraticFilter(), l1TVFilter(), infimalConvolutionFilter()):
    s = e.smooth(F)
    assert maxAngle(e, s) < 1e-9 and s.shape == e.shape
  assert maxAngle(e, smooth(e)) < 1e-9
  assert "smoothing length" in repr(splineFilter()) and "neighbours" in repr(meanFilter(2)) and "TV term" in repr(l1TVFilter())


def test_D2_a_linear_field_survives_the_window_filters():
  # the mean and the median of a window symmetric about its centre are the value at the
  # centre, whatever the slope
  e = angleMapD(lambda i, j: 0.5 * i + 0.25 * j, 9)
  inner = np.array([i + j * 9 for i in range(2, 7) for j in range(2, 7)])
  for F in (meanFilter(), medianFilter(), meanFilter(numNeighbours=2)):
    s = e.smooth(F)
    assert np.max(np.abs(angles(s)[inner] - angles(e)[inner])) < 1e-9


def test_D3_one_wild_pixel():
  # a constant field with one pixel turned by 5 degrees: the median puts it back exactly,
  # the mean spreads it over the five pixels of the window
  e = angleMapD(lambda i, j: np.where((i == 4) & (j == 4), 5.0, 0.0), 9)
  k = 4 + 4 * 9
  assert abs(angles(e.smooth(medianFilter()))[k]) < 1e-9
  m = angles(e.smooth(meanFilter()))
  assert close(m[k], 1.0, 1e-9) and close(m[k + 1], 1.0, 1e-9)


def test_D4_a_step_is_blurred_or_kept():
  e = angleMapD(lambda i, j: np.where(i < 3, 0.0, 20.0), 6)
  left, right = 2 + 3 * 6, 3 + 3 * 6
  m = angles(e.smooth(meanFilter()))
  assert close(m[left], 4.0, 1e-6) and close(m[right], 16.0, 1e-6)          # one fifth of the step
  # the l1 filter does not smooth across a difference above its threshold, and what is left
  # is constant on either side, so nothing moves at all
  for F in (l1TVFilter(threshold=15 * degree), KuwaharaFilter()):
    assert np.max(np.abs(angles(e.smooth(F)) - angles(e))) < 1e-6
  # with the threshold above the step the same filter does smooth across it
  assert angles(e.smooth(l1TVFilter(threshold=25 * degree, alpha=10.0)))[left] > 0.5


def test_D5_noise_is_what_a_filter_takes_out():
  rng = np.random.default_rng(5)
  nu = 0.7 * rng.normal(size=15 * 15)
  e = angleMapD(lambda i, j: nu[i + j * 15], 15)
  for F in (splineFilter(), meanFilter(), halfQuadraticFilter(alpha=5.0)):
    assert np.std(angles(e.smooth(F))) < 0.6 * np.std(angles(e))
  # and the mean of the field stays where it was
  assert abs(np.mean(angles(e.smooth(splineFilter()))) - np.mean(angles(e))) < 0.05


def test_D6_the_pixels_that_carry_no_measurement():
  # a hole in a linear field is filled with the continuation of it, not with a copy of a
  # neighbour: the field is harmonic, so the answer is exact
  e = angleMapD(lambda i, j: 0.5 * i + 0.25 * j, 9, missingAt=lambda i, j: (i == 4) & (j == 4))
  k = 4 + 4 * 9
  assert e.rotations.isnan()[k] and not e.isIndexed[k]
  f = e.fill(splineFilter())
  assert not f.rotations.isnan()[k] and close(angles(f)[k], 0.5 * 4 + 0.25 * 4, 1e-6) and f.isIndexed[k]
  assert e.rotations.isnan()[k]                                              # the original is left alone
  # the measurements themselves do not move, which is what pinning them means
  others = np.arange(81) != k
  assert np.max(np.abs(angles(f)[others] - angles(e)[others])) < 1e-9
  # the edge preserving filter fills too, a little short of the continuation
  assert abs(angles(fill(e, l1TVFilter()))[k] - 3.0) < 0.2 and abs(angles(e.fill())[k] - 3.0) < 0.2
  # a hole at the border is outside the measured region, so it stays empty
  b = angleMapD(lambda i, j: np.full(len(i), 3.0), 7, missingAt=lambda i, j: (i == 0) & (j == 0))
  assert b.fill().rotations.isnan()[0] and not b.fill(extrapolate=True).rotations.isnan()[0]
  # smoothing solves for the holes as well when it is asked to, and a window filter, which
  # can solve for nothing, has them closed first instead
  assert not e.smooth(splineFilter(), fill=True).rotations.isnan()[k] and e.smooth(splineFilter()).rotations.isnan()[k]
  assert not e.smooth(meanFilter(), fill=True).rotations.isnan()[k] and e.smooth(meanFilter()).rotations.isnan()[k]
  with pytest.raises(TypeError):
    e.fill(meanFilter())


def test_D6b_the_properties_of_a_filled_pixel():
  e = angleMapD(lambda i, j: 0.5 * i, 9, missingAt=lambda i, j: (i == 4) & (j == 4))
  i = np.arange(81) % 9
  e.ci = i.astype(float)
  e.label = np.where(i < 4, 1, 2).astype(np.int32)
  k = 4 + 4 * 9
  f = e.fill(splineFilter())
  # a numeric property is filled by the same solve, so a linear one is exact
  assert close(f.ci[k], 4.0, 1e-6) and np.max(np.abs(f.ci - e.ci)) < 1e-6
  # a label is copied from the nearest measurement, since continuing it is meaningless
  assert f.label[k] in (1, 2) and f.label.dtype == np.int32
  # smoothing leaves the properties alone unless it is asked
  n = e.smooth(splineFilter(), fill=True)
  assert np.array_equal(n.ci, e.ci) and not np.array_equal(e.smooth(splineFilter(), fill=True, props=["ci"]).ci, e.ci)
  with pytest.raises(ValueError):
    e.smooth(splineFilter(), props=["label"])
  with pytest.raises(ValueError):
    e.smooth(meanFilter(), props=["ci"])


def test_D7_the_grains_a_map_describes():
  e = angleMapD(lambda i, j: np.where(i < 4, 0.2 * j, 20.0), 8)
  g = calcGrains(e, angle=5 * degree)
  f = e.fillByGrainId(g)
  assert len(np.unique(np.round(angles(f), 9))) == len(g)
  # every pixel of a grain now carries its mean
  assert np.all(f.rotations.angle(g.meanRotation[e.grainId - 1]) < 1e-9) and fillByGrainId(e, g).shape == e.shape
  with pytest.raises(ValueError):
    angleMapD(lambda i, j: 0 * i, 4).fillByGrainId(g)


def test_D8_a_coarser_map_and_the_field_read_at_a_point():
  e = angleMapD(lambda i, j: 0.5 * i, 9)
  c = e.reduce(3)
  assert len(c) == 9 and close(np.sort(np.unique(np.round(angles(c), 6))), [0.0, 1.5, 3.0]) and len(reduce(e, 3)) == 9 == len(subsample(e, 3))
  assert len(e.reduce()) == 25 and c.lattice.shape == "square" and close(c.lattice.dxy, 3)
  with pytest.raises(ValueError):
    e.reduce(0)
  # the field between two measurements, which for a linear one is exact
  p = e.interp(np.array([[2.1, 3.0], [0.0, 0.0], [2.5, 1.5]]))
  assert len(p) == 3 and close(angles(p), [1.05, 0.0, 1.25], 1e-9) and p.frame is e.frame
  assert close(angles(interp(e, [2.1, 0.0], [3.0, 0.0])), [1.05, 0.0], 1e-9)
  # and it does not average across a grain boundary
  two = angleMapD(lambda i, j: np.where(i < 4, 0.0, 20.0), 9)
  calcGrains(two, angle=5 * degree)
  q = two.interp(np.array([[3.5, 4.0]]))
  assert min(abs(angles(q)[0] - 0.0), abs(angles(q)[0] - 20.0)) < 1e-9
  # a position beyond the scan takes the nearest measurement, or nothing
  far = e.interp(np.array([[30.0, 30.0]]))
  assert not far.rotations.isnan()[0] and e.interp(np.array([[30.0, 30.0]]), extrapolate=False).rotations.isnan()[0]


def test_D9_single_precision_and_a_hexagonal_map():
  e = angleMapD(lambda i, j: np.where((i == 3) & (j == 3), 5.0, 0.0), 8, A=HEX)
  k = 3 + 3 * 8
  assert abs(angles(e.smooth(medianFilter()))[k]) < 1e-9 and close(angles(e.smooth(meanFilter()))[k], 5 / 7, 1e-9)   # six neighbours
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
