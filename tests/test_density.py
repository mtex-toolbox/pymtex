# the density estimates on SO(3) and the sphere and the least squares fit
# (docs/plans/functions.md F4: R2, R6, R7 - check_calcDensityEmpty)
import numpy as np

from mtex import (Miller, S2DeLaValleePoussinKernel, S2FunHarmonic, SO3DirichletKernel, SO3FunHarmonic, SO3FunRBF, calcDensity,
                  crystalFrame, deg2dim, degree, interpolate, mean, orientation, rotation, specimenFrame, symmetrise, vector3d,
                  zvector)

rng = np.random.default_rng(42)


# fifty random orientations under m-3m and 1, after three NaNs
def mixedList():
  cs, ss = crystalFrame("m-3m"), specimenFrame("1")
  return orientation(rotation.cat(rotation.nan(3), rotation.rand(50, rng=rng)), cs, ss)


def test_calcDensity_ofEmptyOrNaNOrientationsIsUniformInTheirFrames():
  # check_calcDensityEmpty: empty and all-NaN lists give the uniform ODF in the frames asked
  cs, ss = crystalFrame("m-3m"), specimenFrame("1")
  for o in (orientation(rotation.nan(0), cs, ss), orientation(rotation.nan(3), cs, ss), orientation(rotation.nan(0), cs, cs, antipodal=True)):
    for harmonic in (None, True):
      odf = calcDensity(o, harmonic=harmonic)
      assert abs(mean(odf) - 1) < 1e-10 and odf.CS is o.CS and odf.SS is o.SS
      assert np.all(np.abs(odf(rotation.rand(5, rng=rng)) - 1) < 1e-6)


def test_calcDensity_dropsNaNsKeepsMeanOneAndPeaks():
  # a mixed list drops the NaNs, keeps mean one and, at MATLAB's 10 degrees, peaks above two
  o = mixedList()
  odf = calcDensity(o, halfwidth=10 * degree)
  assert isinstance(odf, SO3FunRBF) and odf.center.size == 50 and abs(mean(odf) - 1) < 1e-10
  assert np.max(odf(o[3:])) > 2


def test_calcDensity_ofFewUniformOrientationsSelectsAWideKernel():
  # without a halfwidth it is selected: fifty uniform orientations resolve nothing
  assert calcDensity(mixedList()).psi.halfwidth() > 20 * degree


def test_calcDensity_isAKernelFormBelow200AndHarmonicAbove():
  # the route: kernel form below 200, harmonic above, or on request
  o = mixedList()
  big = orientation(rotation.rand(300, rng=rng), o.CS, o.SS)
  assert isinstance(calcDensity(big), SO3FunHarmonic) and isinstance(calcDensity(big, harmonic=False), SO3FunRBF)
  assert isinstance(calcDensity(o, harmonic=True), SO3FunHarmonic)
  assert abs(mean(calcDensity(big)) - 1) < 1e-10


def test_calcDensity_normalisesTheWeights():
  o = mixedList()
  w = rng.random(50)
  odfw = calcDensity(o, weights=np.concatenate([np.zeros(3), w]))
  assert np.allclose(odfw.weights, w / np.sum(w) / odfw.psi.mean())


def test_calcDensity_withADirichletKernelOrACrystalSpecimenFrameIsHarmonic():
  # a Dirichlet kernel and a crystal specimen frame take the harmonic route
  o = mixedList()
  assert isinstance(calcDensity(o, kernel=SO3DirichletKernel(10)), SO3FunHarmonic)
  assert isinstance(calcDensity(orientation(rotation(o), o.CS, crystalFrame("-3m"))), SO3FunHarmonic)


def test_calcDensity_passesOnTheHalfwidthAndTheExchangeSymmetry():
  # the halfwidth and the exchange symmetry are passed on
  o = mixedList()
  assert np.isclose(calcDensity(o, halfwidth=20 * degree).psi.halfwidth(), 20 * degree)
  assert calcDensity(orientation(rotation(o), o.CS, o.CS, antipodal=True)).antipodal


def test_calcDensity_onTheSphereIsTheMeanOfTheKernel():
  vs = vector3d.rand(100, rng=rng)
  s = calcDensity(vs, halfwidth=15 * degree)
  assert np.isclose(mean(s), 1) and s.isReal and not s.antipodal and s.frame is None
  psi = S2DeLaValleePoussinKernel(halfwidth=15 * degree)
  for v in vector3d.rand(5, rng=rng):
    assert np.isclose(s(v), np.mean(psi(vs.dot(v))), rtol=1e-3)


def test_calcDensity_ofASingleDirectionIsTheKernel():
  one1 = calcDensity(zvector)
  assert np.isclose(one1(vector3d(np.sin(0.3), 0, np.cos(0.3))), S2DeLaValleePoussinKernel(halfwidth=10 * degree)(np.cos(0.3)), rtol=1e-3)


def test_calcDensity_ofAxesIsEven():
  a = calcDensity(vector3d(vector3d.rand(100, rng=rng), antipodal=True))
  assert a.antipodal and np.isclose(mean(a), 1)


def test_calcDensity_ofMillerIsSymmetric():
  vs = vector3d.rand(100, rng=rng)
  cs = crystalFrame("6/mmm")
  sf = calcDensity(Miller(vs, cs), halfwidth=15 * degree)
  assert sf.frame is cs and np.isclose(mean(sf), 1)
  m = vector3d(Miller(1, 2, 3, cs).data)
  E = cs.sym.rot.flatten()
  for i in range(E.size):
    assert np.isclose(sf(E[i] * m), sf(m), rtol=1e-8)


def test_calcDensity_ofEmptyWeightedOrAngleDataHasMeanOne():
  vs = vector3d.rand(100, rng=rng)
  assert np.isclose(mean(calcDensity(vector3d(np.zeros((0, 3))))), 1)
  assert np.isclose(mean(calcDensity(vs, weights=rng.random(100))), 1)
  # angles go to the circle
  assert np.isclose(mean(calcDensity([0.0, 2.0], sigma=0.2)), 1)


# a random real harmonic function under mmm and 1, and distinct orientations to sample it
def mmmSamples():
  cs, ss = crystalFrame("mmm"), specimenFrame("1")
  f = SO3FunHarmonic(rng.standard_normal(deg2dim(6)) + 1j * rng.standard_normal(deg2dim(6)), cs, ss, real=True)
  # nodes closer than 0.02 are merged, so exact samples are taken at distinct ones
  o = orientation(rotation.rand(3000, rng=rng), cs, ss).unique(tol=0.02)
  return cs, ss, f, o


def test_interpolate_ofExactSamplesRecoversTheFunction():
  cs, ss, f, o = mmmSamples()
  g = interpolate(o, f(o), bandwidth=5, lambda_=0, tol=1e-10)
  assert g.isclose(f, 1e-4)
  rs = rotation.rand(20, rng=rng)
  assert np.allclose(g(rs), f(rs), rtol=1e-4, atol=1e-4 * np.max(np.abs(f(rs))))


def test_interpolate_takesTheMeanOfANodeGivenTwice():
  # a node given twice with two values is one node with their mean
  cs, ss, f, o = mmmSamples()
  o2 = orientation(np.concatenate([o.data, o.data[:1]]), cs, ss)
  v2 = np.concatenate([f(o), [f(o)[0] + 2]])
  v2[0] -= 2
  assert interpolate(o2, v2, bandwidth=5, lambda_=0, tol=1e-10).isclose(f, 1e-4)


def test_interpolate_ofNoisyDataStaysCloseAndIsSymmetric():
  # the regularised fit of noisy data stays close and is symmetric
  cs, ss, f, o = mmmSamples()
  rs = rotation.rand(20, rng=rng)
  gn = SO3FunHarmonic.interpolate(o, f(o) + 0.01 * rng.standard_normal(o.size), bandwidth=5, lambda_=1e-6)
  assert np.max(np.abs(gn(rs) - f(rs))) < 0.1 * np.max(np.abs(f(rs)))
  assert symmetrise(gn).isclose(gn)


def test_interpolate_overOneFundamentalRegionIsNotDividedByTheGroupOrder():
  # samples that cover one fundamental region and nothing else: the fit is the symmetric
  # function they determine, not that function divided by the order of the group
  csC = crystalFrame("m-3m")
  a, b, c = np.meshgrid(np.arange(0, 360, 10) * degree, np.arange(5, 60, 10) * degree, np.arange(0, 90, 10) * degree, indexing="ij")
  grid = orientation(rotation.byEuler(a.ravel(), b.ravel(), c.ravel(), "Bunge"), csC)
  h = interpolate(grid, np.ones(grid.size), bandwidth=12, lambda_=1e-6)
  assert abs(mean(h) - 1) < 1e-2
  assert np.max(np.abs(h(orientation(rotation.rand(50, rng=rng), csC)) - 1)) < 0.05


def test_interpolate_onTheSphereRecoversTheFunction():
  s = S2FunHarmonic(rng.standard_normal(36) + 1j * rng.standard_normal(36), real=True)
  vs = vector3d.rand(500, rng=rng)
  t = interpolate(vs, s(vs), bandwidth=5)
  assert t.isclose(s, 1e-4)


# a sample of 2000 orientations of a 10 degree unimodal ODF under m-3m
def sharpSample():
  from mtex import discreteSample, unimodalODF
  cs = crystalFrame("m-3m")
  center = orientation(rotation.rand(rng=rng), cs)
  return discreteSample(unimodalODF(center, halfwidth=10 * degree), 2000, rng=rng)


def test_calcDensity_selectsANarrowHalfwidthForASharpSampleReproducibly():
  o = sharpSample()
  f = calcDensity(o, harmonic=False)
  # a sharp sample gets a kernel narrower than MATLAB's fixed 10 degrees, reproducibly
  assert 2 * degree <= f.psi.halfwidth() < 10 * degree
  assert np.isclose(calcDensity(o, harmonic=False).psi.halfwidth(), f.psi.halfwidth())


def test_calcDensity_ofPixelsWithoutGrainsKeeps10DegreesWithAWarning():
  import warnings
  o = sharpSample()
  # the pixels of a map without their grains keep 10 degrees, with a warning
  o.notIID = True
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    g = calcDensity(o, harmonic=False)
  assert np.isclose(g.psi.halfwidth(), 10 * degree) and any("not independent" in str(x.message) for x in w)
  # with the grain ids it selects again
  assert calcDensity(o, groups=np.arange(2000) // 10, harmonic=False).psi.halfwidth() != 10 * degree


def test_calcDensity_ofDirectionsSelectsTheBandwidth():
  v = vector3d(rng.standard_normal((3000, 3)))
  assert calcDensity(v).bandwidth < 25
