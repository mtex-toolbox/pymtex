# the texture characteristics (docs/plans/functions.md F6: C1 to C7, check_calcComponents,
# check_axisDistribution, and the definitions checked on functions with known answers)
from types import SimpleNamespace

import numpy as np
import pytest
from numpy import pi

from mtex import (S1FunHandle, S1FunHarmonic, S2FunHandle, Miller, S2DeLaValleePoussinKernel, S2FunBingham, S2FunHarmonic, SO3DeLaValleePoussinKernel, SO3FunHarmonic,
                  angle, bandwidthFor, calcAngleDistribution, calcAxisDistribution, calcComponents, calcDensity, calcError,
                  calcKernel, calcMDF, calcMIndex, centerSpecimen, conv, crystalFrame, degree, discreteSample, entropy,
                  equispacedS2Grid, equispacedSO3Grid, fibreODF, fibreVolume, max, mean, min, norm, orientation, quadratureSO3Grid, rotate,
                  rotation, specimenFrame, textureIndex, unimodalODF, uniformODF, vector3d, volume, yvector, zvector)

rng = np.random.default_rng(71)


# a cubic unimodal ODF of halfwidth 15 degree, as kernel form and harmonic
@pytest.fixture(scope="module")
def cubic15():
  cs, ss = crystalFrame("m-3m"), specimenFrame("1")
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=15 * degree)
  return SimpleNamespace(cs=cs, ss=ss, u=uniformODF(cs, ss), odf=odf, H=SO3FunHarmonic(odf))


def test_textureindex_ofTheUniformODFIsOneItsEntropyZero(cubic15):
  u = cubic15.u
  assert np.isclose(textureIndex(u), 1) and abs(entropy(u)) < 1e-10


def test_textureindex_isTheSquaredNorm(cubic15):
  H, odf = cubic15.H, cubic15.odf
  assert np.isclose(textureIndex(H), norm(H) ** 2) and np.isclose(textureIndex(odf), textureIndex(H), rtol=1e-6)


def test_entropy_ofADensityIsNegativeAndMinusTheIntegralOfFLogF(cubic15):
  # the entropy of a density is at most zero, and -int f log f by an independent rule
  cs, ss, H = cubic15.cs, cubic15.ss, cubic15.H
  e = entropy(H)
  assert e < 0
  g = quadratureSO3Grid(bandwidthFor(80, cs, ss), cs, ss)
  v = H(g.nodes())
  assert np.isclose(e, -np.sum(g.weights * np.where(v > 0, v * np.log(np.where(v > 0, v, 1)), 0)), rtol=1e-3)
  assert np.isclose(entropy(cubic15.odf), e, rtol=1e-2) and np.isclose(H.entropy(), e)


def test_volume_ofABallAroundTheModeOfASharpODFIsNearlyOne(cubic15):
  c = orientation(rotation.rand(rng=rng), cubic15.cs, cubic15.ss)
  sharp = unimodalODF(c, halfwidth=5 * degree)
  assert 0.95 < volume(sharp, c, 20 * degree) <= 1


def test_volume_ofTheUniformODFIsTheShareOfTheBall(cubic15):
  c = orientation(rotation.rand(rng=rng), cubic15.cs, cubic15.ss)
  u = cubic15.u
  assert np.isclose(volume(u, c, 10 * degree), 24 * (10 * degree - np.sin(10 * degree)) / pi, rtol=1e-2)
  assert volume(u, c, 60 * degree) == 1


def test_fibreVolume_ofAFibreODFAroundItsOwnFibreIsNearlyOne(cubic15):
  h, r = Miller(1, 1, 1, cubic15.cs), vector3d.rand(rng=rng)
  fodf = fibreODF(h, r, cubic15.ss, halfwidth=5 * degree)
  assert fibreVolume(fodf, h, r, 15 * degree, resolution=2 * degree) > 0.9


def test_fibreVolume_ofTheUniformODFIsTheShareOfTheTube(cubic15):
  h, r = Miller(1, 1, 1, cubic15.cs), vector3d.rand(rng=rng)
  assert np.isclose(fibreVolume(cubic15.u, h, r, 10 * degree), min(1, 8 * (1 - np.cos(10 * degree)) / 2), rtol=5e-2)


def test_textureindex_onTheSphereIsTheSquaredNormAndTheEntropyNegative():
  s = S2FunHarmonic(S2DeLaValleePoussinKernel(halfwidth=20 * degree))
  assert np.isclose(textureIndex(s), norm(s) ** 2) and entropy(s) < 0


def test_entropy_ofTheUniformSphericalFunctionIsZero():
  assert abs(entropy(S2FunHarmonic([np.sqrt(4 * pi)]))) < 1e-10


# two modes of weight 0.7 and 0.3 in 432
@pytest.fixture(scope="module")
def bimodal():
  cs, ss = crystalFrame("432"), specimenFrame("1")
  c1, c2 = orientation(rotation.rand(rng=rng), cs, ss), orientation(rotation.rand(rng=rng), cs, ss)
  odf = 0.7 * unimodalODF(c1, halfwidth=8 * degree) + 0.3 * unimodalODF(c2, halfwidth=8 * degree)
  return SimpleNamespace(c1=c1, c2=c2, odf=odf)


def test_max_isOneValueAndOneOrientationAtTheHigherMode(bimodal):
  odf, c1 = bimodal.odf, bimodal.c1
  v, o = max(odf)
  assert isinstance(v, float) and isinstance(o, orientation)
  assert angle(o, c1) < 0.5 * degree and np.isclose(v, odf(c1), rtol=1e-3)
  assert odf.max()[0] == v


def test_max_numLocalGivesArraysHighestFirst(bimodal):
  v2, o2 = max(bimodal.odf, numLocal=2)
  assert len(v2) == 2 and v2[0] >= v2[1] and angle(o2[1], bimodal.c2) < 0.5 * degree


def test_max_ofTheHarmonicFormFindsTheSameMode(bimodal):
  v, _ = max(bimodal.odf)
  H = SO3FunHarmonic(bimodal.odf)
  vh, oh = max(H)
  assert angle(oh, bimodal.c1) < 0.5 * degree and np.isclose(vh, H(oh), rtol=1e-6) and np.isclose(vh, v, rtol=5e-2)


def test_min_isAValueAndItsOrientation(bimodal):
  vm, om = min(bimodal.odf)
  assert vm < 1e-3 and np.isclose(vm, bimodal.odf(om))


def test_max_onTheSphereFindsTheBinghamMode():
  s = S2FunHarmonic(rotate(S2FunBingham([-10.0, -4.0, 0.0]), rotation.byAxisAngle(yvector, 0.4)), bandwidth=30)
  vs, ps = max(s)
  assert angle(ps, rotation.byAxisAngle(yvector, 0.4) * zvector) < 0.3 * degree and np.isclose(vs, s(ps), rtol=1e-5)


def test_max_ofASymmetricSphericalFunctionIsInItsFrame():
  cs6 = crystalFrame("6/mmm")
  sf = S2FunHarmonic(S2DeLaValleePoussinKernel(halfwidth=20 * degree), cs6)
  vf, pf = max(sf)
  assert pf.frame is cs6 and abs(vector3d(pf.data).dot(zvector)) > np.cos(0.3 * degree)


def test_max_numLocalOfASymmetricFunctionGivesDistinctOrbits():
  # several maxima of a symmetric function are distinct orbits, not copies of one mode
  cs = crystalFrame("m-3m")
  adf = S2FunHarmonic(S2FunHandle(calcAxisDistribution(cs), cs, bandwidth=64, antipodal=True))
  va, pa = max(adf, numLocal=2)
  assert pa.size == 2 and np.all(np.diff(va) <= 0)
  for i in range(2):
    for j in range(i + 1, 2):
      assert angle(pa[i], pa[j]) > 2 * degree
  assert np.isclose(va[0], np.max(np.real(adf(equispacedS2Grid(resolution=0.5 * degree)))), rtol=1e-3)


@pytest.mark.parametrize("make", [lambda f: S1FunHarmonic(f, bandwidth=8), lambda f: S1FunHandle(f, bandwidth=8)], ids=["harmonic", "handle"])
def test_max_andMin_ofAFunctionOnTheCircleTakeEveryForm(make):
  # MATLAB's max(sF, c), max(sF1, sF2) and max(sF, 'numLocal', n) reach the class's own, the arguments kept
  f = make(lambda x: np.cos(3 * x) + 0.2 * np.sin(x))
  xs = np.linspace(0, 2 * pi, 7)
  for flat, name in ((max, "max"), (min, "min")):
    own = getattr(f, name)
    assert np.allclose(flat(f, 0.5)(xs), own(0.5)(xs), atol=1e-12)
    assert np.allclose(flat(f, 2 * f)(xs), own(2 * f)(xs), atol=1e-12)
    v, x = flat(f, numLocal=3)
    assert len(v) == 3 and np.allclose(v, own(numLocal=3)[0], atol=1e-12) and np.allclose(x, own(numLocal=3)[1], atol=1e-12)


def test_max_andMin_ofPlainNumbersAreThoseOfPython():
  assert max([3, 1, 2]) == 3 and min(4, 2) == 2


def test_clusters_are_the_greedy_ones():
  from mtex.functions.characteristics import _clusterOrientations
  cs = crystalFrame("622")
  c = orientation(rotation.rand(6, rng=rng), cs)
  o = orientation(rotation(c)[rng.integers(0, 6, 300)] * rotation.byAxisAngle(vector3d.rand(300, rng=rng), 3 * degree * rng.random(300)), cs)
  tol = 4 * degree
  ids = _clusterOrientations(o, tol)
  # each orientation joins the cluster of the first of their number within the tolerance of it
  ref, k = np.full(o.size, -1), 0
  for i in range(o.size):
    if ref[i] < 0:
      near = (o[i].angle(o) < tol) & (ref < 0)
      ref[near], ref[i], k = k, k, k + 1
  assert np.array_equal(ids, ref)


def test_clusters_among_the_moved_ones():
  from mtex.functions.characteristics import _clusterOrientations
  cs = crystalFrame("m-3m")
  o = orientation(rotation.byEuler(0.2 * np.arange(12), 0.3, 0.1), cs)
  moved = rotation.cat(rotation(o), rotation(o[3]) * rotation.byAxisAngle(zvector, 1 * degree))
  o = orientation(moved, cs)
  # the twelve are apart, the thirteenth moved next to the fourth
  assert np.array_equal(_clusterOrientations(o, 2 * degree, among=[12]), _clusterOrientations(o, 2 * degree))
  assert _clusterOrientations(o, 2 * degree, among=[12])[12] == 3


def test_grid_basins_climb_to_peaks():
  from mtex.functions.characteristics import _gridBasins
  cs = crystalFrame("222")
  odf = unimodalODF(orientation(rotation.rand(3, rng=rng), cs), halfwidth=10 * degree)
  g = equispacedSO3Grid(cs, odf.SS, resolution=5 * degree)
  v = np.real(odf(g)).ravel()
  peak = _gridBasins(g, v, 1.5 * g.res)
  # every node reaches a peak no lower than itself, a peak is its own peak and beats its neighbours
  assert np.all(v[peak] >= v) and np.array_equal(peak[peak], peak)
  top = np.unique(peak)
  assert all(np.all(v[p] >= v[angle(g, g[p]) < 1.5 * g.res]) for p in top)
  assert np.argmax(v) in top


# two cubic modes of weight 0.6 and 0.4 at known centres
def _cubicModes(halfwidth):
  cs = crystalFrame("m-3m")
  c1, c2 = orientation(rotation.byEuler(0, 0, 0, "Bunge"), cs), orientation(rotation.byEuler(30 * degree, 40 * degree, 50 * degree, "Bunge"), cs)
  odf = 0.6 * unimodalODF(c1, halfwidth=halfwidth) + 0.4 * unimodalODF(c2, halfwidth=halfwidth)
  return SimpleNamespace(cs=cs, c1=c1, c2=c2, odf=odf)


def test_calcComponents_seededGivesOneCentreIdPerSeed():
  t = _cubicModes(6 * degree)
  seed = orientation(rotation.rand(20, rng=rng), t.cs, t.c1.SS)
  modes, weights, centerId = calcComponents(t.odf, seed=seed, radius=5 * degree, exact=True)
  assert len(weights) == modes.size and np.all((0 <= centerId) & (centerId < modes.size)) and len(centerId) == seed.size


def test_calcComponents_ofATinyRadiusCarryNoWeight():
  t = _cubicModes(6 * degree)
  seed = orientation(rotation.rand(20, rng=rng), t.cs, t.c1.SS)
  tm, tw, _ = calcComponents(t.odf, seed=seed, radius=1e-4 * degree, exact=True)
  assert len(tw) == tm.size and np.sum(tw) < 1e-4


def test_calcComponents_weightsSumToOneOverTheFullRadius():
  t = _cubicModes(6 * degree)
  assert abs(np.sum(calcComponents(t.odf, radius=pi, exact=True)[1]) - 1) < 1e-10


def test_calcComponents_findsTheKnownModesAndWeights():
  # two modes at the centres with weights 0.6 and 0.4
  t = _cubicModes(7 * degree)
  modes, weights, centerId = calcComponents(t.odf)
  assert modes.size == 2
  assert np.min(angle(modes, t.c1)) < 1 * degree and np.min(angle(modes, t.c2)) < 1 * degree
  assert np.max(np.abs(np.sort(weights)[::-1] - [0.6, 0.4])) < 0.02 and np.all(centerId >= 0)


def test_calcComponents_ofARadialBasisFunctionAreItsOwnCentres():
  t = _cubicModes(7 * degree)
  rb = unimodalODF(orientation(rotation.cat(rotation(t.c1), rotation(t.c2)), t.cs), halfwidth=7 * degree, weights=[0.6, 0.4])
  om, ow, _ = rb.calcComponents()
  assert om.size == 2 and np.max(np.abs(np.sort(ow)[::-1] - [0.6, 0.4])) < 0.02


def test_components_freeze_light_seeds_on_flat_stretches():
  cs = crystalFrame("m-3m")
  c1, c2 = orientation(rotation.byEuler(0, 0, 0, "Bunge"), cs), orientation(rotation.byEuler(30 * degree, 40 * degree, 50 * degree, "Bunge"), cs)
  odf = 0.6 * unimodalODF(c1, halfwidth=7 * degree) + 0.4 * unimodalODF(c2, halfwidth=7 * degree)
  seed = discreteSample(odf, 1000, rng=rng)
  # the seeds stopped early carry little weight: the two modes keep theirs to a few tenths of a percent
  _, w, _ = calcComponents(odf, seed=seed)
  _, we, _ = calcComponents(odf, seed=seed, exact=True)
  assert len(w) == 2 and np.max(np.abs(w - we[:2])) < 0.005


def test_calcAngleDistribution_ofTwoSymmetriesHasMeanOne():
  cs = crystalFrame("mmm")
  p, omega = calcAngleDistribution(cs, cs)
  assert np.isclose(np.mean(p), 1)


def test_calcAngleDistribution_ofCubicSymmetryIsMackenzies():
  # Mackenzie's maximum and mean angle of the cubic one
  pc, oc = calcAngleDistribution(crystalFrame("m-3m"), crystalFrame("m-3m"))
  assert np.isclose(oc[-1], 62.8 * degree, atol=0.01 * degree)
  assert np.isclose(np.sum(oc * pc) / np.sum(pc), 40.73 * degree, rtol=1e-2)


def test_calcAngleDistribution_ofTheUniformODFIsTheUniformOne():
  cs = crystalFrame("mmm")
  p, omega = calcAngleDistribution(cs, cs)
  pf, of = calcAngleDistribution(uniformODF(cs, cs), resolution=2 * degree)
  assert np.allclose(of, omega) and np.mean(np.abs(pf - p)) < 0.01


def test_calcAngleDistribution_ofRandomMisorientationsIsAHistogramInPercent():
  cs = crystalFrame("mmm")
  p, omega = calcAngleDistribution(cs, cs)
  h, mid = calcAngleDistribution(orientation.rand(20000, cs, cs, rng=rng))
  assert np.isclose(np.sum(h), 100) and np.mean(np.abs(h / 100 * len(h) - np.interp(mid, omega, p))) < 0.1


def test_calcMIndex_ofTheUniformODFIsZero():
  assert calcMIndex(uniformODF(crystalFrame("mmm")), resolution=2 * degree) < 2e-2


def test_calcMDF_ofTheUniformODFIsUniform():
  cs = crystalFrame("mmm")
  m = calcMDF(uniformODF(cs))
  assert m.antipodal and np.isclose(mean(m), 1) and m.CS is cs and m.SS is cs
  assert np.max(np.abs(m(rotation.rand(20, rng=rng)) - 1)) < 1e-6


def test_calcMDF_ofASharpODFConcentratesAtTheIdentity():
  # a large M-index
  odf = unimodalODF(orientation(rotation.rand(rng=rng), crystalFrame("mmm")), halfwidth=5 * degree)
  assert calcMIndex(odf, resolution=2 * degree) > 0.5
  m2 = calcMDF(odf, odf)
  assert m2(rotation.id) > 10 * m2(rotation.rand(rng=rng))


def test_calcMDF_ofTwoODFsLivesBetweenTheirCrystalFrames():
  cs, cs2 = crystalFrame("mmm"), crystalFrame("4/mmm")
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs), halfwidth=5 * degree)
  odf2 = unimodalODF(orientation(rotation.rand(rng=rng), cs2), halfwidth=15 * degree)
  m12 = calcMDF(odf, odf2)
  assert m12.CS is cs2 and m12.SS is cs and not m12.antipodal and np.isclose(mean(m12), 1)


# an orthorhombic misorientation ODF, part uniform, and its axis distribution over all angles
@pytest.fixture(scope="module")
def axes():
  cs = crystalFrame("mmm")
  h = vector3d(equispacedS2Grid(resolution=20 * degree, antipodal=True).data)
  odf = 0.4 * uniformODF(cs, cs, antipodal=True) + 0.6 * unimodalODF(orientation(rotation.byEuler(20 * degree, 30 * degree, 10 * degree, "Bunge"), cs, cs, antipodal=True), halfwidth=15 * degree)
  return SimpleNamespace(cs=cs, h=h, odf=odf, whole=calcAxisDistribution(odf, h, resolution=2 * degree))


def test_calcAxisDistribution_defaultWindowIsAllAngles(axes):
  assert np.array_equal(axes.whole, calcAxisDistribution(axes.odf, axes.h, resolution=2 * degree, minAngle=0, maxAngle=np.inf))


def test_calcAxisDistribution_twoWindowsSplitTheWhole(axes):
  odf, h, whole = axes.odf, axes.h, axes.whole
  lower = calcAxisDistribution(odf, h, resolution=2 * degree, maxAngle=60 * degree)
  upper = calcAxisDistribution(odf, h, resolution=2 * degree, minAngle=60 * degree)
  assert np.max(np.abs(lower + upper - whole)) / np.max(whole) < 5e-3
  tol = 1e-3 * np.max(whole)
  assert np.all(lower <= whole + tol) and np.all(upper <= whole + tol) and np.any(lower > tol) and np.any(upper > tol)


def test_calcAxisDistribution_ofTheUniformODFAgreesWithAFineRule(axes):
  # the uniform ODF against the integral of the uniform density on a fine rule
  h = axes.h
  u = uniformODF(axes.cs, axes.cs, antipodal=True)
  fine = calcAxisDistribution(u, h, resolution=0.5 * degree)
  coarse = calcAxisDistribution(u, h, resolution=2 * degree)
  assert np.max(np.abs(coarse - fine)) / np.max(fine) < 3e-2
  win = calcAxisDistribution(u, h, resolution=2 * degree, minAngle=20 * degree, maxAngle=40 * degree)
  winf = calcAxisDistribution(u, h, resolution=0.5 * degree, minAngle=20 * degree, maxAngle=40 * degree)
  assert np.max(np.abs(win - winf)) / np.max(winf) < 3e-2


def test_calcAxisDistribution_ofAnInvertedWindowIsEmpty(axes):
  assert np.max(np.abs(calcAxisDistribution(axes.odf, axes.h, resolution=2 * degree, minAngle=40 * degree, maxAngle=20 * degree))) < 1e-10


def test_calcAxisDistribution_isASphericalFunctionOfMeanOneInTheCrystalFrame(axes):
  u = uniformODF(axes.cs, axes.cs, antipodal=True)
  adf = calcAxisDistribution(u, resolution=2.5 * degree, bandwidth=32)
  assert adf.frame is axes.cs and adf.antipodal and np.isclose(mean(adf), 1, rtol=2e-2)


# a 432 unimodal ODF of halfwidth 10 degree and 2000 orientations drawn from it
@pytest.fixture(scope="module")
def sampled():
  cs, ss = crystalFrame("432"), specimenFrame("1")
  c = orientation(rotation.rand(rng=rng), cs, ss)
  odf = unimodalODF(c, halfwidth=10 * degree)
  o = discreteSample(odf, 2000, rng=rng)
  return SimpleNamespace(cs=cs, ss=ss, c=c, odf=odf, o=o, share=volume(odf, c, 20 * degree))


def test_discreteSample_ofAnODFIsOrientationsInItsFrame(sampled):
  o = sampled.o
  assert isinstance(o, orientation) and o.size == 2000 and o.CS is sampled.cs


def test_discreteSample_shareNearTheCentreIsTheVolumeThere(sampled):
  assert abs(volume(sampled.o, sampled.c, 20 * degree) - sampled.share) < 0.05


def test_calcDensity_withTheSameKernelIsTheODFConvolvedWithIt(sampled):
  c = sampled.c
  est = calcDensity(sampled.o, halfwidth=10 * degree)
  assert np.isclose(est(c), conv(sampled.odf, SO3DeLaValleePoussinKernel(halfwidth=10 * degree))(c), rtol=0.15)


def test_discreteSample_ofACompositionOfKernelForms(sampled):
  mix = 0.5 * sampled.odf + 0.5 * uniformODF(sampled.cs, sampled.ss)
  om = discreteSample(mix, 1000, rng=rng)
  assert om.size == 1000 and abs(volume(om, sampled.c, 20 * degree) - 0.5 * sampled.share) < 0.06


def test_discreteSample_ofAHarmonicFunction(sampled):
  oh = discreteSample(SO3FunHarmonic(sampled.odf), 1000, rng=rng, resolution=3 * degree)
  assert abs(volume(oh, sampled.c, 20 * degree) - sampled.share) < 0.06


def test_discreteSample_onTheSphere():
  s = S2FunHarmonic(S2DeLaValleePoussinKernel(halfwidth=10 * degree))
  d = discreteSample(s, 1000, rng=rng)
  assert d.size == 1000 and np.count_nonzero(d.angle(zvector) < 20 * degree) > 900
  assert discreteSample(S2FunHarmonic(s, antipodal=True), 5, rng=rng).antipodal


def test_discreteSample_ofASymmetricDensityCoversTheWholeSphere():
  # the sector nodes spread by the group
  adf = calcAxisDistribution(crystalFrame("m-3m"))
  da = discreteSample(adf, 4000, rng=rng)
  assert da.antipodal and np.count_nonzero(da.z < 0) > 100 and np.mean(np.real(adf(da))) > 1.05


# 2000 orientations of a trigonal texture, half uniform
@pytest.fixture(scope="module")
def trigonalSample():
  cs = crystalFrame("321")
  model = 0.5 * uniformODF(cs) + 0.5 * unimodalODF(orientation(rotation.byEuler(35 * degree, 45 * degree, 0.0, "Bunge"), cs), halfwidth=10 * degree)
  return SimpleNamespace(model=model, ori=discreteSample(model, 2000, rng=rng))


def test_calcKernel_everyRuleGivesAUsableWidthTheCrossValidationTheNarrowest(trigonalSample):
  ori = trigonalSample.ori
  hw = {m: calcKernel(ori, m, rng=rng).halfwidth() / degree for m in ("klcv", "ruleOfThumb", "magicRule")}
  assert all(2 <= v <= 40 for v in hw.values()) and hw["klcv"] <= hw["magicRule"]


def test_calcKernel_defaultGivesADensityCloseToTheModel(trigonalSample):
  ori = trigonalSample.ori
  assert calcError(calcDensity(ori, kernel=calcKernel(ori, rng=rng)), trigonalSample.model) < 0.3


def test_calcKernel_rejectsAnUnknownRule(trigonalSample):
  with pytest.raises(ValueError):
    calcKernel(trigonalSample.ori, "nothing")


def test_centerSpecimen_bringsBackAnOrthotropicTextureMountedAskew():
  cubic, ss = crystalFrame("m-3m"), specimenFrame("222")
  odf = unimodalODF(orientation(rotation.byEuler(np.array([135, 60]) * degree, np.array([45, 54.73]) * degree, np.array([120, 45]) * degree, "Bunge"), cubic, ss), halfwidth=12 * degree)
  mount = rotation.byEuler(15 * degree, 12 * degree, -5 * degree, "Bunge")
  os = discreteSample(odf, 1000, rng=rng)
  tilted = calcDensity(orientation(mount * rotation(os), cubic), halfwidth=10 * degree)
  _, correction, v1, v2 = centerSpecimen(tilted)
  assert abs(v1.dot(v2)) < 1e-6
  assert angle(mount, correction.inv()) < 5 * degree


# the compiled sampler of a radial basis function draws from the same distribution as the
# NumPy form: the moments up to degree 4 of both samples agree to their sampling error, with
# symmetry on both sides and a uniform part, and for an exchange symmetric function
def test_rbf_sampler_kernel_against_numpy():
  from mtex import setMTEXpref
  rng = np.random.default_rng(11)
  cs, ss = crystalFrame("321"), specimenFrame("222")
  mixed = 0.7 * unimodalODF(orientation(rotation.rand(3, rng=rng), cs, ss), halfwidth=12 * degree) + 0.3 * uniformODF(cs, ss)
  exchange = unimodalODF(orientation(rotation.rand(2, rng=rng), cs, cs, antipodal=True), halfwidth=15 * degree)
  assert exchange.antipodal
  for f in (mixed, exchange):
    moments = {}
    for kernels in ("numba", "numpy"):
      setMTEXpref("kernels", kernels)
      o = discreteSample(f, 200000, rng=np.random.default_rng(3))
      moments[kernels] = SO3FunHarmonic.quadrature(o, np.ones(o.size), bandwidth=4).fhat
    setMTEXpref("kernels", "numba")
    assert np.max(np.abs(moments["numba"] - moments["numpy"])) < 0.05
