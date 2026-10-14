# the pole figure container, its arithmetic, the simulation from an ODF, the quadrature
# weights, the errors and the corrections (docs/plans/functions.md P4a: P1 to P8 after the
# Julia tests; check_calcPoleFigureSuperposition on the container)
from types import SimpleNamespace

import numpy as np
import pytest

from mtex import (calcError, calcErrorPF, calcNormalization, calcPoleFigure, calcQuadratureWeights, correct, crystalFrame, degree, equispacedS2Grid, isOutlier, max, mean, Miller, min, noisepf, orientation, PoleFigure, radon, refine, regularS2Grid, rotation, S2FunHarmonic, SchulzDefocusingKernel, specimenFrame, uniformODF, unimodalODF, union, unique, vector3d, zvector)
from mtex.functions.polefigure import _poleDensity

rng = np.random.default_rng(3)


# a single cubic pole figure on a 10 degree half grid, a superposition and a data set of both
def _containers():
  cs = crystalFrame("m-3m")
  r = regularS2Grid(resolution=10 * degree, antipodal=True)
  h = Miller(1, 0, 0, cs)
  hs = Miller([1, 0], [1, 1], [0, 1], cs)
  pf = PoleFigure(h, r, np.ones(r.shape))
  both = PoleFigure([h, hs], r, [np.ones(r.size), 2 * np.ones(r.size)], superposition=[None, [0.52, 1.23]])
  return SimpleNamespace(cs=cs, ss=specimenFrame.default, r=r, h=h, hs=hs, pf=pf, both=both)


def test_PoleFigure_holdsOnePoleFigureOnItsGrid():
  t = _containers()
  pf, r = t.pf, t.r
  assert pf.numPF == 1 and len(pf) == 1 and pf.allI[0].shape == r.shape
  assert pf.CS is t.cs and pf.SS is t.ss and pf.frame is t.ss
  # the plotting convention is that of the specimen directions
  assert pf.how2plot is pf.allR[0].how2plot
  assert pf.antipodal and pf.allH[0].size == 1 and pf.c[0] == [1.0]


def test_PoleFigure_specimenDirectionsAreAxesUnlessComplete():
  h = Miller(1, 0, 0, crystalFrame("m-3m"))
  d = regularS2Grid(resolution=20 * degree)
  assert PoleFigure(h, d, np.ones(d.shape)).antipodal
  assert not PoleFigure(h, d, np.ones(d.shape), complete=True).antipodal


def test_PoleFigure_superpositionDefaultsToTheMean():
  t = _containers()
  # a superposition gets one coefficient per direction, the mean by default
  pfs = PoleFigure(t.hs, t.r, np.ones(t.r.size))
  assert pfs.numPF == 1 and pfs.allH[0].size == 2 and np.allclose(pfs.c[0], [0.5, 0.5])
  assert np.allclose(PoleFigure(t.hs, t.r, np.ones(t.r.size), superposition=[0.52, 1.23]).c[0], [0.52, 1.23])


def test_PoleFigure_rejectsWhatDoesNotFit():
  t = _containers()
  h, r = t.h, t.r
  with pytest.raises(ValueError):
    PoleFigure(h, r, np.ones(3))
  with pytest.raises(ValueError):
    PoleFigure(h, r, np.ones(r.size), superposition=[1.0, 1.0])
  with pytest.raises(ValueError):
    PoleFigure(vector3d(1, 0, 0), r, np.ones(r.size))


def test_PoleFigure_dataSetHoldsSeveralPoleFigures():
  # MATLAB's numPF, select and cat
  t = _containers()
  both, r = t.both, t.r
  assert both.numPF == 2 and both.CS is t.cs and both.SS is t.ss
  assert both.allR[0].shape == r.shape and both.allI[1].shape == r.shape
  assert np.allclose(both.c[1], [0.52, 1.23]) and both.c[0] == [1.0]
  assert both.select(1).numPF == 1 and both[1].allH[0].size == 2 and both[[0, 1]].numPF == 2
  assert [p.numPF for p in both] == [1, 1]
  assert PoleFigure.cat(both, t.pf).numPF == 3
  assert both.intensities.size == 2 * r.size and both.h.size == 3 and both.r.size == 2 * r.size


def test_PoleFigure_positionalFramesLandOnDirectionsThatNameNone():
  t = _containers()
  plain = PoleFigure(vector3d(1, 0, 0), vector3d(t.r.data), np.ones(t.r.size), t.cs, specimenFrame("222"))
  assert plain.CS is t.cs and plain.SS.numSym() == 4


def test_PoleFigure_maskKeepsWhatItSaysPerPoleFigure():
  both = _containers().both
  keep = both.r.theta < 60 * degree
  sub = both[keep]
  assert sub.intensities.size == np.count_nonzero(keep) and sub.numPF == 2


def test_PoleFigure_intensitiesAreAssignedAsOneVectorOrOneNumber():
  t = _containers()
  both = t.both
  both.intensities = np.arange(2 * t.r.size)
  assert both.allI[1].ravel()[0] == t.r.size
  both.intensities = 3
  assert np.all(both.intensities == 3)


def test_PoleFigure_reprNamesTheGrid():
  both = _containers().both
  assert "PoleFigure" in repr(both) and "36 × 10 points" in repr(both)


# a real antipodal function of bandwidth 5 and 400 random axes
def _antipodalFunction():
  return S2FunHarmonic(rng.standard_normal(36) + 1j * rng.standard_normal(36), real=True, antipodal=True)


def test_calcQuadratureWeights_areExactOnARegularHalfGrid():
  # on a regular grid the weights are exact: they integrate every harmonic they see
  r = regularS2Grid(resolution=10 * degree, antipodal=True)
  w = calcQuadratureWeights(r)
  assert np.isclose(np.sum(w), 1) and np.all(w > 0) and w.shape == r.shape
  f = _antipodalFunction()
  assert abs(np.sum(w * np.real(f(r))) - mean(f)) < 1e-7
  assert r.calcQuadratureWeights().shape == r.shape


def test_calcQuadratureWeights_areExactOnARegularFullGrid():
  d = regularS2Grid(resolution=6 * degree)
  wd = calcQuadratureWeights(d)
  g = S2FunHarmonic(rng.standard_normal(49) + 1j * rng.standard_normal(49), real=True)
  assert np.isclose(np.sum(wd), 1)
  assert abs(np.sum(wd * np.real(g(d))) - mean(g)) < 1e-7


def test_calcQuadratureWeights_voronoiWeightsOfScatteredPointsArePositiveAndClose():
  f = _antipodalFunction()
  v = vector3d.rand(400, antipodal=True, rng=rng)
  wv = calcQuadratureWeights(v)
  assert np.isclose(np.sum(wv), 1) and np.all(wv > 0)
  assert abs(np.sum(wv * np.real(f(v))) - mean(f)) < 1e-2


def test_calcQuadratureWeights_leastNormWeightsOfScatteredPointsAreExact():
  f = _antipodalFunction()
  v = vector3d.rand(400, antipodal=True, rng=rng)
  wh = calcQuadratureWeights(v, method="harmonic")
  assert np.isclose(np.sum(wh), 1)
  assert abs(np.sum(wh * np.real(f(v))) - mean(f)) < 1e-8
  # equal weights would not do it
  assert abs(np.sum(np.real(f(v))) / v.size - mean(f)) > 1e-3


# a group without an element sending h to -h, so the pole figure is direction valued
@pytest.fixture(scope="module")
def trigonal():
  cs, ss = crystalFrame("3"), specimenFrame.default
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=15 * degree)
  r = regularS2Grid(resolution=10 * degree, antipodal=True)
  h23 = Miller([1, 0], [0, 1], [-1, -1], [1, 1], cs)
  h = [Miller(1, 0, 0, cs), h23]
  return SimpleNamespace(cs=cs, ss=ss, odf=odf, r=r, h23=h23, h=h, pfs=calcPoleFigure(odf, h, r))


def test_calcPoleFigure_superpositionIsTheMeanOfTheComponents(trigonal):
  # check_calcPoleFigureSuperposition: the default is the mean of the components
  t = trigonal
  pfs = t.pfs
  assert pfs.numPF == 2 and np.allclose(pfs.c[1], [0.5, 0.5]) and pfs.c[0] == [1.0]
  ref = 0.5 * _poleDensity(t.odf, t.h23[0], pfs.allR[1]) + 0.5 * _poleDensity(t.odf, t.h23[1], pfs.allR[1])
  assert np.max(np.abs(pfs.allI[1] - ref)) < 1e-10


def test_calcPoleFigure_usesCoefficientsHandedOverAsTheyAre(trigonal):
  t = trigonal
  odf, h, r, pfs = t.odf, t.h, t.r, t.pfs
  pfe = calcPoleFigure(odf, h, r, superposition=[None, [0.52, 1.23]])
  assert np.allclose(calcPoleFigure(odf, h[0], r, superposition=2.0).allI[0], 2 * pfs.allI[0])
  assert np.allclose(calcPoleFigure(odf, [h[0]], r, superposition=[2.0]).allI[0], 2 * pfs.allI[0])
  assert PoleFigure(h[0], r, np.ones(r.size), superposition=2.0).c[0] == [2.0]
  ref2 = 0.52 * _poleDensity(odf, t.h23[0], pfs.allR[1]) + 1.23 * _poleDensity(odf, t.h23[1], pfs.allR[1])
  assert np.max(np.abs(pfe.allI[1] - ref2)) < 1e-10


def test_calcPoleFigure_plainPoleFigureIsItsPoleDensityOnTheGridAskedFor(trigonal):
  t = trigonal
  assert t.pfs.allI[0].shape == t.r.shape
  assert np.allclose(t.pfs.allI[0], _poleDensity(t.odf, t.h[0], t.pfs.allR[0]))


def test_calcPoleFigure_measurementSeesTheEvenPart(trigonal):
  # the pole figure of r and of -r agree
  t = trigonal
  P = radon(t.odf, t.h[0])
  vs = vector3d.rand(20, rng=rng)
  assert not P.antipodal
  assert np.allclose(_poleDensity(t.odf, t.h[0], vector3d(vs, antipodal=True)), np.real(P(vs) + P(-vs)) / 2)


def test_calcPoleFigure_poleDensitiesHaveMeanOne(trigonal):
  # up to what the 10 degree rule integrates of a 15 degree kernel
  t = trigonal
  assert abs(mean(t.pfs)[0] - 1) < 1e-3
  assert np.allclose(calcPoleFigure(uniformODF(t.cs, t.ss), t.h[0], t.r).allI[0], 1)


def test_calcPoleFigure_millerListGivesOnePerElementAndGridKeywordTheRegularGrid(trigonal):
  t = trigonal
  three = calcPoleFigure(t.odf, Miller([1, 0, 0], [0, 1, 0], [0, 0, 1], t.cs), t.r)
  assert three.numPF == 3 and three.allH[2].size == 1
  assert calcPoleFigure(t.odf, t.h[0], resolution=15 * degree).allR[0].shape == (24, 7)


def test_calcPoleFigure_takesOneSetOfSpecimenDirectionsPerPoleFigure(trigonal):
  r2 = [regularS2Grid(resolution=10 * degree, antipodal=True), regularS2Grid(resolution=15 * degree, antipodal=True)]
  assert [p.intensities.size for p in calcPoleFigure(trigonal.odf, trigonal.h, r2)] == [x.size for x in r2]


def test_calcPoleFigure_predictsOnTheDirectionsOfAMeasurementEitherOrder(trigonal):
  t = trigonal
  odf, pfs = t.odf, t.pfs
  sim = calcPoleFigure(odf, pfs)
  assert sim.numPF == 2 and np.allclose(sim.c[1], pfs.c[1]) and np.allclose(sim.allI[1], pfs.allI[1])
  assert np.allclose(calcPoleFigure(pfs, odf).allI[0], pfs.allI[0]) and np.allclose(odf.calcPoleFigure(t.h, t.r).allI[0], pfs.allI[0])


# a cubic ODF of halfwidth 20 degree and its 100 pole figure on a 10 degree half grid
@pytest.fixture(scope="module")
def cubic():
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=20 * degree)
  r = regularS2Grid(resolution=10 * degree, antipodal=True)
  pf = calcPoleFigure(odf, Miller(1, 0, 0, cs), r)
  return SimpleNamespace(cs=cs, ss=ss, odf=odf, r=r, pf=pf, I=pf.allI[0])


def test_PoleFigure_arithmeticActsOnTheIntensities(cubic):
  pf, I = cubic.pf, cubic.I
  assert np.allclose((pf * 2).allI[0], 2 * I) and np.allclose((2 * pf).allI[0], 2 * I)
  assert np.allclose((pf + 1).allI[0], I + 1) and np.allclose((1 - pf).allI[0], 1 - I)
  assert np.all((pf - pf).allI[0] == 0) and np.allclose((-pf).allI[0], -I)
  assert np.isclose(max(max(pf, 1))[0], np.maximum(np.max(I), 1)) and min(max(pf, 1))[0] >= 1
  assert np.allclose((pf / pf).allI[0], 1) and np.allclose((pf * pf).allI[0], I * I)
  assert np.allclose(pf.scale(3).allI[0], 3 * I)


def test_PoleFigure_normalizeGivesMeanOne(cubic):
  pf = cubic.pf
  assert abs(mean(pf)[0] - 1) < 1e-4
  assert abs(mean((pf * 7).normalize())[0] - 1) < 1e-12


def test_PoleFigure_dataSetTakesOneFactorForAllOrOnePerPoleFigure(cubic):
  pf, I = cubic.pf, cubic.I
  both = PoleFigure.cat(pf, pf * 3)
  assert np.allclose(mean(both - 1), [mean(pf)[0] - 1, mean(pf * 3)[0] - 1])
  assert np.allclose(max(both), [np.max(I), 3 * np.max(I)])
  assert np.allclose((both / [1, 3]).allI[1], I)
  with pytest.raises(ValueError):
    both / [1, 2, 3]


def test_PoleFigure_maskAssignsOverAllPoleFiguresAtOnce(cubic):
  both = PoleFigure.cat(cubic.pf, cubic.pf * 3)
  clipped = both - 1
  clipped[clipped.intensities < 0] = 0
  assert np.allclose(clipped.intensities, np.maximum(both.intensities - 1, 0))


def test_PoleFigure_rotateTurnsItsPoleFigures(cubic):
  I = cubic.I
  rot = rotation.byAxisAngle(zvector, 30 * degree)
  turned = cubic.pf.rotate(rot)
  assert np.allclose(turned.allI[0], I) and turned.antipodal and turned.SS is cubic.ss
  assert np.max(np.abs(calcPoleFigure(cubic.odf.rotate(rot), Miller(1, 0, 0, cubic.cs), turned.allR[0]).allI[0] - I)) < 1e-5


def test_PoleFigure_rotateByNoElementOfTheSpecimenGroupStripsIt(cubic):
  sym = PoleFigure(Miller(1, 0, 0, cubic.cs), cubic.r, cubic.I, specimenFrame("222"))
  assert sym.rotate(rotation.byAxisAngle(zvector, np.pi)).SS.numSym() == 4
  assert sym.rotate(rotation.byAxisAngle(zvector, 30 * degree)).SS.numSym() == 1


# three cubic pole figures of one ODF and a second ODF they do not come from
@pytest.fixture(scope="module")
def measured():
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=20 * degree)
  r = regularS2Grid(resolution=10 * degree, antipodal=True)
  pfs = calcPoleFigure(odf, Miller([1, 1], [0, 1], [0, 1], cs), r)
  other = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=20 * degree)
  return SimpleNamespace(odf=odf, pfs=pfs, other=other)


def test_calcNormalization_carriesTheMassOfTheFirstOntoTheSecond(measured):
  pfs = measured.pfs
  assert np.isclose(calcNormalization(pfs[0], pfs[0] * 3)[0], 1 / 3)
  assert np.allclose(calcNormalization(pfs, pfs * 2), [0.5, 0.5])


def test_calcError_ofTheODFTheDataCameFromIsZeroInEveryMode(measured):
  for mode in ("relative", "RP", "l1", "l2"):
    assert np.all(calcError(measured.pfs, measured.odf, mode) < 1e-9)


@pytest.mark.parametrize("mode", [None, "relative", "RP", "rp", "l1", "L1", "l2", "L2"])
def test_calcError_takesTheODFFirstOrSecondAndTheModeInAnyCase(measured, mode):
  # MATLAB's calcError(odf, pf, 'RP') turns to calcError(pf, odf, 'RP'), its flags in any case
  pfs, other = measured.pfs, measured.other
  args = () if mode is None else (mode,)
  canonical = () if mode is None else ({"rp": "RP"}.get(mode.lower(), mode.lower()),)
  e = calcError(pfs, other, *canonical)
  assert np.array_equal(calcError(other, pfs, *args), e) and np.array_equal(other.calcError(pfs, *args), e)
  assert np.array_equal(calcError(pfs, other, *args), e)


def test_calcError_ofADifferentODFIsPositive(measured):
  assert np.all(calcError(measured.pfs, measured.other) > 0.1)


def test_calcError_isInvariantUnderAScalingOfTheData(measured):
  pfs = measured.pfs
  assert np.allclose(calcError(pfs[0] * 5, measured.odf), calcError(pfs[0], measured.odf), atol=1e-12)


def test_calcErrorPF_livesOnTheSameDirections(measured):
  # the RP mode only where it is defined
  pfs, other = measured.pfs, measured.other
  e = calcErrorPF(pfs[0], other)
  assert e.intensities.size == pfs[0].intensities.size and np.all(e.intensities >= 0)
  assert calcErrorPF(pfs[0], other, "RP").intensities.size <= pfs[0].intensities.size


def test_calcError_rejectsAnUnknownMode(measured):
  with pytest.raises(ValueError):
    calcError(measured.pfs, measured.other, "l3")


def test_PoleFigure_normalizeAgainstAnODFKeepsTheMean(measured):
  pfs = measured.pfs
  assert abs(mean((pfs[0] * 4).normalize(measured.odf))[0] - mean(pfs[0])[0]) < 1e-6


def test_calcError_betweenTwoODFsGoesThroughTheSameName(measured):
  odf = measured.odf
  assert calcError(odf, odf) < 1e-12 and calcError(odf, measured.other, "l2") > 0


# a cubic 100 pole figure with orthorhombic specimen symmetry and a defocusing curve
@pytest.fixture(scope="module")
def orthorhombic():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=20 * degree)
  r = regularS2Grid(resolution=10 * degree, antipodal=True, frame=ss)
  pf = calcPoleFigure(odf, Miller(1, 0, 0, cs), r)
  d = pf._rebuild([0.5 + 0.5 * np.cos(r.theta) ** 2])
  return SimpleNamespace(cs=cs, ss=ss, odf=odf, r=r, pf=pf, I=pf.allI[0], d=d)


def test_correct_takesOffABackgroundAndADefocusingCurve(orthorhombic):
  t = orthorhombic
  bg = t.pf._rebuild([np.full(t.I.shape, 0.3)])
  raw = t.pf * t.d + bg
  assert np.max(np.abs(correct(raw, background=bg, defocusing=t.d).allI[0] - t.I)) < 1e-12
  assert np.max(np.abs(correct(raw, background=0.3, defocusing=t.d).allI[0] - t.I)) < 1e-12


def test_correct_interpolatesACoarserDefocusingCurveInThePolarAngle(orthorhombic):
  t = orthorhombic
  th = np.arange(0, 91, 10) * degree
  curve = PoleFigure(Miller(1, 0, 0, t.cs), vector3d.byPolar(th, np.zeros(th.size), antipodal=True), 0.5 + 0.5 * np.cos(th) ** 2, t.ss)
  assert np.max(np.abs(correct(t.pf * t.d, defocusing=curve).allI[0] - t.I)) < 1e-3


def test_correct_takesADefocusingBackgroundOffTheDefocusingFirst(orthorhombic):
  t = orthorhombic
  assert np.max(np.abs(correct(t.pf * t.d, defocusing=t.d + 0.2, defocusingBackground=0.2).allI[0] - t.I)) < 1e-12


def test_isOutlier_findsASpikeAndNotTheRest(orthorhombic):
  t = orthorhombic
  spiked = t.pf._rebuild([t.I.ravel().copy()])
  spiked.allI[0].flat[100] += 20 * np.max(t.I)
  out = isOutlier(spiked)
  assert out[100] and np.count_nonzero(out) < 5


def test_noisepf_keepsTheMeanAndTheShape(orthorhombic):
  # Poisson noise
  pf = orthorhombic.pf
  noisy = noisepf(pf, 100, rng=rng)
  assert noisy.allI[0].shape == orthorhombic.I.shape and abs(mean(noisy)[0] / 100 - mean(pf)[0]) < 0.05
  assert min(noisepf(pf, 10, background=5, nonnegative=True, rng=rng))[0] >= 0


def test_unique_mergesEquivalentSpecimenDirectionsKeepingThePoleDensity(orthorhombic):
  t = orthorhombic
  u = unique(t.pf)
  assert u.intensities.size == 91                      # the 222 sector of the 360 point half grid
  assert np.max(np.abs(u.allI[0] - _poleDensity(t.odf, t.pf.allH[0], u.allR[0]))) < 1e-7


def test_union_joinsPoleFiguresOfTheSameCrystalDirections(orthorhombic):
  t = orthorhombic
  keep = np.zeros(t.r.size, dtype=bool)
  keep[:180] = True
  half1, half2 = t.pf[keep], t.pf[~keep]
  m = union(half1, half2, calcPoleFigure(t.odf, Miller(1, 1, 1, t.cs), t.r))
  assert m.numPF == 2 and m.allI[0].size == 360 and union(half1, None).numPF == 1


# a flat cubic pole figure up to 80 degree tilt and the Schulz defocusing kernel
def _schulz():
  cs = crystalFrame("m-3m")
  r = regularS2Grid(resolution=10 * degree, maxTheta=80 * degree, antipodal=True)
  pf = PoleFigure(Miller(1, 0, 0, cs), r, np.ones(r.shape))
  return pf, SchulzDefocusingKernel(r=100, w=0.5)


def test_SchulzDefocusingKernel_isOneAtThePoleGrowsWithTheTiltAndDivergesAt90():
  _, psi = _schulz()
  assert psi.eval(1.0) == 1 and psi.eval(0.0) == np.inf
  curve = psi(np.cos(np.linspace(0, 80 * degree, 9)))
  assert np.all(np.diff(curve) >= -1e-9) and curve[-1] > 1.5


def test_SchulzDefocusingKernel_matchesMATLAB():
  # pinned against MATLAB's SchulzDefocusingKernel
  _, psi = _schulz()
  assert np.allclose(psi(np.cos(np.array([30, 50, 70]) * degree)), [1.0, 1.685394, 3.885520], atol=1e-5)


def test_SchulzDefocusingKernel_defocusingIsFiniteAndOneAtThePole():
  pf, psi = _schulz()
  d = psi.defocusing(pf)
  assert d.allI[0].shape == pf.allI[0].shape and np.all(np.isfinite(d.allI[0]))
  assert np.isclose(d.allI[0][0, 0], 1)                # the pole is not defocused


def test_correct_recoversAMeasurementDampedByTheDefocusingCurve():
  pf, psi = _schulz()
  d = psi.defocusing(pf)
  assert np.max(np.abs(correct(pf * d, defocusing=d).allI[0] - 1)) < 1e-12


# a triclinic cubic ODF and its 100 pole figure on a 5 degree half grid
@pytest.fixture(scope="module")
def fine():
  cs, ss = crystalFrame("m-3m"), specimenFrame("1")
  odf = unimodalODF(orientation.byEuler(30 * degree, 50 * degree, 10 * degree, cs, ss), halfwidth=20 * degree)
  h = Miller(1, 0, 0, cs)
  r = regularS2Grid(resolution=5 * degree, antipodal=True, frame=ss)
  return SimpleNamespace(ss=ss, h=h, r=r, pf=calcPoleFigure(odf, h, r))


def test_S2FunHarmonic_ofAPoleFigureTakesItsFrameAndTheBandwidthOfItsGrid(fine):
  pf = fine.pf
  f = S2FunHarmonic(pf)
  assert f.frame is pf.SS and f.antipodal
  # half of what the point density of a five degree grid resolves
  assert f.bandwidth == 23
  assert S2FunHarmonic(pf, bandwidth=12).bandwidth == 12


def test_S2FunHarmonic_ofAPoleFigureGivesTheIntensitiesBack(fine):
  # up to what the bandwidth drops
  pf = fine.pf
  vals = np.real(S2FunHarmonic(pf)(fine.r))
  assert np.sqrt(np.mean((vals - pf.allI[0]) ** 2)) < 0.05 * np.max(pf.allI[0])


def test_S2FunHarmonic_ofScatteredPoleFigureTakesTheBandwidthItsCountSupports(fine):
  rl = vector3d.rand(500, antipodal=True, rng=rng)
  assert S2FunHarmonic(PoleFigure(fine.h, rl, rng.random(500), fine.ss)).bandwidth == 14


def test_S2FunHarmonic_refusesSeveralPoleFigures(fine):
  with pytest.raises(ValueError):
    S2FunHarmonic(PoleFigure.cat(fine.pf, fine.pf))


def test_refine_addsPointsInsideACapAndNoneBelowItsRim():
  v = equispacedS2Grid(resolution=20 * degree, maxTheta=80 * degree)
  w, new = refine(v)
  assert w.size == v.size + new.size and new.size > v.size / 2
  assert np.all(new.theta <= 80 * degree + 1e-6)
  assert v.refine()[1].size == new.size
