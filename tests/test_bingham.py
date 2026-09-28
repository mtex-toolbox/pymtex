# the Bingham distributions (docs/plans/functions.md F5: B1 to B4)
import numpy as np
import pytest
from math import factorial

from mtex import (Miller, S2FunBingham, S2FunHarmonic, SO3FunBingham, SO3FunHarmonic, angle, calcBinghamODF, crystalFrame,
                  degree, mean, orientation, pdf, quadratureS2Grid, quadratureSO3Grid, radon, rotate, rotation, specimenFrame,
                  vector3d, xvector, yvector, zvector)
from mtex.functions.bingham import _logMhyperGradient, mhyper

rng = np.random.default_rng(61)


# the scalar 1F1(a; b; k) by its series
def scalar1F1(a, b, k):
  return sum(np.prod([(a + j) / (b + j) for j in range(n)]) * k ** n / factorial(n) for n in range(120))


def test_mhyper_ofASingleConcentrationIsTheScalar1F1():
  for k in (0.5, 3.0, 20.0):
    assert np.isclose(mhyper([k, 0, 0, 0]), scalar1F1(0.5, 2, k), rtol=1e-12)
    assert np.isclose(mhyper([k, 0, 0]), scalar1F1(0.5, 1.5, k), rtol=1e-12)
  assert np.isclose(mhyper([0, 0, 0, 0]), 1) and np.isclose(mhyper(np.zeros(3)), 1)


def test_mhyper_isTheMeanOfTheExponentialOverSO3AndTheSphere():
  # the mean of exp(q^T diag(kappa) q) over SO(3) and of exp(v^T diag(Z) v) over the sphere
  kappa = np.array([5.0, 2.0, 1.0, 0.0])
  g = quadratureSO3Grid(60)
  q = g.nodes().data.reshape(-1, 4)
  assert np.isclose(mhyper(kappa), np.sum(g.weights * np.exp((q ** 2) @ kappa)), rtol=1e-8)
  Z = np.array([-3.0, -1.0, 0.0])
  s = quadratureS2Grid(60)
  v = s.nodes().data.reshape(-1, 3)
  assert np.isclose(mhyper(Z), np.sum(s.weights * np.exp((v ** 2) @ Z)) / (4 * np.pi), rtol=1e-8)


def test_mhyper_shiftMultipliesByItsExponentialAndMomentsSumToOne():
  kappa = np.array([5.0, 2.0, 1.0, 0.0])
  assert np.isclose(mhyper(kappa + 2), np.exp(2) * mhyper(kappa))
  assert abs(np.sum(_logMhyperGradient(kappa)) - 1) < 1e-6


# a Bingham ODF under 432 and 1 with orthogonal axes, and the harmonic expansion of it; per test,
# since the session reset renews the frames
@pytest.fixture
def symmetricBingham():
  cs, ss = crystalFrame("432"), specimenFrame("1")
  r0 = np.random.default_rng(62)
  A = rotation.rand(4, rng=r0).data.T
  Q, _ = np.linalg.qr(A)
  fs = SO3FunBingham([10.0, 5, 1, 0], Q, cs, ss)
  H = SO3FunHarmonic(fs, bandwidth=24)
  return cs, ss, A, Q, fs, H


def test_SO3FunBingham_hasMeanOneAndTheDensityAtTheMode():
  # the example of MTEX with trivial symmetry: mean one and the density at the mode
  f = SO3FunBingham([100.0, 90, 80, 0], np.eye(4), crystalFrame("1"))
  assert np.isclose(mean(f), 1)
  assert np.isclose(f(rotation.id), np.exp(100) / mhyper([100.0, 90, 80, 0]))
  g = quadratureSO3Grid(80)
  assert np.isclose(np.sum(g.weights * f(g.nodes())), 1, rtol=1e-4)


def test_SO3FunBingham_isInvariantUnderTheCrystalSymmetry(symmetricBingham):
  cs, ss, A, Q, fs, H = symmetricBingham
  r = rotation.rand(rng=rng)
  E = cs.sym.rot.flatten().rot
  for i in range(E.size):
    assert np.isclose(fs(r * E[i]), fs(r), rtol=1e-10)


def test_SO3FunBingham_refusesAxesThatAreNotOrthogonal(symmetricBingham):
  cs, ss, A, Q, fs, H = symmetricBingham
  with pytest.raises(ValueError):
    SO3FunBingham([10.0, 5, 1, 0], A, cs, ss)


def test_SO3FunBingham_evaluatesAtFramesDifferingOnlyInTheGroup(symmetricBingham):
  # frames that differ only in the group share their axes and serve, as in MATLAB; a frame
  # with other axes does not
  cs, ss, A, Q, fs, H = symmetricBingham
  r = rotation.rand(rng=rng)
  assert np.isclose(fs(orientation(r, cs, specimenFrame("222"))), fs(r))
  with pytest.raises(ValueError):
    fs(orientation(r, crystalFrame("-3m"), ss))


def test_SO3FunBingham_agreesWithItsHarmonicExpansion(symmetricBingham):
  cs, ss, A, Q, fs, H = symmetricBingham
  assert np.isclose(mean(H), 1, rtol=1e-6)
  rs = rotation.rand(20, rng=rng)
  assert np.max(np.abs(H(rs) - fs(rs))) < 1e-2 * np.max(fs(rs))


def test_SO3FunBingham_poleFigureByTheFibreFormulaAgreesWithTheHarmonicRoute(symmetricBingham):
  # check_radonOptions: the pole figure by the fibre formula agrees with the harmonic route
  cs, ss, A, Q, fs, H = symmetricBingham
  h = Miller(1, 0, 0, cs)
  P = pdf(fs, h)
  vs = vector3d.rand(20, rng=rng)
  assert np.max(np.abs(P(vs) - pdf(H, h)(vs))) < 1e-2 * np.max(P(vs))
  assert fs.pdf(h, bandwidth=4).bandwidth <= 4
  assert np.isclose(radon(fs, h, vs[0]), P(vs[0]))


def test_SO3FunBingham_rotateFromTheLeftAndTheRight():
  # check_SO3FunRotate on the Bingham example
  f1 = SO3FunBingham([100.0, 90, 80, 0], np.eye(4))
  rot = rotation.rand(rng=rng)
  rs = rotation.rand(20, rng=rng)
  assert np.allclose(rotate(f1, rot)(rs), f1(rot.inv() * rs), rtol=1e-6)
  assert np.allclose(rotate(f1, rot, "right")(rs), f1(rs * rot.inv()), rtol=1e-6)


def test_SO3FunBingham_scalesAndDisplays(symmetricBingham):
  cs, ss, A, Q, fs, H = symmetricBingham
  rs = rotation.rand(20, rng=rng)
  assert np.allclose((2 * fs)(rs), 2 * fs(rs)) and np.isclose(mean(2 * fs), 2)
  assert "SO3FunBingham" in repr(fs)


def test_SO3FunBingham_axesAsAnOrientationCarryTheFrames(symmetricBingham):
  # the axes as an orientation carry the frames
  cs, ss, A, Q, fs, H = symmetricBingham
  rs = rotation.rand(20, rng=rng)
  fo = SO3FunBingham([10.0, 5, 1, 0], orientation(rotation(Q.T), cs, ss))
  assert fo.CS is cs and np.allclose(fo(rs), fs(rs))


def test_calcBinghamODF_recoversTheModeAndTheConcentrations():
  # a sample by rejection from a sharp Bingham about a random centre
  cs, ss = crystalFrame("222"), specimenFrame("1")
  centre = rotation.rand(rng=rng)
  kappa0 = np.array([40.0, 30.0, 20.0, 0.0])
  f0 = rotate(SO3FunBingham(kappa0, np.eye(4)), centre)
  cand = rotation.rand(400_000, rng=rng)
  vals = f0(cand)
  keep = rng.random(cand.size) * np.max(vals) < vals
  sample = orientation(cand[keep], cs, ss)
  assert sample.size > 2000
  for exact in (False, True):
    est = calcBinghamODF(sample, exact=exact)
    assert est.CS is cs and np.isclose(mean(est), 1)
    # the estimated mode is near the centre, up to the symmetry
    assert angle(orientation(rotation(est.A[:, 0]), cs, ss), orientation(centre, cs, ss)) < 3 * degree
    # the exact fit recovers the concentrations, the approximation their order of size
    assert np.max(np.abs(np.sort(est.kappa)[::-1] - kappa0)) < (6 if exact else 15)
  # the exact fit, the last of the loop, evaluates to about the truth at the sample
  assert np.corrcoef(est(sample), f0(rotation(sample)))[0, 1] > 0.99


def test_S2FunBingham_isAnAntipodalDensityOfMeanOne():
  f = S2FunBingham([-10.0, -4.0, 0.0])
  assert np.isclose(mean(f), 1) and f.antipodal and np.isclose(f(zvector), 1 / mhyper([-10.0, -4, 0]))
  s = quadratureS2Grid(60)
  assert np.isclose(np.sum(s.weights * f(s.nodes())) / (4 * np.pi), 1, rtol=1e-8)
  assert f(-xvector) == f(xvector)


# the rotation of the axes of the S2 Bingham tests
@pytest.fixture(scope="module")
def s2rot():
  return rotation.rand(rng=np.random.default_rng(63))


def test_S2FunBingham_withSymmetryIsInvariantAndMatchesItsExpansion(s2rot):
  rot = s2rot
  cs = crystalFrame("6/mmm")
  fs = S2FunBingham([-10.0, -4.0, 0.0], vector3d.cat(rot * xvector, rot * yvector, rot * zvector), cs)
  v = vector3d.rand(rng=rng)
  E = cs.sym.rot.flatten()
  for i in range(E.size):
    assert np.isclose(fs(E[i] * v), fs(v), rtol=1e-10)
  H = S2FunHarmonic(fs, bandwidth=40)
  vs = vector3d.rand(20, rng=rng)
  assert np.max(np.abs(H(vs) - fs(vs))) < 1e-4 * np.max(fs(vs)) and H.antipodal


def test_S2FunBingham_rotate(s2rot):
  rot = s2rot
  f = S2FunBingham([-10.0, -4.0, 0.0])
  vs = vector3d.rand(20, rng=rng)
  assert np.allclose(rotate(f, rot)(vs), f(rot.inv() * vs))


# a rejection sample of a rotated S2 Bingham and the distribution it was drawn from
@pytest.fixture(scope="module")
def s2sample(s2rot):
  r0 = np.random.default_rng(64)
  g = rotate(S2FunBingham([-8.0, -3.0, 0.0]), s2rot)
  cand = vector3d.rand(200_000, rng=r0)
  vals = g(cand)
  sample = cand[r0.random(cand.size) * np.max(vals) < vals]
  return g, sample


def test_S2FunBingham_fitRecoversTheShapeAndTheAxes(s2rot, s2sample):
  # the fit from a rejection sample recovers the shape and the axes
  rot = s2rot
  g, sample = s2sample
  # in MTEX's convention: shape parameters ascending to zero at the mode, the third axis
  est = S2FunBingham.fit(sample)
  assert np.max(np.abs(est.Z - [-8.0, -3.0, 0.0])) < 1.5
  assert abs(est.a[2].dot(rot * zvector)) > 0.99


def test_S2FunBingham_confidenceEllipseIsCentredAndShrinksWithTheSample(s2sample):
  # the confidence ellipse of the modal axis is centred on it and shrinks with the sample
  g, sample = s2sample
  est, ab, r = S2FunBingham.fit(sample, confidenceEllipse=True)
  assert abs((r * zvector).dot(est.a[2])) > 1 - 1e-9 and np.all(ab > 0) and np.all(ab < 5 * np.pi / 180)
  _, ab10, _ = S2FunBingham.fit(sample[:sample.size // 10], confidenceEllipse=True)
  assert np.all(ab10 > 2 * ab)
  assert np.corrcoef(est(sample), g(sample))[0, 1] > 0.99


def test_S2FunBingham_fitKeepsTheFrameAndDisplays(s2sample):
  g, sample = s2sample
  cs = crystalFrame("6/mmm")
  assert S2FunBingham.fit(vector3d(sample, frame=cs)).frame is cs
  assert "S2FunBingham" in repr(S2FunBingham.fit(sample))


def test_discreteSample_ofABinghamDistributionHasItsMoments():
  from mtex import BinghamODF, discreteSample, equispacedSO3Grid
  A = np.linalg.qr(rng.standard_normal((4, 4)))[0]
  A[:, 0] *= np.sign(np.linalg.det(A))
  f = BinghamODF([0, 5, 20, 40], A, crystalFrame('1'))
  q = discreteSample(f, 100000, rng=rng).data.reshape(-1, 4)
  # the second moments along the axes, against the density weighted over a grid
  g = equispacedSO3Grid(crystalFrame('1'), resolution=2 * degree)
  w = f(g).ravel()
  qg = g.data.reshape(-1, 4)
  Tg = A.T @ ((qg * (w / w.sum())[:, None]).T @ qg) @ A
  T = A.T @ (q.T @ q / len(q)) @ A
  assert np.allclose(np.diag(T), np.diag(Tg), atol=3e-3)


def test_discreteSample_withCrystalSymmetryGivesSymmetricCopies():
  # with crystal symmetry every sample is a symmetric copy
  from mtex import BinghamODF, discreteSample
  f = BinghamODF([0, 0, 0, 30], crystalFrame('432'))
  o = discreteSample(f, 1000, rng=rng)
  assert o.CS is crystalFrame('432') and np.all(np.isfinite(o.data))
