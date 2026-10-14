# handles and sums (docs/plans/functions.md F5: H5a to H5c, check_S1Fun, the handle and
# composition cases of check_radonOptions)
import numpy as np
import pytest
from numpy import pi

from mtex import (angle, crystalFrame, degree, fibre, fibreODF, mean, Miller, orientation, radon, rotate, rotation, S1FunHandle, S1FunHarmonic, S2DeLaValleePoussinKernel, S2FunHandle, S2FunHarmonic, S2FunSum, SO3FunComposition, SO3FunHandle, SO3FunHarmonic, SO3FunRBF, specimenFrame, unimodalODF, vector3d, zvector)

rng = np.random.default_rng(51)


# check_radonOptions's handle: 1 + cos(angle), under 432 and 222, averaged over the groups when asked
def cosHandle(symmetrise=False):
  cs, ss = crystalFrame("432"), specimenFrame("222")
  return SO3FunHandle(lambda o: 1 + np.cos(angle(o, symmetry=False)), cs, ss, bandwidth=16, symmetrise=symmetrise)


def test_SO3FunHandle_evaluatesItsFunction():
  f = cosHandle()
  o = orientation(rotation.rand(7, rng=rng), f.CS, f.SS)
  assert np.allclose(f(o), 1 + np.cos(angle(o, symmetry=False))) and f(o[0]) == f(o)[0] and f(o[0]) == f(rotation(o[0]))


def test_SO3FunHandle_refusesAnotherFrame():
  # another frame is refused; the same frames without their groups are not another frame
  f = cosHandle()
  with pytest.raises(ValueError):
    f(orientation(rotation.rand(2, rng=rng), f.CS, specimenFrame.rolling))


def test_SO3FunHandle_symmetriseAveragesOverBothGroups():
  g = cosHandle(symmetrise=True)
  cs, ss = g.CS, g.SS
  o = orientation(rotation.rand(7, rng=rng), cs, ss)
  E, S = cs.sym.rot.flatten().rot, ss.sym.rot.flatten().rot
  for i in range(E.size):
    for j in range(S.size):
      assert np.isclose(g(S[j] * rotation(o[0]) * E[i]), g(o[0]), rtol=1e-10)


def test_SO3FunHandle_operationsGoThroughItsHarmonicForm():
  # the harmonic form and the operations through it
  g = cosHandle(symmetrise=True)
  h = SO3FunHarmonic(g)
  assert h.bandwidth == 16 and h.isReal
  rs = rotation.rand(20, rng=rng)
  assert np.max(np.abs(h(rs) - g(rs))) < 1e-6 * np.max(np.abs(g(rs)))
  assert isinstance(radon(g, Miller(1, 0, 0, g.CS)), S2FunHarmonic)
  assert np.isclose(mean(g), mean(h))
  assert np.allclose((2 * g)(rs), 2 * g(rs))
  assert "SO3FunHandle" in repr(g)


def test_SO3FunHandle_exchangeSymmetryAveragesWithTheInverses():
  # exchange symmetry averages with the inverses
  cs = crystalFrame("432")
  rs = rotation.rand(20, rng=rng)
  fx = SO3FunHandle(lambda o: o.a + o.b, cs, cs, antipodal=True, symmetrise=True)
  assert np.isclose(fx(rs[0]), fx(rs[0].inv()))


def test_S2FunHandle_evaluatesItsFunctionAndRefusesAnotherFrame():
  cs = crystalFrame("6/mmm")
  f = S2FunHandle(lambda v: v.z ** 2 + 0.3 * v.x, cs, bandwidth=16)
  vs = vector3d(vector3d.rand(6, rng=rng), frame=cs)
  assert np.allclose(f(vs), vs.z ** 2 + 0.3 * vs.x) and f(vs[0]) == f(vs)[0]
  with pytest.raises(ValueError):
    f(Miller(1, 0, 0, crystalFrame("m-3m")))


# the handle v.z^2 + 0.3 v.x under 6/mmm, symmetrised
def symmetricS2Handle():
  return S2FunHandle(lambda v: v.z ** 2 + 0.3 * v.x, crystalFrame("6/mmm"), bandwidth=16, symmetrise=True)


def test_S2FunHandle_symmetriseIsInvariant():
  g = symmetricS2Handle()
  v0 = vector3d.rand(rng=rng)
  E = g.frame.sym.rot.flatten()
  for i in range(E.size):
    assert np.isclose(g(E[i] * v0), g(v0))


def test_S2FunHandle_antipodalIsEven():
  a = S2FunHandle(lambda v: v.z ** 2 + 0.3 * v.x, bandwidth=16, antipodal=True)
  v0 = vector3d.rand(rng=rng)
  assert a.antipodal and np.isclose(a(v0), a(-v0)) and np.isclose(a(v0), v0.z ** 2)
  assert S2FunHarmonic(a, bandwidth=8).antipodal


def test_S2FunHandle_agreesWithItsHarmonicForm():
  g = symmetricS2Handle()
  h = S2FunHarmonic(g, bandwidth=8)
  ds = vector3d.rand(20, rng=rng)
  assert np.max(np.abs(h(ds) - g(ds))) < 1e-7
  assert np.isclose(mean(g), mean(h)) and np.allclose((2 * g)(ds), 2 * g(ds))


def test_S1FunHandle_evaluatesItsFunction():
  s = S1FunHandle(lambda x: np.cos(2 * x) + 0.5 * np.sin(x) + 0.3, bandwidth=8)
  xs = np.linspace(0, 2 * pi, 13)
  assert np.allclose(s(xs), np.cos(2 * xs) + 0.5 * np.sin(xs) + 0.3)


def test_S1FunHarmonic_addsAndSubtractsScalars():
  # check_S1Fun
  s = S1FunHandle(lambda x: np.cos(2 * x) + 0.5 * np.sin(x) + 0.3, bandwidth=8)
  xs = np.linspace(0, 2 * pi, 13)
  S = S1FunHarmonic(s, bandwidth=4)
  assert np.max(np.abs((S + 3)(xs) - (s(xs) + 3))) < 1e-10
  assert np.max(np.abs((3 + S)(xs) - (s(xs) + 3))) < 1e-10
  assert np.max(np.abs((S - 1.5)(xs) - (s(xs) - 1.5))) < 1e-10
  assert np.isclose(mean(S + 3), mean(S) + 3) and S.isReal
  assert np.allclose((S1FunHarmonic([2.0]) + 5)(xs), 7.0)


# a unimodal ODF, a fibre ODF and a symmetrised handle under 432 and 1
def odfTerms():
  cs, ss = crystalFrame("432"), specimenFrame("1")
  u = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss))
  fb = fibreODF(fibre(orientation(rotation.rand(rng=rng), cs, ss), Miller(1, 1, 1, cs)))
  h = SO3FunHandle(lambda o: 1 + np.cos(angle(o, symmetry=False)), cs, ss, bandwidth=16, symmetrise=True)
  return cs, ss, u, fb, h


def test_SO3FunComposition_sumsItsTerms():
  cs, ss, u, fb, h = odfTerms()
  s = 0.5 * u + 0.5 * fb
  assert isinstance(s, SO3FunComposition) and len(s.components) == 2 and np.isclose(mean(s), 1)
  rs = rotation.rand(10, rng=rng)
  assert np.allclose(s(rs), 0.5 * u(rs) + 0.5 * fb(rs))
  t = s + h - u
  assert isinstance(t, SO3FunComposition) and len(t.components) == 4 and np.allclose(t(rs), s(rs) + h(rs) - u(rs))
  assert np.allclose((s + 2)(rs), s(rs) + 2) and np.isclose(mean(s + 2), 3)


def test_abs_ofADifferenceIsAHandle():
  # the absolute value of a difference is a handle, MATLAB's mean(abs(f - g))
  cs, ss, u, fb, h = odfTerms()
  rs = rotation.rand(10, rng=rng)
  a = abs(h - u)
  assert isinstance(a, SO3FunHandle) and np.allclose(a(rs), np.abs(h(rs) - u(rs)))


def test_SO3FunComposition_agreesWithItsHarmonicForm():
  cs, ss, u, fb, h = odfTerms()
  s = 0.5 * u + 0.5 * fb
  rs = rotation.rand(10, rng=rng)
  H = SO3FunHarmonic(s)
  assert H.isReal and H.CS is cs
  assert np.max(np.abs(H(rs) - s(rs))) < 3e-2 * np.max(s(rs))


def test_poleFigure_ofACompositionIsAxisValuedForALaueCrystal():
  # check_radonOptions on the composition: the pole figure exists and is axis valued for a Laue crystal
  cs, ss, u, fb, h = odfTerms()
  s = 0.5 * u + 0.5 * fb
  assert isinstance(radon(s, Miller(1, 0, 0, cs)), S2FunHarmonic)
  csL = crystalFrame("m-3m")
  sL = 0.5 * unimodalODF(orientation(rotation.rand(rng=rng), csL, ss)) + 0.5 * fibreODF(fibre(orientation(rotation.rand(rng=rng), csL, ss), Miller(1, 1, 1, csL)))
  assert radon(sL, Miller(1, 0, 0, csL)).antipodal
  assert radon(s, vector3d(1, 0, 0, antipodal=True)).antipodal


def test_sum_ofRBFsWithOneKernelStaysAnRBF():
  # two RBFs with the same kernel stay one RBF, with different kernels a sum
  cs, ss, u, fb, h = odfTerms()
  csL = crystalFrame("m-3m")
  u2 = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss))
  assert isinstance(u + u2, SO3FunRBF) and np.isclose(mean(u + u2), 2)
  assert isinstance(u + unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=5 * degree), SO3FunComposition)
  with pytest.raises(ValueError):
    u + unimodalODF(orientation(rotation.rand(rng=rng), csL, ss))


def test_sum_ofAHarmonicAndAKernelFormIsAComposition():
  # a harmonic function plus a kernel form is a composition too
  cs, ss, u, fb, h = odfTerms()
  assert isinstance(SO3FunHarmonic(u) + fb, SO3FunComposition) and isinstance(fb + SO3FunHarmonic(u), SO3FunComposition)


def test_rotate_carriesEveryKindOfTerm():
  # a turn of the specimen carries every kind of term with it, so the rotated function takes
  # at q g the value the original takes at g; a triclinic pair, so no group folds the two
  # sides differently
  cs1, ss1 = crystalFrame("1", [1, 1, 1]), specimenFrame("1")
  q = rotation.byAxisAngle(zvector, 30 * degree)
  gs = orientation(rotation.rand(20, rng=rng), cs1, ss1)
  ub = unimodalODF(orientation(rotation.rand(rng=rng), cs1, ss1))
  fbb = fibreODF(fibre(orientation(rotation.rand(rng=rng), cs1, ss1), Miller(1, 1, 1, cs1)), halfwidth=10 * degree)
  for f in (ub, fbb, 0.4 * ub + 0.6 * fbb):
    rf = rotate(f, q)
    assert np.allclose(rf(orientation(q * rotation(gs), rf.CS, rf.SS)), f(gs))
    rr = rotate(f, q, "right")
    assert np.allclose(rr(orientation(rotation(gs) * q, rr.CS, rr.SS)), f(gs))


def test_S2FunSum_sumsAndDividesItsTerms():
  # on the sphere
  a = S2FunHandle(lambda v: v.z ** 2, bandwidth=16)
  b = S2FunHarmonic(S2DeLaValleePoussinKernel(20))
  ab = a + b
  ds = vector3d.rand(10, rng=rng)
  assert isinstance(ab, S2FunSum) and np.allclose(ab(ds), a(ds) + b(ds)) and np.isclose(mean(ab), mean(a) + mean(b))
  assert np.max(np.abs(S2FunHarmonic(ab)(ds) - ab(ds))) < 1e-6
  assert isinstance(b + a, S2FunSum) and np.allclose((a / b)(ds), a(ds) / b(ds))


def test_S1FunHandle_maxAndMinWithANumberOrAHandleArePointwiseAndExact():
  f, g = S1FunHandle(np.cos), S1FunHandle(lambda x: np.sin(2 * x))
  xs = rng.random(10) * 2 * pi
  assert np.allclose(f.max(0.3)(xs), np.maximum(np.cos(xs), 0.3)) and np.allclose(f.min(g)(xs), np.minimum(np.cos(xs), np.sin(2 * xs)))
  assert isinstance(f.max(g), S1FunHandle)


@pytest.mark.parametrize("make", [lambda cs, ss: unimodalODF(orientation(rotation.rand(rng=rng), cs, ss)),
                                  lambda cs, ss: fibreODF(fibre(orientation(rotation.rand(rng=rng), cs, ss), Miller(1, 1, 1, cs)))],
                         ids=["RBF", "CBF"])
def test_sum_alongAnAxisOfASingleFunctionIsTheFunction(make):
  # MATLAB's sum(odf, d) on a single function: a list of one along every axis
  odf = make(crystalFrame("432"), specimenFrame("1"))
  rs = rotation.rand(5, rng=rng)
  assert np.allclose(odf.sum(0)(rs), odf(rs)) and np.allclose(odf.sum(axis=1)(rs), odf(rs)) and np.isclose(odf.sum(), 8 * pi ** 2 * mean(odf))
