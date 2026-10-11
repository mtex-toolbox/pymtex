# the radial basis and fibre representations (docs/plans/functions.md F4: R1, R3, R4, R5 -
# check_odfGrad, check_radonOptions, check_calcPoleFigureSuperposition)
import numpy as np
import pytest

from mtex import (crystalFrame, degree, fibre, fibreODF, FourierODF, grad, mean, Miller, orientation, radon, rotation, S2DeLaValleePoussinKernel, SO3DeLaValleePoussinKernel, SO3FunCBF, SO3FunHarmonic, SO3FunRBF, specimenFrame, uniformODF, unimodalODF, vector3d, xvector, yvector, zvector)
from mtex.geometry.symreduce import reductionGroups

rng = np.random.default_rng(41)


def test_uniformODF_isOneEverywhere():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  u = uniformODF(cs, ss)
  assert np.isclose(mean(u), 1) and u(rotation.rand(rng=rng)) == 1 and u.CS is cs
  assert SO3FunHarmonic(u).isclose(SO3FunHarmonic([1.0], cs, ss))


# five 15 degree unimodal components under m-3m and 222
def unimodal():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  o = orientation(rotation.rand(5, rng=rng), cs, ss)
  return cs, ss, o, unimodalODF(o, halfwidth=15 * degree)


def test_unimodalODF_isTheMeanOfTheKernelOverTheEquivalentsOfTheCentres():
  cs, ss, o, f = unimodal()
  assert np.isclose(mean(f), 1) and f.center.size == 5
  psi = SO3DeLaValleePoussinKernel(halfwidth=15 * degree)
  # the value is the mean of the kernel over the equivalents of the centres
  L, R = reductionGroups(o)
  L, R = rotation(L), rotation(R)
  for r in rotation.rand(4, rng=rng):
    direct = np.sum([psi(L[j] * o[k] * R[i], r) for k in range(5) for i in range(R.size) for j in range(L.size)]) / (5 * L.size * R.size * psi.mean())
    assert np.isclose(f(r), direct, rtol=1e-4)
    assert f(orientation(r, cs, ss)) == f(r)


def test_unimodalODF_evaluatesOnStrippedFramesAndRefusesAnotherFrame():
  # the frames stripped of their groups are the same frames, another frame is not
  cs, ss, o, f = unimodal()
  assert np.isclose(f(orientation(rotation.id, cs.stripSym(), ss.stripSym())), f(rotation.id))
  with pytest.raises(ValueError):
    f(orientation(rotation.id, cs, specimenFrame.rolling))


def test_unimodalODF_agreesWithItsHarmonicForm():
  # the harmonic form agrees away from the truncation, is symmetric and has mean one
  cs, ss, o, f = unimodal()
  h = SO3FunHarmonic(f)
  assert np.isclose(mean(h), 1) and h.CS is cs and h.isReal
  rs = rotation.rand(200, rng=rng)
  assert np.max(np.abs(h(rs) - f(rs))) < 2e-2 * np.max(f(rs))
  assert f(rs.reshape(20, 10)).shape == (20, 10)


def test_unimodalODF_weightsScaleTheComponents():
  # weights scale the mean
  cs, ss, o, f = unimodal()
  g = unimodalODF(o, halfwidth=15 * degree, weights=[1, 2, 3, 4, 5])
  assert np.isclose(mean(g), 1) and np.isclose(g.weights[4], 5 * g.weights[0])
  assert "SO3FunRBF" in repr(f)


def test_unimodalODF_exchangeSymmetryAveragesWithTheInverses():
  # exchange symmetry averages with the inverses
  cs = crystalFrame("m-3m")
  fx = unimodalODF(orientation(rotation.rand(3, rng=rng), cs, cs, antipodal=True), halfwidth=15 * degree)
  for r in rotation.rand(3, rng=rng):
    assert np.isclose(fx(r), fx(r.inv()), rtol=1e-8)
  assert SO3FunHarmonic(fx).antipodal


def test_unimodalODF_antipodalWithUnequalWeightsKeepsEachWeightOnItsInverse():
  # each centre and its inverse carry the centre's weight, over the equivalents of both
  cs = crystalFrame("m-3m")
  o = orientation(rotation.rand(4, rng=rng), cs, cs, antipodal=True)
  psi = SO3DeLaValleePoussinKernel(halfwidth=15 * degree)
  f = SO3FunRBF(o, psi, [1.0, 2.0, 3.0, 4.0])
  L, R = reductionGroups(o)
  L, R = rotation(L), rotation(R)
  r = rotation.rand(rng=rng)
  direct = sum(w * psi(L[j] * c * R[i], r) for w, k in zip(f.weights, range(4)) for c in (o[k], o[k].inv())
               for i in range(R.size) for j in range(L.size)) / (2 * L.size * R.size)
  # the sum leaves out the terms below 1e-9 of the kernel's peak
  assert np.isclose(f(r), direct, rtol=1e-8)


def test_eval_ofManyCentresSearchesTheirTreeAsTheBlocksSum(monkeypatch):
  # the tree of the centres, its hits in small batches, against the sum by blocks
  from mtex.functions import rbf
  cs = crystalFrame("m-3m")
  f = SO3FunRBF(orientation(rotation.rand(300, rng=rng), cs, cs, antipodal=True), SO3DeLaValleePoussinKernel(halfwidth=5 * degree),
                rng.random(300))
  r = rotation.rand(60, rng=rng)
  blocks = f(r)
  monkeypatch.setattr(rbf, "BLOCK", 1000)
  monkeypatch.setattr(rbf, "HITS", 50)
  assert np.allclose(f(r), blocks, rtol=1e-12, atol=1e-12)


def test_unimodalODF_takesAKernelPositionalTheIdentityAsDefaultAndTheAlgebra():
  # a kernel positional, the identity as the default centre, the algebra
  cs, ss, o, f = unimodal()
  psi = SO3DeLaValleePoussinKernel(halfwidth=15 * degree)
  rs = rotation.rand(200, rng=rng)
  assert isinstance(unimodalODF(o, psi).psi, SO3DeLaValleePoussinKernel)
  assert unimodalODF(cs, ss).center.size == 1
  assert np.isclose((2 * f + 1)(rs[0]), 2 * f(rs[0]) + 1)
  assert np.isclose(f.inv()(rs[0].inv()), f(rs[0])) and f.inv().CS is ss


# the central difference of f at o along v on one side
def fd(f, o, v, side, d=0.005 * degree):
  r = rotation(o)
  a = rotation.byAxisAngle(v, d) * r if side == "left" else r * rotation.byAxisAngle(v, d)
  b = rotation.byAxisAngle(v, -d) * r if side == "left" else r * rotation.byAxisAngle(v, -d)
  return (f(a) - f(b)) / (2 * d)


def test_grad_ofAUnimodalODFIsTheFiniteDifferenceOnBothSides():
  # check_odfGrad: the unimodal ODF, both tangent spaces
  cs, ss = crystalFrame("-3m1"), crystalFrame("m-3m")
  odf = unimodalODF(orientation(rotation.rand(20, rng=rng), cs, ss))
  os = orientation(rotation.rand(30, rng=rng), cs, ss)
  for side in ("left", "right"):
    ts = grad(odf, os, side)
    assert ts.shape == (30,) and ts.side == side
    worst = 0.0
    for i in range(30):
      g = np.array([fd(odf, os[i], v, side) for v in (xvector, yvector, zvector)])
      worst = max(worst, np.linalg.norm(ts[i].data - g) / (1 + np.linalg.norm(g)))
    assert worst < 1e-3
    assert np.max(np.linalg.norm(ts.data, axis=-1)) > 1e-3


def test_grad_ofAFibreODFIsTheFiniteDifferenceOnBothSides():
  # the fibre ODF
  cs2, ss2 = crystalFrame("432"), specimenFrame("222")
  fodf = fibreODF(Miller(vector3d.rand(rng=rng), cs2), vector3d.rand(rng=rng), ss2)
  os2 = orientation(rotation.rand(30, rng=rng), cs2, ss2)
  for side in ("left", "right"):
    worst = 0.0
    for i in range(30):
      t = grad(fodf, os2[i], side)
      g = np.array([fd(fodf, os2[i], v, side) for v in (xvector, yvector, zvector)])
      worst = max(worst, np.linalg.norm(t.data - g) / (1 + np.linalg.norm(g)))
    assert worst < 1e-3


# the 12 degree fibre ODF of (110) and a random specimen direction under 432 and 222
def fibreOf110():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  h, r = Miller(1, 1, 0, cs), vector3d.rand(rng=rng)
  return cs, ss, h, r, fibreODF(h, r, ss, halfwidth=12 * degree)


def test_fibreODF_isTheMeanOfTheS2KernelOverTheEquivalentsAndInvariant():
  cs, ss, h, r, f = fibreOf110()
  assert np.isclose(mean(f), 1) and f.CS is cs and f.SS is ss
  psi = S2DeLaValleePoussinKernel(halfwidth=12 * degree)
  L, R = reductionGroups(orientation(rotation.id, cs, ss))
  L, R = rotation(L), rotation(R)
  hn = vector3d(h.data).normalize()
  for g in rotation.rand(4, rng=rng):
    direct = np.sum([psi((L[j] * g * R[i] * hn).dot(r)) for i in range(R.size) for j in range(L.size)]) / (L.size * R.size * psi.mean())
    assert np.isclose(f(g), direct, rtol=1e-8)
    for i in range(0, R.size, 5):
      for j in range(0, L.size, 3):
        assert np.isclose(f(L[j] * g * R[i]), f(g), rtol=1e-8)


def test_fibreODF_agreesWithItsHarmonicFormWhichIsFourierODF():
  cs, ss, h, r, f = fibreOf110()
  # the harmonic form
  H = SO3FunHarmonic(f)
  assert np.isclose(mean(H), 1) and H.isReal
  # MATLAB's FourierODF is the same
  assert isinstance(FourierODF(f), SO3FunHarmonic) and np.allclose(FourierODF(f).fhat, H.fhat)
  # FourierODF(odf, L) cuts at the bandwidth L
  assert FourierODF(f, 8).bandwidth == 8
  rs = rotation.rand(100, rng=rng)
  assert np.max(np.abs(H(rs) - f(rs))) < 2e-2 * np.max(f(rs))


def test_fibreODF_ofAFibreOrOfSeveralFibres():
  # from a fibre of P1, and several fibres
  cs, ss, h, r, f = fibreOf110()
  fb = fibre(orientation(rotation.rand(rng=rng), cs, ss), h)
  f2 = fibreODF(fb, halfwidth=12 * degree)
  assert np.isclose(mean(f2), 1) and f2.SS is ss
  f3 = fibreODF(Miller(vector3d.cat(h, Miller(1, 0, 0, cs)), cs), vector3d.cat(r, vector3d.rand(rng=rng)), ss, weights=[1, 3])
  assert np.isclose(mean(f3), 1) and np.isclose(f3.weights[1], 3 * f3.weights[0])
  assert "SO3FunCBF" in repr(f) and "fibre : (110) ||" in repr(f)
  assert isinstance(SO3FunCBF(fb, S2DeLaValleePoussinKernel(halfwidth=12 * degree)), SO3FunCBF)


def test_fibreODF_ofALaueGroupIsTheFibreOfHAndMinusH():
  # a Laue group holds the inversion: the fibre of h is that of -h too, so a -3m fibre ODF
  # takes the same value where g h and where -g h meet r (MATLAB 15.31 on both)
  laue = fibreODF(Miller(1, 1, 0, crystalFrame("-3m")), zvector)
  o = fibre(Miller(1, 0, 0, crystalFrame("-3m")), vector3d.X).orientation(360)[[90, 270]]
  assert np.allclose(laue(o), 15.3, atol=0.1) and np.allclose(SO3FunHarmonic(laue)(o), laue(o), atol=0.1)


# a 15 degree unimodal ODF under 432 and 1, its (100) pole figure and twenty directions
def cubicPoleFigure():
  cs, ss = crystalFrame("432"), specimenFrame("1")
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=15 * degree)
  h = Miller(1, 0, 0, cs)
  return cs, ss, odf, h, radon(odf, h), vector3d.rand(20, rng=rng)


def test_poleFigure_ofAUnimodalODFAgreesWithTheHarmonicRouteAndIsAxisValued():
  cs, ss, odf, h, P, vs = cubicPoleFigure()
  Ph = radon(SO3FunHarmonic(odf), h)
  assert np.max(np.abs(P(vs) - Ph(vs))) < 2e-2 * np.max(P(vs))
  assert P.antipodal and P.frame is ss   # a two-fold axis of 432 sends (100) to (-100)
  assert radon(unimodalODF(orientation(rotation.rand(rng=rng), crystalFrame("m-3m"), ss)), Miller(1, 0, 0, crystalFrame("m-3m"))).antipodal
  assert odf.radon(h, bandwidth=4).bandwidth <= 4


def test_poleFigure_ofTheUniformODFIsOne():
  # the pole figure is the S2 kernel at the images of h, and the uniform ODF gives one
  cs, ss, odf, h, P, vs = cubicPoleFigure()
  assert np.allclose(radon(uniformODF(cs, ss), h)(vs), 1)


def test_inversePoleFigure_radonAndTheMeanOfAPoleFigure():
  # ipdf, radon and the mean of a pole figure
  cs, ss, odf, h, P, vs = cubicPoleFigure()
  I = radon(odf, r=vs[0])
  assert I.frame is cs and np.isclose(mean(P), 1) and np.isclose(mean(I), 1)
  assert np.isclose(radon(odf, h, vs[0]), P(vs[0]))


def test_radon_superposesThePoleFiguresOfAList():
  # check_calcPoleFigureSuperposition
  cs3 = crystalFrame("321")
  vs = vector3d.rand(20, rng=rng)
  odf3 = unimodalODF(orientation(rotation.rand(rng=rng), cs3), halfwidth=15 * degree)
  hs = Miller(vector3d.cat(Miller(1, 0, 0, cs3), Miller(0, 0, 1, cs3)), cs3)
  ref = 0.5 * radon(odf3, hs[0])(vs) + 0.5 * radon(odf3, hs[1])(vs)
  assert np.max(np.abs(radon(odf3, hs, vs) - ref)) < 1e-10
  ref2 = 0.52 * radon(odf3, hs[0])(vs) + 1.23 * radon(odf3, hs[1])(vs)
  assert np.max(np.abs(radon(odf3, hs, vs, superposition=[0.52, 1.23]) - ref2)) < 1e-10
  assert np.allclose(radon(odf3, hs[0], vs), radon(odf3, hs[0])(vs))
  assert np.max(np.abs(radon(odf3, hs, superposition=[0.52, 1.23])(vs) - ref2)) < 1e-10
  assert np.max(np.abs(radon(SO3FunHarmonic(odf3), hs, vs, superposition=[0.52, 1.23]) - ref2)) < 2e-2 * np.max(ref2)
  with pytest.raises(ValueError):
    radon(odf3, hs, vs, superposition=[1.0])


def test_poleFigure_ofAFibreODFGoesThroughItsHarmonicForm():
  # the fibre ODF goes through its harmonic form
  cs, ss, odf, h, P, vs = cubicPoleFigure()
  fodf = fibreODF(Miller(1, 1, 1, cs), zvector, ss)
  assert np.allclose(radon(fodf, h)(vs), radon(SO3FunHarmonic(fodf), h)(vs))


# a narrow kernel is summed over the centres within its support through a cell list: the sum of
# its definition over every centre and every symmetric copy, for the two families summed in the
# compiled loop and for one summed over the pairs, antipodal centres included
@pytest.mark.parametrize("kernel", ["dlvp", "vmf", "bump"])
@pytest.mark.parametrize("antipodal", [False, True])
def test_SO3FunRBF_narrowKernelIsTheSumOfItsDefinition(kernel, antipodal):
  from mtex import SO3BumpKernel, SO3vonMisesFisherKernel
  import mtex.functions.rbf as rbf
  cs = crystalFrame("432")
  ss = cs if antipodal else specimenFrame.default
  psi = {"dlvp": SO3DeLaValleePoussinKernel(halfwidth=4 * degree), "vmf": SO3vonMisesFisherKernel(halfwidth=4 * degree),
         "bump": SO3BumpKernel(halfwidth=4 * degree)}[kernel]
  c, q = orientation.rand(300, cs, ss, rng=rng), orientation.rand(40, cs, ss, rng=rng)
  c.antipodal = antipodal
  w = rng.random(c.size)
  L, R = reductionGroups(c)
  cut = rbf.supportCosine(psi)
  got = rbf._cellSum(q.data.reshape(-1, 4), c.data.reshape(-1, 4), w, L, R, antipodal, psi, cut)
  cs_, ww = SO3FunRBF(c, psi, w)._expandedCentres()
  assert np.allclose(got, rbf._blockSum(q.data.reshape(-1, 4), cs_, ww, psi, cut), rtol=1e-10, atol=1e-12)


# the fibre ODF of the de la Vallee Poussin kernel is summed in a compiled loop, with the values of
# the kernel at the arguments of every fibre pair
def test_SO3FunCBF_compiledSumIsTheKernelAtEveryFibrePair():
  from mtex.geometry.representations import quat2mat
  cs = crystalFrame("m-3m")
  f = fibreODF(fibre(Miller(1, 1, 0, cs), vector3d(1, 2, 3)), halfwidth=8 * degree)
  q = orientation.rand(200, cs, rng=rng)
  H, Rv, w = f._expandedFibres()
  t = np.einsum("nij,ki,kj->nk", quat2mat(q.data.reshape(-1, 4)), Rv, H)
  assert np.allclose(f.eval(q), f.psi(np.clip(t, -1, 1)) @ w, rtol=1e-12)


# the harmonic form kept for evaluating many terms follows the function when its weights change
def test_SO3FunRBF_harmonicFormFollowsChangedWeights():
  cs = crystalFrame("m-3m")
  f = SO3FunRBF(orientation.rand(2000, cs, rng=rng), SO3DeLaValleePoussinKernel(halfwidth=10 * degree), np.full(2000, 1 / 2000))
  q = orientation.rand(2000, cs, rng=rng)
  before = f.eval(q)
  f.weights = 2 * f.weights
  assert np.allclose(f.eval(q), 2 * before)
