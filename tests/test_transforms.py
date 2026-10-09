# the transforms of docs/plans/functions.md F2 against the direct sums of F0
import numpy as np
import pytest
from numpy import pi

import mtex
from mtex.functions.harmonics import _WignerD
from mtex import (Miller, S1FejerKernel, S1FourierPlan, S2FourierPlan, S2GridPlan, SO3FourierPlan, SO3GridPlan,
                  adjointTransform, bandwidthFor, crystalFrame, deg2dim, degreeBlock, folds, fourierSeries,
                  fourierSeriesAdjoint, invariantProjection, isRealFourier, isRealSpherical, isRealWigner, project,
                  quadratureS1Grid, quadratureS2Grid, quadratureSO3Grid, rotation, setMTEXpref, specimenFrame,
                  sphericalSeries, sphericalSeriesAdjoint, sphericalTransform, sphericalTransformAdjoint, vector3d,
                  wignerSeries, wignerSeriesAdjoint, wignerTransform, wignerTransformAdjoint)
from mtex.functions.harmonics import wignerSeries as directWigner
from mtex.functions.transforms import _sphericalSeriesDirect, foldedDims, halfPiDegrees, usableFolds

rng = np.random.default_rng(21)


# a random coefficient vector, real valued when asked
def randomWigner(L, real=False):
  fhat = rng.standard_normal(deg2dim(L + 1)) + 1j * rng.standard_normal(deg2dim(L + 1))
  if real:
    for n in range(L + 1):
      F = degreeBlock(fhat, n)
      k = np.arange(-n, n + 1)
      sign = np.where((k[:, None] - k[None, :]) % 2, -1.0, 1.0)
      F[...] = (F + (sign * np.conj(F))[::-1, ::-1]) / 2
  return fhat


def randomSpherical(L, real=False):
  fhat = rng.standard_normal((L + 1) ** 2) + 1j * rng.standard_normal((L + 1) ** 2)
  if real:
    for l in range(L + 1):
      m = np.arange(-l, l + 1)
      c = fhat[l * l:(l + 1) ** 2]
      c[...] = (c + np.where(m % 2, -1.0, 1.0) * np.conj(c)[::-1]) / 2
  return fhat


def relerr(a, b):
  return np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b)))


# the kernels a test runs on, the compiled ones restored afterwards
@pytest.fixture(params=["numba", "numpy"])
def kernels(request):
  setMTEXpref("kernels", request.param)
  yield request.param
  setMTEXpref("kernels", "numba")


def test_SO3FourierPlan_againstTheDirectSum(kernels):
  for L in (3, 8, 24):
    for real in (False, True):
      fhat = randomWigner(L, real)
      rots = rotation.rand(200, rng=rng)
      direct = directWigner(fhat, rots)
      assert isRealWigner(fhat) == real
      if real:
        assert np.max(np.abs(np.imag(direct))) < 1e-10
        direct = np.real(direct)
      p = SO3FourierPlan(rots, L, real=real)
      f = p(fhat)
      assert (f.dtype.kind != "c") == real
      assert relerr(f, direct) < 1e-6
      # the accuracy is set by the tolerance asked
      assert relerr(SO3FourierPlan(rots, L, real=real, eps=1e-10)(fhat), direct) < 1e-9
      # the same through the cost rule, and for few nodes
      assert np.allclose(wignerSeries(fhat, rots), f)
      assert np.allclose(wignerSeries(fhat, rots[:10]), direct[:10])


# the real half array is the conjugate symmetric full one
def test_wignerTransform_realHalfIsTheConjugateSymmetricFull(kernels):
  fhat = randomWigner(5, real=True)
  G = wignerTransform(fhat, 5)
  assert np.max(np.abs(G - np.conj(G[::-1, ::-1, ::-1]))) < 1e-12
  Gh = wignerTransform(fhat, 5, real=True)
  assert np.allclose(Gh[1:], G[6:]) and np.allclose(Gh[0], G[5] / 2)


# a container keeps its shape
def test_wignerSeries_keepsTheShapeOfTheContainer(kernels):
  fhat = randomWigner(5, real=True)
  rs = rotation.rand(3, 40, rng=rng)
  assert wignerSeries(fhat, rs).shape == (3, 40)


# single precision stays single
def test_SO3FourierPlan_singlePrecisionStaysSingle(kernels):
  f32 = randomWigner(8, real=True).astype(np.complex64)
  rots32 = rotation(rotation.rand(100, rng=rng), dtype=np.float32)
  p32 = SO3FourierPlan(rots32, 8)
  assert p32(f32).dtype == np.float32
  assert relerr(p32(f32), np.real(directWigner(f32.astype(complex), rots32))) < 1e-4


# random coefficients, nodes, weights and values for the adjoint
def adjointCases():
  for L in (4, 12):
    for real in (False, True):
      fhat = randomWigner(L, real)
      rots = rotation.rand(300, rng=rng)
      w = rng.random(300)
      v = rng.standard_normal(300) if real else rng.standard_normal(300) + 1j * rng.standard_normal(300)
      yield L, real, fhat, rots, w, v


def test_SO3FourierPlan_adjointIsTheAdjoint():
  for L, real, fhat, rots, w, v in adjointCases():
    p = SO3FourierPlan(rots, L, real=real)
    lhs = np.sum(w * p(fhat) * np.conj(v))
    rhs = np.sum(fhat * np.conj(adjointTransform(p, v, w)))
    assert abs(lhs - rhs) < 1e-7 * abs(lhs)


# against the direct adjoint sum
def test_adjointTransform_againstTheDirectAdjointSum():
  for L, real, fhat, rots, w, v in adjointCases():
    p = SO3FourierPlan(rots, L, real=real)
    split = wignerSeriesAdjoint(rots[:20], v[:20], L, weights=w[:20]) + SO3FourierPlan(rots[20:], L, real=real).adjoint(v[20:], w[20:])
    assert relerr(adjointTransform(p, v, w), split) < 1e-6


def test_quadratureSO3Grid_recoversTheCoefficients():
  for L in (3, 10, 25):
    for real in (False, True):
      fhat = randomWigner(L, real)
      g = quadratureSO3Grid(L)
      rots, w = g.nodes(), g.weights
      f = wignerSeries(fhat, rots)
      assert relerr(wignerSeriesAdjoint(rots, f, L, weights=w), fhat) < 1e-5
      p = SO3FourierPlan(rots, L, real=real, eps=1e-11)
      assert relerr(p.adjoint(p(fhat), w), fhat) < 1e-10


# a function of higher degree is truncated, not aliased into low degrees exactly
def test_quadratureSO3Grid_doesNotRecoverAHigherDegreeExactly():
  for L in (3, 10, 25):
    for real in (False, True):
      g = quadratureSO3Grid(L)
      rots, w = g.nodes(), g.weights
      fhat2 = randomWigner(L + 2, real)
      got = wignerSeriesAdjoint(rots, wignerSeries(fhat2, rots), L, weights=w)
      assert relerr(got, fhat2[:deg2dim(L + 1)]) > 1e-3


def test_sphericalSeries_againstTheDirectSum():
  for L in (3, 8, 40):
    for real in (False, True):
      fhat = randomSpherical(L, real)
      assert isRealSpherical(fhat) == real
      vs = vector3d.rand(300, rng=rng)
      direct = _sphericalSeriesDirect(fhat, vs)
      if real:
        assert np.max(np.abs(np.imag(direct))) < 1e-10
        direct = np.real(direct)
      f = sphericalSeries(fhat, vs)
      assert (f.dtype.kind != "c") == real
      assert relerr(f, direct) < 1e-6
      assert np.allclose(sphericalSeries(fhat, vs[:10]), direct[:10])
      # the trigonometric sum at a few hundred nodes is exact
      assert relerr(sphericalSeries(fhat, vs[:120]), direct[:120]) < 1e-12


# adjoint identity on the sphere
def test_S2FourierPlan_adjointIsTheAdjoint():
  for L in (3, 8, 40):
    for real in (False, True):
      fhat = randomSpherical(L, real)
      vs = vector3d.rand(300, rng=rng)
      w = rng.random(300)
      v = rng.standard_normal(300) + 1j * rng.standard_normal(300)
      p = S2FourierPlan(vs, L)
      lhs, rhs = np.sum(w * p(fhat) * np.conj(v)), np.sum(fhat * np.conj(p.adjoint(v, w)))
      assert abs(lhs - rhs) < 1e-7 * abs(lhs)


# quadrature on both schemes
def test_quadratureS2Grid_recoversTheCoefficients():
  for L in (3, 8, 40):
    for real in (False, True):
      fhat = randomSpherical(L, real)
      for scheme in ("GaussLegendre", "ClenshawCurtis"):
        g = quadratureS2Grid(L, scheme)
        ns, wq = g.nodes(), g.weights
        assert relerr(sphericalSeriesAdjoint(ns, sphericalSeries(fhat, ns), L, weights=wq), fhat) < 1e-5


# framed directions and a container keep their shape
def test_sphericalSeries_ofFramedDirectionsKeepsTheirShape():
  cs = crystalFrame("m-3m")
  fhat = randomSpherical(4, real=True)
  ms = Miller(vector3d.rand(3, 40, rng=rng), cs)
  assert sphericalSeries(fhat, ms).shape == (3, 40)
  assert np.allclose(sphericalSeries(fhat, ms), np.real(_sphericalSeriesDirect(fhat, vector3d(ms.data))))


# the grid itself takes the FFT route, exact where the NUFFT on its nodes is accurate to 1e-7,
# and brings the weights of its rule with it
def test_wignerSeries_onAGridTakesTheFFTRoute():
  for L in (4, 11):
    for real in (False, True):
      fhat = randomWigner(L, real)
      g = quadratureSO3Grid(L)
      v = wignerSeries(fhat, g)
      assert relerr(v, wignerSeries(fhat, g.nodes())) < 1e-5
      assert relerr(wignerSeriesAdjoint(g, v, L), fhat) < 1e-12


# a Gauss-Legendre grid has no equispaced polar rule and goes back through the NUFFT
def test_wignerSeriesAdjoint_onAGaussLegendreGridRecoversTheCoefficients():
  for L in (4, 11):
    for real in (False, True):
      fhat = randomWigner(L, real)
      gl = quadratureSO3Grid(L, "GaussLegendre")
      assert relerr(wignerSeriesAdjoint(gl, wignerSeries(fhat, gl.nodes()), L), fhat) < 1e-5


# a bandwidth the grid cannot hold is evaluated at its nodes rather than refused
def test_wignerSeries_onAGridTooSmallForTheBandwidthEvaluatesAtItsNodes():
  for L in (4, 11):
    for real in (False, True):
      g = quadratureSO3Grid(L)
      wide = randomWigner(L + 1, real)
      assert relerr(wignerSeries(wide, g), wignerSeries(wide, g.nodes())) < 1e-5


# an adjoint's real half lattice follows the values, independently of its forward plan
def test_SO3GridPlan_adjointFollowsTheValuesNotThePlan():
  for folded in (False, True):
    for real in (False, True):
      for planReal in (False, True):
        L = 4
        fhat = randomWigner(L, real)
        if folded:
          for n in range(L + 1):
            F = degreeBlock(fhat, n)
            k = np.arange(-n, n + 1)
            F[np.ix_(k % 2 == 1, np.ones(2 * n + 1, bool))] = 0
            F[:, k % 2 == 1] = 0
        g = quadratureSO3Grid(7, rA=2 if folded else 1, rB=2 if folded else 1)
        p = SO3GridPlan(g, L, real=planReal)
        v = wignerSeries(fhat, g, real=real)
        assert relerr(p.adjoint(v, g.weights), fhat) < 1e-12
        if real == planReal:
          assert relerr(p(fhat), v) < 1e-12


# the sphere: the grid takes the FFT route and brings its weights
def test_sphericalSeries_onAGridTakesTheFFTRoute():
  for L in (4, 11):
    for real in (False, True):
      fhat = randomSpherical(L, real)
      g = quadratureS2Grid(L)
      v = sphericalSeries(fhat, g)
      assert relerr(v, sphericalSeries(fhat, g.nodes())) < 1e-5
      assert relerr(sphericalSeriesAdjoint(g, v, L), fhat) < 1e-12


# the sphere: a Gauss-Legendre grid goes back through the NUFFT
def test_sphericalSeriesAdjoint_onAGaussLegendreGridRecoversTheCoefficients():
  for L in (4, 11):
    for real in (False, True):
      fhat = randomSpherical(L, real)
      gl = quadratureS2Grid(L, "GaussLegendre")
      assert relerr(sphericalSeriesAdjoint(gl, sphericalSeries(fhat, gl.nodes()), L), fhat) < 1e-5


# a wider spherical grid and reused plans retain both complex and real adjoints
def test_S2GridPlan_onAWiderGridKeepsComplexAndRealAdjoints():
  for dtype in (np.float32, np.float64):
    for real in (False, True):
      for planReal in (False, True):
        L = 4
        fhat = randomSpherical(L, real).astype(np.result_type(dtype, 1j))
        g = quadratureS2Grid(7, dtype=dtype)
        p = S2GridPlan(g, L, real=planReal)
        v = sphericalSeries(fhat, g, real=real)
        tol = 5e-6 if dtype is np.float32 else 1e-12
        assert relerr(p.adjoint(v, g.weights), fhat) < tol
        if real == planReal:
          assert relerr(p(fhat), v) < tol


# a function with a 4-fold axis along z on the crystal side and a 2-fold on the specimen side
def symmetricFunction():
  cs, ss = crystalFrame("4/mmm"), specimenFrame("222")
  L = bandwidthFor(10, cs, ss)
  fhat = randomWigner(L, real=True)
  P, Q = invariantProjection(ss, L), invariantProjection(cs, L)
  project(fhat, P, Q)
  return cs, ss, L, fhat, P, Q


# the symmetric function is integrated on one period of gamma and alpha, and the projection
# removes what the smaller node set aliases
def test_quadratureSO3Grid_ofTheFoldsRecoversASymmetricFunction():
  cs, ss, L, fhat, P, Q = symmetricFunction()
  g = quadratureSO3Grid(L, cs, ss)
  rots, w = g.nodes(), g.weights
  got = project(wignerSeriesAdjoint(rots, wignerSeries(fhat, rots), L, weights=w), P, Q)
  assert relerr(got, fhat) < 1e-5
  onGrid = project(wignerSeriesAdjoint(g, wignerSeries(fhat, g), L), P, Q)
  assert relerr(onGrid, fhat) < 1e-12


# told the folds, the transforms at arbitrary nodes run on the reduced index set
def test_wignerSeriesAdjoint_toldTheFoldsAgreesAtArbitraryNodes():
  cs, ss, L, fhat, P, Q = symmetricFunction()
  os = rotation.rand(400, rng=rng)
  vals = wignerSeries(fhat, os)
  cut = project(wignerSeriesAdjoint(os, vals, L, folds=folds(cs, ss)), P, Q)
  raw = project(wignerSeriesAdjoint(os, vals, L), P, Q)
  assert relerr(cut, raw) < 1e-5


def test_usableFolds_dropsTheFoldsABandwidthCannotUse():
  assert usableFolds(6, True, (6, 1)) == (1, 1) and usableFolds(32, True, (2, 1)) == (2, 1)


def test_foldedDims_ofAHalfLatticeWithAFold():
  assert foldedDims(32, True, (2, 1)) == (17, 65, 65)


# a function of the crystal group 222 and the specimen group 222, its values at random nodes
def orthorhombicFunction():
  cs2, ss = crystalFrame("222"), specimenFrame("222")
  L2 = 32
  fh = project(randomWigner(L2, real=True), invariantProjection(ss, L2), invariantProjection(cs2, L2))
  ns = rotation.rand(2000, rng=rng)
  return cs2, L2, fh, ns, wignerSeries(fh, ns, real=True)


def test_wignerSeries_toldTheFoldsAgreesWithTheFullSeries():
  cs2, L2, fh, ns, full = orthorhombicFunction()
  assert relerr(wignerSeries(fh, ns, real=True, folds=folds(cs2)), full) < 1e-5


# where it is taken, the coefficients the folds forbid are not computed at all
def test_wignerSeriesAdjoint_toldTheFoldsSkipsTheForbiddenCoefficients():
  cs2, L2, fh, ns, full = orthorhombicFunction()
  a0 = wignerSeriesAdjoint(ns, full, L2)
  a1 = wignerSeriesAdjoint(ns, full, L2, folds=folds(cs2))
  allowed = np.zeros(len(a0), bool)
  for n in range(L2 + 1):
    B = degreeBlock(allowed, n)
    l = np.arange(-n, n + 1)
    B[:, l % 2 == 0] = True
  assert relerr(a1[allowed], a0[allowed]) < 1e-6
  assert np.all(a1[~allowed] == 0) and np.linalg.norm(a0[~allowed]) > 1e-3


# random Fourier coefficients and angles on the circle
def circleCases():
  for L in (3, 20):
    chat = rng.standard_normal(2 * L + 1) + 1j * rng.standard_normal(2 * L + 1)
    xs = rng.random(300) * 2 * pi
    yield L, chat, xs


def test_fourierSeries_againstTheDirectSum():
  for L, chat, xs in circleCases():
    direct = np.exp(1j * np.outer(xs, np.arange(-L, L + 1))) @ chat
    assert relerr(fourierSeries(chat, xs), direct) < 1e-6
    assert np.allclose(fourierSeries(chat, xs[:5]), direct[:5])


def test_quadratureS1Grid_recoversTheCoefficients():
  for L, chat, xs in circleCases():
    g = quadratureS1Grid(L)
    assert relerr(fourierSeriesAdjoint(g, fourierSeries(chat, g), L), chat) < 1e-6


# a real function: c_{-k} = conj(c_k)
def test_fourierSeries_ofConjugateSymmetricCoefficientsIsReal():
  for L, chat, xs in circleCases():
    creal = np.concatenate([np.conj(chat[L + 1:][::-1]), [1.0], chat[L + 1:]])
    assert isRealFourier(creal) and not isRealFourier(chat)
    assert fourierSeries(creal, xs).dtype == np.float64


# the kernel series of F1
def test_fourierSeries_ofAKernelIsTheKernel():
  for L, chat, xs in circleCases():
    psi = S1FejerKernel(L)
    assert relerr(fourierSeries(psi.fhat, xs), psi(xs)) < 1e-6


def test_wignerdDegrees_agreeWithTheHalfPiTable():
  from mtex.functions.harmonics import wignerdDegrees
  for dtype in (np.float32, np.float64):
    tol = 2e-5 if dtype is np.float32 else 1e-12
    table = list(halfPiDegrees(32, dtype))
    for (n, d), (m, ref) in zip(wignerdDegrees(dtype(pi / 2), 32), table):
      assert n == m and np.max(np.abs(d - ref)) < tol
      # d_n^{j0}(pi/2) vanishes for odd n + j, to the rounding of cos(pi/2)
      assert np.max(np.abs(d[1::2, n])) < tol if n else True


def test_transforms_onStreamedDegreesAgreeWithTheTable():
  from mtex.functions.harmonics import wignerdDegrees
  for dtype in (np.float32, np.float64):
    tol = 2e-5 if dtype is np.float32 else 1e-12
    L = 16
    for real in (False, True):
      fhat = randomWigner(L, real).astype(np.result_type(dtype, 1j))
      streamed = wignerdDegrees(dtype(pi / 2), L)
      G = wignerTransform(fhat, L, real=real)
      assert relerr(wignerTransform(fhat, L, real=real, degrees=streamed), G) < tol
      assert relerr(wignerTransformAdjoint(G, L, real=real, degrees=wignerdDegrees(dtype(pi / 2), L)), wignerTransformAdjoint(G, L, real=real)) < tol
      c = randomSpherical(L, real).astype(np.result_type(dtype, 1j))
      H = sphericalTransform(c, L, real=real)
      assert relerr(sphericalTransform(c, L, real=real, degrees=wignerdDegrees(dtype(pi / 2), L)), H) < tol
      assert relerr(sphericalTransformAdjoint(H, L, real=real, degrees=wignerdDegrees(dtype(pi / 2), L)), sphericalTransformAdjoint(H, L, real=real)) < tol


# random quaternions and directions with one node that is not finite
def badNodes():
  q = rotation.rand(500, rng=rng).data.reshape(-1, 4).copy()
  q[7] = np.nan
  v = vector3d.rand(500, rng=rng).data.reshape(-1, 3).copy()
  v[3] = np.nan
  return q, v


# the nodes of a plan from one compiled loop and from NumPy, on the same quaternions and directions
def nodesBothWays():
  from mtex.functions.kernels import s2Nodes, so3Nodes
  q, v = badNodes()
  out = {}
  for kernels in ("numba", "numpy"):
    setMTEXpref("kernels", kernels)
    out[kernels] = (so3Nodes(q, (4, 2), 17), s2Nodes(v, 9))
  setMTEXpref("kernels", "numba")
  return out


# the nodes of a plan from one compiled loop: the same as the Euler angles
def test_planNodes_kernelAgainstTheNumpyForm():
  out = nodesBothWays()
  for a, b in zip(out["numba"][0] + out["numba"][1], out["numpy"][0] + out["numpy"][1]):
    assert np.allclose(np.where(np.isnan(a), 0, a) if a is not None else 0, np.where(np.isnan(b), 0, b) if b is not None else 0, atol=1e-12)


# the nodes kernel marks the node that is not finite
def test_planNodes_kernelMarksTheNodesNotFinite():
  out = nodesBothWays()
  assert np.flatnonzero(out["numba"][0][4]).tolist() == [7] and np.flatnonzero(out["numba"][1][3]).tolist() == [3]


# a node that is not finite evaluates to NaN
def test_SO3FourierPlan_atANodeNotFiniteIsNaN():
  q, _ = badNodes()
  L, fhat = 6, randomWigner(6, real=True)
  f = SO3FourierPlan(rotation(q), L)(fhat)
  assert np.isnan(f[7]) and relerr(np.delete(f, 7), directWigner(fhat, rotation(np.delete(q, 7, axis=0))).real) < 1e-6


# a node that is not finite adds nothing to the adjoint
def test_SO3FourierPlan_adjointOfANodeNotFiniteAddsNothing():
  q, _ = badNodes()
  L = 6
  good = np.delete(np.arange(500), 7)
  w = rng.random(500)
  a = SO3FourierPlan(rotation(q), L).adjoint(np.ones(500), w)
  assert relerr(a, SO3FourierPlan(rotation(q[good]), L).adjoint(np.ones(499), w[good])) < 1e-10


# a plan builds the direction it is asked for
def test_SO3FourierPlan_isLazy():
  L, fhat = 8, randomWigner(8, real=True)
  p = SO3FourierPlan(rotation.rand(300, rng=rng), L)
  assert not p._plans
  p(fhat)
  assert list(p._plans) == [2]


# a plan follows new nodes of the same count
def test_setNodes_movesThePlanToTheNewNodes():
  L, fhat = 8, randomWigner(8, real=True)
  r1, r2 = rotation.rand(300, rng=rng), rotation.rand(300, rng=rng)
  p = SO3FourierPlan(r1, L)
  p(fhat)
  p.setNodes(r2)
  assert relerr(p(fhat), SO3FourierPlan(r2, L)(fhat)) < 1e-12
  assert relerr(p.adjoint(np.ones(300), np.ones(300)), SO3FourierPlan(r2, L).adjoint(np.ones(300), np.ones(300))) < 1e-12
  c = randomSpherical(8, real=True)
  v1, v2 = vector3d.rand(300, rng=rng), vector3d.rand(300, rng=rng)
  q = S2FourierPlan(v1, 8, real=True)
  q.setNodes(v2)
  assert relerr(q(c), S2FourierPlan(v2, 8, real=True)(c)) < 1e-12


# real valued coefficients are told from complex ones, on the compiled pass and the NumPy form
def test_isReal_detectsAnImaginaryDefect(kernels):
  for L in (0, 1, 5):
    w, s = randomWigner(L, real=True), randomSpherical(L, real=True)
    assert isRealWigner(w) and isRealSpherical(s)
    if L:
      assert not isRealWigner(w + 1e-6j) and not isRealSpherical(s + 1e-6j)


# realizing makes a list of coefficients real and keeps real ones, on the compiled pass and the
# NumPy form, on single vectors and lists
def test_realize_projectsOntoTheRealCoefficients(kernels):
  from mtex.functions.kernels import realizeSpherical, realizeWigner
  for L in (0, 1, 5):
    w, s = randomWigner(L, real=True), randomSpherical(L, real=True)
    W = np.stack([randomWigner(L), randomWigner(L)])
    S = np.stack([randomSpherical(L), randomSpherical(L)])
    assert isRealWigner(realizeWigner(W)) and isRealSpherical(realizeSpherical(S))
    assert np.allclose(realizeWigner(w.copy()), w) and np.allclose(realizeSpherical(s.copy()), s)


# the compiled quadrant table holds the lower right quadrant of the table at pi/2
def test_halfPiQuadrants_isTheQuadrantOfTheHalfPiTable():
  from mtex.functions.kernels import halfPiQuadrants, quadrantOffset
  T = halfPiQuadrants(30)
  for n, d in halfPiDegrees(30):
    assert np.max(np.abs(T[quadrantOffset(n):quadrantOffset(n + 1)] - d[n:, n:].ravel())) < 1e-14


# the compiled Wigner transforms on the quadrant table against the NumPy form on the full
# matrices, every fold pair, real and complex, the bandwidths 0 to 3 included
def test_wigner_kernels_against_the_numpy_form():
  for L in (0, 1, 2, 3, 8, 17):
    for real in (False, True):
      for fo in ((1, 1), (2, 1), (1, 3), (4, 2), (6, 6)):
        f = randomWigner(L)
        H = rng.standard_normal(foldedDims(L, real, fo)) + 1j * rng.standard_normal(foldedDims(L, real, fo))
        setMTEXpref("kernels", "numpy")
        G0, A0 = wignerTransform(f, L, real, fo), wignerTransformAdjoint(H, L, real, fo)
        setMTEXpref("kernels", "numba")
        assert np.max(np.abs(wignerTransform(f, L, real, fo) - G0)) < 1e-13 * max(1, np.max(np.abs(G0)))
        assert np.max(np.abs(wignerTransformAdjoint(H, L, real, fo) - A0)) < 1e-13 * max(1, np.max(np.abs(A0)))


def test_direct_series_kernel_against_the_numpy_form():
  from mtex.functions.kernels import wignerSeriesDirect
  edges = rotation.cat(rotation.byEuler(0.3, 0, 0.2), rotation.byEuler(0.3, pi, 0.2), rotation.byEuler(0, 0, 0))
  for L in (0, 1, 2, 5, 17, 40):
    f = np.stack([randomWigner(L), randomWigner(L)])
    for q in (rotation.rand(7, rng=rng).reshape(7, 1), rotation.rand(rng=rng), edges):
      A = directWigner(f, q)
      B = wignerSeriesDirect(f, q)
      assert B.shape == A.shape and np.max(np.abs(B - A)) < 1e-12 * max(1, np.max(np.abs(A)))
      assert np.allclose(wignerSeriesDirect(f[0], q), A[0], atol=1e-12 * max(1, np.max(np.abs(A))))


# the adjoint is linear: complex values on a real plan are the sum of two real adjoints
def test_adjoint_ofComplexValuesOnARealPlanIsTheComplexAdjoint():
  rots, v = rotation.rand(400, rng=rng), vector3d.rand(400, rng=rng)
  w, vals = rng.random(400), rng.random(400) + 1j * rng.random(400)
  a = SO3FourierPlan(rots, 7, real=False).adjoint(vals, w)
  assert relerr(SO3FourierPlan(rots, 7, real=True).adjoint(vals, w), a) < 1e-8
  b = S2FourierPlan(v, 7, real=False).adjoint(vals, w)
  assert relerr(S2FourierPlan(v, 7, real=True).adjoint(vals, w), b) < 1e-8


# the NFFT backend where its library is configured: the adjoint equal to finufft's
@pytest.mark.skipif(not __import__("os").environ.get("MTEX_NFFT_LIBRARY"), reason="no NFFT library configured")
def test_nfftAdjoint_againstFinufft():
  from mtex.functions.nfft import nfftLibrary
  import os
  assert nfftLibrary() is not None
  for L, folds in ((0, (1, 1)), (3, (1, 1)), (12, (4, 2)), (21, (2, 3))):
    q = rotation.rand(900, rng=rng).data.reshape(-1, 4).copy()
    q[5] = np.nan
    rots, w = rotation(q), rng.random(900)
    for vals in (rng.random(900), rng.random(900) + 1j * rng.random(900)):
      p = SO3FourierPlan(rots, L, real=True, folds=folds)
      a1 = p.adjoint(vals, w)
      os.environ["MTEX_NFFT_LIBRARY"], lib = "", os.environ["MTEX_NFFT_LIBRARY"]
      try:
        a0 = SO3FourierPlan(rots, L, real=vals.dtype.kind != "c", folds=folds).adjoint(vals, w)
      finally:
        os.environ["MTEX_NFFT_LIBRARY"] = lib
      assert relerr(a1, a0) < 1e-7
      p.setNodes(rotation.rand(900, rng=rng))
      assert not any(isinstance(k, tuple) for k in p._plans)


# the spherical kernels against the NumPy form
def test_spherical_kernels_against_the_numpy_form():
  for L in (0, 1, 2, 3, 17, 60):
    for real in (False, True):
      f = randomSpherical(L)
      H = rng.standard_normal((L + 1 if real else 2 * L + 1, 2 * L + 1)) + 1j * rng.standard_normal((L + 1 if real else 2 * L + 1, 2 * L + 1))
      setMTEXpref("kernels", "numpy")
      G0, A0 = sphericalTransform(f, L, real=real), sphericalTransformAdjoint(H, L, real=real)
      setMTEXpref("kernels", "numba")
      assert relerr(sphericalTransform(f, L, real=real), G0) < 1e-13
      assert relerr(sphericalTransformAdjoint(H, L, real=real), A0) < 1e-13


# beyond the underflow of 2^-m against the direct sum: the rows past m = 1074 start below the
# smallest double and are carried scaled
def test_sphericalSeries_beyondTheUnderflowAgainstTheDirectSum():
  from mtex.functions.transforms import _sphericalSeriesTrig
  L = 1100
  f = randomSpherical(L)
  v = vector3d.rand(4, rng=rng)
  assert relerr(_sphericalSeriesTrig(f, v, False), _sphericalSeriesDirect(f, v)) < 1e-12


# the FFT route on a Clenshaw-Curtis grid is exact: against the direct sums at its nodes, for
# the folds of the frames, real and complex, a bandwidth below the grid's
def test_SO3GridPlan_againstTheDirectSums():
  for cs, ss in (("1", None), ("432", None), ("622", "222"), ("-3m", None)):
    CS = crystalFrame(cs)
    SS = specimenFrame(ss) if ss else specimenFrame.default
    Lg = bandwidthFor(5, CS, SS)
    g = quadratureSO3Grid(Lg, CS, SS)
    rots = rotation(g.nodes())
    for L in (Lg, Lg - 2):
      P, Q = invariantProjection(SS, L), invariantProjection(CS, L)
      D = _WignerD(rots, bandwidth=L)
      for real in (False, True):
        fhat = project(randomWigner(L, real), P, Q)
        f = directWigner(fhat, rots).ravel()
        p = SO3GridPlan(g, L, real=real)
        assert relerr(p(fhat), np.real(f) if real else f) < 1e-12
        vals = rng.standard_normal(len(g)) + (0 if real else 1j * rng.standard_normal(len(g)))
        expected = np.concatenate([np.sqrt(2 * n + 1) * np.einsum("i,ikl->lk", g.weights * vals, np.conj(D[n])).ravel() for n in range(L + 1)])
        # the grid covers the fundamental region of the folds only: equal on the orders they keep
        keep = project(np.ones(deg2dim(L + 1), complex), P, Q) != 0
        assert relerr(p.adjoint(vals, g.weights)[keep], expected[keep]) < 1e-12


# the FFT route on a Clenshaw-Curtis grid of the sphere is exact: against the direct sums at its
# nodes, real and complex, a bandwidth below the grid's
def test_S2GridPlan_againstTheDirectSums():
  from mtex.functions.harmonics import _sphericalY
  g = quadratureS2Grid(11)
  for L in (8, 11):
    for real in (False, True):
      c = randomSpherical(L, real)
      f = _sphericalSeriesDirect(c, g.nodes()).ravel()
      p = S2GridPlan(g, L, real=real)
      assert relerr(p(c), np.real(f) if real else f) < 1e-12
      vals = rng.standard_normal(len(g)) + (0 if real else 1j * rng.standard_normal(len(g)))
      expected = np.concatenate([np.conj(_sphericalY(l, g.nodes())) @ (g.weights * vals) for l in range(L + 1)])
      assert relerr(p.adjoint(vals, g.weights), expected) < 1e-12


# the gridded adjoint rounds the rotations onto a grid of twice the bandwidth: close to the
# exact adjoint on the coefficients the folds keep, and a NaN rotation adds nothing
def test_wignerSeriesAdjointGridded_againstTheExactAdjoint():
  from mtex.functions.transforms import wignerSeriesAdjointGridded
  for fo in ((1, 1), (3, 2)):
    L = 10
    q = rotation.rand(200000, rng=rng).data.reshape(-1, 4).copy()
    q[3] = np.nan
    rots, w = rotation(q), rng.random(200000)
    keep = np.ones(200000, bool)
    keep[3] = False
    exact = SO3FourierPlan(rotation(q[keep]), L, real=True, folds=fo).adjoint(np.ones(keep.sum()), w[keep])
    got = wignerSeriesAdjointGridded(rots, np.ones(200000), L, weights=w, folds=fo)
    # the coefficients on the lattice of the folds, k = 0 mod rB and l = 0 mod rA
    mask = np.zeros(deg2dim(L + 1), bool)
    for n in range(L + 1):
      k = np.arange(-n, n + 1)
      degreeBlock(mask, n)[...] = (k[:, None] % fo[1] == 0) & (k[None, :] % fo[0] == 0)
    assert relerr(got[mask], exact[mask]) < 0.02


# the adjoint reads the order j at the index centre + j, so the output of an FFT, whose negative
# orders wrap around, needs no unwrapping
def test_wignerTransformAdjoint_readsAWrappedPolarAxis(kernels):
  L = 6
  H = rng.standard_normal(foldedDims(L, True, (1, 1))) + 1j * rng.standard_normal(foldedDims(L, True, (1, 1)))
  W = np.zeros(H.shape[:2] + (4 * L,), dtype=complex)
  W[..., :L + 1], W[..., -L:] = H[..., L:], H[..., :L]
  assert np.allclose(wignerTransformAdjoint(W, L, True, centre=0), wignerTransformAdjoint(H, L, True), atol=1e-13)


# the series on the circle and its adjoint as compiled direct sums, by Horner's scheme and the
# powers of e^{-ix}, are the sums of the exponentials
def test_fourierSum_andItsAdjointAreTheExponentialSums():
  from mtex.functions.kernels import fourierAdjointSum, fourierSum
  L, x = 50, rng.uniform(0, 2 * np.pi, 700)
  c, u = rng.standard_normal(2 * L + 1) + 1j * rng.standard_normal(2 * L + 1), rng.standard_normal(700) + 1j * rng.standard_normal(700)
  E = np.exp(1j * np.outer(x, np.arange(-L, L + 1)))
  assert relerr(fourierSum(c, x), E @ c) < 1e-13 and relerr(fourierAdjointSum(u, x, L), np.conj(E).T @ u) < 1e-13
