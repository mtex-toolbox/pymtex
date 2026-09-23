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


@pytest.mark.parametrize("kernels", ["numba", "numpy"])
def test_T1_wigner_transform_against_the_direct_sum(kernels):
  setMTEXpref("kernels", kernels)
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
  fhat = randomWigner(5, real=True)
  G = wignerTransform(fhat, 5)
  assert np.max(np.abs(G - np.conj(G[::-1, ::-1, ::-1]))) < 1e-12
  Gh = wignerTransform(fhat, 5, real=True)
  assert np.allclose(Gh[1:], G[6:]) and np.allclose(Gh[0], G[5] / 2)
  # a container keeps its shape, single precision stays single
  rs = rotation.rand(3, 40, rng=rng)
  assert wignerSeries(fhat, rs).shape == (3, 40)
  f32 = randomWigner(8, real=True).astype(np.complex64)
  rots32 = rotation(rotation.rand(100, rng=rng), dtype=np.float32)
  p32 = SO3FourierPlan(rots32, 8)
  assert p32(f32).dtype == np.float32
  assert relerr(p32(f32), np.real(directWigner(f32.astype(complex), rots32))) < 1e-4
  setMTEXpref("kernels", "numba")


def test_T2_the_adjoint_is_the_adjoint():
  for L in (4, 12):
    for real in (False, True):
      fhat = randomWigner(L, real)
      rots = rotation.rand(300, rng=rng)
      w = rng.random(300)
      v = rng.standard_normal(300) if real else rng.standard_normal(300) + 1j * rng.standard_normal(300)
      p = SO3FourierPlan(rots, L, real=real)
      lhs = np.sum(w * p(fhat) * np.conj(v))
      rhs = np.sum(fhat * np.conj(adjointTransform(p, v, w)))
      assert abs(lhs - rhs) < 1e-7 * abs(lhs)
      # against the direct adjoint sum
      split = wignerSeriesAdjoint(rots[:20], v[:20], L, weights=w[:20]) + SO3FourierPlan(rots[20:], L, real=real).adjoint(v[20:], w[20:])
      assert relerr(adjointTransform(p, v, w), split) < 1e-6


def test_T3_quadrature_recovers_the_coefficients():
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
      fhat2 = randomWigner(L + 2, real)
      got = wignerSeriesAdjoint(rots, wignerSeries(fhat2, rots), L, weights=w)
      assert relerr(got, fhat2[:deg2dim(L + 1)]) > 1e-3


def test_T4_the_sphere():
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
      # adjoint identity and quadrature on both schemes
      w = rng.random(300)
      v = rng.standard_normal(300) + 1j * rng.standard_normal(300)
      p = S2FourierPlan(vs, L)
      lhs, rhs = np.sum(w * p(fhat) * np.conj(v)), np.sum(fhat * np.conj(p.adjoint(v, w)))
      assert abs(lhs - rhs) < 1e-7 * abs(lhs)
      for scheme in ("GaussLegendre", "ClenshawCurtis"):
        g = quadratureS2Grid(L, scheme)
        ns, wq = g.nodes(), g.weights
        assert relerr(sphericalSeriesAdjoint(ns, sphericalSeries(fhat, ns), L, weights=wq), fhat) < 1e-5
  # framed directions and a container keep their shape
  cs = crystalFrame("m-3m")
  fhat = randomSpherical(4, real=True)
  ms = Miller(vector3d.rand(3, 40, rng=rng), cs)
  assert sphericalSeries(fhat, ms).shape == (3, 40)
  assert np.allclose(sphericalSeries(fhat, ms), np.real(_sphericalSeriesDirect(fhat, vector3d(ms.data))))


def test_T5_grids_and_folds():
  for L in (4, 11):
    for real in (False, True):
      fhat = randomWigner(L, real)
      g = quadratureSO3Grid(L)
      rots, w = g.nodes(), g.weights
      assert relerr(wignerSeriesAdjoint(rots, wignerSeries(fhat, rots), L, weights=w), fhat) < 1e-5
      # the grid itself takes the FFT route, exact where the NUFFT on its nodes is accurate to
      # 1e-7, and brings the weights of its rule with it
      v = wignerSeries(fhat, g)
      assert relerr(v, wignerSeries(fhat, rots)) < 1e-5
      assert relerr(wignerSeriesAdjoint(g, v, L), fhat) < 1e-12
      # a Gauss-Legendre grid has no equispaced polar rule and goes back through the NUFFT
      gl = quadratureSO3Grid(L, "GaussLegendre")
      assert relerr(wignerSeriesAdjoint(gl, wignerSeries(fhat, gl.nodes()), L), fhat) < 1e-5
      # a bandwidth the grid cannot hold is evaluated at its nodes rather than refused
      wide = randomWigner(L + 1, real)
      assert relerr(wignerSeries(wide, g), wignerSeries(wide, rots)) < 1e-5
  # an adjoint's real half lattice follows the values, independently of its forward plan
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
  # the sphere, the same claims
  for L in (4, 11):
    for real in (False, True):
      fhat = randomSpherical(L, real)
      g = quadratureS2Grid(L)
      v = sphericalSeries(fhat, g)
      assert relerr(v, sphericalSeries(fhat, g.nodes())) < 1e-5
      assert relerr(sphericalSeriesAdjoint(g, v, L), fhat) < 1e-12
      gl = quadratureS2Grid(L, "GaussLegendre")
      assert relerr(sphericalSeriesAdjoint(gl, sphericalSeries(fhat, gl.nodes()), L), fhat) < 1e-5
  # a wider spherical grid and reused plans retain both complex and real adjoints
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
  # a function with a 4-fold axis along z on the crystal side and a 2-fold on the specimen
  # side is integrated on one period of gamma and alpha, and the projection removes what
  # the smaller node set aliases
  cs, ss = crystalFrame("4/mmm"), specimenFrame("222")
  L = bandwidthFor(10, cs, ss)
  fhat = randomWigner(L, real=True)
  P, Q = invariantProjection(ss, L), invariantProjection(cs, L)
  project(fhat, P, Q)
  g = quadratureSO3Grid(L, cs, ss)
  rots, w = g.nodes(), g.weights
  got = project(wignerSeriesAdjoint(rots, wignerSeries(fhat, rots), L, weights=w), P, Q)
  assert relerr(got, fhat) < 1e-5
  onGrid = project(wignerSeriesAdjoint(g, wignerSeries(fhat, g), L), P, Q)
  assert relerr(onGrid, fhat) < 1e-12
  # told the folds, the transforms at arbitrary nodes run on the reduced index set
  os = rotation.rand(400, rng=rng)
  vals = wignerSeries(fhat, os)
  cut = project(wignerSeriesAdjoint(os, vals, L, folds=folds(cs, ss)), P, Q)
  raw = project(wignerSeriesAdjoint(os, vals, L), P, Q)
  assert relerr(cut, raw) < 1e-5
  assert usableFolds(6, True, (6, 1)) == (1, 1) and usableFolds(32, True, (2, 1)) == (2, 1)
  assert foldedDims(32, True, (2, 1)) == (17, 65, 65)
  # where it is taken, the coefficients the folds forbid are not computed at all
  cs2 = crystalFrame("222")
  L2 = 32
  fh = project(randomWigner(L2, real=True), invariantProjection(ss, L2), invariantProjection(cs2, L2))
  ns = rotation.rand(2000, rng=rng)
  full = wignerSeries(fh, ns, real=True)
  assert relerr(wignerSeries(fh, ns, real=True, folds=folds(cs2)), full) < 1e-5
  a0 = wignerSeriesAdjoint(ns, full, L2)
  a1 = wignerSeriesAdjoint(ns, full, L2, folds=folds(cs2))
  allowed = np.zeros(len(a0), bool)
  for n in range(L2 + 1):
    B = degreeBlock(allowed, n)
    l = np.arange(-n, n + 1)
    B[:, l % 2 == 0] = True
  assert relerr(a1[allowed], a0[allowed]) < 1e-6
  assert np.all(a1[~allowed] == 0) and np.linalg.norm(a0[~allowed]) > 1e-3


def test_T6_the_circle():
  for L in (3, 20):
    chat = rng.standard_normal(2 * L + 1) + 1j * rng.standard_normal(2 * L + 1)
    xs = rng.random(300) * 2 * pi
    direct = np.exp(1j * np.outer(xs, np.arange(-L, L + 1))) @ chat
    assert relerr(fourierSeries(chat, xs), direct) < 1e-6
    assert np.allclose(fourierSeries(chat, xs[:5]), direct[:5])
    g = quadratureS1Grid(L)
    assert relerr(fourierSeriesAdjoint(g, fourierSeries(chat, g), L), chat) < 1e-6
    # a real function: c_{-k} = conj(c_k), and the kernel series of F1
    creal = np.concatenate([np.conj(chat[L + 1:][::-1]), [1.0], chat[L + 1:]])
    assert isRealFourier(creal) and not isRealFourier(chat)
    assert fourierSeries(creal, xs).dtype == np.float64
    psi = S1FejerKernel(L)
    assert relerr(fourierSeries(psi.fhat, xs), psi(xs)) < 1e-6


def test_streamed_degrees_agree_with_the_table():
  from mtex.functions.harmonics import wignerdDegrees
  for dtype in (np.float32, np.float64):
    tol = 2e-5 if dtype is np.float32 else 1e-12
    table = list(halfPiDegrees(32, dtype))
    for (n, d), (m, ref) in zip(wignerdDegrees(dtype(pi / 2), 32), table):
      assert n == m and np.max(np.abs(d - ref)) < tol
      # d_n^{j0}(pi/2) vanishes for odd n + j, to the rounding of cos(pi/2)
      assert np.max(np.abs(d[1::2, n])) < tol if n else True
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
