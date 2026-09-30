# rotation, convolution, gradient, Laplacian, Radon transforms and the pointwise operations
# of the harmonic functions (docs/plans/functions.md F3f to F3j)
import numpy as np
import pytest
from numpy import pi

from mtex import (conv, crystalFrame, deg2dim, degreeBlock, dot, grad, invRadon, laplace, Miller, norm, orientation, quadratureS2Grid, quadratureSO3Grid, radon, rotate, rotation, S1FejerKernel, S1FunHarmonic, S2DeLaValleePoussinKernel, S2FunHarmonic, SO3DeLaValleePoussinKernel, SO3FunHarmonic, SO3TangentVector, specimenFrame, vector3d, xvector, yvector, zvector)

rng = np.random.default_rng(36)


def randomCoefficients(n):
  return rng.standard_normal(n) + 1j * rng.standard_normal(n)


def relclose(a, b, rtol=1e-6):
  return np.allclose(a, b, rtol=rtol, atol=rtol * np.max(np.abs(b)))


# a random real function of degree 7 under -3m and 222
def trigonalFunction():
  cs, ss = crystalFrame("-3m"), specimenFrame("222")
  return cs, ss, SO3FunHarmonic(randomCoefficients(deg2dim(7)), cs, ss, real=True)


def test_rotate_isTheFunctionOfTheMovedArgumentAndStripsTheMovedSide():
  # check_SO3FunRotate: the rotated function is the function of the moved argument
  cs, ss, f = trigonalFunction()
  r = rotation.rand(rng=rng)
  rs = rotation.rand(8, rng=rng)
  fl = rotate(f, r)
  assert relclose(fl(rs), f(r.inv() * rs))
  assert fl.SS is ss.stripSym() and fl.CS is cs
  fr = rotate(f, r, "right")
  assert relclose(fr(rs), f(rs * r.inv()))
  assert fr.CS is cs.stripSym() and fr.SS is ss


def test_rotate_byAGroupElementKeepsTheFrameAndTheFunction():
  # an element of the group keeps the frame and the function
  cs, ss, f = trigonalFunction()
  e = ss.sym.rotations().flatten()[1]
  assert rotate(f, e).isclose(f) and rotate(f, e).SS is ss


def test_rotate_byAnOrientationMovesTheFrameOfTheRotatedSide():
  # an orientation moves the frame of the rotated side
  cs, ss, f = trigonalFunction()
  r = rotation.rand(rng=rng)
  rs = rotation.rand(8, rng=rng)
  ss2 = specimenFrame("mmm")
  o = orientation(r, ss, ss2)
  assert rotate(f, o).SS is ss2
  assert relclose(rotate(f, o)(rs), rotate(f, r)(rs))
  with pytest.raises(ValueError):
    rotate(f, orientation(r, ss2, ss))


def test_rotate_onTheSphere():
  cs = crystalFrame("-3m")
  r = rotation.rand(rng=rng)
  s = S2FunHarmonic(randomCoefficients(36), real=True)
  vs = vector3d.rand(8, rng=rng)
  assert relclose(rotate(s, r)(vs), s(r.inv() * vs))
  s2 = S2FunHarmonic(randomCoefficients(36), cs, real=True)
  e = cs.sym.rotations().flatten()[1]
  assert rotate(s2, e).isclose(s2) and rotate(s2, e).frame is cs
  assert rotate(s2, r).frame is cs.stripSym()


# a random function under 4/mmm and 222, and the quadrature of degree 8 to check integrals over SO(3)
def tetragonalFunction():
  csA, ss = crystalFrame("4/mmm"), specimenFrame("222")
  f = SO3FunHarmonic(randomCoefficients(deg2dim(5)), csA, ss, real=True)
  return csA, ss, f, quadratureSO3Grid(8)


def test_conv_ofTwoSO3FunctionsIsTheirIntegral():
  # SO(3) x SO(3): (f * g)(R) = int f(R S^-1) g(S) dS, checked by quadrature
  csA, ss, f, q = tetragonalFunction()
  csB = crystalFrame("m-3m")
  g = SO3FunHarmonic(randomCoefficients(deg2dim(5)), csB, csA, real=True)
  h = conv(f, g)
  assert h.CS is csB and h.SS is ss
  S, w = q.nodes(), q.weights
  for R in rotation.rand(3, rng=rng):
    assert np.isclose(h(R), np.sum(w * f(R * S.inv()) * g(S)), rtol=1e-5)
  with pytest.raises(ValueError):
    conv(g, f)


def test_conv_withAKernelIsItsCoefficientsOver2nPlus1():
  # a kernel convolves as its coefficients over 2n+1
  csA, ss, f, q = tetragonalFunction()
  psi = SO3DeLaValleePoussinKernel(20)
  k = SO3FunHarmonic(psi, csA, csA, symmetrise=False)
  assert conv(f, psi).isclose(conv(f, k), 1e-10)


def test_conv_ofAnSO3AndAnS2FunctionIsTheirIntegralInTheCrystalFrame():
  # SO(3) x S2: (f * s)(h) = int f(R) s(R h) dR in the crystal frame
  csA, ss, f, q = tetragonalFunction()
  S, w = q.nodes(), q.weights
  s = S2FunHarmonic(randomCoefficients(25), ss, real=True)
  fs = conv(f, s)
  assert fs.frame is csA
  scale = np.max(np.abs(fs(vector3d.rand(20, rng=rng))))
  for hh in vector3d.rand(3, rng=rng):
    assert abs(fs(hh) - np.sum(w * f(S) * s(S * hh))) < 1e-6 * scale


def test_conv_ofTwoS2FunctionsIsTheirIntegral():
  # S2 x S2: (s * t)(R) = (1/4pi) int s(xi) t(R xi) dxi
  s1 = S2FunHarmonic(randomCoefficients(25), real=True)
  t1 = S2FunHarmonic(randomCoefficients(25), real=True)
  st = conv(s1, t1)
  qs = quadratureS2Grid(8)
  xi, wxi = qs.nodes(), qs.weights
  for R in rotation.rand(3, rng=rng):
    assert np.isclose(st(R), np.sum(wxi * s1(xi) * t1(R * xi)) / (4 * pi), rtol=1e-5)


def test_conv_ofAnS2FunctionAndAKernelIsTheirIntegral():
  # S2 x kernel: (s * psi)(v) = (1/4pi) int s(xi) psi(<xi, v>) dxi
  s1 = S2FunHarmonic(randomCoefficients(25), real=True)
  psi2 = S2DeLaValleePoussinKernel(20)
  sk = conv(s1, psi2)
  qk = quadratureS2Grid(12)
  for v in vector3d.rand(3, rng=rng):
    assert np.isclose(sk(v), np.sum(qk.weights * s1(qk.nodes()) * psi2(qk.nodes().dot(v))) / (4 * pi), rtol=1e-5)


def test_conv_withAMultiplierPerDegree():
  # a multiplier per degree
  csA, ss, f, q = tetragonalFunction()
  assert np.allclose(conv(f, [1, 0, 1]).fhat[1:10], 0) and conv(f, [1, 0, 1]).bandwidth == 2


def test_grad_isTheFiniteDifferenceOnBothSides():
  cs, ss, f = trigonalFunction()
  d = 1e-5
  os = orientation(rotation.rand(4, rng=rng), cs, ss)
  for side in ("left", "right"):
    for i in range(4):
      o = os[i]
      t = grad(f, o, side)
      assert isinstance(t, SO3TangentVector) and t.side == side
      r = rotation(o)
      for v in (xvector, yvector, zvector):
        plus = rotation.byAxisAngle(v, d) * r if side == "left" else r * rotation.byAxisAngle(v, d)
        minus = rotation.byAxisAngle(v, -d) * r if side == "left" else r * rotation.byAxisAngle(v, -d)
        fd = (f(plus) - f(minus)) / (2 * d)
        assert abs(vector3d(t.data).dot(v) - fd) < 1e-6 * (1 + abs(fd))


def test_grad_keepsTheShapeAndIsAVectorField():
  cs, ss, f = trigonalFunction()
  os = orientation(rotation.rand(4, rng=rng), cs, ss)
  ts = grad(f, orientation(rotation.rand(3, 2, rng=rng), cs, ss))
  assert ts.shape == (3, 2)
  G = grad(f)
  assert np.allclose(G.component(2)(rotation(os)), grad(f, os).z)


def test_laplace_multipliesByTheEigenvalues():
  # the Laplacian by the eigenvalues, checked on a kernel
  cs, ss = crystalFrame("-3m"), specimenFrame("222")
  psi = SO3DeLaValleePoussinKernel(10)
  k = SO3FunHarmonic(psi, cs, ss, symmetrise=False)
  expect = np.concatenate([-(n * (n + 1)) * k.fhat[deg2dim(n):deg2dim(n + 1)] for n in range(k.bandwidth + 1)])
  assert np.allclose(laplace(k).fhat, expect)
  s = S2FunHarmonic(randomCoefficients(36), real=True)
  assert np.allclose(laplace(s).fhat[4:9], -6 * s.fhat[4:9])


def test_norm_ofOrderZeroIsTheL2NormAndTheSobolevNormIsLarger():
  cs, ss, f = trigonalFunction()
  assert np.isclose(norm(f, 0), norm(f)) and f.norm(1) > norm(f)


def test_diff_ofACircleFunctionIsItsGradient():
  assert np.allclose(S1FunHarmonic(S1FejerKernel(4)).diff()(0.3), grad(S1FunHarmonic(S1FejerKernel(4)), 0.3))


# the pole figure is the mean of f over the fibre R h = r
def fibreMean(f, h, r):
  R0 = rotation.map(h.normalize(), r.normalize())
  phis = 2 * pi * np.arange(16) / 16
  return np.mean([f(rotation.byAxisAngle(r.normalize(), p) * R0) for p in phis])


# a random positive-mean function of degree 6 under 432 and 1
def cubicFunction():
  cs, ss = crystalFrame("432"), specimenFrame("1")
  return cs, ss, SO3FunHarmonic(randomCoefficients(deg2dim(6)), cs, ss, real=True) + 3


def test_poleFigure_isTheMeanOverTheFibre():
  cs, ss, f = cubicFunction()
  h = Miller(1, 2, 0, cs)
  P = radon(f, h)
  # a two-fold axis of 432 sends (120) to (-1-20), so this pole figure is axis valued; one
  # of a trigonal crystal is not
  assert P.frame is ss and P.isReal and P.antipodal
  assert not radon(SO3FunHarmonic(randomCoefficients(deg2dim(4)), crystalFrame("3"), ss, real=True), Miller(1, 2, 0, crystalFrame("3"))).antipodal
  for r in vector3d.rand(4, rng=rng):
    assert np.isclose(P(r), fibreMean(f, vector3d(h.data), r), rtol=1e-8)
    assert np.isclose(radon(f, h, r), P(r))
  assert f.radon(h).isclose(P)


def test_inversePoleFigure_isTheMeanOverTheFibre():
  cs, ss, f = cubicFunction()
  r = vector3d.rand(rng=rng)
  I = radon(f, r=r)
  assert I.frame is cs
  for hh in vector3d.rand(4, rng=rng):
    assert np.isclose(I(hh), fibreMean(f, hh, r), rtol=1e-8)
  assert f.radon(r=r).isclose(I)


def test_poleFigure_ofTheUniformFunctionIsOneAndOfALaueCrystalIsAxisValued():
  # the uniform function has pole figure one; a Laue crystal gives an axis valued pole figure
  cs, ss, f = cubicFunction()
  h = Miller(1, 2, 0, cs)
  r = vector3d.rand(rng=rng)
  u = SO3FunHarmonic([1.0], cs, ss)
  assert np.isclose(radon(u, h)(r), 1)
  fL = SO3FunHarmonic(randomCoefficients(deg2dim(6)), crystalFrame("m-3m"), ss, real=True)
  PL = radon(fL, Miller(1, 1, 0, crystalFrame("m-3m")))
  assert PL.antipodal and np.isclose(PL(r), PL(-r))
  assert radon(f, vector3d(1, 0, 0, antipodal=True)).antipodal


def test_poleFigure_refusesMillerOfAnotherFrame():
  cs, ss, f = cubicFunction()
  with pytest.raises(ValueError):
    radon(f, Miller(1, 0, 0, crystalFrame("m-3m")))


def test_radon_onTheSphereIsTheMeanOverTheGreatCircleAndInverts():
  # the Funk transform is the mean over the great circle orthogonal to v
  s = S2FunHarmonic(randomCoefficients(49), real=True) + 2
  Fs = radon(s)
  assert Fs.antipodal
  for v in vector3d.rand(3, rng=rng):
    R0 = rotation.map(zvector, v)
    circle = R0 * vector3d(np.cos(2 * pi * np.arange(20) / 20), np.sin(2 * pi * np.arange(20) / 20), 0)
    assert np.isclose(Fs(v), np.mean(s(circle)), rtol=1e-8)
  assert invRadon(Fs).isclose(s.even())


def test_pointwiseOperations_onSO3():
  cs, ss = crystalFrame("mmm"), specimenFrame("1")
  f = SO3FunHarmonic(randomCoefficients(deg2dim(5)), cs, ss, real=True) + 40
  g = SO3FunHarmonic(randomCoefficients(deg2dim(4)), cs, ss, real=True)
  rs = rotation.rand(10, rng=rng)
  assert np.all(f(rs) > 0)
  assert relclose((f * g)(rs), f(rs) * g(rs))
  assert (f * g).bandwidth == f.bandwidth + g.bandwidth == 7
  assert relclose((f ** 2)(rs), f(rs) ** 2)
  assert relclose(f.sqrt()(rs), np.sqrt(f(rs)), 1e-3)
  assert relclose((1 / f)(rs), 1 / f(rs), 1e-2)


def test_pointwiseOperations_onTheSphere():
  s = S2FunHarmonic(randomCoefficients(16), real=True) + 10
  vs = vector3d.rand(10, rng=rng)
  assert relclose((s * s)(vs), s(vs) ** 2)
  assert relclose((0.1 * s).exp()(vs), np.exp(0.1 * s(vs)), 1e-4)


def test_pointwiseOperations_onTheCircle():
  c = S1FunHarmonic(S1FejerKernel(4))
  xs = rng.random(10) * 2 * pi
  assert relclose((c * c)(xs), c(xs) ** 2, 1e-8)
