# vector fields on SO(3) (docs/plans/functions.md F7: V0 to V6, check_SO3TangentVectors and the
# approximation part of check_SO3VectorFieldApprox)
import numpy as np
import pytest

from mtex import (Miller, SO3DeLaValleePoussinKernel, SO3FunHandle, SO3FunHarmonic, SO3TangentVector, SO3VectorFieldHandle,
                  SO3VectorFieldHarmonic, SO3VectorFieldRBF, antiderivative, bandwidthFor, component, cross, crystalFrame, curl,
                  deg2dim, degreeBlock, degree, div, dot, equispacedSO3Grid, grad, interpolate, inv, laplace, left, norm,
                  orientation, quadratureSO3Grid, right, rotate, rotation, specimenFrame, symmetrise, vector3d, xvector,
                  yvector, zvector)
from mtex.functions.so3vectorfield import differentiate

rng = np.random.default_rng(71)


# the L2 norm of a field, over its coefficients
def l2(v):
  return float(np.linalg.norm(v._fhat))


def randomField(L, cs, ss, real=True, gen=rng):
  c = gen.standard_normal(deg2dim(L + 1)) + 1j * gen.standard_normal(deg2dim(L + 1))
  return SO3FunHarmonic(c, cs, ss, real=real)


def relerr(a, b):
  return np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b)))


def relative(h, g):
  return l2(h - g) / l2(g)


# a random function of -3m and 222 of bandwidth 10 and its gradient on the two sides
@pytest.fixture(scope="module")
def gradientField():
  cs, ss = crystalFrame("-3m"), specimenFrame("222")
  f = randomField(10, cs, ss) + 5
  g1 = grad(f)
  return cs, ss, f, g1, right(g1)


# a random field of -3m and 222 of bandwidth 6, not a gradient
def randomVectorField(cs, ss, L=6):
  return SO3VectorFieldHarmonic(rng.standard_normal((deg2dim(L + 1), 3)) + 1j * rng.standard_normal((deg2dim(L + 1), 3)), cs, ss)


def test_differentiate_isTheAngularMomentumMatrixOnEitherSide():
  for L in (0, 1, 8):
    for side in ("left", "right"):
      fhat = rng.standard_normal(deg2dim(L + 1)) + 1j * rng.standard_normal(deg2dim(L + 1))
      before = fhat.copy()
      actual = differentiate(fhat, side)
      expected = np.empty_like(actual)
      for n in range(L + 1):
        m = np.arange(-n, n)
        c = np.sqrt(n * (n + 1) - m * (m + 1))
        Jp = np.zeros((2 * n + 1, 2 * n + 1), dtype=complex)
        Jp[np.arange(1, 2 * n + 1), np.arange(2 * n)] = c
        Jm = Jp.T.copy()
        J = ((Jp + Jm) / 2, (Jp - Jm) / 2j, np.diag(np.arange(-n, n + 1)).astype(complex))
        F = degreeBlock(fhat, n)
        for i in range(3):
          degreeBlock(expected[:, i], n)[...] = (-1j * J[i].T) @ F if side == "left" else F @ (-1j * J[i].T)
      assert np.allclose(actual, expected) and np.array_equal(fhat, before)
      if L == 0:
        assert np.all(actual == 0)


def test_grad_sidesCarryTheirComponentFramesAndBandwidth(gradientField):
  cs, ss, f, g1, g2 = gradientField
  assert g1.tangentSpace == "left" and g2.tangentSpace == "right" and g2.bandwidth == g1.bandwidth + 1
  assert g1.CS is cs and g1.SS is ss and g1.componentFrames() == (cs, ss.stripSym())
  assert g2.componentFrames() == (cs.stripSym(), ss)


def test_left_right_roundTripKeepsTheFieldAndAgreesPointwise(gradientField):
  # the round trip through the other side keeps the field, and the values agree pointwise
  cs, ss, f, g1, g2 = gradientField
  os = orientation(rotation.rand(6, rng=rng), cs, ss)
  h = left(g2)
  scale = np.max(np.abs(g1(os).data))
  assert np.max(np.abs(g1(os).data - h(os).data)) < 1e-5 * scale
  assert np.max(np.abs(g2(os).data - right(g1(os)).data)) < 1e-5 * scale
  assert g1.isclose(h, 1e-5)


def test_SO3VectorFieldHarmonic_leftTurnsWithTheSpecimenGroupRightWithTheCrystalGroup(gradientField):
  # evaluation at symmetrically equivalent orientations: a left field turns with the
  # specimen group and is invariant under the crystal group, a right field the mirror
  cs, ss, f, g1, g2 = gradientField
  scale = np.max(np.abs(g1(orientation(rotation.rand(6, rng=rng), cs, ss)).data))
  r = rotation.rand(rng=rng)
  S = ss.sym.properGroup().rotations().flatten()
  for i in range(S.size):
    assert np.linalg.norm((S[i].inv() * vector3d(g1(S[i] * r).data)).data - g1(r).data) < 1e-10 * scale
    assert np.linalg.norm(g2(S[i] * r).data - g2(r).data) < 1e-5 * scale
  R = cs.sym.properGroup().rotations().flatten()
  for i in range(R.size):
    assert np.linalg.norm(g1(r * R[i]).data - g1(r).data) < 1e-10 * scale
    assert np.linalg.norm((R[i] * vector3d(g2(r * R[i]).data)).data - g2(r).data) < 1e-5 * scale


def test_SO3TangentVector_carriesBothFramesWrittenInTheFrameOfItsSide(gradientField):
  # the tangent vectors carry both frames, and are written in the frame of their side
  cs, ss, f, g1, g2 = gradientField
  os = orientation(rotation.rand(6, rng=rng), cs, ss)
  t = g1(os)
  assert isinstance(t, SO3TangentVector) and t.oriRef.CS is cs and t.oriRef.SS is ss
  assert t.frame is ss and right(t).frame is cs and g2(os).frame is cs


def test_component_isAFunctionOnTheComponentFrames(gradientField):
  # a component is a function on the component frames
  cs, ss, f, g1, g2 = gradientField
  os = orientation(rotation.rand(6, rng=rng), cs, ss)
  t = g1(os)
  c = component(g1, 2)
  assert c.CS is cs and c.SS is ss.stripSym()
  assert np.allclose(c(rotation(os)), t.z)
  assert dot(g1, zvector).isclose(c)


def test_grad_meanIsZero(gradientField):
  _, _, _, g1, _ = gradientField
  assert np.linalg.norm(g1.mean().data) < 1e-12


def test_curl_ofAGradientVanishesOnBothSides(gradientField):
  # the curl of a gradient vanishes on both sides, exactly on the side it was taken on and
  # to the accuracy of the transform on the other
  _, _, _, g1, g2 = gradientField
  assert l2(curl(g1)) < 1e-10 * l2(g1)
  assert l2(curl(g2)) < 1e-5 * l2(g2)


def test_antiderivative_ofTheGradientIsTheFunctionUpToItsMean(gradientField):
  # the antiderivative of the gradient is the function up to its mean
  cs, ss, f, g1, g2 = gradientField
  for g, tol in ((g1, 1e-10), (g2, 1e-5)):
    F = antiderivative(g) + f.mean()
    assert norm(F - f) < tol * norm(f) and F.CS is cs and F.SS is ss
  assert isinstance(antiderivative(g1, orientation(rotation.rand(rng=rng), cs, ss)), float)


def test_div_isIndependentOfTheSideAndTheCentralDifference(gradientField):
  # the divergence does not depend on the side, and agrees with its central difference
  cs, ss, f, g1, g2 = gradientField
  d1, d2 = div(g1), div(g2)
  assert norm(d1 - d2) < 1e-5 * norm(d1)
  assert d1.CS is cs and d1.SS is ss and d1.isReal
  delta = 0.05 * degree
  for r in rotation.rand(3, rng=rng):
    fd = sum((g1(rotation.byAxisAngle(e, delta / 2) * r).data[i] - g1(rotation.byAxisAngle(e, -delta / 2) * r).data[i]) / delta for i, e in enumerate((xvector, yvector, zvector)))
    assert np.isclose(d1(r), fd, rtol=1e-4) and np.isclose(div(g1, r), d1(r))


def test_div_ofTheGradientIsTheLaplacian(gradientField):
  # the divergence of the gradient is the Laplacian
  _, _, f, g1, _ = gradientField
  d1 = div(g1)
  assert norm(d1 - laplace(f)) < 1e-10 * norm(d1)


def test_curl_ofANonGradientIsTheSameOnBothSidesAndHasNoAntiderivative(gradientField):
  # a field that is not a gradient: its curl on the two sides is the same tangent vector,
  # and it has no antiderivative
  cs, ss, _, _, _ = gradientField
  h = randomVectorField(cs, ss)
  c1, c2 = curl(h), curl(right(h))
  for r in rotation.rand(3, rng=rng):
    assert np.linalg.norm((r * vector3d(c2(r).data)).data - c1(r).data) < 1e-5 * l2(c1)
    assert np.linalg.norm(curl(h, r).data - c1(r).data) < 1e-12
  with pytest.raises(ValueError):
    antiderivative(h)


def test_div_curl_areLinear(gradientField):
  # div and curl are linear
  cs, ss, _, g1, _ = gradientField
  h = randomVectorField(cs, ss)
  d1 = div(g1)
  assert norm(div(g1 + h) - d1 - div(h)) < 1e-10 * norm(d1)
  assert l2(curl(g1 + h) - curl(h)) < 1e-8 * l2(h)


def test_inv_ofAGradientIsTheRightGradientOfTheInverse(gradientField):
  # the push-forward under the inversion commutes with the gradient
  cs, ss, f, g1, _ = gradientField
  gi = inv(g1)
  assert gi.tangentSpace == "right" and gi.CS is ss and gi.SS is cs
  assert np.linalg.norm(gi.fhat - grad(inv(f), "right").fhat) < 1e-10 * l2(g1)


def test_rotate_commutesWithGrad():
  cs, ss = crystalFrame("-3m"), specimenFrame("1")
  f = randomField(8, cs, ss)
  q = rotation.byAxisAngle(vector3d(1, 2, 3).normalize(), 37 * degree)
  rots = rotation.rand(5, rng=rng)
  for R in ("left", "right"):
    fr = rotate(f, q, R)
    os = orientation(rots, fr.CS, fr.SS)
    for S in ("left", "right"):
      ref = left(grad(fr, S)(os))
      g = grad(f, S)
      for field in (g, SO3VectorFieldHandle(g), SO3VectorFieldHarmonic(SO3VectorFieldHandle(g), bandwidth=8)):
        gr = rotate(field, q, R)
        assert gr.CS is fr.CS and gr.SS is fr.SS
        assert np.max(np.abs(left(gr(os)).data - ref.data)) < 1e-5 * np.max(np.abs(ref.data))


def test_rotate_byAnOrientationMovesTheFrame():
  # with an orientation the frame moves
  cs, ss = crystalFrame("-3m"), specimenFrame("1")
  f = randomField(8, cs, ss)
  q = rotation.byAxisAngle(vector3d(1, 2, 3).normalize(), 37 * degree)
  ss2 = specimenFrame("222")
  o = orientation(q, ss, ss2)
  gr = rotate(grad(f), o)
  assert gr.SS is ss2 and gr.CS is cs
  with pytest.raises(ValueError):
    rotate(grad(f), orientation(q, ss2, ss))


def test_rotate_byANormaliserKeepsTheGroup():
  # a rotation that normalises the group keeps it
  fz = randomField(6, crystalFrame("-3m"), specimenFrame("222"))
  gz = rotate(grad(fz), rotation.byAxisAngle(zvector, 90 * degree))
  assert gz.SS is specimenFrame("222")
  assert np.linalg.norm(symmetrise(gz).fhat - gz.fhat) < 1e-10 * l2(gz)


# the bandwidth of the quadrature rule for the gradient field and its nodes
def quadratureNodes(cs, ss, g):
  cA, cB = g.componentFrames()
  L = bandwidthFor(g.bandwidth, cA, cB)
  rule = quadratureSO3Grid(L, cA, cB)
  return L, rule, orientation(rotation(rule.nodes()), cs, ss)


def test_SO3VectorFieldHarmonic_fromSamplesOnTheQuadratureRule(gradientField):
  cs, ss, f, g, _ = gradientField
  L, rule, os = quadratureNodes(cs, ss, g)
  h = SO3VectorFieldHarmonic(g(os), bandwidth=L, weights=rule.weights)
  assert relative(h, g) < 1e-5
  h = SO3VectorFieldHarmonic(os, g(os).data, bandwidth=L, weights=rule.weights)
  assert relative(h, g) < 1e-5


def test_SO3VectorFieldHarmonic_fromACallableAHandleOrTheOtherSide(gradientField):
  # from a callable, from a handle, from the field on the other side
  cs, ss, f, g, _ = gradientField
  L, rule, os = quadratureNodes(cs, ss, g)
  assert relative(SO3VectorFieldHarmonic(lambda o: g(o), cs, ss, bandwidth=L), g) < 1e-5
  hd = SO3VectorFieldHandle(g)
  assert isinstance(hd(os[3]), SO3TangentVector) and np.linalg.norm(hd(os[3]).data - g(os[3]).data) < 1e-12
  assert relative(SO3VectorFieldHarmonic(hd, bandwidth=L), g) < 1e-5
  assert relative(SO3VectorFieldHarmonic(right(g), "left"), g) < 1e-4
  assert relative(left(right(hd)), g) < 1e-4


def test_SO3VectorFieldHarmonic_fromThreeComponentsOnTheComponentFrames(gradientField):
  # the constructor from three components on the component frames
  cs, ss, f, g, _ = gradientField
  fs = [component(g, i) for i in range(3)]
  assert relative(SO3VectorFieldHarmonic(fs, cs, ss), g) < 1e-12
  with pytest.raises(ValueError):
    SO3VectorFieldHarmonic(fs[:2], cs, ss)


def test_interpolate_throughTheValuesOnTheFullRuleReturnsTheField(gradientField):
  # interpolation through the values on the full rule returns the field
  cs, ss, f, g, _ = gradientField
  L, _, _ = quadratureNodes(cs, ss, g)
  full = quadratureSO3Grid(L)
  osf = orientation(full.nodes(), cs, ss)
  h = interpolate(g(osf), bandwidth=L, lambda_=0, weights=full.weights)
  assert relative(h, g) < 1e-4


def test_SO3VectorFieldHarmonic_ofAHandleKeepsItsSideAndBandwidth():
  # a random handle field, its harmonic form evaluated back
  cs, ss = crystalFrame("-3m"), specimenFrame("222")
  fun = lambda o: np.stack([np.sin(o.axis().x * o.angle()), np.cos(o.axis().y * o.angle()), o.axis().z * o.angle()], axis=-1)
  w = SO3VectorFieldHandle(fun, cs, ss, "right", bandwidth=12)
  wh = SO3VectorFieldHarmonic(w)
  assert wh.tangentSpace == "right" and wh.bandwidth == 12


def test_grad_ofAHandleByDifferencesIsTheHarmonicGradient(gradientField):
  # the gradient of a handle function by differences agrees with the harmonic gradient
  cs, ss, f, _, _ = gradientField
  fh = SO3FunHandle(lambda o: f(o), cs, ss, bandwidth=10)
  gh = grad(fh, "right")
  for o in orientation(rotation.rand(3, rng=rng), cs, ss):
    assert np.linalg.norm(gh(o).data - grad(f, o, "right").data) < 1e-5 * np.linalg.norm(grad(f, o, "right").data)


def test_SO3VectorFieldHarmonic_quadratureCutsAPolynomialFieldToItsDegree():
  cs = crystalFrame("222")
  F = SO3VectorFieldHarmonic.quadrature(lambda o: SO3TangentVector(vector3d(np.ones((o.size, 3))), o), cs)
  assert F.bandwidth <= 1 and np.allclose(F(orientation.rand(3, cs, rng=rng)).data, 1, atol=1e-8)


# a gradient field of 432 and a random field, and orientations to evaluate them at, drawn from
# their own generator: the direction's tolerance holds for this draw, whatever ran before
@pytest.fixture(scope="module")
def twoFields():
  gen = np.random.default_rng(71)
  cs, ss = crystalFrame("432"), specimenFrame("1")
  f = randomField(6, cs, ss, gen=gen) + 3
  g = grad(f)
  h = SO3VectorFieldHarmonic(gen.standard_normal((deg2dim(5), 3)) + 1j * gen.standard_normal((deg2dim(5), 3)), cs, ss)
  os = orientation(rotation.rand(4, rng=gen), cs, ss)
  return cs, ss, f, g, h, os


def test_dot_isPointwiseOnEitherSide(twoFields):
  cs, ss, f, g, h, os = twoFields
  d = dot(g, h)
  assert d.CS is cs and d.SS is ss and d.bandwidth == g.bandwidth + h.bandwidth
  assert np.allclose(d(os), np.sum(g(os).data * h(os).data, axis=-1), rtol=1e-5)
  assert np.allclose(dot(g, right(h))(os), d(os), rtol=1e-5)


def test_cross_isPointwise(twoFields):
  cs, ss, f, g, h, os = twoFields
  c = cross(g, h)
  assert c.tangentSpace == "left"
  assert np.allclose(c(os).data, np.cross(g(os).data, h(os).data), rtol=1e-5)


def test_SO3VectorFieldHarmonic_timesAFunctionIsPointwise(twoFields):
  cs, ss, f, g, h, os = twoFields
  p = f * g
  assert np.allclose(p(os).data, f(os)[:, None] * g(os).data, rtol=1e-5)


def test_abs_normalize_approximateThePointwiseLengthAndDirection(twoFields):
  # the pointwise norm and the normalised field are not bandlimited: approximations
  cs, ss, f, g, h, os = twoFields
  n = abs(g)
  assert np.allclose(n(os), np.linalg.norm(g(os).data, axis=-1), rtol=5e-2)
  u = g.normalize()
  assert np.allclose(np.linalg.norm(u(os).data, axis=-1), 1, rtol=1e-1)
  # the direction is singular where the gradient vanishes: compare where it is not small
  G = g(os).data
  big = np.linalg.norm(G, axis=-1) > 0.5 * np.mean(np.linalg.norm(G, axis=-1))
  assert np.allclose(u(os).data[big], (G / np.linalg.norm(G, axis=-1, keepdims=True))[big], rtol=1e-1, atol=1e-2)


def test_SO3VectorFieldHarmonic_linearCombinationsAndDivision(twoFields):
  cs, ss, f, g, h, os = twoFields
  assert np.allclose((2 * g - g + h)(os[0]).data, (g + h)(os[0]).data)
  assert np.isclose(l2(g / 2), l2(g) / 2)


# the radial basis field through the gradient of a random -3m, 222 function at the centres of a 15 degree grid
@pytest.fixture(scope="module")
def rbfField():
  cs, ss = crystalFrame("-3m"), specimenFrame("222")
  f = randomField(6, cs, ss) + 5
  g = grad(f)
  psi = SO3DeLaValleePoussinKernel(halfwidth=15 * degree)
  centres = orientation(rotation(equispacedSO3Grid(cs, ss, resolution=15 * degree)), cs, ss)
  t = g(centres)
  v = SO3VectorFieldRBF(t, kernel=psi)
  os = orientation(rotation.rand(20, rng=rng), cs, ss)
  return cs, ss, g, psi, centres, t, v, os


def test_SO3VectorFieldRBF_isExactAtItsCentres(rbfField):
  # the field through the values at the centres, exact there
  cs, ss, g, psi, centres, t, v, os = rbfField
  assert v.tangentSpace == "left" and v.CS is cs and v.SS is ss
  assert np.max(np.abs(v(centres).data - t.data)) < 1e-5 * np.max(np.abs(t.data))


def test_SO3VectorFieldRBF_approximatesTheFieldBetweenItsCentres(rbfField):
  # the approximation of the field between the centres
  cs, ss, g, psi, centres, t, v, os = rbfField
  assert relerr(v(os).data, g(os).data) < 0.1
  va = SO3VectorFieldRBF(g, resolution=15 * degree, kernel=psi)
  assert np.max(np.abs(va(os).data - v(os).data)) < 1e-8


def test_SO3VectorFieldRBF_isEquivariantOnTheLeftAndInvariantOnTheRight(rbfField):
  # exactly equivariant on the left and invariant on the right
  cs, ss, g, psi, centres, t, v, os = rbfField
  r = rotation.rand(rng=rng)
  S = ss.sym.properGroup().rotations().flatten()
  for i in range(S.size):
    assert np.linalg.norm((S[i].inv() * vector3d(v(S[i] * r).data)).data - v(r).data) < 1e-10
  R = cs.sym.properGroup().rotations().flatten()
  for i in range(R.size):
    assert np.linalg.norm(v(r * R[i]).data - v(r).data) < 1e-10


def test_SO3VectorFieldHarmonic_ofAnRBFFieldReproducesTheKernelSumOnEitherSide(rbfField):
  # the harmonic form reproduces the kernel sum away from the truncation, on either side
  cs, ss, g, psi, centres, t, v, os = rbfField
  vh = SO3VectorFieldHarmonic(v, bandwidth=2 * psi.bandwidth)
  scale = np.max(np.abs(v(os).data))
  assert np.max(np.abs(vh(os).data - v(os).data)) < 1e-4 * scale
  vr = SO3VectorFieldHarmonic(v, "right", bandwidth=2 * psi.bandwidth)
  assert np.max(np.abs(left(vr(os)).data - v(os).data)) < 1e-4 * scale


def test_SO3VectorFieldRBF_rotatesAndScales(rbfField):
  # rotation and scaling
  cs, ss, g, psi, centres, t, v, os = rbfField
  r = rotation.rand(rng=rng)
  q = rotation.byAxisAngle(zvector, 90 * degree)
  vq = rotate(v, q)
  assert vq.SS is ss
  assert np.linalg.norm(vq(q * r).data - (q * vector3d(v(r).data)).data) < 1e-10
  assert np.linalg.norm((2 * v)(r).data - 2 * v(r).data) < 1e-12


# a gradient field of 432 and orientations to evaluate it at
def gradientOf432():
  cs, ss = crystalFrame("432"), specimenFrame("1")
  return cs, grad(randomField(6, cs, ss) + 5), orientation(rotation.rand(20, rng=rng), cs, ss)


def test_norm_isTheLengthOfTheVectors():
  # MATLAB's norm is the length of the vectors
  cs, g, o = gradientOf432()
  lengths = np.linalg.norm(g(o).data, axis=-1)
  # the length has a kink where the field vanishes, so the typical error is compared
  assert np.median(np.abs(np.real(norm(g)(o)) - lengths)) < 1e-2 * np.max(lengths)


def test_dot_withACrystalDirectionTakesTheRightComponents():
  # a crystal direction meets a left field through its right components
  cs, g, o = gradientOf432()
  h = Miller(1, 0, 0, cs)
  assert np.allclose(np.real(dot(g, h)(o)), np.real(dot(right(g), h)(o)))
  assert np.allclose(np.real(dot(g, h)(o)), right(g(o)).data @ np.array([1.0, 0, 0]), atol=1e-6)
