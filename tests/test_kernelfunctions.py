# kernels, docs/plans/functions.md F1: the coefficient rules against the closed forms, the
# normalisation, the halfwidths (check_kernelHalfwidth, check_S2KernelNormalization),
# radon, convolution and derivatives
import numpy as np
import pytest
from numpy import pi

from mtex import (S1BumpKernel, S1DeLaValleePoussinKernel, S1DirichletKernel, S1FejerKernel, S1GaussKernel, S1Kernel,
                  S1KernelHandle, S2BumpKernel, S2DeLaValleePoussinKernel, S2DirichletKernel, S2Kernel, S2KernelHandle,
                  S2RestrictedDistanceKernel, SO3AbelPoissonKernel, SO3BumpKernel, SO3DeLaValleePoussinKernel,
                  SO3DirichletKernel, SO3GaussWeierstrassKernel, SO3Kernel, SO3KernelHandle, SO3LaplaceKernel,
                  SO3RestrictedDistanceKernel, SO3SobolevKernel, SO3SquareSingularityKernel, SO3vonMisesFisherKernel,
                  calcFourier, conv, degree, fibreVonMisesFisherKernel, gaussLegendre, grad, halfwidth, mean, norm,
                  radon, rotation, vector3d, zvector)

rng = np.random.default_rng(5)


# the mean of a kernel by quadrature against the measure of its domain
def meanByQuadrature(psi):
  x, w = gaussLegendre(400)
  if isinstance(psi, SO3Kernel):
    omega = (x + 1) * (pi / 2)
    return np.sum(w * psi(np.cos(omega / 2)) * np.sin(omega / 2) ** 2)
  if isinstance(psi, S2Kernel):
    return np.sum(w * psi(x)) / 2
  x = np.linspace(0, 2 * pi, 4001)[:-1]
  return np.mean(psi(x))


def centralDifference(f, x, h=1e-5):
  return (f(x + h) - f(x - h)) / (2 * h)


# the families with a closed form; the others evaluate through their series
so3Families = [SO3DeLaValleePoussinKernel(20), SO3DeLaValleePoussinKernel(halfwidth=10 * degree), SO3AbelPoissonKernel(0.7),
               SO3vonMisesFisherKernel(10), SO3vonMisesFisherKernel(halfwidth=5 * degree), SO3BumpKernel(10 * degree),
               SO3DirichletKernel(12), SO3RestrictedDistanceKernel(64)]
so3Series = [SO3GaussWeierstrassKernel(0.1), SO3SquareSingularityKernel(0.5), fibreVonMisesFisherKernel(20)]
s2Families = [S2DeLaValleePoussinKernel(20), S2DeLaValleePoussinKernel(halfwidth=10 * degree), S2BumpKernel(10 * degree),
              S2DirichletKernel(12), S2RestrictedDistanceKernel(64)]
s1Families = [S1DeLaValleePoussinKernel(12), S1DirichletKernel(12), S1FejerKernel(12), S1BumpKernel(10 * degree), S1GaussKernel(0.1)]


# the closed form of every family against the Clenshaw sum of its coefficient rule, far
# enough out that the truncation is below the tolerance
@pytest.mark.parametrize("psi", so3Families)
def test_calcFourier_ofAnSO3FamilyIsItsClosedForm(psi):
  t = np.linspace(-0.99, 0.99, 41)
  L = psi.bandwidth if isinstance(psi, (SO3DirichletKernel, SO3RestrictedDistanceKernel)) else 400
  tol = 5e-2 if isinstance(psi, (SO3BumpKernel, SO3RestrictedDistanceKernel)) else 1e-6
  sel = np.abs(t) >= 0.02 if isinstance(psi, SO3BumpKernel) else np.ones(t.shape, bool)
  assert np.allclose(psi(t)[sel], SO3Kernel(calcFourier(psi, L))(t)[sel], atol=tol * max(1, abs(psi(1.0))), rtol=0)


@pytest.mark.parametrize("psi", s2Families)
def test_calcFourier_ofAnS2FamilyIsItsClosedForm(psi):
  t = np.linspace(-0.99, 0.99, 41)
  L = psi.bandwidth if isinstance(psi, (S2DirichletKernel, S2RestrictedDistanceKernel)) else 400
  tol = 5e-2 if isinstance(psi, (S2BumpKernel, S2RestrictedDistanceKernel)) else 1e-6
  sel = np.abs(t - np.cos(10 * degree)) >= 0.02 if isinstance(psi, S2BumpKernel) else np.ones(t.shape, bool)
  assert np.allclose(psi(t)[sel], S2Kernel(calcFourier(psi, L))(t)[sel], atol=tol * max(1, abs(psi(1.0))), rtol=0)


@pytest.mark.parametrize("psi", s1Families)
def test_calcFourier_ofAnS1FamilyIsItsClosedForm(psi):
  x = np.linspace(-3, 3, 41)
  L = 2000 if isinstance(psi, S1BumpKernel) else psi.bandwidth
  sel = np.abs(np.abs(x) - 10 * degree) >= 0.05 if isinstance(psi, S1BumpKernel) else np.ones(x.shape, bool)
  assert np.allclose(psi(x)[sel], S1Kernel(calcFourier(psi, L))(x)[sel], atol=5e-2 if isinstance(psi, S1BumpKernel) else 1e-9, rtol=0)


def test_calcFourier_ofTheSquareSingularityIsStableFarBeyondMATLABsCut():
  A = calcFourier(SO3SquareSingularityKernel(0.5), 400)
  assert np.all(np.isfinite(A)) and A[0] == 1 and abs(A[400]) < 1e-100
  assert np.isclose(A[1], 3 * ((1 + 0.25) / 1 - 1 / np.log(3)))


# the generic Clenshaw path is what a coefficient kernel evaluates
def test_S2Kernel_ofTheRadonCoefficientsIsTheRadonTransform():
  psi = SO3DeLaValleePoussinKernel(20)
  assert np.isclose(S2Kernel(calcFourier(radon(psi), 20))(0.7), radon(psi)(0.7))


def test_A_isTheCoefficientsUpToTheBandwidth():
  psi = SO3DeLaValleePoussinKernel(20)
  assert psi.A.shape == (psi.bandwidth + 1,) and np.array_equal(psi.A, calcFourier(psi))


def test_fhat_ofACircleKernelIsTwoSided():
  assert np.array_equal(S1FejerKernel(3).fhat, [0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25])


def test_S1Kernel_ofTwoSidedCoefficientsIsTheirKernel():
  assert S1Kernel([0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25]) == S1FejerKernel(3)


densities = [psi for psi in so3Families + so3Series + s2Families + s1Families
             if not isinstance(psi, (SO3RestrictedDistanceKernel, S2RestrictedDistanceKernel))]


@pytest.mark.parametrize("psi", densities)
def test_mean_ofADensityIsOne(psi):
  assert np.isclose(mean(psi), 1)


@pytest.mark.parametrize("psi", [psi for psi in densities if not isinstance(psi, SO3BumpKernel)])
def test_mean_ofADensityIsItsIntegralByQuadrature(psi):
  assert np.isclose(meanByQuadrature(psi), 1, atol=2e-2 if isinstance(psi, (S2BumpKernel, S1BumpKernel)) else 1e-6, rtol=0)


def test_mean_ofLaplaceAndSobolevIsZero():
  assert mean(SO3LaplaceKernel()) == 0
  assert mean(SO3SobolevKernel(1, bandwidth=20)) == 0


# check_S2KernelNormalization
def test_S2BumpKernel_isTheIndicatorOfItsCapScaledToADensity():
  hw = 10 * degree
  psi = S2BumpKernel(hw)
  assert psi(np.cos(2 * hw)) == 0
  assert np.isclose(psi(np.cos(hw / 2)), 2 / (1 - np.cos(hw)))
  assert psi(np.cos(hw / 4)) == psi(np.cos(hw / 2))


# check_S2KernelNormalization
def test_A_ofAnS2DensityStartsWithOne():
  hw = 10 * degree
  for psi in (S2DeLaValleePoussinKernel(halfwidth=hw), S2BumpKernel(hw), S2DirichletKernel(16)):
    assert abs(psi.A[0] - 1) < 1e-4


# check_kernelHalfwidth: the analytic halfwidth survives the round trip through the
# coefficients, within max(0.02 degree, 1 %)
@pytest.mark.parametrize("psi", [
  SO3AbelPoissonKernel(0.4), SO3AbelPoissonKernel(0.7), SO3DeLaValleePoussinKernel(1), SO3DeLaValleePoussinKernel(100),
  SO3DirichletKernel(1), SO3DirichletKernel(50), SO3GaussWeierstrassKernel(0.1), SO3GaussWeierstrassKernel(0.5),
  SO3GaussWeierstrassKernel(1), SO3SquareSingularityKernel(0.091), SO3SquareSingularityKernel(0.5),
  SO3SquareSingularityKernel(0.99), SO3vonMisesFisherKernel(0.347), SO3vonMisesFisherKernel(10), SO3LaplaceKernel(),
  SO3SobolevKernel(0.1, bandwidth=20), SO3SobolevKernel(1.1, bandwidth=20), SO3SobolevKernel(5, bandwidth=20)])
def test_halfwidth_survivesTheRoundTripThroughTheCoefficients(psi):
  h1, h2 = halfwidth(psi), halfwidth(SO3Kernel(psi.A))
  assert abs(h1 - h2) <= max(0.02 * degree, 0.01 * h1)
  assert 0 < h1 < pi


@pytest.mark.parametrize("hw", (2 * degree, 10 * degree, 30 * degree))
def test_halfwidth_ofAKernelConstructedFromOneIsThatOne(hw):
  assert np.isclose(SO3DeLaValleePoussinKernel(halfwidth=hw).halfwidth(), hw)
  assert np.isclose(S2DeLaValleePoussinKernel(halfwidth=hw).halfwidth(), hw)
  assert np.isclose(SO3vonMisesFisherKernel(halfwidth=hw).halfwidth(), hw)
  assert np.isclose(fibreVonMisesFisherKernel(halfwidth=hw).halfwidth(), hw)
  assert np.isclose(SO3AbelPoissonKernel(halfwidth=hw).halfwidth(), hw, rtol=1e-3)
  assert np.isclose(SO3GaussWeierstrassKernel(halfwidth=hw).halfwidth(), hw, rtol=1e-3)
  assert np.isclose(SO3SquareSingularityKernel(halfwidth=hw).halfwidth(), hw, rtol=1e-3)
  assert SO3BumpKernel(halfwidth=hw).halfwidth() == hw
  assert np.isclose(SO3Kernel(calcFourier(SO3BumpKernel(hw), 500)).halfwidth(), hw, atol=0.3 * degree)
  assert np.isclose(S1GaussKernel(halfwidth=hw).halfwidth(), hw)


def test_halfwidth_ofTheS1DeLaValleePoussinKernelConstructedFromOneIsNearIt():
  assert abs(S1DeLaValleePoussinKernel(halfwidth=3 * degree).halfwidth() - 3 * degree) < 0.3 * degree


# the halfwidth of the S1 families against a direct scan
@pytest.mark.parametrize("psi", s1Families)
def test_halfwidth_ofAnS1KernelIsWhereItFallsToHalf(psi):
  x = np.linspace(0, pi, 100001)
  i = np.flatnonzero(psi(x) <= psi(0.0) / 2)[0]
  assert np.isclose(psi.halfwidth(), x[i], atol=1e-4)


# the Radon transform of an SO(3) kernel has the same coefficients as a Legendre series,
# and the closed forms MTEX gives for it agree with that series
@pytest.mark.parametrize("psi", [
  SO3DeLaValleePoussinKernel(20), SO3AbelPoissonKernel(0.7), SO3vonMisesFisherKernel(10), fibreVonMisesFisherKernel(20),
  SO3SquareSingularityKernel(0.5), SO3BumpKernel(20 * degree), SO3DirichletKernel(12)])
def test_radon_ofAnSO3KernelIsTheLegendreSeriesOfItsCoefficients(psi):
  t = np.linspace(-0.95, 0.95, 21)
  R = radon(psi)
  assert isinstance(R, S2Kernel)
  assert np.allclose(calcFourier(R, 30), calcFourier(psi, 30))
  L = 2000 if isinstance(psi, SO3BumpKernel) else 12 if isinstance(psi, SO3DirichletKernel) else 400
  assert np.allclose(R(t), S2Kernel(calcFourier(psi, L))(t), atol=5e-2 if isinstance(psi, SO3BumpKernel) else 1e-6, rtol=0)
  assert np.isclose(mean(R), 1)


def test_radon_ofTheDeLaValleePoussinKernelIsItsS2Member():
  assert radon(SO3DeLaValleePoussinKernel(20)) == S2DeLaValleePoussinKernel(20)
  assert isinstance(radon(SO3DeLaValleePoussinKernel(20)), S2DeLaValleePoussinKernel)


def test_conv_multipliesTheCoefficientsAndDividesBy2nPlus1():
  psi1, psi2 = SO3DeLaValleePoussinKernel(20), SO3AbelPoissonKernel(0.7)
  c = conv(psi1, psi2)
  L = c.bandwidth
  assert np.allclose(c.A, calcFourier(psi1, L) * calcFourier(psi2, L) / (2 * np.arange(L + 1) + 1))


def test_conv_withAnS2KernelIsAnS2Kernel():
  assert isinstance(conv(SO3DeLaValleePoussinKernel(20), radon(SO3AbelPoissonKernel(0.7))), S2Kernel)


def test_conv_withADirichletKernelCutsToItsBandwidth():
  psi1 = SO3DeLaValleePoussinKernel(20)
  assert conv(psi1, SO3DirichletKernel(5)).bandwidth == 5
  assert np.allclose(conv(psi1, SO3DirichletKernel(5)).A, calcFourier(psi1, 5))


def test_conv_ofCircleKernelsMultipliesTheirFhat():
  assert np.allclose(conv(S1FejerKernel(3), S1DirichletKernel(3)).fhat, S1FejerKernel(3).fhat)


def test_kernelAlgebra_actsOnTheCoefficients():
  psi1, psi2 = SO3DeLaValleePoussinKernel(20), SO3AbelPoissonKernel(0.7)
  s = 2 * psi1 + psi2
  assert np.allclose(s.A, 2 * calcFourier(psi1, s.bandwidth) + calcFourier(psi2, s.bandwidth))
  assert np.allclose((psi1 - psi1 / 2).A, psi1.A / 2)


def test_norm_ofADirichletKernelIsTheRootOfItsDimension():
  assert np.isclose(norm(SO3DirichletKernel(3)), np.sqrt(1 + 9 + 25 + 49))
  assert np.isclose(norm(S2DirichletKernel(3)), np.sqrt(1 + 3 + 5 + 7))
  assert np.isclose(norm(S1DirichletKernel(3)), np.sqrt(7))


def test_radon_ofASphereKernelIsTheFunkTransform():
  assert np.allclose(radon(S2DirichletKernel(4)).A, [1, 0, -5 / 2, 0, 9 * 3 / 8])


# the closed forms, and the series formula on kernels that are their own series
so3Derivatives = [SO3DeLaValleePoussinKernel(20), SO3vonMisesFisherKernel(10), SO3Kernel(calcFourier(SO3AbelPoissonKernel(0.7), 60)),
                  SO3Kernel(calcFourier(SO3GaussWeierstrassKernel(0.1), 60)), SO3Kernel(calcFourier(SO3SquareSingularityKernel(0.5), 60)),
                  SO3Kernel(calcFourier(SO3DeLaValleePoussinKernel(20), 20))]
s2Derivatives = [S2DeLaValleePoussinKernel(20), S2Kernel(calcFourier(S2DeLaValleePoussinKernel(20), 20)),
                 S2Kernel(calcFourier(SO3AbelPoissonKernel(0.7), 60)), S2Kernel(calcFourier(radon(SO3vonMisesFisherKernel(10)), 60))]


@pytest.mark.parametrize("psi", so3Derivatives)
def test_grad_ofAnSO3KernelIsTheDerivativeInTheAngle(psi):
  for omega in (0.2, 0.7, 1.5, 2.8):
    assert np.isclose(grad(psi, np.cos(omega / 2)), centralDifference(lambda w: psi(np.cos(w / 2)), omega), atol=1e-6 * max(1, psi(1.0)), rtol=0)


@pytest.mark.parametrize("psi", so3Derivatives)
def test_grad_ofAnSO3KernelVanishesAtTheIdentity(psi):
  assert grad(psi, 1.0) == 0


@pytest.mark.parametrize("psi", so3Derivatives)
def test_grad_ofAnSO3KernelWithoutArgumentIsItsDerivativeKernel(psi):
  assert np.isclose(psi.grad()(0.8), psi.grad(0.8))


@pytest.mark.parametrize("psi", s2Derivatives)
def test_grad_ofAnS2KernelIsTheDerivativeInThePolarAngle(psi):
  for theta in (0.2, 0.7, 1.5, 2.8):
    assert np.isclose(grad(psi, np.cos(theta)), centralDifference(lambda w: psi(np.cos(w)), theta), atol=1e-6 * max(1, psi(1.0)), rtol=0)


@pytest.mark.parametrize("psi", s2Derivatives)
def test_grad_ofAnS2KernelAsPolynomialIsTheDerivativeInT(psi):
  assert np.isclose(psi.grad(0.3, polynomial=True), centralDifference(psi, 0.3), atol=1e-6 * max(1, psi(1.0)), rtol=0)


@pytest.mark.parametrize("psi", s2Derivatives)
def test_grad_ofAnS2KernelWithoutArgumentIsItsDerivativeKernel(psi):
  assert np.isclose(psi.grad()(0.3), psi.grad(0.3))


# the Legendre series of dpsi/dt, exact for a polynomial
@pytest.mark.parametrize("psi", [psi for psi in s2Derivatives if type(psi) is S2Kernel])
def test_grad_ofAnS2SeriesAsPolynomialIsTheLegendreSeriesOfTheDerivative(psi):
  assert np.isclose(psi.grad(polynomial=True)(0.3), centralDifference(psi, 0.3), atol=1e-6 * max(1, psi(1.0)), rtol=0)


@pytest.mark.parametrize("psi", [S1FejerKernel(12), S1DeLaValleePoussinKernel(12), S1GaussKernel(0.3)])
def test_grad_ofAnS1KernelIsItsDerivative(psi):
  for x in (0.2, 0.7, 1.5):
    assert np.isclose(grad(psi, x), centralDifference(psi, x), atol=1e-5, rtol=0)


def test_grad_ofABumpKernelIsZeroInsideItsCap():
  assert SO3BumpKernel(0.2).grad(0.5) == 0 and S2BumpKernel(0.2).grad(0.5) == 0 and S1BumpKernel(0.2).grad(0.5) == 0


# a handle recovers the coefficients of the family it wraps
def test_SO3KernelHandle_recoversTheCoefficientsOfItsFunction():
  psi = SO3DeLaValleePoussinKernel(20)
  h = SO3KernelHandle(lambda t: psi(t), bandwidth=20)
  assert np.allclose(h.A, calcFourier(psi, 20), atol=1e-10)


def test_S2KernelHandle_recoversTheCoefficientsOfItsFunction():
  psi2 = S2DeLaValleePoussinKernel(20)
  h2 = S2KernelHandle(lambda t: psi2(t), bandwidth=20)
  assert np.allclose(h2.A, calcFourier(psi2, 20), atol=1e-10)


def test_S1KernelHandle_recoversTheCoefficientsOfItsFunction():
  h1 = S1KernelHandle(lambda x: S1FejerKernel(12)(x), bandwidth=12)
  assert np.allclose(h1.fhat, S1FejerKernel(12).fhat, atol=1e-10)


def test_kernelHandle_evaluatesItsFunction():
  psi, psi2 = SO3DeLaValleePoussinKernel(20), S2DeLaValleePoussinKernel(20)
  h = SO3KernelHandle(lambda t: psi(t), bandwidth=20)
  h2 = S2KernelHandle(lambda t: psi2(t), bandwidth=20)
  h1 = S1KernelHandle(lambda x: S1FejerKernel(12)(x), bandwidth=12)
  assert np.isclose(h(0.7), psi(0.7)) and np.isclose(h2(0.7), psi2(0.7)) and np.isclose(h1(0.7), S1FejerKernel(12)(0.7))


# an integer de la Vallee Poussin parameter is a polynomial of that degree, and the
# cutoff rule drops its negligible top coefficients
def test_SO3DeLaValleePoussinKernel_ofAnIntegerIsAPolynomialOfThatDegree():
  psi = SO3DeLaValleePoussinKernel(20)
  assert 10 < psi.bandwidth <= 20
  assert np.all(np.abs(calcFourier(psi, 30)[21:]) < 1e-12)


def test_bandwidth_growsAsTheHalfwidthShrinks():
  assert SO3DeLaValleePoussinKernel(halfwidth=10 * degree).bandwidth == 25
  assert SO3vonMisesFisherKernel(halfwidth=10 * degree).bandwidth < SO3vonMisesFisherKernel(halfwidth=2 * degree).bandwidth


def test_bandwidth_ofADirichletKernelIsItsDegree():
  assert SO3DirichletKernel(7).bandwidth == 7


def test_SO3Kernel_ofARotationIsItsValueAtTheHalfAngleCosine():
  psi = SO3DeLaValleePoussinKernel(20)
  r = rotation.rand(rng=rng)
  assert np.isclose(psi(r), psi(np.cos(r.angle() / 2)))
  assert np.isclose(psi(r, rotation.id), psi(r))


def test_SO3Kernel_ofAContainerHasItsShape():
  assert SO3DeLaValleePoussinKernel(20)(rotation.rand(3, 4, rng=rng)).shape == (3, 4)


def test_S2Kernel_ofAVectorIsItsValueAtZ():
  psi2 = S2DeLaValleePoussinKernel(20)
  v = vector3d.rand(rng=rng)
  assert np.isclose(psi2(v), psi2(v.z))
  assert np.isclose(psi2(v, zvector), psi2(v))


def test_kernel_ofASinglePrecisionArgumentIsSinglePrecision():
  psi = SO3DeLaValleePoussinKernel(20)
  c = SO3Kernel(calcFourier(psi, 20))
  assert c(np.float32(0.9)).dtype == np.float32
  assert np.isclose(c(np.float32(0.9)), psi(0.9), atol=1e-5 * psi(1.0), rtol=0)
  assert radon(c)(np.float32(0.9)).dtype == np.float32
  assert S1FejerKernel(12)(np.float32(0.3)).dtype == np.float32


def test_str_ofAKernelIsItsFamilyAndHalfwidth():
  assert str(SO3DeLaValleePoussinKernel(halfwidth=10 * degree)) == "de la Vallee Poussin, halfwidth 10°"


def test_repr_ofAKernelListsBandwidthAndHalfwidth():
  assert repr(SO3DeLaValleePoussinKernel(halfwidth=10 * degree)) == "SO3DeLaValleePoussinKernel\n  bandwidth: 25\n  halfwidth: 10°"


def test_str_ofARadonTransformNamesItsKernel():
  assert str(radon(SO3AbelPoissonKernel(0.7))).startswith("Radon transform of Abel Poisson")
