# kernels, docs/plans/functions.md F1: the coefficient rules against the closed forms, the
# normalisation, the halfwidths (check_kernelHalfwidth, check_S2KernelNormalization),
# radon, convolution and derivatives
import numpy as np
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


def test_K1_closed_forms_agree_with_their_series():
  # the closed form of every family against the Clenshaw sum of its coefficient rule, far
  # enough out that the truncation is below the tolerance
  t = np.linspace(-0.99, 0.99, 41)
  for psi in so3Families:
    L = psi.bandwidth if isinstance(psi, (SO3DirichletKernel, SO3RestrictedDistanceKernel)) else 400
    tol = 5e-2 if isinstance(psi, (SO3BumpKernel, SO3RestrictedDistanceKernel)) else 1e-6
    sel = np.abs(t) >= 0.02 if isinstance(psi, SO3BumpKernel) else np.ones(t.shape, bool)
    assert np.allclose(psi(t)[sel], SO3Kernel(calcFourier(psi, L))(t)[sel], atol=tol * max(1, abs(psi(1.0))), rtol=0)
  for psi in s2Families:
    L = psi.bandwidth if isinstance(psi, (S2DirichletKernel, S2RestrictedDistanceKernel)) else 400
    tol = 5e-2 if isinstance(psi, (S2BumpKernel, S2RestrictedDistanceKernel)) else 1e-6
    sel = np.abs(t - np.cos(10 * degree)) >= 0.02 if isinstance(psi, S2BumpKernel) else np.ones(t.shape, bool)
    assert np.allclose(psi(t)[sel], S2Kernel(calcFourier(psi, L))(t)[sel], atol=tol * max(1, abs(psi(1.0))), rtol=0)
  x = np.linspace(-3, 3, 41)
  for psi in s1Families:
    L = 2000 if isinstance(psi, S1BumpKernel) else psi.bandwidth
    sel = np.abs(np.abs(x) - 10 * degree) >= 0.05 if isinstance(psi, S1BumpKernel) else np.ones(x.shape, bool)
    assert np.allclose(psi(x)[sel], S1Kernel(calcFourier(psi, L))(x)[sel], atol=5e-2 if isinstance(psi, S1BumpKernel) else 1e-9, rtol=0)
  # the square singularity coefficients are stable far beyond MATLAB's cut
  A = calcFourier(SO3SquareSingularityKernel(0.5), 400)
  assert np.all(np.isfinite(A)) and A[0] == 1 and abs(A[400]) < 1e-100
  assert np.isclose(A[1], 3 * ((1 + 0.25) / 1 - 1 / np.log(3)))
  # the generic Clenshaw path is what a coefficient kernel evaluates, in the given precision
  psi = SO3DeLaValleePoussinKernel(20)
  assert np.isclose(SO3Kernel(calcFourier(psi, 20))(0.7), psi(0.7))
  assert np.isclose(S2Kernel(calcFourier(radon(psi), 20))(0.7), radon(psi)(0.7))
  assert psi.A.shape == (psi.bandwidth + 1,) and np.array_equal(psi.A, calcFourier(psi))
  # the two sided Fourier array of a circle kernel
  assert np.array_equal(S1FejerKernel(3).fhat, [0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25])
  assert S1Kernel([0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25]) == S1FejerKernel(3)


def test_K2_densities_have_mean_one():
  for psi in so3Families + so3Series + s2Families + s1Families:
    if isinstance(psi, (SO3RestrictedDistanceKernel, S2RestrictedDistanceKernel)):
      continue
    assert np.isclose(mean(psi), 1)
    if isinstance(psi, SO3BumpKernel):
      continue
    assert np.isclose(meanByQuadrature(psi), 1, atol=2e-2 if isinstance(psi, (S2BumpKernel, S1BumpKernel)) else 1e-6, rtol=0)
  assert mean(SO3LaplaceKernel()) == 0
  assert mean(SO3SobolevKernel(1, bandwidth=20)) == 0
  # the S2 bump is the indicator of its cap scaled to a density, check_S2KernelNormalization
  hw = 10 * degree
  psi = S2BumpKernel(hw)
  assert psi(np.cos(2 * hw)) == 0
  assert np.isclose(psi(np.cos(hw / 2)), 2 / (1 - np.cos(hw)))
  assert psi(np.cos(hw / 4)) == psi(np.cos(hw / 2))
  for psi in (S2DeLaValleePoussinKernel(halfwidth=hw), S2BumpKernel(hw), S2DirichletKernel(16)):
    assert abs(psi.A[0] - 1) < 1e-4


def test_K3_halfwidth():
  # check_kernelHalfwidth: the analytic halfwidth survives the round trip through the
  # coefficients, within max(0.02 degree, 1 %)
  cases = [SO3AbelPoissonKernel(0.4), SO3AbelPoissonKernel(0.7), SO3DeLaValleePoussinKernel(1), SO3DeLaValleePoussinKernel(100),
           SO3DirichletKernel(1), SO3DirichletKernel(50), SO3GaussWeierstrassKernel(0.1), SO3GaussWeierstrassKernel(0.5),
           SO3GaussWeierstrassKernel(1), SO3SquareSingularityKernel(0.091), SO3SquareSingularityKernel(0.5),
           SO3SquareSingularityKernel(0.99), SO3vonMisesFisherKernel(0.347), SO3vonMisesFisherKernel(10), SO3LaplaceKernel(),
           SO3SobolevKernel(0.1, bandwidth=20), SO3SobolevKernel(1.1, bandwidth=20), SO3SobolevKernel(5, bandwidth=20)]
  for psi in cases:
    h1, h2 = halfwidth(psi), halfwidth(SO3Kernel(psi.A))
    assert abs(h1 - h2) <= max(0.02 * degree, 0.01 * h1)
    assert 0 < h1 < pi
  # construction from a halfwidth
  for hw in (2 * degree, 10 * degree, 30 * degree):
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
  assert abs(S1DeLaValleePoussinKernel(halfwidth=3 * degree).halfwidth() - 3 * degree) < 0.3 * degree
  # the halfwidth of the S1 families against a direct scan
  x = np.linspace(0, pi, 100001)
  for psi in s1Families:
    i = np.flatnonzero(psi(x) <= psi(0.0) / 2)[0]
    assert np.isclose(psi.halfwidth(), x[i], atol=1e-4)


def test_K4_radon_and_convolution():
  # the Radon transform of an SO(3) kernel has the same coefficients as a Legendre series,
  # and the closed forms MTEX gives for it agree with that series
  t = np.linspace(-0.95, 0.95, 21)
  for psi in (SO3DeLaValleePoussinKernel(20), SO3AbelPoissonKernel(0.7), SO3vonMisesFisherKernel(10),
              fibreVonMisesFisherKernel(20), SO3SquareSingularityKernel(0.5), SO3BumpKernel(20 * degree), SO3DirichletKernel(12)):
    R = radon(psi)
    assert isinstance(R, S2Kernel)
    assert np.allclose(calcFourier(R, 30), calcFourier(psi, 30))
    L = 2000 if isinstance(psi, SO3BumpKernel) else 12 if isinstance(psi, SO3DirichletKernel) else 400
    assert np.allclose(R(t), S2Kernel(calcFourier(psi, L))(t), atol=5e-2 if isinstance(psi, SO3BumpKernel) else 1e-6, rtol=0)
    assert np.isclose(mean(R), 1)
  # the Radon transform of the de la Vallee Poussin kernel is the S2 member of the family
  assert radon(SO3DeLaValleePoussinKernel(20)) == S2DeLaValleePoussinKernel(20)
  assert isinstance(radon(SO3DeLaValleePoussinKernel(20)), S2DeLaValleePoussinKernel)
  # convolution multiplies coefficients and divides by 2n+1
  psi1, psi2 = SO3DeLaValleePoussinKernel(20), SO3AbelPoissonKernel(0.7)
  c = conv(psi1, psi2)
  L = c.bandwidth
  assert np.allclose(c.A, calcFourier(psi1, L) * calcFourier(psi2, L) / (2 * np.arange(L + 1) + 1))
  assert isinstance(conv(psi1, radon(psi2)), S2Kernel)
  assert conv(psi1, SO3DirichletKernel(5)).bandwidth == 5
  assert np.allclose(conv(psi1, SO3DirichletKernel(5)).A, calcFourier(psi1, 5))
  assert np.allclose(conv(S1FejerKernel(3), S1DirichletKernel(3)).fhat, S1FejerKernel(3).fhat)
  # the algebra
  s = 2 * psi1 + psi2
  assert np.allclose(s.A, 2 * calcFourier(psi1, s.bandwidth) + calcFourier(psi2, s.bandwidth))
  assert np.allclose((psi1 - psi1 / 2).A, psi1.A / 2)
  assert np.isclose(norm(SO3DirichletKernel(3)), np.sqrt(1 + 9 + 25 + 49))
  assert np.isclose(norm(S2DirichletKernel(3)), np.sqrt(1 + 3 + 5 + 7))
  assert np.isclose(norm(S1DirichletKernel(3)), np.sqrt(7))
  # the Funk transform of a sphere kernel
  assert np.allclose(radon(S2DirichletKernel(4)).A, [1, 0, -5 / 2, 0, 9 * 3 / 8])


def test_K5_derivatives():
  # the closed forms, and the series formula on kernels that are their own series
  for psi in (SO3DeLaValleePoussinKernel(20), SO3vonMisesFisherKernel(10), SO3Kernel(calcFourier(SO3AbelPoissonKernel(0.7), 60)),
              SO3Kernel(calcFourier(SO3GaussWeierstrassKernel(0.1), 60)), SO3Kernel(calcFourier(SO3SquareSingularityKernel(0.5), 60)),
              SO3Kernel(calcFourier(SO3DeLaValleePoussinKernel(20), 20))):
    for omega in (0.2, 0.7, 1.5, 2.8):
      assert np.isclose(grad(psi, np.cos(omega / 2)), centralDifference(lambda w: psi(np.cos(w / 2)), omega), atol=1e-6 * max(1, psi(1.0)), rtol=0)
    assert grad(psi, 1.0) == 0
    assert np.isclose(psi.grad()(0.8), psi.grad(0.8))
  for psi in (S2DeLaValleePoussinKernel(20), S2Kernel(calcFourier(S2DeLaValleePoussinKernel(20), 20)),
              S2Kernel(calcFourier(SO3AbelPoissonKernel(0.7), 60)), S2Kernel(calcFourier(radon(SO3vonMisesFisherKernel(10)), 60))):
    for theta in (0.2, 0.7, 1.5, 2.8):
      assert np.isclose(grad(psi, np.cos(theta)), centralDifference(lambda w: psi(np.cos(w)), theta), atol=1e-6 * max(1, psi(1.0)), rtol=0)
    assert np.isclose(psi.grad(0.3, polynomial=True), centralDifference(psi, 0.3), atol=1e-6 * max(1, psi(1.0)), rtol=0)
    assert np.isclose(psi.grad()(0.3), psi.grad(0.3))
    # the Legendre series of dpsi/dt, exact for a polynomial
    if type(psi) is S2Kernel:
      assert np.isclose(psi.grad(polynomial=True)(0.3), centralDifference(psi, 0.3), atol=1e-6 * max(1, psi(1.0)), rtol=0)
  for psi in (S1FejerKernel(12), S1DeLaValleePoussinKernel(12), S1GaussKernel(0.3)):
    for x in (0.2, 0.7, 1.5):
      assert np.isclose(grad(psi, x), centralDifference(psi, x), atol=1e-5, rtol=0)
  assert SO3BumpKernel(0.2).grad(0.5) == 0 and S2BumpKernel(0.2).grad(0.5) == 0 and S1BumpKernel(0.2).grad(0.5) == 0


def test_K6_handles_bandwidth_arguments_precision():
  # a handle recovers the coefficients of the family it wraps
  psi = SO3DeLaValleePoussinKernel(20)
  h = SO3KernelHandle(lambda t: psi(t), bandwidth=20)
  assert np.allclose(h.A, calcFourier(psi, 20), atol=1e-10)
  psi2 = S2DeLaValleePoussinKernel(20)
  h2 = S2KernelHandle(lambda t: psi2(t), bandwidth=20)
  assert np.allclose(h2.A, calcFourier(psi2, 20), atol=1e-10)
  h1 = S1KernelHandle(lambda x: S1FejerKernel(12)(x), bandwidth=12)
  assert np.allclose(h1.fhat, S1FejerKernel(12).fhat, atol=1e-10)
  assert np.isclose(h(0.7), psi(0.7)) and np.isclose(h2(0.7), psi2(0.7)) and np.isclose(h1(0.7), S1FejerKernel(12)(0.7))
  # an integer de la Vallee Poussin parameter is a polynomial of that degree, and the
  # cutoff rule drops its negligible top coefficients
  assert 10 < psi.bandwidth <= 20
  assert np.all(np.abs(calcFourier(psi, 30)[21:]) < 1e-12)
  assert SO3DeLaValleePoussinKernel(halfwidth=10 * degree).bandwidth == 25
  assert SO3vonMisesFisherKernel(halfwidth=10 * degree).bandwidth < SO3vonMisesFisherKernel(halfwidth=2 * degree).bandwidth
  assert SO3DirichletKernel(7).bandwidth == 7
  # rotations and vectors as arguments
  r = rotation.rand(rng=rng)
  assert np.isclose(psi(r), psi(np.cos(r.angle() / 2)))
  assert np.isclose(psi(r, rotation.id), psi(r))
  rs = rotation.rand(3, 4, rng=rng)
  assert psi(rs).shape == (3, 4)
  v = vector3d.rand(rng=rng)
  assert np.isclose(psi2(v), psi2(v.z))
  assert np.isclose(psi2(v, zvector), psi2(v))
  # single precision arguments evaluate in single precision
  c = SO3Kernel(calcFourier(psi, 20))
  assert c(np.float32(0.9)).dtype == np.float32
  assert np.isclose(c(np.float32(0.9)), psi(0.9), atol=1e-5 * psi(1.0), rtol=0)
  assert radon(c)(np.float32(0.9)).dtype == np.float32
  assert S1FejerKernel(12)(np.float32(0.3)).dtype == np.float32
  # display
  assert str(SO3DeLaValleePoussinKernel(halfwidth=10 * degree)) == "de la Vallee Poussin, halfwidth 10°"
  assert repr(SO3DeLaValleePoussinKernel(halfwidth=10 * degree)) == "SO3DeLaValleePoussinKernel\n  bandwidth: 25\n  halfwidth: 10°"
  assert str(radon(SO3AbelPoissonKernel(0.7))).startswith("Radon transform of Abel Poisson")
