import numpy as np

from mtex import Gaussian, calcDensity, discreteSample
from mtex.functions.kde import griddedDensity


# a sample of a Gaussian on the unit interval
def gaussianSample():
  rng = np.random.default_rng(1)
  f = lambda t: Gaussian(0.5, 0.1, t)
  return f, discreteSample(f, 4000, range=[0, 1], rng=rng)


def test_Gaussian_integratesToOne():
  x = np.linspace(-1, 1, 20001)
  assert np.isclose(np.trapezoid(Gaussian(0.1, 0.2, x), x), 1)


def test_Gaussian_withoutPointsIsAFunction():
  assert np.isclose(Gaussian(0, 0.5)(0.0), 1 / 0.5 / np.sqrt(np.pi))


def test_discreteSample_ofAFunctionFollowsIt():
  rng = np.random.default_rng(0)
  x = discreteSample(lambda t: Gaussian(0.3, 0.05, t), 5000, range=[0, 1], rng=rng)
  assert x.shape == (5000,) and abs(np.mean(x) - 0.3) < 0.01 and x.min() >= 0 and x.max() <= 1


def test_calcDensity_inOneDimensionEstimatesTheDensity():
  f, x = gaussianSample()
  fN = calcDensity(x, range=[0, 1])
  assert isinstance(fN, griddedDensity) and 0 < fN.bandwidth < 0.1
  t = np.linspace(0, 1, 2001)
  assert np.isclose(np.trapezoid(fN(t), t), 1, atol=1e-3)
  assert np.max(np.abs(fN(t) - f(t))) < 0.4


# a fixed bandwidth is kept
def test_calcDensity_inOneDimensionKeepsAFixedBandwidth():
  _, x = gaussianSample()
  assert np.isclose(calcDensity(x, range=[0, 1], bandwidth=0.02).bandwidth, 0.02)


def test_calcDensity_inTwoDimensionsEstimatesTheDensity():
  rng = np.random.default_rng(2)
  X = rng.normal(0.5, 0.1, (200, 2))
  fN = calcDensity(X, range=[[0, 0], [1, 1]], rng=rng)
  g = np.linspace(0, 1, 101)
  xx, yy = np.meshgrid(g, g, indexing='ij')
  v = fN(xx, yy)
  assert v.shape == (101, 101) and np.isclose(np.trapezoid(np.trapezoid(v, g), g), 1, atol=0.05)
  assert np.unravel_index(np.argmax(v), v.shape) in [(i, j) for i in range(40, 61) for j in range(40, 61)]


def test_calcDensity_ofPeriodicDataStaysOnTheCircle():
  assert type(calcDensity([0.0, 2.0], periodic=True)).__name__ == "S1FunHarmonic"
