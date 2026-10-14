# S1FunHarmonic and the density on the circle (docs/plans/functions.md F3e, check_S1Fun)
import numpy as np
import pytest
from numpy import pi

from mtex import S1FejerKernel, S1FunHarmonic, S1GaussKernel, calcDensity, conv, mean, norm, quadratureS1Grid

rng = np.random.default_rng(35)

# the angles to evaluate at and a random series of bandwidth 4
xs = rng.random(20) * 2 * pi
c = rng.standard_normal(9) + 1j * rng.standard_normal(9)
g = S1FunHarmonic(c)


# a density of angles with weights, the one the density tests share
def weightedDensity():
  return calcDensity([0.0, 2.0], weights=[1, 3], sigma=0.2)


def test_S1FunHarmonic_ofAKernelIsTheRealEvenKernelOfMeanOne():
  f = S1FunHarmonic(S1FejerKernel(12))
  assert np.allclose(f(xs), S1FejerKernel(12)(xs)) and np.isclose(mean(f), 1) and f.isReal and f.even


def test_S1FunHarmonic_evaluatesItsFourierSeries():
  assert np.allclose(g(xs), [np.sum(c * np.exp(-1j * np.arange(-4, 5) * x)) for x in xs])


def test_S1FunHarmonic_algebraIsPointwise():
  f = S1FunHarmonic(S1FejerKernel(12))
  assert np.allclose((f + g)(xs), f(xs) + g(xs)) and np.allclose((2 * g - 1)(xs), 2 * g(xs) - 1)


def test_diff_isTheDerivative():
  assert np.allclose(g.diff()(xs), [(g(x + 1e-6) - g(x - 1e-6)) / 2e-6 for x in xs], atol=1e-6)


def test_rotate_shiftsTheArgument():
  assert np.allclose(g.rotate(0.4)(xs), g(xs - 0.4))


def test_S1FunHarmonic_fromACallableIsTheSeries():
  assert S1FunHarmonic(lambda x: g(x), bandwidth=4).isclose(g)


def test_quadrature_recoversTheSeries():
  q = quadratureS1Grid(4)
  assert S1FunHarmonic.quadrature(q.nodes(), g(q.nodes()), bandwidth=4, weights=q.weights).isclose(g)


def test_norm_isTheL2Norm():
  q8 = quadratureS1Grid(8)
  assert np.isclose(norm(g) ** 2, np.sum(np.abs(g(q8.nodes())) ** 2) / len(q8))


def test_conj_isPointwise():
  assert np.allclose(g.conj()(xs), np.conj(g(xs)))


def test_repr_namesTheClass():
  assert "S1FunHarmonic" in repr(S1FunHarmonic(S1FejerKernel(12)))


def test_antipodal_isASeriesOfEvenModes():
  assert not g.antipodal and S1FunHarmonic([1, 0, 2, 0, 1]).antipodal


# a density of angles has mean one; MATLAB's cutoff at round(4 / sigma) leaves ripples of e^-8
def test_calcDensity_ofAnglesHasMeanOne():
  d = weightedDensity()
  assert d.bandwidth == 20 and np.isclose(mean(d), 1) and d.min()[0] > -1e-3


# a density of angles follows its weights
def test_calcDensity_followsItsWeights():
  d = weightedDensity()
  assert np.isclose(d(2.0), 3 * d(0.0), rtol=1e-3)
  v, x = d.max()
  assert abs(x - 2.0) < 1e-3 and np.isclose(v, d(2.0), rtol=1e-6)


# smoothing twice adds the variances
def test_conv_ofADensityAddsTheVariances():
  d = weightedDensity()
  assert conv(d, S1GaussKernel(0.3)).isclose(calcDensity([0.0, 2.0], weights=[1, 3], sigma=np.sqrt(0.2 ** 2 + 0.3 ** 2), bandwidth=20))


# axes: the density repeats after half a turn and keeps only the even modes
def test_calcDensity_ofAxesRepeatsAfterHalfATurn():
  a = calcDensity([0.3, 1.0], sigma=0.2, antipodal=True)
  assert np.isclose(a(0.3), a(0.3 + pi)) and a.antipodal


# Silverman's rule gives a width, and a kernel of one's own is taken as it is
def test_calcDensity_withoutAWidthHasMeanOne():
  assert np.isclose(mean(calcDensity(xs, periodic=True)), 1) and np.isclose(mean(calcDensity(xs, kernel=S1FejerKernel(12))), 1)


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_max_numLocalFindsTheMaximaOfAPeriodicWave(k):
  # cos(k (x - x0)) has k maxima of value one, 2 pi / k apart, and k minima between them
  x0 = 0.3
  f = S1FunHarmonic(lambda x: np.cos(k * (x - x0)), bandwidth=8)
  for op, top in ((S1FunHarmonic.max, 1), (S1FunHarmonic.min, -1)):
    v, x = op(f, numLocal=k + 2)
    shift = 0 if top == 1 else pi / k
    assert len(v) == k and np.allclose(v, top, atol=1e-9)
    expected = np.sort((x0 + shift + 2 * pi * np.arange(k) / k) % (2 * pi))
    assert np.allclose(np.sort(x), expected, atol=1e-5)


@pytest.mark.parametrize("other", [0.2, -0.5, S1FunHarmonic([0.5, 0.0, 0.5])])
def test_max_withANumberOrAFunctionIsThePointwiseExtremumByQuadrature(other):
  # MATLAB's max(sF, c) and max(sF1, sF2): the quadrature of the pointwise extremum
  f = S1FunHarmonic(lambda x: np.sin(x) + 0.3 * np.cos(2 * x), bandwidth=4)
  value = (lambda x: other(x)) if callable(other) else (lambda x: other)
  for op, npop in ((S1FunHarmonic.max, np.maximum), (S1FunHarmonic.min, np.minimum)):
    h = op(f, other, bandwidth=32)
    assert isinstance(h, S1FunHarmonic) and h.isclose(S1FunHarmonic(lambda x: npop(f(x), value(x)), bandwidth=32), rtol=1e-12)
