# S1FunHarmonic and the density on the circle (docs/plans/functions.md F3e, check_S1Fun)
import numpy as np
from numpy import pi

from mtex import S1FejerKernel, S1FunHarmonic, S1GaussKernel, calcDensity, conv, degree, mean, norm, quadratureS1Grid

rng = np.random.default_rng(35)


def test_F3e_S1FunHarmonic():
  f = S1FunHarmonic(S1FejerKernel(12))
  xs = rng.random(20) * 2 * pi
  assert np.allclose(f(xs), S1FejerKernel(12)(xs)) and np.isclose(mean(f), 1) and f.isReal and f.even
  c = rng.standard_normal(9) + 1j * rng.standard_normal(9)
  g = S1FunHarmonic(c)
  assert np.allclose(g(xs), [np.sum(c * np.exp(1j * np.arange(-4, 5) * x)) for x in xs])
  assert np.allclose((f + g)(xs), f(xs) + g(xs)) and np.allclose((2 * g - 1)(xs), 2 * g(xs) - 1)
  assert np.allclose(g.diff()(xs), [(g(x + 1e-6) - g(x - 1e-6)) / 2e-6 for x in xs], atol=1e-6)
  assert np.allclose(g.rotate(0.4)(xs), g(xs - 0.4))
  assert S1FunHarmonic(lambda x: g(x), bandwidth=4).isclose(g)
  q = quadratureS1Grid(4)
  assert S1FunHarmonic.quadrature(q.nodes(), g(q.nodes()), bandwidth=4, weights=q.weights).isclose(g)
  q8 = quadratureS1Grid(8)
  assert np.isclose(norm(g) ** 2, np.sum(np.abs(g(q8.nodes())) ** 2) / len(q8))
  assert np.allclose(g.conj()(xs), np.conj(g(xs)))
  assert "S1FunHarmonic" in repr(f)
  assert not g.antipodal and S1FunHarmonic([1, 0, 2, 0, 1]).antipodal
  # the Gaussian on the circle: its halfwidth is where it has fallen to half its peak
  psi = S1GaussKernel(halfwidth=10 * degree)
  assert np.isclose(psi.halfwidth(), 10 * degree) and np.isclose(psi(10 * degree), psi(0) / 2, rtol=1e-6)
  # a density of angles has mean one, follows its weights, and smoothing twice adds the variances
  d = calcDensity([0.0, 2.0], weights=[1, 3], sigma=0.2)
  assert np.isclose(mean(d), 1) and d.min()[0] > -1e-6
  assert np.isclose(d(2.0), 3 * d(0.0), rtol=1e-6)
  v, x = d.max()
  assert abs(x - 2.0) < 1e-3 and np.isclose(v, d(2.0), rtol=1e-6)
  assert conv(d, S1GaussKernel(0.3)).isclose(calcDensity([0.0, 2.0], weights=[1, 3], sigma=np.sqrt(0.2 ** 2 + 0.3 ** 2)))
  # axes: the density repeats after half a turn and keeps only the even modes
  a = calcDensity([0.3, 1.0], sigma=0.2, antipodal=True)
  assert np.isclose(a(0.3), a(0.3 + pi)) and a.antipodal
  # Silverman's rule gives a width, and a kernel of one's own is taken as it is
  assert np.isclose(mean(calcDensity(xs)), 1) and np.isclose(mean(calcDensity(xs, kernel=S1FejerKernel(12))), 1)
