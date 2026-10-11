# the arithmetic the harmonic series on the circle, the sphere and SO(3) share
import numpy as np

from mtex import S1FunHarmonic, S2FunHarmonic, SO3FunHarmonic, crystalFrame, orientation, unimodalODF, vector3d


def test_difference_isTheSumWithTheNegation():
  f = S2FunHarmonic.quadrature(lambda v: v.z ** 2, bandwidth=4)
  v = vector3d.rand(5)
  assert np.allclose((f - 1)(v), f(v) - 1) and np.allclose((1 - f)(v), 1 - f(v))


def test_quotientByANumber_scalesTheCoefficients():
  f = SO3FunHarmonic(unimodalODF(orientation.id(crystalFrame("432"))))
  o = orientation.rand(4, crystalFrame("432"))
  assert np.allclose((f / 2)(o), f(o) / 2)


def test_integerPower_isThePointwisePowerAtTheMultipliedBandwidth():
  f = S1FunHarmonic.quadrature(np.cos, bandwidth=4)
  x = np.linspace(0, 2 * np.pi, 7)
  assert np.allclose((f ** 2)(x), np.cos(x) ** 2)
