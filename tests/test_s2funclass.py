# the family of functions on the sphere, MATLAB's S2Fun, as the base class of every representation
from mtex import S2Fun, S2FunHandle, S2FunHarmonic, vector3d


def test_S2Fun_isTheBaseOfEveryRepresentation():
  assert isinstance(S2FunHarmonic.quadrature(lambda v: v.z ** 2, bandwidth=4), S2Fun)
  assert isinstance(S2FunHandle(lambda v: v.z), S2Fun)
  assert not isinstance(vector3d(0, 0, 1), S2Fun)
