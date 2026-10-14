# the family of functions on the sphere, MATLAB's S2Fun, as the base class of every representation
import numpy as np

from mtex import S2Fun, S2FunHandle, S2FunHarmonic, vector3d


def test_S2Fun_isTheBaseOfEveryRepresentation():
  assert isinstance(S2FunHarmonic.quadrature(lambda v: v.z ** 2, bandwidth=4), S2Fun)
  assert isinstance(S2FunHandle(lambda v: v.z), S2Fun)
  assert not isinstance(vector3d(0, 0, 1), S2Fun)


def test_smiley_evenIsAnAntipodalFunction():
  # the flag says what the values do: f(-v) = f(v) exactly when even
  v = vector3d.rand(20)
  for even in (False, True):
    for f in (S2Fun.smiley(even=even, bandwidth=32), S2Fun.smiley(even=even, exact=True)):
      assert f.antipodal == even == bool(np.allclose(f(v), f(-v), atol=1e-12))
