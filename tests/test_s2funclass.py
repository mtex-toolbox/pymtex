# the family of functions on the sphere, MATLAB's S2Fun, as the base class of every representation
import numpy as np
import pytest

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



@pytest.mark.parametrize("halfwidth", [20, 30])
def test_unimodal_isTheKernelGivenAboutTheDirectionGiven(halfwidth):
  # the second positional argument is the kernel, S2Fun.unimodal(v, psi)(w) = psi(v . w)
  from mtex import S2DeLaValleePoussinKernel, degree
  rng = np.random.default_rng(8)
  v, w = vector3d.rand(1, rng=rng), vector3d.rand(30, rng=rng)
  psi = S2DeLaValleePoussinKernel(halfwidth=halfwidth * degree)
  t = np.clip(w.data @ v.data.reshape(3), -1, 1)
  assert np.allclose(np.real(S2Fun.unimodal(v, psi)(w)), psi(t), atol=1e-4 * psi(1.0))
  assert np.allclose(np.real(S2Fun.unimodal(v, halfwidth=halfwidth * degree)(w)), psi(t), atol=1e-4 * psi(1.0))
