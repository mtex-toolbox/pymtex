# the verbs as functions fall through to NumPy for plain arrays
import numpy as np

from mtex import angle, dot, cross, norm, mean, unique, exp, log, inv, vector3d, rotation, rotate, cat
from conftest import close


def test_numpy_fallback():
  assert close(dot([1, 2], [3, 4]), 11) and close(cross([1, 0, 0], [0, 1, 0]), [0, 0, 1])
  assert close(norm([3, 4]), 5) and close(mean([1, 3]), 2) and unique([2, 1, 2]).tolist() == [1, 2]
  assert close(exp(0), 1) and close(log(1), 0) and close(angle(1j), np.pi / 2)
  assert close(inv(np.eye(2)), np.eye(2))


def test_methods_through_functions():
  r = rotation.rand(3, rng=np.random.default_rng(1))
  v = vector3d.rand(3, rng=np.random.default_rng(2))
  assert np.all(rotate(v, r) == r * v) and close(angle(r), r.angle()) and np.all(inv(r) == r.inv())
  assert cat(v, v).shape == (6,)
