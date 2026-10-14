# the verbs as functions fall through to NumPy for plain arrays
import numpy as np

from mtex import angle, dot, cross, norm, mean, unique, exp, log, inv, vector3d, rotation, rotate, cat, project
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


# from mtex import * replaces max, min, round and sum, which must still do what Python's do on
# Python's numbers and iterables
def test_the_star_import_keeps_the_builtins():
  ns = {}
  exec("from mtex import *", ns)
  assert ns["round"](2.6) == 3 and isinstance(ns["round"](2.6), int) and ns["round"](2.567, 1) == 2.6
  assert list(range(ns["round"](2.6))) == [0, 1, 2]
  assert ns["sum"](x * x for x in range(4)) == 14 and ns["sum"]([[1], [2]], []) == [1, 2]
  assert ns["max"](3, 5) == 5 and ns["max"](["aa", "b"], key=len) == "aa" and ns["max"]([], default=0) == 0
  assert ns["min"]("ab", "c") == "ab"
  # arrays and axes stay NumPy's
  assert np.allclose(ns["round"](np.array([1.26, 2.71]), 1), [1.3, 2.7])
  assert np.allclose(ns["sum"](np.ones((2, 3)), axis=0), [2, 2, 2])
  # and objects their own
  v = vector3d([[1.2, 0, 0], [0, 2.7, 0]])
  assert np.allclose(ns["round"](v).data, v.round().data) and np.allclose(ns["sum"](v).data, [1.2, 2.7, 0])


def _projectCases():
  from mtex import crystalFrame, embedding, orientation, plane3d
  ori = orientation.rand(3, crystalFrame("m-3m"), rng=np.random.default_rng(4))
  return [(embedding(ori), ()), (plane3d(vector3d(0, 0, 2), 4), (vector3d(1, 1, 5),))]


def test_project_isTheMethodOfTheObjectPassed():
  # project(x, ...) reaches x.project(...) for every object that has one
  for x, args in _projectCases():
    assert close(project(x, *args).data, x.project(*args).data)
