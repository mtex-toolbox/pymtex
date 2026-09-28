# the Euler conventions, matrices, products and angles against SciPy
import numpy as np
from numpy import pi
from scipy.spatial.transform import Rotation as SciRot

import mtex
from conftest import close, need

rng = np.random.default_rng(1)
TRIPLES = np.column_stack([rng.uniform(0, 2 * pi, 50), rng.uniform(0, pi, 50), rng.uniform(0, 2 * pi, 50)])


# SciPy stores (x, y, z, w); MTEX and the statements store (a, b, c, d) = (w, x, y, z)
def abcd(sci):
  q = sci.as_quat()
  return np.roll(q, 1, axis=-1)


def same_rotation(q1, q2, tol=1e-12):
  return np.all(np.abs(np.abs(np.sum(q1 * q2, axis=-1)) - 1) < tol)


def test_euler_against_scipy():
  need("rotation")
  r = mtex.rotation.byEuler(*TRIPLES.T)
  assert same_rotation(r.data, abcd(SciRot.from_euler("ZXZ", TRIPLES)))
  r = mtex.rotation.byEuler(*TRIPLES.T, "ZYZ")
  assert same_rotation(r.data, abcd(SciRot.from_euler("ZYZ", TRIPLES)))


def test_matrix_and_product_against_scipy():
  need("rotation", "matrix", "vector3d")
  s1, s2 = SciRot.from_euler("ZXZ", TRIPLES[:10]), SciRot.from_euler("ZXZ", TRIPLES[10:20])
  r1, r2 = mtex.rotation.byEuler(*TRIPLES[:10].T), mtex.rotation.byEuler(*TRIPLES[10:20].T)
  assert close(mtex.matrix(r1), s1.as_matrix())
  assert same_rotation((r1 * r2).data, abcd(s1 * s2))
  v = np.array([1.0, 2.0, 3.0])
  assert close((r1 * mtex.vector3d(*v)).xyz, s1.apply(v))


def test_axis_angle_against_scipy():
  need("rotation", "angle")
  s = SciRot.from_euler("ZXZ", TRIPLES)
  r = mtex.rotation.byEuler(*TRIPLES.T)
  assert close(mtex.angle(r), np.linalg.norm(s.as_rotvec(), axis=-1))

