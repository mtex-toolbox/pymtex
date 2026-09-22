# quaternion, rotation and orthogonalTransform: the algebra, lists, means, precision
import numpy as np
import pytest
from numpy import pi

from mtex import (quaternion, rotation, orthogonalTransform, reflection, vector3d, xvector, yvector,
                  zvector, degree, angle, dot, inv, unique, mean, exp, log, cat)
from conftest import close

rng = np.random.default_rng(0)


def same(p, q, tol=1e-13):
  return np.all(1 - np.abs(np.sum(p.data[..., :4] * q.data[..., :4], axis=-1)) < tol)


def test_construction():
  q = quaternion(1, 2, 3, 4)
  assert q.shape == () and close(q.data, [1, 2, 3, 4]) and close(q.norm(), np.sqrt(30))
  assert quaternion([[1, 0, 0, 0], [0, 1, 0, 0]]).shape == (2,) and quaternion(np.eye(4)[:, :2]).shape == (2,)
  assert quaternion().isempty and quaternion.id == quaternion(1, 0, 0, 0)
  r = rotation(1, 1, 1, 1)
  assert close(r.data, [0.5, 0.5, 0.5, 0.5])
  assert close(rotation(q).norm(), 1) and isinstance(rotation(q), rotation)
  assert isinstance(quaternion(r), quaternion) and not isinstance(quaternion(r), rotation)
  assert rotation.id.shape == () and np.all(np.isnan(rotation.nan(3).data))
  with pytest.raises(ValueError):
    rotation(rotation.inversion)
  assert rotation(orthogonalTransform(r)) == r
  with pytest.raises(TypeError):
    rotation(vector3d(1, 2, 3))
  assert rotation.rand(4, rng=rng).shape == (4,) and rotation.rand(2, 3, rng=rng).shape == (2, 3)
  r = rotation.rand(3, rng=rng)
  assert close(r.a, r.data[:, 0]) and close(r.abcd, r.data)
  r.a = 1
  assert np.all(r.data[:, 0] == 1)


def test_quaternion_algebra():
  p, q = quaternion(1, 2, 3, 4), quaternion(-1, 0.5, 2, -3)
  # the Hamilton product against its matrix form
  P = np.array([[1, -2, -3, -4], [2, 1, -4, 3], [3, 4, 1, -2], [4, -3, 2, 1]])
  assert close((p * q).data, P @ q.data)
  assert close((p + q).data, [0, 2.5, 5, 1]) and close((p - q).data, [2, 1.5, 1, 7]) and close((-p).data, [-1, -2, -3, -4])
  assert close((2 * p).data, [2, 4, 6, 8]) and close((p * 2).data, [2, 4, 6, 8]) and close((p / 2).data, [0.5, 1, 1.5, 2])
  assert close(p.conj().data, [1, -2, -3, -4]) and close((p * p.inv()).data, [1, 0, 0, 0])
  assert close(p.normalize().norm(), 1) and close(dot(p, q), -1 + 1 + 6 - 12)
  assert quaternion(0.5, 0.5, 0.5, 0.5) != quaternion(-0.5, -0.5, -0.5, -0.5)
  assert quaternion(0.5, 0.5, 0.5, 0.5) == quaternion(0.5, 0.5, 0.5, 0.5)
  # the class of a product follows the rotation, quaternion/mtimes
  assert isinstance(p * rotation.id, rotation) and isinstance(rotation.id * p, rotation) and isinstance(p * q, quaternion)


def test_rotation_products_and_sign():
  r1, r2 = rotation.rand(rng=rng), rotation.rand(rng=rng)
  assert isinstance(r1 * r2, rotation)
  assert close((r1 * r2).matrix(), r1.matrix() @ r2.matrix())
  v = vector3d.rand(rng=rng)
  assert (r1 * r2) * v == r1 * (r2 * v)
  assert inv(r1) * (r1 * v) == v and same(r1 * inv(r1), rotation.id)
  n = rotation._fromData(-r1.data)
  assert n == r1 and close((n * r2).data, -(r1 * r2).data)
  assert same(rotation(0.5, 0.5, 0.5, 0.5), rotation(-0.5, -0.5, -0.5, -0.5))
  # broadcasting
  rs = rotation.rand(5, rng=rng)
  vs = vector3d.rand(3, rng=rng)
  assert (rs[:, None] * vs[None, :]).shape == (5, 3) and (rs * rs).shape == (5,)
  assert (rs[:, None] * rs[None, :]).shape == (5, 5)
  assert isinstance(2 * r1, quaternion) and not isinstance(2 * r1, rotation)


def test_angle_axis_dot():
  r = rotation.byAxisAngle(vector3d(1, 2, 3), 40 * degree)
  assert close(r.angle(), 40 * degree) and r.axis() == vector3d(1, 2, 3)
  assert close(angle(r), 40 * degree)
  assert close(rotation.byAxisAngle(zvector, -30 * degree).angle(), 30 * degree)
  assert rotation.id.axis() == xvector
  r1, r2 = rotation.rand(10, rng=rng), rotation.rand(10, rng=rng)
  assert close(angle(r1, r2), 2 * np.arccos(np.abs(dot(r1, r2))), tol=1e-7)
  assert close(angle(r1, r1), 0, tol=1e-15)
  assert close(dot(r1, rotation._fromData(-r1.data)), 1) and close(dot(r1, rotation._fromData(-r1.data), noAntipodal=True), -1)
  tiny = rotation.byAxisAngle(zvector, 1e-9)
  assert close(angle(tiny), 1e-9, tol=1e-20) and close(angle(tiny, rotation.id), 1e-9, tol=1e-20)


def test_equality_within_tolerance():
  r = rotation.rand(rng=rng)
  assert r == r * rotation.byAxisAngle(xvector, 1e-4)
  assert r != r * rotation.byAxisAngle(xvector, 1e-2)
  assert r.__eq__(r * rotation.byAxisAngle(xvector, 1e-2), tolerance=0.1)


def test_orthogonal_transforms():
  r = rotation.byAxisAngle(vector3d(1, 1, 1), 70 * degree)
  s = -r
  assert isinstance(s, orthogonalTransform) and s.isImproper and s.i and not orthogonalTransform(r).isImproper
  assert s == rotation.inversion * r and orthogonalTransform.inversion == rotation.inversion
  assert rotation.inversion * vector3d(1, 2, 3) == vector3d(-1, -2, -3)
  n = vector3d(1, 1, 1)
  m = reflection(n)
  assert m * n == -n and m * vector3d(1, -1, 0) == vector3d(1, -1, 0)
  assert m == -rotation.byAxisAngle(n, pi)
  assert not (m * m).isImproper and not (s * s).isImproper and (s * r).isImproper and (r * s).isImproper
  assert close(np.linalg.det(m.matrix()), -1) and close(m.matrix() @ vector3d(0.2, 0.3, 0.4).data, (m * vector3d(0.2, 0.3, 0.4)).data)
  assert close(s.angle(), r.angle()) and s.axis() == r.axis() and s.rot == r
  assert close(inv(s).matrix(), np.linalg.inv(s.matrix()))
  assert not (s == r) and (orthogonalTransform(r) == r) and not (r == s)
  os = cat(orthogonalTransform(r), s)
  assert os.shape == (2,) and os.isImproper.tolist() == [False, True] and os[1].isImproper
  assert os.Euler()[0].shape == (2,) and "Inv." in repr(os)
  o = orthogonalTransform.byMatrix(-r.matrix())
  assert o.isImproper and close(o.matrix(), -r.matrix())
  assert orthogonalTransform.byMatrix(np.stack([r.matrix(), -r.matrix()])).isImproper.tolist() == [False, True]
  assert not hasattr(rotation.id, "isImproper")


def test_unique():
  r = rotation.rand(100, rng=rng)
  assert unique(r).shape == (100,)
  assert unique(cat(r, r, rotation._fromData(-r.data))).shape == (100,)
  o = cat(orthogonalTransform(r), -r)
  assert unique(o).shape == (200,)
  assert unique(cat(rotation.id, rotation.byAxisAngle(zvector, 1e-5))).shape == (1,)
  u, iv, iu = unique(cat(r[:3], r[:3]), return_index=True, return_inverse=True)
  assert u.shape == (3,) and np.all(u[iu] == cat(r[:3], r[:3]))
  q = cat(quaternion(1, 0, 0, 0), quaternion(-1, 0, 0, 0))
  assert unique(q).shape == (2,) and unique(q, antipodal=True).shape == (1,)


def test_mean():
  ref = rotation.rand(rng=rng)
  small = rotation.byAxisAngle(vector3d.rand(500, rng=rng), 0.2 * rng.random(500))
  cloud = cat(ref * small, ref * inv(small))
  assert same(mean(cloud), ref, 1e-12) and isinstance(mean(cloud), rotation)
  assert same(mean(cat(ref, ref, ref)), ref, 1e-14)
  assert same(mean(cat(ref, rotation.rand(rng=rng)), weights=[1, 0]), ref, 1e-12)
  assert np.all(np.isnan(mean(rotation()).data))
  outliers = cat(ref * small[:20], rotation.rand(rng=rng))
  assert angle(mean(outliers, robust=True), ref) < 5 * degree
  assert mean(cloud.reshape(50, 20)).shape == ()


def test_random_rotations_are_Haar_uniform():
  r = rotation.rand(200000, rng=rng)
  # the density of the rotation angle is (1 - cos w)/pi: its mean is pi/2 + 2/pi
  assert abs(np.mean(r.angle()) - (pi / 2 + 2 / pi)) < 5e-3
  assert np.linalg.norm(np.mean(r.axis().data[:50000], axis=0)) < 2e-2


def test_exp_and_log():
  r = rotation.rand(100, rng=rng)
  v = log(r)
  assert isinstance(v, vector3d) and close(v.norm(), r.angle())
  assert same(exp(v), r) and same(exp(log(r)), r)
  assert close(log(rotation.id).data, [0, 0, 0]) and exp(vector3d(0, 0, 0)) == rotation.id
  assert exp(vector3d(0, 0, pi / 2)) * xvector == yvector
  # a small rotation keeps its digits
  small = rotation.byAxisAngle(xvector, 1e-7)
  assert close(log(small).norm(), 1e-7, tol=1e-20)


def test_float32():
  r64 = rotation.rand(200, rng=rng)
  r32 = r64.astype(np.float32)
  assert r32.dtype == np.float32
  assert (r32 * r32).dtype == np.float32 and r32.Euler()[0].dtype == np.float32 and r32.matrix().dtype == np.float32
  assert np.all(angle(r32.astype(np.float64), r64) < 1e-6)
  assert same(rotation.byEuler(*r32.Euler()).astype(np.float64), r64, 1e-6)
  assert same(rotation.byMatrix(r32.matrix()).astype(np.float64), r64, 1e-6)
  v32 = vector3d.rand(200, rng=rng).astype(np.float32)
  assert (r32 * v32).dtype == np.float32
  assert np.all(norm := np.linalg.norm((r32 * v32).data - (r64 * v32.astype(np.float64)).data, axis=-1) < 1e-6)
  # mixed precision promotes
  assert (r32 * r64).dtype == np.float64


def test_repr():
  s = repr(rotation.byEuler(10 * degree, 20 * degree, 30 * degree))
  assert s.startswith("rotation\n  Bunge Euler angles in degree\n  phi1  Phi  phi2\n") and "10" in s and "30" in s
  assert "size: 3" in repr(rotation.rand(3, rng=rng)) and "phi1" not in repr(rotation.rand(30, rng=rng))
  assert repr(quaternion(1, 2, 3, 4)) == "quaternion\n  a  b  c  d\n  1  2  3  4"
