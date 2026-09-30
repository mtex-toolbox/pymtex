# quaternion, rotation and orthogonalTransform: the algebra, lists, means, precision
import numpy as np
import pytest
from numpy import pi

from mtex import (quaternion, rotation, orthogonalTransform, reflection, vector3d, xvector, yvector,
                  zvector, degree, angle, dot, inv, unique, mean, exp, log, cat)
from conftest import close


def rng():
  return np.random.default_rng(0)


def same(p, q, tol=1e-13):
  return np.all(1 - np.abs(np.sum(p.data[..., :4] * q.data[..., :4], axis=-1)) < tol)


def test_quaternion_ofFourScalarsOrRows():
  q = quaternion(1, 2, 3, 4)
  assert q.shape == () and close(q.data, [1, 2, 3, 4]) and close(q.norm(), np.sqrt(30))
  assert quaternion([[1, 0, 0, 0], [0, 1, 0, 0]]).shape == (2,) and quaternion(np.eye(4)[:, :2]).shape == (2,)
  assert quaternion().isempty and quaternion.id == quaternion(1, 0, 0, 0)


def test_rotation_normalizes():
  q = quaternion(1, 2, 3, 4)
  assert close(rotation(1, 1, 1, 1).data, [0.5, 0.5, 0.5, 0.5])
  assert close(rotation(q).norm(), 1) and isinstance(rotation(q), rotation)


def test_quaternion_ofARotationIsAPlainQuaternion():
  r = rotation(1, 1, 1, 1)
  assert isinstance(quaternion(r), quaternion) and not isinstance(quaternion(r), rotation)


def test_rotation_idAndNan():
  assert rotation.id.shape == () and np.all(np.isnan(rotation.nan(3).data))


def test_rotation_refusesAnImproperTransform():
  with pytest.raises(ValueError):
    rotation(rotation.inversion)


def test_rotation_ofAProperOrthogonalTransform():
  r = rotation(1, 1, 1, 1)
  assert rotation(orthogonalTransform(r)) == r


def test_rotation_refusesAVector():
  with pytest.raises(TypeError):
    rotation(vector3d(1, 2, 3))


def test_rotation_randShapes():
  g = rng()
  assert rotation.rand(4, rng=g).shape == (4,) and rotation.rand(2, 3, rng=g).shape == (2, 3)


def test_rotation_componentGettersAndSetters():
  r = rotation.rand(3, rng=rng())
  assert close(r.a, r.data[:, 0]) and close(r.abcd, r.data)
  r.a = 1
  assert np.all(r.data[:, 0] == 1)


def test_quaternion_HamiltonProductAgainstItsMatrixForm():
  p, q = quaternion(1, 2, 3, 4), quaternion(-1, 0.5, 2, -3)
  P = np.array([[1, -2, -3, -4], [2, 1, -4, 3], [3, 4, 1, -2], [4, -3, 2, 1]])
  assert close((p * q).data, P @ q.data)


def test_quaternion_linearOperations():
  p, q = quaternion(1, 2, 3, 4), quaternion(-1, 0.5, 2, -3)
  assert close((p + q).data, [0, 2.5, 5, 1]) and close((p - q).data, [2, 1.5, 1, 7]) and close((-p).data, [-1, -2, -3, -4])
  assert close((2 * p).data, [2, 4, 6, 8]) and close((p * 2).data, [2, 4, 6, 8]) and close((p / 2).data, [0.5, 1, 1.5, 2])


def test_quaternion_conjInverseNormalizeDot():
  p, q = quaternion(1, 2, 3, 4), quaternion(-1, 0.5, 2, -3)
  assert close(p.conj().data, [1, -2, -3, -4]) and close((p * p.inv()).data, [1, 0, 0, 0])
  assert close(p.normalize().norm(), 1) and close(dot(p, q), -1 + 1 + 6 - 12)


def test_quaternion_equalityIsBySign():
  assert quaternion(0.5, 0.5, 0.5, 0.5) != quaternion(-0.5, -0.5, -0.5, -0.5)
  assert quaternion(0.5, 0.5, 0.5, 0.5) == quaternion(0.5, 0.5, 0.5, 0.5)


def test_quaternion_productClassFollowsTheRotation():
  # the class of a product follows the rotation, quaternion/mtimes
  p, q = quaternion(1, 2, 3, 4), quaternion(-1, 0.5, 2, -3)
  assert isinstance(p * rotation.id, rotation) and isinstance(rotation.id * p, rotation) and isinstance(p * q, quaternion)


def test_rotation_productIsTheMatrixProduct():
  g = rng()
  r1, r2 = rotation.rand(rng=g), rotation.rand(rng=g)
  assert isinstance(r1 * r2, rotation)
  assert close((r1 * r2).matrix(), r1.matrix() @ r2.matrix())


def test_rotation_actionIsAssociativeAndInverted():
  g = rng()
  r1, r2 = rotation.rand(rng=g), rotation.rand(rng=g)
  v = vector3d.rand(rng=g)
  assert (r1 * r2) * v == r1 * (r2 * v)
  assert inv(r1) * (r1 * v) == v and same(r1 * inv(r1), rotation.id)


def test_rotation_equalityIgnoresTheSign():
  g = rng()
  r1, r2 = rotation.rand(rng=g), rotation.rand(rng=g)
  n = rotation._fromData(-r1.data)
  assert n == r1 and close((n * r2).data, -(r1 * r2).data)
  assert same(rotation(0.5, 0.5, 0.5, 0.5), rotation(-0.5, -0.5, -0.5, -0.5))


def test_rotation_productsBroadcast():
  g = rng()
  rs = rotation.rand(5, rng=g)
  vs = vector3d.rand(3, rng=g)
  assert (rs[:, None] * vs[None, :]).shape == (5, 3) and (rs * rs).shape == (5,)
  assert (rs[:, None] * rs[None, :]).shape == (5, 5)


def test_rotation_scaledIsAQuaternion():
  r1 = rotation.rand(rng=rng())
  assert isinstance(2 * r1, quaternion) and not isinstance(2 * r1, rotation)


def test_angle_andAxis_ofAnAxisAngleRotation():
  r = rotation.byAxisAngle(vector3d(1, 2, 3), 40 * degree)
  assert close(r.angle(), 40 * degree) and r.axis() == vector3d(1, 2, 3)
  assert close(angle(r), 40 * degree)
  assert close(rotation.byAxisAngle(zvector, -30 * degree).angle(), 30 * degree)


def test_axis_ofTheIdentityIsX():
  assert rotation.id.axis() == xvector


def test_angle_betweenRotationsFromTheDot():
  g = rng()
  r1, r2 = rotation.rand(10, rng=g), rotation.rand(10, rng=g)
  assert close(angle(r1, r2), 2 * np.arccos(np.abs(dot(r1, r2))), tol=1e-7)
  assert close(angle(r1, r1), 0, tol=1e-15)


def test_dot_ignoresTheSignUnlessNoAntipodal():
  r1 = rotation.rand(10, rng=rng())
  assert close(dot(r1, rotation._fromData(-r1.data)), 1) and close(dot(r1, rotation._fromData(-r1.data), noAntipodal=True), -1)


def test_angle_smallAngleStaysAccurate():
  tiny = rotation.byAxisAngle(zvector, 1e-9)
  assert close(angle(tiny), 1e-9, tol=1e-20) and close(angle(tiny, rotation.id), 1e-9, tol=1e-20)


def test_equality_within_tolerance():
  r = rotation.rand(rng=rng())
  assert r == r * rotation.byAxisAngle(xvector, 1e-4)
  assert r != r * rotation.byAxisAngle(xvector, 1e-2)
  assert r.__eq__(r * rotation.byAxisAngle(xvector, 1e-2), tol=0.1)


def ra():
  return rotation.byAxisAngle(vector3d(1, 1, 1), 70 * degree)


def test_orthogonalTransform_negatedRotationIsImproper():
  r = ra()
  s = -r
  assert isinstance(s, orthogonalTransform) and s.isImproper and s.i and not orthogonalTransform(r).isImproper
  assert s == rotation.inversion * r and orthogonalTransform.inversion == rotation.inversion


def test_inversion_negatesAVector():
  assert rotation.inversion * vector3d(1, 2, 3) == vector3d(-1, -2, -3)


def test_reflection_mirrorsInThePlane():
  n = vector3d(1, 1, 1)
  m = reflection(n)
  assert m * n == -n and m * vector3d(1, -1, 0) == vector3d(1, -1, 0)
  assert m == -rotation.byAxisAngle(n, pi)


def test_orthogonalTransform_productParity():
  r = ra()
  s = -r
  m = reflection(vector3d(1, 1, 1))
  assert not (m * m).isImproper and not (s * s).isImproper and (s * r).isImproper and (r * s).isImproper


def test_reflection_matrixHasDeterminantMinusOne():
  m = reflection(vector3d(1, 1, 1))
  assert close(np.linalg.det(m.matrix()), -1) and close(m.matrix() @ vector3d(0.2, 0.3, 0.4).data, (m * vector3d(0.2, 0.3, 0.4)).data)


def test_orthogonalTransform_angleAxisAndRotationPart():
  r = ra()
  s = -r
  assert close(s.angle(), r.angle()) and s.axis() == r.axis() and s.rot == r


def test_orthogonalTransform_inverseIsTheMatrixInverse():
  s = -ra()
  assert close(inv(s).matrix(), np.linalg.inv(s.matrix()))


def test_orthogonalTransform_equalityRespectsTheParity():
  r = ra()
  s = -r
  assert not (s == r) and (orthogonalTransform(r) == r) and not (r == s)


def test_orthogonalTransform_listCarriesTheParityPerElement():
  r = ra()
  os = cat(orthogonalTransform(r), -r)
  assert os.shape == (2,) and os.isImproper.tolist() == [False, True] and os[1].isImproper
  assert os.Euler()[0].shape == (2,) and "Inv." in repr(os)


def test_orthogonalTransform_byMatrixReadsTheParity():
  r = ra()
  o = orthogonalTransform.byMatrix(-r.matrix())
  assert o.isImproper and close(o.matrix(), -r.matrix())
  assert orthogonalTransform.byMatrix(np.stack([r.matrix(), -r.matrix()])).isImproper.tolist() == [False, True]


def test_rotation_hasNoParity():
  assert not hasattr(rotation.id, "isImproper")


def test_unique_ofRotationsIgnoresTheSign():
  r = rotation.rand(100, rng=rng())
  assert unique(r).shape == (100,)
  assert unique(cat(r, r, rotation._fromData(-r.data))).shape == (100,)


def test_unique_keepsBothParities():
  r = rotation.rand(100, rng=rng())
  o = cat(orthogonalTransform(r), -r)
  assert unique(o).shape == (200,)


def test_unique_withinTheTolerance():
  assert unique(cat(rotation.id, rotation.byAxisAngle(zvector, 1e-5))).shape == (1,)


def test_unique_indexAndInverse():
  r = rotation.rand(100, rng=rng())
  u, iv, iu = unique(cat(r[:3], r[:3]), return_index=True, return_inverse=True)
  assert u.shape == (3,) and np.all(u[iu] == cat(r[:3], r[:3]))


def test_unique_ofQuaternionsRespectsTheSignUnlessAntipodal():
  q = cat(quaternion(1, 0, 0, 0), quaternion(-1, 0, 0, 0))
  assert unique(q).shape == (2,) and unique(q, antipodal=True).shape == (1,)


# a reference rotation and a symmetric cloud of small rotations about it
def meanCloud():
  g = rng()
  ref = rotation.rand(rng=g)
  small = rotation.byAxisAngle(vector3d.rand(500, rng=g), 0.2 * g.random(500))
  return g, ref, small, cat(ref * small, ref * inv(small))


def test_mean_ofASymmetricCloudIsItsCentre():
  _, ref, _, cloud = meanCloud()
  assert same(mean(cloud), ref, 1e-12) and isinstance(mean(cloud), rotation)


def test_mean_ofCopiesIsTheRotation():
  _, ref, _, _ = meanCloud()
  assert same(mean(cat(ref, ref, ref)), ref, 1e-14)


def test_mean_weighted():
  g, ref, _, _ = meanCloud()
  assert same(mean(cat(ref, rotation.rand(rng=g)), weights=[1, 0]), ref, 1e-12)


def test_mean_ofNothingIsNan():
  assert np.all(np.isnan(mean(rotation()).data))


def test_mean_robustIgnoresAnOutlier():
  g, ref, small, _ = meanCloud()
  outliers = cat(ref * small[:20], rotation.rand(rng=g))
  assert angle(mean(outliers, robust=True), ref) < 5 * degree


def test_mean_ofAnArrayIsOneRotation():
  _, _, _, cloud = meanCloud()
  assert mean(cloud.reshape(50, 20)).shape == ()


def test_random_rotations_are_Haar_uniform():
  r = rotation.rand(200000, rng=rng())
  # the density of the rotation angle is (1 - cos w)/pi: its mean is pi/2 + 2/pi
  assert abs(np.mean(r.angle()) - (pi / 2 + 2 / pi)) < 5e-3
  assert np.linalg.norm(np.mean(r.axis().data[:50000], axis=0)) < 2e-2


def test_log_isTheRotationVectorAndExpItsInverse():
  r = rotation.rand(100, rng=rng())
  v = log(r)
  assert isinstance(v, vector3d) and close(v.norm(), r.angle())
  assert same(exp(v), r) and same(exp(log(r)), r)


def test_exp_andLog_ofTheIdentityAndAQuarterTurn():
  assert close(log(rotation.id).data, [0, 0, 0]) and exp(vector3d(0, 0, 0)) == rotation.id
  assert exp(vector3d(0, 0, pi / 2)) * xvector == yvector


def test_log_smallRotationKeepsItsDigits():
  small = rotation.byAxisAngle(xvector, 1e-7)
  assert close(log(small).norm(), 1e-7, tol=1e-20)


# 200 random rotations in double and in single precision
def precisions():
  g = rng()
  r64 = rotation.rand(200, rng=g)
  return g, r64, r64.astype(np.float32)


def test_float32_isKeptByTheOperations():
  _, _, r32 = precisions()
  assert r32.dtype == np.float32
  assert (r32 * r32).dtype == np.float32 and r32.Euler()[0].dtype == np.float32 and r32.matrix().dtype == np.float32


def test_float32_roundTripsWithinSinglePrecision():
  _, r64, r32 = precisions()
  assert np.all(angle(r32.astype(np.float64), r64) < 1e-6)
  assert same(rotation.byEuler(*r32.Euler()).astype(np.float64), r64, 1e-6)
  assert same(rotation.byMatrix(r32.matrix()).astype(np.float64), r64, 1e-6)


def test_float32_rotatesVectorsInSinglePrecision():
  g, r64, r32 = precisions()
  v32 = vector3d.rand(200, rng=g).astype(np.float32)
  assert (r32 * v32).dtype == np.float32
  assert np.all(np.linalg.norm((r32 * v32).data - (r64 * v32.astype(np.float64)).data, axis=-1) < 1e-6)


def test_float32_mixedPrecisionPromotes():
  _, r64, r32 = precisions()
  assert (r32 * r64).dtype == np.float64


def test_repr_ofARotationShowsTheEulerAngles():
  s = repr(rotation.byEuler(10 * degree, 20 * degree, 30 * degree))
  assert s.startswith("rotation\n  Bunge Euler angles in degree\n  phi1  Phi  phi2\n") and "10" in s and "30" in s


def test_repr_ofALongListIsCut():
  g = rng()
  assert "size: 3" in repr(rotation.rand(3, rng=g)) and "⋮" in repr(rotation.rand(30, rng=g))


def test_repr_ofAQuaternion():
  assert repr(quaternion(1, 2, 3, 4)) == "quaternion\n  a  b  c  d\n  1  2  3  4"


def test_rand_within_a_maximum_angle_follows_the_haar_measure():
  r = rotation.rand(50000, maxAngle=12 * degree, rng=np.random.default_rng(5))
  w = r.angle()
  assert w.max() <= 12 * degree + 1e-12
  # the mean angle of the Haar measure restricted to the ball
  t = np.linspace(0, 12 * degree, 2001)
  expected = np.trapezoid(t * (1 - np.cos(t)), t) / np.trapezoid(1 - np.cos(t), t)
  assert abs(np.mean(w) - expected) < 0.05 * degree
