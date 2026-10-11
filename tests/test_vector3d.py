# vector3d: construction shapes (check_vector3d), the flags, products, means, unique
import numpy as np
import pytest
from numpy import pi

from mtex import vector3d, xvector, yvector, zvector, rotation, degree, angle, dot, cross, norm, cat, orthProj
from conftest import close


def rng():
  return np.random.default_rng(0)


def test_vector3d_ofThreeCoordinateArrays():
  n = 7
  g = rng()
  x, y, z = g.random(n), g.random(n), g.random(n)
  assert vector3d(x, y, z).shape == (n,)


def test_vector3d_ofThreeScalarsIsZeroDimensional():
  assert vector3d(1, 2, 3).shape == () and vector3d(1, 2, 3).size == 1


def test_vector3d_ofAThreeRowOrThreeColumnMatrix():
  n = 7
  xyz = rng().random((3, n))
  assert vector3d(xyz).shape == (n,) and vector3d(xyz.T).shape == (n,)
  assert close(vector3d(xyz).data, vector3d(xyz.T).data)


def test_vector3d_ofArraysWithTheCoordinatesLast():
  g = rng()
  assert vector3d(np.zeros((0, 3))).shape == (0,)
  assert vector3d([1, 2, 3]).shape == () and vector3d(g.random((2, 5, 3))).shape == (2, 5)
  assert vector3d(g.random((3, 3))).data.shape == (3, 3)


def test_vector3d_refusesAnArrayWithoutThreeCoordinates():
  with pytest.raises(ValueError):
    vector3d(rng().random((4, 5)))


def test_vector3d_emptyZerosAndNan():
  assert vector3d().isempty and vector3d.zeros(3).shape == (3,)
  assert np.all(np.isnan(vector3d.nan(2).data))


def test_vector3d_broadcastsTheCoordinates():
  assert vector3d(np.zeros(4), 1, 2).shape == (4,)


@pytest.mark.parametrize("n", [0, 1, 5])
def test_byXYZ_ofTwoOrThreeColumns(n):
  v = vector3d.byXYZ(np.zeros((n, 2)))
  assert v.shape == (n,) and np.all(v.z == 0)
  assert vector3d.byXYZ(np.zeros((n, 3))).shape == (n,)


def test_byXYZ_ofAnEmptyList():
  assert vector3d.byXYZ([]).isempty


def test_byXYZ_readsTheRows():
  assert vector3d.byXYZ([[1, 2, 3], [4, 5, 6]]).z.tolist() == [3, 6]


@pytest.mark.parametrize("shape", [(0, 4), (3, 4), (5, 1)])
def test_byXYZ_refusesOtherColumnCounts(shape):
  with pytest.raises(ValueError):
    vector3d.byXYZ(rng().random(shape))


def test_vector3d_copyKeepsTheFlagsAndOwnsItsData():
  a = vector3d(1, 2, 3, antipodal=True)
  b = vector3d(a)
  assert b.antipodal and not b.isNormalized
  b.x = 5
  assert a.x == 1


def test_vector3d_axesAreNormalized():
  assert vector3d.X.isNormalized and xvector == vector3d.X


def test_vector3d_isNormalizedIsCarriedByIndexingAndReshape():
  v = vector3d(rng().random((4, 3)), isNormalized=True)
  assert v.isNormalized and v[1:3].isNormalized and v.reshape(2, 2).isNormalized


def test_byPolar_givesThePolarAngles():
  v = vector3d.byPolar(0.7, 2.1)
  assert v.isNormalized and close(v.theta, 0.7) and close(v.rho, 2.1)


def test_polar_ofAVectorAlongMinusZ():
  th, rh, r = vector3d(0, 0, -2).polar()
  assert close(th, pi) and close(r, 2)


def test_rho_isInZeroTwoPi():
  assert close(vector3d(-1, 0, 0).rho, pi) and close(vector3d(0, -1, 0).rho, 3 * pi / 2)


def test_byPolar_ofArrays():
  assert vector3d.byPolar([0.1, 0.2], [1.0, 2.0]).shape == (2,)


def test_normalize_setsTheFlag():
  d = vector3d(3, 0, 4).normalize()
  assert close(d.data, [0.6, 0, 0.8]) and d.isNormalized


def test_arithmetic_clearsIsNormalized():
  d = vector3d(3, 0, 4).normalize()
  assert not (d + d).isNormalized and not (2 * d).isNormalized and not (d / 2).isNormalized


def test_arithmetic_keepsAntipodalButForTheSum():
  a = vector3d(0, 0, 1, antipodal=True)
  assert not (a + a).antipodal and (2 * a).antipodal and (-a).antipodal and (a * vector3d(1, 1, 1)).antipodal


def test_arithmetic_withScalarsAndElementwise():
  d = vector3d(3, 0, 4).normalize()
  assert close((d - 1).data, [-0.4, -1, -0.2]) and close((1 - d).data, [0.4, 1, 0.2])
  assert close((d * np.array([2.0])).data, [[1.2, 0, 1.6]])
  assert close((vector3d(1, 2, 3) * vector3d(2, 2, 2)).data, [2, 4, 6])


def test_rotation_keepsTheFlags():
  d = vector3d(3, 0, 4).normalize()
  a = vector3d(0, 0, 1, antipodal=True)
  r = rotation.rand(rng=rng())
  assert (r * d).isNormalized and (r * a).antipodal


def test_dot_andCross():
  u, v = vector3d(1, 0, 0), vector3d(1, 1, 0)
  assert close(dot(u, v), 1) and close(u.dot(v), 1)
  assert close(cross(u, v).data, [0, 0, 1]) and cross(xvector, yvector).isclose(zvector)


def test_angle_betweenTwoVectors():
  assert close(angle(vector3d(1, 0, 0), vector3d(1, 1, 0)), pi / 4)
  assert close(angle(vector3d(1, 0, 0), vector3d(-1, 0, 0)), pi)


def test_angle_andDot_ofAntipodalVectors():
  u = vector3d(1, 0, 0)
  a = vector3d(1, 0, 0, antipodal=True)
  assert close(angle(a, vector3d(-1, 0.1, 0)), np.arctan2(0.1, 1)) and angle(a, vector3d(-1, 0.1, 0)) <= pi / 2
  assert close(dot(a, vector3d(-1, 0, 0)), 1) and close(dot(a, vector3d(-1, 0, 0), antipodal=False), -1)
  assert close(dot(u, vector3d(-1, 0, 0), antipodal=True), 1)


def test_angle_broadcastsToAnOuterProduct():
  g = rng()
  vs = vector3d.rand(5, rng=g)
  ws = vector3d.rand(3, rng=g)
  assert angle(vs[:, None], ws[None, :]).shape == (5, 3)
  assert close(angle(vs[:, None], ws[None, :])[2, 1], angle(vs[2], ws[1]))


def test_angle_inAPlaneCounterclockwiseAboutN():
  assert close(angle(xvector, yvector, N=zvector), pi / 2) and close(angle(yvector, xvector, N=zvector), 3 * pi / 2)
  assert close(angle(yvector, xvector, N=zvector, antipodal=True), pi / 2)


def test_angle_smallAngleStaysAccurate():
  assert close(angle(xvector, vector3d(1, 1e-9, 0)), 1e-9, tol=1e-20)


# isclose compares directions, == the coordinates
def test_isclose_isByDirection_andEqualityByCoordinates():
  assert vector3d(1, 0, 0).isclose(vector3d(2, 0, 0)) and not vector3d(1, 0, 0) == vector3d(2, 0, 0)
  assert not (vector3d(1, 0, 0) == vector3d(-1, 0, 0))
  assert vector3d(1, 0, 0, antipodal=True).isclose(vector3d(-1, 0, 0))
  assert vector3d(1, 0, 0) != vector3d(0, 1, 0)


def test_equality_isElementwise():
  g = rng()
  assert (vector3d.rand(4, rng=g) == vector3d.rand(4, rng=g)).shape == (4,)


def test_norm_andNormalizeOfTheZeroVector():
  v = vector3d([[3, 0, 4], [0, 0, 0]])
  assert close(norm(v), [5, 0])
  assert np.all(np.isnan(v.normalize().data[1])) and close(v.normalize(zeroToZero=True).data[1], [0, 0, 0])


def test_orth_isOrthogonal():
  assert close(vector3d(1, 2, 3).orth().dot(vector3d(1, 2, 3)), 0)
  assert close(vector3d(0, 2, 3).orth().data, [1, 0, 0])


# poles on the girdle about z
def girdle():
  return rotation.byAxisAngle(zvector, np.linspace(0, pi, 20)) * xvector


def test_perp_isTheNormalOfAGirdle():
  N = girdle().perp()
  assert N.antipodal and N.isclose(zvector)


def test_perp_ofATiltedGirdle():
  tilted = girdle() + 0.01 * vector3d.rand(20, rng=rng())
  assert angle(tilted.perp(), zvector) < 2 * degree


def test_eig_smallestAxisIsTheGirdleNormal():
  w, axes = girdle().eig()
  assert axes[0].isclose(zvector) and close(w[0], 0)


def test_mean_andSum():
  v = vector3d([[1, 0, 0], [0, 1, 0]])
  assert close(v.mean().data, [0.5, 0.5, 0]) and not v.mean().isNormalized
  assert close(v.sum().data, [1, 1, 0])


def test_mean_weightedIsNormalized():
  m = vector3d([[1, 0, 0], [0, 1, 0]]).mean(weights=[1, 3])
  assert m.isNormalized and close(m.data, np.array([1, 3, 0]) / np.sqrt(10))


def test_mean_ofAntipodalAxes():
  axes = cat(vector3d(0, 0, 1, antipodal=True), vector3d(0, 0, -1, antipodal=True))
  assert axes.mean().isclose(zvector) and axes.mean().antipodal and axes.mean().isNormalized
  tilted = cat(vector3d(0, 0, 1, antipodal=True), vector3d(0, 0.01, -1, antipodal=True))
  assert angle(tilted.mean(), zvector) < 0.01
  assert vector3d([[1, 0, 0], [0, 1, 0]]).mean(antipodal=True).antipodal


def test_mean_andSumAlongAnAxis():
  grid = vector3d(rng().random((2, 3, 3)))
  assert grid.mean(axis=0).shape == (3,) and grid.sum(axis=1).shape == (2,)


def test_mean_robustIgnoresAnOutlier():
  # robust: an outlier does not pull the mean
  g = rng()
  cloud = cat(vector3d.byPolar(0.01 * g.random(20), 2 * pi * g.random(20)), -zvector)
  assert angle(cloud.mean(robust=True), zvector) < 1 * degree


def test_unique_withAndWithoutAntipodal():
  v = vector3d([[1, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]])
  assert v.unique().shape == (3,)
  assert v.unique(antipodal=True).shape == (2,)
  a = vector3d(v, antipodal=True)
  assert a.unique().shape == (2,) and a.unique(antipodal=False).shape == (3,)


def test_unique_indexAndInverse():
  v = vector3d([[1, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]])
  u, iv, iu = v.unique(return_index=True, return_inverse=True)
  assert np.all(u.data == v.data[iv]) and np.all(u[iu].data == v.data)
  u = v.unique(stable=True, return_index=True)[1]
  assert list(u) == [0, 2, 3]


def test_unique_withinTheTolerance():
  assert vector3d([[1, 0, 0], [1 + 1e-9, 0, 0]]).unique().shape == (1,)


def test_rand_is_uniform():
  v = vector3d.rand(20000, rng=rng())
  assert v.isNormalized and abs(np.mean(v.z)) < 0.03 and abs(np.mean(v.z ** 2) - 1 / 3) < 0.02


def test_rand_shapesAndAntipodal():
  g = rng()
  assert vector3d.rand(2, 3, rng=g).shape == (2, 3) and vector3d.rand((2, 3), rng=g).shape == (2, 3)
  assert vector3d.rand(5, antipodal=True, rng=g).antipodal


def test_det_ofTheAxes():
  assert close(vector3d(1, 0, 0).det(yvector, zvector), 1)


def test_isnan_perVector():
  assert vector3d([[1, np.nan, 0], [1, 2, 3]]).isnan().tolist() == [True, False]


def test_repr_listsFewVectorsAndSizesMany():
  g = rng()
  assert repr(vector3d(1, 2, 3)) == "vector3d (y↓→x)\n  x  y  z\n  1  2  3"
  assert "size: 2" in repr(vector3d.rand(2, rng=g)) and "antipodal: true" in repr(vector3d(1, 0, 0, antipodal=True))
  assert "x  y  z" not in repr(vector3d.rand(30, rng=g))


def test_outer_dot_and_angle_pair_every_entry():
  a = cat(xvector, yvector)
  b = cat(xvector, yvector, zvector)
  D = dot(a, b, outer=True)
  assert D.shape == (2, 3) and np.allclose(D, [[1, 0, 0], [0, 1, 0]])
  A = angle(a, b, outer=True) / degree
  assert np.allclose(A, [[0, 90, 90], [90, 0, 90]])


def test_orthProj_dropsTheComponentAlongTheNormal():
  w = orthProj(vector3d(1, 2, 3), zvector)
  assert np.allclose(w.data, [1, 2, 0])
