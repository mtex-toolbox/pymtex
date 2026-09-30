# spatial transforms, MATLAB geometry/spatialTransforms
import numpy as np
import pytest

from mtex import (EBSD, crystalFrame, degree, discretize, displacement, fitStage, inv, norm, phaseList, rotation, spatialTransformComposite,
                  spatialTransformDrift, spatialTransformField, spatialTransformHandle, spatialTransformId, spatialTransformInverse,
                  spatialTransformPoly, spatialTransformProjective, spatialTransformRigid, spatialTransformShift, spatialTransformTilt,
                  transform, vector3d)

rng = np.random.default_rng(5)


def points(n=300):
  return vector3d(np.column_stack([rng.uniform(0, 50, n), rng.uniform(0, 40, n), np.zeros(n)]))


H0 = [[1, 0.02, 3], [-0.01, 1.05, -2], [1e-4, 2e-4, 1]]


def test_the_transforms_move_x_and_y_and_keep_the_rest():
  p = points(5)
  p3 = vector3d(p.data + [0, 0, 7.0])
  R = spatialTransformRigid(vector3d(5, -2, 0))
  assert np.allclose((R * p3).data, p3.data + [5, -2, 0]) and (R * p3).shape == p3.shape
  S = spatialTransformShift([[2, 0, 1], [0, 3, -1], [0, 0, 1]])
  assert np.allclose((S * p).data[:, :2], p.data[:, :2] * [2, 3] + [1, -1])
  assert np.allclose(spatialTransformId().eval(p).data, p.data) and spatialTransformId().isid
  assert np.allclose(displacement(R, p).data, [[5, -2, 0]] * 5)
  with pytest.raises(ValueError):
    spatialTransformShift([[1, 0, 0], [0, 1, 0], [0.1, 0, 1]])
  # a transform is a function of the positions
  assert np.allclose(R(p).data, (R * p).data)


def test_products_absorb_matrices_and_sums_chain_stages():
  p = points()
  R = spatialTransformRigid(vector3d(1, -2, 0))
  S = spatialTransformShift([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]])
  P = spatialTransformProjective(H0)
  # T1 * T2 moves by T2 first, and two matrices become one
  RS = S * R
  assert isinstance(RS, spatialTransformShift) and np.allclose((RS * p).data, (S * (R * p)).data)
  assert isinstance(P * S, spatialTransformProjective) and np.allclose(((P * S) * p).data, (P * (S * p)).data)
  # + keeps the stages, the first applied first
  C = R + S
  assert isinstance(C, spatialTransformComposite) and len(C.stages) == 2 and np.allclose((C * p).data, (S * (R * p)).data)
  assert spatialTransformId() + R is R and R + spatialTransformId() is R
  # a product with a transform that has no matrix is a composite
  D = spatialTransformDrift([0, 40], [[1, 0], [3, 0]])
  assert isinstance(D * R, spatialTransformComposite) and np.allclose(((D * R) * p).data, (D * (R * p)).data, equal_nan=True)
  assert spatialTransformShift() * D is D
  assert "shift-drift" in repr(spatialTransformShift() + D)


def test_inverses():
  p = points()
  for T in (spatialTransformRigid(vector3d(3, 1, 0)), spatialTransformShift([[1.1, 0.2, 3], [-0.1, 0.9, 1], [0, 0, 1]]),
            spatialTransformProjective(H0), spatialTransformPoly(np.array([[1, 0.5], [0.01, 0.02], [0.03, -0.01]]))):
    assert np.max(norm(inv(T) * (T * p) - p)) < 1e-9
  Q = spatialTransformPoly(np.array([[1, 0.5], [0.01, 0.02], [0.03, -0.01], [1e-4, 0], [2e-4, 1e-4], [0, -1e-4]]))
  Qi = inv(Q)
  assert isinstance(Qi, spatialTransformInverse) and np.max(norm(Qi.eval(Q * p) - p)) < 1e-8 and inv(Qi) is Q
  C = spatialTransformRigid(vector3d(1, 0, 0)) + spatialTransformProjective(H0)
  assert np.max(norm(inv(C) * (C * p) - p)) < 1e-9
  H = spatialTransformHandle(lambda v: v * 2, lambda v: v * 0.5)
  assert np.allclose((inv(H) * (H * p)).data, p.data)
  with pytest.raises(ValueError):
    inv(spatialTransformHandle(lambda v: v * 2))


def test_the_fits_recover_the_transforms_and_resist_outliers():
  p = points(458)
  T0 = spatialTransformProjective(H0)
  q = T0 * p
  assert np.max(norm(spatialTransformProjective.fit(p, q) * p - q)) < 1e-9
  bad = vector3d(q.data.copy())
  bad.data.reshape(-1, 3)[::10] += [30, -40, 0]
  assert np.max(norm(spatialTransformProjective.fit(p, bad) * p - q)) < 1e-8
  # the plain least squares is pulled away by the outliers
  assert np.max(norm(spatialTransformProjective.fit(p, bad, robust=False) * p - q)) > 1
  S0 = spatialTransformShift([[1.02, 0.03, 2], [-0.01, 0.98, -1], [0, 0, 1]])
  assert np.allclose(spatialTransformShift.fit(p, S0 * p).M, S0.M)
  assert np.allclose(spatialTransformRigid.fit(p, p + vector3d(2, -1, 0)).u.data, [2, -1, 0])
  P2 = spatialTransformPoly.fit(p, q, degree=2)
  assert P2.degree == 2 and np.max(norm(P2 * p - q)) < 0.2
  # a weight of zero leaves a point out
  w = np.ones(p.size)
  w[::10] = 0
  assert np.max(norm(spatialTransformProjective.fit(p, bad, weights=w, robust=False) * p - q)) < 1e-8
  with pytest.raises(ValueError):
    spatialTransformProjective.fit(p[:3], q[:3])


def test_drift_field_and_tilt():
  p = points()
  # a drift along the slow scan direction y, the median of every scan line
  lines = vector3d(np.column_stack([np.tile(np.arange(10.0), 5), np.repeat(np.arange(5.0), 10), np.zeros(50)]))
  shift = np.repeat([[0.0, 0], [0.1, 0], [0.3, 0.1], [0.4, 0.1], [0.4, 0.2]], 10, axis=0)
  moved = vector3d(lines.data + np.column_stack([shift, np.zeros(50)]))
  D = spatialTransformDrift.fit(lines, moved)
  assert np.allclose(D.s, np.arange(5)) and np.allclose((D * lines).data, moved.data)
  assert np.isnan((D * vector3d(0, 7, 0)).x) and not np.isnan((spatialTransformDrift(D.s, D.u, extrapolate=True) * vector3d(0, 7, 0)).x)
  # beyond its knots a drift is NaN unless it extrapolates, and so is its inverse there
  De = spatialTransformDrift(D.s, D.u, extrapolate=True)
  assert np.max(norm(inv(De) * (De * lines) - lines)) < 1e-9
  # a field reproduces its own nodes, and a discretised transform is exact there
  C = spatialTransformRigid(vector3d(1, -2, 0)) + spatialTransformShift([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]])
  F = discretize(C, p)
  assert isinstance(F, spatialTransformField) and np.max(norm(F * p - C * p)) < 1e-12
  assert np.max(norm(spatialTransformField.fit(p, C * p) * p - C * p)) < 1e-12
  # the tilt is fitted stage by stage; an unfitted stage is the identity
  T = spatialTransformTilt()
  assert len(T.stages) == 3 and T.isid
  q = spatialTransformProjective(H0) * p
  T = fitStage(T, 0, p, q)
  assert isinstance(T, spatialTransformTilt) and np.max(norm(T * p - q)) < 1e-9
  with pytest.raises(IndexError):
    fitStage(T, 3, p, q)


def test_by_tilt_and_a_transformed_map():
  c = vector3d(10, 20, 0)
  T = spatialTransformProjective.byTilt(20 * degree, 500, c)
  # the centre stays, and the tilt about x foreshortens along y
  assert np.allclose((T * c).data, c.data)
  a, b = T * vector3d(10, 10, 0), T * vector3d(10, 30, 0)
  assert abs(b.y - a.y) < 20 * np.cos(20 * degree) + 1 and abs(b.y - a.y) > 20 * np.cos(20 * degree) - 1
  cs = crystalFrame("m-3m")
  i, j = np.meshgrid(np.arange(20.0), np.arange(15.0))
  e = EBSD(np.column_stack([i.ravel(), j.ravel(), np.zeros(i.size)]), rotation.rand(i.size, rng=rng), np.ones(i.size, dtype=np.int8), phaseList(cs))
  S = spatialTransformShift([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]])
  eT = transform(e, S)
  assert np.max(norm(transform(eT, inv(S)).pos - e.pos)) < 1e-12
  assert np.array_equal(transform(e, T).lattice.ij, e.lattice.ij)
