# every representation round trips to the same rotation (check_eulerquat), the Euler
# conventions against their definitions, matrices, axis angle, Rodrigues, pairs, fit
import numpy as np
import pytest
from numpy import pi

import mtex
from mtex import rotation, orthogonalTransform, vector3d, xvector, yvector, zvector, degree, Euler, matrix, Rodrigues, angle, setMTEXpref, getMTEXpref
from conftest import close

rng = np.random.default_rng(0)
Rz = lambda w: rotation.byAxisAngle(zvector, w)
Ry = lambda w: rotation.byAxisAngle(yvector, w)
Rx = lambda w: rotation.byAxisAngle(xvector, w)


def same(p, q, tol=1e-13):
  return np.all(1 - np.abs(np.sum(p.data[..., :4] * q.data[..., :4], axis=-1)) < tol)


# the sample of check_eulerquat: general rotations plus the awkward ones
def sample():
  return mtex.cat(rotation.rand(2000, rng=rng), rotation.id,
                  rotation.byEuler(30 * degree, 0, 50 * degree, "Bunge"),
                  rotation.byEuler(200 * degree, 0, 100 * degree, "Bunge"),
                  rotation.byEuler(30 * degree, 1e-8, 50 * degree, "Bunge"),
                  rotation.byEuler(30 * degree, pi, 50 * degree, "Bunge"),
                  rotation.byAxisAngle(xvector, pi))


@pytest.mark.parametrize("conv", ["Bunge", "ABG", "Matthies", "Roe", "Kocks", "Canova", "ZXZ", "ZYZ", "nfft"])
def test_euler_round_trip(conv):
  r = sample()
  a, b, c = Euler(r, conv)
  assert np.all((0 <= a) & (a < 2 * pi + 1e-12)) and np.all((0 <= b) & (b <= pi + 1e-12)) and np.all((0 <= c) & (c < 2 * pi + 1e-12))
  assert same(rotation.byEuler(a, b, c, conv), r)


def test_euler_definitions():
  a, b, c = 0.3, 1.1, 2.5
  assert same(rotation.byEuler(a, b, c, "ZYZ"), Rz(a) * Ry(b) * Rz(c))
  assert same(rotation.byEuler(a, b, c, "Bunge"), Rz(a) * Rx(b) * Rz(c))
  assert same(rotation.byEuler(a, b, c, "Bunge"), rotation.byEuler(a - pi / 2, b, c + pi / 2, "ZYZ"))
  assert same(rotation.byEuler(a, b, c, "Roe"), rotation.byEuler(a, b, c, "Matthies"))
  assert same(rotation.byEuler(a, b, c, "Kocks"), rotation.byEuler(a, b, pi - c, "Matthies"))
  assert same(rotation.byEuler(a, b, c, "Canova"), rotation.byEuler(pi / 2 - a, b, 3 * pi / 2 - c, "Matthies"))
  assert same(rotation.byEuler(a, b, c), rotation.byEuler(a, b, c, "Bunge"))
  with pytest.raises(ValueError):
    rotation.byEuler(a, b, c, "Wrong")


def test_euler_degenerate_case():
  r = rotation.byEuler(10 * degree, 0, 20 * degree)
  for conv in ("Bunge", "Matthies", "Roe", "Kocks", "Canova"):
    a, b, c = Euler(r, conv)
    assert b == 0 and c == 0 and same(rotation.byEuler(a, b, c, conv), r)
  assert close(Euler(r)[0], 30 * degree)
  assert close(r.phi1, 30 * degree) and close(r.Phi, 0) and close(r.phi2, 0)
  a, b, c = Euler(rotation.rand(5, rng=rng))
  assert a.shape == (5,)


def test_session_convention():
  r = rotation.byEuler(0.3, 1.1, 2.5)
  assert getMTEXpref("EulerAngleConvention") == "Bunge"
  setMTEXpref("EulerAngleConvention", "Matthies")
  assert same(rotation.byEuler(*Euler(r)), r) and close(Euler(r)[0], Euler(r, "Matthies")[0])
  assert same(rotation.byEuler(0.3, 1.1, 2.5), Rz(0.3) * Ry(1.1) * Rz(2.5))
  assert "alpha" in repr(r)
  mtex.reset()
  assert getMTEXpref("EulerAngleConvention") == "Bunge"


def test_matrix():
  r = sample()
  M = matrix(r)
  assert M.shape == (2006, 3, 3)
  assert same(rotation.byMatrix(M), r)
  assert np.all(np.abs(M @ np.swapaxes(M, -1, -2) - np.eye(3)) < 1e-12)
  v = vector3d.rand(2006, rng=rng)
  assert close(np.einsum("nij,nj->ni", M, v.data), (r * v).data)
  with pytest.raises(ValueError):
    rotation.byMatrix(-M[0])
  o = orthogonalTransform.byMatrix(-M)
  assert np.all(o.isImproper) and close(o.matrix(), -M)
  back = orthogonalTransform.byMatrix(o.matrix())
  assert np.all(back.isImproper) and same(back, o)


def test_axis_angle():
  r = sample()
  assert same(rotation.byAxisAngle(r.axis(), r.angle()), r)
  assert same(rotation.byAxisAngle(vector3d(1, 2, 3), 40 * degree), rotation.byAxisAngle(vector3d(2, 4, 6), 40 * degree))
  assert same(rotation.byAxisAngle([0, 0, 1], pi / 2), Rz(pi / 2))
  assert rotation.byAxisAngle(zvector, [0.1, 0.2, 0.3]).shape == (3,)
  assert rotation.byAxisAngle(vector3d(0, 0, 0), 0) == rotation.id and rotation.byAxisAngle(vector3d(0, 0, 0), 2 * pi) == rotation.id
  with pytest.raises(ValueError):
    rotation.byAxisAngle(vector3d(0, 0, 0), 1.0)


def test_rodrigues():
  r = sample()
  keep = r.angle() < pi - 1e-3
  rr = r[keep]
  assert same(rotation.byRodrigues(Rodrigues(rr)), rr)
  assert np.sum(keep) > 0.9 * r.size
  assert np.all(Rodrigues(rr) == np.tan(rr.angle() / 2) * rr.axis())
  assert same(rotation.byRodrigues([0, 0, 1]), Rz(pi / 2))


def test_map():
  for _ in range(20):
    u, v = vector3d.rand(rng=rng), vector3d.rand(rng=rng)
    r = rotation.map(u, v)
    assert r * u == v and close(r.angle(), angle(u, v))
    u1, u2, s = vector3d.rand(rng=rng), vector3d.rand(rng=rng), rotation.rand(rng=rng)
    assert same(rotation.map(u1, s * u1, u2, s * u2), s)
  assert rotation.map(xvector, xvector) == rotation.id
  assert same(rotation.map(xvector, -xvector), rotation.byAxisAngle(yvector, pi))
  assert same(rotation.map(vector3d(1, 1, 1), zvector, vector3d(0, -1, 1), xvector) * vector3d(1, 1, 1), zvector)
  us, vs = vector3d.rand(4, rng=rng), vector3d.rand(4, rng=rng)
  assert rotation.map(us, vs).shape == (4,)
  with pytest.raises(ValueError):
    rotation.map(xvector, yvector, yvector, yvector)
  with pytest.raises(ValueError):
    rotation.map(xvector, yvector, xvector, zvector)


def test_fit():
  r = rotation.rand(rng=rng)
  u = vector3d.rand(50, rng=rng)
  v = r * u + 0.02 * vector3d.rand(50, rng=rng)
  rec = rotation.fit(u, v)
  assert angle(r, rec) < 2 * degree
  assert same(rotation.fit(u, r * u), r, 1e-12)
  assert angle(rotation.fit(u, v, weights=np.ones(50)), rec) < 1e-10
