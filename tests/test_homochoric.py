# port of check_homochoric, plus the cubochoric round trip
import numpy as np
from numpy import pi

from mtex import rotation, vector3d, zvector, homochoric, cubochoric, homochoricRadius, angle
from conftest import close

rng = np.random.default_rng(0)


def qdist(p, q):
  return np.minimum(np.linalg.norm(p.data - q.data, axis=-1), np.linalg.norm(p.data + q.data, axis=-1))


def test_round_trips():
  R = homochoricRadius()
  rot = rotation.rand(20000, rng=rng)
  assert np.max(qdist(rot, rotation.byHomochoric(homochoric(rot)))) < 1e-13
  v = vector3d.rand(20000, rng=rng) * (R * rng.random(20000) ** (1 / 3))
  assert np.max((v - homochoric(rotation.byHomochoric(v))).norm()) < 1e-13


def test_identity_and_surface():
  R = homochoricRadius()
  id = rotation.byHomochoric(vector3d(0, 0, 0))
  assert not np.isnan(id.a) and qdist(id, rotation.id) < 1e-15
  assert np.max(np.abs(angle(rotation.byHomochoric(R * vector3d.rand(1000, rng=rng))) - pi)) < 1e-12


def test_small_angles():
  om = 10.0 ** (-np.linspace(7, 1, 4000))
  w2 = om ** 2
  rho = (0.75 * om ** 3 / 6 * (1 - w2 / 20 * (1 - w2 / 42 * (1 - w2 / 72)))) ** (1 / 3)
  rec = rotation.byHomochoric(vector3d(1, 2, 3).normalize() * rho)
  omRec = 2 * np.arcsin(np.minimum(np.linalg.norm(rec.data[:, 1:], axis=-1), 1))
  assert np.max(np.abs(omRec - om) / om) < 1e-9


def test_monotone_and_shapes():
  omg = np.linspace(0, pi, 2000)
  assert np.all(np.diff(homochoric(rotation.byAxisAngle(zvector, omg)).norm()) > 0)
  assert rotation.byHomochoric(vector3d.rand(4, 5, rng=rng)).shape == (4, 5)
  assert qdist(rotation.byHomochoric([0.1, 0.2, 0.3]), rotation.byHomochoric(vector3d(0.1, 0.2, 0.3))) < 1e-15
  assert rotation.byHomochoric(vector3d.rand(0, rng=rng)).isempty


def test_cubochoric():
  a = pi ** (2 / 3) / 2
  rot = rotation.rand(2000, rng=rng)
  c = cubochoric(rot)
  assert np.all(np.abs(c.data) <= a + 1e-12)
  assert np.max(qdist(rotation.byCubochoric(c), rot)) < 1e-12
  assert close(cubochoric(rotation.id).data, [0, 0, 0])
  assert close(rotation.byCubochoric(vector3d(a, 0.3, -0.2)).homochoric().norm(), homochoricRadius(), tol=1e-12)
