# one test per statement of docs/design/semantics.md
import numpy as np
import pytest
from numpy import pi

import mtex
from conftest import close, need

pytestmark = pytest.mark.filterwarnings("error")


def rot():
  need("rotation", "vector3d", "degree", "inv", "matrix", "angle")
  return mtex


def test_S1_active():
  m = rot()
  r = m.rotation.byAxisAngle(m.vector3d.Z, 90 * m.degree)
  assert (r * m.vector3d.X).isclose(m.vector3d.Y)


def test_S2_right_to_left():
  m = rot()
  r1 = m.rotation.byAxisAngle(m.vector3d.Z, 30 * m.degree)
  r2 = m.rotation.byAxisAngle(m.vector3d.X, 70 * m.degree)
  v = m.vector3d(1, 2, 3)
  assert ((r1 * r2) * v).isclose(r1 * (r2 * v))
  assert (r1 * r2) * v != (r2 * r1) * v


def test_S3_inverse():
  m = rot()
  r = m.rotation.byEuler(0.3, 0.7, 1.1)
  v = m.vector3d(1, 2, 3)
  assert (m.inv(r) * (r * v)).isclose(v)
  assert close(m.inv(r).data, r.data * [1, -1, -1, -1])


def test_S4_matrix():
  m = rot()
  need("orthogonalTransform")
  r = m.rotation.byEuler(0.3, 0.7, 1.1)
  v = m.vector3d(1, 2, 3)
  M = m.matrix(r)
  assert close(M @ v.xyz, (r * v).xyz)
  assert close(M @ M.T, np.eye(3)) and close(np.linalg.det(M), 1)
  assert m.rotation.byMatrix(M).isclose(r)
  F = np.diag([1.0, 1.0, -1.0])
  with pytest.raises(ValueError):
    m.rotation.byMatrix(F)
  assert isinstance(m.orthogonalTransform.byMatrix(F), m.orthogonalTransform)


def test_S5_sign():
  m = rot()
  need("quaternion")
  r = m.rotation(0.5, 0.5, 0.5, 0.5)
  assert r.isclose(m.rotation(-0.5, -0.5, -0.5, -0.5))
  assert m.quaternion(0.5, 0.5, 0.5, 0.5) != m.quaternion(-0.5, -0.5, -0.5, -0.5)
  s = m.rotation(-0.5, -0.5, -0.5, -0.5)
  assert close((s * s).data, (r * r).data)
  assert close(m.angle(r), 2 * np.arccos(0.5)) and close(m.angle(s), 2 * np.arccos(0.5))
  eps = 1e-9
  tiny = m.rotation.byAxisAngle(m.vector3d.Z, eps)
  assert close(m.angle(tiny), eps, tol=1e-20)
  nearHalfTurn = m.rotation.byAxisAngle(m.vector3d.Z, np.pi - eps)
  assert close(m.angle(nearHalfTurn), np.pi - eps, tol=1e-15)


def test_S6_axis_angle():
  m = rot()
  n = m.vector3d(1, 2, 3)
  r = m.rotation.byAxisAngle(n, 200 * m.degree)
  assert close(r.angle(), 160 * m.degree) and close(m.angle(r), 160 * m.degree)
  assert r.axis().isclose(-m.normalize(n)) and m.axis(r).isclose(r.axis())
  assert r.axis().isNormalized
  assert m.rotation.byAxisAngle(r.axis(), r.angle()).isclose(r)
  assert m.rotation.id.axis().isclose(m.vector3d.X)
  assert (m.rotation.byAxisAngle(m.vector3d.Z, 90 * m.degree) * m.vector3d.X).isclose(m.vector3d.Y)
  # what MATLAB declares as a property is a property
  assert close(m.rotation.byEuler(0.3, 0.2, 0.1).phi1, 0.3) and n.isNormalized is False
  assert m.reflection(m.vector3d.Z).isImproper


def test_S7_rodrigues():
  m = rot()
  need("Rodrigues")
  r = m.rotation.byAxisAngle(m.vector3d(1, 2, 3), 100 * m.degree)
  rv = m.Rodrigues(r)
  assert rv.isclose(np.tan(50 * m.degree) * r.axis())
  assert m.rotation.byRodrigues(rv).isclose(r)


def test_S8_map():
  m = rot()
  u, v = m.vector3d(1, 0, 0), m.vector3d(1, 1, 0)
  r = m.rotation.map(u, v)
  assert close(m.angle(r * u, v), 0) and close(r.angle(), 45 * m.degree)
  u1, v1, u2, v2 = m.vector3d(1, 0, 0), m.vector3d(0, 1, 0), m.vector3d(0, 1, 0), m.vector3d(0, 0, 1)
  r = m.rotation.map(u1, v1, u2, v2)
  assert close(m.angle(r * u1, v1), 0) and close(m.angle(r * u2, v2), 0)
  with pytest.raises(ValueError):
    m.rotation.map(u1, v1, m.vector3d(1, 1, 0), v2)


def Rz(m, a):
  return m.rotation.byAxisAngle(m.vector3d.Z, a)


def Ry(m, a):
  return m.rotation.byAxisAngle(m.vector3d.Y, a)


def Rx(m, a):
  return m.rotation.byAxisAngle(m.vector3d.X, a)


A, B, C = 0.3, 0.7, 1.1


def test_S9_matthies():
  m = rot()
  r = m.rotation.byEuler(A, B, C, "ZYZ")
  assert r.isclose(Rz(m, A) * Ry(m, B) * Rz(m, C))
  assert r.isclose(m.rotation.byEuler(A, B, C, "Matthies"))
  q = [np.cos(B / 2) * np.cos((A + C) / 2), -np.sin(B / 2) * np.sin((A - C) / 2),
       np.sin(B / 2) * np.cos((A - C) / 2), np.cos(B / 2) * np.sin((A + C) / 2)]
  assert close(r.data, q)


def test_S10_bunge():
  m = rot()
  need("getMTEXpref")
  r = m.rotation.byEuler(A, B, C, "Bunge")
  assert r.isclose(Rz(m, A) * Rx(m, B) * Rz(m, C))
  assert r.isclose(m.rotation.byEuler(A - pi / 2, B, C + pi / 2, "ZYZ"))
  assert m.getMTEXpref("EulerAngleConvention") == "Bunge"
  assert r.isclose(m.rotation.byEuler(A, B, C))


def test_S11_roe():
  m = rot()
  assert m.rotation.byEuler(A, B, C, "Roe").isclose(m.rotation.byEuler(A, B, C, "ZYZ"))


def test_S12_kocks():
  m = rot()
  assert m.rotation.byEuler(A, B, C, "Kocks").isclose(m.rotation.byEuler(A, B, pi - C, "ZYZ"))


def test_S13_canova():
  m = rot()
  assert m.rotation.byEuler(A, B, C, "Canova").isclose(m.rotation.byEuler(pi / 2 - A, B, 3 * pi / 2 - C, "ZYZ"))


def test_S14_extraction():
  m = rot()
  need("Euler")
  rng = np.random.default_rng(0)
  a, b, c = rng.uniform(0, 2 * pi, 100), rng.uniform(0, pi, 100), rng.uniform(0, 2 * pi, 100)
  r = m.rotation.byEuler(a, b, c)
  e1, e2, e3 = m.Euler(r)
  assert np.all((0 <= e1) & (e1 < 2 * pi)) and np.all((0 <= e2) & (e2 <= pi)) and np.all((0 <= e3) & (e3 < 2 * pi))
  assert np.all(m.rotation.byEuler(e1, e2, e3).isclose(r))
  assert close(r.phi1, e1) and close(r.Phi, e2) and close(r.phi2, e3)
  z1, z2, z3 = m.Euler(m.rotation.byEuler(0.4, 0.0, 0.5))
  assert close([z1, z2, z3], [0.9, 0.0, 0.0])
  f1, f2, f3 = m.Euler(m.rotation.byEuler(A, B, C, "ZYZ"), "ZYZ")
  assert close([f1, f2, f3], [A, B, C])


def test_S15_degree():
  m = rot()
  assert close(30 * m.degree, 30 * pi / 180)
  assert close(m.rotation.byEuler(0, 30 * m.degree, 0).angle(), 30 * m.degree)


def test_S16_improper():
  m = rot()
  need("orthogonalTransform", "reflection")
  r = m.rotation.byEuler(A, B, C)
  v = m.vector3d(1, 2, 3)
  s = -r
  assert isinstance(s, m.orthogonalTransform) and s.isImproper
  assert s * v == -(r * v)
  assert m.rotation.inversion * v == -v
  n = m.vector3d(0, 0, 1)
  f = m.reflection(n)
  assert (f * m.vector3d(1, 2, 0)).isclose(m.vector3d(1, 2, 0)) and (f * n).isclose(-n)
  assert f.isclose(-m.rotation.byAxisAngle(n, pi))
  assert not (f * f).isImproper and not (f * m.rotation.inversion).isImproper and (f * r).isImproper and (r * f).isImproper
  assert close(np.linalg.det(m.matrix(f)), -1)
  assert not hasattr(m.rotation.id, "isImproper")


def test_S17_normalised_and_antipodal():
  m = rot()
  need("normalize")
  v = m.vector3d(1, 2, 3)
  assert not v.isNormalized and m.normalize(v).isNormalized
  a = m.vector3d(1, 2, 3, antipodal=True)
  b = m.vector3d(-1, -2, -3, antipodal=True)
  assert a == b and m.angle(a, b) == 0
  assert m.angle(a, m.vector3d(-1, 0, 0)) <= pi / 2
  assert m.angle(a, m.vector3d(-1, -2, -3)) == 0
  assert not (a + a).antipodal and not (m.normalize(v) + m.normalize(v)).isNormalized
  r = m.rotation.byEuler(A, B, C)
  assert (r * a).antipodal and (r * m.normalize(v)).isNormalized


def test_S18_polar():
  m = rot()
  v = m.vector3d.byPolar(0.4, 2.0)
  assert v.isNormalized and close(v.theta, 0.4) and close(v.rho, 2.0)
  assert close(m.vector3d.byPolar(0.4, -2.0).rho, 2 * pi - 2.0)
  assert close(m.vector3d.Z.theta, 0) and close(m.vector3d(0, 0, -1).theta, pi)
  assert close(m.vector3d.X.rho, 0) and close(m.vector3d.Y.rho, pi / 2)
  assert close(m.vector3d.byPolar(np.array([0.1, 0.2]), np.array([1.0, 2.0])).theta, [0.1, 0.2])


# ------------------------------------------------------------------ orientations and frames


def ori():
  need("orientation", "crystalFrame", "specimenFrame", "Miller", "rotate", "symmetrise", "transformReferenceFrame")
  return mtex


def test_S19_orientation_maps_crystal_to_specimen():
  m = ori()
  cs, ss = m.crystalFrame("m-3m", [4, 4, 4]), m.specimenFrame.specimen
  o = m.orientation.byEuler(A, B, C, cs, ss)
  h = m.Miller(1, 1, 0, cs)
  assert o.frameA is cs and o.frameB is ss and o.CS is cs and o.SS is ss
  assert m.inv(o).CS is ss
  v = o * h
  assert v.frame is ss and v == m.rotate(h, o)
  assert close(m.matrix(o) @ h.xyz, v.xyz)


def test_S20_left_specimen_right_crystal():
  m = ori()
  cs, ss = m.crystalFrame("m-3m", [4, 4, 4]), m.specimenFrame("222")
  o = m.orientation.byEuler(A, B, C, cs, ss)
  h = m.Miller(1, 1, 0, cs)
  r = m.rotation.byAxisAngle(m.vector3d(1, 2, 3), 0.5)
  assert ((r * o) * h).isclose(r * (o * h))
  assert (r * o).SS.numSym() == 1 and (r * o).SS.isAligned(ss)
  assert (o * r).CS.numSym() == 1 and (o * r).CS.isAligned(cs)
  assert (m.rotation.byAxisAngle(m.vector3d.Z, pi) * o).SS is ss


def test_S21_products_of_orientations():
  m = ori()
  cs, ss = m.crystalFrame("m-3m", [4, 4, 4]), m.specimenFrame.specimen
  o1 = m.orientation.byEuler(A, B, C, cs, ss)
  o2 = m.orientation.byEuler(C, B, A, cs, ss)
  mori = m.inv(o1) * o2
  assert mori.frameA is cs and mori.frameB is cs
  with pytest.raises(ValueError):
    o1 * o2
  # a compatible crystal frame, the same lattice set up against other axes, is absorbed
  ca = m.crystalFrame("2/m", [8.6, 13, 7.2], [90, 116, 90], alignment=["X||a"])
  cb = m.crystalFrame("2/m", [8.6, 13, 7.2], [90, 116, 90], alignment=["X||a*"])
  o = m.orientation.byEuler(A, B, C, ca, ss)
  mori = m.orientation.byEuler(C, B, A, cb, cb)
  p = o * mori
  h = m.Miller(1, 2, 3, cb)
  assert p.frameA is cb and p.frameB is ss and m.angle(p * h, o * m.transformReferenceFrame(mori * h, ca)) < 1e-12


def test_S22_symmetry_sides():
  m = ori()
  cs, ss = m.crystalFrame("m-3m", [4, 4, 4]), m.specimenFrame("222")
  o = m.orientation.byEuler(A, B, C, cs, ss)
  s = m.symmetrise(o)
  assert s.size == cs.numProper() * ss.numProper()
  assert np.all(m.angle(s, o) < 1e-12)
  sc, ssym = cs.properGroup().rotations()[3], ss.properGroup().rotations()[1]
  assert np.any(s == ssym * o * sc) and m.angle(ssym * o * sc, o) < 1e-12
  # the specimen symmetry acts on the specimen directions, the crystal symmetry on the crystal ones
  h = m.Miller(1, 2, 3, cs)
  assert close((o * sc * h).xyz, (o * (sc * h)).xyz) and close((ssym * o * h).xyz, (ssym * (o * h)).xyz)


def test_S23_misorientation():
  m = ori()
  cs, ss = m.crystalFrame("m-3m", [4, 4, 4]), m.specimenFrame.specimen
  o1 = m.orientation.byEuler(A, B, C, cs, ss)
  o2 = m.orientation.byEuler(C, B, A, cs, ss)
  mori = m.inv(o1) * o2
  assert close(m.angle(o1, o2), min(m.angle(m.symmetrise(mori), m.rotation.id)))
  mori.antipodal = True
  assert close(m.angle(mori), min(m.angle(m.inv(mori), m.rotation.id), m.angle(mori, m.rotation.id)))
  cs2 = m.crystalFrame("6/mmm", [3, 3, 5])
  with pytest.raises(ValueError):
    m.orientation.byEuler(A, B, C, cs, cs2, antipodal=True)


def test_S24_crystal_direction_reduces_over_its_group():
  m = ori()
  cs = m.crystalFrame("m-3m", [4, 4, 4])
  m1, m2 = m.Miller(1, 0, 0, cs), m.Miller(0, 1, 0, cs)
  assert m1.isclose(m2) and close(m.dot(m1, m2), m1.norm() ** 2) and close(m.angle(m1, m2), 0)
  assert close(m.dot(m1, m2, symmetry=False), 0)
  ss = m.specimenFrame("222")
  v1, v2 = m.vector3d(1, 0, 0), m.vector3d(0, 1, 0)
  v1.frame, v2.frame = ss, ss
  assert close(m.dot(v1, v2), 0)


def test_S25_frame_change_is_not_rotation():
  m = ori()
  cs = m.crystalFrame("2/m", [8.6, 13, 7.2], [90, 116, 90], alignment=["X||a"])
  cs2 = m.crystalFrame("2/m", [8.6, 13, 7.2], [90, 116, 90], alignment=["X||a*"])
  h = m.Miller(1, 2, 3, cs)
  g = m.transformReferenceFrame(h, cs2)
  assert g.frame is cs2 and close(g.hkl, h.hkl) and not close(g.xyz, h.xyz)
  r = m.rotation.byAxisAngle(m.vector3d.Z, 0.3)
  assert m.rotate(h, r).frame is cs and not close(m.rotate(h, r).hkl, h.hkl)
  o = m.orientation.byEuler(A, B, C, cs, m.specimenFrame.specimen)
  assert (o * h).frame is o.frameB


def test_S26_default_convention():
  m = ori()
  need("plottingConvention")
  assert m.specimenFrame.default is m.specimenFrame.specimen
  assert m.specimenFrame.default.how2plot == m.plottingConvention.ij
  assert m.plottingConvention.ij.east.isclose(m.vector3d.X)
  assert m.plottingConvention.ij.outOfScreen.isclose(-m.vector3d.Z)
  # a frame without a convention is drawn in the default of the moment it is drawn
  from mtex.plotting.spherical import _conventionOf
  fr = m.specimenFrame("222")
  fr.how2plot = None
  m.plottingConvention.default("y↑→x")
  assert _conventionOf(fr) == m.plottingConvention("y↑→x")
