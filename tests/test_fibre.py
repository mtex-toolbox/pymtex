# fibres: the Julia RotationFibre page, the MATLAB @fibre methods, and the fit
import numpy as np
import pytest
from numpy import pi

from mtex import (fibre, orientation, rotation, crystalFrame, specimenFrame, vector3d, Miller, zvector, degree,
                  angle, symmetrise, multiplicity, project2FundamentalRegion, cat)
from conftest import close

rng = np.random.default_rng(0)


def test_segment_between_two_components():
  cs = crystalFrame("432")
  oA = orientation.goss(cs)
  oB = project2FundamentalRegion(orientation.brass(cs), oA)
  f = fibre(oA, oB)
  assert f.CS is cs and f.SS is oA.SS and f.h.frame is cs and f.r.frame is oA.SS and f.shape == ()
  assert f.h.round() == Miller(0, 1, 1, cs, "uvw") and f.r == zvector
  # every orientation on the fibre maps h to r, the ends exactly
  assert angle(oA * f.h.round(), f.r) < 1e-10 and angle(oB * f.h.round(), f.r) < 1e-10
  assert not f.isFull and close(f.arcAngle(), angle(oA, oB)) and close(f.arcAngle(), 35 * degree)
  os = f.orientation(50)
  assert os.shape == (50,) and os.CS is cs and angle(os[0], oA) < 1e-10 and angle(os[-1], oB) < 1e-10
  assert np.max(angle(os * f.h, f.r)) < 1e-10
  # the full fibre keeps the pair and drops the ends
  full = fibre(f.h, f.r)
  assert full.isFull and close(full.arcAngle(), 2 * pi) and full.CS is cs and full.SS is f.SS
  assert fibre(oA, oB, full=True).isFull and fibre(oA, f.h).isFull and fibre(oA, f.h) == full
  os = full.orientation(400)
  assert np.max(angle(os * f.h, f.r)) < 1e-10 and angle(os[0], oA) < 1e-10 and angle(os[-1], oA) < 1e-10
  assert type(full.rotation(10)) is rotation
  # the ends have to share their frames
  with pytest.raises(ValueError):
    fibre(oA, orientation(oB, crystalFrame("m-3m")))
  with pytest.raises(ValueError):
    fibre(oA, Miller(1, 0, 0, crystalFrame("622"), "uvw"))
  # two coinciding ends turn about the c axis
  assert fibre(oA, oA).h == Miller(0, 0, 1, cs, "uvw")


def test_distance_to_a_fibre():
  cs = crystalFrame("432")
  oA = orientation.goss(cs)
  full = fibre(oA, project2FundamentalRegion(orientation.brass(cs), oA), full=True)
  assert angle(full, oA) < 1e-10 and angle(full, orientation.brass(cs)) < 1e-10
  # <011> to z is 45 degrees for the cube orientation, however it is written
  assert close(angle(full, orientation.cube(cs)), 45 * degree) and close(angle(orientation.cube(cs), full), 45 * degree)
  assert close(angle(full, orientation(cs.properGroup().rotations()[5], cs)), 45 * degree)
  assert angle(full, orientation.cube(cs), noSymmetry=True) > 45 * degree - 1e-10
  # the distance is symmetric under both groups and vanishes along the fibre
  ss = specimenFrame("222")
  f = fibre.rand(cs, ss, rng=rng)
  ori = orientation.rand(20, cs, ss, rng=rng)
  d = angle(f, ori)
  assert d.shape == (20,)
  for s in ss.properGroup().rotations():
    assert np.max(np.abs(angle(f, orientation(s * rotation(ori), cs, ss)) - d)) < 1e-10
  for g in cs.properGroup().rotations():
    assert np.max(np.abs(angle(f, orientation(rotation(ori) * g, cs, ss)) - d)) < 1e-10
  assert np.max(angle(f, f.orientation(100))) < 1e-10
  # a small step off the fibre reads as a small angle, to full precision
  step = rotation.byAxisAngle(f.r.perp(), 1e-6)
  assert close(angle(f, orientation(step * rotation(f.o1), cs, ss)), 1e-6, tol=1e-12)
  # fibre to fibre
  assert angle(f, f) < 1e-8 and angle(fibre.alpha(cs), fibre.gamma(cs)) > 5 * degree
  with pytest.raises(ValueError):
    angle(fibre.rand(3, cs, rng=rng), ori)


def test_named_fibres():
  cs = crystalFrame("432")
  a = fibre.alpha(cs)
  # the geodesic between MATLAB's ends, (001)[110] at (315, 0, 0) and (111)[110] at (180, 54.74, 45)
  assert not a.isFull and close(a.arcAngle(), 102.2 * degree, tol=1e-3 * degree)
  assert close(angle(a.o1, a.o2), 54.7356 * degree, tol=1e-4 * degree)
  assert np.max(np.abs(a.h.data.reshape(3) - [0, -0.5907, -0.8069])) < 1e-3 and a.r.antipodal is False
  for name in ("alpha", "beta", "gamma", "epsilon", "eta", "tau"):
    f = getattr(fibre, name)(cs, specimenFrame("222"))
    assert f.CS is cs and f.SS is specimenFrame("222") and not f.isFull
    assert getattr(fibre, name)(cs, full=True).isFull
    assert angle(f, f.o1) < 1e-10 and angle(f, f.o2) < 1e-10
  th = fibre.theta(cs)
  assert th.isFull and th.h == Miller(1, 0, 0, cs, "uvw") and th.r == zvector
  assert fibre.theta().CS.pointGroup == cs.pointGroup


def test_symmetrise_and_equality():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  th = fibre.theta(cs, ss)
  all_ = symmetrise(th)
  assert all_.shape == (24 * 4,) and np.all(all_ == th)
  # <100> has six members, z two under 222, and (h, r) is the same fibre as (-h, -r)
  u = symmetrise(th, unique=True)
  assert u.size == 6 and multiplicity(th) == 6 and np.all(u == th)
  # a generic direction has no antipode in its orbit, so nothing is identified
  assert multiplicity(fibre.theta(cs)) == 6 and multiplicity(fibre.rand(cs, ss, rng=rng)) == 24 * 4
  assert multiplicity(fibre.rand(crystalFrame("1"), rng=rng)) == 1
  # equality is up to the symmetry of the frames; a segment is not the full fibre
  a = fibre.alpha(cs)
  assert a == a and a != fibre.alpha(cs, full=True) and a != fibre.gamma(cs)
  assert not (a == fibre.alpha(cs, ss))
  assert fibre(th.h, -th.r) == th and fibre(-th.h, -th.r) == th
  with pytest.raises(ValueError):
    symmetrise(fibre.rand(2, cs, rng=rng), unique=True)


def test_lists():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  f = fibre.rand(3, cs, ss, rng=rng)
  assert f.shape == (3,) and f.size == 3 and len(f) == 3 and f.CS is cs and f.SS is ss
  assert f[1].shape == () and f[1] == f[1] and f[1:].shape == (2,)
  assert all(g.shape == () for g in f)
  assert cat(f, f[0]).shape == (4,) and f.reshape(3, 1).shape == (3, 1)
  assert f.orientation(10).shape == (10, 3) and np.all(f.isFull) and f.arcAngle().shape == (3,)
  ori = f.orientation(10)
  assert np.max(angle(ori * f.h, f.r)) < 1e-10
  # ends and a direction broadcast to one shape
  o = orientation.rand(4, cs, rng=rng)
  g = fibre(o, Miller(1, 1, 1, cs, "uvw"))
  assert g.shape == (4,) and g.h.shape == (4,) and np.all(g.isFull)
  assert fibre.rand(2, 3, cs, rng=rng).shape == (2, 3)


def test_fit_recovers_a_fibre():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  f = fibre.rand(cs, ss, rng=rng)
  # the representatives as sampled lie on one circle, which the raw tensor sees
  ori = orientation(f.orientation(500) * rotation.byAxisAngle(vector3d.rand(500, rng=rng), 2 * degree), cs, ss)
  fRec, lam, delta = fibre.fit(ori, diagnostics=True, noSymmetry=True)
  assert fRec.isFull and fRec.CS is cs and fRec.SS is ss and angle(fRec, f) < 1 * degree and delta < 2 * degree
  assert lam.shape == (4,) and close(np.sum(lam), 1) and lam[-2] > 100 * lam[-3]
  # the fit fixes the sign of (h, r) by r in the upper hemisphere, whatever the sample
  for k in range(4):
    sub = ori[rng.permutation(500)[:200]]
    r = fibre.fit(sub, noSymmetry=True).r.data.reshape(3)
    assert r[2] > 0 or (abs(r[2]) < 1e-9 and r[1] > 0)
  # a concentrated cloud scattered over the crystal group is gathered at its mean first;
  # with a specimen symmetry on top the fundamental region is too small for a 30 degree
  # arc to count as concentrated, so the cloud has none, as measured orientations do
  f = fibre.rand(cs, rng=rng)
  seg = fibre(f.o1, f.orientation(13)[1])
  ori = orientation(seg.orientation(500) * rotation.byAxisAngle(vector3d.rand(500, rng=rng), 2 * degree), cs)
  ori = orientation(rotation(ori) * cs.properGroup().rotations()[rng.integers(24, size=500)], cs)
  fRec, lam, delta = fibre.fit(ori, diagnostics=True)
  assert angle(fRec, f) < 2 * degree and abs(delta - np.mean(angle(f, ori))) < 0.2 * degree
  assert type(fibre.fit(ori)) is fibre and fibre.fit(ori) == fRec
  # NaN orientations are dropped
  o = orientation.cat(ori, orientation.nan(cs))
  assert angle(fibre.fit(o), fRec) < 1e-6


@pytest.mark.parametrize("signs", [(1, 1), (-1, 1), (1, -1), (-1, -1)])
def test_fit_hemisphere_does_not_depend_on_eigenvector_signs(monkeypatch, signs):
  cs = crystalFrame("1")
  f = fibre(Miller(1, 2, 3, cs), vector3d(0.2, 0.3, -1).normalize())
  ori = f.orientation(100)
  eigh = np.linalg.eigh

  def signed(A):
    vals, vecs = eigh(A)
    vecs[:, -2:] *= signs
    return vals, vecs

  monkeypatch.setattr(np.linalg, "eigh", signed)
  fitted = fibre.fit(ori, noSymmetry=True)
  assert fitted.r.z > 0 and angle(fitted, f) < 1e-7


def test_display():
  cs = crystalFrame("432")
  s = repr(fibre.alpha(cs))
  assert s.startswith("fibre (432 → y↓→x)") and "h || r :" in s and "o1 → o2:" in s
  assert "o1 → o2" not in repr(fibre.theta(cs)) and "size: 3" in repr(fibre.rand(3, cs, rng=rng))
  assert "[100] || (0,0,1)" in repr(fibre.theta(cs))
