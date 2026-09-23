# orientations and their container: construction, frames, indexing, inverse, exchange
import numpy as np
import pytest
from numpy import pi

import mtex
from mtex import orientation, rotation, quaternion, crystalFrame, specimenFrame, Miller, vector3d, xvector, zvector, degree, inv, Euler, angle, cat
from conftest import close


def same(p, q, tol=1e-13):
  return np.all(1 - np.abs(np.sum(p.data[..., :4] * q.data[..., :4], axis=-1)) < tol)


def test_orientation_scalars():
  cs = crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Al")
  ss = specimenFrame("222")
  o = orientation.byEuler(10 * degree, 20 * degree, 30 * degree, cs, ss)
  assert isinstance(o, orientation) and isinstance(o, rotation) and o.shape == ()
  assert o.CS is cs and o.SS is ss and o.frameA is cs and o.frameB is ss
  assert same(o, rotation.byEuler(10 * degree, 20 * degree, 30 * degree))
  assert close(Euler(o), [10 * degree, 20 * degree, 30 * degree]) and close(o.phi1, 10 * degree)
  # the default specimen frame is the session frame, the default crystal frame the trivial one
  assert orientation(rotation.id, cs).SS is specimenFrame.default and orientation(rotation.id).CS.numSym() == 1
  assert orientation(rotation.id).CS.pointGroup == "1" and isinstance(orientation(rotation.id).CS, crystalFrame)
  # inv swaps the frames
  i = inv(o)
  assert i.CS is ss and i.SS is cs and same(inv(i), o)
  # a misorientation is one between two crystal frames, and only one within a frame takes grain exchange
  cs2 = crystalFrame("6/mmm", [3, 3, 5], mineral="Ti")
  m = orientation(rotation.rand(), cs, cs2)
  assert m.isMisorientation and not o.isMisorientation and not m.antipodal
  assert orientation(rotation.rand(), cs, cs, antipodal=True).antipodal
  for frames in ((cs, ss), (cs, cs2), (cs, crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Al", noIntern=True))):
    with pytest.raises(ValueError):
      orientation(rotation.rand(), *frames, antipodal=True)
  # relabelling restates the frames without moving anything
  r = orientation(o, cs2)
  assert r.CS is cs2 and r.SS is ss and same(r, o)
  o.CS = cs2
  assert o.frameA is cs2
  o.SS = specimenFrame.rolling
  assert o.frameB is specimenFrame.rolling
  # the constructor forms
  assert same(orientation(1, 1, 1, 1, cs), rotation(0.5, 0.5, 0.5, 0.5)) and orientation(1, 1, 1, 1, cs).CS is cs
  assert orientation(quaternion(1, 2, 3, 4), cs, ss).SS is ss and close(orientation(quaternion(1, 2, 3, 4)).norm(), 1)
  assert orientation(cs, ss).isempty and orientation().isempty
  assert orientation(o, antipodal=False).CS is cs2
  with pytest.raises(TypeError):
    -o
  assert repr(o).startswith("orientation (Ti → rolling)") and "misorientation (Al → Ti)" in repr(m)


def test_constructors_with_frames():
  cs = crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Al")
  ss = specimenFrame("222")
  a, b, c = 0.3, 0.7, 1.1
  o = orientation.byEuler(a, b, c, "ZYZ", cs, ss)
  assert same(o, rotation.byEuler(a, b, c, "ZYZ")) and o.CS is cs and o.SS is ss
  assert same(orientation.byEuler(a, b, c, cs), rotation.byEuler(a, b, c)) and orientation.byEuler(a, b, c, cs).SS is specimenFrame.default
  assert orientation.byEuler(a, b, c, cs, cs, antipodal=True).antipodal
  assert orientation.byAxisAngle(zvector, 0.5, cs, ss).CS is cs and same(orientation.byAxisAngle(zvector, 0.5, cs), rotation.byAxisAngle(zvector, 0.5))
  M = rotation.byEuler(a, b, c).matrix()
  assert same(orientation.byMatrix(M, cs), rotation.byEuler(a, b, c)) and orientation.byMatrix(M, cs).CS is cs
  with pytest.raises(ValueError):
    orientation.byMatrix(-M, cs)
  assert orientation.byRodrigues(vector3d(0, 0, 1), cs).CS is cs and same(orientation.byRodrigues([0, 0, 1], cs), rotation.byRodrigues([0, 0, 1]))
  assert same(orientation.byHomochoric(vector3d(0.1, 0.2, 0.3), cs), rotation.byHomochoric([0.1, 0.2, 0.3]))
  # map adopts the frames of framed inputs: CS from the crystal direction, SS from the framed vector
  h = Miller(1, 0, 0, cs)
  v = vector3d(0, 1, 0, frame=specimenFrame.rolling)
  om = orientation.map(h, v)
  assert om.CS is cs and om.SS is specimenFrame.rolling and (om * h) == v
  assert orientation.map(h, xvector).SS is specimenFrame.default and orientation.map(xvector, xvector).CS.numSym() == 1
  om2 = orientation.map(Miller(1, 0, 0, cs), xvector, Miller(0, 1, 0, cs), vector3d(0, 1, 0), cs, ss)
  assert om2.CS is cs and om2.SS is ss and same(om2, rotation.id)
  assert orientation.map(xvector, vector3d(0, 1, 0), cs).CS is cs
  # rand, id, nan, fit, and the standard orientations
  assert orientation.rand(7, cs).shape == (7,) and orientation.rand(7, cs).CS is cs and orientation.rand(cs, ss).SS is ss
  assert orientation.rand(2, 3, cs).shape == (2, 3) and orientation.rand(cs).shape == ()
  assert orientation.id(cs).shape == () and orientation.id(3, cs).shape == (3,) and same(orientation.id(cs), rotation.id)
  assert np.all(np.isnan(orientation.nan(2, cs).data)) and orientation.nan(2, cs).CS is cs
  u = vector3d.rand(20, rng=np.random.default_rng(1))
  r = rotation.rand(rng=np.random.default_rng(2))
  assert angle(orientation.fit(u, r * u, cs), r) < 1e-10 and orientation.fit(u, r * u, cs).CS is cs
  assert same(orientation.cube(cs), rotation.id) and orientation.cube(cs).CS is cs
  assert close(orientation.goss(cs, ss).Phi, 45 * degree) and orientation.goss(cs, ss).SS is ss
  assert close(orientation.brass(cs).phi1, 35 * degree) and close(orientation.SR(cs).phi2, 63 * degree)


def test_orientation_arrays():
  cs = crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Al")
  rng = np.random.default_rng(1)
  rots = rotation.rand(10, rng=rng)
  os = orientation(rots, cs)
  assert os.shape == (10,) and os.CS is cs
  assert isinstance(os[3], orientation) and same(os[3], rots[3]) and os[3].CS is cs
  assert isinstance(os[2:4], orientation) and os[2:4].size == 2 and os[2:4].CS is cs
  os[0] = os[5]
  assert same(os[0], rots[5])
  os[1] = rots[2]
  assert same(os[1], rots[2])
  with pytest.raises(ValueError):
    os[0] = orientation(rots[0], specimenFrame.rolling, cs)
  # from representations
  e = orientation.byEuler(0.1 * np.arange(1, 6), 0.2, 0.3, cs)
  assert e.size == 5 and close(Euler(e)[0][1], 0.2)
  # inverse, concatenation, reshape
  assert inv(os).CS is specimenFrame.default and inv(os).SS is cs
  assert cat(os, os).size == 20 and cat(os, os).CS is cs
  with pytest.raises(ValueError):
    cat(os, orientation(rots, crystalFrame("6/mmm", [3, 3, 5])))
  assert cat(os, orientation(rots, cs.Laue())).CS is cs
  assert os.reshape(2, 5).shape == (2, 5) and same(os.reshape(2, 5)[1, 4], os[9])
  assert cat(os[0], os[1]).size == 2 and cat(os[0], os[1]).CS is cs
  assert cat(orientation(rots, cs, cs, antipodal=True), orientation(rots, cs, cs)).antipodal is False
  assert cat(orientation(rots, cs, cs, antipodal=True), orientation(rots, cs, cs, antipodal=True)).antipodal
  # the Euler angles of a list, and the display
  a = angle(os)
  assert a.shape == (10,) and close(a[2], os[2].angle())
  assert Euler(os)[0].shape == (10,)
  assert "size: 10" in repr(os) and repr(os).startswith("orientation (Al → specimen)")
  # float32
  assert os.astype(np.float32).CS is cs and os.astype(np.float32).dtype == np.float32
  assert orientation(rots.astype(np.float32), cs).dtype == np.float32


def test_products():
  cs = crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Al")
  ss = specimenFrame("222")
  o = orientation.byEuler(0.3, 0.7, 1.1, cs, ss)
  h = Miller(1, 1, 0, cs)
  v = o * h
  assert v.frame is ss and v.dispStyle == "xyz" and close(o.matrix() @ h.data, v.data)
  assert (rotation.byEuler(0.3, 0.7, 1.1) * h).frame is cs
  back = inv(o) * v
  assert back.frame is cs and back.dispStyle == "hkl" and close(back.hkl, [1, 1, 0])
  # a specimen direction in crystal coordinates; a frame-free vector passes
  assert (o * vector3d(1, 2, 3)).frame is ss
  with pytest.raises(ValueError):
    o * v                                          # a specimen direction on the crystal side
  with pytest.raises(ValueError):
    o * Miller(1, 0, 0, crystalFrame("6/mmm", [3, 3, 5]))
  # orientations compose when the inner frames fit, and the result names the outer ones
  o2 = orientation.byEuler(1.1, 0.7, 0.3, cs, ss)
  mori = inv(o) * o2
  assert mori.CS is cs and mori.SS is cs and mori.isMisorientation and same(mori, inv(rotation(o)) * rotation(o2))
  with pytest.raises(ValueError):
    o * o2
  assert (o * inv(o2)).CS is ss and (o * inv(o2)).SS is ss
  # a rotation on the left turns the specimen and drops its group unless it is an element
  r = rotation.byAxisAngle(vector3d(1, 2, 3), 0.5)
  assert (r * o).SS.numSym() == 1 and (r * o).SS.isAligned(ss) and (r * o).CS is cs and same(r * o, r * rotation(o))
  assert (rotation.byAxisAngle(zvector, pi) * o).SS is ss
  assert (o * r).CS.numSym() == 1 and (o * r).SS is ss and same(o * r, rotation(o) * r)
  assert (o * rotation.byAxisAngle(zvector, pi / 2)).CS is cs
  with pytest.raises(TypeError):
    rotation.inversion * o
  assert isinstance(2 * o, quaternion) and not isinstance(2 * o, orientation)
  # an aligned crystal frame fits, a compatible one is absorbed
  cs2 = crystalFrame("432", [4.05, 4.05, 4.05], mineral="Al")
  assert (o * Miller(1, 0, 0, cs2)).frame is ss
  csA = crystalFrame("121", [2, 3, 4], np.array([90, 105, 90]) * degree)
  csB = crystalFrame("112", [4, 2, 3], np.array([90, 90, 105]) * degree)
  oA = orientation.byEuler(0.3, 0.7, 1.1, csA, ss)
  hB = Miller(1, 0, 0, csB)
  assert (oA * hB).frame is ss
  assert (oA * hB) == (oA * Miller(vector3d(rotation.byMatrix(csB.transformationMatrix(csA)) * hB), csA))
  # lists broadcast
  os = orientation.rand(5, cs, ss, rng=np.random.default_rng(0))
  hs = Miller([[1, 0, 0], [1, 1, 0]], cs)
  assert (os[:, None] * hs[None, :]).shape == (5, 2)
  assert (inv(os[:, None]) * os[None, :]).shape == (5, 5)


def test_project2FundamentalRegion_antipodal():
  # MATLAB's 'antipodal': the projection with grain exchange, the misorientation or its inverse
  cs = crystalFrame("m-3m")
  m = orientation.rand(50, cs, cs, rng=np.random.default_rng(1))
  p = m.project2FundamentalRegion(antipodal=True)
  q = orientation._fromRotation(m.data, (cs, cs), True).project2FundamentalRegion()
  assert same(p, q) and not m.antipodal
  # each projected one is the misorientation or its inverse up to symmetry
  assert np.all(np.minimum(angle(p, m), angle(p, inv(m))) < 1e-6)
