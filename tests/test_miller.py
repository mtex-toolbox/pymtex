# crystal directions: index conventions, conversion, rounding, brackets (check_Miller
# without the symmetrise parts, which come with G5)
import numpy as np
import pytest
from numpy import pi

import mtex
from mtex import Miller, crystalFrame, specimenFrame, vector3d, xvector, degree, angle, dspacing, cross, isCrystalDirection, setMTEXpref, cat
from conftest import close

rng = np.random.default_rng(0)


def parallel(a, b):
  return bool(np.all(1 - np.abs(a.normalize().dot(b.normalize(), noAntipodal=True)) < 1e-10))


def test_four_index_constraint_and_round_trip():
  cs = crystalFrame("6/mmm", [3, 3, 5])
  csT = crystalFrame("-3m", [4.9, 4.9, 5.4])
  for fr in (cs, csT):
    h = Miller(np.round(5 * rng.normal(size=20)), np.round(5 * rng.normal(size=20)), np.round(5 * rng.normal(size=20)), fr)
    assert np.max(np.abs(h.h + h.k + h.i)) < 1e-10 and h.shape == (20,) and h.dispStyle == "hkil"
    d = Miller(np.round(5 * rng.normal(size=20)), np.round(5 * rng.normal(size=20)), np.round(5 * rng.normal(size=20)), fr, "uvw")
    assert np.max(np.abs(d.U + d.V + d.T)) < 1e-10 and d.dispStyle == "uvw"
  for _ in range(30):
    h = Miller(*np.round(4 * rng.normal(size=3)), cs)
    assert parallel(h, Miller(*h.hkil, cs)) and parallel(h, Miller(h.hkil, cs, "hkil"))
    d = Miller(*np.round(4 * rng.normal(size=3)), cs, "uvw")
    assert parallel(d, Miller(*d.UVTW, cs, "UVTW"))
  assert close(Miller(1, 2, 3, cs).hkl, [1, 2, 3]) and close(Miller(1, 2, 3, cs, "uvw").uvw, [1, 2, 3])
  with pytest.warns(UserWarning):
    Miller(1, 1, 1, 0, cs)


def test_round():
  cs = crystalFrame("6/mmm", [3, 3, 5])
  csC = crystalFrame("m-3m")
  h = Miller(1, 3, -4, 8, cs, "UVTW")
  assert close(h.uvw, np.array([5, 7, 8]) / 3)
  assert close(h.round().UVTW, [1, 3, -4, 8]) and h.round().dispStyle == "UVTW"
  hUvw = h.round("uvw")
  assert close(hUvw.uvw, [5, 7, 8]) and parallel(h, hUvw) and hUvw.dispStyle == "UVTW"
  assert close(hUvw.UVTW, np.round(hUvw.UVTW)) and not close(hUvw.UVTW, [1, 3, -4, 8])
  for style in ("hkl", "hkil", "uvw", "UVTW"):
    g = Miller(1, 3, -4, 8, cs, "UVTW")
    g.dispStyle = style
    assert g.round().dispStyle == style and g.round("uvw").dispStyle == style
  for fr in (cs, csC):
    hs = Miller(rng.normal(size=40), rng.normal(size=40), rng.normal(size=40), fr)
    hr = hs.round(maxHKL=12)
    assert np.max(angle(hs, hr)) < 5 * degree and np.max(np.abs(hr.hkl - np.round(hr.hkl))) < 1e-9
    assert mtex.round(hs).shape == (40,)
  for m in ([1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 0]):
    assert parallel(Miller(*m, csC), Miller(*m, csC, "uvw"))


def test_dspacing_cross_and_mixing():
  csC = crystalFrame("m-3m")
  al = crystalFrame("m-3m", [4.05, 4.05, 4.05])
  assert close(dspacing(Miller(1, 0, 0, csC)), 1) and close(dspacing(Miller(1, 0, 0, al)), 4.05)
  with pytest.raises(ValueError):
    dspacing(xvector)
  c = cross(Miller(1, 0, 0, csC), Miller(0, 1, 0, csC))
  assert c.dispStyle == "uvw" and parallel(c, Miller(0, 0, 1, csC, "uvw")) and c.frame is csC
  # plane normals and lattice directions are the same vectors: a mix is shown in the style of the first
  assert (Miller(1, 0, 0, csC) + Miller(0, 1, 0, csC, "uvw")).dispStyle == "hkl"
  s = Miller(1, 0, 0, csC) + Miller(0, 1, 0, csC)
  assert s.frame is csC and s.dispStyle == "hkl" and close(s.hkl, [1, 1, 0])
  assert isCrystalDirection(Miller(1, 0, 0, csC)) and not isCrystalDirection(xvector) and not isCrystalDirection(csC)
  v = xvector
  v.frame = specimenFrame.rolling
  assert not isCrystalDirection(v) and v.CS is None
  w = vector3d.X
  w.frame = csC
  assert isCrystalDirection(w) and w.dispStyle == "hkl"
  # Cartesian components restated in a frame keep their numbers
  assert close(Miller(vector3d(0.3, 0.2, 0.1), csC).data, [0.3, 0.2, 0.1])
  assert Miller(vector3d(0.3, 0.2, 0.1), csC, "xyz").dispStyle == "xyz" and close(Miller(0.3, 0.2, 0.1, csC, "xyz").data, [0.3, 0.2, 0.1])
  assert Miller(vector3d(0.3, 0.2, 0.1, frame=csC)).frame is csC
  with pytest.raises(ValueError):
    Miller(1, 0, 0)
  # a list of indices, one per row
  hs = Miller([[1, 0, 0], [1, 1, 0], [1, 1, 1]], csC)
  assert hs.shape == (3,) and close(hs.hkl, [[1, 0, 0], [1, 1, 0], [1, 1, 1]])
  assert Miller(antipodal=True, *[1, 0, 0], **{}) if False else Miller(1, 0, 0, csC, antipodal=True).frame is csC.Laue()
  assert Miller(1, 0, 0, crystalFrame("432"), antipodal=True).frame.pointGroup == "m-3m"


def test_brackets():
  csC = crystalFrame("m-3m")
  csH = crystalFrame("6/mmm", [3, 3, 5])
  cs1 = crystalFrame(csH.basis)
  assert str(Miller(1, 1, 1, csC)) == "{111}" and Miller(1, 1, 1, csC).char() == "{111}"
  assert str(Miller(1, 0, 0, csC, "uvw")) == "<100>"
  assert str(Miller(1, 1, 1, cs1)) == "(111)" and str(Miller(1, 0, 0, cs1, "uvw")) == "[100]"
  s = str(Miller(1, 0, -1, 0, csH))
  assert s.startswith("{") and s.endswith("}") and "̅" in s
  assert Miller(1, 0, -1, 0, csH).char(utf8=False) == "{10-10}"
  setMTEXpref("UTF8Output", False)
  assert str(Miller(1, 0, -1, 0, csH)) == "{10-10}"
  mtex.reset()
  assert str(Miller(1, 0, 0, csC, "uvw")) == "<100>"
  assert str(Miller(1, 2, 3, csC).round()) == "{123}" and str(Miller(12, 0, 0, csC)) == "{12 0 0}"
  assert str(Miller([[1, 0, 0], [1, 1, 0]], csC)) == "{100}{110}"
  assert Miller(1, 0, 0, csC).char("uvw") == "<100>"
  # both direct brackets mean uvw, both reciprocal ones hkl
  assert parallel(Miller("<100>", csC), Miller("[100]", csC)) and Miller("<100>", csC).dispStyle == "uvw"
  assert parallel(Miller("⟨100⟩", csC), Miller("<100>", csC))
  assert parallel(Miller("{110}", csC), Miller("(110)", csC)) and Miller("{110}", csC).dispStyle == "hkl"
  # the single forms carry the group-free sibling, the family forms the frame
  for s in ("(100)", "[100]"):
    assert Miller(s, csC).frame.numSym() == 1 and Miller(s, csC).frame is csC.stripSym()
  for s in ("{100}", "<100>", "⟨100⟩"):
    assert Miller(s, csC).frame is csC
  assert parallel(Miller("(110)", csC), Miller(1, 1, 0, csC))
  assert parallel(Miller("(1-10)", csC), Miller(1, -1, 0, csC)) and parallel(Miller("(1,-1,0)", csC), Miller(1, -1, 0, csC))
  assert Miller("{10-10}", csH).dispStyle == "hkil"
  with pytest.raises(ValueError):
    Miller("110", csC)
  assert repr(Miller(1, 1, 1, csC)).startswith("vector3d (m3̅m)\n  h  k  l\n  1  1  1")


def test_frame_membership_and_setters():
  cs = crystalFrame("321", [3, 3, 5], "X||a")
  cs2 = crystalFrame("321", [3, 3, 5], "Y||a")
  m = Miller(1, 0, 0, cs)
  assert m.frame is cs and m.how2plot == cs.how2plot and m.CS is cs and m.lattice is cs.lattice
  mIdx = Miller(1, 2, 3, cs)
  mIdx.CS = cs2
  assert mIdx.frame is cs2 and close(mIdx.hkl, [1, 2, 3])
  mCmp = Miller(1, 2, 3, cs)
  xyzBefore = mCmp.data.copy()
  mCmp.frame = cs2
  assert close(mCmp.data, xyzBefore) and mCmp.frame is cs2
  mSpec = Miller(1, 2, 3, cs)
  mSpec.frame = specimenFrame.rolling
  assert not isCrystalDirection(mSpec) and mSpec.CS is None and mSpec.dispStyle == "xyz"
  with pytest.raises(ValueError):
    mSpec.hkl
  v = vector3d(Miller(1, 0, 0, cs))
  assert v.frame is cs and isCrystalDirection(v)
  # the index setters
  h = Miller(1, 0, 0, cs)
  h.hkl = [0, 0, 1]
  assert parallel(h, cs.cAxisRec) and h.dispStyle == "hkil"
  h.uvw = [0, 0, 1]
  assert parallel(h, cs.cAxis) and h.dispStyle == "uvw"
  h.UVTW = [2, -1, -1, 0]
  assert close(h.uvw, [1, 0, 0]) and h.dispStyle == "UVTW"
  h.coordinates = [1, 0, -1, 0]
  assert close(h.uvw, np.array([1, 0, 0]) / 3 * 1) or close(h.UVTW, [1, 0, -1, 0])
  with pytest.raises(ValueError):
    h.uvw = [1, 0, 0, 0]
  # a plotting convention or a frame given positionally
  pC = mtex.plottingConvention("z↑→x")
  fv = vector3d(1, 2, 3, pC)
  assert fv.frame is specimenFrame.frameFor(pC) and fv.how2plot == pC
  assert vector3d(1, 2, 3, specimenFrame.rolling).frame is specimenFrame.rolling
  assert vector3d.rand(5, rng=rng).frame is None and vector3d.rand(5, rng=rng).how2plot == mtex.plottingConvention.default
  with pytest.raises(TypeError):
    vector3d(1, 2, 3).frame = 5


def test_join_rule():
  cs = crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Al")
  sF = specimenFrame.rolling
  m = vector3d(1, 0, 0, frame=cs)
  r = vector3d(0, 1, 0, frame=sF)
  assert (m + m).frame is cs and (2 * m).frame is cs and (-m).frame is cs and (m + vector3d(0, 0, 1)).frame is cs
  assert (vector3d(0, 0, 1) + m).frame is cs
  with pytest.raises(ValueError):
    m + r
  assert m.dot(vector3d(1, 1, 0)) == 1 and m.normalize().frame is cs and m.normalize().dispStyle == "hkl"
  L = crystalFrame("432", [4.05, 4.05, 4.05], mineral="Al")
  assert (vector3d(0, 1, 0, frame=L) + m).frame is L
  assert (mtex.rotation.byAxisAngle(mtex.zvector, 0.3) * m).frame is cs
  v = vector3d(rng.random((5, 3)), frame=cs)
  assert v[2].frame is cs and v[2:3].frame is cs and v.reshape(1, 5).frame is cs and v.copy().frame is cs
  assert cat(v, v).frame is cs and cat(v, vector3d(rng.random((2, 3)))).frame is cs and cat(vector3d(rng.random((2, 3))), v).frame is cs
  assert cat(v, vector3d(rng.random((2, 3)), frame=cs.stripSym())).frame is cs.stripSym()
  with pytest.raises(ValueError):
    cat(v, vector3d(rng.random((2, 3)), frame=sF))
  v[0] = vector3d(1, 2, 3)
  with pytest.raises(ValueError):
    v[0] = vector3d(1, 0, 0, frame=sF)
  assert v.mean().frame is cs and v.sum().frame is cs and v.perp().frame is cs and v.perp().dispStyle == "uvw"
  assert (v * 2).frame is cs and (v / 2).dispStyle == "hkl"
