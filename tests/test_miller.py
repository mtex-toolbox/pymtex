# crystal directions: index conventions, conversion, rounding, brackets (check_Miller
# without the symmetrise parts, which come with G5)
import numpy as np
import pytest
from numpy import pi

import mtex
from mtex import Miller, crystalFrame, specimenFrame, vector3d, xvector, degree, angle, dspacing, cross, isCrystalDirection, setMTEXpref, cat, symmetrise, unique
from conftest import close

hexFrames = [crystalFrame("6/mmm", [3, 3, 5]), crystalFrame("-3m", [4.9, 4.9, 5.4])]


def parallel(a, b):
  return bool(np.all(1 - np.abs(a.normalize().dot(b.normalize(), noAntipodal=True)) < 1e-10))


def hexagonal():
  return crystalFrame("6/mmm", [3, 3, 5])


def cubic():
  return crystalFrame("m-3m")


def trigonal():
  return crystalFrame("321", [3, 3, 5], "X||a")


def aluminium():
  return crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Al")


@pytest.mark.parametrize("fr", hexFrames, ids=["6/mmm", "-3m"])
def test_Miller_fourIndicesSumToZero(fr):
  rng = np.random.default_rng(0)
  h = Miller(np.round(5 * rng.normal(size=20)), np.round(5 * rng.normal(size=20)), np.round(5 * rng.normal(size=20)), fr)
  assert np.max(np.abs(h.h + h.k + h.i)) < 1e-10 and h.shape == (20,) and h.dispStyle == "hkil"
  d = Miller(np.round(5 * rng.normal(size=20)), np.round(5 * rng.normal(size=20)), np.round(5 * rng.normal(size=20)), fr, "uvw")
  assert np.max(np.abs(d.U + d.V + d.T)) < 1e-10 and d.dispStyle == "uvw"


def test_Miller_fourIndexRoundTrip():
  rng = np.random.default_rng(0)
  cs = hexagonal()
  for _ in range(30):
    h = Miller(*np.round(4 * rng.normal(size=3)), cs)
    assert parallel(h, Miller(*h.hkil, cs)) and parallel(h, Miller(h.hkil, cs, "hkil"))
    d = Miller(*np.round(4 * rng.normal(size=3)), cs, "uvw")
    assert parallel(d, Miller(*d.UVTW, cs, "UVTW"))


def test_Miller_threeIndicesReadBack():
  cs = hexagonal()
  assert close(Miller(1, 2, 3, cs).hkl, [1, 2, 3]) and close(Miller(1, 2, 3, cs, "uvw").uvw, [1, 2, 3])


def test_Miller_warnsOfFourIndicesNotSummingToZero():
  with pytest.warns(UserWarning):
    Miller(1, 1, 1, 0, hexagonal())


def test_round_restoresTheIntegerIndicesInTheirStyle():
  h = Miller(1, 3, -4, 8, hexagonal(), "UVTW")
  assert close(h.uvw, np.array([5, 7, 8]) / 3)
  assert close(h.round().UVTW, [1, 3, -4, 8]) and h.round().dispStyle == "UVTW"


def test_round_writesZeroNeverMinusZero():
  # a component rounded to zero is written 0, never -0
  assert str(vector3d(-1e-9, 0.0, 1).round()) == "(0,0,1)"


def test_round_inAnotherIndexStyle():
  h = Miller(1, 3, -4, 8, hexagonal(), "UVTW")
  hUvw = h.round("uvw")
  assert close(hUvw.uvw, [5, 7, 8]) and parallel(h, hUvw) and hUvw.dispStyle == "UVTW"
  assert close(hUvw.UVTW, np.round(hUvw.UVTW)) and not close(hUvw.UVTW, [1, 3, -4, 8])


@pytest.mark.parametrize("style", ["hkl", "hkil", "uvw", "UVTW"])
def test_round_keepsTheDisplayStyle(style):
  g = Miller(1, 3, -4, 8, hexagonal(), "UVTW")
  g.dispStyle = style
  assert g.round().dispStyle == style and g.round("uvw").dispStyle == style


@pytest.mark.parametrize("fr", [hexagonal(), cubic()], ids=["6/mmm", "m-3m"])
def test_round_maxHKLGivesNearbyIntegerIndices(fr):
  rng = np.random.default_rng(0)
  hs = Miller(rng.normal(size=40), rng.normal(size=40), rng.normal(size=40), fr)
  hr = hs.round(maxHKL=12)
  assert np.max(angle(hs, hr)) < 5 * degree and np.max(np.abs(hr.hkl - np.round(hr.hkl))) < 1e-9
  assert mtex.round(hs).shape == (40,)


@pytest.mark.parametrize("m", [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 0]])
def test_Miller_cubicPlaneNormalIsTheDirectionOfTheSameIndices(m):
  csC = cubic()
  assert parallel(Miller(*m, csC), Miller(*m, csC, "uvw"))


def test_dspacing_isTheLatticePlaneDistance():
  assert close(dspacing(Miller(1, 0, 0, cubic())), 1) and close(dspacing(Miller(1, 0, 0, crystalFrame("m-3m", [4.05, 4.05, 4.05]))), 4.05)


def test_dspacing_refusesAVectorWithoutCrystalFrame():
  with pytest.raises(ValueError):
    dspacing(xvector)


def test_cross_ofTwoPlaneNormalsIsADirection():
  csC = cubic()
  c = cross(Miller(1, 0, 0, csC), Miller(0, 1, 0, csC))
  assert c.dispStyle == "uvw" and parallel(c, Miller(0, 0, 1, csC, "uvw")) and c.frame is csC


def test_Miller_sumIsShownInTheStyleOfTheFirst():
  # plane normals and lattice directions are the same vectors: a mix is shown in the style of the first
  csC = cubic()
  assert (Miller(1, 0, 0, csC) + Miller(0, 1, 0, csC, "uvw")).dispStyle == "hkl"
  s = Miller(1, 0, 0, csC) + Miller(0, 1, 0, csC)
  assert s.frame is csC and s.dispStyle == "hkl" and close(s.hkl, [1, 1, 0])


def test_isCrystalDirection_isAVectorInACrystalFrame():
  csC = cubic()
  assert isCrystalDirection(Miller(1, 0, 0, csC)) and not isCrystalDirection(xvector) and not isCrystalDirection(csC)
  v = xvector
  v.frame = specimenFrame.rolling
  assert not isCrystalDirection(v) and v.CS is None
  w = vector3d.X
  w.frame = csC
  assert isCrystalDirection(w) and w.dispStyle == "hkl"


def test_Miller_ofCartesianComponentsKeepsTheNumbers():
  # Cartesian components restated in a frame keep their numbers
  csC = cubic()
  assert close(Miller(vector3d(0.3, 0.2, 0.1), csC).data, [0.3, 0.2, 0.1])
  assert Miller(vector3d(0.3, 0.2, 0.1), csC, "xyz").dispStyle == "xyz" and close(Miller(0.3, 0.2, 0.1, csC, "xyz").data, [0.3, 0.2, 0.1])
  assert Miller(vector3d(0.3, 0.2, 0.1, frame=csC)).frame is csC


def test_Miller_needsACrystalFrame():
  with pytest.raises(ValueError):
    Miller(1, 0, 0)


def test_Miller_listOfIndicesOnePerRow():
  csC = cubic()
  hs = Miller([[1, 0, 0], [1, 1, 0], [1, 1, 1]], csC)
  assert hs.shape == (3,) and close(hs.hkl, [[1, 0, 0], [1, 1, 0], [1, 1, 1]])


def test_Miller_antipodalTakesTheLaueFrame():
  csC = cubic()
  assert Miller(1, 0, 0, csC, antipodal=True).frame is csC.Laue()
  assert Miller(1, 0, 0, crystalFrame("432"), antipodal=True).frame.pointGroup == "m-3m"


def test_str_familiesInBracesAndAngleBrackets():
  csC = cubic()
  assert str(Miller(1, 1, 1, csC)) == "{111}" and Miller(1, 1, 1, csC).char() == "{111}"
  assert str(Miller(1, 0, 0, csC, "uvw")) == "<100>"


def test_str_withoutSymmetryInParenthesesAndSquareBrackets():
  cs1 = crystalFrame(hexagonal().basis)
  assert str(Miller(1, 1, 1, cs1)) == "(111)" and str(Miller(1, 0, 0, cs1, "uvw")) == "[100]"


def test_str_negativeIndexOverlined():
  s = str(Miller(1, 0, -1, 0, hexagonal()))
  assert s.startswith("{") and s.endswith("}") and "̅" in s


def test_char_withoutUTF8WritesAMinus():
  assert Miller(1, 0, -1, 0, hexagonal()).char(utf8=False) == "{10-10}"


def test_str_followsTheUTF8Preference():
  setMTEXpref("UTF8Output", False)
  assert str(Miller(1, 0, -1, 0, hexagonal())) == "{10-10}"
  mtex.reset()
  assert str(Miller(1, 0, 0, cubic(), "uvw")) == "<100>"


def test_str_ofRoundedLargeAndListedIndices():
  csC = cubic()
  assert str(Miller(1, 2, 3, csC).round()) == "{123}" and str(Miller(12, 0, 0, csC)) == "{12 0 0}"
  assert str(Miller([[1, 0, 0], [1, 1, 0]], csC)) == "{100}{110}"


def test_char_inAGivenStyle():
  assert Miller(1, 0, 0, cubic()).char("uvw") == "<100>"


def test_Miller_directBracketsMeanUvwReciprocalOnesHkl():
  # both direct brackets mean uvw, both reciprocal ones hkl
  csC = cubic()
  assert parallel(Miller("<100>", csC), Miller("[100]", csC)) and Miller("<100>", csC).dispStyle == "uvw"
  assert parallel(Miller("⟨100⟩", csC), Miller("<100>", csC))
  assert parallel(Miller("{110}", csC), Miller("(110)", csC)) and Miller("{110}", csC).dispStyle == "hkl"


@pytest.mark.parametrize("s", ["(100)", "[100]"])
def test_Miller_singleBracketFormCarriesTheGroupFreeSibling(s):
  csC = cubic()
  assert Miller(s, csC).frame.numSym() == 1 and Miller(s, csC).frame is csC.stripSym()


@pytest.mark.parametrize("s", ["{100}", "<100>", "⟨100⟩"])
def test_Miller_familyBracketFormCarriesTheFrame(s):
  csC = cubic()
  assert Miller(s, csC).frame is csC


def test_Miller_bracketStringGivesTheIndices():
  csC = cubic()
  assert parallel(Miller("(110)", csC), Miller(1, 1, 0, csC))
  assert parallel(Miller("(1-10)", csC), Miller(1, -1, 0, csC)) and parallel(Miller("(1,-1,0)", csC), Miller(1, -1, 0, csC))
  assert Miller("{10-10}", hexagonal()).dispStyle == "hkil"


def test_Miller_refusesAStringWithoutBrackets():
  with pytest.raises(ValueError):
    Miller("110", cubic())


def test_repr_listsTheIndices():
  assert repr(Miller(1, 1, 1, cubic())).startswith("vector3d (m3̅m)\n  h  k  l\n  1  1  1")


def test_Miller_carriesItsFrame():
  cs = trigonal()
  m = Miller(1, 0, 0, cs)
  assert m.frame is cs and m.how2plot == cs.how2plot and m.CS is cs and m.lattice is cs.lattice


def test_Miller_settingCSKeepsTheIndices():
  cs2 = crystalFrame("321", [3, 3, 5], "Y||a")
  mIdx = Miller(1, 2, 3, trigonal())
  mIdx.CS = cs2
  assert mIdx.frame is cs2 and close(mIdx.hkl, [1, 2, 3])


def test_Miller_settingTheFrameKeepsTheComponents():
  cs2 = crystalFrame("321", [3, 3, 5], "Y||a")
  mCmp = Miller(1, 2, 3, trigonal())
  xyzBefore = mCmp.data.copy()
  mCmp.frame = cs2
  assert close(mCmp.data, xyzBefore) and mCmp.frame is cs2


def test_Miller_settingASpecimenFrameDropsTheIndices():
  mSpec = Miller(1, 2, 3, trigonal())
  mSpec.frame = specimenFrame.rolling
  assert not isCrystalDirection(mSpec) and mSpec.CS is None and mSpec.dispStyle == "xyz"
  with pytest.raises(ValueError):
    mSpec.hkl


def test_vector3d_ofAMillerKeepsTheFrame():
  cs = trigonal()
  v = vector3d(Miller(1, 0, 0, cs))
  assert v.frame is cs and isCrystalDirection(v)


def test_Miller_indexSettersSetTheStyle():
  cs = trigonal()
  h = Miller(1, 0, 0, cs)
  h.hkl = [0, 0, 1]
  assert parallel(h, cs.cAxisRec) and h.dispStyle == "hkil"
  h.uvw = [0, 0, 1]
  assert parallel(h, cs.cAxis) and h.dispStyle == "uvw"
  h.UVTW = [2, -1, -1, 0]
  assert close(h.uvw, [1, 0, 0]) and h.dispStyle == "UVTW"
  h.coordinates = [1, 0, -1, 0]
  assert close(h.UVTW, [1, 0, -1, 0])


def test_Miller_uvwSetterRefusesFourIndices():
  h = Miller(1, 0, 0, trigonal())
  with pytest.raises(ValueError):
    h.uvw = [1, 0, 0, 0]


def test_vector3d_plottingConventionGivenPositionally():
  pC = mtex.plottingConvention("z↑→x")
  fv = vector3d(1, 2, 3, pC)
  assert fv.frame is specimenFrame.frameFor(pC) and fv.how2plot == pC


def test_vector3d_frameGivenPositionally():
  assert vector3d(1, 2, 3, specimenFrame.rolling).frame is specimenFrame.rolling


def test_vector3d_randHasNoFrameAndTheDefaultConvention():
  rng = np.random.default_rng(0)
  assert vector3d.rand(5, rng=rng).frame is None and vector3d.rand(5, rng=rng).how2plot == mtex.plottingConvention.default


def test_vector3d_frameSetterRefusesANonFrame():
  with pytest.raises(TypeError):
    vector3d(1, 2, 3).frame = 5


def test_vector3d_arithmeticKeepsTheFrame():
  cs = aluminium()
  m = vector3d(1, 0, 0, frame=cs)
  assert (m + m).frame is cs and (2 * m).frame is cs and (-m).frame is cs and (m + vector3d(0, 0, 1)).frame is cs
  assert (vector3d(0, 0, 1) + m).frame is cs


def test_vector3d_sumRefusesTwoFrames():
  m = vector3d(1, 0, 0, frame=aluminium())
  r = vector3d(0, 1, 0, frame=specimenFrame.rolling)
  with pytest.raises(ValueError):
    m + r


def test_vector3d_dotAndNormalizeKeepTheFrame():
  cs = aluminium()
  m = vector3d(1, 0, 0, frame=cs)
  assert m.dot(vector3d(1, 1, 0)) == 1 and m.normalize().frame is cs and m.normalize().dispStyle == "hkl"


def test_vector3d_sumOfAlignedFramesTakesTheFirst():
  cs = aluminium()
  L = crystalFrame("432", [4.05, 4.05, 4.05], mineral="Al")
  assert (vector3d(0, 1, 0, frame=L) + vector3d(1, 0, 0, frame=cs)).frame is L


def test_vector3d_rotationKeepsTheFrame():
  cs = aluminium()
  assert (mtex.rotation.byAxisAngle(mtex.zvector, 0.3) * vector3d(1, 0, 0, frame=cs)).frame is cs


def test_vector3d_indexingAndReshapeKeepTheFrame():
  cs = aluminium()
  v = vector3d(np.random.default_rng(0).random((5, 3)), frame=cs)
  assert v[2].frame is cs and v[2:3].frame is cs and v.reshape(1, 5).frame is cs and v.copy().frame is cs


def test_cat_joinsTheFrames():
  rng = np.random.default_rng(0)
  cs = aluminium()
  v = vector3d(rng.random((5, 3)), frame=cs)
  assert cat(v, v).frame is cs and cat(v, vector3d(rng.random((2, 3)))).frame is cs and cat(vector3d(rng.random((2, 3))), v).frame is cs
  assert cat(v, vector3d(rng.random((2, 3)), frame=cs.stripSym())).frame is cs.stripSym()


def test_cat_refusesTwoFrames():
  rng = np.random.default_rng(0)
  v = vector3d(rng.random((5, 3)), frame=aluminium())
  with pytest.raises(ValueError):
    cat(v, vector3d(rng.random((2, 3)), frame=specimenFrame.rolling))


def test_vector3d_assignmentRefusesAnotherFrame():
  v = vector3d(np.random.default_rng(0).random((5, 3)), frame=aluminium())
  v[0] = vector3d(1, 2, 3)
  with pytest.raises(ValueError):
    v[0] = vector3d(1, 0, 0, frame=specimenFrame.rolling)


def test_vector3d_reductionsAndScalingKeepTheFrame():
  cs = aluminium()
  v = vector3d(np.random.default_rng(0).random((5, 3)), frame=cs)
  v[0] = vector3d(1, 2, 3)
  assert v.mean().frame is cs and v.sum().frame is cs and v.perp().frame is cs and v.perp().dispStyle == "uvw"
  assert (v * 2).frame is cs and (v / 2).dispStyle == "hkl"


def test_unique_ofTheSymmetrisedDirectionIsOne():
  assert unique(symmetrise(Miller(1, 2, 3, crystalFrame("m-3m")))).size == 1
