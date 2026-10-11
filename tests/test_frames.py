# reference frames: the register, siblings, transitions, comparisons by value, the named
# specimen frames and the session default (check_symmetryCompare, the frame parts of
# check_referenceFrame and check_plottingConventionOwnership)
import numpy as np
import pytest
from numpy import pi

import mtex
from mtex import (crystalFrame, specimenFrame, notIndexedFrame, referenceFrame, plottingConvention, vector3d,
                  xvector, yvector, zvector, rotation, degree, Laue, properGroup, stripSym, fullSym, isAligned,
                  transformationMatrix, isCompatible, isclose, sim, numSym, symmetry,
                  alignmentSetups, closestAlignment)
from mtex.geometry.session import SESSION


def aluminium():
  return crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Aluminium")


def aluminium432():
  return crystalFrame("432", [4.05, 4.05, 4.05], mineral="Aluminium")


def test_crystalFrame_sameKeyIsTheSameFrame():
  cs = aluminium()
  assert crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Aluminium") is cs


def test_crystalFrame_cellWithinTheToleranceIsTheSameFrame():
  cs = aluminium()
  assert crystalFrame("m-3m", [4.06, 4.06, 4.06], mineral="Aluminium") is cs        # 0.2 percent is one phase


def test_crystalFrame_nameCellOrGroupSplitsTheFrame():
  cs = aluminium()
  assert crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Nickel") is not cs       # a name splits
  assert crystalFrame("m-3m", [8.1, 8.1, 8.1], mineral="Aluminium") is not cs       # a different cell splits
  assert crystalFrame("432", [4.05, 4.05, 4.05], mineral="Aluminium") is not cs     # a group splits


def test_crystalFrame_noInternMakesAFreshFrame():
  cs = aluminium()
  assert crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Aluminium", intern=False) is not cs


def test_crystalFrame_colourIsNotInTheKey():
  # the first colour stays unless asked for explicitly
  fo = crystalFrame("mmm", [4.8, 10, 6], mineral="Forsterite", color="green")
  assert crystalFrame("mmm", [4.8, 10, 6], mineral="Forsterite") is fo and fo.color == "green"
  crystalFrame("mmm", [4.8, 10, 6], mineral="Forsterite", color="red")
  assert fo.color == "red"


def test_crystalFrame_identityIsTheOnlyEquality():
  cs = aluminium()
  assert cs == cs and cs != aluminium432()


def test_crystalFrame_registeredFrameIsSealed():
  cs = aluminium()
  with pytest.raises(AttributeError):
    cs.basis = cs.basis
  with pytest.raises(AttributeError):
    cs.sym = symmetry("432")


def test_crystalFrame_nameAndColourCanBeSet():
  cs = aluminium()
  cs.color = "blue"
  assert cs.color == "blue"
  cs.mineral = "Al"
  assert cs.name == "Al"


def test_crystalFrame_cubicProperties():
  cs = aluminium()
  assert np.allclose(cs.abc, [4.05, 4.05, 4.05]) and np.allclose(cs.abg, pi / 2)
  assert np.all(cs.axes.data == cs.basis.data) and numSym(cs) == 48 and cs.isLaue and str(cs.lattice) == "cubic"
  assert cs.mineral == cs.name and cs.isIndexed and cs.id == 45 and cs.pointGroup == "m-3m" and cs.rot.size == 48
  assert cs.Schoenflies == "Oh" and cs.numProper() == 24 and cs.nfold() == 4 and len(cs) == 48


def test_crystalFrame_containsItsRotations():
  assert rotation.byAxisAngle(zvector, pi / 2) in aluminium()


def test_register_holdsOneFramePerKey():
  aluminium()
  crystalFrame("m-3m", [4.06, 4.06, 4.06], mineral="Aluminium")
  crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Nickel")
  crystalFrame("m-3m", [8.1, 8.1, 8.1], mineral="Aluminium")
  aluminium432()
  crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Aluminium", intern=False)
  crystalFrame("mmm", [4.8, 10, 6], mineral="Forsterite", color="green")
  crystalFrame("mmm", [4.8, 10, 6], mineral="Forsterite")
  assert len(SESSION.frames) == 5


def test_Laue_isAnAlignedSiblingOfTheSameName():
  cs = aluminium432()
  L = cs.Laue()
  assert L is not cs and L.pointGroup == "m-3m" and isAligned(L, cs) and L.name == cs.name and isinstance(L, crystalFrame)
  assert Laue(cs) is L and Laue(L) is L


def test_Laue_isTheRegisteredFrameOfItsKey():
  cs = aluminium432()
  L = cs.Laue()
  csL = aluminium()
  assert csL is L and properGroup(csL) is cs and properGroup(cs) is cs


def test_stripSym_isTheAlignedFrameWithoutSymmetry():
  cs = aluminium432()
  t = stripSym(cs)
  assert t.numSym() == 1 and isAligned(t, cs) and t.name == "Aluminium" and crystalFrame(cs) is t


def test_fullSym_isTheLaueSibling():
  cs = aluminium432()
  L = cs.Laue()
  t = stripSym(cs)
  assert fullSym(t) is L and fullSym(cs) is L


def test_crystalFrame_refusesASpecimenFrame():
  with pytest.raises(ValueError):
    crystalFrame(specimenFrame.rolling)


def test_crystalFrame_byPointId():
  bare = crystalFrame([2.95, 2.95, 4.68], pointId=40)
  assert bare.id == 40 and bare.numSym() == 24


def test_properSubGroup_ofTheLaueFrame():
  cs = aluminium432()
  assert aluminium().properSubGroup() is cs


def test_crystalFrame_byElements():
  # the group of a frame given by elements
  cE = crystalFrame(mtex.cat(rotation.byAxisAngle(zvector, pi / 2), rotation.byAxisAngle(vector3d(1, 1, 1), 2 * pi / 3)))
  assert cE.pointGroup == "432" and crystalFrame.byElements(rotation.byAxisAngle(zvector, 2 * pi / 3)).pointGroup == "3"


def test_union_andDisjoint_joinOrIntersectTheGroup():
  cs = aluminium432()
  L = cs.Laue()
  assert cs.union(crystalFrame("-1", [4.05, 4.05, 4.05], mineral="Aluminium")) is L
  assert L.disjoint(cs) is cs and cs.disjoint(rotation.byAxisAngle(zvector, pi / 2)).pointGroup == "4"


def test_crystalFrame_orderedBySubgroup():
  cs = aluminium432()
  L = cs.Laue()
  assert cs <= L and L >= cs and not L <= cs


# two triclinic settings of one cell
def triclinicPair():
  cs1 = crystalFrame("triclinic", [1, 2, 3], np.array([70, 80, 120]) * degree, "Z||a*")
  cs2 = crystalFrame("triclinic", [1, 2, 3], np.array([70, 80, 120]) * degree, "Z||b", "X||a*")
  return cs1, cs2


def test_transformationMatrix_isTheChangeOfBasis():
  cs1, cs2 = triclinicPair()
  A1 = cs1.basis.normalize().data
  A2 = cs2.basis.normalize().data
  assert np.allclose(transformationMatrix(cs1, cs2), np.linalg.inv(A2) @ A1)


def test_transformationMatrix_keepsTheIndicesOfADirection():
  # the transition takes coordinates of a direction from cs1 into cs2: the same indices in both
  cs1, cs2 = triclinicPair()
  d1 = np.array([1, 2, 3]) @ cs1.basisDual().data
  d2 = np.array([1, 2, 3]) @ cs2.basisDual().data
  assert np.linalg.norm(transformationMatrix(cs1, cs2) @ d1 - d2) < 1e-10


def test_isCompatible_monoclinicSettingsRepairedByLength():
  # the monoclinic settings 121 and 112 differ by a permutation of the axes: re-paired by length
  csA = crystalFrame("121", [2, 3, 4], np.array([90, 105, 90]) * degree)
  csB = crystalFrame("112", [4, 2, 3], np.array([90, 90, 105]) * degree)
  ok, M = isCompatible(csA, csB)
  assert ok and np.linalg.norm(M @ M.T - np.eye(3)) < 1e-10
  Mpure = np.linalg.inv(csB.basis.normalize().data) @ csA.basis.normalize().data
  assert np.linalg.norm(Mpure @ Mpure.T - np.eye(3)) > 0.1


def test_crystalFrame_fromItsAxesAgreesWithTheLattice():
  cs1, _ = triclinicPair()
  cF = crystalFrame(cs1.basis)
  assert np.all((cF.basis - cs1.basis).norm() < 1e-10) and cF.numSym() == 1 and cF.axesNames == ("a", "b", "c")


def test_crystalFrame_fromHexagonalAxesKeepsTheAlignment():
  cH = crystalFrame("6/mmm", [3, 3, 5], "X||a")
  assert crystalFrame(cH.basis).alignment() == cH.alignment()


def test_crystalFrame_fromCellAndAnglesWithoutGroup():
  cH = crystalFrame("6/mmm", [3, 3, 5], "X||a")
  cF2 = crystalFrame([3, 3, 5], np.array([90, 90, 120]) * degree, "X||a")
  assert np.all((cF2.basis - cH.basis).norm() < 1e-10) and "X||a" in cF2.alignment()


def test_crystalFrame_reprShowsTheAlignment():
  out = repr(crystalFrame("6/mmm", [3, 3, 5], "X||a"))
  assert "X||a" in out and "⊙c→a" in out


def test_eqTol_andSim_ofSpecimenFrames():
  triclinic1, ortho, ortho2 = specimenFrame("1"), specimenFrame("mmm"), specimenFrame("mmm")
  assert isclose(triclinic1, triclinic1) and isclose(ortho, ortho2) and sim(ortho, ortho2)
  assert not isclose(triclinic1, ortho) and not sim(triclinic1, ortho) and not isclose(ortho, triclinic1)


def test_eqTol_ofCrystalGroups():
  cs = crystalFrame("432")
  assert isclose(cs, cs) and not isclose(cs, crystalFrame("222"))


def test_eqTol_twoMineralsSharingALatticeAreTwoPhases():
  ni = crystalFrame("m-3m", [3.52, 3.52, 3.52], mineral="Nickel")
  fe = crystalFrame("m-3m", [3.52, 3.52, 3.52], mineral="Iron fcc")
  assert not isclose(ni, fe) and not sim(ni, fe) and isclose(ni, crystalFrame("m-3m", [3.52, 3.52, 3.52], mineral="Nickel"))


def test_eqTol_wantsAlignedBasesSimOnlyTheCell():
  q1 = crystalFrame("321", [4.9, 4.9, 5.4], mineral="Quartz")
  q2 = crystalFrame("321", [4.9, 4.9, 5.4], mineral="Quartz", alignment=["X||a"])
  assert not isclose(q1, q2) and sim(q1, q2)
  assert not isAligned(q1, q2) and isAligned(q1, Laue(q1))


def test_eqTol_properGroupAndItsLaueGroup():
  assert isclose(crystalFrame("432", [2, 2, 2], mineral="X"), crystalFrame("m-3m", [2, 2, 2], mineral="X"))


def test_eqTol_crystalAgainstSpecimenFrame():
  cs = crystalFrame("432")
  with pytest.warns(UserWarning):
    ss432 = specimenFrame("432")
  assert not isclose(cs, ss432) and not sim(cs, specimenFrame("mmm"))


def test_notIndexedFrame_isOneFrameWithoutSymmetry():
  n1, n2 = notIndexedFrame(), notIndexedFrame("NotIndexed")
  assert isclose(n1, n2) and sim(n1, n2) and not n1.isIndexed and n1.numSym() == 1 and n1.sibling(symmetry("mmm")) is n1


def test_specimenFrame_rolling():
  sF = specimenFrame.rolling
  assert sF.name == "rolling" and sF.axesNames == ("RD", "TD", "ND") and specimenFrame.rolling is sF
  assert sF.how2plot.char() == "y←↑x" and sF.conventionChar() == "TD←RD↑" and "TD←RD↑" in repr(sF)


def test_specimenFrame_specimenAndMeasurementAxes():
  assert specimenFrame.specimen.axesNames == ("X", "Y", "Z") and specimenFrame.measurement.axesNames == ("X1", "Y1", "Z1")


def test_specimenFrame_constructorMakesAFreshFrame():
  # the constructor makes a fresh frame, the factories return the registered one
  assert specimenFrame("rolling") is not specimenFrame.rolling and specimenFrame("rolling", intern=False).name == "rolling"


def test_specimenFrame_defaultIsTheSpecimenFrameDrawnAsSEMImages():
  d = specimenFrame.default
  assert d is specimenFrame.specimen and plottingConvention.default == plottingConvention.ij and d.how2plot.char() == "y↓→x"
  assert specimenFrame("1") is d and specimenFrame() is d


def test_specimenFrame_withAGroupIsAnAlignedSibling():
  d = specimenFrame.default
  ss = specimenFrame("222")
  assert ss is not d and ss.numSym() == 4 and isAligned(ss, d) and ss.how2plot == d.how2plot
  assert specimenFrame("222") is ss and d.numSym() == 1


def test_specimenFrame_rollingWithAGroup():
  sF = specimenFrame.rolling
  assert specimenFrame.rolling("mmm") is not sF and specimenFrame.rolling("mmm").numSym() == 8 and specimenFrame.rolling("mmm").name == "rolling"
  assert specimenFrame.rolling("mmm") is specimenFrame.rolling("mmm") and specimenFrame.rolling() is sF


def test_specimenFrame_warnsOfACubicGroup():
  with pytest.warns(UserWarning):
    specimenFrame("m-3m")


def test_specimenFrame_customConventionForksTheFrame():
  d = specimenFrame.default
  pC = plottingConvention("z↑→x")
  ssF = specimenFrame(pC)
  assert ssF is not d and ssF.how2plot == pC and d.how2plot != pC and specimenFrame.frameFor(pC) is ssF
  assert specimenFrame.frameFor(plottingConvention.default) is d and specimenFrame.frameFor("z↑→x") is ssF


def test_makeDefault_replacesTheDefaultFramesConvention():
  d = specimenFrame.default
  pC = plottingConvention("z↑→x")
  ssF = specimenFrame(pC)
  plottingConvention("y↑→x").makeDefault()
  assert plottingConvention.default == plottingConvention("y↑→x") and d.how2plot == plottingConvention("y↑→x")
  assert ssF.how2plot == pC and d.numSym() == 1


def test_default_setsTheEastDirectionOfTheDefaultConvention():
  plottingConvention.default(east=xvector)
  assert plottingConvention.default.east.isclose(xvector)


def test_makeDefault_anySpecimenFrameCanTakeOverTheDefault():
  specimenFrame.rolling.makeDefault()
  assert specimenFrame.default is specimenFrame.rolling and plottingConvention.default == specimenFrame.rolling.how2plot
  assert specimenFrame("1") is specimenFrame.rolling


def test_reset_restoresThePristineState():
  # the register forgets, rolling reminted with its seed
  sF = specimenFrame.rolling
  sF.makeDefault()
  mtex.reset()
  assert specimenFrame.default.name == "specimen" and plottingConvention.default == plottingConvention.ij
  assert specimenFrame.rolling is not sF and specimenFrame.rolling.how2plot == plottingConvention("y←↑x")


def test_stripSym_afterAResetComesFromTheNewRegister():
  stripSym(crystalFrame("432", [3.6, 3.6, 3.6], mineral="gamma"))
  mtex.reset()
  t = stripSym(crystalFrame("432", [3.6, 3.6, 3.6], mineral="gamma"))
  assert any(r is t for r in mtex.geometry.session.SESSION.frames)


def test_specimenFrame_refusesACrystalFrame():
  with pytest.raises(ValueError):
    specimenFrame(crystalFrame("m-3m"))


def test_specimenFrame_ofASpecimenFrameStripsTheGroup():
  assert specimenFrame(specimenFrame("222")) is specimenFrame.default


def test_specimenFrame_byElements():
  assert specimenFrame(rotation.byAxisAngle(zvector, pi)).numSym() == 2


def test_frameFor_leavesTheOtherFramesConventions():
  pC = plottingConvention("z↑→x")
  dflt = specimenFrame.default.how2plot
  assert dflt != pC
  ss = specimenFrame("222")
  before = ss.how2plot
  ss2 = specimenFrame.frameFor(pC)
  assert ss2.how2plot == pC and ss2 is not ss and ss.how2plot == before and specimenFrame.default.how2plot == dflt


def test_Laue_sharesTheConvention():
  cs = crystalFrame("432")
  csBefore = cs.how2plot
  assert cs.Laue() is not cs and cs.Laue().how2plot == cs.how2plot
  assert cs.how2plot == csBefore


def test_default_changesTheDefaultFramesConvention():
  dflt = specimenFrame.default.how2plot
  ssd = specimenFrame("1")
  assert ssd is specimenFrame.default
  plottingConvention.default(north=xvector)
  assert plottingConvention.default != dflt and ssd.how2plot == plottingConvention.default


def test_crystalFrame_drawsCStarOutOfTheScreenAndAEast():
  cs = crystalFrame("432")
  assert cs.how2plot == plottingConvention(cs.cAxisRec, cs.aAxis)


def test_crystalFrame_monoclinicDrawsBEast():
  mono = crystalFrame("2/m", [8.6, 13, 7.2], [90, 116, 90])
  assert mono.how2plot.east.isclose(mono.bAxis.normalize()) and mono.conventionChar() == "⊙c*→b"


def test_crystalFrame_cubicConventionYUpXRightNamesTheSameFrame():
  # for a cubic frame y↑→x is the crystal convention itself, so it names the same frame
  assert crystalFrame("432", how2plot="y↑→x") is crystalFrame("432")


def test_crystalFrame_otherConventionForksTheFrame():
  cs = crystalFrame("432")
  cs2 = crystalFrame("432", how2plot="z↑→x")
  assert cs2.how2plot == plottingConvention("z↑→x") and cs2 is not cs


def test_crystalFrame_conventionCanBeSet():
  cs = crystalFrame("432")
  cs.how2plot = "x↑→y"
  assert cs.how2plot == plottingConvention("x↑→y")


def test_crystalFrame_repr():
  r = repr(crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Aluminium", color="blue"))
  assert r.startswith("crystalFrame (⊙") and "mineral : Aluminium" in r and "color   : blue" in r and "elements: 48" in r


def test_frame_strIsItsName():
  cs = crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Aluminium", color="blue")
  assert str(cs) == "Aluminium" and str(crystalFrame("432")) == "432" and str(specimenFrame.rolling) == "rolling"


def test_crystalFrame_reprShowsTheAnglesTheLatticeDoesNotImply():
  # as MATLAB shows them
  r = repr(crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Aluminium", color="blue"))
  assert "alpha" not in r and "alpha" not in repr(crystalFrame("6/mmm", [3, 3, 5]))
  assert "alpha, beta, gamma: 80°, 90°, 100°" in repr(crystalFrame("-1", [3, 4, 5], [80 * degree, 90 * degree, 100 * degree]))


def test_notIndexedFrame_reprAndTheDefaultsStr():
  assert repr(notIndexedFrame()) == "notIndexedFrame (notIndexed)" and str(specimenFrame.default) == "specimen"


def test_specimenFrame_reprShowsTheConvention():
  assert repr(specimenFrame("222")).startswith("specimenFrame (specimen, y↓→x)")


def test_alignmentSetups_areTwelve():
  setups = alignmentSetups()
  assert len(setups) == 12 and setups[1] == ["X||a", "Z||c*"]


def test_closestAlignment_givesTheSetupATriclinicFrameWasBuiltIn():
  setups = alignmentSetups()
  tric = [crystalFrame("-1", [4, 5, 6], np.array([80, 85, 95]) * degree, *s) for s in setups]
  for s, fr in zip(setups, tric):
    assert closestAlignment(fr) == s


def test_closestAlignment_hexagonalReportsOneOfTheCoincidingNames():
  # a hexagonal lattice reports Z||c whichever of the coinciding names was asked for
  hexa = crystalFrame("6/mmm", [3, 3, 5], np.array([90, 90, 120]) * degree, "X||a", "Z||c")
  assert closestAlignment(hexa) == ["X||a", "Z||c*"]
  assert closestAlignment(crystalFrame("6/mmm", [3, 3, 5], np.array([90, 90, 120]) * degree, "X||a*", "Z||c")) == ["X||a*", "Z||c"]


def test_closestAlignment_orthogonalLatticeIsStillPlaced():
  # an orthogonal lattice states no alignment of its own and is still placed
  assert closestAlignment(crystalFrame("m-3m", [4, 4, 4])) == ["X||a", "Z||c*"]


def test_crystalFrame_readsTheCenteringOffTheSpaceGroup():
  assert crystalFrame("I4/mmm", [3.3, 3.3, 5.8]).centering == "I" and crystalFrame("Fm-3m", [4, 4, 4]).centering == "F"
  assert crystalFrame("Cmcm", [1, 2, 3]).centering == "C" and crystalFrame("R-3m", [1, 1, 6]).centering == "R"
  assert crystalFrame("4/mmm", [3.3, 3.3, 5.9], centering="I").centering == "I"
  assert crystalFrame("4/mmm", [3.3, 3.3, 5.7]).centering == "P"


def test_crystalFrame_withoutACenteringIsTheFrameThatStatesIt():
  plain = crystalFrame("m-3m", [4, 4, 4], mineral="centred")
  assert crystalFrame("Fm-3m", [4, 4, 4], mineral="centred") is plain
  assert plain.centering == "F" and plain.stripSym().centering == "F"


def test_crystalFrame_withCenteringsThatDisagreeAreTwoFrames():
  f = crystalFrame("Fm-3m", [4, 4, 4], mineral="two lattices")
  i = crystalFrame("Im-3m", [4, 4, 4], mineral="two lattices")
  assert f is not i and (f.centering, i.centering) == ("F", "I")


def test_crystalFrame_refusesAnUnknownCentering():
  with pytest.raises(ValueError):
    crystalFrame("m-3m", centering="Q")


@pytest.mark.parametrize("symbol, share", [("Pm-3m", 1), ("Im-3m", 1 / 2), ("Fm-3m", 1 / 4), ("Cmmm", 1 / 2), ("R-3m", 1 / 3)])
def test_primitiveAxes_spanTheShareOfTheCellOneLatticePointHolds(symbol, share):
  cs = crystalFrame(symbol, [1, 2, 3] if symbol == "Cmmm" else [1, 1, 6] if symbol == "R-3m" else [4, 4, 4])
  P, C = cs.primitiveAxes().data, cs.axes.data
  assert np.isclose(abs(np.linalg.det(P)), share * abs(np.linalg.det(C)))
  # every conventional axis is a lattice vector of the primitive cell
  assert np.allclose(np.linalg.solve(P.T, C.T), np.rint(np.linalg.solve(P.T, C.T)))
