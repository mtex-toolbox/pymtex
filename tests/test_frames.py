# reference frames: the register, siblings, transitions, comparisons by value, the named
# specimen frames and the session default (check_symmetryCompare, the frame parts of
# check_referenceFrame and check_plottingConventionOwnership)
import numpy as np
import pytest
from numpy import pi

import mtex
from mtex import (crystalFrame, specimenFrame, notIndexedFrame, referenceFrame, plottingConvention, vector3d,
                  xvector, yvector, zvector, rotation, degree, Laue, properGroup, stripSym, fullSym, isAligned,
                  transformationMatrix, isCompatible, eqTol, sim, numSym, plotx2east, plotx2north, symmetry)
from mtex.geometry.session import SESSION


def test_the_register():
  cs = crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Aluminium")
  assert crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Aluminium") is cs
  assert crystalFrame("m-3m", [4.06, 4.06, 4.06], mineral="Aluminium") is cs        # 0.2 percent is one phase
  assert crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Nickel") is not cs       # a name splits
  assert crystalFrame("m-3m", [8.1, 8.1, 8.1], mineral="Aluminium") is not cs       # a different cell splits
  assert crystalFrame("432", [4.05, 4.05, 4.05], mineral="Aluminium") is not cs     # a group splits
  assert crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Aluminium", noIntern=True) is not cs
  # colour is not in the key: the first colour stays unless asked for explicitly
  fo = crystalFrame("mmm", [4.8, 10, 6], mineral="Forsterite", color="green")
  assert crystalFrame("mmm", [4.8, 10, 6], mineral="Forsterite") is fo and fo.color == "green"
  crystalFrame("mmm", [4.8, 10, 6], mineral="Forsterite", color="red")
  assert fo.color == "red"
  # identity is the only equality, and a registered frame is sealed except name and colour
  assert cs == cs and cs != crystalFrame("432", [4.05, 4.05, 4.05], mineral="Aluminium")
  with pytest.raises(AttributeError):
    cs.basis = cs.basis
  with pytest.raises(AttributeError):
    cs.sym = symmetry("432")
  cs.color = "blue"
  assert cs.color == "blue"
  cs.mineral = "Al"
  assert cs.name == "Al"
  assert np.allclose(cs.abc, [4.05, 4.05, 4.05]) and np.allclose(cs.abg, pi / 2)
  assert np.all(cs.axes.data == cs.basis.data) and numSym(cs) == 48 and cs.isLaue and str(cs.lattice) == "cubic"
  assert cs.mineral == cs.name and cs.isIndexed and cs.id == 45 and cs.pointGroup == "m-3m" and cs.rot.size == 48
  assert cs.Schoenflies == "Oh" and cs.numProper() == 24 and cs.nfold() == 4 and len(cs) == 48
  assert rotation.byAxisAngle(zvector, pi / 2) in cs
  assert len(SESSION.frames) == 5


def test_siblings():
  cs = crystalFrame("432", [4.05, 4.05, 4.05], mineral="Aluminium")
  L = cs.Laue()
  assert L is not cs and L.pointGroup == "m-3m" and isAligned(L, cs) and L.name == cs.name and isinstance(L, crystalFrame)
  assert Laue(cs) is L and Laue(L) is L
  csL = crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Aluminium")
  assert csL is L and properGroup(csL) is cs and properGroup(cs) is cs
  t = stripSym(cs)
  assert t.numSym() == 1 and isAligned(t, cs) and t.name == "Aluminium" and crystalFrame(cs) is t
  assert fullSym(t) is L and fullSym(cs) is L
  with pytest.raises(ValueError):
    crystalFrame(specimenFrame.rolling)
  bare = crystalFrame([2.95, 2.95, 4.68], pointId=40)
  assert bare.id == 40 and bare.numSym() == 24
  assert csL.properSubGroup() is cs
  # the group of a frame given by elements
  cE = crystalFrame(mtex.cat(rotation.byAxisAngle(zvector, pi / 2), rotation.byAxisAngle(vector3d(1, 1, 1), 2 * pi / 3)))
  assert cE.pointGroup == "432" and crystalFrame.byElements(rotation.byAxisAngle(zvector, 2 * pi / 3)).pointGroup == "3"
  # the frame with a group joined or intersected
  assert cs.union(crystalFrame("-1", [4.05, 4.05, 4.05], mineral="Aluminium")) is L
  assert L.disjoint(cs) is cs and cs.disjoint(rotation.byAxisAngle(zvector, pi / 2)).pointGroup == "4"
  assert cs <= L and L >= cs and not L <= cs


def test_frame_transitions():
  cs1 = crystalFrame("triclinic", [1, 2, 3], np.array([70, 80, 120]) * degree, "Z||a*")
  cs2 = crystalFrame("triclinic", [1, 2, 3], np.array([70, 80, 120]) * degree, "Z||b", "X||a*")
  A1 = cs1.basis.normalize().data
  A2 = cs2.basis.normalize().data
  assert np.allclose(transformationMatrix(cs1, cs2), np.linalg.inv(A2) @ A1)
  # the transition takes coordinates of a direction from cs1 into cs2: the same indices in both
  d1 = np.array([1, 2, 3]) @ cs1.basisDual().data
  d2 = np.array([1, 2, 3]) @ cs2.basisDual().data
  assert np.linalg.norm(transformationMatrix(cs1, cs2) @ d1 - d2) < 1e-10
  # the monoclinic settings 121 and 112 differ by a permutation of the axes: re-paired by length
  csA = crystalFrame("121", [2, 3, 4], np.array([90, 105, 90]) * degree)
  csB = crystalFrame("112", [4, 2, 3], np.array([90, 90, 105]) * degree)
  ok, M = isCompatible(csA, csB)
  assert ok and np.linalg.norm(M @ M.T - np.eye(3)) < 1e-10
  Mpure = np.linalg.inv(csB.basis.normalize().data) @ csA.basis.normalize().data
  assert np.linalg.norm(Mpure @ Mpure.T - np.eye(3)) > 0.1
  # a frame built from its axes agrees with the lattice construction
  cF = crystalFrame(cs1.basis)
  assert np.all((cF.basis - cs1.basis).norm() < 1e-10) and cF.numSym() == 1 and cF.axesNames == ("a", "b", "c")
  cH = crystalFrame("6/mmm", [3, 3, 5], "X||a")
  assert crystalFrame(cH.basis).alignment() == cH.alignment()
  cF2 = crystalFrame([3, 3, 5], np.array([90, 90, 120]) * degree, "X||a")
  assert np.all((cF2.basis - cH.basis).norm() < 1e-10) and "X||a" in cF2.alignment()
  out = repr(cH)
  assert "X||a" in out and "⊙c→a" in out


def test_comparing_frames_by_value():
  triclinic1, ortho, ortho2 = specimenFrame("1"), specimenFrame("mmm"), specimenFrame("mmm")
  assert eqTol(triclinic1, triclinic1) and eqTol(ortho, ortho2) and sim(ortho, ortho2)
  assert not eqTol(triclinic1, ortho) and not sim(triclinic1, ortho) and not eqTol(ortho, triclinic1)
  cs = crystalFrame("432")
  assert eqTol(cs, cs) and not eqTol(cs, crystalFrame("222"))
  # two minerals sharing a lattice are two phases
  ni = crystalFrame("m-3m", [3.52, 3.52, 3.52], mineral="Nickel")
  fe = crystalFrame("m-3m", [3.52, 3.52, 3.52], mineral="Iron fcc")
  assert not eqTol(ni, fe) and not sim(ni, fe) and eqTol(ni, crystalFrame("m-3m", [3.52, 3.52, 3.52], mineral="Nickel"))
  # eqTol wants aligned bases, sim only the cell
  q1 = crystalFrame("321", [4.9, 4.9, 5.4], mineral="Quartz")
  q2 = crystalFrame("321", [4.9, 4.9, 5.4], mineral="Quartz", alignment=["X||a"])
  assert not eqTol(q1, q2) and sim(q1, q2)
  assert eqTol(crystalFrame("432", [2, 2, 2], mineral="X"), crystalFrame("m-3m", [2, 2, 2], mineral="X"))
  assert not isAligned(q1, q2) and isAligned(q1, Laue(q1))
  with pytest.warns(UserWarning):
    ss432 = specimenFrame("432")
  assert not eqTol(cs, ss432) and not sim(cs, ortho)
  n1, n2 = notIndexedFrame(), notIndexedFrame("NotIndexed")
  assert eqTol(n1, n2) and sim(n1, n2) and not n1.isIndexed and n1.numSym() == 1 and n1.sibling(symmetry("mmm")) is n1


def test_named_specimen_frames_and_the_default():
  sF = specimenFrame.rolling
  assert sF.name == "rolling" and sF.axesNames == ("RD", "TD", "ND") and specimenFrame.rolling is sF
  assert sF.how2plot.char() == "y←↑x" and sF.conventionChar() == "TD←RD↑" and "TD←RD↑" in repr(sF)
  assert specimenFrame.specimen.axesNames == ("X", "Y", "Z") and specimenFrame.measurement.axesNames == ("X1", "Y1", "Z1")
  # the constructor makes a fresh frame, the factories return the registered one
  assert specimenFrame("rolling") is not sF and specimenFrame("rolling", noIntern=True).name == "rolling"
  # the default frame is the generic specimen frame drawn as SEM images are
  d = specimenFrame.default
  assert d is specimenFrame.specimen and plottingConvention.default == plottingConvention.ij and d.how2plot.char() == "y↓→x"
  assert specimenFrame("1") is d and specimenFrame() is d
  ss = specimenFrame("222")
  assert ss is not d and ss.numSym() == 4 and isAligned(ss, d) and ss.how2plot == d.how2plot
  assert specimenFrame("222") is ss and d.numSym() == 1
  assert specimenFrame.rolling("mmm") is not sF and specimenFrame.rolling("mmm").numSym() == 8 and specimenFrame.rolling("mmm").name == "rolling"
  assert specimenFrame.rolling("mmm") is specimenFrame.rolling("mmm") and specimenFrame.rolling() is sF
  with pytest.warns(UserWarning):
    specimenFrame("m-3m")
  # a custom convention forks the frame; a new default replaces the frame's convention
  pC = plottingConvention("z↑→x")
  ssF = specimenFrame(pC)
  assert ssF is not d and ssF.how2plot == pC and d.how2plot != pC and specimenFrame.frameFor(pC) is ssF
  assert specimenFrame.frameFor(plottingConvention.default) is d and specimenFrame.frameFor("z↑→x") is ssF
  plottingConvention("y↑→x").makeDefault()
  assert plottingConvention.default == plottingConvention("y↑→x") and d.how2plot == plottingConvention("y↑→x")
  assert ssF.how2plot == pC and d.numSym() == 1
  plotx2east()
  assert plottingConvention.default.east == xvector
  # any specimen frame can take over the default
  specimenFrame.rolling.makeDefault()
  assert specimenFrame.default is specimenFrame.rolling and plottingConvention.default == specimenFrame.rolling.how2plot
  assert specimenFrame("1") is specimenFrame.rolling
  # reset restores the pristine state: the register forgets, rolling reminted with its seed
  mtex.reset()
  assert specimenFrame.default.name == "specimen" and plottingConvention.default == plottingConvention.ij
  assert specimenFrame.rolling is not sF and specimenFrame.rolling.how2plot == plottingConvention("y←↑x")
  with pytest.raises(ValueError):
    specimenFrame(crystalFrame("m-3m"))
  assert specimenFrame(specimenFrame("222")) is specimenFrame.default
  assert specimenFrame(rotation.byAxisAngle(zvector, pi)).numSym() == 2


def test_convention_ownership():
  pC = plottingConvention("z↑→x")
  dflt = specimenFrame.default.how2plot
  assert dflt != pC
  ss = specimenFrame("222")
  before = ss.how2plot
  ss2 = specimenFrame.frameFor(pC)
  assert ss2.how2plot == pC and ss2 is not ss and ss.how2plot == before and specimenFrame.default.how2plot == dflt
  cs = crystalFrame("432")
  csBefore = cs.how2plot
  assert cs.Laue() is not cs and cs.Laue().how2plot == cs.how2plot
  assert cs.how2plot == csBefore
  ssd = specimenFrame("1")
  assert ssd is specimenFrame.default
  plotx2north()
  assert plottingConvention.default != dflt and ssd.how2plot == plottingConvention.default
  # a crystal frame draws c* out of the screen and a east, b for the monoclinic settings
  assert cs.how2plot == plottingConvention(cs.cAxisRec, cs.aAxis)
  mono = crystalFrame("2/m", [8.6, 13, 7.2], [90, 116, 90])
  assert mono.how2plot.east == mono.bAxis.normalize() and mono.conventionChar() == "⊙c*→b"
  # for a cubic frame y↑→x is the crystal convention itself, so it names the same frame
  assert crystalFrame("432", how2plot="y↑→x") is cs
  cs2 = crystalFrame("432", how2plot="z↑→x")
  assert cs2.how2plot == plottingConvention("z↑→x") and cs2 is not cs
  cs.how2plot = "x↑→y"
  assert cs.how2plot == plottingConvention("x↑→y")


def test_repr_and_char():
  cs = crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Aluminium", color="blue")
  r = repr(cs)
  assert r.startswith("crystalFrame (⊙") and "mineral : Aluminium" in r and "color   : blue" in r and "elements: 48" in r
  assert str(cs) == "Aluminium" and str(crystalFrame("432")) == "432" and str(specimenFrame.rolling) == "rolling"
  # the angles only where the lattice does not imply them, as MATLAB shows them
  assert "alpha" not in r and "alpha" not in repr(crystalFrame("6/mmm", [3, 3, 5]))
  assert "alpha, beta, gamma: 80°, 90°, 100°" in repr(crystalFrame("-1", [3, 4, 5], [80 * degree, 90 * degree, 100 * degree]))
  assert repr(notIndexedFrame()) == "notIndexedFrame (notIndexed)" and str(specimenFrame.default) == "specimen"
  assert repr(specimenFrame("222")).startswith("specimenFrame (specimen, y↓→x)")


# the twelve alignment setups, and the setup a frame is reported in
def test_alignment_setups():
  from mtex import alignmentSetups, closestAlignment, degree
  setups = alignmentSetups()
  assert len(setups) == 12 and setups[1] == ["X||a", "Z||c*"]
  tric = [crystalFrame("-1", [4, 5, 6], np.array([80, 85, 95]) * degree, *s) for s in setups]
  for s, fr in zip(setups, tric):
    assert closestAlignment(fr) == s
  # a hexagonal lattice reports Z||c whichever of the coinciding names was asked for
  hexa = crystalFrame("6/mmm", [3, 3, 5], np.array([90, 90, 120]) * degree, "X||a", "Z||c")
  assert closestAlignment(hexa) == ["X||a", "Z||c*"]
  assert closestAlignment(crystalFrame("6/mmm", [3, 3, 5], np.array([90, 90, 120]) * degree, "X||a*", "Z||c")) == ["X||a*", "Z||c"]
  # an orthogonal lattice states no alignment of its own and is still placed
  assert closestAlignment(crystalFrame("m-3m", [4, 4, 4])) == ["X||a", "Z||c*"]
