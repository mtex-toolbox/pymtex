# products, frame rules and frame changes: the product and transition parts of
# check_referenceFrame beyond the semantics statements
import numpy as np
import pytest
from numpy import pi

import mtex
from mtex import (orientation, rotation, crystalFrame, specimenFrame, Miller, vector3d, xvector, zvector, degree,
                  angle, symmetrise, inv, transformReferenceFrame, rotate)
from conftest import close


def test_rotations_drop_the_group_on_their_side():
  cs, ss = crystalFrame("cubic"), specimenFrame("222")
  ori = orientation.byEuler(10 * degree, 20 * degree, 30 * degree, cs, ss)
  h = Miller(1, 0, 0, cs, "uvw")
  rot = rotation.byAxisAngle(xvector, 60 * degree)
  assert close(((rot * ori) * h).data, (rot * (ori * h)).data)
  assert angle(ori * (rot * vector3d(h.data)), (rot * ori) * h) > 1e-3
  l = rot * ori
  assert l.SS.numSym() == 1 and l.SS is ss.stripSym() and l.CS is cs
  r = ori * rot
  assert r.CS.numSym() == 1 and r.CS is cs.stripSym() and r.SS is ss
  assert (ss.sym.rotations()[1] * ori).SS is ss
  assert (rotation.byAxisAngle(zvector, pi + 1e-9) * ori).SS.numSym() == 4
  assert (rotation.byAxisAngle(zvector, 180.5 * degree) * ori).SS.numSym() == 1
  # a group of order two is not dropped: there is no direction to lose
  o2 = orientation.byEuler(0.1, 0.2, 0.3, crystalFrame("-1"), specimenFrame("-1"))
  assert (rot * o2).SS.numSym() == 2 and (o2 * rot).CS.numSym() == 2


def test_symmetrise_and_composition():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  o1 = orientation.byEuler(10 * degree, 20 * degree, 30 * degree, cs, ss)
  o2 = orientation.byEuler(40 * degree, 50 * degree, 60 * degree, cs, ss)
  m = inv(o1) * o2
  assert m.isMisorientation and m.CS is cs and m.SS is cs and m.CS.numSym() == 48
  assert np.max(angle(symmetrise(o1), o1)) < 1e-10 and symmetrise(o1).size == 24 * 4
  assert symmetrise(o1, unique=True).size == 96
  # a compatible crystal frame is absorbed into the product, an incompatible one refused
  csA = crystalFrame("triclinic", [1, 2, 3], np.array([70, 80, 120]) * degree, "Z||a*")
  csB = crystalFrame("triclinic", [1, 2, 3], np.array([70, 80, 120]) * degree, "Z||b", "X||a*")
  oriB = orientation(rotation.rand(rng=np.random.default_rng(2)), csB)
  mA = Miller(1, 2, 3, csA)
  assert np.linalg.norm((oriB * mA).data - (oriB * transformReferenceFrame(mA, csB)).data) < 1e-8
  with pytest.raises(ValueError):
    orientation(rotation.rand(), cs) * vector3d(1, 0, 0, frame=specimenFrame.default)
  assert (inv(orientation(rotation.rand(), cs)) * vector3d(1, 0, 0, frame=specimenFrame.default)).frame is cs
  assert (orientation(rotation.rand(), cs) * vector3d(1, 0, 0)).frame is specimenFrame.default
  assert (orientation(rotation.rand(5), cs) * Miller(1, 0, 0, cs)).shape == (5,)
  assert rotate(Miller(1, 0, 0, cs), 0.5).frame is cs and rotate(Miller(1, 0, 0, cs), 0.5) == rotation.byAxisAngle(zvector, 0.5) * Miller(1, 0, 0, cs)


def test_frame_change_is_not_rotation():
  cs1 = crystalFrame("triclinic", [1, 2, 3], np.array([70, 80, 120]) * degree, "Z||a*")
  cs2 = crystalFrame("triclinic", [1, 2, 3], np.array([70, 80, 120]) * degree, "Z||b", "X||a*")
  m = Miller(1, 2, 3, cs1)
  m2 = transformReferenceFrame(m, cs2)
  assert m2.frame is cs2 and close(m2.hkl, [1, 2, 3]) and m2.dispStyle == "hkl"
  assert np.linalg.norm(m2.data - Miller(1, 2, 3, cs2).data) < 1e-10
  assert transformReferenceFrame(Miller(1, 2, 3, cs1, "uvw"), cs2).dispStyle == "uvw"
  o = orientation.align(cs1, cs2)
  assert o.CS is cs1.stripSym() and o.SS is cs2.stripSym() and angle(o * m, m2, symmetry=False) < 1e-10
  with pytest.raises(ValueError):
    orientation.align(cs1, crystalFrame("m-3m", [4, 4, 4]))
  ori = orientation.byEuler(30 * degree, 50 * degree, 120 * degree, cs1)
  ori2 = transformReferenceFrame(ori, cs2)
  assert ori2.CS is cs2 and np.linalg.norm((ori * Miller(1, 0, 0, cs1)).data - (ori2 * Miller(1, 0, 0, cs2)).data) < 1e-10
  assert transformReferenceFrame(ori, cs1) is ori
  mori = orientation.rand(cs1, cs1, rng=np.random.default_rng(0))
  mori2 = transformReferenceFrame(mori, cs2, cs2)
  assert mori2.CS is cs2 and mori2.SS is cs2
  h = Miller(1, 2, 3, cs1)
  assert angle(transformReferenceFrame(mori * h, cs2), mori2 * transformReferenceFrame(h, cs2), symmetry=False) < 1e-10
  assert rotate(m, rotation.byAxisAngle(zvector, 0.3)).frame is cs1
  # byScreenAlignment and the explicit rule agree
  fa = specimenFrame("transitionA", axesNames=("A", "B", "C"), how2plot="y↑→x")
  fb = specimenFrame("transitionB", axesNames=("A", "B", "C"), how2plot="x↑→y")
  w = vector3d(1, 2, 3, frame=fa)
  byPlot = transformReferenceFrame(w, fb, rule="byScreenAlignment")
  byOri = transformReferenceFrame(w, fb, rule=orientation.byScreenAlignment(fa, fb))
  assert angle(byPlot, byOri) < 1e-10 and byPlot.frame is fb
  assert angle(orientation.byScreenAlignment(specimenFrame("i", how2plot=mtex.plottingConvention.ij), specimenFrame("j", how2plot=mtex.plottingConvention.ij))) < 1e-10
  with pytest.raises(ValueError):
    transformReferenceFrame(w, fb, rule="nonsense")
  assert transformReferenceFrame(w, fb, rule="byAxes").frame is fb
