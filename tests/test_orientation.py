# orientations and their container: construction, frames, indexing, inverse, exchange
import numpy as np
import pytest
from numpy import pi

import mtex
from mtex import orientation, rotation, quaternion, crystalFrame, specimenFrame, Miller, vector3d, xvector, zvector, degree, inv, Euler, angle, cat
from conftest import close


def same(p, q, tol=1e-13):
  return np.all(1 - np.abs(np.sum(p.data[..., :4] * q.data[..., :4], axis=-1)) < tol)


def al():
  return crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Al")


def ti():
  return crystalFrame("6/mmm", [3, 3, 5], mineral="Ti")


def ss222():
  return specimenFrame("222")


def eulerOri(CS=None, SS=None):
  return orientation.byEuler(10 * degree, 20 * degree, 30 * degree, CS or al(), SS or ss222())


# ten random orientations of aluminium and the rotations they were built from
def alList():
  rots = rotation.rand(10, rng=np.random.default_rng(1))
  return rots, orientation(rots, al())


def test_orientation_byEulerCarriesItsFrames():
  cs, ss = al(), ss222()
  o = eulerOri(cs, ss)
  assert isinstance(o, orientation) and isinstance(o, rotation) and o.shape == ()
  assert o.CS is cs and o.SS is ss and o.frameA is cs and o.frameB is ss


def test_orientation_byEulerIsTheRotation():
  o = eulerOri()
  assert same(o, rotation.byEuler(10 * degree, 20 * degree, 30 * degree))
  assert close(Euler(o), [10 * degree, 20 * degree, 30 * degree]) and close(o.phi1, 10 * degree)


def test_orientation_defaultFrames():
  # the default specimen frame is the session frame, the default crystal frame the trivial one
  assert orientation(rotation.id, al()).SS is specimenFrame.default and orientation(rotation.id).CS.numSym() == 1
  assert orientation(rotation.id).CS.pointGroup == "1" and isinstance(orientation(rotation.id).CS, crystalFrame)


def test_inv_swapsTheFrames():
  cs, ss = al(), ss222()
  o = eulerOri(cs, ss)
  i = inv(o)
  assert i.CS is ss and i.SS is cs and same(inv(i), o)


def test_orientation_betweenTwoCrystalFramesIsAMisorientation():
  m = orientation(rotation.rand(), al(), ti())
  assert m.isMisorientation and not eulerOri().isMisorientation and not m.antipodal


def test_orientation_antipodalWithinOneFrame():
  # only a misorientation within a frame takes grain exchange
  cs = al()
  assert orientation(rotation.rand(), cs, cs, antipodal=True).antipodal


def test_orientation_antipodalRefusedBetweenOtherFrames():
  cs = al()
  for frames in ((cs, ss222()), (cs, ti()), (cs, crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Al", intern=False))):
    with pytest.raises(ValueError):
      orientation(rotation.rand(), *frames, antipodal=True)


def test_orientation_relabellingRestatesTheFramesWithoutMoving():
  ss, cs2 = ss222(), ti()
  o = eulerOri(SS=ss)
  r = orientation(o, cs2)
  assert r.CS is cs2 and r.SS is ss and same(r, o)


def test_orientation_framesCanBeSet():
  cs2 = ti()
  o = eulerOri()
  o.CS = cs2
  assert o.frameA is cs2
  o.SS = specimenFrame.rolling
  assert o.frameB is specimenFrame.rolling


def test_orientation_fromQuaternionComponents():
  cs, ss = al(), ss222()
  assert same(orientation(1, 1, 1, 1, cs), rotation(0.5, 0.5, 0.5, 0.5)) and orientation(1, 1, 1, 1, cs).CS is cs
  assert orientation(quaternion(1, 2, 3, 4), cs, ss).SS is ss and close(orientation(quaternion(1, 2, 3, 4)).norm(), 1)


def test_orientation_empty():
  assert orientation(al(), ss222()).isempty and orientation().isempty


def test_orientation_copyKeepsTheFrames():
  cs2 = ti()
  o = eulerOri(cs2)
  assert orientation(o, antipodal=False).CS is cs2


def test_orientation_hasNoNegation():
  with pytest.raises(TypeError):
    -eulerOri()


def test_orientation_reprNamesTheFrames():
  o = eulerOri(ti(), specimenFrame.rolling)
  m = orientation(rotation.rand(), al(), ti())
  assert repr(o).startswith("orientation (Ti → TD←RD↑)") and "misorientation (Al → Ti)" in repr(m)


a, b, c = 0.3, 0.7, 1.1


def test_orientation_byEulerWithFrames():
  cs, ss = al(), ss222()
  o = orientation.byEuler(a, b, c, "ZYZ", cs, ss)
  assert same(o, rotation.byEuler(a, b, c, "ZYZ")) and o.CS is cs and o.SS is ss
  assert same(orientation.byEuler(a, b, c, cs), rotation.byEuler(a, b, c)) and orientation.byEuler(a, b, c, cs).SS is specimenFrame.default
  assert orientation.byEuler(a, b, c, cs, cs, antipodal=True).antipodal


def test_orientation_byAxisAngle():
  cs, ss = al(), ss222()
  assert orientation.byAxisAngle(zvector, 0.5, cs, ss).CS is cs and same(orientation.byAxisAngle(zvector, 0.5, cs), rotation.byAxisAngle(zvector, 0.5))


def test_orientation_byMatrixRefusesAnImproperMatrix():
  cs = al()
  M = rotation.byEuler(a, b, c).matrix()
  assert same(orientation.byMatrix(M, cs), rotation.byEuler(a, b, c)) and orientation.byMatrix(M, cs).CS is cs
  with pytest.raises(ValueError):
    orientation.byMatrix(-M, cs)


def test_orientation_byRodriguesAndByHomochoric():
  cs = al()
  assert orientation.byRodrigues(vector3d(0, 0, 1), cs).CS is cs and same(orientation.byRodrigues([0, 0, 1], cs), rotation.byRodrigues([0, 0, 1]))
  assert same(orientation.byHomochoric(vector3d(0.1, 0.2, 0.3), cs), rotation.byHomochoric([0.1, 0.2, 0.3]))


def test_map_adoptsTheFramesOfFramedInputs():
  # CS from the crystal direction, SS from the framed vector
  cs = al()
  h = Miller(1, 0, 0, cs)
  v = vector3d(0, 1, 0, frame=specimenFrame.rolling)
  om = orientation.map(h, v)
  assert om.CS is cs and om.SS is specimenFrame.rolling and (om * h).isclose(v)
  assert orientation.map(h, xvector).SS is specimenFrame.default and orientation.map(xvector, xvector).CS.numSym() == 1


def test_map_ofTwoPairsWithFramesGiven():
  cs, ss = al(), ss222()
  om2 = orientation.map(Miller(1, 0, 0, cs), xvector, Miller(0, 1, 0, cs), vector3d(0, 1, 0), cs, ss)
  assert om2.CS is cs and om2.SS is ss and same(om2, rotation.id)
  assert orientation.map(xvector, vector3d(0, 1, 0), cs).CS is cs


def test_orientation_rand():
  cs, ss = al(), ss222()
  assert orientation.rand(7, cs).shape == (7,) and orientation.rand(7, cs).CS is cs and orientation.rand(cs, ss).SS is ss
  assert orientation.rand(2, 3, cs).shape == (2, 3) and orientation.rand(cs).shape == ()


def test_orientation_id():
  cs = al()
  assert orientation.id(cs).shape == () and orientation.id(3, cs).shape == (3,) and same(orientation.id(cs), rotation.id)


def test_orientation_nan():
  cs = al()
  assert np.all(np.isnan(orientation.nan(2, cs).data)) and orientation.nan(2, cs).CS is cs


def test_orientation_fitRecoversTheRotation():
  cs = al()
  u = vector3d.rand(20, rng=np.random.default_rng(1))
  r = rotation.rand(rng=np.random.default_rng(2))
  assert angle(orientation.fit(u, r * u, cs), r) < 1e-10 and orientation.fit(u, r * u, cs).CS is cs


def test_orientation_standardOrientations():
  cs, ss = al(), ss222()
  assert same(orientation.cube(cs), rotation.id) and orientation.cube(cs).CS is cs
  assert close(orientation.goss(cs, ss).Phi, 45 * degree) and orientation.goss(cs, ss).SS is ss
  assert close(orientation.brass(cs).phi1, 35 * degree) and close(orientation.SR(cs).phi2, 63 * degree)


def test_orientation_indexingKeepsTheFrame():
  cs = al()
  rots, os = alList()
  assert os.shape == (10,) and os.CS is cs
  assert isinstance(os[3], orientation) and same(os[3], rots[3]) and os[3].CS is cs
  assert isinstance(os[2:4], orientation) and os[2:4].size == 2 and os[2:4].CS is cs


def test_orientation_assignment():
  rots, os = alList()
  os[0] = os[5]
  assert same(os[0], rots[5])
  os[1] = rots[2]
  assert same(os[1], rots[2])


def test_orientation_assignmentRefusesOtherFrames():
  cs = al()
  rots, os = alList()
  with pytest.raises(ValueError):
    os[0] = orientation(rots[0], specimenFrame.rolling, cs)


def test_orientation_byEulerOfAList():
  e = orientation.byEuler(0.1 * np.arange(1, 6), 0.2, 0.3, al())
  assert e.size == 5 and close(Euler(e)[0][1], 0.2)


def test_inv_ofAListSwapsTheFrames():
  cs = al()
  _, os = alList()
  assert inv(os).CS is specimenFrame.default and inv(os).SS is cs


def test_cat_keepsTheFrame():
  # a Laue sibling is absorbed
  cs = al()
  rots, os = alList()
  assert cat(os, os).size == 20 and cat(os, os).CS is cs
  assert cat(os, orientation(rots, cs.Laue())).CS is cs
  assert cat(os[0], os[1]).size == 2 and cat(os[0], os[1]).CS is cs


def test_cat_refusesAnotherCrystalFrame():
  rots, os = alList()
  with pytest.raises(ValueError):
    cat(os, orientation(rots, crystalFrame("6/mmm", [3, 3, 5])))


def test_cat_isAntipodalOnlyWhenAllAre():
  cs = al()
  rots, _ = alList()
  assert cat(orientation(rots, cs, cs, antipodal=True), orientation(rots, cs, cs)).antipodal is False
  assert cat(orientation(rots, cs, cs, antipodal=True), orientation(rots, cs, cs, antipodal=True)).antipodal


def test_orientation_reshape():
  _, os = alList()
  assert os.reshape(2, 5).shape == (2, 5) and same(os.reshape(2, 5)[1, 4], os[9])


def test_angle_andEuler_ofAList():
  _, os = alList()
  a = angle(os)
  assert a.shape == (10,) and close(a[2], os[2].angle())
  assert Euler(os)[0].shape == (10,)


def test_orientation_reprOfAList():
  _, os = alList()
  assert "size: 10" in repr(os) and repr(os).startswith("orientation (Al → y↓→x)")


def test_orientation_float32():
  cs = al()
  rots, os = alList()
  assert os.astype(np.float32).CS is cs and os.astype(np.float32).dtype == np.float32
  assert orientation(rots.astype(np.float32), cs).dtype == np.float32


def productOri(CS=None, SS=None):
  return orientation.byEuler(0.3, 0.7, 1.1, CS or al(), SS or ss222())


def test_orientation_timesMillerIsASpecimenDirection():
  cs, ss = al(), ss222()
  o = productOri(cs, ss)
  h = Miller(1, 1, 0, cs)
  v = o * h
  assert v.frame is ss and v.dispStyle == "xyz" and close(o.matrix() @ h.data, v.data)
  assert (rotation.byEuler(0.3, 0.7, 1.1) * h).frame is cs


def test_inv_timesASpecimenDirectionIsMiller():
  cs = al()
  o = productOri(cs)
  back = inv(o) * (o * Miller(1, 1, 0, cs))
  assert back.frame is cs and back.dispStyle == "hkl" and close(back.hkl, [1, 1, 0])


def test_orientation_timesAFrameFreeVector():
  ss = ss222()
  assert (productOri(SS=ss) * vector3d(1, 2, 3)).frame is ss


def test_orientation_refusesVectorsOfOtherFrames():
  cs = al()
  o = productOri(cs)
  v = o * Miller(1, 1, 0, cs)
  with pytest.raises(ValueError):
    o * v                                          # a specimen direction on the crystal side
  with pytest.raises(ValueError):
    o * Miller(1, 0, 0, crystalFrame("6/mmm", [3, 3, 5]))


def test_orientation_composeWhenTheInnerFramesFit():
  # the result names the outer ones
  cs, ss = al(), ss222()
  o = productOri(cs, ss)
  o2 = orientation.byEuler(1.1, 0.7, 0.3, cs, ss)
  mori = inv(o) * o2
  assert mori.CS is cs and mori.SS is cs and mori.isMisorientation and same(mori, inv(rotation(o)) * rotation(o2))
  assert (o * inv(o2)).CS is ss and (o * inv(o2)).SS is ss


def test_orientation_refusesComposingWhenTheInnerFramesDiffer():
  cs, ss = al(), ss222()
  with pytest.raises(ValueError):
    productOri(cs, ss) * orientation.byEuler(1.1, 0.7, 0.3, cs, ss)


def test_rotation_onTheLeftTurnsTheSpecimenAndDropsItsGroup():
  # unless it is an element
  cs, ss = al(), ss222()
  o = productOri(cs, ss)
  r = rotation.byAxisAngle(vector3d(1, 2, 3), 0.5)
  assert (r * o).SS.numSym() == 1 and (r * o).SS.isAligned(ss) and (r * o).CS is cs and same(r * o, r * rotation(o))
  assert (rotation.byAxisAngle(zvector, pi) * o).SS is ss


def test_rotation_onTheRightTurnsTheCrystalAndDropsItsGroup():
  cs, ss = al(), ss222()
  o = productOri(cs, ss)
  r = rotation.byAxisAngle(vector3d(1, 2, 3), 0.5)
  assert (o * r).CS.numSym() == 1 and (o * r).SS is ss and same(o * r, rotation(o) * r)
  assert (o * rotation.byAxisAngle(zvector, pi / 2)).CS is cs


def test_orientation_refusesTheInversionOnTheLeft():
  with pytest.raises(TypeError):
    rotation.inversion * productOri()


def test_orientation_timesAScalarIsAQuaternion():
  o = productOri()
  assert isinstance(2 * o, quaternion) and not isinstance(2 * o, orientation)


def test_orientation_alignedCrystalFrameFits():
  ss = ss222()
  cs2 = crystalFrame("432", [4.05, 4.05, 4.05], mineral="Al")
  assert (productOri(SS=ss) * Miller(1, 0, 0, cs2)).frame is ss


def test_orientation_compatibleCrystalFrameIsAbsorbed():
  ss = ss222()
  csA = crystalFrame("121", [2, 3, 4], np.array([90, 105, 90]) * degree)
  csB = crystalFrame("112", [4, 2, 3], np.array([90, 90, 105]) * degree)
  oA = orientation.byEuler(0.3, 0.7, 1.1, csA, ss)
  hB = Miller(1, 0, 0, csB)
  assert (oA * hB).frame is ss
  assert (oA * hB) == (oA * Miller(vector3d(rotation.byMatrix(csB.transformationMatrix(csA)) * hB), csA))


def test_orientation_productsOfListsBroadcast():
  cs, ss = al(), ss222()
  os = orientation.rand(5, cs, ss, rng=np.random.default_rng(0))
  hs = Miller([[1, 0, 0], [1, 1, 0]], cs)
  assert (os[:, None] * hs[None, :]).shape == (5, 2)
  assert (inv(os[:, None]) * os[None, :]).shape == (5, 5)


def test_projectIntoFundamentalRegion_antipodal():
  # MATLAB's 'antipodal': the projection with grain exchange, the misorientation or its inverse
  cs = crystalFrame("m-3m")
  m = orientation.rand(50, cs, cs, rng=np.random.default_rng(1))
  p = m.projectIntoFundamentalRegion(antipodal=True)
  q = orientation._fromRotation(m.data, (cs, cs), True).projectIntoFundamentalRegion()
  assert same(p, q) and not m.antipodal
  # each projected one is the misorientation or its inverse up to symmetry
  assert np.all(np.minimum(angle(p, m), angle(p, inv(m))) < 1e-6)


def test_notIID_isOffByDefault():
  cs = crystalFrame("m-3m")
  assert not orientation(rotation.rand(6), cs).notIID
  assert not orientation(rotation.rand(2), cs).notIID


def test_notIID_isCarried():
  cs = crystalFrame("m-3m")
  o = orientation(rotation.rand(6), cs)
  o.notIID = True
  # indexing, reshaping, copies, the copy constructor, products and lists keep the mark
  assert o[1:3].notIID and o.flatten().notIID and o.copy().notIID and orientation(o).notIID
  assert (o * rotation.rand()).notIID and (inv(o[0]) * o).notIID
  assert cat(orientation(rotation.rand(2), cs), o).notIID


def test_defaultFrames_fillsTheTrivialCrystalAndTheDefaultSpecimen():
  from mtex.geometry.orientation import _defaultFrames
  cs = crystalFrame("m-3m")
  assert _defaultFrames([cs]) == (cs, specimenFrame.default) and _defaultFrames([], "432")[0].pointGroup == "432"
