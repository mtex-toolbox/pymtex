# ODF sections (docs/plans/plotting.md S2), the geometry only: the numbers of
# ../mtexJulia/test/sections.jl E1 to E7
import numpy as np
import pytest

from mtex import (alphaSections, axisAngleSections, crystalFrame, degree, eulerRegion, fundamentalRegion, gammaSections,
                  newODFSectionPlot, orientation, pfSections, phi1Sections, phi2Sections, rotation, sigmaSections,
                  specimenFrame, sphericalRegion, vector3d)
from mtex.plotting.sections import sectionIndex, sectionTolerance

rng = np.random.default_rng(4)


# the cubic crystal and the orthorhombic specimen most tests section
def cubicPair():
  return crystalFrame("m-3m"), specimenFrame("222")


# the phi2 sections at 0, 15, ..., 75 degree
def phi2Six():
  cs, ss = cubicPair()
  return phi2Sections(cs, ss, phi2=np.arange(0, 90, 15) * degree)


def test_eulerRegion_isTheBoxOfTheGroups():
  cs, ss, hexa, triv = crystalFrame("m-3m"), specimenFrame("222"), crystalFrame("6/mmm"), specimenFrame("1")
  assert np.allclose(np.array(eulerRegion(cs, ss)) / degree, (90, 90, 90))
  assert np.allclose(np.array(eulerRegion(hexa, ss)) / degree, (90, 90, 60))
  assert np.allclose(np.array(eulerRegion(cs, triv)) / degree, (360, 90, 90))
  assert np.allclose(np.array(eulerRegion(crystalFrame("1"), triv)) / degree, (360, 180, 360))


def test_eulerRegion_carriesEveryOrientationAtLeastOnce():
  for cname in ("1", "2/m", "222", "4", "-3m", "622", "m-3m", "mm2"):
    for sname in ("1", "2/m", "222", "mmm"):
      frA, frB = crystalFrame(cname), specimenFrame(sname)
      p1, pPhi, p2 = eulerRegion(frA, frB)
      for _ in range(20):
        es = orientation(rotation.rand(rng=rng), frA, frB).symmetrise()
        a, b, c = es.Euler("Bunge")
        assert np.any((a <= p1 + 1e-9) & (b <= pPhi + 1e-9) & (c <= p2 + 1e-9))


def test_phi2Sections_carryTheAnglesLabelsAndFrames():
  cs, ss = cubicPair()
  oS = phi2Six()
  assert oS.numSections == 6 and np.allclose(oS.phi2, oS.values)
  assert oS.axisLabels == ("φ1", "Φ") and oS.sectionLabel(1) == "φ2 = 15°"
  assert oS.CS is cs and oS.SS is ss and oS.frameA is cs


def test_phi2Sections_sectionData_placesTheEquivalentsAndGivesThemBack():
  cs, ss = cubicPair()
  oS = phi2Six()
  o = orientation.byEuler(35 * degree, 40 * degree, 30 * degree, cs, ss)
  vs, sec, src = oS.sectionData(o)
  assert np.all(src == 0)
  # the equivalent that sits at phi2 = 30 lands in section three at (Phi, phi1)
  k = [i for i in range(len(sec)) if sec[i] == 2 and abs(vs[i].theta - 40 * degree) < 1e-9]
  assert k and abs(vs[k[0]].rho - 35 * degree) < 1e-9
  # every point of every section gives back an orientation of the list
  for i in range(len(sec)):
    assert oS.orientationsAt(sec[i], vs[i]).angle(o) <= oS.tol + 1e-9


def test_phi1Sections_readTheBoxTheOtherWayRoundAndTakeAnglesSetAfterwards():
  cs, ss = cubicPair()
  oS1 = phi1Sections(cs, ss)
  assert oS1.axisLabels == ("φ2", "Φ") and np.isclose(oS1.values[-1], 90 * degree * 5 / 6)
  # the angles set afterwards, as MATLAB's sP.phi1 = (0:10:90)*degree
  oS1.phi1 = np.arange(0, 91, 10) * degree
  assert oS1.numSections == 10 and np.isclose(oS1.tol, 10 * degree / 2.1)


def test_gammaAndAlphaSections_labelTheirAxes():
  cs, ss = cubicPair()
  assert gammaSections(cs, ss).axisLabels == ("α", "β") and alphaSections(cs, ss).axisLabels == ("γ", "β")


def test_phi2Sections_chartIsThePlainBoxInDegrees():
  c = phi2Six().sectionCharts(0)
  assert len(c) == 1 and c[0].isPlain and c[0].bounds() == (0.0, 90.0, 0.0, 90.0)


def test_phi2Sections_refuseSigma():
  cs, ss = cubicPair()
  with pytest.raises(TypeError):
    phi2Sections(cs, ss, sigma=[0.0])


def test_phi2Sections_degenerateMiddleAngleFallsInEverySection():
  cs, ss = cubicPair()
  oS = phi2Six()
  for Phi in (0.0, np.pi):
    o = orientation.byEuler(50 * degree, Phi, 20 * degree, cs, ss)
    vs, sec, _ = oS.sectionData(o)
    assert sorted(set(sec.tolist())) == list(range(6))
    for i in range(len(sec)):
      assert oS.orientationsAt(sec[i], vs[i]).angle(o) < 1e-4


def test_sigmaSections_periodLabelAndCharts():
  cs, ss, hexa, triv = crystalFrame("m-3m"), specimenFrame("222"), crystalFrame("6/mmm"), specimenFrame("1")
  oS = sigmaSections(cs, ss)
  assert oS.numSections == 6 and np.isclose(oS.period, 90 * degree) and np.isclose(oS.sigma[1], 15 * degree)
  assert oS.sectionLabel(1) == "σ = 15°"
  assert np.isclose(sigmaSections(hexa, ss).period, 60 * degree)
  assert len(sigmaSections(crystalFrame("mmm"), triv).sectionCharts(0)) == 1
  assert len(sigmaSections(crystalFrame("1"), triv).sectionCharts(0)) == 2
  assert sigmaSections(crystalFrame("mmm"), triv).sectionCharts(0)[0].upper


def test_sigmaSections_coordinateIsPhi1MinusPhi2():
  triv = specimenFrame("1")
  # the coordinate against phi1 - phi2, where c* is z and a is x
  plain = sigmaSections(crystalFrame("1"), triv, sigma=[0.0])
  for p1, Phi, p2 in ((35, 40, 30), (100, 70, 20), (10, 5, 355)):
    o = orientation.byEuler(p1 * degree, Phi * degree, p2 * degree, crystalFrame("1"), triv)
    v = vector3d((rotation(o) * vector3d(plain.h1.data)).data)
    w = vector3d((rotation(o) * vector3d(plain.h2.data)).data)
    om = plain.referenceField(v).angle(w, N=v)
    assert np.mod(om + (p1 - p2) * degree, 2 * np.pi) < 1e-9 or abs(np.mod(om + (p1 - p2) * degree, 2 * np.pi) - 2 * np.pi) < 1e-9


def test_sigmaSections_pointOfASectionGivesBackTheOrientation():
  cs, ss = cubicPair()
  oS = sigmaSections(cs, ss)
  o = orientation.byEuler(30 * degree, 50 * degree, 10 * degree, cs, ss)
  vs, sec, _ = oS.sectionData(o)
  assert vs.size > 0 and vs.frame is ss
  for i in range(len(sec)):
    assert oS.orientationsAt(sec[i], vs[i]).angle(o) <= oS.tol + 1e-9


def test_pfSections_h2EntersOnlyThroughItsPartPerpendicularToH1():
  triv = specimenFrame("1")
  mono = pfSections(crystalFrame("2/m", [1.0, 2.0, 3.0], [90 * degree, 100 * degree, 90 * degree]), triv)
  assert abs(mono.h1.data @ mono.h2.data) < 1e-12
  om = orientation.byEuler(20 * degree, 60 * degree, 40 * degree, mono.frameA, triv)
  vs, sec, _ = mono.sectionData(om)
  for i in range(len(sec)):
    assert mono.orientationsAt(sec[i], vs[i]).angle(om) <= mono.tol + 1e-9


def test_sigmaSections_sectionFieldIsTangent():
  cs, ss = cubicPair()
  # the field of a section, at an equispaced grid of the region
  vs, field = sigmaSections(cs, ss).sectionField(1)
  assert vs.size > 5 and field.size == vs.size and np.all(np.abs(np.sum(vs.data * field.data, axis=-1)) < 1e-9)


def test_pfSections_refuseParallelDirections():
  cs, ss = cubicPair()
  with pytest.raises(ValueError):
    pfSections(cs, ss, h1=vector3d(0, 0, 1), h2=vector3d(0, 0, 2))


def test_axisSector_isTheAxesOfTheRegionAtThatAngle():
  for a, b in (("m-3m", "m-3m"), ("6/mmm", "6/mmm"), ("m-3m", "1"), ("222", "222")):
    frA = crystalFrame(a)
    frB = specimenFrame("1") if b == "1" else crystalFrame(b)
    oR = fundamentalRegion(frA, frB)
    wmax = oR.maxAngle()
    for w in (5 * degree, 20 * degree, 45 * degree):
      if w > wmax:
        continue
      sR = oR.axisSector(w)
      v = vector3d.rand(200, rng=rng)
      inRegion = oR.checkInside(rotation.byAxisAngle(v, w), tol=1e-8)
      assert np.array_equal(inRegion, sR.checkInside(v, tol=1e-8))
    assert oR.axisSector(0).volume(2 * degree) >= oR.axisSector(wmax).volume(2 * degree)


def test_thetaIntervals_capTakenOutLeavesTheMeridianInTwoPieces():
  out = sphericalRegion(vector3d(-1, 0, 0), [-np.cos(40 * degree)])
  segs = out.thetaIntervals(0.0)
  assert len(segs) == 2 and np.isclose(segs[0][1], 50 * degree) and np.isclose(segs[1][0], 130 * degree)


def test_thetaIntervals_ofAnAxisAngleChartCoverTheRegion():
  oS = axisAngleSections(crystalFrame("m-3m"), specimenFrame("222"), angles=[50 * degree])
  sR = oS.sectionCharts(0)[0].sR
  for v in vector3d.rand(300, rng=rng):
    if not sR.checkInside(v, tol=-1e-6):
      continue
    assert any(lo - 1e-6 <= v.theta <= hi + 1e-6 for lo, hi in sR.thetaIntervals(float(v.rho)))


def test_axisAngleSections_anglesLabelsSectorsAndScale():
  cs, ss = cubicPair()
  oS = axisAngleSections(cs, ss, angles=np.arange(10, 91, 20) * degree)
  wmax = oS.oR.maxAngle()
  assert np.all(oS.angles <= wmax) and oS.sectionLabel(0) == "ω = 10°"
  assert oS.jointCS.numSym() == 4 and len(oS.axesSectors) == oS.numSections
  assert np.isclose(oS.sectionScale(0), np.sin(np.max(oS.values) / 2) / np.sin(5 * degree))
  assert axisAngleSections(cs, ss, angles=np.arange(10, 91, 20) * degree, volumeScaling=False).sectionScale(0) == 1
  assert oS.outerSector is oS.jointSector and len(oS.sectionCharts(0)) == 1


def test_axisAngleSections_sectionDataGivesBackTheOrientation():
  cs, ss = cubicPair()
  oS = axisAngleSections(cs, ss, angles=np.arange(10, 91, 20) * degree)
  o = orientation.byAxisAngle(vector3d(1, 2, 3), 30 * degree, cs, ss)
  vs, sec, src = oS.sectionData(o)
  assert vs.size == 1 and np.isclose(oS.values[sec[0]], 30 * degree) and src[0] == 0
  assert oS.orientationsAt(sec[0], vs).angle(o)[0] < 1e-9


def test_axisAngleSections_countPutsSectionsInTheMiddleOfEqualBands():
  cs, ss = cubicPair()
  n = axisAngleSections(cs, ss, sections=4)
  assert n.numSections == 4 and np.isclose(n.values[0], n.oR.maxAngle() / 8)


def test_axisAngleSections_refuseAnAngleBeyondTheRegion():
  cs, ss = cubicPair()
  with pytest.raises(ValueError):
    axisAngleSections(cs, ss, angles=[170 * degree])


def test_axisAngleSections_cubicAndHexagonalShareTheRotationsOf222():
  m = axisAngleSections(crystalFrame("m-3m"), crystalFrame("6/mmm"), sections=3)
  assert m.jointCS.numSym() == 4


def test_sectionIndex_isTheSectionACoordinateFallsIn():
  vals = np.arange(0, 90, 15) * degree
  assert sectionIndex(16 * degree, vals, 5 * degree, 90 * degree) == 1
  assert sectionIndex(10 * degree, vals, 2 * degree, 90 * degree) == -1
  assert sectionIndex(89 * degree, vals, 5 * degree, 90 * degree) == 0
  assert sectionIndex(89 * degree, vals, 5 * degree, np.inf) == -1
  assert np.array_equal(sectionIndex([16 * degree, 40 * degree], vals, 5 * degree, 90 * degree), [1, -1])


def test_sectionTolerance_isTheSpacingOver2point1CappedByTheGiven():
  vals = np.arange(0, 90, 15) * degree
  assert np.isclose(sectionTolerance(np.arange(0, 9, 4) * degree, 5 * degree), 4 * degree / 2.1)
  assert sectionTolerance(vals, 5 * degree) == 5 * degree


def test_newODFSectionPlot_choosesTheSectionsByTheFramesAndName():
  cs, ss = cubicPair()
  assert isinstance(newODFSectionPlot(cs, ss), phi2Sections)
  assert isinstance(newODFSectionPlot(cs, crystalFrame("6/mmm")), axisAngleSections)
  assert isinstance(newODFSectionPlot(cs, ss, "sigma", sections=4), sigmaSections)
  assert newODFSectionPlot(cs, ss, "sigma", sections=4).numSections == 4
  assert isinstance(newODFSectionPlot(cs, ss, "axisAngle", volumeScaling=False), axisAngleSections)


def test_newODFSectionPlot_passesSectionsThrough():
  cs, ss = cubicPair()
  oS = sigmaSections(cs, ss)
  assert newODFSectionPlot(cs, ss, oS) is oS
  assert "sigmaSections" in repr(oS)


def test_newODFSectionPlot_refusesAnUnknownName():
  cs, ss = cubicPair()
  with pytest.raises(ValueError):
    newODFSectionPlot(cs, ss, "beta")


def test_values_on_the_nodes_of_makeGrid():
  import matplotlib
  matplotlib.use("Agg")
  from mtex import axisAngleSections, orientation
  cs = crystalFrame("622")
  sP = axisAngleSections(cs, cs)
  ori = sP.makeGrid(resolution=10 * degree)
  assert isinstance(ori, orientation) and ori.ndim == 1
  hs = sP.plot(ori.angle(), smooth=True)
  assert any(h is not None for h in hs)


def test_eulerSections_aDegenerateOrientationFallsIntoEverySection():
  oS = phi2Sections(crystalFrame("m-3m"), specimenFrame.default, sections=4)
  _, sec, src = oS.sectionData(orientation.byEuler(10 * degree, 0, 20 * degree, "Bunge", oS.frameA, oS.frameB))
  assert set(sec.tolist()) == set(range(4)) and set(src.tolist()) == {0}
