# orientations from directions, the relationships and their variants
import numpy as np
import pytest

from mtex import (orientation, rotation, crystalFrame, specimenFrame, Miller, vector3d, xvector, zvector, degree,
                  angle, symmetrise, variants)


def test_by_miller_and_the_texture_components():
  cs = crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Al")
  ori = orientation.map(Miller(1, 0, 0, cs), vector3d(0, 0, 1), Miller(0, 1, 0, cs, "uvw"), vector3d(0, 1, 0))
  assert ori.CS is cs and angle(ori * Miller(1, 0, 0, cs), zvector) < 1e-10
  fr = specimenFrame("lab", axesNames=("A", "B", "C"), how2plot="y←↑x")
  assert orientation.map(Miller(1, 0, 0, cs), vector3d(1, 0, 0, frame=fr)).SS is fr
  o3 = orientation.byMiller([1, 1, 2], [-1, 1, -1], cs)
  assert angle(o3 * Miller(1, 1, 2, cs), zvector) < 1e-10 and o3.CS is cs
  assert np.min(angle(o3 * symmetrise(Miller(-1, 1, -1, cs, "uvw")), xvector)) < 1e-10
  assert orientation.byMiller(Miller(1, 1, 2, cs), Miller(-1, 1, -1, cs, "uvw"), specimenFrame("222")).SS.numSym() == 4
  # a single plane named group-free still belongs to the phase
  assert orientation.byMiller(Miller("(112)", cs), Miller("[-11-1]", cs)).CS is cs
  assert angle(orientation.copper(cs), orientation.byEuler(90 * degree, 35 * degree, 45 * degree, cs)) < 1 * degree
  assert angle(orientation.copper2(cs), orientation.byEuler(270 * degree, 35 * degree, 45 * degree, cs)) < 1 * degree
  assert angle(orientation.goss(cs), orientation.byEuler(0, 45 * degree, 0, cs)) < 1e-10
  with pytest.warns(UserWarning):
    orientation.byMiller([1, 0, 0], [1, 1, 1], cs)


def test_relationships_and_variants():
  csG = crystalFrame("m-3m", [3.6, 3.6, 3.6], mineral="gamma")
  csA = crystalFrame("m-3m", [2.87, 2.87, 2.87], mineral="alpha")
  ks = orientation.KurdjumovSachs(csG, csA)
  assert ks.isMisorientation and ks.CS is csG and ks.SS is csA
  assert angle(ks * Miller(1, 1, 1, csG), Miller(0, 1, 1, csA)) < 1e-10
  assert angle(ks * Miller(-1, 0, 1, csG, "uvw"), Miller(-1, -1, 1, csA, "uvw")) < 1e-10
  assert variants(ks).size == 24 and variants(orientation.NishiyamaWassermann(csG, csA)).size == 12 and variants(orientation.Bain(csG, csA)).size == 3
  assert variants(ks, "parent").size == 24 and ks.variants(parent=True).size == 24 and variants(ks, "child").size == 24
  assert variants(orientation.Pitsch(csG, csA)).size == 12 and variants(orientation.GreningerTrojano(csG, csA)).size == 24
  # the child orientations of one parent
  mag = crystalFrame("m-3m", [8.396, 8.396, 8.396], mineral="Magnetite")
  hem = crystalFrame("-3m", [5.038, 5.038, 13.772], mineral="Hematite")
  m2h = orientation.map(Miller(1, 1, 1, mag), Miller(0, 0, 0, 1, hem), Miller(-1, 0, 1, mag), Miller(1, 0, -1, 0, hem))
  par = orientation.byEuler(20 * degree, 30 * degree, 40 * degree, mag)
  kids = variants(m2h, par)
  assert kids.size == 8 and kids.CS is hem and kids.SS is par.SS
  ts = par * symmetrise(Miller(1, 1, 1, mag), unique=True)
  for k in range(8):
    assert np.min(angle(kids[k] * Miller(0, 0, 0, 1, hem), ts)) < 1e-8
  assert variants(m2h, par, [1, 2]).size == 2 and np.all(variants(m2h, par, [1, 2]) == kids[:2])
  assert variants(m2h, orientation.rand(3, mag)).size == 24
  cs2 = crystalFrame("6/mmm", [2.95, 2.95, 4.68], mineral="beta")
  assert variants(orientation.Burgers(csG, cs2)).size == 12 and orientation.Burgers(cs2, csG).CS is cs2
