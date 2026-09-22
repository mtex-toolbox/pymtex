# reading pole figure files: the Dubna, Siemens, Philips, Bruker and Aachen formats on the
# fixtures, the Geesthacht table and the generic columns on the MTEX data folder
# (docs/plans/io.md I6)
import os

import numpy as np
import pytest

from mtex import Miller, PoleFigure, crystalFrame, degree, loadPoleFigure, mean, specimenFrame
from mtex.io.loadpolefigure import DubnaGrid, loadPoleFigure_generic, string2Miller

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "polefigure")
LOCAL = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data", "PoleFigure")
QUARTZ = crystalFrame("321", [4.916, 4.916, 5.4054], mineral="Quartz")


def fixture(name):
  return os.path.join(FIXTURES, name)


def test_the_reflection_in_a_name():
  assert np.allclose(string2Miller("Q(10-10)_amp.cnv", QUARTZ).hkil, [1, 0, -1, 0])
  h = string2Miller("Q(10-11)(01-11)_amp.cnv", QUARTZ)
  assert h.size == 2 and np.allclose(h[1].hkil, [0, 1, -1, 1])
  # three digits, without a sign before with
  assert np.allclose(string2Miller("seifert-111.nja").hkl, [1, 1, 1])
  assert np.allclose(string2Miller("gt9104.ptx", crystalFrame("mmm")).hkl, [1, 0, 4])
  assert np.allclose(string2Miller("nothing here").hkl, [1, 0, 0])


def test_the_dubna_files():
  pf = loadPoleFigure(fixture("Q(10-10)_amp.cnv"), QUARTZ)
  assert pf.numPF == 1 and pf.CS is QUARTZ and pf.antipodal and pf.allI[0].shape == (72, 19)
  assert np.isclose(pf.max()[0], 89.8) and np.isclose(pf.resolution, 5 * degree)
  assert np.allclose(pf.allH[0].hkil, [1, 0, -1, 0])
  # the goniometer's rings start at their own azimuths
  r = DubnaGrid(19)
  assert r.shape == (72, 19) and r.antipodal and np.isclose(r.theta[0, 3], 15 * degree)
  assert np.isclose(np.mod(r.rho[1, 1], 2 * np.pi), (272.5 + 360 / 71) * degree)
  # a superposed reflection from its name, the coefficients given or the mean
  p3 = loadPoleFigure(fixture("Q(10-11)(01-11)_amp.cnv"), QUARTZ, superposition=[0.52, 1.23])
  assert p3.allH[0].size == 2 and np.allclose(p3.c[0], [0.52, 1.23]) and np.isclose(p3.max()[0], 1360)
  assert np.allclose(loadPoleFigure(fixture("Q(10-11)(01-11)_amp.cnv"), QUARTZ).c[0], [0.5, 0.5])
  # several files, the directions and coefficients handed over
  both = PoleFigure.load([fixture("Q(10-10)_amp.cnv"), fixture("Q(10-11)(01-11)_amp.cnv")],
                         [Miller(1, 0, -1, 0, QUARTZ), Miller([0, 1], [1, 0], [-1, -1], [1, 1], QUARTZ)], QUARTZ,
                         superposition=[1, [0.52, 1.23]])
  assert both.numPF == 2 and both.allH[1].size == 2 and np.allclose(both.c[1], [0.52, 1.23])
  assert np.allclose(loadPoleFigure(fixture("Q(10-10)_amp.cnv"), Miller(0, 0, 0, 1, QUARTZ)).allH[0].hkil, [0, 0, 0, 1])
  with pytest.raises(ValueError):
    loadPoleFigure(fixture("Q(10-10)_amp.cnv"), Miller([1, 1], [0, 0], [-1, -1], [0, 1], QUARTZ), QUARTZ)
  with pytest.raises(ValueError):
    loadPoleFigure(fixture("Q(10-10)_amp.cnv"), interface="mars")
  assert loadPoleFigure(fixture("Q(10-10)_amp.cnv"), QUARTZ, specimenFrame("222")).SS.numSym() == 4


def test_the_siemens_dump():
  pf = loadPoleFigure(fixture("gt9104.ptx"), crystalFrame("mmm"))
  assert pf.numPF == 1 and pf.allI[0].shape == (72, 17) and np.allclose(pf.allH[0].hkl, [1, 0, 4])
  r = pf.allR[0]
  assert np.isclose(r.theta[0, 16], 80 * degree) and np.isclose(r.rho[1, 1], 5 * degree)
  assert np.isclose(pf.allI[0][0, 0], 0.0895) and np.isclose(pf.resolution, 5 * degree)


def test_the_philips_binary():
  pf = loadPoleFigure(fixture("AcAR110p.rw1"), crystalFrame("m-3m"))
  assert pf.numPF == 1 and pf.allI[0].shape == (73, 18) and np.allclose(pf.allH[0].hkl, [1, 1, 1])
  assert np.isclose(pf.allR[0].theta[0, 17], 85 * degree) and pf.antipodal
  assert np.all(pf.allI[0] >= 0) and pf.max()[0] > 1000
  assert loadPoleFigure(fixture("AcAR110p.rw1"), interface="rw1").numPF == 1


def test_the_bruker_text():
  pf = loadPoleFigure(fixture("ZnCuTi_Wal_50_5x5_PF_002_R.UXD"), crystalFrame("6/mmm"))
  assert pf.numPF == 1 and pf.allI[0].size == 16 * 72 and np.allclose(pf.allH[0].hkil, [0, 0, 0, 2])
  assert np.isclose(pf.allR[0].theta.max(), 75 * degree) and np.isclose(pf.allR[0].rho[73], 5 * degree)
  assert np.isclose(pf.allI[0][0], 351.67) and np.isclose(pf.resolution, 5 * degree)


def test_the_aachen_file():
  pf = loadPoleFigure(fixture("aachen_exp.EXP"))
  assert pf.numPF == 4 and np.allclose(pf.h.hkl, [[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]])
  assert pf.allI[0].shape == (72, 17) and pf.CS.pointGroup == "m-3m" and pf.antipodal
  assert np.isclose(pf.allI[0][0, 0], 76) and np.isclose(pf.allI[0][1, 0], 100)
  assert np.isclose(pf.allR[0].theta[0, 0], 5 * degree) and np.isclose(pf.allR[0].theta[0, 16], 85 * degree)


@pytest.mark.skipif(not os.path.isdir(LOCAL), reason="the MTEX data folder is not here")
def test_the_geesthacht_table_and_the_generic_columns():
  pf = loadPoleFigure(os.path.join(LOCAL, "geesthacht", "ST42-104-110.dat"), crystalFrame("m-3m"))
  assert pf.numPF == 4 and [I.size for I in pf.allI] == [679, 16, 679, 16]
  assert np.isclose(pf.allI[0][0], 520) and np.isclose(pf.allR[0].theta[0], 0)
  # the generic columns of the Seifert file, the polar angle in degrees; the block found
  # by itself or the header lines counted
  fname = os.path.join(LOCAL, "nja", "seifert-111.nja")
  g = loadPoleFigure_generic(fname, crystalFrame("m-3m"), columnNames=["polar angle", "azimuth angle", "intensity"])
  assert loadPoleFigure_generic(fname, header=20, columnNames=["theta", "rho", "intensity"]).allI[0].size == 2121
  assert g.numPF == 1 and g.allI[0].size == 2121 and np.allclose(g.allH[0].hkl, [1, 1, 1])
  assert np.isclose(g.allI[0][0], 74.9) and np.isclose(g.allR[0].theta.max(), 70 * degree)
  assert loadPoleFigure(fname, Miller(1, 1, 1, crystalFrame("m-3m")), header=21, columnNames=["polar angle", "azimuth angle", "intensity"], columns=[0, 1, 2]).allI[0].size == 2120
  with pytest.raises(ValueError, match="columnNames"):
    loadPoleFigure(fname, header=21)
  assert mean(g)[0] > 0
