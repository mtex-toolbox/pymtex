# the vendor pole figure formats, on the fixtures of the Julia port and the numbers its
# tests pin (test/polefigureformats.jl there, several of them against MATLAB)
import os
import warnings

import numpy as np
import pytest

from mtex import Miller, crystalFrame, degree, loadPoleFigure
from mtex.io import polefigureformats as F

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "polefigure")
CUB = crystalFrame("m-3m")


def load(name, *args, **kwargs):
  return loadPoleFigure(os.path.join(FIXTURES, name), *args, **kwargs)


def means(pf):
  return [round(float(np.mean(I)), 4) for I in pf.allI]


def thetaMax(pf, k=0):
  return np.degrees(pf.allR[k].theta).max()


def indices(pf, k=0):
  return np.round(pf.allH[k].hkl).astype(int).reshape(-1).tolist()


def test_LaboTEX_states_its_structure_and_one_ladder_per_pole_figure():
  pf = load("LaboTEX.epf")
  assert pf.numPF == 3 and pf.CS.sym.pointGroup == "432" and all(I.shape == (72, 16) for I in pf.allI)
  assert [np.round(h.hkl).astype(int).reshape(-1).tolist() for h in pf.allH] == [[1, 1, 1], [1, 0, 0], [1, 1, 0]]
  assert np.isclose(thetaMax(pf), 75) and (pf.allI[0].min(), pf.allI[0].max()) == (19, 15182)
  assert means(pf)[0] == 1021.3556 and means(pf)[2] == 372.0465
  ppf = load("LaboTEX.PPF")
  assert all(I.shape == (72, 17) for I in ppf.allI) and np.isclose(thetaMax(ppf), 80) and means(ppf)[0] == 57.1511
  # a crystal frame the caller passes overrules the structure code
  assert load("LaboTEX.epf", CUB).CS is CUB


def test_Popla_writes_its_header_and_counts_in_fixed_fields():
  pf = load("popla.epf", CUB)
  assert pf.numPF == 4 and all(I.shape == (72, 17) for I in pf.allI)
  assert [np.round(h.hkl).astype(int).reshape(-1).tolist() for h in pf.allH] == [[1, 0, 3], [1, 1, 0], [1, 1, 2], [2, 0, 0]]
  assert np.isclose(pf.allR[0].theta.min(), 0) and np.isclose(thetaMax(pf), 80)
  # the header's scale is a divisor, so the counts are the densities it means
  assert means(pf)[0] == 1.0625 and means(pf)[3] == 1.0412
  # a row may carry a nineteenth field the format does not have
  assert F.fixedFields("   1   2   3   4", 4, 18) == [1, 2, 3, 4] and F.fixedFields("   1   2   3   4", 4, 2) == [1, 2]
  gpf = load("popla.GPF", CUB)
  assert gpf.numPF == 6 and all(I.shape == (72, 19) for I in gpf.allI) and np.isclose(thetaMax(gpf), 90) and means(gpf)[0] == 0.7254


def test_Aachen_states_the_format_of_its_own_block():
  pf = load("aachen.txt")
  assert pf.numPF == 7 and all(I.shape == (72, 19) and np.all(np.isfinite(I)) for I in pf.allI)
  assert np.allclose(pf.allH[1].hkil, [1, 1, -2, 0]) and np.allclose(pf.allH[2].hkil, [1, 0, -1, 4])
  assert (pf.allI[0].min(), pf.allI[0].max()) == (0.1, 5.053) and means(pf)[0] == 1.7273
  # the five character symbol field names the point group as far as it names one
  assert F.latticeFromSymbol(" D3He") is not None and F.latticeFromSymbol("     ") is None
  # a header without its Fortran format is one of eighteen fields of four
  assert F.isAachenHeader("104  D3    5.00 5.0075.00    0    0") and not F.isAachenHeader(" 875 897 897 912 901 891 950 964 934")


def test_the_Aachen_experimental_file_one_ring_at_a_time():
  pf = load("aachen_exp.EXP", CUB)
  assert pf.numPF == 4 and all(I.shape == (72, 17) for I in pf.allI)
  assert np.isclose(np.degrees(pf.allR[0].theta).min(), 5) and np.isclose(thetaMax(pf), 85)
  # every ring ends in the polar angle of the next, written with a minus; no intensity
  assert pf.allI[0].min() == 11 and means(pf) == [687.1936, 362.9444, 281.1283, 288.9853]


def test_a_ladder_of_twenty_numbers_and_one_on_its_own_named_lines():
  a = load("EMSE102.ana", CUB)
  assert a.allI[0].shape == (800, 27) and indices(a) == [1, 0, 2] and means(a) == [0.0618]
  assert np.isclose(np.degrees(a.allR[0].theta).min(), 50.65) and np.isclose(thetaMax(a), 62.35)
  p = load("philips-111.txt", CUB)
  assert p.allI[0].shape == (360, 81) and indices(p) == [1, 1, 1] and np.isclose(thetaMax(p), 80) and means(p) == [184.7031]
  # the caller's direction replaces what the name says
  assert indices(load("EMSE102.ana", Miller(1, 1, 1, CUB), CUB)) == [1, 1, 1]


def test_the_formats_whose_grid_is_a_constant_or_a_count_of_their_own():
  plf = load("Queens_alu.plf", CUB)
  assert plf.numPF == 4 and all(I.shape == (90, 17) for I in plf.allI) and means(plf) == [0.6296, 0.7914, 0.9048, 0.9797]
  ibm = load("IBM.ibm", CUB)
  assert ibm.numPF == 3 and all(I.shape == (72, 19) for I in ibm.allI) and means(ibm) == [414.182, 761.6155, 136.5892]
  slc = load("slc110.slc", CUB)
  assert slc.allI[0].shape == (36, 360) and means(slc) == [1059.0944]
  bt = load("beartex.XPa")
  assert bt.numPF == 7 and bt.CS.sym.pointGroup == "312" and np.allclose(bt.allH[0].hkil, [1, 1, -2, 2])
  assert means(bt) == [976.3114, 585.5731, 487.8406, 524.5943, 421.1864, 1159.7383, 196.3289]
  x = load("Rigaku_xrd_111.asc", CUB)
  assert x.allI[0].shape == (145, 31) and means(x) == [626.8652] and np.isclose(thetaMax(x), 75)


def test_a_file_that_names_no_reflections_asks_for_them():
  with pytest.raises(ValueError, match="names no crystal directions"):
    load("out1.out", CUB)
  hs = [Miller(*x, CUB) for x in [(1, 0, 0), (1, 1, 0), (1, 0, 2), (2, 0, 0), (2, 0, 1), (1, 1, 2), (2, 1, 1), (1, 1, 3)]]
  out = load("out1.out", hs, CUB)
  # a point not measured carries a negative intensity and is no point
  assert out.numPF == 8 and all(I.size == 1224 and I.min() >= 0 for I in out.allI)
  assert means(out) == [3.5059, 3.5257, 2.7263, 1.9823, 5.8509, 2.4005, 2.0374, 5.8786]
  with pytest.raises(ValueError):
    load("out1.out", hs[:3], CUB)


def test_the_formats_whose_sections_a_marker_opens():
  pf = load("siemens.dat", CUB)
  assert pf.numPF == 4 and pf.allI[0].shape == (72, 15) and np.isclose(thetaMax(pf), 70) and indices(pf) == [1, 1, 1]
  # eight character fields, a five digit count filling its own: the first count less the
  # mean of the two backgrounds under its *Khi line
  assert np.isclose(pf.allI[0][0, 0], 3809.60 - (3763.2 + 2934.8) / 2) and np.isclose(pf.allI[0].max(), 22885.6)
  hs = [Miller(1, 1, 1, CUB), Miller(2, 0, 0, CUB), Miller(2, 2, 0, CUB), Miller(3, 1, 1, CUB)]
  uxd = load("bruker.UXD", hs, CUB)
  assert uxd.numPF == 4 and all(I.size == 1224 for I in uxd.allI) and np.isclose(thetaMax(uxd), 80)
  assert np.isclose(np.mean(uxd.allI[0]), 1149.736111) and np.isclose(np.mean(uxd.allI[1]), 672.846405)
  j = load("juelich130", CUB)
  assert j.allI[0].size == 480 and np.isclose(np.mean(j.allI[0]), 9.570833) and indices(j) == [1, 3, 0]
  g = load("geesthacht_cut.dat", [Miller(1, 0, 0, CUB)] * 2, CUB)
  assert [I.size for I in g.allI] == [679, 16] and np.isclose(np.mean(g.allI[0]), 741.621502) and np.isclose(np.mean(g.allI[1]), 245.4375)
  # nja names its reflection and its count, so both are read by name
  n = load("seifert-111.nja", CUB)
  assert n.allI[0].size == 2121 and indices(n) == [1, 1, 1] and np.isclose(np.mean(n.allI[0]), 67.998727) and np.isclose(thetaMax(n), 70)
  pwd = load("siemens.pwd", CUB)
  assert pwd.numPF == 4 and all(I.size == 15 for I in pwd.allI) and np.isclose(np.mean(pwd.allI[0]), 6072.857333)
  assert np.allclose(pwd.allR[0].rho, 0) and np.isclose(thetaMax(pwd), 70)
  r = load("Rigaku002.txt", CUB)
  assert r.allI[0].size == 1241 and np.isclose(np.mean(r.allI[0]), 2459.469746) and np.isclose(thetaMax(r), 80)
  rd = load("rigaku_cut.DAT", Miller(1, 0, 0, CUB), CUB)
  assert rd.allI[0].size == 1500 and np.isclose(thetaMax(rd), 90, atol=1e-3)
  # the polar ladder is nowhere in the file: the azimuth step names it, theta states it
  rt = load("Rigaku_txt.TXT", Miller(1, 1, 0, CUB), CUB)
  assert rt.allI[0].size == 3751 and np.isclose(np.mean(rt.allI[0]), 97.992002) and np.isclose(thetaMax(rt), 90)
  stated = load("Rigaku_txt.TXT", Miller(1, 1, 0, CUB), CUB, theta=np.arange(31) * 2 * degree)
  assert np.isclose(thetaMax(stated), 60)
  with pytest.raises(ValueError):
    load("bruker.UXD", [Miller(1, 1, 1, CUB)], CUB)


def test_the_binary_formats_and_the_XML_one():
  g = load("Al_111.gpol", CUB)
  assert g.allI[0].size == 18204 and indices(g) == [1, 1, 1] and g.allI[0][:5].tolist() == [435, 596, 403, 308, 121]
  assert np.isclose(np.mean(g.allI[0]), 422.216766)
  assert np.isclose(np.degrees(g.allR[0].theta).min(), 0.7105, atol=1e-4) and np.isclose(thetaMax(g), 78.4167, atol=1e-4)
  w = load("philipsXpert.rw1", CUB)
  assert w.allI[0].shape == (72, 17) and np.isclose(np.mean(w.allI[0]), 448.283721) and np.isclose(w.allI[0][0, 0], 1348.082)
  assert indices(w) == [0, 0, 2]
  x = load("xrdml.xrdml", CUB)
  assert x.allI[0].size == 3600 and indices(x) == [1, 1, 1] and np.isclose(np.mean(x.allI[0]), 14631.728889)
  assert np.allclose(x.allI[0][:3], [32028.4, 31882.8, 31767.0])


def test_the_Miller_indices_of_a_group():
  assert F.millerIndices("10-11") == [1, 0, -1, 1] and F.millerIndices("111") == [1, 1, 1] and F.millerIndices("12") is None
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    assert F.ladderOf(0, 0.45, 11.7).size == 27
