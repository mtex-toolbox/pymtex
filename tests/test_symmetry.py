# point groups from their symbols: orders, closure, names, relations, and the Crystalline
# fixture of the Julia port
import json
import pathlib

import numpy as np
import pytest
from numpy import pi

import mtex
from mtex import (symmetry, latticeType, rotation, orthogonalTransform, reflection, vector3d, xvector, yvector,
                  zvector, degree, findPointGroup, isPointGroupName, hm2point, monoclinicSetting,
                  spaceGroupToPointGroup, identifyGroup, Laue, properGroup, properSubGroup, disjoint, union,
                  factor, nfold, numSym, numProper, LaueName)
from mtex.geometry.data.pointgroupnames import POINT_GROUP_NAMES
from mtex.geometry.symmetry import _isin, _sameElements, _uniqueElements

GROUP_ORDERS = {"1": 1, "-1": 2, "211": 2, "m11": 2, "2/m11": 4, "121": 2, "1m1": 2, "12/m1": 4, "112": 2, "11m": 2,
  "112/m": 4, "222": 4, "2mm": 4, "m2m": 4, "mm2": 4, "mmm": 8, "3": 3, "-3": 6, "321": 6, "3m1": 6, "-3m1": 12,
  "312": 6, "31m": 6, "-31m": 12, "4": 4, "-4": 4, "4/m": 8, "422": 8, "4mm": 8, "-42m": 8, "-4m2": 8, "4/mmm": 16,
  "6": 6, "-6": 6, "6/m": 12, "622": 12, "6mm": 12, "-62m": 12, "-6m2": 12, "6/mmm": 24, "23": 12, "m-3": 24,
  "432": 24, "-43m": 24, "m-3m": 48, "532": 60, "-5-32/m": 120}
LAUE_OF = {"1": "-1", "211": "2/m11", "m11": "2/m11", "121": "12/m1", "1m1": "12/m1", "112": "112/m", "11m": "112/m",
  "222": "mmm", "2mm": "mmm", "m2m": "mmm", "mm2": "mmm", "3": "-3", "321": "-3m1", "3m1": "-3m1", "312": "-31m",
  "31m": "-31m", "4": "4/m", "-4": "4/m", "422": "4/mmm", "4mm": "4/mmm", "-42m": "4/mmm", "-4m2": "4/mmm", "6": "6/m",
  "-6": "6/m", "622": "6/mmm", "6mm": "6/mmm", "-62m": "6/mmm", "-6m2": "6/mmm", "23": "m-3", "432": "m-3m",
  "-43m": "m-3m", "532": "-5-32/m"}
PROPER_OF = {"-1": "1", "m11": "1", "2/m11": "211", "1m1": "1", "12/m1": "121", "11m": "1", "112/m": "112",
  "2mm": "211", "m2m": "121", "mm2": "112", "mmm": "222", "-3": "3", "3m1": "3", "-3m1": "321", "31m": "3",
  "-31m": "312", "-4": "112", "4/m": "4", "4mm": "4", "-42m": "222", "-4m2": "222", "4/mmm": "422", "-6": "3",
  "6/m": "6", "6mm": "6", "-62m": "321", "-6m2": "312", "6/mmm": "622", "m-3": "23", "-43m": "23", "m-3m": "432",
  "-5-32/m": "532"}


def isGroup(pg):
  return bool(np.all(_isin((pg.rot[:, None] * pg.rot[None, :]).flatten(), pg.rot)))


def has(pg, kind, v):
  o = pg.rot
  along = o.axis().angle(v, antipodal=True) < 1e-10
  if kind == "2":
    sel = ~o.isImproper & (np.abs(o.angle() - pi) < 1e-10)
  elif kind == "3":
    sel = ~o.isImproper & (np.abs(o.angle() - 2 * pi / 3) < 1e-10)
  else:
    sel = o.isImproper & (np.abs(o.angle() - pi) < 1e-10)
  return bool(np.any(sel & along))


SYMBOLS = [row[0] for row in POINT_GROUP_NAMES]


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_symmetry_hasTheOrderOfItsSymbol(symbol):
  pg = symmetry(symbol)
  assert pg.pointGroup == symbol and pg.numSym() == GROUP_ORDERS[symbol] and len(pg) == numSym(pg)


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_symmetry_elementsAreDistinctAndClosed(symbol):
  pg = symmetry(symbol)
  assert _uniqueElements(pg.rot).size == pg.numSym()
  assert isGroup(pg)


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_isLaue_iffTheInversionIsAnElement(symbol):
  pg = symmetry(symbol)
  assert pg.isLaue == (rotation.inversion in pg)


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_identifyGroup_namesTheElementsBySymbol(symbol):
  assert identifyGroup(symmetry(symbol).rot) == symbol


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_Laue_isTheLaueGroupContainingIt(symbol):
  pg = symmetry(symbol)
  L = pg.Laue()
  assert L.isLaue and pg <= L and L.pointGroup == LAUE_OF.get(symbol, symbol) and LaueName(pg) == L.pointGroup


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_properSubGroup_isTheProperPart(symbol):
  pg = symmetry(symbol)
  P = pg.properSubGroup()
  assert P.isProper and P <= pg and numProper(pg) == P.numSym()
  # the proper subgroup of -4m2 is 222 turned by 45 degree about c: its type by the signature, as MATLAB names it
  assert P.pointGroup == PROPER_OF.get(symbol, symbol)


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_properGroup_andLaue_asFunctions(symbol):
  pg = symmetry(symbol)
  L = pg.Laue()
  assert properGroup(pg) == properSubGroup(L) and Laue(pg) == L


c4 = rotation.byAxisAngle(zvector, pi / 2)
c3 = rotation.byAxisAngle(vector3d(1, 1, 1), 2 * pi / 3)


def test_symmetry_closureOfRotations():
  assert symmetry(mtex.cat(c4, c3)).numSym() == 24 and symmetry(mtex.cat(c4, c3)).pointGroup == "432"


def test_symmetry_closureWithTheInversion():
  assert symmetry(orthogonalTransform.cat(orthogonalTransform(c4), orthogonalTransform(c3), rotation.inversion)).pointGroup == "m-3m"


def test_symmetry_ofOneReflection():
  assert symmetry(reflection(zvector)).pointGroup == "11m"


def test_symmetry_ofOneThreefold():
  assert symmetry(rotation.byAxisAngle(zvector, 2 * pi / 3)).pointGroup == "3"


def test_symmetry_ofAFivefoldIsUnnamed():
  five = symmetry(rotation.byAxisAngle(zvector, 2 * pi / 5))
  assert five.pointGroup == "unknown" and five.id == 0 and five.Schoenflies == "unknown"
  assert five == symmetry(rotation.byAxisAngle(zvector, 4 * pi / 5)) and five.numSym() == 5


def test_symmetry_ofASymmetry():
  assert symmetry(symmetry("432")) == symmetry("432")


@pytest.mark.parametrize("name, symbol", (("m-3m", "m-3m"), ("Oh", "m-3m"), ("cubic", "m-3m"), ("hexagonal", "6/mmm"), ("2/m", "12/m1"),
  ("2", "121"), ("m", "1m1"), ("orthotropic", "mmm"), ("32", "321"), ("3m", "3m1"), ("-3m", "-3m1"),
  ("triclinic", "-1"), ("tetragonal", "4/mmm"), ("trigonal", "-31m"), ("monoclinic", "112/m"), ("C1", "1"),
  ("D2", "222"), ("Td", "-43m"), ("m3m", "m-3m"), ("S6", "-3"), ("icosahedral", "-5-32/m")))
def test_findPointGroup_ofAName(name, symbol):
  assert findPointGroup(name) == symbol and symmetry(name).pointGroup == symbol


def test_findPointGroup_ofASpaceGroupSymbol():
  assert findPointGroup("P6_3/mmc") == "6/mmm" and findPointGroup("Fm-3m") == "m-3m" and findPointGroup("P2_1/c") == "12/m1"
  assert findPointGroup("Pnma") == "mmm"


def test_hm2point():
  assert hm2point("Pnma") == "mmm" and hm2point("I4_1/amd") == "4/mmm" and hm2point("P6_322") == "622"


def test_isPointGroupName():
  assert isPointGroupName("m-3m") and not isPointGroupName("measurement") and not isPointGroupName("rolling")


def test_findPointGroup_refusesAnUnknownName():
  with pytest.raises(ValueError):
    findPointGroup("foo")


def test_monoclinicSetting_followsTheOddAngle():
  assert monoclinicSetting("2/m", np.array([90, 116, 90]) * degree) == "12/m1"
  assert monoclinicSetting("2", np.array([110, 90, 90]) * degree) == "211"
  assert monoclinicSetting("m", np.array([90, 90, 100]) * degree) == "11m"
  assert symmetry("2/m", abg=np.array([90, 116, 90]) * degree).pointGroup == "12/m1"


def test_spaceGroupToPointGroup():
  assert spaceGroupToPointGroup(225) == "m-3m" and spaceGroupToPointGroup(1) == "1" and spaceGroupToPointGroup(194) == "6/mmm"


def test_symmetry_byNumbers():
  assert symmetry(spaceId=225).pointGroup == "m-3m" and symmetry(laueId=7).pointGroup == "m-3m" and symmetry(laueId=11).pointGroup == "6/mmm"
  assert symmetry(pointId=45).pointGroup == "m-3m" and symmetry().pointGroup == "1"


def test_symmetry_names():
  assert symmetry("m-3m").Schoenflies == "Oh" and symmetry("m-3m").id == 45 and symmetry("m-3m").name == "m-3m"
  assert symmetry("321").LaueName() == "-3m1"


def test_symmetry_char():
  assert symmetry("m-3m").char() == "m3̅m" and symmetry("m-3m").char(utf8=False) == "m-3m"


def test_symmetry_pointGroupsTable():
  assert symmetry.pointGroups[44][0] == "m-3m"


def test_symmetry_42mAnd4m2PlaceTheTwofoldAndTheMirrorOppositely():
  # -42m has its two-fold along a and its mirror across the diagonal, -4m2 the other way round
  assert has(symmetry("-42m"), "2", xvector) and has(symmetry("-42m"), "m", vector3d(1, 1, 0))
  assert has(symmetry("-4m2"), "m", xvector) and has(symmetry("-4m2"), "2", vector3d(1, 1, 0))


# the axes of a hexagonal cell
hexAxes = vector3d([[np.sqrt(3) / 2, -0.5, 0], [0, 1, 0], [0, 0, 1]])


def test_symmetry_321And312PlaceTheTwofoldAlongAOrAMinusB():
  # 321 has a two-fold along a, 312 along a - b; in the axes of a hexagonal cell
  axes = hexAxes
  assert has(symmetry("321", axes), "2", axes[0]) and not has(symmetry("312", axes), "2", axes[0])
  assert has(symmetry("312", axes), "2", axes[0] - axes[1])


def test_identifyGroup_inTheGivenAxes():
  assert identifyGroup(symmetry("312", hexAxes).rot, hexAxes) == "312"


def test_symmetry_axesDoNotEnterTheEquality():
  assert symmetry("312", hexAxes) == symmetry("312") and symmetry("312", hexAxes.data) == symmetry("312")


def test_symmetry_monoclinicUniqueAxis():
  # the unique axis of the monoclinic settings
  assert has(symmetry("211"), "2", xvector) and has(symmetry("121"), "2", yvector) and has(symmetry("11m"), "m", zvector)


def test_symmetry_cubicThreefoldAndMirrors():
  # cubic: the three-fold on the body diagonal, the mirrors of -43m across the face diagonals
  assert has(symmetry("23"), "3", vector3d(1, 1, 1)) and has(symmetry("-43m"), "m", vector3d(1, 1, 0)) and has(symmetry("m-3"), "m", xvector)


def test_symmetry_multiplicities():
  assert symmetry("321").multiplicityZ == 3 and symmetry("321").multiplicityPerpZ == 2
  assert symmetry("m-3m").multiplicityZ == 4 and symmetry("4/m").multiplicityPerpZ == 1


def test_nfold_ofTheMainAxis():
  assert nfold(symmetry("m-3m")) == 4 and symmetry("6/mmm").nfold() == 6 and symmetry("-3m").nfold() == 3
  assert symmetry("-4").nfold() == 4 and symmetry("mmm").nfold() == 2 and symmetry("-1").nfold() == 1


def test_nfold_ofAnAxis():
  assert symmetry("m-3m").nfold(vector3d(1, 1, 1)) == 3 and symmetry("m-3m").nfold(zvector) == 4
  assert symmetry("m-3m").nfold(mtex.cat(zvector, vector3d(1, 1, 0), vector3d(1, 2, 3))).tolist() == [4, 2, 1]


def test_elements_axesAndMultiplicities():
  ax, mult = symmetry("432").elements()
  assert sorted(mult.tolist()) == [2] * 12 + [3] * 8 + [4] * 6 and ax.size == 26 and ax.antipodal
  assert symmetry("432").elements(4).size == 6
  assert symmetry("1").elements()[0].isempty


@pytest.mark.parametrize("s, l", (("1", "triclinic"), ("-1", "triclinic"), ("12/m1", "monoclinic"), ("m", "monoclinic"),
  ("mm2", "orthorhombic"), ("mmm", "orthorhombic"), ("3", "trigonal"), ("-3m1", "trigonal"), ("-4", "tetragonal"),
  ("4/mmm", "tetragonal"), ("-6", "hexagonal"), ("6/mmm", "hexagonal"), ("23", "cubic"), ("m-3m", "cubic"), ("532", "icosahedral")))
def test_symmetry_lattice(s, l):
  assert symmetry(s).lattice is latticeType[l] and str(symmetry(s).lattice) == l


def test_rotations_areTheProperElements():
  assert symmetry("m-3m").rotations().size == 24 and symmetry("-43m").rotations().size == 12


m3m, o, mmm, c222 = symmetry("m-3m"), symmetry("432"), symmetry("mmm"), symmetry("222")


def test_symmetry_orderedBySubgroup():
  assert o <= m3m and o < m3m and o != m3m and mmm <= m3m and c222 <= o and not m3m <= o and m3m > o and m3m >= m3m


def test_disjoint_isTheIntersection():
  assert disjoint(m3m, symmetry("6/mmm")) == mmm and disjoint(o, mmm) == c222 and disjoint(o, o) is o


def test_union_isTheGroupGenerated():
  assert union(o, symmetry("-1")) == m3m and union(c222, symmetry("4")).pointGroup == "422" and union(o, c222) is o


def test_Laue_andProperGroup_betweenGroups():
  assert Laue(o) == m3m and properGroup(m3m) == o and properSubGroup(symmetry("-43m")).pointGroup == "23"
  assert properSubGroup(symmetry("-4m2")).numSym() == 4 and properSubGroup(symmetry("-4m2")).lattice is latticeType.orthorhombic


def test_symmetry_derivedGroupsAreBuiltOnce():
  # the derived groups are built once per group
  assert m3m.properGroup() is m3m.properGroup() and m3m.properSubGroup() is m3m.properSubGroup() and o.properGroup() is o


def test_factor_splitsBothGroupsThroughTheCommonPart():
  p, q = symmetry("432"), symmetry("622")
  rot1, d, rot2 = factor(p, q)
  assert _sameElements(_uniqueElements((rot1[:, None] * d[None, :]).flatten()), p.rot)
  assert _sameElements(_uniqueElements((d[:, None] * rot2[None, :]).flatten()), q.rot)
  assert factor(p, p)[1].size == 24


def test_symmetry_equalityByElements():
  assert symmetry("m-3m") == symmetry("Oh") and symmetry("m-3m") != symmetry("432")


def test_symmetry_contains():
  assert rotation.byAxisAngle(zvector, pi / 2) in m3m and rotation.byAxisAngle(zvector, pi / 3) not in m3m
  assert reflection(zvector) in m3m and reflection(zvector) not in o


def test_symmetry_repr():
  assert "48" in repr(m3m) and "cubic" in repr(m3m)


def test_against_crystalline_fixture():
  fixture = json.loads((pathlib.Path(__file__).parent / "fixtures" / "pointgroups.json").read_text())
  for label, sig in fixture.items():
    pg = symmetry(label)
    assert pg.numSym() == len(sig)
    mine = sorted(zip(np.round(pg.rot.angle() / degree).astype(int).tolist(), np.where(pg.rot.isImproper, -1, 1).tolist()))
    assert mine == [tuple(s) for s in sig], label


def test_latticeType_hklForm():
  assert latticeType.hexagonal.isTriHex and latticeType.hexagonal.hklForm == "hkil" and latticeType.cubic.hklForm == "hkl"


def test_latticeType_isOrthogonal():
  assert latticeType.cubic.isOrthogonal and not latticeType.trigonal.isOrthogonal


def test_latticeType_freeLengthsAndAngles():
  assert latticeType.cubic.freeLengths == (True, False, False) and latticeType.tetragonal.freeLengths == (True, False, True)
  assert latticeType.monoclinic.freeAngles() == (False, True, False) and latticeType.triclinic.freeAngles() == (True, True, True)


def test_latticeType_defaultAngles():
  assert np.allclose(latticeType.hexagonal.defaultAngles, (pi / 2, pi / 2, 2 * pi / 3))


def test_sameElement_needsTheParityToAgree():
  from mtex.geometry.symmetry import _sameElement
  e = np.array([[1.0, 0, 0, 0, 1], [1.0, 0, 0, 0, -1]])
  assert _sameElement(e, e, 1e-6).tolist() == [[True, False], [False, True]]


def test_cosetRepresentatives_coverTheGroupWithTheSubgroup():
  from mtex.geometry.symmetry import _cosetRepresentatives
  G, d = symmetry("432").rot, symmetry("422").rot
  reps = _cosetRepresentatives(G, d, left=True)
  assert reps.size == 3 and (reps[:, None] * d[None, :]).flatten().unique().size == G.size
