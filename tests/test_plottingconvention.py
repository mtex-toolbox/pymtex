import numpy as np
import pytest

import mtex
from mtex import plottingConvention, vector3d, xvector, yvector, zvector, rotation, setMTEXpref


def test_pictogram_and_parsing():
  pC = plottingConvention("y↑→x")
  assert pC.north.isclose(yvector) and pC.east.isclose(xvector) and pC.outOfScreen.isclose(zvector)
  assert pC.char() == "y↑→x" and str(pC) == "y↑→x" and repr(pC) == "plottingConvention (y↑→x)"
  for s in ("y↑x→", "↑y→x", "y^->x", "x->y^"):
    assert plottingConvention(s) == pC
  for oos in (zvector, -zvector):
    for e in (xvector, -xvector, yvector, -yvector):
      q = plottingConvention(oos, e)
      assert q.isPictogram and plottingConvention(q.char()) == q and plottingConvention(q.char(utf8=False)) == q
  assert plottingConvention("z⊙→x") == plottingConvention("y↑→x") and plottingConvention("z(x)->x") == plottingConvention.ij
  assert plottingConvention("-x→↑y").east.isclose(-xvector)
  setMTEXpref("UTF8Output", False)
  assert pC.char() == "y^->x"


def test_named_conventions():
  ij = plottingConvention.ij
  assert ij.east.isclose(xvector) and ij.south.isclose(yvector) and ij.intoScreen.isclose(zvector) and ij.char() == "y↓→x"
  d3 = plottingConvention.default3D
  assert not d3.isPictogram and d3.char() == "xyz" and "outOfScreen" in repr(d3)
  assert plottingConvention.edax(1).east.isclose(yvector) and plottingConvention.edax(2).east.isclose(-yvector)
  assert plottingConvention.edax(3).east.isclose(xvector) and plottingConvention.edax(4).east.isclose(-xvector)
  with pytest.raises(ValueError):
    plottingConvention.edax(5)
  assert plottingConvention().east.isclose(xvector) and plottingConvention(rotation.id) == plottingConvention()
  assert plottingConvention(plottingConvention.ij) == plottingConvention.ij


def test_setters_move_one_direction():
  pC = plottingConvention("y↑→x")
  pC.east = yvector
  assert pC.east.isclose(yvector)
  pC.north = -xvector
  assert pC.north.isclose(-xvector)
  pC.west = zvector
  assert pC.east.isclose(-zvector)
  pC.intoScreen = yvector
  assert pC.outOfScreen.isclose(-yvector)


def test_rejected_strings():
  for s in ("q↑→x", "y↑x", "y↑→y", "x↑→y↑z"):
    with pytest.raises(ValueError):
      plottingConvention(s)


def test_session_default():
  assert plottingConvention.default == plottingConvention.ij
  plottingConvention.default(north=xvector)
  assert plottingConvention.default.north.isclose(xvector)
  plottingConvention.default(intoScreen=zvector)
  assert plottingConvention.default.intoScreen.isclose(zvector)
  plottingConvention.default(east=xvector)
  assert plottingConvention.default == plottingConvention("y↓→x")
  pC = plottingConvention("y↑→x")
  pC.makeDefault()
  pC.east = yvector
  assert plottingConvention.default == plottingConvention("y↑→x")
  mtex.reset()
  assert plottingConvention.default == plottingConvention.ij


# the eight axis aligned conventions the import wizard offers
def test_axis_conventions():
  from mtex.geometry.plottingconvention import AXIS_CONVENTIONS, closestConvention, conventionNames
  from mtex.io.ang import eulerCorrectionRotation
  assert len(AXIS_CONVENTIONS) == 8
  assert [plottingConvention(s).char(utf8=True) for s in AXIS_CONVENTIONS] == list(AXIS_CONVENTIONS)
  assert conventionNames(("X1", "Y1", "Z1"))[0] == "Y1↑→X1"
  assert conventionNames(("RD", "TD", "ND"))[5] == "RD←↓TD"
  # the vendor corrections are members: the Oxford half turn about z and the EDAX one about (1, -1, 0)
  k, d = closestConvention(rotation.byAxisAngle(zvector, np.pi))
  assert AXIS_CONVENTIONS[k] == "x←↓y" and d < 1e-12
  k, d = closestConvention(eulerCorrectionRotation(2))
  assert AXIS_CONVENTIONS[k] == "y←↓x" and d < 1e-12
  k, d = closestConvention(plottingConvention.ij)
  assert AXIS_CONVENTIONS[k] == "y↓→x" and d < 1e-12
  # a tilted view is far from all eight
  assert closestConvention(plottingConvention.default3D)[1] > 0.1
