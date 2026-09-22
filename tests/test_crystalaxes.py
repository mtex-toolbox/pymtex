# the crystal axes of a cell and their alignment (check_crystalAxes)
import numpy as np
import pytest
from numpy import pi

from mtex import crystalFrame, vector3d, xvector, degree, angle, alignment
from mtex.geometry.crystalaxes import reciprocalAxes, completeCell


def axesOf(fr):
  return fr.basis.data


def test_orthogonal_lattices():
  for pg, cell in (("mmm", [1, 2, 3]), ("422", [2, 2, 3]), ("m-3m", [2, 2, 2])):
    ref = axesOf(crystalFrame(pg, cell))
    assert np.max(np.abs(ref - np.diag(cell))) < 1e-12
    # naming either of a/a* and c/c* is the same request
    for al in (["X||a"], ["X||a*"], ["X||a", "Z||c"], ["X||a*", "Z||c"], ["X||a", "Z||c*"], ["EDAX"]):
      assert np.max(np.abs(axesOf(crystalFrame(pg, cell, *al)) - ref)) < 1e-12
      assert np.max(np.abs(axesOf(crystalFrame(pg, cell, alignment=al)) - ref)) < 1e-12
  assert alignment(crystalFrame([2, 3, 4])) == []


def test_hexagonal_setups():
  a, c = 2.0, 3.0
  wantAStar = np.array([[a * np.sqrt(0.75), -a / 2, 0], [0, a, 0], [0, 0, c]])
  wantA = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(0.75), 0], [0, 0, c]])
  for pg in ("321", "6/mmm", "3", "-6m2"):
    assert np.max(np.abs(axesOf(crystalFrame(pg, [a, a, c])) - wantAStar)) < 1e-12
    for al in (["X||a*"], ["X||a*", "Z||c"]):
      assert np.max(np.abs(axesOf(crystalFrame(pg, [a, a, c], *al)) - wantAStar)) < 1e-12
    for al in (["X||a"], ["X||a", "Z||c"], ["EDAX"], ["TSL"]):
      assert np.max(np.abs(axesOf(crystalFrame(pg, [a, a, c], *al)) - wantA)) < 1e-12
    cs1, cs2 = crystalFrame(pg, [a, a, c]), crystalFrame(pg, [a, a, c], "X||a")
    assert abs(angle(vector3d(cs1.aAxis.data), vector3d(cs2.aAxis.data)) - 30 * degree) < 1e-12
    assert "X||a" in cs2.alignment() and "X||a*" in cs1.alignment()


def test_naming_does_not_align_and_the_general_path():
  for pg, cell, ang in (("m-3m", [2, 2, 2], [90, 90, 90]), ("321", [2, 2, 3], [90, 90, 120]),
    ("2/m", [1, 2, 3], [90, 100, 90]), ("1", [1, 2, 3], [80, 95, 110])):
    ref = axesOf(crystalFrame(pg, cell, np.array(ang) * degree))
    got = axesOf(crystalFrame(pg, cell, np.array(ang) * degree, mineral="Test", color="red"))
    assert np.all(got == ref)
  cs = crystalFrame("2/m", [1, 2, 3], np.array([90, 100, 90]) * degree)
  csA = crystalFrame("2/m", [1, 2, 3], np.array([90, 100, 90]) * degree, "X||a")
  assert np.max(np.abs(axesOf(cs) - axesOf(csA))) > 0.1
  assert angle(csA.aAxis, xvector) < 1e-10 and angle(cs.aAxisRec, xvector) < 1e-10
  with pytest.raises(ValueError, match="alignment"):
    crystalFrame("1", [1, 2, 3], np.array([80, 95, 110]) * degree, "X||a", "Z||c")
  # the monoclinic unique axis follows the angles, and the lattice checks its cell
  assert crystalFrame("2/m", [8.6, 13, 7.2], np.array([90, 116, 90]) * degree, mineral="orthoclase").pointGroup == "12/m1"
  assert crystalFrame("2/m", [1, 2, 3], np.array([90, 90, 100]) * degree).pointGroup == "112/m"
  assert crystalFrame("2/m", [1, 2, 3], [90, 90, 100]).pointGroup == "112/m"   # degrees are read as such
  with pytest.raises(ValueError):
    crystalFrame("m-3m", [1, 2, 3])
  with pytest.raises(ValueError):
    crystalFrame("2/m", [1, 2, 3], np.array([100, 100, 90]) * degree)
  # the group is written in the axes: the two-fold of 321 lies along a
  cs = crystalFrame("321", [3, 3, 5], "X||a")
  o = cs.rot
  assert np.any(~o.isImproper & (np.abs(o.angle() - pi) < 1e-10) & (o.axis().angle(cs.aAxis, antipodal=True) < 1e-10))


def test_kernels():
  axes = crystalFrame("1", [1, 2, 3], np.array([80, 95, 110]) * degree).basis.data
  dual = reciprocalAxes(axes)
  assert np.allclose(axes @ dual.T, np.eye(3))
  assert completeCell([2, 0, 0], [0, 0, 0]) == ((2, 2, 2), (pi / 2, pi / 2, pi / 2))
