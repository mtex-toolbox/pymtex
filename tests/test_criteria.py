# M3: the grain boundary criteria, the Julia criteria tests
import numpy as np
import pytest

from mtex import EBSD, phaseList, crystalFrame, rotation, vector3d, degree, angle, gbcAngle, gbcSoft, gbcEqual, gbcCustom, connectivity
from mtex.geometry.kernels import qmul
from conftest import close
from test_calcgrains import blockMap, SQ3

rng = np.random.default_rng(903)


def test_the_symmetry_search_agrees_with_exhaustive_equivalents():
  for symbol in ("1", "6/mmm", "m-3m"):
    for dtype in (np.float32, np.float64):
      fr = crystalFrame(symbol)
      G = fr.sym.properGroup().rotations().data.reshape(-1, 4)
      rot = rotation.rand(120, rng=rng).astype(dtype)
      pos = np.stack([np.arange(120.0), np.zeros(120), np.zeros(120)], axis=-1)
      e = EBSD(pos, rot, np.ones(120, dtype=np.int8), phaseList(fr), lattice=None)
      i, j = np.arange(119), np.arange(1, 120)
      q = rot.data
      for levels in ((0, 10, 40), (40, 10), (180,), (10,)):
        t = np.array(levels) * degree
        m = qmul(q[j] * [1, -1, -1, -1], q[i])
        d = np.max(np.abs(m @ G.T), axis=1)
        expected = np.sum(d[:, None] > np.cos(t / 2)[None, :], axis=1) / len(t)
        out = connectivity(gbcAngle(t), e, i, j)
        assert out.dtype == dtype and close(out, expected, tol=1e-6)


def test_S2_the_angle_criterion():
  e = blockMap(SQ3, 2, 3)
  i, j = [0, 0, 0], [1, 3, 4]                     # neighbours in the same block, and across
  c = connectivity(gbcAngle(5 * degree), e, i, j)
  assert np.all((0 <= c) & (c <= 1)) and c[0] == 1 and c[1] == 0
  assert close(gbcAngle(5 * degree).evaluate(e, i, j), c)
  # a pair of levels answers on three levels
  c2 = connectivity(gbcAngle([20 * degree, 3 * degree]), e, [0], [3])
  assert c2[0] == 0.5
  # a phase can be given its own, by name, without touching the levels
  c3 = connectivity(gbcAngle(5 * degree, byPhase={"test": 90 * degree}), e, [0], [3])
  assert c3[0] == 1
  # MATLAB's list in CSList order
  c4 = connectivity(gbcAngle([[10 * degree], [90 * degree]]), e, [0], [3])
  assert c4[0] == 1
  # different phases never join, two not indexed always do
  e2 = blockMap(SQ3, 2, 3, notIndexed=((1, 1, 1.5),))
  ni = np.flatnonzero(~e2.isIndexed)
  ind = np.flatnonzero(e2.isIndexed)[0]
  assert connectivity(gbcAngle(5 * degree), e2, [ni[0]], [ni[1]])[0] == 1
  assert connectivity(gbcAngle(5 * degree), e2, [ni[0]], [ind])[0] == 0
  assert "gbcAngle" in repr(gbcAngle(5 * degree))
  with pytest.raises(ValueError):
    connectivity(gbcAngle(5 * degree), e, [0, 1], [1])


def test_S3_the_soft_criterion_is_the_angle_one_smoothed():
  e = blockMap(SQ3, 2, 3)
  hard = connectivity(gbcAngle(7 * degree), e, [0, 0], [1, 3])
  soft = connectivity(gbcSoft(7 * degree), e, [0, 0], [1, 3])
  assert soft[0] > 0.9 and hard[0] == 1          # far below the threshold
  assert 0 < soft[1] < 1                         # at the threshold it is neither
  assert close(connectivity(gbcSoft([7 * degree, 3.5 * degree]), e, [0, 0], [1, 3]), soft)
  assert gbcSoft(7 * degree).width == 3.5 * degree and "gbcSoft" in repr(gbcSoft())


def test_a_property_and_a_custom_function():
  e = blockMap(SQ3, 2, 3)
  e.variantId = (np.arange(36) % 2).astype(int)
  c = connectivity(gbcEqual("variantId"), e, [0, 0], [1, 2])
  assert c[0] == 0 and c[1] == 1
  assert close(connectivity(gbcEqual(e.variantId), e, [0, 0], [1, 2]), c)
  # a per pixel property with a tolerance, and a function of the map and the pair
  e.bc = np.arange(36.0)
  c = connectivity(gbcCustom(e.bc, 1.5), e, [0, 0], [1, 2])
  assert c[0] == 1 and c[1] == 0
  c = connectivity(gbcCustom(lambda ebsd, i, j: np.abs(ebsd.bc[i] - ebsd.bc[j]) < 3), e, [0, 0], [1, 5])
  assert c[0] == 1 and c[1] == 0
  # an axial property compared as an axis
  ax = vector3d.rand(36, rng=rng)
  ax.data[1] = -ax.data[0]
  c = connectivity(gbcCustom(ax, 1 * degree, antipodal=True), e, [0], [1])
  assert c[0] == 1 and connectivity(gbcCustom(ax, 1 * degree), e, [0], [1])[0] == 0
