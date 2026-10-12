# the orientation gradients, curvature, GND, KAM, fill and profiles of a volume; the gradients
# against MATLAB master's EBSD3square/gradientX, Y, Z
import os

import numpy as np
import pytest

from mtex import EBSD3, EBSD3square, crystalFrame, degree, dislocationSystem, phaseList, rotation, vector3d

DREAM3D = os.path.join(os.path.dirname(__file__), "..", "..", "master", "data", "EBSD", "SmallIN100_MeshStats.dream3d")
A = 0.5 * degree


# a volume bent about z along x: the orientation turns by A per unit of x
def bent(n=(6, 5, 4)):
  x = np.arange(n[0])[:, None, None] * np.ones(n)
  r = rotation.byAxisAngle(vector3d(0, 0, 1), A * x)
  return EBSD3square(r, np.ones(n, dtype=np.int8), phaseList([crystalFrame("m-3m", mineral="Iron")]))


def test_the_gradients_and_the_curvature_of_a_bend():
  e = bent()
  gx, gy, gz = e.gradientX().data, e.gradientY().data, e.gradientZ().data
  assert np.allclose(gx[..., 2], A) and np.allclose(gx[..., :2], 0, atol=1e-12)
  assert np.allclose(gy, 0, atol=1e-12) and np.allclose(gz, 0, atol=1e-12)
  k = e.curvature()
  M = np.asarray(k.M)
  assert M.shape == (6, 5, 4, 3, 3) and np.allclose(M[..., 2, 0], A) and np.isclose(np.abs(M).sum(axis=(-1, -2)), A).all()
  gnd, rho = e.calcGND(dislocationSystem.bcc(e.CS))
  assert gnd.shape == (6, 5, 4) and np.all(np.isfinite(gnd)) and np.all(gnd > 0)


def test_KAM_averages_over_the_face_neighbours():
  e = bent()
  kam = e.KAM()
  # an inner voxel: two of its six neighbours turned by A
  assert np.isclose(kam[2, 2, 2], 2 * A / 6) and np.isclose(kam[0, 2, 2], A / 5)
  assert np.isclose(e.KAM(max=True)[2, 2, 2], A)


def test_shifted_pairs_every_cell_with_the_one_the_offset_away():
  from mtex.maps3d.intragranular3d import _shifted
  a = np.arange(24).reshape(2, 3, 4)
  src, dst = _shifted((1, -1, 0), a.shape)
  assert np.array_equal(a[dst] - a[src], np.full((1, 2, 4), 12 - 4))


def test_fill_closes_an_inner_hole_and_extrapolates_on_request():
  e = bent()
  f = e.copy()
  f.phaseId[2, 2, 2] = 0
  f.phaseId[0, 0, 0] = 0
  g = f.fill()
  assert g.phaseId[2, 2, 2] == 1 and g.phaseId[0, 0, 0] == 0 and f.fill(extrapolate=True).phaseId[0, 0, 0] == 1


def test_a_profile_along_x_steps_voxel_by_voxel():
  e = bent()
  p, d = e.spatialProfile(np.array([[0.0, 1.0, 1.0], [5.0, 1.0, 1.0]]))
  assert p.size == 6 and np.allclose(d, np.arange(6)) and np.all(p.phaseId == 1)


@pytest.mark.skipif(not os.path.exists(DREAM3D), reason="the IN100 volume of ../master/data")
def test_IN100_gradients_as_MATLAB():
  s = EBSD3.load(DREAM3D)[:20, :20, :20]
  ref = {"gradientX": ([-0.00244005, 0.0345608, -0.0167611], [0.00539828, -0.0074289, 0.0225272], 733),
         "gradientY": ([-0.00303671, 0.0331253, 0.00226433], [-0.00942081, 0.050318, -0.0188765], 737),
         "gradientZ": ([0.00788212, 0.0573358, 0.0229568], [np.nan] * 3, 1013)}
  for name, (a, b, nNan) in ref.items():
    v = getattr(s, name)().data
    assert np.allclose(v[1, 2, 3], a, atol=1e-6) and np.allclose(v[19, 4, 6], b, atol=1e-6, equal_nan=True)
    assert int(np.isnan(v[..., 0]).sum()) == nNan
