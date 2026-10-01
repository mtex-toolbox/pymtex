# the boundary normal distribution of 3D grains, against MATLAB master (bench/matlab/probe_gbnd3d.m,
# where MATLAB's mean orientations are corrected for its reading of DREAM.3D's quaternions)
import os

import numpy as np
import pytest

from mtex import EBSD3square, Miller, calcGBND, calcGrains, crystalFrame, degree, grain3d, orientation, phaseList, rotation, vector3d

DREAM3D = os.path.join(os.path.dirname(__file__), "..", "..", "master", "data", "EBSD", "SmallIN100_MeshStats.dream3d")


def test_the_normals_of_a_box():
  r = np.zeros((4, 4, 4, 4))
  r[..., 0] = 1
  e = EBSD3square(rotation(r), np.ones((4, 4, 4), dtype=np.int8), phaseList([crystalFrame("m-3m", mineral="Iron")]))
  g = calcGrains(e)
  f = calcGBND(g.boundary)
  v = np.real(np.asarray(f.eval(vector3d(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])))))
  # the six faces of a cube: the three axes alike, the diagonal low
  assert np.allclose(v[:3], v[0], rtol=1e-6) and v[3] < v[0]


needsDream3d = pytest.mark.skipif(not os.path.exists(DREAM3D), reason="the IN100 volume of ../master/data")
V = vector3d(np.array([[0, 0, 1], [1, 0, 0], [1, 1, 1], [0.3, 0.5, 0.8]]))


# the IN100 grains and their boundaries between two grains of phase 1
@pytest.fixture(scope="module")
def in100():
  g = grain3d.load(DREAM3D)
  return g, g.boundary[np.all(g.boundary.phaseId == 1, axis=1)]


@needsDream3d
def test_calcGBND_ofIN100InTheSpecimenAsMATLAB(in100):
  g, gB = in100
  assert np.allclose(np.real(calcGBND(gB).eval(V)), [1.739678, 1.364896, 0.935329, 1.158495], atol=1e-6)


@needsDream3d
def test_calcGBND_ofIN100InTheCrystalAsMATLAB(in100):
  g, gB = in100
  cs = g.CSList[1]
  assert np.allclose(np.real(calcGBND(gB, g).eval(Miller(V, cs))), [0.7411072, 0.7411072, 2.004266, 1.047731], atol=1e-6)


@needsDream3d
def test_calcGBND_ofIN100AtOneMisorientationAsMATLAB(in100):
  g, gB = in100
  cs = g.CSList[1]
  m = orientation.byAxisAngle(vector3d(1, 1, 1), 60 * degree, cs, cs)
  assert np.allclose(np.real(calcGBND(gB, g, m).eval(Miller(V, cs))), [0.3129911, 0.3129911, 16.30684, 3.716864], atol=1e-4)
