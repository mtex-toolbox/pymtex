# the import of grain meshes, DREAM.3D and Neper, against MATLAB master (bench/matlab/probe_loadgrains3d.m)
import os

import numpy as np
import pytest

from mtex import EBSD3, degree, grain3d, loadNeperTess, plane3d, vector3d

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "master", "data")
DREAM3D = os.path.join(DATA, "EBSD", "SmallIN100_MeshStats.dream3d")
TESS = os.path.join(DATA, "Neper", "my100grains.tess")


@pytest.mark.skipif(not os.path.exists(TESS), reason="the Neper data of ../master/data")
def test_a_neper_tessellation():
  t = loadNeperTess(TESS)
  assert t.dimension == 3 and len(t.V) == 6134 and len(t.F) == 7131 and t.I_GF.shape == (1000, 7131) and t.crysym == "triclinic"
  g = grain3d.load(TESS)
  assert len(g) == 1000 and not g.boundary.isTriangulated and np.isclose(g.volume.sum(), 32) and np.all(g.volume > 0)
  assert np.allclose(g.volume[:5], [0.013374218, 0.060431542, 0.074319907, 0.013167039, 0.031180139])
  assert np.allclose(g.meanRotation.data[:2], [[0.501942, 0.461763, 0.70884, -0.179931], [0.26803, -0.370155, 0.00807199, -0.889427]], atol=1e-6)
  assert np.allclose(g.centroid.data[0], [1.40696921, 0.25880703, 1.14319614])
  assert list(g.numNeighbors[:10]) == [9, 12, 16, 14, 15, 14, 16, 9, 11, 15]
  assert np.allclose(g.diameter[:5], [0.46414168, 0.68085808, 0.71262603, 0.43514128, 0.61517775])
  s = g.slice(plane3d(vector3d(0, 0, 1), vector3d(0, 0, 1)))
  assert len(s) == 232 and np.isclose(s.area.sum(), 16)


@pytest.mark.skipif(not os.path.exists(DREAM3D), reason="the IN100 volume of ../master/data")
def test_the_dream3d_mesh():
  g = grain3d.load(DREAM3D)
  assert len(g) == 794 and len(g.F) == 757564 and np.isclose(g.volume.sum(), 15625) and np.all(g.volume > 0)
  assert np.allclose(g.volume[:5], [4.0710272, 16.927367, 2.5024644, 37.923512, 27.314664])
  # after orientFaces every inner face is +1 for one grain and -1 for the other
  assert list(np.bincount((np.asarray(g.I_GF.sum(axis=0)).ravel() + 2).astype(int))) == [0, 0, 637564, 120000]
  assert g.boundary.grainId[:3].tolist() == [[793, 0]] * 3
  # the mean of a grain is the orientation of its voxels: DREAM.3D's (x, y, z, w) is passive
  e = EBSD3.load(DREAM3D)
  fid = e.grainId.reshape(-1).astype(int)
  ori = e.orientations.flatten()
  for k in (1, 2, 100):
    sel = np.flatnonzero(fid == g.id[k])[:5]
    assert np.all(np.asarray(ori[sel].angle(g.meanOrientation[k])) < 3 * degree)
  raw = grain3d.load(DREAM3D, noOrientFaces=True)
  assert not np.all(raw.volume > 0)
