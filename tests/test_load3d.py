# the import of volumes: xnovo and a stack of maps, against MATLAB master (bench/matlab/probe_ebsd3.m)
import glob
import os

import numpy as np
import pytest

from mtex import EBSD3

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "master", "data")
XNOVO = os.path.join(DATA, "EBSD3", "SimulatedMultiPhase.h5")


@pytest.mark.skipif(not os.path.exists(XNOVO), reason="the xnovo volume of ../master/data")
def test_the_xnovo_volume():
  x = EBSD3.load(XNOVO)
  assert x.shape == (50, 50, 50) and x.scanUnit == "mm" and len(x.CSList) == 10
  assert [f.pointGroup for f in x.CSList.frames[1:]] == ["m-3m", "m-3m", "6/mmm", "4/mmm", "-3m1", "-3m1", "mmm", "12/m1", "-1"]
  assert np.allclose(x.pos.data[0, 0, 0], [-0.245] * 3) and np.allclose(x.pos.data[19, 29, 39], [-0.055, 0.045, 0.145])
  assert x.phaseId[19, 29, 39] == 7 and np.allclose(x.rotations.data[19, 29, 39], [0.92317598, -0.36615592, 0.03882264, 0.11031204])
  # Corundum's rhombohedral cell read with the angles of the hexagonal one, as MATLAB
  assert np.allclose(x.CSList.frames[5].axes.data.reshape(3, 3)[1], [-2.564, 4.441, 0], atol=1e-3)


@pytest.mark.skipif(not glob.glob(os.path.join(DATA, "EBSD", "3dData", "*.ANG")), reason="the stack of ../master/data")
def test_a_stack_of_maps():
  files = sorted(glob.glob(os.path.join(DATA, "EBSD", "3dData", "*.ANG")))
  e = EBSD3.load(files, z=0.1 * np.arange(len(files)))
  assert e.lattice.shape == (35, 40, len(files)) and np.allclose(e.lattice.spacing, [0.4, 0.4, 0.1])
  assert e.gridify().isPadding.sum() == 0
  with pytest.raises(ValueError):
    EBSD3.load(files[:2], z=[0.0])
