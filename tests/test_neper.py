# the Neper interface: tessellations simulated and read back as grains
import numpy as np
import pytest

from mtex import KurdjumovSachs, crystalFrame, degree, neper, orientation, uniformODF

np_ = neper.init()
pytestmark = pytest.mark.skipif(not np_.available(), reason="neper does not run here")


def test_simulated_grains():
  n = neper.init()
  n.geometry = "cube(1,1,1)"
  ori = orientation.rand(20, crystalFrame("m-3m"), rng=np.random.default_rng(0))
  g = n.simulateGrains(ori)
  assert len(g) == 20 and np.isclose(g.volume.sum(), 1) and np.all(g.volume > 0)
  assert np.allclose(np.asarray(g.meanOrientation.angle(ori)), 0, atol=1e-6)


def test_child_lamellae():
  n = neper.init()
  n.geometry = "cube(1,1,1)"
  p2c = orientation.KurdjumovSachs(crystalFrame("432", mineral="fcc"), crystalFrame("432", mineral="bcc"))
  g = n.simulateChildGrains(3, p2c, uniformODF(p2c.CS), rng=np.random.default_rng(0))
  assert len(g) > 3 and np.isclose(g.volume.sum(), 1)
