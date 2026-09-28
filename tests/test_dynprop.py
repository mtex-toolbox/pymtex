import numpy as np

from mtex import EBSD, crystalFrame, rotation, vector3d
from mtex.maps.dynprop import dynProp


def test_the_display_lists_every_property_with_its_shape():
  p = dynProp(mad=np.zeros((2, 3)), axis=vector3d(np.ones((4, 3))))
  lines = repr(p).splitlines()
  assert lines == [" mad: [2×3 float64]", "axis: [4 vector3d]"]
  assert repr(dynProp()) == "no properties"


def test_container_property_follows_selection():
  n = 5
  pos = np.column_stack([np.arange(n, dtype=float), np.zeros(n), np.zeros(n)])
  ebsd = EBSD(pos, rotation.rand(n), np.ones(n, dtype=int), ['notIndexed', crystalFrame('m-3m')])
  ebsd.myAxis = vector3d(np.arange(3 * n, dtype=float).reshape(n, 3))
  assert isinstance(ebsd.prop, dynProp) and isinstance(ebsd.myAxis, vector3d)
  sub = ebsd[np.array([True, False, True, False, False])]
  assert isinstance(sub.myAxis, vector3d) and np.allclose(sub.myAxis.data, [[0, 1, 2], [6, 7, 8]])
  assert isinstance(ebsd.copy().myAxis, vector3d)
