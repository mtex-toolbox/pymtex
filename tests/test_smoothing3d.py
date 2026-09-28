# the surfaces of 3D grains smoothed, reduced and refined, against MATLAB master
# (bench/matlab/probe_smoothing3d.m)

import numpy as np

from mtex import EBSD3square, calcGrains, crystalFrame, degree, phaseList, reduceBoundary, refineBoundary, rotation
from mtex import smoothBoundary, taubinFilter, vector3d



# a 4 x 4 x 4 volume split into two grains along x
def grains():
  r = np.zeros((4, 4, 4, 4))
  r[..., 0] = 1
  r[2:] = rotation.byAxisAngle(vector3d(0, 0, 1), 30 * degree).data
  e = EBSD3square(rotation(r), np.ones((4, 4, 4), dtype=np.int8), phaseList([crystalFrame("m-3m", mineral="Iron")]))
  return calcGrains(e, angle=10 * degree)


def test_a_plane_boundary_stays_where_it_is():
  g = grains()
  s = g.smoothBoundary(3)
  # a flat boundary between two grains and the surface of the box do not move
  assert np.allclose(s.volume, g.volume) and np.allclose(s.boundary.allV, g.boundary.allV)
  assert np.allclose(smoothBoundary(g, taubinFilter(5)).volume, g.volume)
  r = g.reduceBoundary()
  assert len(r.boundary) < len(g.boundary) and np.allclose(r.volume, g.volume)
  f = refineBoundary(g)
  assert len(f.boundary) == 4 * len(g.boundary) and np.allclose(f.volume, g.volume)
  assert np.allclose(reduceBoundary(g, quadric=True).volume, g.volume)


def test_IN100_as_MATLAB_smooths_it(in100):
  g = in100[1]
  r = g.reduceBoundary()
  assert len(r.F) == 282646 and len(r.boundary.idV) == 122476 and np.isclose(r.volume.sum(), 15625)
  assert (r.F[:2] + 1).tolist() == [[109575, 1, 109574], [109576, 109574, 1]]
  s = r.smoothBoundary(taubinFilter(10))
  assert np.allclose(s.volume[:3], [0.006625666, 0.016074441, 0]) and np.isclose(s.boundary.area.sum(), 16786.064893)
  l = g.smoothBoundary(3)
  assert np.allclose(l.volume[:3], [0.010216056, 0.021717784, 0.011437194]) and np.isclose(l.boundary.area.sum(), 18603.877946)
  f = g.refineBoundary()
  assert len(f.F) == 2995544 and len(f.boundary.allV) == 1435777
