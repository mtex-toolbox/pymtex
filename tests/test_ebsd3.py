# EBSD3: the container, gridify, selections, interp and slice, against MATLAB master
# (bench/matlab/probe_ebsd3.m)
import os

import numpy as np
import pytest

from mtex import EBSD3, EBSD3square, crystalFrame, phaseList, plane3d, rotation, vector3d

DREAM3D = os.path.join(os.path.dirname(__file__), "..", "..", "master", "data", "EBSD", "SmallIN100_MeshStats.dream3d")


def volume():
  rng = np.random.default_rng(3)
  rot = rotation(rng.standard_normal((4, 5, 6, 4)))
  ph = np.ones((4, 5, 6), dtype=np.int8)
  ph[0, 0, 0] = 0
  return EBSD3square(rot, ph, phaseList([crystalFrame("m-3m", mineral="Iron")]), (0.5, 0.5, 1.0), prop={"ci": rng.random((4, 5, 6))})


def test_the_container_and_its_lattice():
  e = volume()
  assert e.isGridded and e.shape == (4, 5, 6) and e.size == 120
  assert np.allclose(e.d1.data, [0.5, 0, 0]) and np.allclose(e.d3.data, [0, 0, 1]) and (e.dx, e.dy, e.dz) == (0.5, 0.5, 1.0)
  assert np.allclose(e.pos.data[1, 2, 3], [0.5, 1.0, 3.0]) and np.allclose(e.extent(), [0, 1.5, 0, 2, 0, 5])
  assert np.all(np.isnan(e.N.data)) and "EBSD3" in repr(e) and "voxel     : 0.5 × 0.5 × 1" in repr(e)
  # a voxel, a phase, a condition: lists that keep their lattice
  v = e[1, 2, 3]
  assert isinstance(v, EBSD3) and v.size == 1
  iron = e["Iron"]
  assert isinstance(iron, EBSD3) and iron.size == 119 and iron.lattice.shape == (4, 5, 6)
  assert np.allclose(iron.d2.data, [0, 0.5, 0])


def test_gridify_puts_a_selection_back_with_padding():
  e = volume()
  sub = e[e.prop["ci"] > 0.5]
  g, newId = sub.gridify(return_index=True)
  assert g.isGridded and g.shape == (4, 5, 6)
  assert np.array_equal(g.rotations.data.reshape(-1, 4)[newId], sub.rotations.data.reshape(-1, 4))
  assert int(g.isPadding.sum()) == 120 - sub.size and np.all(g.phaseId.reshape(-1)[g.isPadding.reshape(-1)] == 0)
  assert np.all(np.isnan(g.prop["ci"][g.isPadding]))
  # a list from positions alone finds its lattice
  f = EBSD3(e.pos.data.reshape(-1, 3), rotation(e.rotations.data.reshape(-1, 4)), e.phaseId.reshape(-1), e.CSList)
  assert f.lattice.shape == (4, 5, 6) and np.allclose(f.gridify().rotations.data, e.rotations.data)


def test_interp_takes_the_nearest_voxel_and_nothing_outside():
  e = volume()
  s = e.interp(vector3d([[0.4, 1.1, 2.6], [9.0, 0, 0]]))
  assert s.size == 2 and s.phaseId[0] == 1 and s.phaseId[1] == 0
  assert np.allclose(s.rotations.data[0], e.rotations.data[1, 2, 3])


def test_a_slice_through_a_layer_is_a_gridded_map():
  e = volume()
  cut = e.slice(plane3d(vector3d(0, 0, 1), vector3d(0, 0, 2)))
  assert cut.isGridded and cut.ndim == 2
  assert cut.isIndexed.sum() == 20


@pytest.mark.skipif(not os.path.exists(DREAM3D), reason="the IN100 volume of ../master/data")
def test_the_IN100_volume_as_MATLAB_reads_it():
  e = EBSD3.load(DREAM3D)
  assert e.shape == (100, 100, 100) and np.allclose(e.d1.data, [0.25, 0, 0])
  # the voxels probe_ebsd3.m prints, MATLAB counting from one
  for (i, j, k), p, g, q in (((1, 1, 1), (-37, 10.25, -29.25), 793, (0.68155782, -0.26168791, -0.58734516, 0.34931939)),
                             ((50, 30, 70), (-24.75, 17.5, -12), 714, (-0.20691131, -0.97421194, 0.07778052, 0.04526576)),
                             ((13, 87, 42), (-34, 31.75, -19), 173, (0.49013488, -0.49994486, -0.54695766, 0.45897739))):
    assert np.allclose(e.pos.data[i - 1, j - 1, k - 1], p) and e.grainId[i - 1, j - 1, k - 1] == g
    assert abs(abs(np.dot(e.rotations.data[i - 1, j - 1, k - 1], q)) - 1) < 1e-6
  s = e.slice(plane3d(vector3d(0, 0, 1), vector3d(0, 0, -20)))
  assert s.shape == (102, 102) and s.isIndexed.sum() == 10000 and np.nansum(s.grainId) == 4724611
  s = e.slice(plane3d(vector3d(1, 1, 1), vector3d(-25, 22, -17)))
  assert s.shape == (145, 165) and s.isIndexed.sum() == 12677


def test_repr_givesTheSharesOfTheMeasuredVoxels():
  e = volume()
  g = e[e.prop["ci"] > 0.5].gridify()
  assert "57 (98%)" in repr(g)


@pytest.mark.parametrize("how", ["rotate", "keepXY", "keepEuler", "shift"])
def test_a_moved_volume_is_a_volume_on_its_lattice(how):
  # the lattice carried through the motion is the one the moved positions give
  rng = np.random.default_rng(2)
  e = EBSD3square(rotation.rand(4, 3, 2, rng=rng), None, [crystalFrame("m-3m")], dxyz=(0.5, 0.25, 2.0), origin=(1.0, -2.0, 0.5))
  r = rotation.byAxisAngle(vector3d(1, 2, 3), 0.7)
  m = {"rotate": lambda: e.rotate(r), "keepXY": lambda: e.rotate(r, keepXY=True), "keepEuler": lambda: e.rotate(r, keepEuler=True),
       "shift": lambda: e.shift(vector3d(1, 2, 3))}[how]()
  assert type(m) is EBSD3 and m.shape == e.shape and np.array_equal(m.lattice.ijk, e.lattice.ijk)
  assert np.allclose(m.lattice.positions().reshape(m.shape + (3,)), m.pos.data, atol=1e-12)
  turned = how in ("rotate", "keepXY")
  assert np.allclose(m.rotations.data, (r * e.rotations).data if turned else e.rotations.data)
  assert np.allclose(m.pos.data, e.pos.data) == (how == "keepXY")
