# the lattice of a volume: the translations inferred along x, y, z, the index checked
import numpy as np
import pytest

from mtex.maps3d.voxellattice import voxelLattice


def test_the_translations_and_the_index():
  rng = np.random.default_rng(1)
  D = np.array([[0.3, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.2]])
  ijk = np.stack(np.unravel_index(np.arange(4 * 5 * 6), (4, 5, 6)), axis=1)
  P = np.array([1.0, -2.0, 0.5]) + ijk @ D
  perm = rng.permutation(len(P))
  l = voxelLattice.fromPositions(P[perm])
  assert np.allclose(l.D, D) and l.shape == (4, 5, 6)
  assert np.allclose(l.positions(), P[perm]) and np.array_equal(l.ijk, ijk[perm])
  # the unit cell as MATLAB lists its corners, and a subset keeps the translations
  assert np.allclose(l.unitCell[0], [0.15, 0.25, 0.1]) and np.allclose(l.unitCell[6], [-0.15, -0.25, -0.1])
  assert np.allclose(l.subset(perm[:10]).D, D)


def test_a_sheared_lattice_and_the_refusals():
  D = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 2.0]])
  ijk = np.stack(np.unravel_index(np.arange(27), (3, 3, 3)), axis=1)
  l = voxelLattice.fromPositions(ijk @ D, D)
  assert np.array_equal(l.ijk, ijk)
  with pytest.raises(ValueError):
    voxelLattice.fromPositions(ijk @ D + np.random.default_rng(0).uniform(-0.4, 0.4, (27, 3)), np.eye(3))
  with pytest.raises(ValueError):
    voxelLattice.fromPositions(np.zeros((3, 3)))
