# the lattice of a volume: the translations inferred along x, y, z, the index checked
import numpy as np
import pytest

from mtex.maps3d.voxellattice import voxelLattice


# a 4 x 5 x 6 lattice of translations D, its positions shuffled by perm
def shuffledLattice():
  rng = np.random.default_rng(1)
  D = np.array([[0.3, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.2]])
  ijk = np.stack(np.unravel_index(np.arange(4 * 5 * 6), (4, 5, 6)), axis=1)
  P = np.array([1.0, -2.0, 0.5]) + ijk @ D
  perm = rng.permutation(len(P))
  return D, ijk, P, perm, voxelLattice.fromPositions(P[perm])


def test_box_indexesEveryVoxelInCOrder():
  l = voxelLattice._box(np.zeros(3), np.eye(3), (2, 3, 4))
  assert np.array_equal(l.ijk, np.stack(np.unravel_index(np.arange(24), (2, 3, 4)), axis=1)) and l.shape == (2, 3, 4)


def test_fromPositions_infersTheTranslationsAndTheIndex():
  D, ijk, P, perm, l = shuffledLattice()
  assert np.allclose(l.D, D) and l.shape == (4, 5, 6)
  assert np.allclose(l.positions(), P[perm]) and np.array_equal(l.ijk, ijk[perm])


def test_unitCell_listsTheCornersAsMATLAB():
  # the unit cell as MATLAB lists its corners
  D, ijk, P, perm, l = shuffledLattice()
  assert np.allclose(l.unitCell[0], [0.15, 0.25, 0.1]) and np.allclose(l.unitCell[6], [-0.15, -0.25, -0.1])


def test_subset_keepsTheTranslations():
  # a subset keeps the translations
  D, ijk, P, perm, l = shuffledLattice()
  assert np.allclose(l.subset(perm[:10]).D, D)


def shearedLattice():
  D = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 2.0]])
  return D, np.stack(np.unravel_index(np.arange(27), (3, 3, 3)), axis=1)


def test_fromPositions_indexesAShearedLattice():
  D, ijk = shearedLattice()
  l = voxelLattice.fromPositions(ijk @ D, D)
  assert np.array_equal(l.ijk, ijk)


def test_fromPositions_refusesPositionsOffTheLattice():
  D, ijk = shearedLattice()
  with pytest.raises(ValueError):
    voxelLattice.fromPositions(ijk @ D + np.random.default_rng(0).uniform(-0.4, 0.4, (27, 3)), np.eye(3))
  with pytest.raises(ValueError):
    voxelLattice.fromPositions(np.zeros((3, 3)))
