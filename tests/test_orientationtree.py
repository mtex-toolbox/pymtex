# the quaternion k-d tree search of the SO(3) grids: nearest, k nearest and radius, up to
# symmetry, against a brute-force symmetry-reduced distance to every grid point
import numpy as np
import pytest

import mtex
from mtex import crystalFrame, equispacedSO3Grid, orientation, specimenFrame, superfibonacciSO3Grid
from scipy.spatial import cKDTree

rng = np.random.default_rng(9)


def _grids(name):
  cs = crystalFrame(name)
  return [equispacedSO3Grid(cs, resolution=10 * mtex.degree), superfibonacciSO3Grid(cs, resolution=10 * mtex.degree)]


@pytest.mark.parametrize("name", ["m-3m", "222", "6/mmm"])
def test_nearest_is_the_reduced_closest(name):
  for g in _grids(name):
    o = orientation.rand(400, g.frameA, rng=rng)
    ind, d = mtex.find(g, o)
    dall = mtex.angle(o[:, None], g[None, :])
    assert np.allclose(d, np.cos(dall.min(axis=1) / 2), atol=1e-9)
    assert np.allclose(d, np.cos(mtex.angle(o, g[ind]) / 2), atol=1e-9)


@pytest.mark.parametrize("name", ["m-3m", "222"])
def test_knn_matches_brute(name):
  for g in _grids(name):
    o = orientation.rand(200, g.frameA, rng=rng)
    ind, d = mtex.find(g, o, 4)
    dall = mtex.angle(o[:, None], g[None, :])
    assert ind.shape == (200, 4)
    assert np.allclose(np.sort(d, axis=1), np.sort(dall, axis=1)[:, :4], atol=1e-9)


@pytest.mark.parametrize("name", ["m-3m", "222", "-3m"])
def test_radius_matches_brute(name):
  eps = 12 * mtex.degree
  for g in _grids(name):
    o = orientation.rand(300, g.frameA, rng=rng)
    I, D = mtex.find(g, o, eps)
    dall = mtex.angle(o[:, None], g[None, :])
    assert I.shape == (300, g.size)
    for q in range(0, 300, 30):
      ref = set(np.flatnonzero(dall[q] <= eps + 1e-9).tolist())
      assert set(I[q].indices.tolist()) == ref
      got = np.array(sorted(ref))
      if len(got):
        assert np.allclose(D[q].toarray()[0][got], dall[q][got], atol=1e-9)


def test_grid_finds_itself_and_shapes():
  cs = crystalFrame("m-3m")
  g = equispacedSO3Grid(cs, resolution=6 * mtex.degree)
  ii, dd = mtex.find(g, g[:100])
  assert np.array_equal(ii, np.arange(100)) and np.min(dd) > 1 - 1e-12
  o = orientation.rand((3, 4), cs, rng=rng)
  ind, d = mtex.find(g, o)
  assert ind.shape == (3, 4) and d.shape == (3, 4)


def test_radius_empty_when_far():
  cs = crystalFrame("m-3m")
  g = superfibonacciSO3Grid(cs, resolution=8 * mtex.degree)
  o = orientation.rand(50, cs, rng=rng)
  I, D = mtex.find(g, o, 0.01 * mtex.degree)
  assert I.nnz == 0
