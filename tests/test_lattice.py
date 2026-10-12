# M1: the lattice a map is measured on, the Julia L1 to L6 and the lattice parts of check_ebsdGrid
import numpy as np
import pytest

from mtex import lattice, latticeBasis, yvector, vector3d
from mtex.maps.lattice import cellShapeOf, mapPlane
from mtex.maps.polygon import signedArea2, voronoiCell
from conftest import close


# the positions of an n x n patch of the lattice A, optionally distorted smoothly
def latticePositions(A, n, distort=0.0, dtype=np.float64):
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  x, y = A[0, 0] * i + A[0, 1] * j, A[1, 0] * i + A[1, 1] * j
  x, y = x * (1 + distort * y / n), y * (1 + distort * x / n)
  return np.stack([x, y, np.zeros_like(x)], axis=-1).astype(dtype)


SQ = np.eye(2)
RECT = np.diag([0.8, 1.4])
HEXA = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])


def rot2(w):
  return np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])


def test_L1_shape_and_basis():
  for A, S in ((SQ, "square"), (RECT, "square"), (HEXA, "hex")):
    for w in (0.0, 0.4, 1.1, 2.3):
      l = lattice(latticePositions(rot2(w) @ A, 20))
      assert l.shape == S
      # the basis holds the two shortest translations, whatever the map's rotation
      assert close(l.dxy, (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1])) / 2, tol=1e-6)
      assert close(abs(np.linalg.det(l.A)), abs(np.linalg.det(A)), tol=1e-6)
      assert len(l.stencil) == (6 if S == "hex" else 4)
  # a lattice needs two independent directions
  with pytest.raises(ValueError):
    lattice(np.stack([np.arange(10.0), np.zeros(10), np.zeros(10)], axis=-1))


def test_L2_the_index_is_a_bijection_onto_the_stencil_graph():
  for A in (SQ, RECT, HEXA):
    for w in (0.0, 0.4, 2.3):
      P3 = latticePositions(rot2(w) @ A, 20)
      l = lattice(P3)
      assert l.ij.shape == (400, 2) and l.ij.dtype == np.int32
      assert len(np.unique(l.ij, axis=0)) == 400 and l.ij.min() == 0
      # the ideal position of every index is the measured one
      assert np.max(np.linalg.norm(l.position(l.ij) - P3, axis=1)) < 1e-9
      # a neighbour in the index is a neighbour in space, however the map is turned
      idx = {tuple(ij): k for k, ij in enumerate(l.ij)}
      P = l.plane.to2d(P3)
      for k in range(0, 400, 7):
        for o in l.stencil:
          j = idx.get((l.ij[k, 0] + o[0], l.ij[k, 1] + o[1]))
          if j is not None:
            assert np.linalg.norm(P[j] - P[k]) <= 1.5 * l.dxy


def test_L3_a_smooth_distortion_keeps_the_index():
  for A in (SQ, HEXA):
    l0 = lattice(latticePositions(A, 20))
    l = lattice(latticePositions(A, 20, distort=0.08))
    assert l.shape == l0.shape and np.array_equal(l.ij, l0.ij)
  # a distortion the affine model cannot absorb is grown over the neighbour graph
  P3 = latticePositions(SQ, 40, distort=0.6)
  l = lattice(P3)
  assert len(np.unique(l.ij, axis=0)) == 1600
  idx = {tuple(ij): k for k, ij in enumerate(l.ij)}
  P = l.plane.to2d(P3)
  for k in range(0, 1600, 13):
    for o in l.stencil:
      j = idx.get((l.ij[k, 0] + o[0], l.ij[k, 1] + o[1]))
      if j is not None:
        assert np.linalg.norm(P[j] - P[k]) <= 1.7 * l.dxy


def test_L4_the_unit_cell_is_the_Voronoi_cell_of_the_lattice():
  for A, nc in ((SQ, 4), (RECT, 4), (HEXA, 6)):
    l = lattice(latticePositions(A, 12))
    cell = l.unitCell
    assert cell.shape == (nc, 3)
    # its area is the area of one lattice cell, |det A|
    V = l.plane.to2d(cell + l.plane.o)
    assert close(abs(signedArea2(V)) / 2, abs(np.linalg.det(A)), tol=1e-6)
    # and the basis reads back off the cell
    B, stencil, dxy = latticeBasis(cell)
    assert len(stencil) == nc and close(abs(np.linalg.det(B)), abs(np.linalg.det(A)), tol=1e-6) and close(dxy, l.dxy, tol=1e-6)
    # a known basis gives the same lattice
    l2 = lattice(latticePositions(A, 12), basis=(B, l.shape))
    assert np.array_equal(l2.ij, l.ij)
  with pytest.raises(ValueError):
    latticeBasis(np.zeros((5, 3)))


def test_L5_out_of_the_xy_plane_and_in_single_precision():
  # the same square lattice laid into the xz plane
  i, j = np.meshgrid(np.arange(10.0), np.arange(10.0), indexing="xy")
  l = lattice(np.stack([i.reshape(-1), np.zeros(100), j.reshape(-1)], axis=-1))
  assert l.shape == "square"
  assert close(abs(l.plane.normal @ yvector.data), 1)
  assert len(np.unique(l.ij, axis=0)) == 100
  l32 = lattice(latticePositions(HEXA, 10, dtype=np.float32))
  assert l32.shape == "hex" and l32.A.dtype == np.float32 and l32.origin.dtype == np.float32
  assert len(np.unique(l32.ij, axis=0)) == 100


def test_L6_the_shape_counts_neighbours_not_equal_steps():
  # a square lattice is exactly degenerate: any shear splits its four cocircular corners
  # into slivers, which must not turn it into a six neighbour lattice
  assert cellShapeOf([1.0, 0.0], [0.03, 1.0]) == "square"
  assert cellShapeOf([1.0, 0.0], [0.0, 3.0]) == "square"
  # a sheared hexagonal lattice keeps its six neighbours although no two steps agree
  assert cellShapeOf([1.0, 0.0], [0.55, 0.83]) == "hex"
  assert cellShapeOf([1.0, 0.0], [0.5, np.sqrt(3) / 2]) == "hex"


def test_map_plane_and_polygon():
  P = latticePositions(SQ, 5) + np.array([10.0, 20.0, 0.0])
  pl = mapPlane(P)
  # the origin is the foot of the specimen origin, so the coordinates in the plane are the specimen's own
  assert close(pl.o, [0, 0, 0]) and close(pl.to2d([13.0, 24.0, 0.0]), [13.0, 24.0]) and close(pl.from2d([13.0, 24.0]), [13, 24, 0])
  assert close(pl.normal, [0, 0, 1])
  # the Voronoi cell of the square lattice is the unit square, anticlockwise
  vs = voronoiCell([0.0, 0.0], [[1, 0], [-1, 0], [0, 1], [0, -1]], 3.0)
  assert vs.shape == (4, 2) and close(abs(signedArea2(vs)), 2) and signedArea2(vs) > 0
  assert close(np.max(np.abs(vs)), 0.5)
  # and of the hexagonal one a hexagon of area sqrt(3)/2
  nb = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]]) @ HEXA.T
  vs = voronoiCell([0.0, 0.0], nb, 3.0)
  assert vs.shape == (6, 2) and close(signedArea2(vs) / 2, np.sqrt(3) / 2)


def test_the_display_names_shape_sites_spacing_and_basis():
  l = lattice(latticePositions(HEXA, 6))
  s = repr(l)
  assert s.startswith("lattice (hex)") and "sites  : 36" in s and "spacing: 1" in s and "basis  : (1, 0), (0.5, 0.866)" in s


# duplicate positions on an exact grid: the affine index is kept, the neighbour graph not searched
def test_duplicates_on_an_exact_grid_keep_the_affine_index(monkeypatch):
  import sys
  L = sys.modules["mtex.maps.lattice"]
  pos = latticePositions(np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]]), 20)
  pos = np.concatenate([pos, np.repeat(pos[:1], 5, axis=0)])
  monkeypatch.setattr(L, "growIndex", lambda *a: pytest.fail("the neighbour graph was searched"))
  l = lattice(pos)
  assert l.shape == "hex" and (l.ij[-5:] == l.ij[0]).all()
  assert len(np.unique(l.ij[:-5], axis=0)) == len(pos) - 5


# scattered points lie on no lattice: fewer than nine in ten of their neighbouring steps are whole
# cells, where growing an index over them would put two in one cell
def test_scattered_points_are_refused_before_an_index_is_grown():
  rng = np.random.default_rng(5)
  with pytest.raises(ValueError):
    lattice(np.c_[rng.random((3000, 2)) * 100, np.zeros(3000)])


def test_an_index_grown_over_the_neighbours_is_the_rounded_one_on_a_rigid_lattice():
  from mtex.maps.lattice import growIndex
  A = np.array([[1.0, 0.3], [0.0, 0.9]])
  i, j = np.meshgrid(np.arange(-3, 9), np.arange(5), indexing="ij")
  ij = np.stack([i.reshape(-1), j.reshape(-1)], axis=1)
  P = (ij @ A.T)[np.random.default_rng(2).permutation(len(ij))]
  assert np.array_equal(growIndex(P, A), np.rint((P - P[0]) @ np.linalg.inv(A).T).astype(int))
