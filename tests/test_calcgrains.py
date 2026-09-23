# M3: calcGrains, the Julia S4 to S10 and the parts of check_calcGrainsCases that need no polygons
import numpy as np
import pytest

from mtex import (EBSD, phaseList, crystalFrame, orientation, rotation, vector3d, xvector, yvector, zvector, degree, angle,
                  calcGrains, gbcAngle, gridify, transform)
from conftest import close

CS = crystalFrame("m-3m", mineral="test")
SQ3 = np.eye(2)
HEX3 = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])


# a map of nb x nb blocks of bs x bs pixels, each block one orientation
def blockMap(A, nb, bs, notIndexed=(), step=7 * degree, dtype=np.float64):
  n = nb * bs
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  pos = np.stack([A[0, 0] * i + A[0, 1] * j, A[1, 0] * i + A[1, 1] * j, np.zeros(n * n)], axis=-1).astype(dtype)
  rot = rotation.byAxisAngle(zvector, ((i // bs + (j // bs) * nb) * step).astype(dtype))
  ph = np.ones(n * n, dtype=np.int8)
  for ci, cj, r in notIndexed:
    ph[np.hypot(i - ci, j - cj) <= r] = 0
  return EBSD(pos, rot, ph, phaseList(CS))


def test_S4_grains_of_a_block_map():
  for A in (SQ3, HEX3):
    e = blockMap(A, 4, 5)
    g = calcGrains(e, angle=5 * degree)
    assert len(g) == 16 and np.all(g.numPixel == 25) and g.numPixel.sum() == len(e)
    assert g.GOS.max() < 1e-6                      # a block has one orientation
    assert np.array_equal(np.unique(e.grainId), np.arange(1, 17)) and e.grainId.shape == e.shape
    assert np.all(g.phaseId[e.grainId - 1] == e.phaseId)
    # the mean orientation of a block is its orientation
    m = g.meanOrientation
    assert m.CS is CS and m.SS is e.frame and len(m) == 16
    k = e.grainId[0] - 1
    assert angle(m[k], e.orientations[0]) < 1e-10
    assert len(e[g[:3]]) == 75 and g["test"].size == 16 and g["indexed"].size == 16 and g[["test"]].size == 16
    assert g[5].size == 1 and g[np.arange(16) < 4].size == 4 and g["id", 3].id[0] == 3


def test_S5_the_threshold_decides_how_much_is_one_grain():
  e = blockMap(SQ3, 4, 5)                        # neighbouring blocks differ by 7 degrees along a row
  assert len(calcGrains(e, angle=5 * degree)) == 16
  assert len(calcGrains(e, angle=8 * degree)) < 16
  assert len(calcGrains(e, angle=90 * degree)) == 1
  # a criterion object instead of a threshold, positionally as MATLAB writes it or by name
  assert len(calcGrains(e, criterion=gbcAngle(5 * degree))) == 16
  assert len(calcGrains(e, gbcAngle(5 * degree))) == 16
  assert len(calcGrains(e, threshold=5 * degree)) == 16
  # a pair of levels keeps the blocks in one grain with boundaries drawn inside
  assert len(calcGrains(e, angle=[90 * degree, 5 * degree])) == 1


def test_S6_not_indexed_regions_become_their_own_grains():
  e = blockMap(SQ3, 4, 5, notIndexed=((10, 10, 3.0),))
  g = calcGrains(e, angle=5 * degree)
  assert (~g.isIndexed).sum() == 1 and g.isIndexed.sum() == 16
  ni = g[~g.isIndexed]
  assert ni.numPixel[0] == (~e.isIndexed).sum() and np.isnan(ni.meanRotation.data).all()
  with pytest.raises(ValueError):
    ni.meanOrientation
  assert "notIndexed" in repr(g) and g.mineralList == ["notIndexed", "test"]


def test_S7_minPixel_culls_and_segments_again():
  # cut one block down to four pixels; it is dropped and its pixels become not indexed
  e = blockMap(SQ3, 4, 5)
  ph = e.phaseId.reshape(20, 20).copy()
  ph[0:5, 0:5] = 0
  ph[0:2, 0:2] = 1
  e = EBSD(e.pos, e.rotations, ph.reshape(-1), e.CSList)
  g0 = calcGrains(e.copy(), angle=5 * degree)
  assert g0.isIndexed.sum() == 16 and g0.numPixel[g0.isIndexed].min() == 4      # the stump is a grain
  g = calcGrains(e, angle=5 * degree, minPixel=10)
  assert g.isIndexed.sum() == 15 and np.all(g.numPixel[g.isIndexed] >= 10)     # the small one is gone
  # its pixels are marked not indexed, and those the closing gives to a neighbour take
  # its phase without a rotation, as MATLAB's [grains, ebsd] = calcGrains returns them
  given = (e.grainId.reshape(-1) > 0) & np.isnan(e.rotations.data.reshape(-1, 4)[:, 0])
  assert (~e.isIndexed).sum() + given.sum() == 25 and np.all(e.isIndexed.reshape(-1)[given])


def test_S9_a_filled_hole_is_absorbed_a_boundary_through_it_is_not():
  n = 30
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  pos = np.stack([i, j, np.zeros(n * n)], axis=-1).astype(float)
  rot = rotation.byAxisAngle(zvector, (i // 10 + (j // 10) * 3) * 20 * degree)
  # a hole inside one grain: the closing fills it, and every corner of each of its pixels
  # lands in that grain, so the pixels join it
  ph = np.ones(n * n, dtype=np.int8)
  ph[np.hypot(i - 15, j - 15) <= 1.5] = 0
  e = EBSD(pos, rot, ph, phaseList(CS))
  hole = ~e.isIndexed
  g = calcGrains(e, angle=5 * degree)
  assert (e.grainId == 0).sum() == 0 and len(g) == 9
  assert len(np.unique(e.grainId[hole])) == 1
  # the filled pixels take the phase of their grain and no rotation
  assert np.all(e.isIndexed[hole]) and np.all(np.isnan(e.rotations.data[hole.reshape(-1)]))
  # the same hole straddling the boundary between two grains: the pixels the boundary
  # runs through belong to neither and keep zero
  ph2 = np.ones(n * n, dtype=np.int8)
  ph2[np.hypot(i - 10, j - 15) <= 1.5] = 0
  e3 = EBSD(pos, rot, ph2, phaseList(CS))
  calcGrains(e3, angle=5 * degree)
  assert (e3.grainId == 0).sum() > 0 and np.all(e3.grainId[~e3.isIndexed] >= 0)


def test_S10_the_rim_closes_as_the_inside_does():
  # the edge of the map is not the edge of a hole: a cell on the rim is absorbed exactly
  # when a cell in its place further in would be; both gaps here are one cell, so both go
  e = blockMap(SQ3, 1, 7, notIndexed=((3, 3, 0.0), (0, 3, 0.0)))
  ix = e[e.isIndexed]
  assert len(ix) == 47
  for alpha in (3.1, 20.0):
    g = calcGrains(ix, angle=5 * degree, alpha=alpha)
    assert len(g) == 1 and g.numPixel[0] == 47


# ------------------------------------------------------------------ check_calcGrainsCases

CS1 = crystalFrame("1", mineral="test")


# nb x nb blocks of blk x blk pixels on a square lattice, one orientation per block, with a
# few not indexed holes and a cropped corner that exercises the map edge
def squareBlockGrid(blk, nb, oris, dxy=1.0):
  n = nb * blk
  r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
  blockId = (r // blk) * nb + c // blk
  rot = oris[blockId.reshape(-1)]
  ph = np.ones((n, n), dtype=np.int8)
  ph[7, 7] = ph[14, 19] = ph[21, 9] = 0
  ph[0:3, 0:3] = 0
  pos = np.stack([c.reshape(-1) * dxy, r.reshape(-1) * dxy, np.zeros(n * n)], axis=-1)
  return gridify(EBSD(pos, rot, ph.reshape(-1), phaseList(CS1)))


def hexBlockGrid(blk, nb, oris):
  n = nb * blk
  r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
  x = c + (r % 2) * 0.5
  y = r * np.sqrt(3) / 2
  blockId = (r // blk) * nb + c // blk
  rot = oris[blockId.reshape(-1)]
  pos = np.stack([x.reshape(-1), y.reshape(-1), np.zeros(n * n)], axis=-1)
  return gridify(EBSD(pos, rot, np.ones(n * n, dtype=np.int8), phaseList(CS1)))


def test_block_grids_with_holes_and_a_cropped_corner():
  oris = rotation.byAxisAngle(zvector, np.arange(1, 17) * 7 * degree)
  thr = 5 * degree
  for build in (squareBlockGrid, hexBlockGrid):
    e = build(6, 4, oris)
    g0 = calcGrains(e.copy(), angle=thr)
    assert g0.isIndexed.sum() == 16
    # the grain count survives a trapezoidal distortion and a rotation of the map
    x, y = e.x.reshape(-1), e.y.reshape(-1)
    xc, yc, yh = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2, (y.max() - y.min()) / 2
    eT = transform(e, lambda p: vector3d(xc + (p.x - xc) * (1 + 0.02 * (p.y - yc) / yh), p.y, p.z))
    gT = calcGrains(eT, angle=thr)
    assert len(gT) == len(g0) and np.array_equal(np.sort(gT.numPixel), np.sort(g0.numPixel))
    eR = e.rotate(27 * degree)
    gR = calcGrains(eR, angle=thr)
    assert len(gR) == len(g0) and np.array_equal(np.sort(gR.numPixel), np.sort(g0.numPixel))


def test_a_diagonal_neighbour_does_not_save_an_undersized_grain():
  # the sizing pass has to be as connected as the segmentation: four neighbours on a square lattice
  o2 = rotation.byAxisAngle(zvector, 30 * degree)
  n = 8
  rot = rotation.byAxisAngle(zvector, np.zeros((n, n)))
  rot[0:2, 0:2] = o2
  rot[2, 2] = o2
  r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
  e = EBSD(np.stack([c, r, np.zeros((n, n))], axis=-1).astype(float), rot, np.ones((n, n), dtype=np.int8), phaseList(CS1))
  g = calcGrains(e, angle=5 * degree, minPixel=2)
  gi = g[g.isIndexed]
  assert gi.size == 2 and gi.numPixel.min() >= 2
  # hex: a single stray pixel in the middle of a block
  e = hexBlockGrid(n, 1, rotation.byAxisAngle(zvector, np.zeros(1)))
  cx, cy = e.x.mean(), e.y.mean()
  k = np.argmin(np.hypot(e.x.reshape(-1) - cx, e.y.reshape(-1) - cy))
  rot = e.rotations.data.reshape(-1, 4).copy()
  rot[k] = o2.data
  e = EBSD(e.pos, rotation(rot.reshape(e.shape + (4,))), e.phaseId, e.CSList)
  g = calcGrains(e, angle=5 * degree, minPixel=2)
  assert g.isIndexed.sum() == 1 and g[g.isIndexed].numPixel[0] == n * n - 1


def test_gridify_padding_segments_like_the_list():
  oris = rotation.byAxisAngle(zvector, np.arange(1, 17) * 7 * degree)
  base = squareBlockGrid(6, 4, oris)
  hole = np.zeros(base.shape, dtype=bool)
  hole[9:16, 5:14] = True                         # spans four orientation blocks
  # the hole as missing measurements, as a list and padded back by gridify, and the same
  # cells measured but not indexed
  lst = base[~hole]
  a = gridify(lst)
  b = base.copy()
  b.phaseId[hole] = 0
  assert np.array_equal(a.isPadding, hole) and not b.isPadding.any()
  gL, gA, gB = (calcGrains(m, angle=5 * degree) for m in (lst, a, b))
  # the padding is no measurement: the grid segments as its list does, the hole is a hole
  assert len(gA) == len(gL) and len(gA.boundary) == len(gL.boundary)
  assert np.isclose(gA.boundary.segLength.sum(), gL.boundary.segLength.sum())
  assert not (~gA.isIndexed).any() and np.all(a.grainId[hole] == 0)
  # while measured notIndexed cells make a notIndexed grain
  assert (~gB.isIndexed).sum() == 1
