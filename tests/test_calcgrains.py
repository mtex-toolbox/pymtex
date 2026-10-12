# M3: calcGrains, the Julia S4 to S10 and the parts of check_calcGrainsCases that need no polygons
import os

import numpy as np
import pytest

from mtex import (EBSD, phaseList, crystalFrame, orientation, rotation, vector3d, xvector, yvector, zvector, degree, angle,
                  calcGrains, gbcAngle, gridify)
from conftest import close

CS = crystalFrame("m-3m", mineral="test")
LOCAL = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data")
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


def test_grains_from_the_grain_ids_stored_with_a_planar_map():
  # one orientation everywhere, so only the stored ids part the grains; an id of 0 is no grain
  from mtex import gbcCustom
  for flag in ({"grainId": True}, {"angle": "grainId"}, {"criterion": None}):
    e = blockMap(SQ3, 2, 5, step=0)
    i = np.arange(len(e)) % 10
    e.prop["grainId"] = np.where(i < 5, 1, 2).astype(float)
    e.prop["grainId"][0] = 0
    if "criterion" in flag:
      g = calcGrains(e, gbcCustom("grainId", 0.5))
    else:
      g = calcGrains(e, **flag)
    indexed = g[g.phaseId > 0]
    # the criterion alone only compares the ids, so the pixel of id 0 is a grain of its own
    assert sorted(indexed.numPixel.tolist()) == ([49, 50] if "criterion" not in flag else [1, 49, 50])
    assert calcGrains(blockMap(SQ3, 2, 5, step=0)).size == 1


def test_grain_ids_must_be_stored():
  with pytest.raises(ValueError, match="grainId"):
    calcGrains(blockMap(SQ3, 2, 5), grainId=True)


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
  # the map calcGrains returned segments again into the same grains, the absorbed pixels
  # taken as the not indexed pixels they were
  gid = e.grainId.copy()
  g2 = calcGrains(e, angle=5 * degree)
  assert len(g2) == len(g) and np.array_equal(e.grainId, gid) and np.all(e.isIndexed[hole])
  # the same hole straddling the boundary between two grains: the pixels the boundary
  # runs through belong to neither and keep zero
  ph2 = np.ones(n * n, dtype=np.int8)
  ph2[np.hypot(i - 10, j - 15) <= 1.5] = 0
  e3 = EBSD(pos, rot, ph2, phaseList(CS))
  calcGrains(e3, angle=5 * degree)
  assert (e3.grainId == 0).sum() > 0 and np.all(e3.grainId[~e3.isIndexed] >= 0)


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


def test_the_compiled_absorption_is_the_numpy_one():
  # holes of several sizes in and across grains: the compiled corner search gives every pixel
  # the grain the Python loop gives it
  e = blockMap(SQ3, 5, 8, notIndexed=((8, 8, 1.5), (16, 20, 2.5), (31, 7, 1.0), (24, 24, 3.2)))
  a, b = e.copy(), e.copy()
  calcGrains(a, angle=5 * degree, backend="numba")
  calcGrains(b, angle=5 * degree, backend="numpy")
  assert np.array_equal(a.grainId, b.grainId) and (a.grainId > 0).sum() > (b.isIndexed & ~b.isPadding).sum() - 20


def test_the_compiled_grain_table_is_the_numpy_one():
  # blocks with a spread inside, holes and two phases: the same grains, and the mean
  # orientations and spreads of the compiled passes those of the NumPy form to 1e-12
  from mtex.maps.calcgrains import grainTable
  from mtex.maps.decomposition import decompose
  from mtex.maps.segment import segment
  e = blockMap(SQ3, 5, 8, notIndexed=((8, 8, 1.5), (24, 24, 3.2)))
  q = e.rotations.data.reshape(-1, 4)
  noise = rotation.byAxisAngle(vector3d(1, 2, 3), np.linspace(0, 2 * degree, len(q))).data.reshape(-1, 4)
  from mtex.geometry.kernels import qmul
  ph = e.phaseId.reshape(-1).copy()
  ph[(ph > 0) & (np.arange(len(ph)) % 97 < 40)] = 2
  e2 = EBSD(e.pos, rotation(qmul(q, noise)), ph, phaseList(CS, crystalFrame("6/mmm", mineral="hex")))
  c = decompose(e2)
  seg = segment(c, e2, gbcAngle(5 * degree), "numba")
  segN = segment(c, e2, gbcAngle(5 * degree), "numpy")
  assert np.array_equal(seg.grainId, segN.grainId) and seg.nGrains == segN.nGrains and np.array_equal(seg.connect, segN.connect)
  a, b = grainTable(seg, e2, backend="numba"), grainTable(seg, e2, backend="numpy")
  assert np.array_equal(np.asarray(a.phaseId), np.asarray(b.phaseId)) and np.array_equal(np.asarray(a.numPixel), np.asarray(b.numPixel))
  ma, mb = a.meanRotation.data.reshape(-1, 4), b.meanRotation.data.reshape(-1, 4)
  assert np.array_equal(np.isnan(ma), np.isnan(mb)) and np.nanmax(np.abs(ma - mb)) < 1e-12
  assert np.max(np.abs(np.asarray(a.GOS) - np.asarray(b.GOS))) < 1e-12 and np.asarray(a.GOS).max() > 0


def test_the_default_angle_is_matlabs_15_degrees():
  from mtex.io.sampledata import testGrainsMap as grainsMap
  assert np.isclose(gbcAngle().threshold, 15 * degree)
  a, b = grainsMap(), grainsMap()
  assert np.array_equal(calcGrains(a).id, calcGrains(b, angle=15 * degree).id) and np.array_equal(a.grainId, b.grainId)


@pytest.mark.skipif(not os.path.isdir(os.path.join(LOCAL, "EBSD")), reason="the MTEX data folder is not here")
def test_the_default_reconstruction_of_csl_is_matlabs():
  # MATLAB's calcGrains(ebsd) on the csl map, 15 degrees: 885 grains (calcgrains-release-audit.md)
  from mtex import mtexdata
  ebsd = mtexdata("csl", root=LOCAL, download=False)
  assert len(calcGrains(ebsd)) == 885


@pytest.mark.skipif(not os.path.isdir(os.path.join(LOCAL, "EBSD")), reason="the MTEX data folder is not here")
def test_no_indexed_grain_below_minPixel_survives_the_second_pass():
  # the second segmentation cut a 239 pixel grain of small into a one pixel fragment; it
  # becomes not indexed where it lies, as MATLAB's enforceMinPixel, on a turned map as well
  from mtex import mtexdata
  ebsd = mtexdata("small", root=LOCAL, download=False)
  for e in (ebsd.copy(), ebsd.copy().rotate(rotation.byAxisAngle(zvector, 30 * degree))):
    g = calcGrains(e, angle=15 * degree, minPixel=3)
    assert not np.any(g.isIndexed & (g.numPixel < 3))
    frag = ~np.asarray(g.isIndexed).reshape(-1) & (np.asarray(g.numPixel).reshape(-1) < 3)
    assert frag.any() and np.all(np.isnan(g.meanRotation.data.reshape(-1, 4)[frag, 0]))
    assert np.all(~e.isIndexed.reshape(-1)[np.isin(e.grainId.reshape(-1), g.id[frag])])


def test_the_grain_loops_follow_the_kernels_preference():
  from mtex import getMTEXpref, setMTEXpref
  from mtex.maps.kernels import grainKernel
  old = getMTEXpref("kernels")
  setMTEXpref("kernels", "numpy")
  try:
    assert grainKernel("relabel") is None
  finally:
    setMTEXpref("kernels", old)


def test_undersized_marks_the_indexed_pixels_of_small_grains():
  from mtex.maps.calcgrains import undersized
  assert np.array_equal(undersized(np.array([1, 1, 2, 0, 2, 2]), 2, np.array([True, True, True, True, False, True]), 3), [1, 1, 0, 0, 0, 0])


def test_a_zero_row_stays_zero_when_the_rows_are_normalised():
  from mtex.maps.calcgrains import _normalizeRows
  assert np.array_equal(_normalizeRows(np.array([[0.0, 0, 0, 0], [0, 3, 0, 4]])), [[0, 0, 0, 0], [0, 0.6, 0, 0.8]])
