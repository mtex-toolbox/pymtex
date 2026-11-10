# the kinds of the notIndexed parts, MATLAB's checkNotIndexedKinds in check_calcGrainsCases.m
import numpy as np

from mtex import EBSD, phaseList, crystalFrame, rotation, zvector, degree, classifyNotIndexed
from mtex.maps.kinds import KINDS, kindCodes

CS = crystalFrame("m-3m", mineral="test")
THR = 5 * degree


# a map of nr x nc pixels in blocks of bs, each block one orientation 7 degrees apart, the pixels
# notIndexed(r, c) says left notIndexed; r and c count from 1 as MATLAB's meshgrid
def blockMap(nr, nc, bs, notIndexed):
  r, c = np.meshgrid(np.arange(1, nr + 1), np.arange(1, nc + 1), indexing="ij")
  blockId = (r - 1) // bs + (nr // bs) * ((c - 1) // bs)
  rot = rotation.byAxisAngle(zvector, (blockId.reshape(-1) + 1) * 7 * degree)
  pos = np.stack([c.reshape(-1), r.reshape(-1), np.zeros(nr * nc)], axis=-1).astype(float)
  ph = np.where(notIndexed(r, c).reshape(-1), 0, 1).astype(np.int8)
  return EBSD(pos, rot, ph, phaseList(CS)), r, c


# MATLAB's 48 x 48 map of 4 x 4 blocks of 12 pixels: the crack cuts the second block column, the
# pore sits inside block (2, 3), the pocket where four blocks meet; with the three masks
def kindsMap():
  masks = lambda r, c: ((c >= 15) & (c <= 21), np.hypot(r - 18.5, c - 30.5) <= 4.3, np.hypot(r - 36.5, c - 36.5) <= 4.3)
  e, r, c = blockMap(48, 48, 12, lambda r, c: np.any(masks(r, c), axis=0))
  return e, dict(zip(("crack", "pore", "pocket"), masks(r, c)))


def test_a_pore_a_pocket_and_a_crack_are_told_apart():
  e, masks = kindsMap()
  kinds = classifyNotIndexed(e, threshold=THR)[0].reshape(48, 48)
  assert all(np.array_equal(kinds == k, m) for k, m in masks.items()) and set(np.unique(kinds)) == {*masks, "indexed"}


def test_the_parts_table_measures_the_parts():
  e, masks = kindsMap()
  p = classifyNotIndexed(e, threshold=THR)[1]
  k = {kind: int(np.flatnonzero(p["kind"] == kind)[0]) for kind in masks}
  assert [p["area"][k[kind]] for kind in masks] == [int(m.sum()) for m in masks.values()]
  assert p["elong"][k["crack"]] >= 5 and p["gap"][k["crack"]] <= p["rim"][k["crack"]] and p["border"][k["crack"]]
  assert p["share"][k["pore"]] == 1 and p["share"][k["pocket"]] == 0.25 and not p["border"][k["pore"]]


def test_a_kind_given_by_name_replaces_the_computed_one():
  e, masks = kindsMap()
  kinds = classifyNotIndexed(e, threshold=THR)[0]
  given = np.where(kinds == "crack", "pocket", kinds)
  assert np.array_equal(classifyNotIndexed(e, threshold=THR, kinds=given)[0], given)


def test_what_lies_between_two_parts_of_a_crack_is_crack():
  # a 7 wide crack down a 96 row map, pinched to 3 wide over six rows: the closing fills the neck,
  # which lies between the two crack parts and so is crack; the four corner cells beside the neck
  # are filled too, each a piece touching one part alone, so they are leaves and stay speck
  e, r, c = blockMap(96, 48, 12, lambda r, c: ((c >= 15) & (c <= 21) & ((r < 46) | (r > 51))) | ((c >= 17) & (c <= 19)))
  kinds = classifyNotIndexed(e, threshold=THR)[0].reshape(96, 48)
  corner = ((r == 45) | (r == 52)) & ((c == 15) | (c == 21))
  assert np.array_equal(kinds == "crack", ~e.isIndexed.reshape(96, 48) & ~corner) and np.array_equal(kinds == "speck", corner)


def test_a_thin_region_is_a_speck_and_a_corner_outside_the_hull_a_void():
  # a disc of radius 1.5 is filled by the closing; of a 3 x 3 corner the cells beyond the hull edge
  # from (1, 4) to (4, 1) are void, the rest filled
  disc, corner = lambda r, c: np.hypot(r - 18.5, c - 30.5) <= 1.5, lambda r, c: (r <= 3) & (c <= 3)
  e, r, c = blockMap(48, 48, 12, lambda r, c: disc(r, c) | corner(r, c))
  kinds = classifyNotIndexed(e, threshold=THR)[0].reshape(48, 48)
  assert np.array_equal(kinds == "void", corner(r, c) & (r + c < 5)) and np.array_equal(kinds == "speck", disc(r, c) | corner(r, c) & (r + c >= 5))


def test_kind_codes_of_names_and_numbers():
  assert np.array_equal(kindCodes(np.array([["crack", "pore"], ["indexed", "pocket"]])), [3, 1, 0, 2])
  assert np.array_equal(kindCodes([4, 0]), [4, 0]) and kindCodes(None) is None and list(KINDS) == ["pore", "pocket", "crack", "void", "speck", "indexed"]
