# M1: the phases of a map
import numpy as np
import pytest

from mtex import phaseList, crystalFrame
from mtex.maps.phaselist import distinctMinerals
from mtex.geometry.frames import notIndexedFrame


def test_the_not_indexed_frame_comes_first():
  cs, cs2 = crystalFrame("m-3m", mineral="Iron"), crystalFrame("6/mmm", mineral="Titanium")
  pl = phaseList(cs, cs2)
  assert len(pl) == 3 and isinstance(pl[0], notIndexedFrame) and pl[1] is cs and pl[2] is cs2
  assert pl.mineralList == ["notIndexed", "Iron", "Titanium"] and list(pl.indexedPhaseId) == [1, 2]
  assert pl.isIndexed(1) and not pl.isIndexed(0)
  # a given not indexed frame, or its name as MATLAB writes the list, stays where it is
  assert len(phaseList(["notIndexed", cs])) == 2 and phaseList(["notIndexed", cs])[0].mineral == "notIndexed"
  assert phaseList(notIndexedFrame("holes"), cs).mineralList == ["holes", "Iron"]
  assert len(phaseList()) == 1 and len(phaseList([])) == 1


def test_a_phase_is_named_by_its_mineral_its_frame_or_its_number():
  cs, cs2 = crystalFrame("m-3m", mineral="Iron"), crystalFrame("6/mmm", mineral="Titanium")
  pl = phaseList(cs, cs2)
  assert pl.phaseId("iron") == 1 and pl.phaseId(cs2) == 2 and pl.phaseId(2) == 2 and pl["IRON"] is cs
  assert "Iron" in pl and cs2 in pl and "Quartz" not in pl and 7 not in pl
  with pytest.raises(ValueError):
    pl.phaseId("Quartz")
  with pytest.raises(ValueError):
    pl.phaseId(crystalFrame("m-3m", mineral="Copper"))
  with pytest.raises(IndexError):
    pl[5]
  with pytest.raises(TypeError):
    phaseList(cs, 3)
  assert [f for f in pl] == pl.CSList
  s = repr(pl)
  assert "Iron" in s and "m-3m" in s and "Mineral" in s


def test_the_selection_names_are_the_minerals_then_indexed_and_not_indexed():
  pl = phaseList(crystalFrame("m-3m", mineral="Iron"), crystalFrame("6/mmm", mineral="Titanium"))
  assert pl.selectionNames() == ["Iron", "Titanium", "indexed", "notIndexed"]


def test_the_selection_names_of_a_boundary_add_every_pair_of_minerals():
  pl = phaseList(crystalFrame("m-3m", mineral="Iron"), crystalFrame("6/mmm", mineral="Titanium"))
  assert pl.selectionNames(pairs=True)[4:] == [("Iron", "Iron"), ("Iron", "Titanium"), ("Titanium", "Iron"), ("Titanium", "Titanium")]


def test_the_indexed_mask_is_indexed_by_phase_id():
  pl = phaseList(crystalFrame("m-3m", mineral="Iron"), crystalFrame("6/mmm", mineral="Mg"))
  assert np.array_equal(pl._indexedMask, [False, True, True])


def test_a_selection_mask_reads_indexed_notIndexed_and_minerals():
  pl = phaseList(crystalFrame("m-3m", mineral="Iron"), crystalFrame("6/mmm", mineral="Mg"))
  ph = np.array([[0, 1], [2, 1]])
  assert (np.array_equal(pl._selectionMask(["indexed"], ph), ph > 0) and np.array_equal(pl._selectionMask(["notIndexed"], ph), ph == 0)
          and np.array_equal(pl._selectionMask(["Mg", "notIndexed"], ph), ph != 1))


@pytest.mark.parametrize("names, distinct", [
  (["Iron", "Titanium"], ["Iron", "Titanium"]),
  (["Iron", "Iron", "Titanium"], ["Iron A", "Iron B", "Titanium"]),
  (["Iron", "iron ", "IRON"], ["Iron A", "iron B", "IRON C"]),
  (["", "", "Iron"], ["A", "B", "Iron"]),
  (["", "Iron"], ["", "Iron"]),
  (["A", "", ""], ["A", "B", "C"]),
  (["x"] * 28, [f"x {c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + ["x AA", "x AB"])])
def test_distinctMinerals_givesPhasesOfOneNameALetterEach(names, distinct):
  # equal cells, so that only the name tells the phases apart
  frames = distinctMinerals(["notIndexed"] + [crystalFrame("m-3m", [3, 3, 3], mineral=n) for n in names])
  pl = phaseList(frames)
  assert pl.mineralList[1:] == distinct and (len(set(distinct)) < len(distinct) or [pl.phaseId(n) for n in distinct if n] == [k + 1 for k, n in enumerate(distinct) if n])
