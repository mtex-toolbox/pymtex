# M1: the phases of a map
import numpy as np
import pytest

from mtex import phaseList, crystalFrame
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
