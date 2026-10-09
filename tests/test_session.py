import sys

import mtex
from mtex import getMTEXpref, setMTEXpref


def test_preferences_and_reset():
  assert getMTEXpref("EulerAngleConvention") == "Bunge" and getMTEXpref("UTF8Output") is True
  assert getMTEXpref("nothing") is None and getMTEXpref("nothing", 3) == 3
  setMTEXpref("EulerAngleConvention", "Matthies")
  assert getMTEXpref("EulerAngleConvention") == "Matthies"
  mtex.reset()
  assert getMTEXpref("EulerAngleConvention") == "Bunge"


def test_all_preferences():
  prefs = getMTEXpref()
  assert prefs["EulerAngleConvention"] == "Bunge" and prefs["UTF8Output"] is True
  # the dict is a copy: changing it leaves the session alone
  prefs["UTF8Output"] = False
  assert getMTEXpref("UTF8Output") is True


# NumPy's request for huge pages follows the preference, 'auto' by default and after a reset
def test_huge_pages_follow_the_preference():
  from numpy._core.multiarray import _get_madvise_hugepage
  from mtex.geometry.session import _hugePagesHelp
  auto = _hugePagesHelp() and sys.platform.startswith("linux")
  assert getMTEXpref("hugePages") == "auto" and _get_madvise_hugepage() is auto
  setMTEXpref("hugePages", True)
  assert _get_madvise_hugepage() is sys.platform.startswith("linux")
  setMTEXpref("hugePages", False)
  assert _get_madvise_hugepage() is False
  mtex.reset()
  assert _get_madvise_hugepage() is auto


# 'auto' asks for huge pages where the free blocks of their size hold a quarter of the memory available
def test_huge_pages_auto_reads_the_free_blocks():
  from mtex.geometry.session import _hugePagesHelp
  mem = "MemAvailable:   8388608 kB\nHugepagesize:       2048 kB\n"
  # orders 0 to 10: 2000 free blocks of 4 MB are 8 GB; the fragmented one is a laptop's, where huge
  # pages made a fresh 46 MB array fill in 101 ms instead of 30
  rich = "Node 0, zone   Normal " + " ".join(["100"] * 10) + " 2000\n"
  poor = ("Node 0, zone    DMA32  20306  13433   7473   3277   1511    814    403    324     93     69      3\n"
          "Node 0, zone   Normal 1215576 600051 153063  27820   6859   1882   1026    419     42      6      0\n")
  assert _hugePagesHelp("always [madvise] never", rich, mem) and not _hugePagesHelp("always [madvise] never", poor, mem)
  assert not _hugePagesHelp("always madvise [never]", rich, mem)
