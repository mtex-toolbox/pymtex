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


# NumPy's request for huge pages follows the preference, off by default and after a reset
def test_huge_pages_follow_the_preference():
  from numpy._core.multiarray import _get_madvise_hugepage
  assert getMTEXpref("hugePages") is False and _get_madvise_hugepage() is False
  setMTEXpref("hugePages", True)
  assert _get_madvise_hugepage() is True
  mtex.reset()
  assert _get_madvise_hugepage() is False
