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
