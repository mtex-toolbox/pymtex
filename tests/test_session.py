import mtex
from mtex import getMTEXpref, setMTEXpref


def test_preferences_and_reset():
  assert getMTEXpref("EulerAngleConvention") == "Bunge" and getMTEXpref("UTF8Output") is True
  assert getMTEXpref("nothing") is None and getMTEXpref("nothing", 3) == 3
  setMTEXpref("EulerAngleConvention", "Matthies")
  assert getMTEXpref("EulerAngleConvention") == "Matthies"
  mtex.reset()
  assert getMTEXpref("EulerAngleConvention") == "Bunge"
