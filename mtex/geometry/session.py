# the session: the one place holding per-session state - preferences now, the frame
# register and the default frame from G3 on
from __future__ import annotations

import threading

_DEFAULT_PREFS = {
  "EulerAngleConvention": "Bunge",
  "UTF8Output": True,
  "degreeChar": "°",
  "FontSize": 13,
  "markerSize": 10,
  "stopOnSymmetryMissmatch": True,
  "maxS1Bandwidth": 1024,
  "maxS2Bandwidth": 512,
  "maxSO3Bandwidth": 64,
  "FFTAccuracy": 1e-2,
}


class Session:
  def __init__(self):
    self.lock = threading.RLock()
    self.reset()

  # the state of a fresh session: default preferences, no frames registered, the default
  # frame minted again on first use; data objects keep the handles they hold
  def reset(self):
    self.prefs = dict(_DEFAULT_PREFS)
    self.frames = []            # the register, one handle per key
    self.namedFrames = {}       # the named specimen frames, by name
    self.defaultFrame = None    # the specimen frame supplying the default plotting convention
    self.mintDefaultFrame = None   # set by frames.py: how a fresh default frame is made


SESSION = Session()


# one preference, or with no name all of them as a new dict, MATLAB's struct
def getMTEXpref(name=None, default=None):
  if name is None:
    return dict(SESSION.prefs)
  return SESSION.prefs.get(name, default)


def setMTEXpref(name, value):
  SESSION.prefs[name] = value


# a fresh session: the preferences, the register and the default frame; the way to mint
# a default frame is what frames.py installed and stays
def reset():
  mint = SESSION.mintDefaultFrame
  SESSION.reset()
  SESSION.mintDefaultFrame = mint


__all__ = ["getMTEXpref", "setMTEXpref", "reset"]
