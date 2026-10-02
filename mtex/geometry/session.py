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
  # the bandwidth a callable or scattered values are expanded to when none is given; the
  # maxima above cap the results whose bandwidth is known, as the sum in a product
  "defaultS2Bandwidth": 128,
  "defaultSO3Bandwidth": 64,
  "FFTAccuracy": 1e-2,
  "displayRows": 20,
  # whether NumPy asks the kernel for huge pages on its large arrays; on a machine whose memory is
  # fragmented every page fault of such an array then waits for the kernel to compact memory,
  # which makes fresh memory ten times slower to fill, so MTEX switches the request off
  "hugePages": False,
}


class Session:
  """the per-session state: the preferences, the frame register, the named frames and the default
  frame"""
  def __init__(self):
    self.lock = threading.RLock()
    self.reset()

  def reset(self):
    """the state of a fresh session: default preferences, no frames registered, the default
    frame minted again on first use; data objects keep the handles they hold"""
    self.prefs = dict(_DEFAULT_PREFS)
    self.frames = []            # the register, one handle per key
    self.namedFrames = {}       # the named specimen frames, by name
    self.defaultFrame = None    # the specimen frame supplying the default plotting convention
    self.mintDefaultFrame = None   # set by frames.py: how a fresh default frame is made


SESSION = Session()


def getMTEXpref(name=None, default=None):
  """one preference, or with no name all of them as a new dict, MATLAB's struct"""
  if name is None:
    return dict(SESSION.prefs)
  return SESSION.prefs.get(name, default)


def setMTEXpref(name, value, save=False):
  """sets a preference of the session; with save=True also in the user's settings file, so that
  it holds in every later session

    setMTEXpref('maxS2Bandwidth', 256)
    setMTEXpref('kernels', 'numpy')
    setMTEXpref('displayRows', 40, save=True)"""
  SESSION.prefs[name] = value
  if name == "hugePages":
    applyHugePages()
  if name == "threads":
    from .kernels import applyThreads
    applyThreads()
  if save:
    from .settings import saveSetting
    saveSetting(name, value)


def applyHugePages():
  """hands the preference 'hugePages' to NumPy, which holds it for the whole process"""
  try:
    from numpy._core.multiarray import _set_madvise_hugepage
  except ImportError:
    return
  _set_madvise_hugepage(bool(SESSION.prefs.get("hugePages", False)))


def reset():
  """a fresh session: the preferences, the register and the default frame; the way to mint
  a default frame is what frames.py installed and stays"""
  mint = SESSION.mintDefaultFrame
  SESSION.reset()
  SESSION.mintDefaultFrame = mint
  applyHugePages()
  from .kernels import applyThreads
  applyThreads()


__all__ = ["getMTEXpref", "setMTEXpref", "reset"]
