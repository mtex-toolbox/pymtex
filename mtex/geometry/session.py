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
  # whether NumPy asks the kernel for huge pages on its large arrays: True, False, or 'auto', which
  # asks where the kernel offers them and holds enough free memory in blocks of their size. Where
  # memory is fragmented, every page fault of such an array waits for the kernel to compact it,
  # ten times slower to fill; where it is not, a fresh 46 MB array fills in 7 ms instead of 24
  "hugePages": "auto",
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
    self.siblings = {}          # the register's answer for a frame and a named group
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


# 'auto' asks for huge pages where the free memory in blocks of their size is at least this
# share of the available memory, and at least HUGE_MINIMUM bytes
HUGE_SHARE, HUGE_MINIMUM = 0.25, 1 << 30


def _readText(path):
  try:
    with open(path) as f:
      return f.read()
  except OSError:
    return None


# whether huge pages fill fresh memory faster here, from the kernel's setting (the bracketed word
# of transparent_hugepage/enabled), the free blocks of every order in /proc/buddyinfo and
# /proc/meminfo; no on a system without them, and where the free blocks of a huge page's size
# hold too little of the memory available, since then every fault waits for compaction
def _hugePagesHelp(enabled=None, buddyinfo=None, meminfo=None):
  enabled = _readText("/sys/kernel/mm/transparent_hugepage/enabled") if enabled is None else enabled
  buddyinfo = _readText("/proc/buddyinfo") if buddyinfo is None else buddyinfo
  meminfo = _readText("/proc/meminfo") if meminfo is None else meminfo
  if not enabled or "[never]" in enabled or not buddyinfo or not meminfo:
    return False
  mem = {}
  for line in meminfo.splitlines():
    parts = line.replace(":", " ").split()
    if len(parts) >= 2 and parts[1].isdigit():
      mem[parts[0]] = int(parts[1]) * 1024
  page, huge, avail = 4096, mem.get("Hugepagesize", 2 << 20), mem.get("MemAvailable", 0)
  order = max(0, int(huge // page).bit_length() - 1)
  free = 0
  for line in buddyinfo.splitlines():
    counts = line.split()[4:]
    free += sum(int(c) * (page << k) for k, c in enumerate(counts) if k >= order and c.isdigit())
  return free >= max(HUGE_SHARE * avail, HUGE_MINIMUM)


def applyHugePages():
  """hands the preference 'hugePages' to NumPy, which holds it for the whole process; 'auto'
  decides by the memory this machine has free in blocks of a huge page's size"""
  try:
    from numpy._core.multiarray import _set_madvise_hugepage
  except ImportError:
    return
  want = SESSION.prefs.get("hugePages", "auto")
  _set_madvise_hugepage(_hugePagesHelp() if want == "auto" else bool(want))


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
