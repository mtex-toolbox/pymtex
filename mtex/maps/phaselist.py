# the phases of a map (MATLAB phaseList, Julia phaselist.jl)
#
#   phaseList(cs1, cs2)                 the frames a map's measurements can belong to
#   phaseList(['notIndexed', cs1, cs2]) the MATLAB list, a string naming a not indexed frame
#
# The not indexed frame comes first, at index 0, and the phase id of a measurement is an
# index into this list. MTEX's phaseList also reconciles the numbering a file states; that
# translation belongs to the importer, and what reaches a map is an index into this list.
from __future__ import annotations

import numpy as np

from ..geometry.frames import crystalFrame, notIndexedFrame, referenceFrame
from ..geometry.display import aligned


# the letter of the k-th phase of one name: A to Z, then AA, AB, ...
def _letter(k):
  return (_letter(k // 26 - 1) if k >= 26 else "") + chr(ord("A") + k % 26)


def distinctMinerals(frames):
  """the phases of a file with every indexed one addressable by its mineral: phases that share
  a name, in any case, and phases without one take a letter each in their order, 'Iron A',
  'Iron B', or 'A', 'B' where the file names none of them; a name the file gives once stays"""
  frames = list(frames)
  indexed = [i for i, f in enumerate(frames) if isinstance(f, referenceFrame) and f.isIndexed]
  taken = {frames[i].mineral.strip().lower() for i in indexed}
  count = {}
  for i in indexed:
    key = frames[i].mineral.strip().lower()
    count[key] = count.get(key, 0) + 1
  seen = {}
  for i in indexed:
    name = frames[i].mineral.strip()
    if count[name.lower()] == 1:
      continue
    # a letter another phase already has as its name is passed over
    while True:
      k = seen.get(name.lower(), 0)
      seen[name.lower()] = k + 1
      new = f"{name} {_letter(k)}".strip()
      if new.lower() not in taken:
        break
    taken.add(new.lower())
    fr = frames[i].clone()
    fr.mineral = new
    frames[i] = referenceFrame.intern(fr)
  return frames


class phaseList:
  """the phases of a map, the frames its measurements can belong to, the not indexed frame first at
  index 0, so that a phase id is an index into the list

    phaseList(cs1, cs2)
    phaseList(['notIndexed', cs1, cs2])  # a string names a not indexed frame

  Input
    frames - crystal frames as separate arguments or a list; 'notIndexed' or a notIndexedFrame denotes phase zero"""

  def __init__(self, *frames):
    if len(frames) == 1 and isinstance(frames[0], (list, tuple)):
      frames = frames[0]
    fs = []
    for f in frames:
      if isinstance(f, str):
        fs.append(notIndexedFrame(f))
      elif isinstance(f, referenceFrame):
        fs.append(f)
      else:
        raise TypeError(f"a phase is a crystalFrame or the name of a not indexed one, not {type(f).__name__}")
    if not fs or fs[0].isIndexed:
      fs.insert(0, notIndexedFrame())
    self.frames = fs

  # ------------------------------------------------------------------ the list

  def __len__(self):
    return len(self.frames)

  def __iter__(self):
    return iter(self.frames)

  def __getitem__(self, key):
    return self.frames[self.phaseId(key)]

  @property
  def CSList(self):
    """the frames of the phases as a list, the not indexed one first"""
    return list(self.frames)

  @property
  def mineralList(self):
    """the mineral names of the phases"""
    return [f.mineral for f in self.frames]

  @property
  def colorList(self):
    """the colours the frames of the phases carry, None where they carry none"""
    return [f.color for f in self.frames]

  def isIndexed(self, i):
    """whether phase i carries a lattice"""
    return self.frames[i].isIndexed

  # whether every phase carries a lattice, as an array indexed by phase id
  @property
  def _indexedMask(self):
    return np.array([f.isIndexed for f in self.frames], dtype=bool)

  # the measurements or grains of the phase ids given that a list of selection names picks:
  # 'indexed', 'notIndexed' where no phase is called so, and the phases named
  def _selectionMask(self, names, phaseId):
    indexed = self._indexedMask[phaseId]
    mask = np.zeros(np.shape(phaseId), dtype=bool)
    for name in names:
      if name.lower() == "indexed":
        mask |= indexed
      elif name.lower() == "notindexed" and "notindexed" not in [m.lower() for m in self.mineralList]:
        mask |= ~indexed
      else:
        mask |= phaseId == self.phaseId(name)
    return mask

  @property
  def indexedPhaseId(self):
    """the ids of the phases that carry a lattice"""
    return np.array([i for i, f in enumerate(self.frames) if f.isIndexed], dtype=int)

  def phaseId(self, key):
    """the id of a phase named by its mineral, its frame or its number"""
    if isinstance(key, (int, np.integer)):
      if not 0 <= key < len(self.frames):
        raise IndexError(f"the map has {len(self.frames)} phases, there is no phase {key}")
      return int(key)
    if isinstance(key, referenceFrame):
      for i, f in enumerate(self.frames):
        if f is key:
          return i
      raise ValueError(f"the frame {key.char()} is not one of the map's phases")
    if isinstance(key, str):
      for i, f in enumerate(self.frames):
        if f.mineral.lower() == key.lower():
          return i
      # an abbreviation names the phase when it begins one mineral only, ebsd['Fo']
      hits = [i for i, f in enumerate(self.frames) if f.mineral.lower().startswith(key.lower())]
      if len(hits) == 1:
        return hits[0]
      raise ValueError(f"no phase called {key!r}; the map has " + ", ".join(repr(f.mineral) for f in self.frames))
    raise TypeError(f"a phase is named by its mineral, its frame or its number, not by a {type(key).__name__}")

  def selectionNames(self, pairs=False):
    """the names a selection ebsd['...'] takes: the minerals, 'indexed' and 'notIndexed'

      names = ebsd.CSList.selectionNames()             # the minerals, 'indexed', 'notIndexed'
      names = ebsd.CSList.selectionNames(pairs=True)   # also every pair of minerals

    The pairs are those a boundary selection takes, gB['Forsterite', 'Enstatite'].

    Input
      pl - phaseList
    Output
      names - list of the names, then of the pairs as tuples
    Keywords
      pairs - add every ordered pair of minerals
    See also
      phaseList.phaseId, EBSD.select"""
    minerals = list(dict.fromkeys(f.mineral for f in self.frames if f.isIndexed))
    others = list(dict.fromkeys(f.mineral for f in self.frames if not f.isIndexed))
    names = minerals + ["indexed"] + others
    names += ["notIndexed"] if "notindexed" not in [m.lower() for m in names] else []
    return names + ([(a, b) for a in minerals for b in minerals] if pairs else [])

  def __contains__(self, key):
    try:
      self.phaseId(key)
      return True
    except (ValueError, IndexError, TypeError):
      return False

  # ------------------------------------------------------------------ display

  def __repr__(self):
    return aligned("phaseList\n" + phaseTable(self.frames, range(len(self.frames))))


# the table of the phases with their ids, one extra column of counts when given
# the colours the indexed phases are drawn in when their frames carry none, in MATLAB's
# order (mtex_settings PhaseColorOrder), with the CSS triplets the display and the maps share
PHASE_COLOR_ORDER = ["LightSkyBlue", "DarkSeaGreen", "Goldenrod", "LightCoral", "DarkBlue", "DarkGreen", "DarkRed", "DarkViolet",
                     "Gray", "Thistle", "Aquamarine", "MediumSeaGreen", "LightSalmon", "SteelBlue", "MediumPurple", "Moccasin",
                     "OliveDrab"]
CSS_255 = {"lightskyblue": (135, 206, 250), "darkseagreen": (143, 188, 143), "goldenrod": (218, 165, 32),
           "lightcoral": (240, 128, 128), "darkblue": (0, 0, 139), "darkgreen": (0, 100, 0), "darkred": (139, 0, 0),
           "darkviolet": (148, 0, 211), "gray": (128, 128, 128), "thistle": (216, 191, 216), "aquamarine": (127, 255, 212),
           "mediumseagreen": (60, 179, 113), "lightsalmon": (255, 160, 122), "steelblue": (70, 130, 180),
           "mediumpurple": (147, 112, 219), "moccasin": (255, 228, 181), "olivedrab": (107, 142, 35)}


def phaseColorNames(frames):
  """the colour of every phase of a list: the one its frame carries, else the next name of the
  order for an indexed phase, None for a not indexed one without a colour"""
  out, k = [], 0
  for f in frames:
    if f.color is not None and not (isinstance(f.color, np.ndarray) and np.all(np.isnan(f.color))):
      out.append(f.color)
    elif f.isIndexed:
      out.append(PHASE_COLOR_ORDER[k % len(PHASE_COLOR_ORDER)])
      k += 1
    else:
      out.append(None)
  return out


def colorTriplet(c):
  """a colour as an RGB triplet: a name of the order, or what matplotlib knows, or a triplet"""
  if c is None:
    return None
  if isinstance(c, str):
    if c.lower() in CSS_255:
      return np.array(CSS_255[c.lower()]) / 255
    from matplotlib.colors import to_rgb
    return np.array(to_rgb(c))
  return np.asarray(c, dtype=float).reshape(3)


def phaseTable(frames, ids, counts=None, countLabel="Orientations"):
  """the table of the phases as text: id, mineral, colour, symmetry and crystal frame, with a column
  of counts when given"""
  names = phaseColorNames(frames)
  rows = []
  for k, i in enumerate(ids):
    f = frames[i]
    cs = isinstance(f, crystalFrame)
    row = [str(i)] + ([counts[k]] if counts is not None else []) + [
      f.mineral, "" if names[i] is None else (names[i] if isinstance(names[i], str) else np.array2string(np.asarray(names[i], dtype=float), precision=3)),
      f.pointGroup if cs else "", f.char() if cs else ""]
    rows.append(row)
  labels = ["Phase"] + ([countLabel] if counts is not None else []) + ["Mineral", "Color", "Symmetry", "Crystal reference frame"]
  widths = [max(len(labels[j]), *(len(r[j]) for r in rows)) if rows else len(labels[j]) for j in range(len(labels))]
  def line(r):
    return ("  " + "  ".join(str(x).ljust(w) for x, w in zip(r, widths))).rstrip()
  return "\n".join([line(labels)] + [line(r) for r in rows])


__all__ = ["phaseList", "phaseColorNames", "colorTriplet"]
