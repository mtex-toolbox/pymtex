# what every control of the import wizard does, as one function each over the state (Julia
# callbacks.jl; docs/plans/wizard.md W1)
#
# The window calls these and nothing else; the tests call them with no window open. A load
# runs where the caller says: inline by default, on a worker the window supplies.
from __future__ import annotations

import os
import threading
import time

import numpy as np

from ..geometry.frames import closestAlignment, crystalFrame, notIndexedFrame
from ..geometry.plottingconvention import AXIS_CONVENTIONS, closestConvention, plottingConvention
from ..geometry.units import degree
from ..io.importscript import buildImportWizardScript
from ..io.inspect import ebsdFiles, inspectEBSD, isEBSDFile
from ..maps.phaselist import phaseList
from .state import FileRow, WizardState

IPF_TABS = {"IPF X": "x", "IPF Y": "y", "IPF Z": "z"}


# ------------------------------------------------------------------ the browser

def folderRows(folder, depth=0):
  """the rows of a folder: its folders first, then its maps"""
  rows = []
  try:
    names = sorted(os.listdir(folder), key=str.lower)
  except OSError:
    return rows
  for n in names:
    p = os.path.join(folder, n)
    if os.path.isdir(p) and not n.startswith("."):
      rows.append(FileRow(n, p, depth, True))
  for n in ebsdFiles(folder):
    p = os.path.join(folder, n)
    rows.append(FileRow(n, p, depth, False, size=os.path.getsize(p)))
  return rows


def setFolder(st, folder):
  """makes a folder the one the browser lists, its rows read afresh"""
  st.folder = os.path.abspath(folder)
  st.entries = folderRows(st.folder)
  st.changed("folder", "entries")


def toggleFolder(st, i):
  """a folder unfolds where it stands, or folds its rows away again"""
  row = st.entries[i]
  if not row.isDir:
    return
  if row.open:
    j = i + 1
    while j < len(st.entries) and st.entries[j].depth > row.depth:
      j += 1
    del st.entries[i + 1:j]
    row.open = False
  else:
    st.entries[i + 1:i + 1] = folderRows(row.path, row.depth + 1)
    row.open = True
  st.changed("entries")


# what the file says that the window shows nowhere else: the name is the list's and the status
# band's, the size the list's, the points the band's, the phases the table's, the properties
# the tabs', the data sets their list's; the vendor names the format, which is the extension
INFO_SHOWN_ELSEWHERE = ("file", "format", "size", "points", "phases", "properties", "data sets")


def infoRows(st):
  """the rows of the file information panel not shown elsewhere in the window, the vendor falling
  back to the format; none before a file is selected"""
  if st.info is None:
    return []
  rows = [(k, v) for k, v in st.info.describe() if k not in INFO_SHOWN_ELSEWHERE]
  return [(k, v or st.info.format) if k == "vendor" else (k, v) for k, v in rows]


def parentRow(st, i):
  """the row of the folder an entry was unfolded from, None for one at the top"""
  depth = st.entries[i].depth
  return next((j for j in range(i - 1, -1, -1) if st.entries[j].depth < depth), None)


def letterRow(st, current, letter):
  """the row a letter typed goes to: the first entry whose name starts with it, or the next one
  after the current row when that starts with it already, round to the top; None for none"""
  letter = letter.lower()
  hits = [j for j, r in enumerate(st.entries) if r.name.lower().startswith(letter)]
  if not hits:
    return None
  if current is not None and 0 <= current < len(st.entries) and st.entries[current].name.lower().startswith(letter):
    return next((j for j in hits if j > current), hits[0])
  return hits[0]


def breadcrumbs(st):
  """the crumbs of the path, the root first, each with the folder it reaches"""
  drive, tail = os.path.splitdrive(st.folder)
  here = drive + os.sep
  out = [(here, here)]
  for p in tail.split(os.sep):
    if p:
      here = os.path.join(here, p)
      out.append((p, here))
  return out


def pickFile(st, i):
  """a click on a row: a folder unfolds, a map is described and read"""
  row = st.entries[i]
  if row.isDir:
    toggleFolder(st, i)
  else:
    openEntry(st, row.path)


# ------------------------------------------------------------------ describing and importing

def selectEntry(st, path):
  """the file described from its header alone: the panel, the data sets and the two frames"""
  st.selected = path
  st.dataSet = 0
  try:
    st.info = inspectEBSD(path)
  except Exception as e:
    st.info = None
    st.status = ("failed", f"{os.path.basename(path)}: {e}")
    st.changed("selected", "info", "status")
    return None
  st.eulerAxesNames = st.info.eulerAxesNames
  # the map on the screen is still the last one read: it is drawn anew only when the convention
  # it is drawn in really changes, not on every file the cursor passes
  convention = closestConvention(plottingConvention.default.rot)[0]
  moved = convention != st.mapConvention
  st.mapConvention = convention
  st.eulerConvention = eulerConventionOf(st, st.info.eulerCorrection)
  st.status = ("idle", f"{st.info.name}: click to import" if st.ebsd is None or st.selected != _loadedPath(st) else f"{st.info.name} is loaded")
  st.changed("selected", "info", "status", "eulerConvention", *(["mapConvention"] if moved else []))
  return st.info


def openEntry(st, path, worker=None):
  """selects a file and starts its import, inline or through the worker given"""
  if selectEntry(st, path) is None:
    return
  startImport(st, worker)


def leaveEntry(st):
  """the cursor moved on to another file: the import under way is not taken when it finishes, so
  that walking through a folder draws only the map it stops on"""
  st.imports += 1


def eulerConventionOf(st, correction):
  """the entry of AXIS_CONVENTIONS whose frame the vendor wrote its angles in, given the
  correction the loader applied: correction = mapRot * inv(eulerRot)"""
  mapRot = plottingConvention(AXIS_CONVENTIONS[st.mapConvention]).rot
  return closestConvention(correction.inv() * mapRot)[0]


def correctionOf(st):
  """the Euler correction the two chosen conventions make: the rotation of the map convention
  times the inverse of that of the Euler convention"""
  return plottingConvention(AXIS_CONVENTIONS[st.mapConvention]).rot * plottingConvention(AXIS_CONVENTIONS[st.eulerConvention]).rot.inv()


def warmColours(st):
  """the colour of an orientation is a look-up over a grid built once per phase, and five of
  them are a second and a half - paid, until this, when the first map was drawn and the
  window could not be moved. The phases are known from the header before the map is read,
  so the grids are built beside the read: numpy holds no lock over its own arrays, and the
  register of the grids has one of its own."""
  frames = [f for f in st.info.phases.frames if f.isIndexed] if st.info is not None else []
  if not frames:
    return None

  thread = threading.Thread(target=buildColourGrids, args=(frames,), name="mtex-wizard-colours", daemon=True)
  thread.start()
  return thread


def buildColourGrids(frames):
  """the inverse pole figure colour grid of every frame, each built once and kept"""
  from ..plotting.orientationkeys import keyColorGrid
  for f in frames:
    try:
      keyColorGrid(f)
    except Exception:
      return


def startImport(st, worker=None):
  """the load: inline, or handed to a worker as (load, commit) so the window stays live"""
  if st.info is None:
    return
  path, k, several = st.selected, st.dataSet, len(st.info.dataSets) > 1
  # every import is numbered and only the latest is taken: a map clicked while another was
  # still being read would otherwise be replaced by it when the earlier one finished later
  st.imports += 1
  number = st.imports
  st.status = ("loading", f"reading {os.path.basename(path)} ...")
  st.changed("status")
  if worker is not None:
    warmColours(st)

  def load():
    from ..io.load import loadEBSD
    t = time.perf_counter()
    kw = {"dataSet": k, "silent": True} if several else {}
    ebsd = loadEBSD(path, **kw)
    seconds = time.perf_counter() - t
    # the colour grids of phases the header did not name built here too, not on the main thread
    # when the map is first drawn
    if worker is not None:
      buildColourGrids(f for f in ebsd.CSList.frames if f.isIndexed)
    return ebsd, seconds

  def commit(result):
    if number != st.imports:
      return
    if isinstance(result, BaseException):
      st.status = ("failed", f"{os.path.basename(path)}: {result}")
      st.changed("status")
    else:
      commitMap(st, *result)

  if worker is None:
    try:
      commit(load())
    except Exception as e:
      commit(e)
  else:
    worker(load, commit)


def commitMap(st, ebsd, seconds):
  """the map taken into the state: its phases as declared, every indexed one drawn, the raster,
  the views it offers, the Euler frame it was read in"""
  st.ebsd = ebsd
  st.loadSeconds = seconds
  st.headerPhases = frameKeys(ebsd.CSList)
  st.drawn = {k: True for k in range(len(ebsd.CSList))}
  st.densities = {}
  st.eulerConvention = eulerConventionOf(st, ebsd.EulerCorrection)
  st.images = imagesOf(ebsd)
  st.tabs, st.tabKinds = viewsOf(ebsd, st.images)
  opening = "IPF Z" if "IPF Z" in st.tabs else "phase"
  st.tab = st.tabs.index(opening)
  st.metaRows = metaRowsOf(ebsd, st.images)
  # a branch of fields opens at the top level; a branch of many lines, an .ang header, starts shut
  st.metaOpen = {i for i, (depth, _, _, branch, _) in enumerate(st.metaRows) if branch and depth == 0 and childCount(st.metaRows, i) <= 40}
  st.image = 0
  st.imageCache = {}
  st.keyPhase = dominantPhase(st)
  if st.keyPhase is None:
    st.keyPhase = next((k for k, f in enumerate(ebsd.CSList.frames) if f.isIndexed), None)
  st.status = ("loaded", f"{os.path.basename(st.selected)}: {ebsd.size} points in {seconds:.2f} s")
  st.changed("ebsd", "phases", "eulerConvention", "tabs", "tab", "meta", "images", "keyPhase", "status")


def _loadedPath(st):
  return st.ebsd.meta.get("h5", {}).get("file") if st.ebsd is not None else None


def setDataSet(st, k, worker=None):
  """picking a data set of a project file is the import"""
  if st.info is None or not 0 <= k < max(len(st.info.dataSets), 1):
    return
  st.dataSet = k
  st.changed("dataSet")
  startImport(st, worker)


# ------------------------------------------------------------------ the views

def viewsOf(ebsd, images=None):
  """the views a map has: the map always, IPF and pole figures once something is indexed, one
  tab per property, the images where the file brought some"""
  indexed = any(f.isIndexed for f in ebsd.CSList.frames) and bool(np.any(ebsd.isIndexed))
  tabs = [("phase", "phase")]
  if indexed:
    tabs += [("IPF X", "ipf"), ("IPF Y", "ipf"), ("IPF Z", "ipf")]
  tabs += [(p, "property") for p in sorted(ebsd.prop) if np.asarray(ebsd.prop[p]).ndim == ebsd.ndim]
  if indexed:
    tabs.append(("pole figures", "polefigures"))
  if imagesOf(ebsd) if images is None else images:
    tabs.append(("images", "images"))
  return [t for t, _ in tabs], [k for _, k in tabs]


def pickTab(st, i):
  """shows tab i, when there is one"""
  if 0 <= i < len(st.tabs):
    st.tab = i
    st.changed("tab")


def imagesOf(ebsd):
  """the pictures a file brought: the ones its loader indexed without reading, which is how a
  vendor's dozen electron images stay out of the map, and the ones that came in the header
  small enough to have been read. Each is a name and either an array or an entry to read."""
  out, seen = [], set()
  for entry in ebsd.meta.get("images", []) or []:
    label = pictureName(entry["name"])
    out.append((label, entry))
    seen.add((label.split("/")[-1], tuple(entry["shape"])[:2]))

  def walk(d, prefix):
    for k, v in d.items():
      p = f"{prefix}/{k}" if prefix else str(k)
      if isinstance(v, dict):
        walk(v, p)
        continue
      picture = asPicture(v)
      if picture is not None and (str(k), picture.shape[:2]) not in seen:
        out.append((p, picture))

  walk({k: v for k, v in ebsd.meta.items() if k not in ("h5", "images", "EulerCorrection")}, "")
  return out


MIN_PICTURE = 16
GENERIC = ("data", "header", "")


def pictureName(path):
  """what a picture is called: the last two parts of its path that say anything, since every
  vendor buries it under a Data or a Header the name of which names nothing"""
  parts = [p for p in str(path).split("/") if p.lower() not in GENERIC]
  return "/".join(parts[-2:]) if parts else str(path)


def asPicture(v):
  """the picture an array is, or None: a plane, a plane of three or four channels either way
  round, and a stack of one channel, which is how a grey picture is written"""
  if not isinstance(v, np.ndarray) or v.dtype.kind not in "iuf" or v.ndim not in (2, 3):
    return None
  a = v
  if a.ndim == 3:
    if a.shape[0] in (1, 3, 4) and min(a.shape[1:]) >= MIN_PICTURE:
      a = a[0] if a.shape[0] == 1 else np.moveaxis(a, 0, -1)
    elif a.shape[2] in (3, 4) and min(a.shape[:2]) >= MIN_PICTURE:
      a = a[..., :3] if a.shape[2] == 4 else a
    else:
      return None
  elif min(a.shape) < MIN_PICTURE:
    return None
  # a colour picture is drawn from bytes or from fractions, so an integer one is scaled
  if a.ndim == 3 and a.dtype.kind in "iu" and a.dtype != np.uint8:
    hi = float(a.max()) or 1.0
    a = (a / hi).astype(np.float32)
  return a


def pictureOf(st, k):
  """the pixels of the picture at k, read from the file the first time it is looked at"""
  if not 0 <= k < len(st.images):
    return None
  label, thing = st.images[k]
  if isinstance(thing, np.ndarray):
    return thing
  if k not in st.imageCache:
    from ..io.h5 import h5Image
    st.imageCache[k] = h5Image(st.ebsd.meta["h5"]["file"], thing)
  return st.imageCache[k]


def showImage(st, k):
  """shows image k of the file's pictures, when there is one"""
  if 0 <= k < len(st.images):
    st.image = k
    st.changed("image")


def metaRowsOf(ebsd, images=()):
  """the metadata flattened into rows of (depth, name, preview, isBranch, picture), where the
  picture is the entry of images the row shows, or None. Every branch opens at the top level."""
  rows = []
  where = {label: k for k, (label, thing) in enumerate(images) if isinstance(thing, np.ndarray)}

  def walk(d, depth, prefix):
    for k, v in d.items():
      p = f"{prefix}/{k}" if prefix else str(k)
      if isinstance(v, dict):
        rows.append((depth, str(k), f"{len(v)} fields", True, None))
        walk(v, depth + 1, p)
      else:
        picture = where.get(p)
        rows.append((depth, str(k), pictureLabel(images[picture][1]) if picture is not None else preview(v), False, picture))

  walk({k: v for k, v in ebsd.meta.items() if k not in ("h5", "images", "EulerCorrection")}, 0, "")
  # the pictures the file holds and the map did not read stand in a branch of their own,
  # since nothing of them is in the metadata to hang them on
  indexed = [(k, label) for k, (label, thing) in enumerate(images) if not isinstance(thing, np.ndarray)]
  if indexed:
    rows.append((0, "pictures", f"{len(indexed)} pictures", True, None))
    for k, label in indexed:
      rows.append((1, label, pictureLabel(images[k][1]), False, k))
  return rows


def pictureLabel(thing):
  """what a picture says about itself in the tree, whether it is read or only indexed"""
  shape = thing.shape if isinstance(thing, np.ndarray) else tuple(thing["shape"])
  return f"picture {shape[1]} x {shape[0]}" + (" colour" if len(shape) == 3 else "")


def preview(v):
  """a value in a line: a number, a string, an array by its shape"""
  if isinstance(v, (bytes, np.bytes_)):
    v = v.decode("ascii", errors="replace")
  if isinstance(v, str):
    return v if len(v) <= 60 else v[:57] + "..."
  if isinstance(v, (bool, np.bool_)):
    return str(bool(v))
  if isinstance(v, (int, np.integer)):
    return str(int(v))
  if isinstance(v, (float, np.floating)):
    return f"{float(v):.6g}"
  if isinstance(v, np.ndarray):
    if v.size <= 6:
      return "[" + ", ".join(preview(x) for x in v.reshape(-1)) + "]"
    return f"{v.dtype} array {' x '.join(str(s) for s in v.shape)}"
  if isinstance(v, (list, tuple)):
    return "[" + ", ".join(preview(x) for x in v[:6]) + (", ..." if len(v) > 6 else "") + "]"
  if v is None:
    return "-"
  return repr(v)


def childCount(rows, i):
  """the number of rows below metadata row i that lie inside it, its descendants"""
  depth, n = rows[i][0], 0
  for d, *_ in rows[i + 1:]:
    if d <= depth:
      break
    n += 1
  return n


def visibleMeta(st):
  """the rows whose every ancestor is open"""
  out, hidden = [], None
  for i, (depth, k, v, branch, picture) in enumerate(st.metaRows):
    if hidden is not None and depth > hidden:
      continue
    hidden = None
    out.append((i, depth, k, v, branch, picture))
    if branch and i not in st.metaOpen:
      hidden = depth
  return out


def pickMeta(st, i):
  """a row of the tree: a branch opens or closes, a picture is shown, and showing it moves to
  the view that holds the pictures, since a picture picked where it is named is picked to
  be looked at"""
  if not 0 <= i < len(st.metaRows):
    return
  _, _, _, branch, picture = st.metaRows[i]
  if branch:
    st.metaOpen.symmetric_difference_update({i})
    st.changed("meta")
  elif picture is not None:
    showImage(st, picture)
    if "images" in st.tabs:
      pickTab(st, st.tabs.index("images"))


def setMiller(st, k, text):
  """sets the k-th of the three Miller indices the pole figures show, the densities dropped"""
  if 0 <= k < 3:
    st.millers[k] = str(text).strip()
    st.densities = {}
    st.changed("millers")


# ------------------------------------------------------------------ the coordinates

def setMapConvention(st, i):
  """the map convention only re-views"""
  if not 0 <= i < len(AXIS_CONVENTIONS):
    return
  st.mapConvention = i
  plottingConvention(AXIS_CONVENTIONS[i]).makeDefault()
  st.changed("mapConvention")


def setEulerConvention(st, i):
  """the Euler convention turns the rotations in place and the densities with them"""
  if not 0 <= i < len(AXIS_CONVENTIONS):
    return
  st.eulerConvention = i
  applyEuler(st)
  st.changed("eulerConvention")


def applyEuler(st):
  """turns the loaded map, the Euler angles only, to the correction of the chosen conventions,
  the densities with it; nothing when it is already there"""
  if st.ebsd is None:
    return
  new = correctionOf(st)
  old = st.ebsd.EulerCorrection
  d = new * old.inv()
  if float(d.angle()) < 1e-12:
    return
  ebsd = st.ebsd.rotate(d, keepXY=True)
  ebsd.meta = dict(ebsd.meta, EulerCorrection=new)
  st.ebsd = ebsd
  st.densities = {k: odf.rotate(d) for k, odf in st.densities.items()}
  st.changed("ebsd")


# ------------------------------------------------------------------ the phases

# the columns of the phase table, editable ones through editPhase
PHASE_COLUMNS = ("plot", "phase", "mineral", "pixels", "percent", "symmetry", "a", "b", "c", "alpha", "beta", "gamma", "alignment")


def phaseCounts(st):
  """the number of pixels of every phase id of the loaded map, empty before one is loaded; the
  table, the script and the pole figures ask for it many times per map, so it is counted once"""
  if st.ebsd is None:
    return {}
  ph = st.ebsd.phaseId
  memo = getattr(st, "_phaseCounts", None)
  if memo is None or memo[0] is not ph:
    counts = np.bincount(ph.reshape(-1).astype(np.intp))
    memo = (ph, {int(i): int(c) for i, c in enumerate(counts) if c})
    st._phaseCounts = memo
  return memo[1]


def dominantPhase(st):
  """the id of the drawn indexed phase with the most pixels, which the pole figures show"""
  counts = phaseCounts(st)
  frames = st.ebsd.CSList.frames if st.ebsd is not None else []
  ids = [k for k, on in st.drawn.items() if on and k in counts and k < len(frames) and frames[k].isIndexed]
  return max(ids, key=lambda k: counts[k]) if ids else None


def pickKeyPhase(st, phaseId):
  """the phase whose key the key panel shows, an indexed one"""
  frames = st.ebsd.CSList.frames if st.ebsd is not None else []
  if 0 <= phaseId < len(frames) and frames[phaseId].isIndexed and phaseId != st.keyPhase:
    st.keyPhase = phaseId
    st.changed("keyPhase")


def editPhase(st, phaseId, column, value):
  """a cell edited: the row's frame rebuilt with the lattice snapped to what the group forces.
  The register interns a frame within a shape tolerance, so a rebuilt frame it hands back
  unchanged is minted outside it; a colour is set on the frame itself, which is what reaches
  every map holding it (ADR 0004)"""
  if st.ebsd is None or not 0 <= phaseId < len(st.ebsd.CSList):
    return
  if column == "plot":
    st.drawn[phaseId] = bool(value)
    st.changed("drawn")
    return
  old = st.ebsd.CSList.frames[phaseId]
  if column == "color":
    old.color = value
    st.changed("phases")
    return
  if not old.isIndexed:
    if column == "mineral":
      new = notIndexedFrame(str(value))
    else:
      return
  else:
    symbol, abc, abg = old.pointGroup, list(old.abc), list(old.abg / degree)
    alignment, mineral, color = closestAlignment(old), old.name, old.color
    if column == "mineral":
      mineral = str(value)
    elif column == "symmetry":
      symbol = str(value)
    elif column in ("a", "b", "c"):
      abc["abc".index(column)] = float(value)
    elif column in ("alpha", "beta", "gamma"):
      abg[("alpha", "beta", "gamma").index(column)] = float(value)
    elif column == "alignment":
      alignment = [s.strip() for s in str(value).split(",")] if isinstance(value, str) else list(value)
    else:
      raise ValueError(f"no editable column {column!r}")
    abc, abg = snapLattice(symbol, abc, abg, column)
    new = crystalFrame(symbol, abc, np.array(abg) * degree, *alignment, mineral=mineral, color=color)
    if np.max(np.abs(new.abc - abc)) > 1e-6 * max(abc) or np.max(np.abs(new.abg / degree - abg)) > 1e-6 \
        or closestAlignment(new) != list(alignment):
      new = crystalFrame(symbol, abc, np.array(abg) * degree, *alignment, mineral=mineral, color=color, intern=False)
  frames = list(st.ebsd.CSList.frames)
  frames[phaseId] = new
  st.ebsd.CSList = phaseList(frames)
  st.densities.pop(phaseId, None)
  st.changed("phases")


def snapLattice(symbol, abc, abg, driver=None):
  """the cell the lattice forces follow the one edited: b and c from a on a cubic lattice, the
  angles the lattice implies but for the free ones"""
  from ..geometry.symmetry import symmetry
  from ..geometry.frames import _uniqueMonoclinicAxis
  lat = symmetry(symbol).lattice
  free = lat.freeLengths
  abc = list(abc)
  if not free[1]:
    abc[1] = abc[0] if driver != "b" else abc[1]
    if driver == "b":
      abc[0] = abc[1]
  if not free[2]:
    abc[2] = abc[0] if driver != "c" else abc[2]
    if driver == "c":
      abc[0] = abc[1] = abc[2]
  unique = _uniqueMonoclinicAxis(symbol)
  freeAngles = lat.freeAngles(unique if unique is not None else 1)
  default = np.round(np.array(lat.defaultAngles) / degree, 9)
  abg = [abg[k] if freeAngles[k] else float(default[k]) for k in range(3)]
  return abc, abg


def freeCells(CS):
  """which cells of a row the lattice leaves free"""
  if not CS.isIndexed:
    return {c: False for c in PHASE_COLUMNS}
  from ..geometry.frames import _uniqueMonoclinicAxis
  lat = CS.lattice
  unique = _uniqueMonoclinicAxis(CS.pointGroup)
  lengths, angles = lat.freeLengths, lat.freeAngles(unique if unique is not None else 1)
  return {"a": lengths[0], "b": lengths[1], "c": lengths[2], "alpha": angles[0], "beta": angles[1], "gamma": angles[2]}


# ------------------------------------------------------------------ the outputs

def frameKeys(phases):
  """the value snapshot of a phase list, against which an edit is told: name, group, cell, setup, colour"""
  return [frameKey(f) for f in phases.frames]


def frameKey(f):
  """the value of a frame as a tuple, what an edit of the phase table changes"""
  from ..io.importscript import frameKey as key
  return key(f)


def scriptOf(st):
  """the Python script of the import as the window has set it up, empty before a map is loaded"""
  if st.ebsd is None or st.info is None:
    return ""
  return buildImportWizardScript(st.selected, phases=st.ebsd.CSList, headerPhases=st.headerPhases, dataSet=st.dataSet,
                                 dataSets=st.info.dataSets, mapConvention=AXIS_CONVENTIONS[st.mapConvention],
                                 eulerConvention=AXIS_CONVENTIONS[st.eulerConvention], vendorCorrection=st.info.eulerCorrection,
                                 variable=st.variable, drawn={k: phaseCounts(st).get(k, 0) for k, on in st.drawn.items() if on})


def writeScript(st, path=None):
  """the script written to a file, by default beside the session and never in somebody's data folder"""
  text = scriptOf(st)
  if not text:
    return None
  path = path or defaultScriptPath(st)
  with open(path, "w", encoding="utf-8") as f:
    f.write(text)
  st.status = ("loaded", f"written {path}")
  st.changed("status")
  return path


def defaultScriptPath(st, folder=None):
  """where the script is written by default: import_<file>.py in the given folder, else the
  working directory"""
  stem = os.path.splitext(os.path.basename(st.selected))[0]
  return os.path.join(folder or os.getcwd(), f"import_{stem}.py")


__all__ = ["WizardState", "setFolder", "toggleFolder", "pickFile", "selectEntry", "openEntry", "leaveEntry", "startImport", "setDataSet",
           "pickTab", "setMapConvention", "setEulerConvention", "editPhase", "scriptOf", "writeScript"]
