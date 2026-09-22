# the views of the import wizard on one canvas, and the metadata tree (docs/plans/wizard.md W4)
#
# The map is drawn once per import through the package's own plot, and a tab that is a
# colouring of it - the phase, the three IPF directions, a property - rewrites the picture's
# colours alone. The pole figures come from the binned density of the dominant drawn phase,
# the images from the metadata.
from __future__ import annotations

import numpy as np

import matplotlib
import matplotlib.figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import QMetaObject, Qt, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QAbstractItemView, QFrame, QHBoxLayout, QLabel, QLineEdit, QTreeWidget, QTreeWidgetItem, QWidget

from ..geometry.miller import Miller
from ..geometry.plottingconvention import plottingConvention
from ..geometry.vector3d import xvector, yvector, zvector
from . import callbacks as cb

MAP_KINDS = ("phase", "ipf", "property")
KEY_WIDTH = 0.28
MAX_PIXELS = 1_500_000    # of the picture the map is drawn as, whatever the canvas holds
PICTURE_INK = "#1f4e79"   # a leaf of the metadata that is a picture, and can be looked at


KEY_PIXELS = 300          # the side a key is drawn at, once, before it is a picture
KEY_CACHE = {}


# a key is a picture of its phase and of nothing else: it does not change with the map, the
# tab or the size of the window, and matplotlib re-renders a mesh of tens of thousands of
# quads on every draw. So each is drawn once into a figure of its own and shown as an image
# afterwards, which takes a map of five phases from a second a draw to a millisecond.
def keyPicture(cs, px=KEY_PIXELS):
  import matplotlib.figure
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from ..plotting.directionkeys import HSVDirectionKey
  from ..plotting.mapplot import keyResolution
  from ..plotting.sphericalplot import plotKey
  hit = KEY_CACHE.get(id(cs))
  if hit is not None and hit[0] is cs:
    return hit[1]
  fig = matplotlib.figure.Figure(figsize=(px / 100, px / 100), dpi=100)
  FigureCanvasAgg(fig)
  fig.patch.set_alpha(0)
  ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
  ax.patch.set_alpha(0)
  plotKey(HSVDirectionKey(cs), resolution=keyResolution(px), parent=ax, how2plot=cs.how2plot, fontSize=16)
  fig.canvas.draw()
  picture = np.asarray(fig.canvas.buffer_rgba()).copy()
  KEY_CACHE[id(cs)] = (cs, picture)
  return picture


# an inset in the map's own coordinates, which follows the map when the window is resized
class KeyPlacement:

  def __init__(self, parent, bounds):
    self.parent, self.bounds = parent, bounds

  def __call__(self, ax, renderer):
    from matplotlib.transforms import Bbox, TransformedBbox
    box = TransformedBbox(Bbox.from_bounds(*self.bounds), self.parent.transAxes)
    return TransformedBbox(box, ax.figure.transSubfigure.inverted())
IPF_DIRECTIONS = {"IPF X": lambda: xvector, "IPF Y": lambda: yvector, "IPF Z": lambda: zvector}


class ViewCanvas(FigureCanvasQTAgg):

  def __init__(self, st):
    super().__init__(matplotlib.figure.Figure(layout="constrained"))
    self.st = st
    self.drawnFor = None       # (map identity, map convention) the base picture stands for
    self.image = None
    self.colorbar = None
    self.mappable = None
    self.legend = None
    self.keyAxes = []
    self.keyGrounds = []
    self.keyPending = False
    self.coloured = None
    self.pending = False
    for name in ("ebsd", "drawn", "phases", "mapConvention", "millers", "image", "meta"):
      st.on(name, self.scheduleDraw)
    # the side the keys stand on follows the shape of the canvas
    self.mpl_connect("resize_event", lambda event: self.placeKeys())

  # ------------------------------------------------------------------ the dispatch

  # several changes announced in one go draw once, on the next turn of the event loop
  def scheduleDraw(self):
    if not self.pending:
      self.pending = True
      QMetaObject.invokeMethod(self, "flush", Qt.QueuedConnection)

  @Slot()
  def flush(self):
    self.pending = False
    self.draw_view()

  def draw_view(self):
    st = self.st
    kind = st.tabKinds[st.tab] if 0 <= st.tab < len(st.tabKinds) else None
    if st.ebsd is None or kind is None:
      self.clear()
      return
    try:
      if kind in MAP_KINDS:
        self.drawMap(kind)
      elif kind == "polefigures":
        self.drawPoleFigures()
      elif kind == "images":
        self.drawImage()
      else:
        self.clear("the metadata is in the tree above")
    except Exception as e:
      import traceback
      traceback.print_exc()
      self.clear(f"{e}")
      st.status = ("failed", f"{st.currentTab}: {e}")
      st.changed("status")

  def clear(self, text=None):
    self.figure.clear()
    self.image = self.colorbar = self.mappable = self.legend = self.drawnFor = None
    self.keyAxes, self.keyGrounds = [], []
    if text:
      self.figure.text(0.5, 0.5, text, ha="center", va="center", color="#777")
    self.draw_idle()

  # ------------------------------------------------------------------ the map

  # the map drawn once per import and recoloured per tab. The picture is drawn in the
  # colouring the tab asks for, so the first look at a tab colours nothing twice and builds
  # no second raster; a tab after it rewrites the colours of the picture alone, which is
  # what the raster index is kept for.
  def drawMap(self, kind):
    from ..plotting.mapplot import plotEBSD
    from ..plotting.mtexfigure import mtexFigure
    st = self.st
    name = st.currentTab
    color, kw = self.colorArgs(kind, name)
    shown = np.array([st.drawn.get(k, True) for k in range(len(st.ebsd.CSList))])[np.asarray(st.ebsd.phaseId).reshape(-1)]
    key = (id(st.ebsd), st.mapConvention)
    if self.drawnFor != key or self.image is None:
      self.figure.clear()
      self.colorbar = self.mappable = None
      self.keyAxes, self.keyGrounds = [], []
      mtexFigure(1, figure=self.figure)
      self.image = plotEBSD(st.ebsd, color, legend=False, colorbar=False, maxPixels=MAX_PIXELS, **kw)
      # the artist's own alpha would override the transparency of a phase not drawn
      self.image.set_alpha(None)
      self.ax = self.image.axes
      self.raster, self.drawnFor, self.coloured = None, key, (kind, name)
      # a property is the only colouring that is a number, and it states the range it was
      # drawn over, so the bar needs nothing read back; every other picture carries none
      norm = getattr(self.image, "norm", None)
      scalar = kind == "property" and np.ndim(self.image.get_array() if hasattr(self.image, "get_array") else None) in (1, 2)
      self.setColorbar(matplotlib.colormaps["viridis"] if scalar else None,
                       getattr(norm, "vmin", 0.0) or 0.0, getattr(norm, "vmax", 1.0) or 1.0, name)
      self.setKeyLegend(kind == "ipf")
      if shown.all():
        self.draw_idle()
        return
    self.recolour(kind, name, shown)
    self.setKeyLegend(kind == "ipf")
    self.draw_idle()

  # the colours of the picture rewritten, the map's own never assembled: mapValues colours
  # the measurements the picture shows and no others
  def recolour(self, kind, name, shown):
    from ..plotting.maps import mapRaster
    st = self.st
    if self.raster is None and hasattr(self.image, "set_data"):
      layout = getattr(getattr(self.ax, "_mtexMapPlot", None), "layout", None)
      self.raster = mapRaster(st.ebsd, maxPixels=MAX_PIXELS, layout=layout)
    index = self.raster[2] if self.raster is not None else None
    pick = np.where(index >= 0, index, 0) if index is not None else None
    values, scalar = self.colorsOf(kind, name, at=None if pick is None else pick.reshape(-1))
    lo, hi = _range(values) if scalar else (0.0, 1.0)
    cmap = matplotlib.colormaps["viridis"]
    self.setColorbar(cmap if scalar else None, lo, hi, name)
    self.coloured = (kind, name)
    if index is not None:
      covered = index >= 0
      if scalar:
        # a number keeps the map's own range, so that route is not narrowed
        v = np.asarray(values, dtype=float).reshape(-1)[pick]
        img = cmap((v - lo) / max(hi - lo, 1e-12), bytes=True)
        opaque = covered & shown[pick] & np.isfinite(v)
      else:
        rgb = np.asarray(values).reshape(index.shape + (3,))
        img = np.zeros(index.shape + (4,), dtype=np.uint8)
        img[..., :3] = (np.nan_to_num(np.clip(rgb, 0, 1)) * 255 + 0.5).astype(np.uint8)
        # a measurement whose colour is no colour, an unindexed one, is drawn as nothing
        opaque = covered & shown[pick] & np.isfinite(rgb).all(axis=-1)
      img[..., 3] = np.where(opaque, 255, 0)
      self.image.set_data(img)
    else:
      # a map on no lattice is drawn as its cells, one colour per measurement
      rgba = np.zeros((st.ebsd.size, 4))
      if scalar:
        v = np.asarray(values, dtype=float).reshape(-1)
        finite = np.isfinite(v)
        rgba[finite] = cmap((v[finite] - lo) / max(hi - lo, 1e-12))
      else:
        rgb = np.asarray(values, dtype=float).reshape(-1, 3)
        rgba[:, :3] = np.nan_to_num(rgb)
        rgba[:, 3] = np.isfinite(rgb).all(axis=-1)
      rgba[~shown, 3] = 0
      if hasattr(self.image, "set_facecolors"):
        self.image.set_facecolors(rgba)
      elif hasattr(self.image, "set_color"):
        self.image.set_color(rgba)

  # ------------------------------------------------------------------ the keys

  # the inverse pole figure key of every phase beside an IPF map, as MATLAB shows it. It is
  # built on the turn of the event loop after the map is drawn and not before it: the keys
  # of a map of five phases cost a second, and a second of the map not being there is what
  # the user would see. Once built they are kept and hidden on the tabs that are not IPF.
  def setKeyLegend(self, on):
    if on and not self.keyAxes and not self.keyPending:
      self.keyPending = True
      QMetaObject.invokeMethod(self, "buildKeys", Qt.QueuedConnection)
    for ax in self.keyAxes:
      ax.set_visible(on)
    self.placeKeys()

  @Slot()
  def buildKeys(self):
    st = self.st
    self.keyPending = False
    if self.image is None or st.ebsd is None or st.tabKinds[st.tab] != "ipf":
      return
    pl = st.ebsd.CSList
    present = set(np.asarray(st.ebsd.phaseId).reshape(-1).tolist())
    ids = [i for i in pl.indexedPhaseId if i in present]
    if len(self.keyAxes) >= len(ids):
      return
    # one key a turn: five of them cost two seconds to draw the first time, and a window
    # that cannot be moved for two seconds is what building them together would give
    i = ids[len(self.keyAxes)]
    step = 1 / max(len(ids), 1)
    inset = self.ax.inset_axes([1.08, 0, KEY_WIDTH, 0.7 * step])
    inset.imshow(keyPicture(pl[i]), interpolation="antialiased")
    inset.set_axis_off()
    inset.set_title(pl[i].mineral or pl[i].name, fontsize=max(6, min(10, 22 * step)) + 1)
    self.keyAxes.append(inset)
    self.placeKeys()
    self.draw_idle()
    if len(self.keyAxes) < len(ids):
      self.keyPending = True
      QMetaObject.invokeMethod(self, "buildKeys", Qt.QueuedConnection)

  # the keys stand beside a map that leaves room for them and over one that does not: a map
  # wider than the canvas is drawn to its whole width, and keys outside would take that
  # width from it - the layout counts them, so the map lost a quarter of its width to three
  def placeKeys(self):
    from matplotlib.patches import Rectangle
    for r in self.keyGrounds:
      r.remove()
    self.keyGrounds = []
    if not self.keyAxes:
      return
    w, h = self.figure.get_size_inches()
    xlo, xhi = self.ax.get_xlim()
    ylo, yhi = self.ax.get_ylim()
    over = abs(xhi - xlo) / max(abs(yhi - ylo), 1e-12) > w / max(h, 1e-12)
    n = len(self.keyAxes)
    # over the map the stack is fitted inside it, since a key that runs off the edge is cut
    # and the ground of each is a child of its own axes, dropped when they move back out
    step = (0.9 if over else 1.0) / n
    # beside the map the stack stands clear of it, but never past the edge of the canvas:
    # what is outside the figure is cut, and what is cut is the labels of the key
    box = self.ax.get_position()
    beside = min(1.08, (0.995 - box.x0) / max(box.width, 1e-9) - KEY_WIDTH)
    for row, ax in enumerate(self.keyAxes):
      x0 = 0.985 - KEY_WIDTH if over else beside
      y0 = (0.93 if over else 1.0) - (row + 0.9) * step
      ax.set_axes_locator(KeyPlacement(self.ax, [x0, y0, KEY_WIDTH, 0.7 * step]))
      if over:
        # a key over a map carries its own ground, or it is read against the colours. The
        # ground is a child of the key's own axes, so it follows the box the aspect leaves
        # it; the box is drawn in its coordinates from below its labels to above its name.
        ground = Rectangle((-0.12, -0.1), 1.24, 1.32, transform=ax.transAxes, facecolor="white", alpha=0.78,
                           edgecolor="none", zorder=-5, clip_on=False)
        ax.add_patch(ground)
        self.keyGrounds.append(ground)

  # what the tab colours the map by: its phases, the crystal direction along a specimen
  # direction, or one of its properties
  def colorArgs(self, kind, name):
    if kind == "phase":
      return "phase", {}
    if kind == "ipf":
      return "orientation", {"ipfDirection": IPF_DIRECTIONS[name]()}
    return name, {}

  def colorsOf(self, kind, name, at=None):
    from ..plotting.mapplot import mapValues
    color, kw = self.colorArgs(kind, name)
    return mapValues(self.st.ebsd, color, at=at, **kw)

  # the colour bar of a property, which the picture does not carry: it is made from a
  # mappable of its own. That mappable is kept and re-normed rather than replaced, since a
  # colour bar can only be removed through the one the figure registered it with.
  def setColorbar(self, cmap, lo=0.0, hi=1.0, label=""):
    if cmap is None:
      if self.colorbar is not None:
        self.colorbar.remove()
        self.colorbar, self.mappable = None, None
      return
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    if self.colorbar is None:
      self.mappable = ScalarMappable(norm=Normalize(lo, hi), cmap=cmap)
      self.colorbar = self.figure.colorbar(self.mappable, ax=self.ax, fraction=0.04, pad=0.02)
    else:
      self.mappable.set_norm(Normalize(lo, hi))
      self.mappable.set_cmap(cmap)
      self.colorbar.update_normal(self.mappable)
    self.colorbar.set_label(label)

  # ------------------------------------------------------------------ the pole figures

  # the pole figures of the dominant drawn phase from its binned density, through the
  # package's own plotPDF; the axes of the Euler frame drawn in red on every chart so that a
  # wrong Euler convention is visible
  def drawPoleFigures(self):
    from ..plotting.mtexfigure import mtexFigure
    from ..plotting.sphericalplot import annotate, plotPDF
    from .binning import binnedDensity
    st = self.st
    self.figure.clear()
    self.image = self.colorbar = self.mappable = self.drawnFor = None
    k = cb.dominantPhase(st)
    if k is None:
      self.clear("no indexed phase is drawn")
      return
    if k not in st.densities:
      st.densities[k] = binnedDensity(st.ebsd[st.ebsd.CSList.frames[k]].orientations)
    odf = st.densities[k]
    cs = st.ebsd.CSList.frames[k]
    hs = [millerOf(text, cs) for text in st.millers]
    mtexFigure(len(hs), figure=self.figure)
    plotPDF(odf, hs, colorbar=False)
    corr = st.ebsd.EulerCorrection
    for v, n in zip((xvector, yvector, zvector), st.eulerAxesNames):
      try:
        annotate(corr * v, label=n, MarkerFaceColor="red", MarkerEdgeColor="red", color="red", add2all=True)
      except Exception:
        pass
    self.figure.suptitle(f"{cs.name}: pole figures of the density", fontsize=9)
    self.draw_idle()


  # ------------------------------------------------------------------ the images

  def drawImage(self):
    st = self.st
    self.figure.clear()
    self.image = self.colorbar = self.mappable = self.drawnFor = None
    if not st.images:
      self.clear("the file brought no picture")
      return
    k = min(st.image, len(st.images) - 1)
    label, _ = st.images[k]
    picture = cb.pictureOf(st, k)
    ax = self.figure.add_subplot(111)
    ax.imshow(picture, cmap=None if picture.ndim == 3 else "gray")
    ax.set_axis_off()
    ax.set_title(f"{label}   ({k + 1} of {len(st.images)})" if len(st.images) > 1 else label, fontsize=9)
    self.draw_idle()


# the range a property is coloured over, its finite values
def _range(values):
  v = np.asarray(values, dtype=float).reshape(-1)
  finite = np.isfinite(v)
  return (float(np.min(v[finite])), float(np.max(v[finite]))) if finite.any() else (0.0, 1.0)


# a Miller index typed as '100', '1-10' or '0001', on the frame's lattice
def millerOf(text, cs):
  try:
    digits = [int(c) if c.isdigit() else -int(c[1]) for c in _tokens(text)]
    return Miller(*digits, cs)
  except Exception:
    return Miller(1, 0, 0, cs)


def _tokens(text):
  out, i = [], 0
  text = text.replace(" ", "").replace(",", "")
  while i < len(text):
    if text[i] == "-" and i + 1 < len(text):
      out.append(text[i:i + 2])
      i += 2
    else:
      out.append(text[i])
      i += 1
  return out


# ------------------------------------------------------------------ the Miller indices

class MillerBar(QWidget):
  """the three crystal directions of the pole figures, each a field that is its label"""

  def __init__(self, st):
    super().__init__()
    self.st = st
    lay = QHBoxLayout(self)
    lay.setContentsMargins(6, 0, 6, 0)
    lay.addWidget(QLabel("pole figures of"))
    self.fields = []
    for k in range(3):
      f = QLineEdit(st.millers[k])
      f.setFixedWidth(64)
      f.setToolTip("a crystal direction, as 100 or 1-10")
      f.editingFinished.connect(lambda k=k, f=f: self.commit(k, f))
      lay.addWidget(f)
      self.fields.append(f)
    lay.addStretch(1)
    st.on("tab", self.show_)
    st.on("tabs", self.show_)
    self.show_()

  def commit(self, k, f):
    if f.text().strip() != self.st.millers[k]:
      cb.setMiller(self.st, k, f.text())

  def show_(self):
    st = self.st
    self.setVisible(0 <= st.tab < len(st.tabKinds) and st.tabKinds[st.tab] == "polefigures")


# ------------------------------------------------------------------ the metadata

class MetaTree(QTreeWidget):
  """the metadata of the map as a tree, a value preview per leaf

  A picture is a leaf like any other and says so; clicking it shows it, which is what the
  images view is for.
  """

  def __init__(self, st, small):
    super().__init__()
    self.st = st
    self.small = small
    self.setColumnCount(2)
    # the columns say what they hold; a band reading field and value over them does not
    self.setHeaderLabels(["field", "value"])
    self.setHeaderHidden(True)
    self.setFrameShape(QFrame.NoFrame)
    self.setSelectionMode(QAbstractItemView.SingleSelection)
    self.setIndentation(12)
    self.header().setStretchLastSection(True)
    self.setColumnWidth(0, 180)
    self.itemClicked.connect(self.onClick)
    st.on("meta", self.refresh)
    st.on("image", self.mark)
    self.refresh()

  def refresh(self):
    self.clear()
    stack = [(-1, self.invisibleRootItem())]
    for i, (depth, key, value, branch, picture) in enumerate(self.st.metaRows):
      while stack and stack[-1][0] >= depth:
        stack.pop()
      parent = stack[-1][1] if stack else self.invisibleRootItem()
      item = QTreeWidgetItem(parent, [key, value])
      item.setData(0, Qt.UserRole, i)
      if branch:
        stack.append((depth, item))
        item.setExpanded(i in self.st.metaOpen)
      elif picture is not None:
        item.setForeground(1, QColor(PICTURE_INK))
        item.setToolTip(1, "click to look at it")
    self.mark()

  # the picture being looked at is the one selected in the tree
  def mark(self):
    for item in self.walk():
      i = item.data(0, Qt.UserRole)
      row = self.st.metaRows[i] if i is not None and i < len(self.st.metaRows) else None
      if row is not None and row[4] is not None and row[4] == self.st.image:
        self.setCurrentItem(item)

  def walk(self, parent=None):
    parent = parent or self.invisibleRootItem()
    for k in range(parent.childCount()):
      child = parent.child(k)
      yield child
      yield from self.walk(child)

  def onClick(self, item, column):
    i = item.data(0, Qt.UserRole)
    if i is not None:
      cb.pickMeta(self.st, i)


__all__ = ["ViewCanvas", "MetaTree", "MillerBar"]
