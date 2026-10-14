# the views of the import wizard on one canvas, and the key panel (docs/plans/wizard.md W4, W4f)
#
# The map is drawn once per import through the package's own plot, and a tab that is a
# colouring of it - the phase, the three IPF directions, a property - rewrites the picture's
# colours alone. The pole figures come from the binned density of the dominant drawn phase,
# the images from the metadata.
from __future__ import annotations

import importlib

import numpy as np

import matplotlib
import matplotlib.figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import QMetaObject, Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QLineEdit, QSlider, QToolButton, QVBoxLayout, QWidget

from ..geometry.miller import Miller
from ..io.polefigureformats import millerIndices
from ..geometry.plottingconvention import plottingConvention
from . import callbacks as cb
from .pictogram import SIDE

MAP_KINDS = ("phase", "ipf", "property")
AHEAD_DELAY_MS = 700      # the rest on a file before its pole figures are drawn ahead
MAX_PIXELS = 1_500_000    # of the picture the map is drawn as, whatever the canvas holds
PF_RESOLUTION = 1.5 * np.pi / 180   # the pole figures' grid, fine enough to show no facets
KEY_PIXELS = 300          # the side a key is drawn at, once, before it is a picture
KEY_CACHE = {}
# the module of vector3d, whose xvector, yvector and zvector are minted on every access
_vectors = importlib.import_module("..geometry.vector3d", __package__)


# the window's font as matplotlib takes it: the family Qt resolves its generic one to, where
# matplotlib has that family, mathtext set in it too so that a barred index matches the digits
# around it, and the size in points of a figure of that dpi that give the pixels of the window's
GUI_FALLBACK = "DejaVu Sans"


def guiFont(dpi):
  """the matplotlib settings of the window's font, its family where matplotlib has it and the
  mathtext in it too, and its size in points at the dpi given"""
  from matplotlib import font_manager
  from PySide6.QtGui import QFontInfo
  from PySide6.QtWidgets import QApplication
  family, pixels = GUI_FALLBACK, 13
  if QApplication.instance() is not None:
    info = QFontInfo(QApplication.font())
    pixels = info.pixelSize() or pixels
    try:
      font_manager.findfont(font_manager.FontProperties(family=info.family()), fallback_to_default=False)
      family = info.family()
    except ValueError:
      pass
  rc = {"font.family": family, "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold", "mathtext.sf": family, "mathtext.cal": family, "mathtext.tt": family}
  return rc, pixels * 72 / dpi


def keyPicture(CS, px=KEY_PIXELS):
  """a key is a picture of its phase and of nothing else: it does not change with the map, the
  tab or the size of the window, and matplotlib re-renders a mesh of tens of thousands of
  quads on every draw. So each is drawn once into a figure of its own and shown as an image
  afterwards, which takes a map of five phases from a second a draw to a millisecond. The
  picture is cut to what is drawn, the labels of the key standing outside its sector included.
  A sector reaching into both hemispheres, of a group without the inversion, is drawn as both,
  each named as the key drawn on its own names them, beside or over one another as fills a
  square best."""
  from ..plotting.directionkeys import HSVDirectionKey
  from ..plotting.spherical import makeSphericalProjection
  hit = KEY_CACHE.get((id(CS), px))
  if hit is not None and hit[0] is CS:
    return hit[1]
  key = HSVDirectionKey(CS)
  n = len(makeSphericalProjection(key.keySector, how2plot=CS.how2plot))
  if n == 1:
    picture = _hemispherePicture(key, CS, px, {})
  else:
    # the two are shrunk to share the square, so their labels are drawn larger
    parts = [_hemispherePicture(key, CS, px, {side: True}, fontScale=1.5, label=True) for side in ("upper", "lower")]
    picture = _arrange(parts, gap=max(2, px // 30))
  KEY_CACHE[(id(CS), px)] = (CS, picture)
  return picture


# one chart of a key as a picture cut to what it draws
def _hemispherePicture(key, CS, px, side, fontScale=1.0, label=False):
  import io
  import matplotlib
  import matplotlib.figure
  import matplotlib.image
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from ..plotting.mapplot import keyResolution
  from ..plotting.sphericalplot import plotKey
  rc, _ = guiFont(100)
  with matplotlib.rc_context(rc):
    fig = matplotlib.figure.Figure(figsize=(px / 100, px / 100), dpi=100)
    FigureCanvasAgg(fig)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.patch.set_alpha(0)
    plotKey(key, resolution=keyResolution(px), parent=ax, how2plot=CS.how2plot, fontSize=16 * fontScale * px / KEY_PIXELS, **side)
    # the hemisphere named above all the chart draws, where no label of the key can stand
    if label:
      box = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans).transformed(fig.transFigure.inverted())
      fig.text(box.x0, box.y1 + 0.01, next(iter(side)), ha="left", va="bottom", fontsize=16 * fontScale * px / KEY_PIXELS)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.02, transparent=True)
  buf.seek(0)
  return (matplotlib.image.imread(buf, format="png") * 255 + 0.5).astype(np.uint8)


# two pictures beside or over one another, whichever leaves them larger in a square
def _arrange(parts, gap):
  (h1, w1), (h2, w2) = parts[0].shape[:2], parts[1].shape[:2]
  beside = min(1 / (w1 + w2 + gap), 1 / max(h1, h2))
  over = min(1 / max(w1, w2), 1 / (h1 + h2 + gap))
  if beside >= over:
    out = np.zeros((max(h1, h2), w1 + w2 + gap, 4), dtype=np.uint8)
    out[(out.shape[0] - h1) // 2:(out.shape[0] - h1) // 2 + h1, :w1] = parts[0]
    out[(out.shape[0] - h2) // 2:(out.shape[0] - h2) // 2 + h2, w1 + gap:] = parts[1]
  else:
    out = np.zeros((h1 + h2 + gap, max(w1, w2), 4), dtype=np.uint8)
    out[:h1, (out.shape[1] - w1) // 2:(out.shape[1] - w1) // 2 + w1] = parts[0]
    out[h1 + gap:, (out.shape[1] - w2) // 2:(out.shape[1] - w2) // 2 + w2] = parts[1]
  return out


# the specimen direction of an IPF tab, by the name of the constant of vector3d, which mints a
# fresh vector on every access
IPF_DIRECTIONS = {"IPF X": "xvector", "IPF Y": "yvector", "IPF Z": "zvector"}


class ViewCanvas(FigureCanvasQTAgg):
  """the canvas of the wizard's views: the map in the colouring of the tab, the pole figures of
  the dominant phase drawn ahead from a density computed on the worker, and the pictures of the
  file; redrawn once per turn of the event loop whatever the state announced"""

  def __init__(self, st, worker=None):
    # compressed, so that a map and the colour bar of a property stand centred as one group
    super().__init__(matplotlib.figure.Figure(layout="compressed"))
    self.st = st
    # the window's worker runs the density of the pole figures off the main thread; the
    # (map, phase) pairs being computed there
    self.worker = worker
    self.computing = set()
    # the pole figures drawn ahead as a picture, (what it shows, the pixels), and the timer that
    # draws them anew once a resize has settled
    self.pfPicture = None
    # the height the canvas gains on the pole figure tab, where the bars under it give way: the
    # pole figures drawn ahead are drawn at the size they are shown at
    self.poleFigureRoom = lambda: 0
    self.resized = QTimer(self)
    self.resized.setSingleShot(True)
    self.resized.setInterval(250)
    self.resized.timeout.connect(self.afterResize)
    # the pole figures are drawn ahead only once the cursor has rested on a file, since the
    # drawing holds the window for half a second
    self.ahead = QTimer(self)
    self.ahead.setSingleShot(True)
    self.ahead.setInterval(AHEAD_DELAY_MS)
    self.ahead.timeout.connect(self.drawAhead)
    self.drawnFor = None       # (the map, its convention and layer axis) the base picture stands for; the map is held, so that
                               # no later section can take its identity
    self.image = None
    self.raster = None         # the picture's pixels as measurements, once a tab recolours it
    self.cells = None          # the measurements the cells of a small map stand for, likewise
    self.colorbar = None
    self.mappable = None
    self.legend = None
    self.coloured = None
    self.shownFor = None       # the phases drawn when the picture was last coloured, per measurement
    self.pending = False
    for name in ("ebsd", "section", "drawn", "mapConvention", "millers", "image", "meta"):
      st.on(name, self.scheduleDraw)
    st.on("phases", self.phasesChanged)
    st.on("densities", self.onDensity)
    st.on("selected", self.restartAhead)
    self.mpl_connect("resize_event", lambda event: self.resized.start())

  # ------------------------------------------------------------------ the dispatch

  def scheduleDraw(self):
    """several changes announced in one go draw once, on the next turn of the event loop"""
    if not self.pending:
      self.pending = True
      QMetaObject.invokeMethod(self, "flush", Qt.QueuedConnection)

  def phasesChanged(self):
    """an edited phase recolours the picture, whose tab and drawn phases are the same"""
    self.coloured = None
    self.scheduleDraw()

  @Slot()
  def flush(self):
    """the draw the changes scheduled, run on the event loop"""
    self.pending = False
    self.draw_view()

  def draw_view(self):
    """the view of the tab in force drawn: the map, the pole figures or a picture, the error as
    text and status when the drawing fails"""
    st = self.st
    kind = st.currentKind
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
    except Exception as e:
      import traceback
      traceback.print_exc()
      self.clear(f"{e}")
      st.status = ("failed", f"{st.currentTab}: {e}")
      st.changed("status")

  def clear(self, text=None):
    """the figure emptied, a text in its middle if given"""
    self.figure.clear()
    self.image = self.colorbar = self.mappable = self.legend = self.drawnFor = None
    if text:
      self.figure.text(0.5, 0.5, text, ha="center", va="center", color="#777")
    self.draw_idle()

  # ------------------------------------------------------------------ the map

  def drawMap(self, kind):
    """the map, or the section of a volume, drawn once and recoloured per tab. The picture is drawn in the
    colouring the tab asks for, so the first look at a tab colours nothing twice and builds
    no second raster; a tab after it rewrites the colours of the picture alone, which is
    what the raster index is kept for."""
    from ..plotting.mapplot import plotEBSD
    from ..plotting.mtexfigure import mtexFigure
    st = self.st
    name = st.currentTab
    color, kw = self.colorArgs(kind, name)
    shown = np.array([st.drawn.get(k, True) for k in range(len(st.section.CSList))])[np.asarray(st.section.phaseId).reshape(-1)]
    # a layer across x or y is drawn in its own side view, which the key tells apart from the map's
    key = (st.section, (st.mapConvention, st.sectionAxis if st.isVolume else 2))
    drawn = self.drawnFor if self.image is not None else None
    fresh = drawn is None or drawn[0] is not key[0] or drawn[1] != key[1]
    # another layer across the same axis of a volume lies where the layer drawn lies: the
    # picture stays and takes the colours of the new layer, a fifth of drawing it anew
    relayer = fresh and drawn is not None and drawn[1] == key[1] and st.isVolume and hasattr(self.image, "set_data") \
      and _sameLattice(drawn[0], st.section)
    if relayer:
      self.drawnFor, self.coloured = key, None
    elif fresh:
      self.figure.clear()
      self.colorbar = self.mappable = None
      mtexFigure(1, figure=self.figure)
      # the layer of a volume is a picture however few voxels it holds, so that every layer is drawn alike
      self.image = plotEBSD(st.section, color, legend=False, colorbar=False, maxPixels=MAX_PIXELS,
                            **({"backend": "raster", "how2plot": cb.sectionConvention(st)} if st.isVolume else {}), **kw)
      # the artist's own alpha would override the transparency of a phase not drawn
      self.image.set_alpha(None)
      self.ax = self.image.axes
      self.raster, self.cells, self.drawnFor, self.coloured, self.shownFor = None, None, key, (kind, name), shown
      # a property is the only colouring that is a number, and it states the range it was
      # drawn over, so the bar needs nothing read back; every other picture carries none
      norm = getattr(self.image, "norm", None)
      scalar = kind == "property" and np.ndim(self.image.get_array() if hasattr(self.image, "get_array") else None) in (1, 2)
      self.setColorbar(matplotlib.colormaps["viridis"] if scalar else None,
                       getattr(norm, "vmin", 0.0) or 0.0, getattr(norm, "vmax", 1.0) or 1.0, name)
      if shown.all():
        self.draw_idle()
        self.prepareDensity()
        return
    # the picture shows these colours already: a change that touched none of them draws nothing
    elif self.coloured == (kind, name) and self.shownFor is not None and np.array_equal(self.shownFor, shown):
      return
    self.recolour(kind, name, shown)
    self.shownFor = shown
    self.draw_idle()
    self.prepareDensity()

  def recolour(self, kind, name, shown):
    """the colours of the picture rewritten, the map's own never assembled: mapValues colours
    the measurements the picture shows and no others"""
    from ..plotting.maps import mapCells, mapRaster
    st = self.st
    layout = getattr(getattr(self.ax, "_mtexMapPlot", None), "layout", None)
    if self.raster is None and hasattr(self.image, "set_data"):
      self.raster = mapRaster(st.section, maxPixels=MAX_PIXELS, layout=layout)
    # the cells were drawn without the padding and without the measurements the first colouring
    # had no colour for: every measurement gets its cell, so that a colour meets its own
    if self.cells is None and hasattr(self.image, "set_verts") and st.section.lattice is not None:
      self.cells = np.flatnonzero(~st.section.isPadding.reshape(-1))
      self.image.set_verts(mapCells(st.section, layout=layout)[self.cells])
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
      rgba = np.zeros((st.section.size, 4))
      if scalar:
        v = np.asarray(values, dtype=float).reshape(-1)
        finite = np.isfinite(v)
        rgba[finite] = cmap((v[finite] - lo) / max(hi - lo, 1e-12))
      else:
        rgb = np.asarray(values, dtype=float).reshape(-1, 3)
        rgba[:, :3] = np.nan_to_num(rgb)
        rgba[:, 3] = np.isfinite(rgb).all(axis=-1)
      rgba[~shown, 3] = 0
      if self.cells is not None:
        rgba = rgba[self.cells]
      if hasattr(self.image, "set_facecolors"):
        # cells drawn by a number carry it, and it would colour them over the colours given
        self.image.set_array(None)
        self.image.set_facecolors(rgba)
      elif hasattr(self.image, "set_color"):
        self.image.set_color(rgba)

  def colorArgs(self, kind, name):
    """what the tab colours the map by: its phases, the crystal direction along a specimen
    direction, or one of its properties"""
    if kind == "phase":
      return "phase", {}
    if kind == "ipf":
      return "orientation", {"ipfDirection": getattr(_vectors, IPF_DIRECTIONS[name])}
    return name, {}

  def colorsOf(self, kind, name, at=None):
    """the colours or values of the map the tab asks for, of the measurements at alone if given"""
    from ..plotting.mapplot import mapValues
    color, kw = self.colorArgs(kind, name)
    return mapValues(self.st.section, color, at=at, **kw)

  def setColorbar(self, cmap, lo=0.0, hi=1.0, label=""):
    """the colour bar of a property, which the picture does not carry: it is made from a
    mappable of its own. That mappable is kept and re-normed rather than replaced, since a
    colour bar can only be removed through the one the figure registered it with. A colour bar
    anchors the map at its right side, beside the bar, which removing it does not undo: the map
    is centred again then."""
    if cmap is None:
      if self.colorbar is not None:
        self.colorbar.remove()
        self.colorbar, self.mappable = None, None
        self.ax.set_anchor("C")
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
  # package's own plotPF; the axes of the Euler frame drawn in red on every chart so that a
  # wrong Euler convention is visible

  def density(self, k):
    """the density of a phase, computed on the worker where there is one, so that the tab opens
    on it at once; None while it is computed, the tab redrawn when it is there. A result for a
    map that has since been replaced is dropped."""
    from .binning import previewDensity
    st = self.st
    if k in st.densities:
      return st.densities[k]
    if self.worker is None:
      st.densities[k] = previewDensity(st.ebsd, k)
      return st.densities[k]
    ebsd = st.ebsd
    if (id(ebsd), k) not in self.computing:
      self.computing.add((id(ebsd), k))

      def commit(odf):
        self.computing.discard((id(ebsd), k))
        if st.ebsd is ebsd and not isinstance(odf, BaseException):
          st.densities[k] = odf
          st.changed("densities")
      self.worker(lambda: previewDensity(ebsd, k), commit)
    return None

  def prepareDensity(self):
    """the density of the phase the pole figures will show started behind the map, on the worker"""
    k = cb.dominantPhase(self.st)
    if self.worker is not None and k is not None:
      self.density(k)

  def onDensity(self):
    """a density arrived: the pole figures drawn, or drawn ahead once the cursor has rested on
    the file, so that the tab opens on a finished picture"""
    if self.onPoleFigures():
      self.scheduleDraw()
    else:
      self.ahead.start()

  def restartAhead(self):
    """another file selected: the pole figures waiting to be drawn ahead wait again"""
    if self.ahead.isActive():
      self.ahead.start()

  def onPoleFigures(self):
    """whether the tab in force is the pole figure tab"""
    return self.st.currentKind == "polefigures"

  def poleFigureKey(self):
    """what the picture of the pole figures shows, None when the density is not there yet"""
    st = self.st
    k = cb.dominantPhase(st) if st.ebsd is not None else None
    if k is None or k not in st.densities:
      return None
    return (id(st.densities[k]), k, tuple(st.millers), tuple(st.eulerAxesNames), *self.poleFigureSize())

  def poleFigureSize(self):
    """the size in pixels the pole figure tab shows its picture at, from any tab"""
    w, h = self.figure.get_size_inches()
    room = 0 if self.onPoleFigures() else self.poleFigureRoom()
    return round(w * self.figure.dpi), round(h * self.figure.dpi) + room

  @Slot()
  def drawAhead(self):
    """the picture of the pole figures drawn off the screen, unless the one kept shows the same"""
    key = self.poleFigureKey()
    if key is not None and (self.pfPicture is None or self.pfPicture[0] != key):
      self.pfPicture = (key, self.renderPoleFigures(key))

  @Slot()
  def afterResize(self):
    """a resize draws the pole figures anew at the new size, once it has settled"""
    if self.onPoleFigures():
      self.scheduleDraw()
    else:
      self.drawAhead()

  def drawPoleFigures(self):
    """the tab shows the picture drawn ahead, drawing it first when it is not there or stale"""
    st = self.st
    k = cb.dominantPhase(st)
    if k is None:
      self.clear("no indexed phase is drawn")
      return
    if self.density(k) is None:
      self.clear("computing the pole figures ...")
      return
    self.drawAhead()
    self.figure.clear()
    self.image = self.colorbar = self.mappable = self.drawnFor = None
    # the picture pixel for pixel, as it was drawn at the canvas's size, with no axes around it
    self.figure.figimage(self.pfPicture[1], origin="upper")
    self.draw_idle()

  def renderPoleFigures(self, key):
    """the pole figures of the dominant phase drawn off the screen at the canvas's size: a colour
    bar under each on its own range, the titles close above, the Euler frame's axes as red
    names on a white ground where they point; the row in a band as tall as it needs across the
    middle, since a chart keeps its aspect in a taller cell and its title and bar would stand
    at the ends of the cell, far from it"""
    import matplotlib.figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from ..plotting.mtexfigure import mtexFigure
    from ..plotting.sphericalplot import annotate, plotPF
    st = self.st
    _, k, millers, names, w, h = key
    cs = st.ebsd.CSList.frames[k]
    hs = [millerOf(text, cs) for text in millers]
    # in the window's font: the titles a step above its size, the names and the ticks at it
    rc, size = guiFont(self.figure.dpi)
    with matplotlib.rc_context(rc):
      fig = matplotlib.figure.Figure(figsize=(w / self.figure.dpi, h / self.figure.dpi), dpi=self.figure.dpi, layout="constrained")
      FigureCanvasAgg(fig)
      frac = min(1.0, (0.92 * w / len(hs) + 90) / max(h, 1))
      fig.get_layout_engine().set(rect=(0, (1 - frac) / 2, 1, frac))
      mtexFigure(len(hs), figure=fig)
      plotPF(st.densities[k], hs, colorbar=False, resolution=PF_RESOLUTION, fontSize=size + 1)
      for ax in list(fig.axes):
        sP = getattr(ax, "_mtexSphericalPlot", None)
        if sP is None:
          continue
        ax.set_title(ax.get_title(), fontsize=size + 2, pad=0)
        for t in sP.labels:
          t.set_fontsize(size)
        if sP.mappable is not None:
          bar = fig.colorbar(sP.mappable, ax=ax, location="bottom", shrink=0.7, aspect=25, pad=0)
          bar.ax.tick_params(labelsize=size - 1)
      corr = st.ebsd.EulerCorrection
      for v, n in zip((_vectors.xvector, _vectors.yvector, _vectors.zvector), names):
        try:
          annotate(corr * v, label=n, markerSize=0, backgroundColor="w", color="red", labelFontSize=size, add2all=True)
        except Exception:
          pass
      fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


  # ------------------------------------------------------------------ the images

  def drawImage(self):
    """the picture of the file the state points at, with its label and number"""
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
# whether two maps lie on the same lattice, cell for cell
def _sameLattice(a, b):
  la, lb = a.lattice, b.lattice
  return la is not None and lb is not None and a.shape == b.shape and np.array_equal(la.A, lb.A) and np.array_equal(la.origin, lb.origin)


def _range(values):
  v = np.asarray(values, dtype=float).reshape(-1)
  finite = np.isfinite(v)
  return (float(np.min(v[finite])), float(np.max(v[finite]))) if finite.any() else (0.0, 1.0)


def millerOf(text, CS):
  """a Miller index typed as '100', '1-10' or '0001', on the frame's lattice"""
  digits = millerIndices(text)
  return Miller(1, 0, 0, CS) if digits is None else Miller(*digits, CS)


# ------------------------------------------------------------------ the section of a volume

class SectionBar(QWidget):
  """the layer of a volume the views draw: the axis it lies across, x, y or z, and a slider
  through its layers; shown for the maps of a volume alone, not its pole figures"""

  def __init__(self, st):
    super().__init__()
    self.st = st
    lay = QHBoxLayout(self)
    lay.setContentsMargins(6, 0, 6, 0)
    lay.addWidget(QLabel("section across"))
    self.axes = QButtonGroup(self)
    for k, name in enumerate("xyz"):
      b = QToolButton()
      b.setText(name)
      b.setCheckable(True)
      b.setToolTip(f"the layers of the volume across {name}")
      self.axes.addButton(b, k)
      lay.addWidget(b)
    self.slider = QSlider(Qt.Horizontal)
    self.slider.setToolTip("the layer shown")
    lay.addWidget(self.slider, 1)
    self.label = QLabel()
    lay.addWidget(self.label)
    self.axes.idClicked.connect(lambda k: cb.setSection(self.st, axis=k))
    self.slider.valueChanged.connect(lambda v: cb.setSection(self.st, index=v))
    for name in ("ebsd", "section", "tab", "tabs"):
      st.on(name, self.refresh)
    self.refresh()

  def refresh(self):
    """the controls set to the layer of the state, without announcing it again"""
    st = self.st
    self.setVisible(st.isVolume and st.currentKind != "polefigures")
    if not st.isVolume:
      return
    n = st.ebsd.shape[st.sectionAxis]
    self.axes.button(st.sectionAxis).setChecked(True)
    self.slider.blockSignals(True)
    self.slider.setRange(0, n - 1)
    self.slider.setValue(st.sectionIndex)
    self.slider.blockSignals(False)
    self.label.setText(cb.sectionLabel(st))


# ------------------------------------------------------------------ the Miller indices

class MillerBar(QWidget):
  """the three crystal directions of the pole figures, each a field that is its label; in the
  window it floats over the top left corner of the canvas, above the row of pole figures, and
  takes no room from the maps"""

  def __init__(self, st, parent=None):
    super().__init__(parent)
    self.st = st
    lay = QHBoxLayout(self)
    lay.setContentsMargins(6, 4, 6, 0)
    self.lead = QLabel("pole figures of")
    lay.addWidget(self.lead)
    self.fields = []
    for k in range(3):
      f = QLineEdit(st.millers[k])
      f.setFixedWidth(64)
      f.setToolTip("a crystal direction, as 100 or 1-10")
      f.editingFinished.connect(lambda k=k, f=f: self.commit(k, f))
      lay.addWidget(f)
      self.fields.append(f)
    lay.addStretch(1)
    for name in ("tab", "tabs", "ebsd", "phases", "drawn"):
      st.on(name, self.show_)
    self.show_()

  def commit(self, k, f):
    """the crystal direction of the field k taken into the state when its text changed"""
    if f.text().strip() != self.st.millers[k]:
      cb.setMiller(self.st, k, f.text())

  def show_(self):
    """shown on the pole figure tab alone, naming the phase the pole figures are of"""
    st = self.st
    self.setVisible(st.currentKind == "polefigures")
    k = cb.dominantPhase(st) if st.ebsd is not None else None
    self.lead.setText("pole figures of" if k is None else f"{st.ebsd.CSList.frames[k].name}: pole figures of")
    if self.parentWidget() is not None:
      self.adjustSize()
      self.raise_()


# ------------------------------------------------------------------ the key

class KeyPanel(QWidget):
  """the inverse pole figure key of one phase, at a fixed size in a fixed place: the phase
  picked in the phase table, at first the one with the most pixels"""

  def __init__(self, st):
    super().__init__()
    self.st = st
    lay = QVBoxLayout(self)
    lay.setContentsMargins(6, 4, 6, 4)
    lay.setSpacing(2)
    self.title = QLabel("x")
    self.title.setAlignment(Qt.AlignCenter)
    # as tall as the two frame pictograms beside the phase table, less the frame around it (a
    # border and a margin of a pixel each, above and below), so the row keeps their height;
    # the picture a square of what the title leaves
    height = 2 * SIDE + 3 - 4
    self.side = height - 8 - 2 - self.title.sizeHint().height()
    self.picture = QLabel()
    self.picture.setFixedSize(self.side, self.side)
    self.picture.setAlignment(Qt.AlignCenter)
    lay.addWidget(self.title)
    lay.addWidget(self.picture, 0, Qt.AlignHCenter)
    self.setToolTip("the colour key of the IPF maps; click a phase in the table to show its key")
    self.setFixedSize(self.side + 12, height)
    for name in ("ebsd", "phases", "keyPhase"):
      st.on(name, self.refresh)
    self.refresh()

  def refresh(self):
    """the key of the phase picked drawn anew at the pixels of the screen, empty for none"""
    st = self.st
    frames = st.ebsd.CSList.frames if st.ebsd is not None else []
    k = st.keyPhase
    if k is None or not 0 <= k < len(frames) or not frames[k].isIndexed:
      self.title.setText("")
      self.picture.clear()
      return
    cs = frames[k]
    self.title.setText(f"<b>{cs.name}</b> &nbsp;{cs.pointGroup}")
    # drawn at the pixels of the screen, so the picture is not scaled after
    ratio = self.devicePixelRatioF()
    side = int(round(self.side * ratio))
    rgba = np.ascontiguousarray(keyPicture(cs, px=side))
    h, w = rgba.shape[:2]
    pixmap = QPixmap.fromImage(QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy())
    # the picture with its labels fitted into the square, whatever shape the sector has
    pixmap = pixmap.scaled(side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    pixmap.setDevicePixelRatio(ratio)
    self.picture.setPixmap(pixmap)


__all__ = ["ViewCanvas", "KeyPanel", "MillerBar"]
