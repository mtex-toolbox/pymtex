# an image that knows where on the specimen it sits (MATLAB EBSDAnalysis/@mapImage)
#
#   mg = mapImage(img)                            unit step, a frame of its own
#   mg = mapImage(img, dxy=0.05, name='bse')      square pixels; dxy=(dx, dy) rectangular ones
#   mg = mapImage(ebsd.bc, ebsd, name='bcImg')    a channel of a gridded square map, the map riding along
#   mg.img, mg.origin, mg.d1, mg.d2, mg.frame, mg.pos, mg.layout, mg.dx, mg.dy, mg.nChannel
#   v = interp(mg, pos), pos2ind(mg, pos), subGrid(mg, rows, cols), extent(mg), edgeMap(mg, 2)
#   imgList = rescale(imboxfilt(imgList, 3))       and so every verb over a list of images
#   imgList = transformReferenceFrame(imgList, fr, 'byScreenAlignment')
#
# The grid is regular: an origin, the centre of pixel (0, 0), and two perpendicular steps,
# d1 from row to row and d2 from column to column, the positions derived and never stored.
# A distortion is applied by resampling onto a new regular grid, never by moving the grid.
# An image of k channels is (rows, cols, k); an integer image is scaled to [0, 1]. A list of
# images is a mapImageList, a Python list that displays as MATLAB's table.
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from ..geometry.arguments import refuseFlags
from ..geometry.frames import specimenFrame
from ..geometry.gridlayout import gridLayout, layoutIndex
from ..geometry.plottingconvention import plottingConvention
from ..geometry.vector3d import vector3d
from ..geometry.install import install
from ..geometry.display import aligned


def ownFrame():
  """the frame of an image nothing locates yet: axes iX, iY, iZ, which every display shows as
  the image's own, and a fresh instance each time, so that two images never share one"""
  return specimenFrame("image", axesNames=("iX", "iY", "iZ"), how2plot=plottingConvention.ij)


def _vec(v):
  return np.asarray(v.data if isinstance(v, vector3d) else v, dtype=float).reshape(3)


def _toDouble(img):
  a = np.asarray(img)
  if not (np.issubdtype(a.dtype, np.number) or a.dtype == bool):
    raise TypeError(f"an image is numeric, got {a.dtype}")
  if np.issubdtype(a.dtype, np.integer):
    return a.astype(float) / float(np.iinfo(a.dtype).max)
  return np.array(a, dtype=float)


class mapImage:
  """an image that knows where on the specimen it sits: a regular grid of an origin and two
  perpendicular steps, d1 from row to row and d2 from column to column; an integer image is
  scaled to [0, 1]

    mapImage(img)                          unit step, a frame of its own
    mapImage(img, dxy=0.05, name='bse')    square pixels; dxy=(dx, dy) rectangular ones
    mapImage(ebsd.bc, ebsd, name='bcImg')  a channel of a gridded square map, the map riding along"""

  def __init__(self, img=None, ebsd=None, dxy=1.0, origin=None, name="", scanUnit=None):
    self.img = np.zeros((0, 0))
    self.name = ""
    self.ebsd = None
    self._origin = np.zeros(3)
    self._d1, self._d2 = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    self.frame = None
    self.scanUnit = "um"
    if img is None:
      return
    if ebsd is not None:
      if not ebsd.isGridded:
        ebsd = ebsd.gridify()
      if ebsd.lattice is not None and ebsd.lattice.shape == "hex":
        raise ValueError("a mapImage is a regular grid and a hexagonal map is not one; resample it first, with "
                         "gridify(ebsd, unitCell=...) onto a square cell")
      # the map states its own layout: the row index advances along d1, the column along d2
      P = ebsd.pos.data
      self._origin = np.array(P[0, 0], dtype=float)
      d = ebsd.dPos
      from .gridify import arrayDirections
      u1, u2 = arrayDirections(ebsd)
      self._d1 = np.array(P[1, 0] - P[0, 0], dtype=float) if P.shape[0] > 1 else d * u1
      self._d2 = np.array(P[0, 1] - P[0, 0], dtype=float) if P.shape[1] > 1 else d * u2
      self.scanUnit = ebsd.scanUnit
      self.frame = ebsd.frame
      self.ebsd = ebsd
    else:
      self.frame = ownFrame()
      # laid out as gridify stores a map, rows along y
      dxy = np.atleast_1d(np.asarray(dxy, dtype=float))
      self._d1 = dxy[-1] * np.array([0.0, 1.0, 0.0])
      self._d2 = dxy[0] * np.array([1.0, 0.0, 0.0])
    self.img = _toDouble(img)
    if origin is not None:
      self._origin = _vec(origin)
    if scanUnit is not None:
      self.scanUnit = scanUnit
    if name:
      if not str(name).isidentifier():
        raise ValueError(f"the name becomes a property of the EBSD map, so it has to be an identifier; {name!r} is not")
      self.name = str(name)
    if self.img.ndim not in (2, 3):
      raise ValueError(f"an image is rows x cols x k, got {self.img.shape}")
    a, b = self._d1 / np.linalg.norm(self._d1), self._d2 / np.linalg.norm(self._d2)
    if abs(a @ b) >= 1e-6:
      raise ValueError(f"the row and column steps of a regular grid are perpendicular, but the ones given are "
                       f"{np.degrees(np.arccos(np.clip(a @ b, -1, 1))):.1f}° apart")
    if self.ebsd is not None and self.img.shape[:2] != self.ebsd.shape:
      raise ValueError(f"the image and the map have to be on the same grid, in the same array order; the image is "
                       f"{self.img.shape[:2]} and the map {self.ebsd.shape}")

  # ------------------------------------------------------------------ the geometry

  @property
  def origin(self):
    """the centre of pixel (0, 0)"""
    return vector3d(self._origin)

  @origin.setter
  def origin(self, v):
    """sets the centre of pixel (0, 0)"""
    self._origin = _vec(v)

  @property
  def d1(self):
    """the step from one row to the next"""
    return vector3d(self._d1)

  @d1.setter
  def d1(self, v):
    """sets the step from one row to the next"""
    self._d1 = _vec(v)

  @property
  def d2(self):
    """the step from one column to the next"""
    return vector3d(self._d2)

  @d2.setter
  def d2(self, v):
    """sets the step from one column to the next"""
    self._d2 = _vec(v)

  @property
  def pos(self):
    """the centre of every pixel, (rows, cols)"""
    return vector3d(self._posData())

  def _posData(self):
    r, c = self.gridSize()
    i, j = np.meshgrid(np.arange(r, dtype=float), np.arange(c, dtype=float), indexing="ij")
    return self._origin + i[..., None] * self._d1 + j[..., None] * self._d2

  @property
  def layout(self):
    """the gridLayout of the image, the directions its rows and columns advance along"""
    return gridLayout(self._d1, self._d2)

  @property
  def how2plot(self):
    """the plotting convention of the image's frame, None without a frame"""
    return None if self.frame is None else self.frame.how2plot

  @property
  def dx(self):
    """the column step, the length of d2"""
    return float(np.linalg.norm(self._d2))

  @property
  def dy(self):
    """the row step, the length of d1"""
    return float(np.linalg.norm(self._d1))

  @property
  def nChannel(self):
    """the number of channels, 1 for a two dimensional image"""
    return 1 if self.img.ndim == 2 else self.img.shape[2]

  @property
  def isempty(self):
    """whether the image has no pixel"""
    return self.img.size == 0

  def gridSize(self):
    """the number of rows and columns of the image"""
    return self.img.shape[:2]

  def copy(self):
    """a copy of the image with its own pixel array; the map riding along is shared"""
    out = mapImage.__new__(mapImage)
    out.__dict__.update(self.__dict__)
    out.img = self.img.copy()
    return out

  def __array__(self, dtype=None, copy=None):
    return self.img if dtype is None else self.img.astype(dtype)

  def extent(self):
    """[xmin, xmax, ymin, ymax, zmin, zmax] of the pixel centres"""
    r, c = self.gridSize()
    C = self._origin + np.array([0, 0, r - 1, r - 1])[:, None] * self._d1 + np.array([0, c - 1, 0, c - 1])[:, None] * self._d2
    return np.array([C[:, 0].min(), C[:, 0].max(), C[:, 1].min(), C[:, 1].max(), C[:, 2].min(), C[:, 2].max()])

  def gridCoordinates(self, pos):
    """where positions sit in grid coordinates, fractional rows and columns from zero; the steps
    are perpendicular, so each is one projection"""
    P = np.asarray(pos.data if isinstance(pos, vector3d) else pos, dtype=float)
    if P.shape[-1] == 2:
      P = np.concatenate([P, np.zeros(P.shape[:-1] + (1,))], axis=-1)
    R = P - self._origin
    return R @ self._d1 / (self._d1 @ self._d1), R @ self._d2 / (self._d2 @ self._d2)

  def pos2ind(self, pos, linear=False):
    """the pixel covering each position: row and column, not clipped, and whether it lies on the
    grid; linear=True the index into the flattened grid, -1 outside"""
    u, v = self.gridCoordinates(pos)
    i, j = np.floor(u + 0.5).astype(np.int64), np.floor(v + 0.5).astype(np.int64)
    r, c = self.gridSize()
    inside = (i >= 0) & (i < r) & (j >= 0) & (j < c)
    if linear:
      return np.where(inside, i * c + j, -1)
    return i, j, inside

  def interp(self, pos, method="linear", extrapolationMethod="none"):
    """the image values at positions: bilinear on the grid by default, 'nearest' or 'cubic';
    outside the grid NaN, or the nearest border value with extrapolationMethod='nearest'"""
    u, v = self.gridCoordinates(pos)
    shape = u.shape
    u, v = u.reshape(-1), v.reshape(-1)
    r, c = self.gridSize()
    img = self.img.reshape(r, c, -1)
    lost = ~(np.isfinite(u) & np.isfinite(v))
    outside = lost | (u < -1e-9) | (u > r - 1 + 1e-9) | (v < -1e-9) | (v > c - 1 + 1e-9)
    if extrapolationMethod == "nearest":
      outside = lost
    elif extrapolationMethod != "none":
      raise ValueError("extrapolationMethod is 'none' or 'nearest'")
    u, v = np.clip(np.where(lost, 0, u), 0, r - 1), np.clip(np.where(lost, 0, v), 0, c - 1)
    if method == "nearest":
      out = img[np.floor(u + 0.5).astype(np.int64), np.floor(v + 0.5).astype(np.int64)]
    elif method == "linear":
      i0, j0 = np.minimum(np.floor(u).astype(np.int64), max(r - 2, 0)), np.minimum(np.floor(v).astype(np.int64), max(c - 2, 0))
      i1, j1 = np.minimum(i0 + 1, r - 1), np.minimum(j0 + 1, c - 1)
      s, t = (u - i0)[:, None], (v - j0)[:, None]
      out = (1 - s) * ((1 - t) * img[i0, j0] + t * img[i0, j1]) + s * ((1 - t) * img[i1, j0] + t * img[i1, j1])
    elif method in ("cubic", "spline", "makima"):
      from scipy.interpolate import RegularGridInterpolator
      g = RegularGridInterpolator((np.arange(r), np.arange(c)), img, method="cubic" if method == "cubic" else "quintic" if method == "spline" else "pchip")
      out = g(np.stack([u, v], axis=1))
    else:
      raise ValueError("the interpolation is 'linear', 'nearest', 'cubic', 'spline' or 'makima'")
    out = np.array(out, dtype=float)
    out[outside] = np.nan
    return out.reshape(shape) if img.shape[2] == 1 else out.reshape(shape + (img.shape[2],))

  def subGrid(self, rows, cols=None):
    """a rectangle cut out, rows and cols contiguous and ascending, or the box round a mask; the
    map riding along is cut with it and the origin moves, nothing is resampled"""
    if cols is None:
      mask = np.asarray(rows, dtype=bool)
      rr, cc = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
      rows, cols = np.arange(rr[0], rr[-1] + 1), np.arange(cc[0], cc[-1] + 1)
    r, c = self.gridSize()
    rows = np.arange(r)[rows] if isinstance(rows, slice) else np.asarray(rows, dtype=np.int64).reshape(-1)
    cols = np.arange(c)[cols] if isinstance(cols, slice) else np.asarray(cols, dtype=np.int64).reshape(-1)
    if not (np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)) and np.array_equal(cols, np.arange(cols[0], cols[-1] + 1))):
      raise ValueError("a subGrid is a rectangle, so the indices have to be contiguous and ascending")
    if rows[0] < 0 or rows[-1] >= r or cols[0] < 0 or cols[-1] >= c:
      raise IndexError(f"asked for rows {rows[0]}:{rows[-1]} and columns {cols[0]}:{cols[-1]} of a {r} x {c} grid")
    out = self.copy()
    out._origin = self._origin + rows[0] * self._d1 + cols[0] * self._d2
    out.img = self.img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].copy()
    if self.ebsd is not None:
      out.ebsd = self.ebsd[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return out

  # ------------------------------------------------------------------ the display

  def char(self):
    """the image as one line of text: its size, channels, steps, name and whether a map rides along"""
    r, c = self.gridSize()
    s = f"{r} × {c}" + (f" × {self.nChannel}" if self.nChannel > 1 else "") + f"  step {self.dx:.4g} × {self.dy:.4g}"
    if self.name:
      s += f'  "{self.name}"'
    return s + ("  + map" if self.ebsd is not None else "")

  def __repr__(self):
    return aligned(repr(mapImageList([self])))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    from ..plotting.mapplot import plotMapImage as plot
    def rescale(self, lo=0.0, hi=1.0, perChannel=False) -> Any:
      """the values stretched to [lo, hi], all channels of an image together or each on its own; a
      non finite value is left out of the range and left alone, a flat image unchanged"""
    def imboxfilt(self, k=3) -> Any:
      """a box filter of k pixels, or (rows, cols), odd: the window shrinks at the border and around
      a missing value, so a NaN padding neither spreads nor darkens the frame (MATLAB's own,
      movmean down the columns and then along the rows)"""
    def edgeMap(self, padWidth=1) -> Any:
      """the edge transform: how unlike its eight neighbours at padWidth pixels each pixel is, every
      channel contrast normalised over its 2nd to 98th percentile, a neighbour off the image
      differenced against zero. Absolute brightness is not comparable between a band contrast map
      and a backscatter image, but their boundaries are, so registration correlates this"""
    def transformReferenceFrame(self, target, *args, byScreenAlignment=False) -> Any:
      """the images stated in a frame and laid out that way: a layout reindexes and nothing else, a
      frame restates where they sit by the orientation given, by 'byScreenAlignment', the
      assertion that the frames were drawn the same way up, or by nothing where the frames are
      already related. Only a transpose and flips are applied, never a resampling"""


# ------------------------------------------------------------------ a list of images

def _each(mg, f):
  if isinstance(mg, mapImage):
    return f(mg)
  return mapImageList(f(m) for m in mg)


class mapImageList(list):
  """a list of mapImage, displayed as MATLAB's table of a mapImage array"""

  def __getitem__(self, k):
    out = super().__getitem__(k)
    return mapImageList(out) if isinstance(k, slice) else out

  def __repr__(self):
    if not self:
      return aligned("mapImage\n  empty")
    rows = [(str(n), _sizeStr(m.img), _pixelStr(m), _frameStr(m.frame, m.how2plot), _frameStr(None if m.isempty or m.frame is None else m.layout, m.how2plot),
             m.name, _ebsdStr(m.ebsd)) for n, m in enumerate(self)]
    labels = ("", "image", "pixel", "frame", "layout", "name", "EBSD")
    w = [max(len(labels[j]), *(len(r[j]) for r in rows)) for j in range(len(labels))]
    def line(r):
      return "  " + "  ".join(x.ljust(n) for x, n in zip(r, w)).rstrip()
    pC = self[0].how2plot
    same = pC is not None and all(m.how2plot is not None and m.how2plot == pC for m in self)
    head = f"mapImage ({pC.char()})" if same else "mapImage"
    return aligned("\n".join([head, "", line(labels)] + [line(r) for r in rows]))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    from ..plotting.mapplot import plotMapImage as plot
    def rescale(self, lo=0.0, hi=1.0, perChannel=False) -> Any:
      """the values stretched to [lo, hi], all channels of an image together or each on its own; a
      non finite value is left out of the range and left alone, a flat image unchanged"""
    def imboxfilt(self, k=3) -> Any:
      """a box filter of k pixels, or (rows, cols), odd: the window shrinks at the border and around
      a missing value, so a NaN padding neither spreads nor darkens the frame (MATLAB's own,
      movmean down the columns and then along the rows)"""
    def transformReferenceFrame(self, target, *args, byScreenAlignment=False) -> Any:
      """the images stated in a frame and laid out that way: a layout reindexes and nothing else, a
      frame restates where they sit by the orientation given, by 'byScreenAlignment', the
      assertion that the frames were drawn the same way up, or by nothing where the frames are
      already related. Only a transpose and flips are applied, never a resampling"""


def _sizeStr(img):
  return " × ".join(str(n) for n in img.shape) if img.size else "-"


# one number for a square pixel, two for a rectangular one; the steps of a map differ in the
# last digits, so the test is relative
def _pixelStr(m):
  if m.isempty or m.dx == 0:
    return "-"
  u = "µm" if m.scanUnit == "um" else m.scanUnit
  if abs(m.dy - m.dx) <= 1e-3 * max(m.dx, m.dy):
    return f"{m.dx:.4g} {u}"
  return f"{m.dx:.4g} × {m.dy:.4g} {u}"


# a frame in its own axes names on the screen it is drawn on: 'y↓→x' for a specimen frame,
# 'row↓→col' for a layout
def _frameStr(fr, pC):
  if fr is None:
    return "-"
  if isinstance(fr, gridLayout):
    return _layoutChar(fr, pC)
  s = fr.conventionChar(pC) if pC is not None else ""
  return s if s else fr.char()


def _layoutChar(gL, pC):
  from ..geometry.plottingconvention import _arrowPictogram
  pC = plottingConvention.default if pC is None else pC
  north, east = _vec(pC.north), _vec(pC.east)
  dirs = [_vec(gL.row), _vec(gL.col)]
  names = ("row", "col")
  iN = next((k for k in range(2) if abs(dirs[k] @ north) > 1 - 1e-6), None)
  iE = next((k for k in range(2) if abs(dirs[k] @ east) > 1 - 1e-6), None)
  if iN is None or iE is None:
    return f"row {tuple(np.round(dirs[0], 3))}, col {tuple(np.round(dirs[1], 3))}"
  return _arrowPictogram(names[iN], dirs[iN] @ north > 0, names[iE], dirs[iE] @ east > 0, None)


def _ebsdStr(ebsd):
  if ebsd is None or ebsd.isempty:
    return "-"
  mins = [str(fr.mineral) for fr in ebsd.CSList.frames if fr.isIndexed]
  return ", ".join(mins) if mins else "EBSD"


# ------------------------------------------------------------------ the verbs

def rescale(mg, lo=0.0, hi=1.0, perChannel=False):
  """the values stretched to [lo, hi], all channels of an image together or each on its own; a
  non finite value is left out of the range and left alone, a flat image unchanged"""
  def stretch(v):
    f = v[np.isfinite(v)]
    if f.size == 0 or f.max() <= f.min():
      return v
    return lo + (v - f.min()) * (hi - lo) / (f.max() - f.min())

  def one(m):
    out = m.copy()
    if m.isempty:
      return out
    if perChannel and m.img.ndim == 3:
      out.img = np.stack([stretch(m.img[..., c]) for c in range(m.nChannel)], axis=2)
    else:
      out.img = stretch(m.img)
    return out
  return _each(mg, one)


# the mean over a sliding window of n along an axis that shrinks at the ends, the non finite
# values left out: MATLAB's movmean(v, n, dim, 'omitnan')
def _movmean(v, n, axis):
  v = np.moveaxis(v, axis, 0)
  ok = np.isfinite(v)
  z = np.where(ok, v, 0.0)
  cs = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(z, axis=0)])
  cn = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(ok, axis=0)])
  m = v.shape[0]
  k = np.arange(m)
  lo, hi = np.maximum(k - (n - 1) // 2, 0), np.minimum(k + n // 2 + 1, m)
  s, c = cs[hi] - cs[lo], cn[hi] - cn[lo]
  with np.errstate(invalid="ignore", divide="ignore"):
    out = np.where(c > 0, s / np.maximum(c, 1), np.nan)
  return np.moveaxis(out, 0, axis)


def imboxfilt(mg, k=3):
  """a box filter of k pixels, or (rows, cols), odd: the window shrinks at the border and around
  a missing value, so a NaN padding neither spreads nor darkens the frame (MATLAB's own,
  movmean down the columns and then along the rows)"""
  k = (int(k), int(k)) if np.isscalar(k) else tuple(int(x) for x in k)
  if any(x <= 0 or x % 2 == 0 for x in k):
    raise ValueError(f"the box size has to be odd and positive, got {k}")

  def one(m):
    out = m.copy()
    if not m.isempty:
      out.img = _movmean(_movmean(m.img, k[0], 0), k[1], 1)
    return out
  return _each(mg, one)


def edgeMap(mg, padWidth=1):
  """the edge transform: how unlike its eight neighbours at padWidth pixels each pixel is, every
  channel contrast normalised over its 2nd to 98th percentile, a neighbour off the image
  differenced against zero. Absolute brightness is not comparable between a band contrast map
  and a backscatter image, but their boundaries are, so registration correlates this"""
  img = np.array(mg.img, dtype=float)
  if img.ndim == 2:
    img = img[..., None]
  r, c, _ = img.shape
  for k in range(img.shape[2]):
    ch = img[..., k]
    f = ch[np.isfinite(ch)]
    if f.size:
      lo, hi = np.percentile(f, [2, 98], method="hazen")
      if hi > lo:
        img[..., k] = (ch - lo) / (hi - lo)
  e = np.zeros((r, c))
  p = int(padWidth)
  for dr, dc in ((-p, -p), (-p, 0), (-p, p), (0, -p), (0, p), (p, -p), (p, 0), (p, p)):
    s = np.zeros_like(img)
    r0, r1, c0, c1 = max(0, -dr), min(r, r - dr), max(0, -dc), min(c, c - dc)
    if r1 > r0 and c1 > c0:
      s[r0:r1, c0:c1] = img[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    e += np.sqrt(np.sum((s - img) ** 2, axis=2))
  return e


def transformReferenceFrame(mg, target, *args, byScreenAlignment=False):
  """the images stated in a frame and laid out that way: a layout reindexes and nothing else, a
  frame restates where they sit by the orientation given, by 'byScreenAlignment', the
  assertion that the frames were drawn the same way up, or by nothing where the frames are
  already related. Only a transpose and flips are applied, never a resampling"""
  from ..geometry.frametransform import byScreenAlignment as screenAlignment
  from ..geometry.quaternion import rotation
  if isinstance(target, mapImage):
    target = target.layout
  ori = next((a for a in args if isinstance(a, rotation)), None)
  refuseFlags(args, "transformReferenceFrame")
  byPlot = byScreenAlignment
  isLayout = isinstance(target, gridLayout)
  if isLayout and (ori is not None or byPlot):
    raise ValueError("a gridLayout only says which way round the array is stored; give a reference frame as the "
                     "target to state where the images sit")
  tgt = target if isLayout else gridLayout.assumedFor(target)

  def one(m):
    B = np.array([_vec(v) for v in m.layout.basis])
    R = np.eye(3)
    if ori is not None:
      R = ori.matrix().reshape(3, 3)
    elif byPlot:
      R = screenAlignment(m.frame if m.frame is not None else ownFrame(), target).matrix().reshape(3, 3)
    B = B @ R.T
    r, c = m.gridSize()
    lin, transposed = layoutIndex(tgt, (B[0], B[1]), (r, c), return_transpose=True)
    out = m.copy()
    out.img = m.img.reshape(r * c, -1)[lin].reshape(lin.shape + m.img.shape[2:])
    i, j = divmod(int(lin[0, 0]), c)
    first = m._origin + float(i) * m._d1 + float(j) * m._d2
    if m.ebsd is not None:
      from .gridify import gridify
      out.ebsd = gridify(m.ebsd, tgt)
    step = (m.dy, m.dx)[::-1] if transposed else (m.dy, m.dx)
    out._d1, out._d2 = step[0] * _vec(tgt.row), step[1] * _vec(tgt.col)
    out._origin = R @ first
    if not isLayout:
      out.frame = target
    return out
  return _each(mg, one)


for _name, _f in (("rescale", rescale), ("imboxfilt", imboxfilt), ("edgeMap", edgeMap), ("transformReferenceFrame", transformReferenceFrame)):
  install((mapImage,) if _name == "edgeMap" else (mapImage, mapImageList), _f, _name)

__all__ = ["mapImage", "mapImageList", "rescale", "imboxfilt", "edgeMap"]
