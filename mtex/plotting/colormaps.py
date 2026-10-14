# MTEX's colour maps as stops, and the contour levels (MATLAB plotting/colormaps, mtexColorMap)
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
  from ..geometry.vocabulary import ColorName
  from .vocabulary import ColormapName

# the stops (r, g, b) with their positions in [0, 1]
WHITE_JET_STOPS = ([(1.0, 1.0, 1.0), (0.0, 0.0, 0.7), (0.0, 0.85, 0.65), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
                   np.cumsum([0.0, 0.15, 0.4, 0.25, 0.13, 0.07]))
COLORMAP_STOPS = {
  "whitejet": WHITE_JET_STOPS,
  "labotex": ([(1.0, 1.0, 1.0), (1.0, 0.4, 0.4), (0.4, 0.0, 0.0)], [0.0, 0.5, 1.0]),
  # MATLAB's blue2redColorMap: dark blue, pure blue, white, pure red, dark red
  "blue2red": ([(0.0, 0.0, 0.53), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.51, 0.0, 0.0)], [0.0, 0.16, 0.5, 0.84, 1.0]),
  "red2blue": ([(0.51, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.53)], [0.0, 0.16, 0.5, 0.84, 1.0]),
  "white2black": ([(1.0, 1.0, 1.0), (0.7, 0.7, 0.7), (0.0, 0.0, 0.0)], [0.0, 0.5, 1.0]),
  "black2white": ([(0.0, 0.0, 0.0), (0.7, 0.7, 0.7), (1.0, 1.0, 1.0)], [0.0, 0.5, 1.0]),
  "parula": ([(0.2081, 0.1663, 0.5292), (0.1707, 0.2919, 0.7792), (0.0265, 0.4463, 0.8797), (0.0230, 0.5641, 0.7896), (0.1834, 0.6884, 0.6146),
              (0.4147, 0.7311, 0.4842), (0.7223, 0.7517, 0.2865), (0.9634, 0.7891, 0.1663), (0.9763, 0.9831, 0.0538)], np.linspace(0, 1, 9)),
}
# the number of colours of MATLAB's maps: WhiteJetColorMap 100, the others 128, parula 256
SIZES = {"whitejet": 100, "labotex": 128, "blue2red": 128, "red2blue": 128, "white2black": 128, "black2white": 128, "parula": 256}
NAMES = {"whitejet": "WhiteJet", "labotex": "LaboTeX", "blue2red": "blue2red", "red2blue": "red2blue", "white2black": "white2black", "black2white": "black2white",
         "parula": "parula"}


def colormapStops(name):
  """the colours and positions of the stops of MTEX's colour map of the name, None for another name"""
  return COLORMAP_STOPS.get(str(name).lower())


def colormapColors(name, n=None):
  """the map sampled at n colours, an n x 3 array; by default as many as MATLAB's map has"""
  stops = colormapStops(name)
  if stops is None:
    raise ValueError(f"unknown MTEX colour map {name}; one of {', '.join(NAMES.values())}")
  colors, pos = np.array(stops[0]), np.asarray(stops[1], dtype=float)
  t = np.linspace(0, 1, SIZES[str(name).lower()] if n is None else int(n))
  return np.stack([np.interp(t, pos, colors[:, k]) for k in range(3)], axis=1)


def mtexColormap(name=None):
  """the matplotlib colour map of a name: MTEX's own, registered on first use, or matplotlib's;
  a matrix of colours or a named map function as MATLAB passes them, and without a name the
  preference defaultColorMap"""
  import matplotlib
  from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
  if name is None:
    from ..geometry.session import getMTEXpref
    name = getMTEXpref("defaultColorMap", "WhiteJet")
  if callable(name) and not isinstance(name, Colormap):
    name = name()
  if isinstance(name, Colormap):
    return name
  if not isinstance(name, str):
    return ListedColormap(np.asarray(name, dtype=float))
  stops = colormapStops(name)
  if stops is None:
    return matplotlib.colormaps[name]
  key = NAMES[str(name).lower()]
  # as many colours as MATLAB's map has
  if key not in matplotlib.colormaps:
    C = colormapColors(key)
    matplotlib.colormaps.register(LinearSegmentedColormap.from_list(key, C, N=len(C)), name=key)
  return matplotlib.colormaps[key]


def mtexColorMap(name: ColormapName | Any = "WhiteJet", fig=None):
  """the colour map of the current figure, or of one figure or axes; MATLAB's mtexColorMap

    mtexColorMap('LaboTeX')                          # every axes of the current figure
    mtexColorMap('white2black', gcm())               # every axes of one MTEX figure
    mtexColorMap(ax, 'hot')                          # one axes only
    mtexColorMap(np.array([[1, 1, 1], [0, 0, 1]]))   # an n x 3 matrix of rgb values
    mtexColorMap(blue2redColorMap)                   # a named map function, as MATLAB passes it

  Without a figure the map is given to every axes of the current MTEX figure, so that all the
  sections of one function share the colour scale; to give the subplots of a figure different
  maps, pass each axes as the first argument. A name is one of MTEX's maps (WhiteJet, LaboTeX,
  blue2red, red2blue, white2black, black2white, parula) or one of matplotlib's. In a plain
  matplotlib figure the images and filled contours of the current figure take the map.

  Input
    name - the name of a colour map, a matplotlib Colormap, an n x 3 matrix of rgb values or a
        function returning one; an axes, with the map following as the second argument
  Output
    cmap - the matplotlib Colormap set
  Keywords
    fig - the mtexFigure to colour instead of the current one
  See also
    mtexColorbar, setColorRange, num2rgb"""
  from .mtexfigure import gcm
  # mtexColorMap(ax, name): the map of one axes of the current figure
  if hasattr(name, "get_figure"):
    ax, name = name, fig
    cmap = mtexColormap(name)
    if gcm() is not None:
      gcm().setColormap(cmap, ax=ax)
    return cmap
  import matplotlib.pyplot as plt
  cmap = mtexColormap(name)
  target = gcm() if fig is None else fig
  # an MTEX figure that pyplot no longer has current leaves the map to the current one
  if fig is None and target is not None and getattr(target.fig, "number", None) is not None and target.fig is not plt.gcf():
    target = None
  if target is not None:
    target.setColormap(cmap)
    return cmap
  # a plain matplotlib figure: its images and filled contours take the map, as MATLAB's
  # colormap; contour lines keep their colour
  if plt.get_fignums():
    for ax in plt.gcf().axes:
      for c in ax.collections + ax.images:
        if c.get_array() is not None and getattr(c, "filled", True):
          c.set_cmap(cmap)
  return cmap


def str2rgb(name: ColorName | Any):
  """str2rgb('salmon'): the RGB triplet of a colour name, MATLAB tools/str2rgb"""
  from matplotlib.colors import to_rgb
  return np.array(to_rgb(name), dtype=float)


# the colour range of values: the one given, else the extent of their finite values, (0, 1)
# for none; a range of no width is widened a little, so that it still divides
def _colorLimits(values, colorRange=None):
  if colorRange is None:
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
  else:
    lo, hi = colorRange
  if not hi > lo:
    hi = lo + max(abs(lo), 1.0) * 1e-6
  return lo, hi


def contourLevels(lo, hi, n=10, log=False):
  """n levels between the extremes, logarithmic on request: the levels of a contour plot

    levels = contourLevels(0, 4)                 # ten levels from 0 to 4
    levels = contourLevels(1, 100, 5, log=True)  # five levels equally spaced on a log scale

  A range of no width is widened a little, so that the levels still differ; a logarithmic
  spacing needs a positive lower end and is linear otherwise.

  Input
    lo, hi - the lowest and the highest level
  Output
    levels - an array of n levels
  Keywords
    n - the number of levels
    log - equal ratios instead of equal differences between the levels
  See also
    contour, contourf"""
  lo, hi = float(lo), float(hi)
  if not hi > lo:
    hi = lo + max(abs(lo), 1.0) * 1e-6
  if log and lo > 0:
    return np.logspace(np.log10(lo), np.log10(hi), n)
  return np.linspace(lo, hi, n)


def num2rgb(values, colormap: ColormapName | Any = "WhiteJet", range=None, N=255):
  """the colours a colour map gives those numbers, MATLAB's num2rgb

    rgb = num2rgb(values)                         # WhiteJet over the extent of the values
    rgb = num2rgb(values, 'hot', range=[0, 6])    # another map, a fixed range
    rgb = num2rgb(values, 'LaboTeX', N=16)        # one of 16 colours of a named map
    rgb = num2rgb(values, blue2redColorMap(8))    # one of eight colours of a matrix

  The values are scaled to the range and clipped to it, and take one of the N colours a named
  map is sampled at, or of the rows of a matrix of colours: the value at the fraction t of the
  range the colour ceil(t N), counted from one, the lowest value the first.

  Input
    values - an array of numbers
  Output
    rgb - n x 3 rgb triplets, one row per value
  Keywords
    colormap - the name of a colour map, a matplotlib map or an n x 3 matrix of colours
    range - [lower, upper] limit of the values, by default their minimum and maximum
    N - the number of colours a named map is sampled at; a matrix has its rows
  See also
    ind2color, mtexColorMap"""
  v = np.asarray(values, dtype=float).reshape(-1)
  lo, hi = (np.nanmin(v), np.nanmax(v)) if range is None else (float(range[0]), float(range[1]))
  t = np.zeros_like(v) if not hi > lo else np.clip((v - lo) / (hi - lo), 0, 1)
  # the map is a name, a matplotlib map, or the matrix of colours one of the named maps returns
  if isinstance(colormap, np.ndarray):
    C = np.asarray(colormap, dtype=float).reshape(-1, colormap.shape[-1])[:, :3]
  elif colormapStops(colormap) is not None:
    C = colormapColors(colormap, N)
  else:
    C = mtexColormap(colormap)(np.linspace(0, 1, int(N)))[:, :3]
  k = np.clip(np.ceil(np.nan_to_num(t) * len(C)).astype(int) - 1, 0, len(C) - 1)
  return C[k]


# the named maps as MATLAB names them: blue2redColorMap, LaboTeXColorMap, ...
def _namedMap(key):
  def f(n=SIZES[key]):
    return colormapColors(key, n)
  f.__name__ = NAMES[key] + "ColorMap"
  f.__doc__ = (f"the {NAMES[key]} colour map, as MATLAB's {f.__name__} returns it\n\n"
               f"  rgb = {f.__name__}()     # {SIZES[key]} colours, as MATLAB's\n"
               f"  rgb = {f.__name__}(16)   # 16 colours\n\n"
               f"The colours are interpolated linearly between the stops of the map; pass the result to\n"
               f"mtexColorMap or as a plot's colormap.\n\n"
               f"Output\n  rgb - an n x 3 array of rgb values in [0, 1]\n"
               f"Keywords\n  n - the number of colours\n"
               f"See also\n  mtexColorMap, num2rgb")
  return f


blue2redColorMap = _namedMap("blue2red")
red2blueColorMap = _namedMap("red2blue")
LaboTeXColorMap = _namedMap("labotex")
white2blackColorMap = _namedMap("white2black")
black2whiteColorMap = _namedMap("black2white")
WhiteJetColorMap = _namedMap("whitejet")

# vega20(n): n colours of the twenty of vega's categorical map, repeated beyond twenty
VEGA20 = np.array([(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120), (44, 160, 44), (152, 223, 138),
                   (214, 39, 40), (255, 152, 150), (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                   (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199), (188, 189, 34), (219, 219, 141),
                   (23, 190, 207), (158, 218, 229)]) / 255


def vega20(n=None):
  """n colours of vega's twenty categorical ones, repeated beyond twenty, an n x 3 array;
  MATLAB's vega20

    map = vega20()     # as many as the current colour map has
    map = vega20(5)    # the first five

  The colours of matplotlib 2.0 for categorical data, matplotlib's tab20, in MATLAB's order:
  the ten dark shades first, then the ten light ones.

  Output
    map - an n x 3 array of rgb values in [0, 1]
  Keywords
    n - the number of colours; by default that of the colour map of the current MTEX figure,
        else of the preference defaultColorMap
  See also
    ind2color"""
  if n is None:
    from .mtexfigure import gcm
    fig = gcm()
    n = mtexColormap(getattr(fig, "colormap", None)).N
  return VEGA20[np.r_[0:20:2, 1:20:2]][np.arange(int(n)) % len(VEGA20)]


# the colour order MTEX sets as its preference, plotting/plotting_tools/colors.mat: the
# seven of MATLAB's default order, then the lighter half of vega20 and six darker ones
MTEX_COLORS = np.array([
  (0.0, 0.447, 0.741), (0.85, 0.325, 0.098), (0.929, 0.694, 0.125), (0.466, 0.674, 0.188),
  (0.494, 0.184, 0.556), (0.301, 0.745, 0.933), (0.635, 0.078, 0.184), (0.54902, 0.337255, 0.294118),
  (0.890196, 0.466667, 0.760784), (0.498039, 0.498039, 0.498039), (0.737255, 0.741176, 0.133333),
  (0.682353, 0.780392, 0.909804), (0.596078, 0.87451, 0.541176), (1.0, 0.596078, 0.588235),
  (0.772549, 0.690196, 0.835294), (0.768627, 0.611765, 0.580392), (0.780392, 0.780392, 0.780392),
  (0.858824, 0.858824, 0.552941), (0.619608, 0.854902, 0.898039), (0.619048, 0.464567, 1.0),
  (0.0, 0.362205, 0.095238), (0.253968, 0.11811, 0.714286), (0.634921, 0.480315, 0.0),
  (0.0, 0.669291, 0.571429)])


# ind2color(i): the colour of index i of MTEX's colour order, counted from 0 where MATLAB counts
# from 1, one RGB row per index of an array, NaN where the index is negative or NaN; 'ordered'
# takes MATLAB's order in which neighbouring ids get related colours, a matrix of colours an
# order of its own
ORDERED = np.array([1, 6, 19, 12, 20, 22, 3, 11, 18, 16, 23, 8, 7, 2, 14, 15, 9, 5, 21, 4, 24, 13, 10, 17]) - 1


def ind2color(i, cmap=None):
  """the colour of the index i in MTEX's colour order, counting from 0 (MATLAB counts from 1); an
  RGB row per index of an array, NaN where the index is negative or NaN; MATLAB's ind2color

    rgb = ind2color(i)                  # one colour of the order of MTEX's preference
    rgb = ind2color(ids, 'ordered')     # the order in which neighbouring ids get related colours
    rgb = ind2color(ids, cmap)          # a matrix of colours as the order

  The order has 24 colours and repeats beyond them, so that the phases or grains numbered by
  the indices are told apart.

  Input
    i - an index, or an array of them
  Output
    rgb - an rgb triplet, or one row per index
  Keywords
    cmap - 'ordered' for MATLAB's order of related neighbours, or a matrix of colours
  See also
    vega20, num2rgb"""
  if cmap is None:
    colors = MTEX_COLORS
  elif isinstance(cmap, str) and cmap == "ordered":
    colors = MTEX_COLORS[ORDERED]
  else:
    colors = np.asarray(cmap, dtype=float)
  if np.ndim(i) == 0:
    return colors[int(i) % len(colors)].copy()
  ind = np.asarray(i, dtype=float).reshape(-1)
  out = np.full((len(ind), 3), np.nan)
  ok = np.isfinite(ind) & (ind >= 0)
  out[ok] = colors[ind[ok].astype(int) % len(colors)]
  return out


__all__ = ["mtexColorMap", "contourLevels", "str2rgb", "ind2color", "num2rgb", "blue2redColorMap", "red2blueColorMap", "LaboTeXColorMap",
           "white2blackColorMap", "black2whiteColorMap", "WhiteJetColorMap", "vega20"]
