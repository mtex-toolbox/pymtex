# MTEX's colour maps as stops, and the contour levels (MATLAB plotting/colormaps, mtexColorMap)
from __future__ import annotations

import numpy as np

# the stops (r, g, b) with their positions in [0, 1]
WHITE_JET_STOPS = ([(1.0, 1.0, 1.0), (0.0, 0.0, 0.7), (0.0, 0.85, 0.65), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
                   np.cumsum([0.0, 0.15, 0.4, 0.25, 0.13, 0.07]))
COLORMAP_STOPS = {
  "whitejet": WHITE_JET_STOPS,
  "labotex": ([(1.0, 1.0, 1.0), (1.0, 0.4, 0.4), (0.4, 0.0, 0.0)], [0.0, 0.5, 1.0]),
  "blue2red": ([(0.0, 0.0, 0.5), (1.0, 1.0, 1.0), (0.5, 0.0, 0.0)], [0.0, 0.5, 1.0]),
  "red2blue": ([(0.5, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.5)], [0.0, 0.5, 1.0]),
  "white2black": ([(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)], [0.0, 1.0]),
  "black2white": ([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], [0.0, 1.0]),
  "parula": ([(0.2081, 0.1663, 0.5292), (0.1707, 0.2919, 0.7792), (0.0265, 0.4463, 0.8797), (0.0230, 0.5641, 0.7896), (0.1834, 0.6884, 0.6146),
              (0.4147, 0.7311, 0.4842), (0.7223, 0.7517, 0.2865), (0.9634, 0.7891, 0.1663), (0.9763, 0.9831, 0.0538)], np.linspace(0, 1, 9)),
}
NAMES = {"whitejet": "WhiteJet", "labotex": "LaboTeX", "blue2red": "blue2red", "red2blue": "red2blue", "white2black": "white2black", "black2white": "black2white",
         "parula": "parula"}


def colormapStops(name):
  return COLORMAP_STOPS.get(str(name).lower())


# the map sampled at n colours, an n x 3 array
def colormapColors(name, n=256):
  stops = colormapStops(name)
  if stops is None:
    raise ValueError(f"unknown MTEX colour map {name}; one of {', '.join(NAMES.values())}")
  colors, pos = np.array(stops[0]), np.asarray(stops[1], dtype=float)
  t = np.linspace(0, 1, n)
  return np.stack([np.interp(t, pos, colors[:, k]) for k in range(3)], axis=1)


# the matplotlib colour map of a name: MTEX's own, registered on first use, or matplotlib's
def mtexColormap(name="WhiteJet"):
  import matplotlib
  from matplotlib.colors import Colormap, LinearSegmentedColormap
  if isinstance(name, Colormap):
    return name
  stops = colormapStops(name)
  if stops is None:
    return matplotlib.colormaps[name]
  key = NAMES[str(name).lower()]
  if key not in matplotlib.colormaps:
    matplotlib.colormaps.register(LinearSegmentedColormap.from_list(key, colormapColors(key)), name=key)
  return matplotlib.colormaps[key]


# mtexColorMap WhiteJet: the colour map of the current figure, or of one figure
def mtexColorMap(name="WhiteJet", fig=None):
  from .mtexfigure import gcm
  cmap = mtexColormap(name)
  target = gcm() if fig is None else fig
  if target is not None:
    target.setColormap(cmap)
  return cmap


# str2rgb('salmon'): the RGB triplet of a colour name, MATLAB tools/str2rgb
def str2rgb(name):
  from matplotlib.colors import to_rgb
  return np.array(to_rgb(name), dtype=float)


# n levels between the extremes, logarithmic on request
def contourLevels(lo, hi, n=10, log=False):
  lo, hi = float(lo), float(hi)
  if not hi > lo:
    hi = lo + max(abs(lo), 1.0) * 1e-6
  if log and lo > 0:
    return np.logspace(np.log10(lo), np.log10(hi), n)
  return np.linspace(lo, hi, n)


# num2rgb(values, map, range=[lo, hi]): the colours a colour map gives those numbers,
# MATLAB's num2rgb; the range defaults to the extent of the values
def num2rgb(values, colormap="WhiteJet", range=None):
  v = np.asarray(values, dtype=float).reshape(-1)
  lo, hi = (np.nanmin(v), np.nanmax(v)) if range is None else (float(range[0]), float(range[1]))
  t = np.zeros_like(v) if not hi > lo else np.clip((v - lo) / (hi - lo), 0, 1)
  # the map is a name, a matplotlib map, or the matrix of colours one of the named maps returns
  if isinstance(colormap, np.ndarray):
    C = np.asarray(colormap, dtype=float).reshape(-1, colormap.shape[-1])[:, :3]
    k = np.clip((t * (len(C) - 1)).round().astype(int), 0, len(C) - 1)
    return C[k]
  return mtexColormap(colormap)(t)[:, :3]


# the named maps as MATLAB names them: blue2redColorMap, LaboTeXColorMap, ...
def _namedMap(key):
  def f(n=256):
    return colormapColors(key, n)
  f.__name__ = NAMES[key] + "ColorMap"
  f.__doc__ = f"the {NAMES[key]} colour map, as MATLAB's {f.__name__} returns it"
  return f


blue2redColorMap = _namedMap("blue2red")
red2blueColorMap = _namedMap("red2blue")
LaboTeXColorMap = _namedMap("labotex")
white2blackColorMap = _namedMap("white2black")
black2whiteColorMap = _namedMap("black2white")
WhiteJetColorMap = _namedMap("whitejet")

__all__ = ["mtexColorMap", "contourLevels", "str2rgb", "ind2color", "num2rgb", "blue2redColorMap", "red2blueColorMap", "LaboTeXColorMap",
           "white2blackColorMap", "black2whiteColorMap", "WhiteJetColorMap"]


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


# ind2color(i): the i-th colour of MTEX's colour order, MATLAB's counting from one
def ind2color(i):
  return MTEX_COLORS[(int(i) - 1) % len(MTEX_COLORS)].copy()
