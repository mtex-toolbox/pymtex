# the one spelling of each style keyword the plots take, MATLAB's graphics properties in camelCase:
# lineWidth, markerSize, displayName. MATLAB's and matplotlib's spellings of the same property
# (LineWidth, linewidth, MarkerSize, markersize) are refused with the one to use, so that a keyword
# is never quietly dropped or handed to matplotlib under another name
from typing import TYPE_CHECKING

STYLE_KEYS = ("color", "lineColor", "lineWidth", "lineStyle", "marker", "markerSize", "markerFaceColor", "markerEdgeColor",
              "markerAlpha", "markerFaceAlpha", "markerEdgeAlpha", "displayName", "fontSize", "fontWeight", "faceColor", "faceAlpha", "edgeColor", "edgeAlpha",
              "backgroundColor", "textColor", "showText")

# every other spelling of a style key, lower cased, and the key it spells
_SPELLINGS = {k.lower(): k for k in STYLE_KEYS} | {"markercolor": "markerFaceColor", "linecolour": "lineColor",
                                                   "lw": "lineWidth", "ls": "lineStyle", "ms": "markerSize"}

# the properties of a line in matplotlib's spelling
_LINE = {"lineWidth": "linewidth", "lineStyle": "linestyle", "lineColor": "color", "displayName": "label",
         "markerSize": "markersize", "markerFaceColor": "markerfacecolor", "markerEdgeColor": "markeredgecolor"}


# MATLAB's options of the plots this port spells otherwise or does not take, with what to write
# instead; refused before they could reach matplotlib as a property it does not know
MATLAB_FLAGS = {
  "noBoundary": "write boundary=False",
  "points": "write backend='scatter'",
  "exact": "write backend='patch' or a larger maxPixels",
  "dynamicMarkerSize": "a figure here is drawn once; markerSize sets the size",
  "autoArrowSize": "the arrows take their length from the grid by default; arrowSize scales it",
  "refFrameDirs": "the scale bar draws the axes of the specimen frame, refFrame=False leaves them out",
  "refFrameLabels": "the scale bar draws the axes of the specimen frame, refFrame=False leaves them out",
  "scale": "scale sizes the crystal shapes of plot(grains, cS) and the arrows of quiver",
}


def refuseVariants(kw, flags=()):
  """raises TypeError where a style keyword of kw is spelled otherwise than MTEX spells it, or
  is one of MATLAB's options of the plots this port does not take (MATLAB_FLAGS, the names of
  flags as well where the caller does not read them)"""
  for k in kw:
    c = _SPELLINGS.get(k.lower())
    if c is not None and k != c:
      raise TypeError(f"the keyword {k!r} is spelled {c!r}")
    if k in MATLAB_FLAGS and (k not in _READ_ELSEWHERE or k in flags):
      raise TypeError(f"MATLAB's option {k!r} is not taken here: {MATLAB_FLAGS[k]}")
  return kw


# the flags of MATLAB_FLAGS some other plot reads under the same name (the points of a fibre or
# a region, the scale of crystal shapes and arrows), refused only where the caller names them
_READ_ELSEWHERE = {"points", "scale"}


def lineKeywords(kw):
  """the keywords of a line for matplotlib, their MTEX spelling checked first"""
  refuseVariants(kw)
  return {_LINE.get(k, k): v for k, v in kw.items()}


# the properties of a patch or a collection in matplotlib's spelling
_PATCH = {"lineWidth": "linewidth", "lineStyle": "linestyle", "lineColor": "edgecolor", "edgeColor": "edgecolor",
          "faceColor": "facecolor", "displayName": "label"}


def patchKeywords(kw):
  """the keywords of a patch or a collection for matplotlib, their MTEX spelling checked first"""
  refuseVariants(kw)
  return {_PATCH.get(k, k): v for k, v in kw.items()}


# the keywords of the plots as an editor completes them, with the values of the named ones; any
# other keyword stays allowed, the plots reading more than these
if TYPE_CHECKING:
  from typing import Any, Literal

  from typing_extensions import TypedDict

  from ..geometry.vocabulary import ColorName, ConventionName
  from .vocabulary import ColormapName, FigureSizeName, MarkerName, ProjectionName

  class PlotKeywords(TypedDict, total=False, extra_items=Any):
    # the style
    color: ColorName | Any
    lineColor: ColorName | Any
    lineWidth: float | Any
    lineStyle: Literal["-", "--", ":", "-.", "none"] | Any
    marker: MarkerName | Any
    markerSize: float | Any
    markerFaceColor: ColorName | Any
    markerEdgeColor: ColorName | Any
    markerAlpha: float | Any
    markerFaceAlpha: float | Any
    markerEdgeAlpha: float | Any
    displayName: str | Any
    fontSize: float | Any
    fontWeight: Literal["normal", "bold"] | Any
    faceColor: ColorName | Any
    faceAlpha: float | Any
    edgeColor: ColorName | Any
    edgeAlpha: float | Any
    backgroundColor: ColorName | Any
    textColor: ColorName | Any
    showText: bool | Any
    # the colouring of data
    colormap: ColormapName | Any
    colorRange: Literal["equal", "tight"] | Any
    contours: int | Any
    colorbar: bool | Any
    # the chart
    projection: ProjectionName | Any
    upper: bool | Any
    lower: bool | Any
    complete: bool | Any
    antipodal: bool | Any
    how2plot: ConventionName | Any
    fundamentalRegion: bool | Any
    grid: bool | Any
    grid_res: float | Any
    axisLabels: bool | Any
    resolution: float | Any
    # where it goes
    parent: Any
    add2all: bool | Any
    figSize: FigureSizeName | Any
    layout: Any
    marginx: float | Any
    marginy: float | Any
