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


def refuseVariants(kw):
  """raises TypeError where a style keyword of kw is spelled otherwise than MTEX spells it"""
  for k in kw:
    c = _SPELLINGS.get(k.lower())
    if c is not None and k != c:
      raise TypeError(f"the keyword {k!r} is spelled {c!r}")
  return kw


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
