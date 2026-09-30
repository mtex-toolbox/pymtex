# the one spelling of each style keyword the plots take, MATLAB's graphics properties in camelCase:
# lineWidth, markerSize, displayName. MATLAB's and matplotlib's spellings of the same property
# (LineWidth, linewidth, MarkerSize, markersize) are refused with the one to use, so that a keyword
# is never quietly dropped or handed to matplotlib under another name
STYLE_KEYS = ("color", "lineColor", "lineWidth", "lineStyle", "marker", "markerSize", "markerFaceColor", "markerEdgeColor",
              "markerFaceAlpha", "markerEdgeAlpha", "displayName", "fontSize", "faceColor", "faceAlpha", "edgeColor",
              "edgeAlpha", "backgroundColor", "textColor")

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
