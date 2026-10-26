# the colours of the import wizard, one table for a light desktop and one for a dark one; the
# window follows the desktop's scheme, and every panel reads its colours here when it draws
#
#   colours()            the table in force
#   applyScheme(app)     the scheme of the desktop, or the one asked for, taken by the application
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette

LIGHT = {
  "dark": False,
  # the palette
  "window": "#efefef", "base": "#ffffff", "alternateBase": "#f5f5f5", "text": "#000000", "highlight": "#3874c8",
  "highlightedText": "#ffffff", "toolTip": "#ffffdc", "placeholder": "#808080",
  # the frames of the panels, their grounds, the text said in passing
  "frame": "#c8c8c8", "panel": "#ffffff", "muted": "#777777", "crumb": "#888888",
  # the status band: its ground, the bar on its left and its text, by the kind of the status
  "statusGround": {"idle": "#dbe9f6", "loading": "#fbe5b6", "loaded": "#d8efd3", "failed": "#f4c7c3"},
  "statusMark": {"idle": "#1f4e79", "loading": "#b26a00", "loaded": "#2e7d32", "failed": "#b71c1c"},
  "statusInk": "#1a1a1a",
  # a leaf of the metadata that is a picture, the cells of the phase table the lattice forces
  "pictureInk": "#e65100", "forced": "#e4e4e4",
  # the script: its comments and strings, its buttons disabled and done
  "comment": "#7a7a7a", "string": "#2e7d32", "disabledFill": "#c8c8c8", "disabledInk": "#f4f4f4", "done": "#2e7d32",
  # the tabs: the ink of each kind of view, tinted into the ground
  "tabInk": {"phase": "#1f4e79", "ipf": "#2e7d32", "property": "#333333", "polefigures": "#6a1b9a", "images": "#e65100"},
  # the pictograms: the map's axes, the Euler frame's, the ground chosen, under the pointer and
  # at rest, the frame under the pointer
  "mapInk": "#333333", "eulerInk": "#c62828", "checked": "#e8eef6", "hover": "#f2f2f2", "ground": "#ffffff",
  "hoverFrame": "#8a8a8a",
  # the drawings of matplotlib: the ground, the text and the lines
  "figure": "#ffffff", "figureInk": "#000000",
}

DARK = {
  "dark": True,
  "window": "#2b2b2b", "base": "#1e1e1e", "alternateBase": "#262626", "text": "#e6e6e6", "highlight": "#3d7bd0",
  "highlightedText": "#ffffff", "toolTip": "#3a3a3a", "placeholder": "#8a8a8a",
  "frame": "#4a4a4a", "panel": "#1e1e1e", "muted": "#9a9a9a", "crumb": "#9a9a9a",
  "statusGround": {"idle": "#1c3247", "loading": "#4a3810", "loaded": "#1d3b20", "failed": "#4f1f1f"},
  "statusMark": {"idle": "#64a0d8", "loading": "#f0a030", "loaded": "#66bb6a", "failed": "#ef5350"},
  "statusInk": "#f0f0f0",
  "pictureInk": "#ff9e5e", "forced": "#383838",
  "comment": "#9a9a9a", "string": "#81c784", "disabledFill": "#454545", "disabledInk": "#8a8a8a", "done": "#2e7d32",
  "tabInk": {"phase": "#7fb2e5", "ipf": "#81c784", "property": "#d0d0d0", "polefigures": "#ce93d8", "images": "#ff9e5e"},
  "mapInk": "#e0e0e0", "eulerInk": "#ef5350", "checked": "#2a3a4d", "hover": "#333333", "ground": "#1e1e1e",
  "hoverFrame": "#8a8a8a",
  "figure": "#2b2b2b", "figureInk": "#e6e6e6",
}

_current = LIGHT


def colours():
  """the colours the wizard is drawn in, the light or the dark table"""
  return _current


def desktopIsDark(app):
  """whether the desktop is dark: its colour scheme, or where it states none, its window colour"""
  scheme = app.styleHints().colorScheme()
  if scheme != Qt.ColorScheme.Unknown:
    return scheme == Qt.ColorScheme.Dark
  return app.palette().color(QPalette.Window).lightness() < 128


def applyScheme(app, dark=None):
  """the application drawn in the wizard's light or dark colours, the desktop's scheme unless dark
  is given: the Fusion style, which draws every control in the palette, and the palette of the table"""
  global _current
  _current = DARK if (desktopIsDark(app) if dark is None else dark) else LIGHT
  app.setStyle("Fusion")
  app.setPalette(palette())


def palette():
  """the Qt palette of the table in force"""
  c = _current
  p = QPalette(QColor(c["window"]), QColor(c["window"]))
  for group in (QPalette.Active, QPalette.Inactive):
    for role, key in ((QPalette.Base, "base"), (QPalette.AlternateBase, "alternateBase"), (QPalette.Text, "text"),
                      (QPalette.WindowText, "text"), (QPalette.ButtonText, "text"), (QPalette.ToolTipBase, "toolTip"),
                      (QPalette.ToolTipText, "text"), (QPalette.PlaceholderText, "placeholder"),
                      (QPalette.Highlight, "highlight"), (QPalette.HighlightedText, "highlightedText")):
      p.setColor(group, role, QColor(c[key]))
  return p


def figureRc():
  """matplotlib's settings of the drawings in the table in force: the ground, the text, the axes"""
  c = _current
  return {"figure.facecolor": c["figure"], "axes.facecolor": c["figure"], "savefig.facecolor": c["figure"],
          "text.color": c["figureInk"], "axes.edgecolor": c["figureInk"], "axes.labelcolor": c["figureInk"],
          "xtick.color": c["figureInk"], "ytick.color": c["figureInk"]}


__all__ = ["colours", "applyScheme", "desktopIsDark", "palette", "figureRc", "LIGHT", "DARK"]
