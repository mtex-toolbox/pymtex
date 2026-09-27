# two scalar properties as one colour (MATLAB plotting/planarColorKeys/planarColorKey)
#
#   pK = planarColorKey('twilight'), planarColorKey(rgbTable), planarColorKey(cmap, shading='black')
#   pK.periode = pi, pK.range1, pK.range2, pK.quantileCap, pK.label1, pK.label2
#   rgb = pK.property2color(v1, v2)
#   plot(pK), plot(pK, v1, v2)
#
# The first property picks the hue from a colour table, wrapped over a period when the
# property is periodic, and the second property the saturation, from white or black at the
# low end of its range to the full colour at the high end.
from __future__ import annotations

import numpy as np

from ..geometry.session import getMTEXpref
from ..geometry.units import degree
from .colormaps import mtexColormap
from .mtexfigure import gcm, mtexFigure
from ..geometry.display import aligned


class planarColorKey:
  """a colour key of two properties: hue by the first, saturation by the second"""

  def __init__(self, colorMap="viridis", satRange=(0, 1), shading="white", periode=np.nan, quantileCap=0.975, range1=(np.nan, np.nan),
               range2=(np.nan, np.nan), label1="Property 1", label2="Property 2"):
    if isinstance(colorMap, np.ndarray) and colorMap.ndim == 2 and colorMap.shape[1] == 3:
      self.colorMap = np.asarray(colorMap, dtype=float)
    else:
      self.colorMap = np.asarray(mtexColormap(colorMap)(np.linspace(0, 1, 256)))[:, :3]
    self.satRange = tuple(float(s) for s in satRange)
    self.shading = shading
    self.periode = periode
    self.quantileCap = quantileCap
    self.range1 = tuple(float(r) for r in range1)
    self.range2 = tuple(float(r) for r in range2)
    self.label1, self.label2 = label1, label2

  # the bounds of both properties: those given, else those of the data, the upper bound of
  # the second capped at a quantile
  def _ranges(self, v1, v2):
    mn1, mx1 = self.range1
    mn2, mx2 = self.range2
    if v1 is not None and len(v1):
      mn1 = np.nanmin(v1) if np.isnan(mn1) else mn1
      mx1 = np.nanmax(v1) if np.isnan(mx1) else mx1
    else:
      mn1, mx1 = (0.0 if np.isnan(mn1) else mn1), (1.0 if np.isnan(mx1) else mx1)
    if v2 is not None and len(v2):
      mn2 = np.nanmin(v2) if np.isnan(mn2) else mn2
      if np.isnan(mx2):
        mx2 = np.nanquantile(v2, self.quantileCap) if self.quantileCap != 1 else np.nanmax(v2)
    else:
      mn2, mx2 = (0.0 if np.isnan(mn2) else mn2), (1.0 if np.isnan(mx2) else mx2)
    return mn1, mx1, mn2, mx2

  def property2color(self, v1, v2):
    """rgb = pK.property2color(v1, v2): the colour of every pair, one row per element"""
    v1 = np.asarray(v1, dtype=float).reshape(-1)
    v2 = np.asarray(v2, dtype=float).reshape(-1)
    if v1.size != v2.size:
      raise ValueError("the two properties need one value each per element")
    mn1, mx1, mn2, mx2 = self._ranges(v1, v2)
    if np.isfinite(self.periode):
      # a periodic property, given in degrees when it exceeds a full turn
      w = v1 * degree if np.nanmax(v1) > 2 * np.pi else v1
      t1 = np.mod(w, self.periode) / self.periode
    else:
      t1 = np.clip((v1 - mn1) / (mx1 - mn1) if mx1 > mn1 else np.zeros_like(v1), 0, 1)
    t2 = np.clip((np.clip(v2, mn2, mx2) - mn2) / (mx2 - mn2) if mx2 > mn2 else np.zeros_like(v2), 0, 1)
    # the hue by linear interpolation in the table
    f = np.nan_to_num(t1) * (len(self.colorMap) - 1)
    lo = np.floor(f).astype(int)
    hi = np.minimum(lo + 1, len(self.colorMap) - 1)
    frac = (f - lo)[:, None]
    hue = (1 - frac) * self.colorMap[lo] + frac * self.colorMap[hi]
    sat = (self.satRange[0] + np.nan_to_num(t2) * (self.satRange[1] - self.satRange[0]))[:, None]
    rgb = sat * hue + (1 - sat) if self.shading == "white" else sat * hue
    return np.clip(rgb, 0, 1)

  def plot(self, v1=None, v2=None, nHue=256, nSat=128, **kw):
    """draws the key as an image, hue along x and saturation along y; with the two properties v1, v2
    the axes carry their ranges and labels

      pK.plot()
      pK.plot(v1, v2)"""
    return plotPlanarKey(self, v1, v2, nHue, nSat, **kw)

  def __repr__(self):
    lines = ["planarColorKey", f"  colours: {len(self.colorMap)}, shading: {self.shading}"]
    if np.isfinite(self.periode):
      lines.append(f"  periode: {self.periode:g}")
    lines.append(f"  {self.label1}: {self.range1}, {self.label2}: {self.range2}")
    return aligned("\n".join(lines))


def plotPlanarKey(pK, v1=None, v2=None, nHue=256, nSat=128, parent=None, **kw):
  """the key as an image, hue along x and saturation along y; with the data the axes carry
  their ranges and labels"""
  fig = gcm()
  if parent is not None:
    ax = parent
  elif fig is not None and fig.pending is not None:
    ax = fig.takeAxis()
  elif fig is not None and not fig.children and not fig.hold:
    ax = fig.nextAxis()
  else:
    ax = mtexFigure(1, figSize=(4.5, 3.2)).nextAxis()
  X, Y = np.meshgrid(np.linspace(0, 1, nHue), np.linspace(0, 1, nSat))
  flat = planarColorKey(pK.colorMap, satRange=pK.satRange, shading=pK.shading)
  img = flat.property2color(X.reshape(-1), Y.reshape(-1)).reshape(nSat, nHue, 3)
  h = ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), aspect="auto", interpolation="nearest")
  ax.set_box_aspect(0.5)
  size = getMTEXpref("FontSize", 13)
  if v1 is None or v2 is None:
    ax.set_xticks([])
    ax.set_yticks([])
    return h
  v1 = np.asarray(v1, dtype=float).reshape(-1)
  v2 = np.asarray(v2, dtype=float).reshape(-1)
  mn1, mx1, mn2, mx2 = pK._ranges(v1, v2)
  if np.isfinite(pK.periode):
    inDegrees = np.nanmax(v1) > 2 * np.pi
    top = pK.periode / degree if inDegrees else pK.periode
    ticks = np.linspace(0, top, 7 if inDegrees else 5)
    labels = [f"{t:g}" for t in ticks] if inDegrees else ["0", "π/4", "π/2", "3π/4", "π"] if np.isclose(pK.periode, np.pi) else \
      ["0", "π/2", "π", "3π/2", "2π"]
    ax.set_xticks(ticks / top)
    ax.set_xticklabels(labels)
  else:
    ticks = _niceTicks(mn1, mx1)
    ax.set_xticks((ticks - mn1) / (mx1 - mn1))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
  ticks = _niceTicks(mn2, mx2)
  ax.set_yticks((ticks - mn2) / (mx2 - mn2))
  ax.set_yticklabels([f"{t:g}" for t in ticks])
  ax.set_xlabel(pK.label1, fontsize=size)
  ax.set_ylabel(pK.label2, fontsize=size)
  return h


# three to six round ticks inside a range
def _niceTicks(mn, mx):
  if not mx > mn:
    return np.array([mn])
  mag = 10 ** np.floor(np.log10(mx - mn))
  best = np.array([mn, mx])
  for step in np.array([20, 10, 5, 3, 2, 1, 0.5, 0.2, 0.1]) * mag:
    t = np.arange(np.ceil(mn / step) * step, np.floor(mx / step) * step + step / 2, step)
    t = t[(t >= mn - 1e-9 * step) & (t <= mx + 1e-9 * step)]
    if 3 <= len(t) <= 6:
      best = t
  return best


__all__ = ["planarColorKey"]
