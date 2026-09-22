# the profile of a kernel on SO(3) and the spectra of kernels and harmonic functions
# (MATLAB @SO3Kernel/plot, @SO3Kernel/plotSpektra, @S2Kernel/plotSpektra, @SO3Fun/plotSpektra)
#
#   plot(psi)                          psi(omega) over -180 to 180 degree
#   plotSpektra(psi, bandwidth=32)     the Chebyshev coefficients A_l / (2l + 1) per degree
#   plotSpektra(odf, bandwidth=32)     the harmonic power per degree of an SO3Fun
from __future__ import annotations

import numpy as np

from ..geometry.units import degree
from .mtexfigure import gcm, mtexFigure


# the axes a line plot goes into: the current one when held, else a new figure
def _lineAxes(kw):
  from .mtexfigure import figureSize
  figSize = kw.pop("figSize", None)
  fig = gcm()
  import matplotlib.pyplot as plt
  # a plain figure the user opened after the last MTEX figure takes the plot
  if plt.get_fignums() and (fig is None or plt.gcf() is not fig.fig):
    return plt.gca()
  # a held figure takes the line on its last axes, a fresh one on its first
  if fig is not None and (fig.hold or not fig.children):
    return fig.children[-1] if fig.children else fig.nextAxis()
  return mtexFigure(figSize=figureSize(figSize, (6.0, 4.0))).nextAxis()


def _lineKw(kw):
  kw.pop("silent", None)
  for old, new in (("LineWidth", "linewidth"), ("lineWidth", "linewidth"), ("LineStyle", "linestyle"), ("lineStyle", "linestyle"),
                   ("Color", "color"), ("lineColor", "color"), ("DisplayName", "label"), ("Marker", "marker"), ("MarkerSize", "markersize")):
    if old in kw:
      kw[new] = kw.pop(old)
  return kw


def plotSO3Kernel(psi, **kw):
  """plot(psi): the value of a kernel on SO(3) against the rotation angle, -180 to 180 degree"""
  ax = _lineAxes(kw)
  kw = _lineKw(kw)
  kw.setdefault("linewidth", 2)
  omega = np.linspace(-np.pi, np.pi, 1000)
  h = ax.plot(omega / degree, np.real(psi.eval(np.cos(omega / 2))), **kw)
  ax.set_xlim(-180, 180)
  ax.set_xlabel("angle in degree")
  return h


def plotSpektra(x, bandwidth=None, logarithmic=False, **kw):
  """plotSpektra(psi), plotSpektra(odf): the Chebyshev coefficients of a kernel, or the harmonic
  power of a function, per degree"""
  from ..functions.kernelfunctions import S2Kernel, SO3Kernel
  ax = _lineAxes(kw)
  kw = _lineKw(kw)
  if isinstance(x, (SO3Kernel, S2Kernel)):
    A = np.asarray(x.A, dtype=float).reshape(-1)
    L = min(32 if bandwidth is None else int(bandwidth), len(A) - 1)
    n = np.arange(L + 1)
    y = A[:L + 1] / (2 * n + 1)
    kw.setdefault("marker", "o")
    kw.setdefault("markersize", 5)
    if logarithmic:
      h = ax.loglog(n, np.abs(y), **kw)
    else:
      h = ax.semilogx(n, y, **kw)
    ax.set_xlabel("harmonic degree")
    ax.set_ylabel("Chebyshev coefficient")
    return h
  from ..functions.so3fun import SO3FunHarmonic
  f = x if isinstance(x, SO3FunHarmonic) else SO3FunHarmonic(x)
  L = f.bandwidth if bandwidth is None else int(bandwidth)
  power = np.zeros(L + 1)
  LL = min(L, f.bandwidth)
  power[:LL + 1] = f.power[:LL + 1]
  kw.setdefault("marker", "o")
  kw.setdefault("linestyle", ":")
  h = ax.plot(np.arange(L + 1), power, **kw)
  ax.set_xlim(0, L)
  ax.set_xlabel("harmonic degree")
  ax.set_ylabel("power")
  return h


__all__ = ["plotSO3Kernel", "plotSpektra"]
