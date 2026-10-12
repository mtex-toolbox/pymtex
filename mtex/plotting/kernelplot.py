# the profile of a kernel on SO(3) and the spectra of kernels and harmonic functions
# (MATLAB @SO3Kernel/plot, @SO3Kernel/plotSpektra, @S2Kernel/plotSpektra, @SO3Fun/plotSpektra)
#
#   plot(psi)                          psi(omega) over -180 to 180 degree
#   plotSpectrum(psi, bandwidth=32)     the Chebyshev coefficients A_l / (2l + 1) per degree
#   plotSpectrum(odf, bandwidth=32)     the harmonic power per degree of an SO3Fun
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..geometry.units import degree
from .keywords import lineKeywords, refuseVariants
from .mtexfigure import gcm, mtexFigure

if TYPE_CHECKING:
  from typing import Unpack

  from .keywords import PlotKeywords


# the axes a line plot goes into: the current one when held, else a new figure
def _lineAxes(kw):
  from .mtexfigure import figureSize
  figSize = kw.pop("figSize", None)
  fig = gcm()
  import matplotlib.pyplot as plt
  # a plain figure the user opened after the last MTEX figure takes the plot, and so does
  # an axes the user added to the MTEX figure, as MATLAB's subplot
  if plt.get_fignums() and (fig is None or plt.gcf() is not fig.fig):
    return plt.gca()
  if fig is not None and fig.fig.axes and fig.fig.gca() not in fig.children:
    ax = fig.fig.gca()
    fig.children.append(ax)
    fig.current = ax
    return ax
  # a held figure takes the line on its current axes, a fresh one on its first
  if fig is not None and (fig.hold or not fig.children):
    if fig.children:
      return fig.current if fig.current in fig.children else fig.children[-1]
    return fig.nextAxis()
  return mtexFigure(figSize=figureSize(figSize, (6.0, 4.0))).nextAxis()


def plotSO3Kernel(psi, **kw: Unpack[PlotKeywords]):
  """plot(psi): the value of a kernel on SO(3) against the rotation angle, -180 to 180 degree"""
  ax = _lineAxes(kw)
  kw = lineKeywords(kw)
  kw.setdefault("linewidth", 2)
  omega = np.linspace(-np.pi, np.pi, 1000)
  h = ax.plot(omega / degree, np.real(psi.eval(np.cos(omega / 2))), **kw)
  ax.set_xlim(-180, 180)
  ax.set_xlabel("angle in degree")
  return h


def plotSpectrum(x, bandwidth=None, logarithmic=False, **kw: Unpack[PlotKeywords]):
  """plotSpectrum(psi), plotSpectrum(odf): the Chebyshev coefficients of a kernel, or the harmonic
  power of a function, per degree (MATLAB's plotSpektra)"""
  # a list of functions, MATLAB's [f, g]: one line each in the same axes
  if isinstance(x, (list, tuple)):
    from .mtexfigure import hold
    hs = [plotSpectrum(x[0], bandwidth, logarithmic, **kw)]
    hold(True)
    for f in x[1:]:
      hs.append(plotSpectrum(f, bandwidth, logarithmic, **{k: v for k, v in kw.items() if k != "figSize"}))
    hold(False)
    return hs
  refuseVariants(kw)
  fontSize = kw.pop("fontSize", None)
  ax = _lineAxes(kw)
  kw = lineKeywords(kw)
  h = _spektra(ax, x, bandwidth, logarithmic, kw)
  # the font size of the labels and the ticks
  if fontSize is not None:
    ax.tick_params(labelsize=fontSize)
    ax.xaxis.label.set_size(fontSize)
    ax.yaxis.label.set_size(fontSize)
  return h


def _spektra(ax, x, bandwidth, logarithmic, kw):
  from ..functions.kernelfunctions import S2Kernel, SO3Kernel
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
  from ..functions.s2fun import S2FunHarmonic
  from ..functions.so3fun import SO3FunHarmonic
  # a function on the sphere: the norm of the coefficients of each degree
  if isinstance(x, (SO3FunHarmonic, S2FunHarmonic)):
    f = x
  else:
    f = S2FunHarmonic(x) if hasattr(x, "_harmonicS2") else SO3FunHarmonic(x)
  L = f.bandwidth if bandwidth is None else int(bandwidth)
  # a list of functions draws one line for each
  P = np.asarray(f.power).reshape(-1, f.bandwidth + 1)
  power = np.zeros((len(P), L + 1))
  LL = min(L, f.bandwidth)
  power[:, :LL + 1] = P[:, :LL + 1]
  # open circles on a dotted line, as MATLAB draws the power
  kw.setdefault("marker", "o")
  kw.setdefault("markerfacecolor", "none")
  kw.setdefault("linestyle", ":")
  h = ax.plot(np.arange(L + 1), power.T, **kw)
  ax.set_xlim(0, L)
  ax.set_xlabel("harmonic degree")
  ax.set_ylabel("power")
  return h


__all__ = ["plotSO3Kernel", "plotSpectrum"]
