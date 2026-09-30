# the kernel profile and the spectra: the lines, their axes and the figure they go into
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mtex import SO3DeLaValleePoussinKernel, SO3DirichletKernel, degree, hold, plot, plotSpectrum, uniformODF, unimodalODF, crystalFrame, orientation


def test_plot_ofAKernelIsItsProfileOverTheFullCircle():
  plt.close("all")
  psi = SO3DeLaValleePoussinKernel(halfwidth=20 * degree)
  h = plot(psi)
  x, y = h[0].get_data()
  assert x[0] == -180 and x[-1] == 180 and abs(np.max(y) - psi.eval(1.0)) < 0.1


def test_plot_ofHeldKernelsGoesIntoTheUsersFigure():
  plt.close("all")
  plt.figure()
  hold(True)
  for psi in (SO3DeLaValleePoussinKernel(halfwidth=20 * degree), SO3DirichletKernel(3)):
    plot(psi, displayName="k")
  hold(False)
  assert len(plt.get_fignums()) == 1 and len(plt.gca().lines) == 2 and plt.gca().lines[0].get_label() == "k"


def test_plotSpektra_ofAKernelDrawsItsCoefficients():
  plt.close("all")
  psi = SO3DeLaValleePoussinKernel(halfwidth=20 * degree)
  h = plotSpectrum(psi, bandwidth=8)
  x, y = h[0].get_data()
  assert len(x) == 9 and abs(y[0] - psi.A[0]) < 1e-12


def test_plotSpektra_ofAnODFDrawsItsEnergyOverTheHarmonicDegree():
  plt.close("all")
  cs = crystalFrame("432")
  odf = 0.5 * uniformODF(cs) + 0.5 * unimodalODF(orientation.byEuler(0.1, 0.2, 0.3, cs))
  h = plotSpectrum(odf, bandwidth=12)
  x, y = h[0].get_data()
  assert len(x) == 13 and y[0] > 0 and h[0].axes.get_xlabel() == "harmonic degree"
