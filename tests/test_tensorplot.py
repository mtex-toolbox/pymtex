# the drawing of tensor quantities (docs/plans/plotting.md S1), on the Agg backend
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import QuadMesh  # noqa: E402

from mtex import birefringence, crystalFrame, degree, gcm, plot, plotSeismicVelocities, spectralTransmission, velocity
from mtex.tensors.tensor import refractiveIndexTensor, stiffnessTensor
from mtex.plotting.tensorplot import _extrema, isRGB

CS = crystalFrame("mmm", [4.7646, 10.2296, 5.9942], mineral="olivine")
M = np.array([[320.5, 68.15, 71.6, 0, 0, 0], [68.15, 196.5, 76.8, 0, 0, 0], [71.6, 76.8, 233.5, 0, 0, 0],
              [0, 0, 0, 64, 0, 0], [0, 0, 0, 0, 77, 0], [0, 0, 0, 0, 0, 78.7]])


@pytest.fixture(autouse=True)
def closeFigures():
  yield
  plt.close("all")


def test_the_seismic_overview():
  C = stiffnessTensor(M, CS, density=3.355)
  fig = plotSeismicVelocities(C)
  assert fig is gcm() and len(fig.children) == 6 and fig.nrows == 2 and fig.ncols == 3
  assert [ax.get_title() for ax in fig.children] == ["Vp (km/s)", "S-wave anisotropy (%)", "Vs1 (km/s)", "Vs2 (km/s)", "Vp/Vs1", "Vp/Vs2"]
  labels = [t.get_text() for ax in fig.children for t in ax.texts if "Anisotropy" in t.get_text()]
  assert len(labels) == 6 and labels[0].startswith("Vp Anisotropy = ")
  # the extrema of Vp of olivine: fastest along a, slowest along b, about 24 percent apart
  vp = velocity(C).vp
  (hi, pHi), (lo, pLo) = _extrema(vp)
  assert np.isclose(hi, np.sqrt(320.5 / 3.355), atol=0.02) and np.isclose(lo, np.sqrt(196.5 / 3.355), atol=0.02)
  assert abs(pHi.uvw[0]) > 0.99 * np.linalg.norm(pHi.uvw) and abs(pLo.uvw[1]) > 0.99 * np.linalg.norm(pLo.uvw)
  assert float(labels[0].split("= ")[1]) == pytest.approx(200 * (hi - lo) / (hi + lo), abs=0.1)
  # every chart has its markers and the polarisation panels their segments, six colour bars
  assert all(len(ax.collections) >= 3 for ax in fig.children) and len(fig.colorbars) == 6
  # the method is the function, on the lower hemisphere with the extrema symmetrised
  C.plotSeismicVelocities(lower=True, symmetriseMinMax=True, MarkerSize=6)
  assert len(gcm().children) == 6


def test_the_rgb_plot_of_interference_colours():
  rI = refractiveIndexTensor(np.diag([1.635, 1.651, 1.670]), CS)
  rgb = spectralTransmission(rI, 30000, tau=45 * degree)
  assert isRGB(rgb) and not isRGB(rgb[:2]) and not isRGB((1, 2, 3))
  h = plot(rgb, rgb=True)
  assert isinstance(h, list) and all(isinstance(m, QuadMesh) for m in h) and len(gcm().children) == 1
  fc = h[0].get_facecolors()
  assert fc.shape[1] == 4 and np.all(fc >= 0) and np.all(fc <= 1) and len(np.unique(np.round(fc[:, :3], 2), axis=0)) > 20
  # the flag is implied by three functions, and complete shows the whole hemisphere
  h1 = plot(rgb)
  n1 = sum(len(m.get_facecolors()) for m in h1)
  h2 = plot(rgb, upper=True, complete=True)
  n2 = sum(len(m.get_facecolors()) for m in h2)
  assert n2 > 2 * n1
  # the birefringence and its axes plot as any function and field
  dn, fast, slow = birefringence(rI)
  plot(dn, upper=True, complete=True)
  plot(fast)
  assert len(gcm().children) == 1
