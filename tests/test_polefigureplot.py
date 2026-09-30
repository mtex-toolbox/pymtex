# the pole figures of orientation lists and of measured pole figures (docs/plans/plotting.md S6),
# on the Agg backend
import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mtex import (Miller, PoleFigure, calcODF, crystalFrame, degree, discreteSample, gcm, loadPoleFigure, orientation, plot,
                  plotDiff, plotIPF, plotPF, rotation, specimenFrame, unimodalODF, vector3d, xvector, zvector)
from mtex.plotting.sphericalplot import plotText

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "polefigure")
QUARTZ = crystalFrame("321", [4.916, 4.916, 5.4054], mineral="Quartz")


@pytest.fixture(autouse=True)
def closeFigures():
  yield
  plt.close("all")


def plotsOf():
  return [ax._mtexSphericalPlot for ax in gcm().children if hasattr(ax, "_mtexSphericalPlot")]


@pytest.fixture(scope="module")
def sample():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  odf = unimodalODF(orientation.byEuler(30 * degree, 50 * degree, 10 * degree, cs, ss), halfwidth=15 * degree)
  return cs, ss, odf, discreteSample(odf, 500, rng=np.random.default_rng(1))


# the measured quartz pole figure and the ODF computed from it
@pytest.fixture(scope="module")
def quartz():
  pf = loadPoleFigure(os.path.join(FIXTURES, "Q(10-10)_amp.cnv"), QUARTZ)
  return pf, calcODF(pf)


# the budget of markers of MATLAB: n orientations times the m elements of the specimen group
def budget(ori, SS):
  return min(500, 10000 // (ori.CS.numSym() * ori.SS.numSym())), SS.numSym()


def test_plotPF_onePanelPerDirectionWithOneMarkerPerOrbit(sample):
  cs, ss, odf, ori = sample
  hs = Miller([1, 1], [0, 1], [0, 1], cs)
  h = plotPF(ori, hs)
  fig = gcm()
  # one panel per crystal direction on the specimen sector, titled, one marker per orbit
  assert len(fig.children) == 2 and [ax.get_title() for ax in fig.children] == ["{100}", "{111}"]
  assert all(len(sP.sR.N) == 3 for sP in plotsOf())
  # MATLAB's sample budget: 10000 markers over the elements of both groups
  n, m = budget(ori, ss)
  assert len(h[0].get_offsets()) == 3 * m * n and len(h[1].get_offsets()) == 4 * m * n


def test_plotPF_valuePerOrientationColoursEveryEquivalentInEitherOrder(sample):
  cs, ss, odf, ori = sample
  hs = Miller([1, 1], [0, 1], [0, 1], cs)
  n, m = budget(ori, ss)
  # a value per orientation colours every equivalent, in either order of the arguments
  h = plotPF(ori, np.degrees(ori.angle()), hs, colorbar=True)
  assert len(gcm().colorbars) == 1 and h[0].get_array().shape == (3 * m * n,)
  h2 = plotPF(ori, hs, np.degrees(ori.angle()))
  assert np.allclose(h2[0].get_array(), h[0].get_array())


def test_plotPF_budgetThinsTheListBeforeTheEquivalents(sample):
  cs, ss, odf, ori = sample
  m = ss.numSym()
  # the budget thins a long list before its equivalents are formed
  h = plotPF(ori, Miller(1, 2, 3, cs), points=100)
  assert len(h[0].get_offsets()) == 24 * m * 100
  h = plotPF(ori, Miller(1, 2, 3, cs))
  assert len(h[0].get_offsets()) == 24 * m * (10000 // (ori.CS.numSym() * ori.SS.numSym()))


def test_plotPF_contourfDrawsTheDensityOfThePoles(sample):
  cs, ss, odf, ori = sample
  # the density of the poles instead of the markers
  h = plotPF(ori, Miller([1, 1], [0, 1], [0, 1], cs), contourf=True)
  assert all(isinstance(x, matplotlib.contour.QuadContourSet) for x in h)


def test_plotPF_completeDrawsTheWholeSphere(sample):
  cs, ss, odf, ori = sample
  plotPF(ori, Miller(1, 0, 0, cs), complete=True, title=False)
  assert len(plotsOf()[0].sR.N) == 1 and gcm().children[0].get_title() == ""


def test_plotPF_ofODFDrawsItsDensity(sample):
  cs, ss, odf, ori = sample
  hs = Miller([1, 1], [0, 1], [0, 1], cs)
  # the method, and an ODF still draws its density
  ori.plotPF(hs, markerSize=3)
  plotPF(odf, hs)
  assert all(isinstance(x, matplotlib.collections.QuadMesh) for x in [c for ax in gcm().children for c in ax.collections])


def test_plotPF_refusesValuesWithoutDirections(sample):
  cs, ss, odf, ori = sample
  with pytest.raises(ValueError):
    plotPF(ori, np.zeros(500))


def test_plotIPF_onePanelPerSpecimenDirection(sample):
  cs, ss, odf, ori = sample
  h = plotIPF(ori, [xvector, zvector])
  fig = gcm()
  assert [ax.get_title() for ax in fig.children] == ["X", "Z"] and all(len(sP.sR.N) >= 3 for sP in plotsOf())
  # z is fixed by the specimen two-fold, x is not
  assert len(h[1].get_offsets()) == 500 and len(h[0].get_offsets()) == 500


def test_plotIPF_valuesTakeAColourbar(sample):
  cs, ss, odf, ori = sample
  plotIPF(ori, np.degrees(ori.angle()), [xvector, zvector], colorbar=True)
  assert len(gcm().colorbars) == 1
  ori.plotIPF(zvector, pcolor=True)
  plotIPF(odf, zvector)


def test_plot_measuredPoleFigureMarkersCarryTheIntensities(quartz):
  pf, _ = quartz
  h = plot(pf)
  fig = gcm()
  assert len(fig.children) == 1 and fig.children[0].get_title() == plotText(pf.allH[0][0].char()) == "{10$\\bar{1}$0}"
  # the markers carry the intensities, and their size follows how many there are
  assert len(h[0].get_offsets()) == pf.allI[0].size and np.isclose(h[0].get_array().max(), pf.allI[0].max())
  assert h[0].get_sizes()[0] < 100


def test_plot_measuredPoleFiguresEachOnItsOwnRange(quartz):
  pf, _ = quartz
  # a data set in a row, each panel on its own range with its own bar
  pf2 = PoleFigure([pf.allH[0], pf.allH[0]], [pf.allR[0], pf.allR[0]], [pf.allI[0], 2 * pf.allI[0]], QUARTZ)
  h = plot(pf2, colorbar=True)
  assert len(gcm().children) == 2 and h[0].get_clim() != h[1].get_clim() and len(gcm().colorbars) == 2
  h = plot(pf2, colorRange=None)
  assert h[0].get_clim() == h[1].get_clim()


def test_plot_measuredPoleFigureReadAsAFunction(quartz):
  pf, _ = quartz
  # the pole figure read as a function
  h = pf.plot(smooth=True, minmax=True)
  assert isinstance(h[0], matplotlib.collections.QuadMesh)
  h = plot(pf, contourf=True, bandwidth=12)
  assert isinstance(h[0], matplotlib.contour.QuadContourSet)


def test_plotDiff_RPErrorDropsSmallRecalculations(quartz):
  pf, odf = quartz
  h = plotDiff(pf, odf, RP=True)
  assert len(gcm().children) == 1 and len(gcm().colorbars) == 1 and len(h[0].get_offsets()) <= pf.allI[0].size
  # the RP error drops the directions where the recalculation is small, the l1 error keeps them
  h2 = pf.plotDiff(odf, l1=True, colorbar=False)
  assert len(gcm().colorbars) == 0 and len(h2[0].get_offsets()) == pf.allI[0].size >= len(h[0].get_offsets())


def test_plotDiff_defaultIsEveryDirectionInEitherOrder(quartz):
  pf, odf = quartz
  # MATLAB's default is the regularised relative error at every direction, also as plotDiff(odf, pf)
  h3 = plotDiff(odf, pf, colorbar=False)
  assert len(h3[0].get_offsets()) == pf.allI[0].size


def test_plots_into_axes_named_by_parent(quartz):
  pf, odf = quartz
  plt.figure()
  axes = [plt.subplot(1, 3, k) for k in (1, 2, 3)]
  plot(pf, parent=axes[0])
  plotPF(odf, pf.allH[0], antipodal=True, parent=axes[1])
  plotDiff(odf, pf, parent=axes[2])
  # one figure, each panel in the axes it was given, a bar only beside the difference
  assert plt.get_fignums() == [plt.gcf().number] and all(hasattr(ax, "_mtexSphericalPlot") for ax in axes)
  assert len(gcm().colorbars) == 1


def test_the_equal_range_of_measured_pole_figures(quartz):
  pf, _ = quartz
  h = plot(pf, colorRange='equal')
  assert h[0].get_clim() == (float(np.min(pf.allI[0])), float(np.max(pf.allI[0])))
