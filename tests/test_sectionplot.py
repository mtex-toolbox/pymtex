# the drawing of ODF sections (docs/plans/plotting.md S2), on the Agg backend
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mtex import (vector3d, axisAngleSections, crystalFrame, degree, gcm, orientation, plotSection, sigmaSections, specimenFrame,
                  unimodalODF)
from mtex.plotting.sectionplot import _gridColumns

rng = np.random.default_rng(3)


@pytest.fixture(autouse=True)
def closeFigures():
  yield
  plt.close("all")


def plotsOf(fig):
  return [ax._mtexSphericalPlot for ax in fig.children if hasattr(ax, "_mtexSphericalPlot")]


def test_the_grid_of_cells():
  assert _gridColumns(1) == 1 and _gridColumns(3) == 3 and _gridColumns(6) == 3 and _gridColumns(9) == 3
  assert _gridColumns(7) == 3 and _gridColumns(18) == 6
  # up to three wide Euler sections, 360 by 90 degrees, stack in one column, as MATLAB's
  assert _gridColumns(3, aspect=0.25) == 1 and _gridColumns(2, aspect=0.25) == 1


def test_sections_of_an_odf():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  odf = unimodalODF(orientation.byEuler(30 * degree, 50 * degree, 10 * degree, cs, ss), halfwidth=15 * degree)
  # the default against a specimen: phi2 sections, plain charts labelled phi1 and Phi
  h = plotSection(odf, resolution=10 * degree)
  fig = gcm()
  sPs = plotsOf(fig)
  assert len(sPs) == 6 and all(sP.proj.isPlain for sP in sPs) and len(h) == 6
  assert fig.children[0].get_title() == "φ2 = 0°" and fig.children[0].get_xlabel() == "φ1"
  assert all(isinstance(x, matplotlib.collections.QuadMesh) for x in h)
  # one colour range over the sections
  assert len({x.get_clim() for x in h}) == 1 and fig.colorRange == h[0].get_clim()
  # a vector field on SO(3) as arrows, into Euler and into sigma sections
  from mtex import SO3VectorFieldHarmonic, hold, plot
  F = SO3VectorFieldHarmonic(lambda o: vector3d(np.ones((o.size, 3))), cs, ss, bandwidth=2)
  for kind in ("phi2", "sigma"):
    plotSection(odf, kind, sections=2, resolution=10 * degree)
    hold(True)
    hs = plot(F, color="k")
    hold(False)
    assert len(hs) == 2 and all(h is not None for h in hs)
  # a style flag after the angles, as MATLAB's plotSection(odf, 'phi2', angles, 'contourf')
  h = plotSection(odf, 'phi2', np.array([0, 45]) * degree, 'contourf', resolution=10 * degree)
  assert len(h) == 2 and all(isinstance(x, matplotlib.contour.QuadContourSet) for x in h)
  # noGrid leaves out the reference field of the sigma sections, as MATLAB's
  plotSection(odf, "sigma", sections=2, resolution=10 * degree, noGrid=True)
  assert all(len(ax.collections) == 1 for ax in gcm().children if hasattr(ax, "_mtexSphericalPlot"))
  # sigma sections by name, with the reference field, as filled contours with a bar
  h = plotSection(odf, "sigma", sections=4, resolution=10 * degree, contourf=True, colorbar=True)
  fig = gcm()
  assert len(plotsOf(fig)) == 4 and fig.children[1].get_title() == "σ = 22°" or fig.children[1].get_title() == "σ = 23°"
  assert all(isinstance(x, matplotlib.contour.QuadContourSet) for x in h) and len(fig.colorbars) == 1
  assert any(isinstance(a, matplotlib.quiver.Quiver) for a in fig.children[0].get_children())
  # explicit values and a section object, the method, and no field
  odf.plotSection(sigmaSections(cs, ss, sigma=[0, 30 * degree]), resolution=10 * degree, field=False)
  fig = gcm()
  assert len(plotsOf(fig)) == 2 and not any(isinstance(a, matplotlib.quiver.Quiver) for a in fig.children[0].get_children())
  plotSection(odf, "phi2", phi2=[15 * degree, 45 * degree], resolution=10 * degree, columns=1, minmax=True)
  fig = gcm()
  assert len(fig.children) == 2 and fig.ncols == 1 and sum("Max:" in t.get_text() for t in fig.children[0].texts) == 1
  # axis angle sections between two crystals: the outer sector behind, the axes rescaled
  mdf = unimodalODF(orientation.byAxisAngle([1, 1, 1], 40 * degree, cs, cs), halfwidth=10 * degree)
  h = plotSection(mdf, sections=3, resolution=10 * degree)
  fig = gcm()
  sPs = plotsOf(fig)
  assert len(sPs) == 3 and fig.children[0].get_title().startswith("ω = ")
  w0, w1 = np.diff(fig.children[0].get_xlim())[0], np.diff(fig.children[2].get_xlim())[0]
  assert w0 > w1


def test_sections_of_orientations():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  ori = orientation.byEuler(30 * degree + 5 * degree * rng.standard_normal(300), 50 * degree + 5 * degree * rng.standard_normal(300),
                            10 * degree + 5 * degree * rng.standard_normal(300), cs, ss)
  h = plotSection(ori, sections=6)
  fig = gcm()
  assert len(plotsOf(fig)) == 6 and all(isinstance(x, matplotlib.collections.PathCollection) for x in h)
  assert sum(len(x.get_offsets()) for x in h) >= 300
  # coloured by a value, thinned to a number of points
  h = ori.plotSection("sigma", values=np.arange(300), points=100, colorbar=True)
  fig = gcm()
  assert len(fig.colorbars) == 1 and sum(len(x.get_offsets()) for x in h) <= 100 * 48
  # every orientation with all, once each on axis angle sections: those within the tolerance
  # of a section, and of them those whose axis the sector of the section still allows
  oS = axisAngleSections(cs, ss, sections=4)
  h = plotSection(ori, oS, all=True, MarkerSize=2)
  n = sum(len(x.get_offsets()) for x in h)
  assert 100 < n <= len(oS.sectionData(ori)[0]) <= 300
  h = plotSection(ori, oS, points=50)
  assert sum(len(x.get_offsets()) for x in h) < n


# one section of each function into the cells of a layout, as nextAxis names them
def test_sections_fill_the_cells_of_a_layout():
  from mtex import newMtexFigure, nextAxis, setColorRange
  cs = crystalFrame("432")
  odfs = [unimodalODF(orientation.byEuler(0, 0, 0, cs), halfwidth=h * degree) for h in (10, 20, 30)]
  fig = newMtexFigure(layout=[1, 3])
  for i, odf in enumerate(odfs):
    if i > 0:
      nextAxis()
    plotSection(odf, "sigma", 0 * degree)
  assert gcm() is fig and len(fig.children) == 3 and len(plt.get_fignums()) == 1
  # each cell back to the range of its own values, the widest halfwidth the lowest peak
  setColorRange("equal")
  setColorRange("tight")
  highs = [ax._mtexSphericalPlot.mappable.get_clim()[1] for ax in fig.children]
  assert highs[0] > highs[1] > highs[2]


def test_orientations_coloured_by_rgb_and_nan_dropped(capsys):
  cs = crystalFrame('m-3m')
  ori = orientation.rand(3000, cs, rng=rng)
  ori.data[:10] = np.nan
  rgb = np.tile([1.0, 0.0, 0.0], (3000, 1))
  plt.close("all")
  plotSection(ori, rgb, all=True)
  sets = [c for ax in plt.gcf().axes for c in ax.collections if len(c.get_offsets())]
  assert sets and all(np.allclose(c.get_facecolor()[:, :3], [1, 0, 0]) for c in sets)
  plt.close("all")
  plotSection(ori)
  assert "out of 2990 given" in capsys.readouterr().out
  plt.close("all")
