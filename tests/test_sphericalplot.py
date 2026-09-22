# the matplotlib drawing of spherical plots (docs/plans/plotting.md S1), on the Agg backend
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mtex import (Miller, S2FunHarmonic, annotate, contour, contourf, crystalFrame, degree, fundamentalSector, gcm, grad, hold,
                  orientation, pcolor, plot, plotIPDF, plotPDF, quiver, scatter, specimenFrame, sphericalRegion, text,
                  unimodalODF, vector3d, xvector, yvector, zvector)
from mtex.plotting.sphericalplot import sphericalPlot

rng = np.random.default_rng(7)


@pytest.fixture(autouse=True)
def closeFigures():
  yield
  plt.close("all")


def plotsOf(fig):
  return [ax._mtexSphericalPlot for ax in fig.children]


def test_directions_as_markers():
  ss = specimenFrame("222")
  vs = vector3d(vector3d.rand(40, rng=rng), frame=ss)
  hs = plot(vs, grid=True)
  fig = gcm()
  assert len(fig.children) == 2 and len(hs) == 2
  sPs = plotsOf(fig)
  assert all(isinstance(sP, sphericalPlot) for sP in sPs) and sPs[0].proj.upper and not sPs[1].proj.upper
  assert len(sPs[0].grid) == 11 and len(sPs[0].boundary) == 1 and sPs[0].labels
  # every marker lies inside the bounds of its chart, and the two charts share the list
  n = 0
  for sP, h in zip(sPs, hs):
    xy = h.get_offsets()
    xlo, xhi, ylo, yhi = sP.bounds
    assert np.all((xy[:, 0] >= xlo) & (xy[:, 0] <= xhi) & (xy[:, 1] >= ylo) & (xy[:, 1] <= yhi))
    n += len(xy)
  assert n == 40
  # axes take one chart, and the marker options are MATLAB's
  h = plot(vector3d.rand(40, antipodal=True, rng=rng), MarkerSize=6, MarkerFaceColor="red")
  assert len(gcm().children) == 1 and h[0].get_sizes()[0] == 36
  assert np.allclose(h[0].get_facecolor()[0][:3], (1, 0, 0))
  # upper only, stereographic, labelled
  h = plot(vs, projection="stereo", upper=True, label=[str(i) for i in range(40)])
  fig = gcm()
  assert len(fig.children) == 1 and plotsOf(fig)[0].proj.name == "eangle"
  assert sum(isinstance(t, matplotlib.text.Annotation) for t in fig.children[0].texts) == len(h[0].get_offsets()) > 10
  # the method is the function
  vs.plot(upper=True)
  assert len(gcm().children) == 1
  vs.scatter(upper=True)
  assert len(gcm().children) == 1


def test_markers_coloured_by_data_and_rgb():
  vs = vector3d.rand(30, rng=rng)
  data = rng.random(30)
  h = plot(vs, data, upper=True, colorbar=True)
  fig = gcm()
  sP = plotsOf(fig)[0]
  keep = np.isfinite(sP.proj.project(vs)[0])
  assert sP.mappable is h[0] and np.isclose(sP.minData, data[keep].min()) and len(h[0].get_offsets()) == keep.sum()
  assert len(fig.colorbars) == 1
  rgb = rng.random((30, 3))
  h = plot(vs, rgb, upper=True)
  assert h[0].get_facecolor().shape[1] == 4 and len(h[0].get_facecolor()) == len(h[0].get_offsets())


def test_miller_indices_symmetrised_and_in_the_sector():
  cs = crystalFrame("m-3m")
  m = Miller([1, 1, 1, 2], [0, 1, 1, 1], [0, 0, 1, 0], cs)
  h = plot(m, symmetrised=True, upper=True)
  assert len(h[0].get_offsets()) > 4
  h = plot(m, fundamentalRegion=True, label=[m[i].char() for i in range(4)])
  fig = gcm()
  sP = plotsOf(fig)[0]
  assert len(fig.children) == 1 and len(sP.boundary) == 3
  assert {t.get_text() for t in fig.children[0].texts if isinstance(t, matplotlib.text.Annotation)} >= {"[001]", "[101]", "[111]"}
  assert len(h[0].get_offsets()) == 4
  # a text command at the markers
  scatter(m, upper=True)
  hold(True)
  text(m, ["a", "b", "c", "d"])
  hold(False)
  assert len(gcm().children) == 1 and sum(t.get_text() in "abcd" for t in gcm().children[0].texts) == 4


def test_a_spherical_region_and_annotate():
  tri = crystalFrame("-3m")
  h = plot(fundamentalSector(tri))
  assert len(gcm().children) == 2 and len(h) >= 2
  # annotate draws into every axes of the current figure, the corners of the sector here
  annotate(fundamentalSector(tri).vertices())
  assert len(gcm().children) == 2 and all(len(ax.collections) >= 1 for ax in gcm().children)
  # the outline of a region added to a held plot
  plot(vector3d.rand(10, rng=rng), upper=True)
  hold(True)
  h = plot(sphericalRegion(vector3d(1, 1, 0)))
  hold(False)
  assert len(gcm().children) == 1 and len(h) == 1


def test_functions_on_the_sphere():
  s = S2FunHarmonic(rng.standard_normal(49) + 1j * rng.standard_normal(49), real=True) + 3
  h = plot(s, colorbar=True, minmax=True)
  fig = gcm()
  sPs = plotsOf(fig)
  assert len(fig.children) == 2 and all(isinstance(x, matplotlib.collections.QuadMesh) for x in h)
  # one colour range over the two hemispheres, one colour bar
  assert h[0].get_clim() == h[1].get_clim() and len(fig.colorbars) == 1
  lo, hi = h[0].get_clim()
  assert lo <= min(sP.minData for sP in sPs) and hi >= max(sP.maxData for sP in sPs)
  assert sum("Max:" in t.get_text() for t in fig.children[0].texts) == 1
  h = contourf(s, upper=True, colormap="LaboTeX", contours=8)
  assert len(gcm().children) == 1 and isinstance(h[0], matplotlib.contour.QuadContourSet)
  assert h[0].get_cmap().name == "LaboTeX" and len(h[0].levels) == 8
  h = contour(s, upper=True)
  assert isinstance(h[0], matplotlib.contour.QuadContourSet)
  h = pcolor(s, projection="plain")
  assert len(gcm().children) == 1 and gcm().children[0].yaxis_inverted()
  h = s.plot(upper=True, colorRange=(0, 5))
  assert h[0].get_clim() == (0, 5)
  s.contourf(upper=True)
  s.contour(upper=True)
  s.pcolor(upper=True)
  # a symmetric function is drawn on its sector, complete on the whole sphere
  cs = crystalFrame("-3m")
  sc = S2FunHarmonic(rng.standard_normal(81) + 1j * rng.standard_normal(81), cs, real=True) + 4
  contourf(sc, colorbar=True)
  assert len(gcm().children) == 1 and len(plotsOf(gcm())[0].sR.N) >= 2
  contourf(sc, complete=True)
  assert len(gcm().children) == 2


def test_the_density_of_directions():
  vs = vector3d.rand(200, rng=rng)
  h = contourf(vs, halfwidth=10 * degree, minmax=True)
  fig = gcm()
  assert len(fig.children) == 2 and all(isinstance(x, matplotlib.contour.QuadContourSet) for x in h)
  # data interpolated to a function
  h = plot(vs, rng.random(200), pcolor=True, upper=True)
  assert isinstance(h[0], matplotlib.collections.QuadMesh)


def test_vector_and_axis_fields_as_arrows():
  s = S2FunHarmonic(rng.standard_normal(49) + 1j * rng.standard_normal(49), real=True) + 3
  G = grad(s)
  h = plot(G, upper=True, resolution=10 * degree)
  assert len(gcm().children) == 1 and isinstance(h[0], matplotlib.quiver.Quiver)
  assert abs(np.max(np.hypot(h[0].U, h[0].V)) - 1.2 * 10 * degree) < 1e-9
  h = quiver(G, upper=True, normalize=True)
  L = np.hypot(h[0].U, h[0].V)
  assert np.allclose(L, L[0])
  # arrows on a held filled plot
  contourf(s, upper=True)
  hold(True)
  plot(G, color="white")
  hold(False)
  assert len(gcm().children) == 1
  # a pair of lists
  v = vector3d.rand(10, rng=rng)
  h = quiver(v, v.orth(), upper=True)
  assert isinstance(h[0], matplotlib.quiver.Quiver)


def test_pole_figures_of_an_odf():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  odf = unimodalODF(orientation.byEuler(30 * degree, 50 * degree, 10 * degree, cs, ss), halfwidth=15 * degree)
  h = plotPDF(odf, Miller([1, 1, 1], [0, 1, 1], [0, 0, 1], cs))
  fig = gcm()
  assert len(fig.children) == 3 and [ax.get_title() for ax in fig.children] == ["{100}", "{110}", "{111}"]
  # one colour range over the row, the specimen sector of the 222 group
  clims = {x.get_clim() for x in h}
  assert len(clims) == 1 and fig.colorRange == next(iter(clims))
  assert all(len(sP.sR.N) == 3 for sP in plotsOf(fig))
  assert all(sP.proj.antipodal for sP in plotsOf(fig))
  h = plotPDF(odf, Miller([1, 1], [0, 1], [0, 1], cs), contourf=True, contours=8, colorbar=True, noTitle=True)
  fig = gcm()
  assert len(fig.children) == 2 and fig.children[0].get_title() == "" and len(fig.colorbars) == 1
  assert all(isinstance(x, matplotlib.contour.QuadContourSet) for x in h)
  # complete asks for both hemispheres even of an antipodal pole figure, as MATLAB draws it
  odf.plotPDF(Miller(1, 0, 0, cs), complete=True)
  assert len(gcm().children) == 2 and all(len(sP.sR.N) == 1 for sP in plotsOf(gcm()))
  # the inverse pole figures, titled by the specimen directions, on the crystal sector
  plotIPDF(odf, [xvector, yvector, zvector])
  fig = gcm()
  assert [ax.get_title() for ax in fig.children] == ["X", "Y", "Z"]
  assert all(len(sP.sR.N) >= 3 for sP in plotsOf(fig))
  odf.plotIPDF(zvector, colorbar=True)
  assert len(gcm().children) == 1


def test_where_a_plot_goes():
  vs = vector3d.rand(10, rng=rng)
  fig, ax = plt.subplots()
  h = plot(vs, upper=True, parent=ax)
  assert h[0].axes is ax and hasattr(ax, "_mtexSphericalPlot")
  # a second plot into the same axes reuses its projection
  plot(vector3d.rand(5, rng=rng), parent=ax)
  assert len(ax.collections) == 2 and len(ax._mtexSphericalPlot.boundary) == 1
  # hold keeps the current axes, and a new plot after the hold starts a figure
  plot(vs, upper=True)
  first = gcm()
  hold(True)
  plot(vector3d.rand(5, rng=rng))
  assert gcm() is first and len(first.children) == 1 and len(first.children[0].collections) == 2
  hold(False)
  plot(vs, upper=True)
  assert gcm() is not first
