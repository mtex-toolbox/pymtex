# the matplotlib drawing of spherical plots (docs/plans/plotting.md S1), on the Agg backend
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mtex import (Miller, S2FunHarmonic, S2VectorFieldHarmonic, annotate, cat, contour, contourf, crystalFrame, degree, fundamentalSector, gcm, grad, hold, newMtexFigure, nextAxis,
                  orientation, pcolor, plot, plotIPDF, plotPDF, quiver, scatter, specimenFrame, sphericalRegion, text,
                  unimodalODF, vector3d, xvector, yvector, zvector)
from mtex.functions.s2axisfield import S2AxisFieldHandle
from mtex.plotting.sphericalplot import sphericalPlot

rng = np.random.default_rng(7)


@pytest.fixture(autouse=True)
def closeFigures():
  yield
  plt.close("all")


def plotsOf(fig):
  return [ax._mtexSphericalPlot for ax in fig.children]


# 40 directions of an orthorhombic specimen frame
def specimenDirections():
  return vector3d(vector3d.rand(40, rng=rng), frame=specimenFrame("222"))


def test_plot_directionsOnTwoChartsWithGridBoundaryAndLabels():
  hs = plot(specimenDirections(), grid=True)
  fig = gcm()
  assert len(fig.children) == 2 and len(hs) == 2
  sPs = plotsOf(fig)
  assert all(isinstance(sP, sphericalPlot) for sP in sPs) and sPs[0].proj.upper and not sPs[1].proj.upper
  assert len(sPs[0].grid) == 11 and len(sPs[0].boundary) == 1 and sPs[0].labels


def test_plot_gridLiesOverSurfaceAndUnderBoundary():
  plot(specimenDirections(), grid=True)
  sPs = plotsOf(gcm())
  # the grid lies over a surface, as MATLAB draws it, and under the boundary
  assert sPs[0].grid[0].get_zorder() > 2 and sPs[0].grid[0].get_zorder() < sPs[0].boundary[0].get_zorder()


def test_plot_markersInsideTheirChartAndChartsShareTheList():
  hs = plot(specimenDirections(), grid=True)
  sPs = plotsOf(gcm())
  # every marker lies inside the bounds of its chart, and the two charts share the list
  n = 0
  for sP, h in zip(sPs, hs):
    xy = h.get_offsets()
    xlo, xhi, ylo, yhi = sP.bounds
    assert np.all((xy[:, 0] >= xlo) & (xy[:, 0] <= xhi) & (xy[:, 1] >= ylo) & (xy[:, 1] <= yhi))
    n += len(xy)
  assert n == 40


def test_plot_axesOnOneChartWithMATLABMarkerOptions():
  # axes take one chart, and the marker options are MATLAB's
  h = plot(vector3d.rand(40, antipodal=True, rng=rng), markerSize=6, markerFaceColor="red")
  assert len(gcm().children) == 1 and h[0].get_sizes()[0] == 36
  assert np.allclose(h[0].get_facecolor()[0][:3], (1, 0, 0))


def test_plot_upperStereographicLabelled():
  # upper only, stereographic, labelled
  h = plot(specimenDirections(), projection="stereo", upper=True, label=[str(i) for i in range(40)])
  fig = gcm()
  assert len(fig.children) == 1 and plotsOf(fig)[0].proj.name == "eangle"
  assert sum(isinstance(t, matplotlib.text.Annotation) for t in fig.children[0].texts) == len(h[0].get_offsets()) > 10


def test_vector3d_plotAndScatterMethodsAreTheFunction():
  vs = specimenDirections()
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


# four cubic Miller indices
def cubicMillers():
  cs = crystalFrame("m-3m")
  return Miller([1, 1, 1, 2], [0, 1, 1, 1], [0, 0, 1, 0], cs)


def test_plot_millerSymmetrisedDrawsTheEquivalents():
  h = plot(cubicMillers(), symmetrised=True, upper=True)
  assert len(h[0].get_offsets()) > 4


def test_plot_millerInFundamentalRegionIsTheLabelledSector():
  m = cubicMillers()
  h = plot(m, fundamentalRegion=True, label=[m[i].char() for i in range(4)])
  fig = gcm()
  sP = plotsOf(fig)[0]
  assert len(fig.children) == 1 and len(sP.boundary) == 3
  assert {t.get_text() for t in fig.children[0].texts if isinstance(t, matplotlib.text.Annotation)} >= {"[001]", "[101]", "[111]"}
  assert len(h[0].get_offsets()) == 4


def test_text_atTheMarkers():
  m = cubicMillers()
  # a text command at the markers
  scatter(m, upper=True)
  hold(True)
  text(m, ["a", "b", "c", "d"])
  hold(False)
  assert len(gcm().children) == 1 and sum(t.get_text() in "abcd" for t in gcm().children[0].texts) == 4


def test_text_labeledWritesEachDirectionsNameIntoExistingCharts():
  scatter(cubicMillers(), upper=True)
  # text(v, 'labeled') writes each direction's own name, into the charts already there
  text(cat(xvector, zvector), labeled=True)
  assert len(gcm().children) == 1 and "x" in {t.get_text() for t in gcm().children[0].texts}


def test_plot_polarTrigonalSectorOnBothHemispheres():
  # the polar trigonal sector reaches into both hemispheres
  h = plot(fundamentalSector(crystalFrame("3m")))
  assert len(gcm().children) == 2 and len(h) >= 2


def test_annotate_drawsIntoEveryAxesOfTheFigure():
  tri = crystalFrame("3m")
  plot(fundamentalSector(tri))
  # annotate draws into every axes of the current figure, the corners of the sector here
  annotate(fundamentalSector(tri).vertices())
  assert len(gcm().children) == 2 and all(len(ax.collections) >= 1 for ax in gcm().children)


def test_plot_regionOutlineAddedToAHeldPlot():
  # the outline of a region added to a held plot
  plot(vector3d.rand(10, rng=rng), upper=True)
  hold(True)
  h = plot(sphericalRegion(vector3d(1, 1, 0)))
  hold(False)
  assert len(gcm().children) == 1 and len(h) == 1


# a random positive function on the sphere
def randomS2Fun():
  return S2FunHarmonic(rng.standard_normal(49) + 1j * rng.standard_normal(49), real=True) + 3


# a random function with trigonal symmetry
def trigonalS2Fun():
  cs = crystalFrame("-3m")
  return S2FunHarmonic(rng.standard_normal(81) + 1j * rng.standard_normal(81), cs, real=True) + 4


def test_plot_S2FunOneColourRangeOverBothHemispheres():
  h = plot(randomS2Fun(), colorbar=True, minmax=True)
  fig = gcm()
  sPs = plotsOf(fig)
  assert len(fig.children) == 2 and all(isinstance(x, matplotlib.collections.QuadMesh) for x in h)
  # one colour range over the two hemispheres, one colour bar
  assert h[0].get_clim() == h[1].get_clim() and len(fig.colorbars) == 1
  lo, hi = h[0].get_clim()
  assert lo <= min(sP.minData for sP in sPs) and hi >= max(sP.maxData for sP in sPs)
  assert sum("Max:" in t.get_text() for t in fig.children[0].texts) == 1


def test_contourf_S2FunTakesColormapAndContours():
  h = contourf(randomS2Fun(), upper=True, colormap="LaboTeX", contours=8)
  assert len(gcm().children) == 1 and isinstance(h[0], matplotlib.contour.QuadContourSet)
  assert h[0].get_cmap().name == "LaboTeX" and len(h[0].levels) == 8


def test_contour_S2FunIsAContourSet():
  h = contour(randomS2Fun(), upper=True)
  assert isinstance(h[0], matplotlib.contour.QuadContourSet)


def test_pcolor_S2FunPlainProjectionIsInvertedInY():
  pcolor(randomS2Fun(), projection="plain")
  assert len(gcm().children) == 1 and gcm().children[0].yaxis_inverted()


def test_S2Fun_plotMethodsTakeAColourRange():
  s = randomS2Fun()
  h = s.plot(upper=True, colorRange=(0, 5))
  assert h[0].get_clim() == (0, 5)
  s.contourf(upper=True)
  s.contour(upper=True)
  s.pcolor(upper=True)


def test_contourf_symmetricS2FunOnItsSectorCompleteOnTheSphere():
  sc = trigonalS2Fun()
  # a symmetric function is drawn on its sector, complete on the whole sphere
  contourf(sc, colorbar=True)
  assert len(gcm().children) == 1 and len(plotsOf(gcm())[0].sR.N) >= 2
  contourf(sc, complete=True)
  assert len(gcm().children) == 2


def test_plot_S2FunTakesMATLABPositionalFlags():
  # MATLAB's positional flags, both hemispheres named
  plot(trigonalS2Fun(), 'complete', 'smooth', 'upper', 'lower')
  assert len(gcm().children) == 2


def test_contourf_ofDirectionsIsTheirDensity():
  h = contourf(vector3d.rand(200, rng=rng), halfwidth=10 * degree, minmax=True)
  fig = gcm()
  assert len(fig.children) == 2 and all(isinstance(x, matplotlib.contour.QuadContourSet) for x in h)


def test_plot_dataOnDirectionsPcolorInterpolatesToAFunction():
  # data interpolated to a function
  h = plot(vector3d.rand(200, rng=rng), rng.random(200), pcolor=True, upper=True)
  assert isinstance(h[0], matplotlib.collections.QuadMesh)


def test_plot_vectorFieldLongestArrowIsHalfTheGridSpacing():
  G = grad(randomS2Fun())
  h = plot(G, upper=True, resolution=10 * degree)
  assert len(gcm().children) == 1 and isinstance(h[0], matplotlib.quiver.Quiver)
  # MATLAB's length: the longest arrow half the grid spacing
  assert abs(np.max(np.hypot(h[0].U, h[0].V)) - 0.5 * 10 * degree) < 1e-9


def test_quiver_normalizeGivesArrowsOfOneLength():
  h = quiver(grad(randomS2Fun()), upper=True, normalize=True)
  L = np.hypot(h[0].U, h[0].V)
  assert np.allclose(L, L[0])


def test_plot_arrowsOnAHeldFilledPlot():
  s = randomS2Fun()
  # arrows on a held filled plot
  contourf(s, upper=True)
  hold(True)
  plot(grad(s), color="white")
  hold(False)
  assert len(gcm().children) == 1


def test_quiver_axisFieldNeedleOnTheRimStepsBack():
  # an axis field over the whole hemisphere: a needle on the rim steps back rather than over
  # it, so the interior needles keep their length
  A = S2AxisFieldHandle(lambda v: vector3d(v.cross(zvector).data + 0.3 * zvector.data, antipodal=True), bandwidth=12)
  h = quiver(A, upper=True, resolution=10 * degree)
  lengths = np.linalg.norm(np.diff(np.asarray(h[0].get_segments()), axis=1)[:, 0], axis=-1)
  assert np.median(lengths) > 0.3 * lengths.max()


def test_quiver_ofAPairOfLists():
  # a pair of lists
  v = vector3d.rand(10, rng=rng)
  h = quiver(v, v.orth(), upper=True)
  assert isinstance(h[0], matplotlib.quiver.Quiver)


# a cubic unimodal ODF under orthorhombic specimen symmetry and three cubic directions
@pytest.fixture(scope="module")
def cubicODF():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  odf = unimodalODF(orientation.byEuler(30 * degree, 50 * degree, 10 * degree, cs, ss), halfwidth=15 * degree)
  return cs, odf, Miller([1, 1, 1], [0, 1, 1], [0, 0, 1], cs)


def test_plotPDF_ofODFTitledPanelsEachOnItsOwnRange(cubicODF):
  cs, odf, hs = cubicODF
  h = plotPDF(odf, hs)
  fig = gcm()
  assert len(fig.children) == 3 and [ax.get_title() for ax in fig.children] == ["{100}", "{110}", "{111}"]
  clims = {x.get_clim() for x in h}
  # every pole figure on its own range, as MATLAB's
  assert len(clims) == 3


def test_plotPDF_secondRowContinuesALayoutFromNextAxis(cubicODF):
  cs, odf, hs = cubicODF
  # a second row continues a newMtexFigure layout from the axes nextAxis points to
  newMtexFigure(layout=[2, 3])
  lay = gcm()
  plotPDF(odf, hs)
  nextAxis()
  plotPDF(odf, hs, title=False)
  assert gcm() is lay and len(lay.children) == 6 and all(hasattr(ax, "_mtexSphericalPlot") for ax in lay.children)


def test_setColorRange_equalRecoloursBandsAndKeepsLevels(cubicODF):
  from mtex import setColorRange
  cs, odf, hs = cubicODF
  # a shared colour range recolours the contour bands and keeps each panel's levels, as MATLAB's
  hs = plotPDF(odf, hs, 'contourf')
  before = [tuple(np.round(h.levels, 9)) for h in hs]
  setColorRange('equal')
  after = [tuple(np.round(ax._mtexSphericalPlot.mappable.levels, 9)) for ax in gcm().children]
  assert before == after and gcm().colorRange[1] >= max(max(l) for l in before)


def test_mtexColorbar_oneBarForCommonRangeAndYlabelBesideAChart(cubicODF):
  from mtex import mtexColorbar, setColorRange, ylabel
  cs, odf, hs = cubicODF
  plotPDF(odf, hs, 'contourf')
  setColorRange('equal')
  # one bar for the common range, and a label beside a chart, MATLAB's ylabel(ax, ...)
  assert len(mtexColorbar()) == 1
  ylabel(gcm().children[0], 'row one')
  assert any(t.get_text() == 'row one' for t in gcm().children[0].texts)


def test_plotPDF_colorRangeEqualIsOneRangeOnTheSpecimenSector(cubicODF):
  cs, odf, hs = cubicODF
  # one range over the row when asked
  h = plotPDF(odf, hs, colorRange='equal')
  clims = {x.get_clim() for x in h}
  fig = gcm()
  assert len(clims) == 1 and fig.colorRange == next(iter(clims))
  # the specimen sector of the 222 group
  assert all(len(sP.sR.N) == 3 for sP in plotsOf(fig))
  assert all(sP.proj.antipodal for sP in plotsOf(fig))


def test_plotPDF_ofODFTakesMATLABPositionalFlags(cubicODF):
  cs, odf, _ = cubicODF
  # MATLAB's positional flags
  h = plotPDF(odf, Miller(1, 0, 0, cs), 'antipodal', 'contourf')
  assert isinstance(h[0], matplotlib.contour.QuadContourSet)


def test_plotPDF_ofODFContourfNoTitleABarPerPanel(cubicODF):
  cs, odf, _ = cubicODF
  h = plotPDF(odf, Miller([1, 1], [0, 1], [0, 1], cs), contourf=True, contours=8, colorbar=True, title=False)
  fig = gcm()
  assert len(fig.children) == 2 and fig.children[0].get_title() == "" and len(fig.colorbars) == 2
  assert all(isinstance(x, matplotlib.contour.QuadContourSet) for x in h)


def test_plotPDF_completeDrawsBothHemispheresOfAnAntipodalPoleFigure(cubicODF):
  cs, odf, _ = cubicODF
  # complete asks for both hemispheres even of an antipodal pole figure, as MATLAB draws it
  odf.plotPDF(Miller(1, 0, 0, cs), complete=True)
  assert len(gcm().children) == 2 and all(len(sP.sR.N) == 1 for sP in plotsOf(gcm()))


def test_plotIPDF_ofODFTitledBySpecimenDirectionsOnTheCrystalSector(cubicODF):
  cs, odf, _ = cubicODF
  # the inverse pole figures, titled by the specimen directions, on the crystal sector
  plotIPDF(odf, [xvector, yvector, zvector])
  fig = gcm()
  assert [ax.get_title() for ax in fig.children] == ["X", "Y", "Z"]
  assert all(len(sP.sR.N) >= 3 for sP in plotsOf(fig))
  odf.plotIPDF(zvector, colorbar=True)
  assert len(gcm().children) == 1


def test_plot_parentAxesAreReusedWithTheirProjection():
  vs = vector3d.rand(10, rng=rng)
  fig, ax = plt.subplots()
  h = plot(vs, upper=True, parent=ax)
  assert h[0].axes is ax and hasattr(ax, "_mtexSphericalPlot")
  # a second plot into the same axes reuses its projection
  plot(vector3d.rand(5, rng=rng), parent=ax)
  assert len(ax.collections) == 2 and len(ax._mtexSphericalPlot.boundary) == 1


def test_hold_keepsTheAxesAndAPlotAfterItStartsAFigure():
  vs = vector3d.rand(10, rng=rng)
  # hold keeps the current axes, and a new plot after the hold starts a figure
  plot(vs, upper=True)
  first = gcm()
  hold(True)
  plot(vector3d.rand(5, rng=rng))
  assert gcm() is first and len(first.children) == 1 and len(first.children[0].collections) == 2
  hold(False)
  plot(vs, upper=True)
  assert gcm() is not first


def test_arrows_draw_the_tangential_part_at_its_length():
  # as MATLAB's quiver: the normal field draws nothing, a tangential one its full size
  h = plot(S2VectorFieldHarmonic.normal(), upper=True)[0]
  assert np.hypot(h.U, h.V).max() < 1e-8
  t = S2VectorFieldHarmonic(lambda v: v.cross(zvector), bandwidth=4)
  h = plot(t, upper=True)[0]
  assert np.isclose(np.hypot(h.U, h.V).max(), 0.5 * 10 * degree, rtol=0.02)


def test_text_goes_where_nextAxis_points_in_a_held_figure():
  # MATLAB's text adds to the current axes; without a chosen one to every chart
  newMtexFigure(layout=[1, 3])
  for k in range(3):
    if k:
      nextAxis()
    plot(S2FunHarmonic(lambda v: v.z ** 2, bandwidth=4), upper=True)
  nextAxis(1, 2)
  hold(True)
  text(xvector, ["a"])
  hold(False)
  assert [len([t for t in ax.texts if t.get_text() == "a"]) for ax in gcm().children] == [0, 1, 0]
  text(xvector, ["b"])
  assert all(any(t.get_text() == "b" for t in ax.texts) for ax in gcm().children)


def test_many_open_markers_keep_their_edge():
  plt.close("all")
  plot(vector3d.rand(12000, rng=rng), markerFaceColor='none', markerEdgeColor='k')
  c = plt.gcf().axes[0].collections[-1]
  assert len(c.get_edgecolor()) and np.allclose(c.get_edgecolor()[0][:3], 0)
  plt.close("all")


# a trigonal unimodal ODF and its mode
def trigonalODF():
  cs = crystalFrame("-3m")
  mod = orientation.byEuler(110 * degree, 30 * degree, 80 * degree, cs)
  return cs, mod, unimodalODF(mod)


def test_annotate_orientationMarksEveryInversePoleFigure():
  cs, mod, odf = trigonalODF()
  # annotate(ori) marks an inverse pole figure at the crystal direction along its r, in the sector
  plotIPDF(odf, [xvector, zvector], antipodal=True)
  hs = annotate(mod, marker='s', markerFaceColor='r', label='A', color='w')
  assert len(hs) == 2 and all(h is not None and len(h.get_offsets()) == 1 for h in hs)
  # beside a markerFaceColor, color is the colour of the text
  assert any(t.get_text() == 'A' and matplotlib.colors.to_hex(t.get_color()) == '#ffffff' for t in gcm().children[0].texts)


def test_plot_fibreAdd2allNamedOnceInTheLegend():
  from mtex import fibre, legend
  cs, mod, odf = trigonalODF()
  # a fibre with add2all is drawn on every pole figure as curves, named once per axes
  plotPDF(odf, Miller([[1, 0, 0], [1, 1, 1]], cs))
  plot(fibre(Miller(1, 1, -2, 1, cs), vector3d.Y), color='red', add2all=True, displayName='fibre')
  leg = legend('show')
  assert [t.get_text() for t in leg.get_texts()] == ['fibre']


def test_plot_labelWithSubscriptIsMathematics():
  # a label with a subscript is set as mathematics, as MATLAB's tex interpreter does
  plot(vector3d(1, 1, 1), label='v_1')
  assert any(t.get_text() == '$v_1$' for ax in gcm().children for t in ax.texts)


# a fibre ODF of quartz and the direction it is plotted at
def quartzFibre():
  from mtex import fibreODF
  cs = crystalFrame("321")
  return fibreODF(Miller(1, 1, 0, cs), zvector), Miller(1, 1, 0, cs)


def test_plotPDF_contourfLevelsAreTheBands():
  odf, h = quartzFibre()
  # 'contourf', levels: the bands at the levels, the colours spread over them, labelled with ShowText
  plotPDF(odf, h, contourf=np.arange(0, 6), antipodal=True)
  assert np.allclose(gcm().children[0]._mtexSphericalPlot.mappable.levels, np.arange(0, 6))
  plotPDF(odf, h, contour=[1, 3], antipodal=True, ShowText='on')
  assert gcm().children[0]._mtexSphericalPlot.lines[0].labelTexts


def test_plotPDF_logarithmicIsALogNorm():
  odf, h = quartzFibre()
  # 'logarithmic' is a logarithmic colour scale
  plotPDF(odf, h, antipodal=True, logarithmic=True)
  assert isinstance(gcm().children[0]._mtexSphericalPlot.mappable.norm, matplotlib.colors.LogNorm)


def test_plotPDF_antipodalOfPolarCrystalSuperposesBothHemispheres():
  odf, h = quartzFibre()
  # an antipodal pole figure of a polar crystal superposes both hemispheres: 15.3 = 30.6 / 2
  plotPDF(odf, h, antipodal=True)
  assert abs(gcm().children[0]._mtexSphericalPlot.maxData - 15.3) < 0.2


def test_plotPDF_3dIsTheSphereInSpace():
  from mtex import plottingConvention
  odf, h = quartzFibre()
  # '3d': the pole figure on the sphere in space, seen along the convention given
  pC = plottingConvention()
  pC.east, pC.outOfScreen = vector3d(9, 3, 3), vector3d(6, 10, 9)
  plotPDF(odf, h, '3d', pC, grid=True, grid_res=10 * degree, title=False)
  assert gcm().children[0].name == '3d'


def test_plotHKL_labelsTheCubicPlanes():
  from mtex import plotHKL
  newMtexFigure(layout=[1, 2])
  plotHKL(crystalFrame("m-3m"), projection='edist', grid_res=15 * degree)
  labels = sorted(t.get_text() for t in gcm().children[0].texts if t.get_text().startswith('('))
  assert len(labels) == 17 and '(001)' in labels


def test_plot_symmetrisedLabeledNamesTheEquivalentInsideAnAntipodalSector():
  # an equivalent inside the sector is labelled rather than the antipode of one
  cs = crystalFrame("-3m")
  plotIPDF(unimodalODF(orientation.id(cs)), xvector, axisLabels=False)
  hold(True)
  plot(Miller(0, 0, 0, 1, cs), symmetrised=True, labeled=True)
  hold(False)
  assert [t.get_text() for t in gcm().children[0].texts if t.get_text().startswith('(')] == ['(0001)']


# a label on a marker of size zero, text alone, stands on its point; beside a marker below it
def test_text_alone_stands_on_its_point():
  plot(vector3d.rand(5, rng=rng), upper=True)
  annotate(yvector, label="Y1", markerSize=0, backgroundColor="w", color="red")
  annotate(xvector, label="X1")
  alone = next(t for t in gcm().children[0].texts if t.get_text() == "Y1")
  marked = next(t for t in gcm().children[0].texts if t.get_text() == "X1")
  assert alone.get_va() == "center" and tuple(alone.xyann) == (0, 0) and alone.get_color() == "red"
  assert alone.get_bbox_patch() is not None and marked.get_va() == "top" and marked.xyann[1] < 0
