# the drawing of crystal shapes (docs/plans/plotting.md S5), on the Agg backend
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from mtex import (EBSD, Miller, calcGrains, crystalFrame, crystalShape, degree, gcm, hold, ipfHSVKey, orientation, phaseList, plot,
                  plotInnerDirection, plotInnerFace, plotSlipSystem, rotation, slipSystem, vector3d, zvector)
from mtex.plotting.shapeplot import faceColors

CS = crystalFrame("6/mmm", [3, 3, 4.7], mineral="Titanium")


@pytest.fixture(autouse=True)
def closeFigures():
  yield
  plt.close("all")


def axesOf():
  return gcm().children[-1]


# a map of two cubic grains, turned 30 degree apart, and the cube of their phase
@pytest.fixture(scope="module")
def gridMap():
  fr = crystalFrame("m-3m", mineral="cubes")
  nx, ny = 8, 4
  i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
  i, j = i.reshape(-1, order="F"), j.reshape(-1, order="F")
  pos = np.stack([i, j, np.zeros(nx * ny)], axis=-1).astype(float)
  rot = rotation.byAxisAngle(zvector, np.where(i < 4, 0.0, 30 * degree))
  ebsd = EBSD(pos, rot, np.full(nx * ny, 2, dtype=np.int8), phaseList(crystalFrame("1"), fr))
  return calcGrains(ebsd, angle=5 * degree), crystalShape.cube(fr)


# three hexagonal prisms turned about c
def threeShapes():
  return orientation.byEuler(np.array([0, 30, 60]) * degree, 0, 0, CS) * crystalShape.hex(CS)


def test_plot_ofAShapeDrawsItsFacesInSpaceWithTheirAlpha():
  h = plot(crystalShape.hex(CS), faceAlpha=0.2)
  ax = axesOf()
  assert isinstance(h, Poly3DCollection) and ax.name == "3d" and len(h.get_facecolor()) == 8
  assert np.allclose(h.get_facecolor()[:, 3], 0.2) and ax.get_xlim()[1] < 1


def test_plot_coloredGivesEveryFormItsColourWithALegend():
  cS = crystalShape.hex(CS)
  # the method is the function, and every form its own colour with a legend
  h = cS.plot(colored=True)
  assert len(np.unique(h.get_facecolor(), axis=0)) == 2 and axesOf().get_legend() is not None
  assert [t.get_text() for t in axesOf().get_legend().get_texts()] == [cS.N[0].round().char(), cS.N[1].round().char()]


def test_plot_coloredTakesTheDirectionKeyOfTheNormals():
  plot(crystalShape.quartz(simple=True), colored=ipfHSVKey(crystalShape.quartz().CS))


def test_faceColors_isOneColourAColourPerFaceOrAColourPerShape():
  cS = crystalShape.hex(CS)
  c = faceColors(cS, "red")
  assert c.shape == (1, 8, 4) and np.allclose(c[0, 0], [1, 0, 0, 1])
  assert np.allclose(faceColors(cS)[0, 0, :3], (0.53, 0.81, 0.98))
  assert faceColors(cS, np.random.default_rng(0).random((8, 3))).shape == (1, 8, 4)
  assert faceColors(cS, cS.faceArea).shape == (1, 8, 4)
  fc = faceColors(threeShapes(), np.random.default_rng(0).random((3, 3)))
  assert fc.shape == (3, 8, 4) and np.allclose(fc[1, 0], fc[1, 5]) and not np.allclose(fc[0, 0], fc[1, 0])


def test_faceColors_refusesAColourCountOfNeitherFacesNorShapes():
  with pytest.raises(ValueError):
    faceColors(crystalShape.hex(CS), np.zeros((5, 3)))


def test_plot_ofAListOfShapesAtPositions():
  # a list at positions
  h = plot(threeShapes() + np.arange(3)[:, None] * np.array([[1.2, 0, 0]]), faceColor=np.random.default_rng(0).random((3, 3)))
  assert len(h.get_facecolor()) == 24 and axesOf().get_xlim()[1] > 2


def test_plot_takesThePositionsBeforeTheShape():
  cS = crystalShape.hex(CS)
  plot(np.arange(3) * 2.0, np.zeros(3), cS)
  assert axesOf().get_xlim()[1] > 4
  plot(np.zeros((2, 2)), cS)
  plot(1.0, 2.0, 3.0, cS)


def test_plot_heldShapesJoinOneCollectionSoTheirFacesSort():
  cS = crystalShape.hex(CS)
  twinning = orientation.map(Miller(0, 1, -1, -2, CS), Miller(0, -1, 1, -2, CS), Miller(2, -1, -1, 0, CS), Miller(2, -1, -1, 0, CS))
  plot(cS, faceAlpha=0.5)
  hold(True)
  h = plot(twinning * cS * 0.9, faceColor="orange")
  hold(False)
  assert len(h.get_facecolor()) == 16 and len([c for c in axesOf().collections if isinstance(c, Poly3DCollection)]) == 1


def test_plot_ofASelectedFaceIsDrawnOverTheShape():
  q = crystalShape.quartz(simple=True)
  plot(q)
  hold(True)
  h = plot(q[Miller(0, -1, 1, 0, q.CS)], faceColor="DarkRed")
  hold(False)
  assert len(h.get_facecolor()) == q.numFaces + 1


def test_plot_ofSlipSystemsInsideAShapeJoinItsCollectionAndKeepItsLimits():
  cS = crystalShape.hex(CS)
  sS = [slipSystem.pyramidal2CA(CS), slipSystem.pyramidalA(CS)]
  plot(cS, faceAlpha=0.2)
  hold(True)
  h1 = plot(cS, sS[1], faceColor="blue")
  h2 = plot(cS, sS[0], faceColor="red")
  hold(False)
  ax = axesOf()
  assert h1 is h2 and len(h1.get_facecolor()) == 10 and len(ax.collections) == 3
  # the limits stay those of the shape
  assert ax.get_xlim()[1] < 1


def test_plotInnerFaceAndDirection_drawIntoAHeldShape():
  ol = crystalShape.olivine()
  plot(ol, faceAlpha=0.2, colored=True)
  hold(True)
  h = plotInnerFace(ol, Miller(1, 0, -1, ol.CS), faceColor="red", faceAlpha=0.5)
  a = plotInnerDirection(ol, Miller(1, 0, 1, ol.CS, "uvw"), faceColor="blue", arrowWidth=0.05)
  hold(False)
  assert h is not None and a is not None


def test_plotSlipSystem_drawsOneOrTheSymmetrisedSystemsIntoAShape():
  ol = crystalShape.olivine()
  ol.plotSlipSystem(slipSystem(Miller(1, 0, 1, ol.CS, "uvw"), Miller(1, 0, -1, ol.CS)))
  plotSlipSystem(ol, slipSystem(Miller(1, 0, 1, ol.CS, "uvw"), Miller(1, 0, -1, ol.CS)).symmetrise())


def test_plotInnerFace_takesAPlaneNormalAsMillerOrVector():
  cS = crystalShape.hex(CS)
  plot(cS, Miller(0, 0, 0, 1, CS))
  assert plotInnerFace(cS, vector3d(1, 0, 0)) is not None


def test_plot_ofShapesIntoAHeldMapIsASquarePerGrainFlippedByTheLayout(gridMap):
  g, cube = gridMap
  plot(g)
  hold(True)
  h = plot(g, cube, faceColor="red", faceAlpha=0.7)
  hold(False)
  ax = axesOf()
  # one square per grain, the map's layout flips y
  assert isinstance(h, PolyCollection) and len(h.get_paths()) == 2 and hasattr(ax, "_mtexMapPlot")
  P = h.get_paths()[0].vertices
  assert len(P) == 5 and np.all(P[:, 1] <= 0)


def test_plot_ofShapesAloneMakesAMapWithItsScaleBarAndALegendOfTheForms(gridMap):
  g, cube = gridMap
  h = plot(g, cube, colored=True)
  assert len(h.get_paths()) == 2 and axesOf().get_legend() is not None and getattr(axesOf()._mtexMapPlot, "scaleBarArtists", None)


def test_plot_ofShapesTakesAColourPerGrain(gridMap):
  g, cube = gridMap
  plot(g, cube, faceColor=np.array([[1, 0, 0], [0, 1, 0.0]]))


def test_plot_ofAShapeOfAnotherLatticeIntoAMapDrawsNothing(gridMap):
  g, _ = gridMap
  assert plot(g, crystalShape.hex(CS)) is None


def test_plot_ofExplicitShapesIntoAHeldMap(gridMap):
  g, cube = gridMap
  # the explicit construction into the held map, and a shape at a position above it
  plot(g)
  hold(True)
  h = plot(g.centroid + g.meanOrientation * cube * np.sqrt(g.area), faceColor="red")
  plot(1.5, 1.5, 1.0, g.meanOrientation[0] * cube)
  hold(False)
  assert isinstance(h, PolyCollection) and len(h.get_paths()) == 2
  assert len([c for c in axesOf().collections if isinstance(c, PolyCollection)]) == 2


def test_plotInnerFaceAndDirection_drawIntoAMap(gridMap):
  g, cube = gridMap
  plot(g)
  hold(True)
  s = g.centroid[0] + g.meanOrientation[0] * cube * 2
  plotInnerFace(s, g.meanOrientation[0] * Miller(1, 0, 0, cube.CS))
  plotInnerDirection(s, g.meanOrientation[0] * Miller(1, 0, 0, cube.CS, "uvw"))
  hold(False)
