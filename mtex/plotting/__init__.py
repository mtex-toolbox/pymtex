# the plotting layer: the geometry of a plot in NumPy, matplotlib imported only when a plot
# is drawn; the plot commands are functions here and methods on the objects they draw
from .spherical import (sphericalProjection, eareaProjection, eangleProjection, edistProjection, orthographicProjection,
                        gnomonicProjection, plainProjection, makeSphericalProjection, plotSector, rhoRange, isUpper,
                        isLower, millerLabel, plotS2Grid)
from .colormaps import (mtexColorMap, contourLevels, str2rgb, ind2color, num2rgb, blue2redColorMap, red2blueColorMap,
                        LaboTeXColorMap, white2blackColorMap, black2whiteColorMap, WhiteJetColorMap, vega20)
from .mtexfigure import (saveFigure, logAxis, mtexFigure, newMtexFigure, gcm, hold, legend, nextAxis, mtexTitle, xlabel,
                         ylabel, mtexColorbar, setColorRange, setColorScale)
from .sphericalplot import (plotHKL, plotSymmetry, sphericalPlot, plot, scatter, text, annotate, contourf, contour,
                            pcolor, quiver, plotRegion, plotKernel, circle, ellipse, plotKey)
from .sections import (ODFSections, eulerSections, phi1Sections, phi2Sections, alphaSections, gammaSections, pfSections,
                       sigmaSections, axisAngleSections, newODFSectionPlot, sectionIndex, sectionTolerance)
from .sectionplot import (plotKeySections, plotSection, plotSections, annotateSection, plotFibreSection, quiverSection,
                          quiverSectionField)
from .fibreplot import plotFibre
from .sectorfold import regionCenter, sectorFolds, hasSmoothKey, polarCoordinates
from .directionkeys import (directionColorKey, HSVDirectionKey, TSLDirectionKey, HKLDirectionKey, spotDirectionKey,
                            sphereColor, hsv2rgb, rgb2hsv, hsl2hsv)
from .orientationkeys import (orientationColorKey, ipfColorKey, ipfHSVKey, ipfTSLKey, ipfHKLKey, ipfSpotKey,
                              axisAngleColorKey, BungeColorKey, spotColorKey, spectralTransmissionColorKey, colorGrid,
                              ipfColors)
from .patalakey import PatalaColorKey
from .maps import (mapLayout, mapRaster, mapCells, mapPoints, mapExtent, boundaryPolyline, boundarySegments,
                   grainPolygons, scaleBar, switchUnit, triadGeometry, frameOnScreen, phaseColor, phasePalette)
from .mapplot import (line, scatterXY, mapPlot, plotEBSD, plotGrains, plotBoundary, plotTriplePoints, quiverMap,
                      scaleBarOn, phaseLegend, keyLegend, keyResolution, mapValues, mapAliases)
from .poles import poles, inversePoles, dynamicMarkerSize
from .polefigureplot import plotPDF, plotIPDF, plotPoleFigure, plotDiff
from .space import (sphereMesh, sphereGridLines, smallCircle, frameAxes, axisAnglePoints, maxAngles, regionSurface,
                    regionWalls, regionLines, visibleFaces, shapeOutlines, shapesOnMap)
from .spaceplot import plot3d, scatter3d, spacePlot, arrow3d, quiver3, surf, setCamera
from .shapeplot import plotInnerFace, plotInnerDirection, plotSlipSystem, faceColors, plotPlane
from .tensorplot import plotSeismicVelocities, plotRGB
from .planarkey import planarColorKey
from .polarplot import setView, polarAxes
from .grainplots import hist, histogram, plotEllipse
from .kernelplot import plotSO3Kernel, plotSpektra
from .distributionplot import plotAngleDistribution, plotAxisDistribution
from .scene3d import (scene3d, mesh3d, slices3d, picture3d, lines3d, arrows3d, points3d, ellipsoids3d, grainMesh,
                      boundaryMesh, volumeSlices)
from .render3d import hasPyVista, pyvistaPlotter, renderImage, drawMatplotlib, writeHTML, sceneHTML
from .volumeplot import (plotGrains3d, plotBoundary3d, plotEBSD3, show3d, exportHTML, is3dMap, plot3dMap, gcs,
                         mapPicture, quiver3dMap, quiverVectors, plotEllipsoid, plotPoints3d, scatter3)

__all__ = ["sphericalProjection", "eareaProjection", "eangleProjection", "edistProjection", "orthographicProjection",
           "gnomonicProjection", "plainProjection", "makeSphericalProjection", "plotRegion", "plotSector", "rhoRange",
           "isUpper", "isLower", "millerLabel", "plotS2Grid", "mtexColorMap", "contourLevels", "str2rgb", "ind2color",
           "num2rgb", "blue2redColorMap", "red2blueColorMap", "LaboTeXColorMap", "white2blackColorMap",
           "black2whiteColorMap", "WhiteJetColorMap", "vega20", "saveFigure", "logAxis", "mtexFigure", "newMtexFigure",
           "gcm", "hold", "legend", "nextAxis", "mtexTitle", "xlabel", "ylabel", "mtexColorbar", "setColorRange",
           "setColorScale", "plotHKL", "plotSymmetry", "sphericalPlot", "plot", "scatter", "text", "annotate",
           "contourf", "contour", "pcolor", "quiver", "plotKernel", "circle", "ellipse", "plotPDF", "plotIPDF",
           "plotKey", "ODFSections", "eulerSections", "phi1Sections", "phi2Sections", "alphaSections", "gammaSections",
           "pfSections", "sigmaSections", "axisAngleSections", "newODFSectionPlot", "sectionIndex", "sectionTolerance",
           "plotKeySections", "plotSection", "plotSections", "annotateSection", "plotFibreSection", "quiverSection",
           "quiverSectionField", "plotFibre", "regionCenter", "sectorFolds", "hasSmoothKey", "polarCoordinates",
           "directionColorKey", "HSVDirectionKey", "TSLDirectionKey", "HKLDirectionKey", "spotDirectionKey",
           "sphereColor", "hsv2rgb", "rgb2hsv", "hsl2hsv", "orientationColorKey", "ipfColorKey", "ipfHSVKey",
           "ipfTSLKey", "ipfHKLKey", "ipfSpotKey", "axisAngleColorKey", "BungeColorKey", "spotColorKey",
           "spectralTransmissionColorKey", "colorGrid", "ipfColors", "PatalaColorKey", "mapLayout", "mapRaster",
           "mapCells", "mapPoints", "mapExtent", "boundaryPolyline", "boundarySegments", "grainPolygons", "scaleBar",
           "switchUnit", "triadGeometry", "frameOnScreen", "phaseColor", "phasePalette", "line", "scatterXY", "mapPlot",
           "plotEBSD", "plotGrains", "plotBoundary", "plotTriplePoints", "quiverMap", "scaleBarOn", "phaseLegend",
           "keyLegend", "keyResolution", "mapValues", "mapAliases", "poles", "inversePoles", "dynamicMarkerSize",
           "plotPoleFigure", "plotDiff", "sphereMesh", "sphereGridLines", "smallCircle", "frameAxes", "axisAnglePoints",
           "maxAngles", "regionSurface", "regionWalls", "regionLines", "visibleFaces", "shapeOutlines", "shapesOnMap",
           "plot3d", "scatter3d", "spacePlot", "arrow3d", "quiver3", "surf", "setCamera", "plotInnerFace",
           "plotInnerDirection", "plotSlipSystem", "faceColors", "plotPlane", "plotSeismicVelocities", "plotRGB",
           "planarColorKey", "setView", "polarAxes", "hist", "histogram", "plotEllipse", "plotSO3Kernel", "plotSpektra",
           "plotAngleDistribution", "plotAxisDistribution", "scene3d", "mesh3d", "slices3d", "picture3d", "lines3d",
           "arrows3d", "points3d", "ellipsoids3d", "grainMesh", "boundaryMesh", "volumeSlices", "hasPyVista",
           "pyvistaPlotter", "renderImage", "drawMatplotlib", "writeHTML", "sceneHTML", "plotGrains3d",
           "plotBoundary3d", "plotEBSD3", "show3d", "exportHTML", "is3dMap", "plot3dMap", "gcs", "mapPicture",
           "quiver3dMap", "quiverVectors", "plotEllipsoid", "plotPoints3d", "scatter3"]


# plot(v) and v.plot() are one call: the commands installed on the classes they draw
def _install():
  from ..functions.bingham import S2FunBingham, SO3FunBingham
  from ..functions.handles import S2FunHandle, S2FunSum, SO3FunGeneric
  from ..functions.s2funtri import S2FunTri
  from ..functions.rbf import SO3FunCBF, SO3FunRBF
  from ..functions.s2axisfield import S2AxisField
  from ..functions.s2fun import S2FunHarmonic
  from ..functions.s2vectorfield import S2VectorField
  from ..functions.so3fun import SO3FunHarmonic
  from ..geometry.crystalshape import crystalShape
  from ..geometry.plane import plane3d
  from ..geometry.orientation import orientation
  from ..geometry.sphericalregion import sphericalRegion
  from ..maps.boundary import grainBoundary
  from ..maps.ebsd import EBSD
  from ..maps.grains import grain2d
  from ..functions.polefigure import PoleFigure
  from ..tensors.tensor import stiffnessTensor, tensor
  from . import tensorplot as tp
  from . import polefigureplot as pfp
  from . import spaceplot as spp
  from . import shapeplot as shp
  from ..geometry.orientationregion import orientationRegion
  from ..geometry.vector3d import vector3d
  from . import sphericalplot as sp
  from .sectionplot import plotSection
  from ..geometry.install import install

  for f in (sp.plot, sp.scatter, sp.text, sp.contourf, sp.contour, sp.pcolor, sp.quiver):
    install(vector3d, f)
  install(vector3d, sp.pcolor, "smooth")
  for f in (sp.plot, sp.contourf, sp.contour, sp.pcolor):
    install((S2FunHarmonic, S2FunHandle, S2FunSum, S2FunBingham, S2FunTri), f)
  for f in (sp.plot, sp.quiver):
    install((S2VectorField, S2AxisField), f)
  for f in (sp.plotPDF, sp.plotIPDF, plotSection):
    install((SO3FunHarmonic, SO3FunGeneric, SO3FunBingham, SO3FunRBF, SO3FunCBF), f)
  install(orientation, plotSection)
  install(sphericalRegion, sp.plotRegion, "plot")
  from . import mapplot as mp
  for cls, f in ((EBSD, mp.plotEBSD), (grain2d, mp.plotGrains), (grainBoundary, mp.plotBoundary)):
    install(cls, sp.plot, declare=f)
  install((EBSD, grain2d, grainBoundary), sp.quiver, declare=mp.quiverMap)
  for f in (pfp.plotPDF, pfp.plotIPDF):
    install(orientation, f)
  install(PoleFigure, pfp.plotPoleFigure, "plot")
  install(tensor, sp.plot)
  install(stiffnessTensor, tp.plotSeismicVelocities)
  install(PoleFigure, pfp.plotDiff)
  install(vector3d, spp.plot3d)
  install(vector3d, spp.scatter3d)
  install((S2FunHarmonic, S2FunHandle, S2FunSum, S2FunBingham, S2VectorField, S2AxisField), spp.plot3d)
  from ..geometry.quaternion import rotation
  from ..geometry.tangent import SO3TangentVector
  install((rotation, orientation), spp.plot3d)
  install((rotation, orientation), spp.scatterOrientations, "scatter")
  install(vector3d, spp.arrow3d)
  from ..geometry.frames import referenceFrame
  install(referenceFrame, sp.plotSymmetry, "plot")
  install(referenceFrame, sp.plotHKL)
  install(SO3TangentVector, spp.quiver3)
  for name in ("plot", "plot3d"):
    install(orientationRegion, spp.plotRegion3d, name)
  install(crystalShape, shp.plotShape, "plot")
  from ..maps.mapimage import mapImage, mapImageList
  from .mapplot import plotMapImage
  install((mapImage, mapImageList), plotMapImage, "plot")
  from ..maps.shape import shape2d
  from ..functions.s1fun import S1FunHarmonic
  from . import polarplot as pp
  from . import grainplots as gp
  install(shape2d, pp.plotShape, "plot")
  install(S1FunHarmonic, pp.plotS1, "plot")
  install(grain2d, gp.hist)
  install((grain2d, vector3d), gp.histogram)
  install(plane3d, shp.plotPlane, "plot")
  for f in (shp.plotInnerFace, shp.plotInnerDirection, shp.plotSlipSystem):
    install(crystalShape, f)
  from ..geometry.plottingconvention import plottingConvention
  install(plottingConvention, pp.setView)
  from ..maps3d.ebsd3 import EBSD3
  from ..maps3d.grain3boundary import grain3Boundary
  from ..maps3d.grain3d import grain3d
  from . import volumeplot as vp
  install(EBSD3, vp.plotEBSD3, "plot")
  install(grain3d, vp.plotGrains3d, "plot")
  install(grain3Boundary, vp.plotBoundary3d, "plot")
  install((grain3d, grain3Boundary), vp.quiver3dMap, "quiver")
  install(grain3d, gp.histogram)


_install()
