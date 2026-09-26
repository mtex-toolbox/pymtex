# the plotting layer: the geometry of a plot in NumPy, matplotlib imported only when a plot
# is drawn; the plot commands are functions here and methods on the objects they draw
import importlib as _importlib

__all__ = []
for _name in ("spherical", "colormaps", "mtexfigure", "sphericalplot", "sections", "sectionplot", "fibreplot", "sectorfold", "directionkeys",
              "orientationkeys", "maps", "mapplot", "poles", "polefigureplot", "space", "spaceplot", "shapeplot", "tensorplot", "planarkey", "polarplot", "grainplots", "kernelplot", "distributionplot",
              "scene3d", "render3d", "volumeplot"):
  _module = _importlib.import_module(f"{__name__}.{_name}")
  for _public in _module.__all__:
    globals()[_public] = getattr(_module, _public)
  __all__ += [n for n in _module.__all__ if n not in __all__]


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

  def method(f):
    def m(self, *args, **kwargs):
      return f(self, *args, **kwargs)
    m.__name__, m.__doc__ = f.__name__, f.__doc__
    return m

  for cls in (vector3d,):
    for f in (sp.plot, sp.scatter, sp.text, sp.contourf, sp.contour, sp.pcolor, sp.quiver):
      setattr(cls, f.__name__, method(f))
    setattr(cls, "smooth", method(sp.pcolor))
  for cls in (S2FunHarmonic, S2FunHandle, S2FunSum, S2FunBingham, S2FunTri):
    for f in (sp.plot, sp.contourf, sp.contour, sp.pcolor):
      setattr(cls, f.__name__, method(f))
  for cls in (S2VectorField, S2AxisField):
    for f in (sp.plot, sp.quiver):
      setattr(cls, f.__name__, method(f))
  for cls in (SO3FunHarmonic, SO3FunGeneric, SO3FunBingham, SO3FunRBF, SO3FunCBF):
    for f in (sp.plotPDF, sp.plotIPDF, plotSection):
      setattr(cls, f.__name__, method(f))
  orientation.plotSection = method(plotSection)
  sphericalRegion.plot = method(sp.plotRegion)
  for cls in (EBSD, grain2d, grainBoundary):
    for f in (sp.plot, sp.quiver):
      setattr(cls, f.__name__, method(f))
  for f in (pfp.plotPDF, pfp.plotIPDF):
    setattr(orientation, f.__name__, method(f))
  PoleFigure.plot = method(pfp.plotPoleFigure)
  tensor.plot = method(sp.plot)
  stiffnessTensor.plotSeismicVelocities = method(tp.plotSeismicVelocities)
  PoleFigure.plotDiff = method(pfp.plotDiff)
  vector3d.plot3d = method(spp.plot3d)
  vector3d.scatter3d = method(spp.scatter3d)
  for cls in (S2FunHarmonic, S2FunHandle, S2FunSum, S2FunBingham, S2VectorField, S2AxisField):
    cls.plot3d = method(spp.plot3d)
  from ..geometry.quaternion import rotation
  from ..geometry.tangent import SO3TangentVector
  for cls in (rotation, orientation):
    cls.plot3d = method(spp.plot3d)
    cls.scatter = method(spp.scatterOrientations)
  vector3d.arrow3d = method(spp.arrow3d)
  from ..geometry.frames import referenceFrame
  referenceFrame.plot = method(sp.plotSymmetry)
  referenceFrame.plotHKL = method(sp.plotHKL)
  SO3TangentVector.quiver3 = method(spp.quiver3)
  orientationRegion.plot = method(spp.plotRegion3d)
  orientationRegion.plot3d = method(spp.plotRegion3d)
  crystalShape.plot = method(shp.plotShape)
  from ..maps.mapimage import mapImage, mapImageList
  from .mapplot import plotMapImage
  mapImage.plot = method(plotMapImage)
  mapImageList.plot = method(plotMapImage)
  from ..maps.shape import shape2d
  from ..functions.s1fun import S1FunHarmonic
  from . import polarplot as pp
  from . import grainplots as gp
  shape2d.plot = method(pp.plotShape)
  S1FunHarmonic.plot = method(pp.plotS1)
  grain2d.hist = method(gp.hist)
  grain2d.histogram = method(gp.histogram)
  vector3d.histogram = method(gp.histogram)
  plane3d.plot = method(shp.plotPlane)
  for f in (shp.plotInnerFace, shp.plotInnerDirection, shp.plotSlipSystem):
    setattr(crystalShape, f.__name__, method(f))
  from ..geometry.plottingconvention import plottingConvention
  plottingConvention.setView = method(pp.setView)
  from ..maps3d.ebsd3 import EBSD3
  from ..maps3d.grain3boundary import grain3Boundary
  from ..maps3d.grain3d import grain3d
  from . import volumeplot as vp
  for cls in (grain3d, grain3Boundary, EBSD3):
    cls.plot = method(vp.plot3dMap)
  for cls in (grain3d, grain3Boundary):
    cls.quiver = method(vp.quiver3dMap)
  grain3d.histogram = method(gp.histogram)


_install()
