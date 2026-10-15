# the verbs of the maps layer as functions: gridify(ebsd) and ebsd.gridify() are one call
from __future__ import annotations

from ..geometry.funcs import _method

gridify = _method("gridify")
extent = _method("extent")
shift = _method("shift")
transform = _method("transform")
findByLocation = _method("findByLocation")
findByOrientation = _method("findByOrientation")
inpolygon = _method("inpolygon")
reduceBoundary = _method("reduceBoundary")
simplifyBoundary = _method("simplifyBoundary")
flip = _method("flip")
refineBoundary = _method("refineBoundary")
smoothBoundary = _method("smoothBoundary")
smoothBoundary.__doc__ += """

Example
Compare reconstructed and smoothed boundaries on the small Forsterite sample.
The sample file is downloaded and cached on first use.

  import matplotlib.pyplot as plt
  from mtex import mtexdata, calcGrains, smoothBoundary, degree, plot, hold

  ebsd = mtexdata('small', verbose=False)['Forsterite']
  grains = calcGrains(ebsd, angle=10*degree, minPixel=5)
  smoothed = smoothBoundary(grains, 5)
  plot(grains.boundary, lineColor='black', lineWidth=2, displayName='Reconstructed')
  hold(True)
  plot(smoothed.boundary, lineColor='red', lineWidth=1, displayName='Smoothed')
  hold(False)
  print('Grains before and after:', grains.size, smoothed.size)
  plt.show()

Black shows the pixel-derived outlines and red the smoothed geometry. Smoothing changes
the curves, while the pixel grain ids and the grain membership remain those of the reconstruction.
"""
principalComponents = _method("principalComponents")
fitEllipse = _method("fitEllipse")
longAxis = _method("longAxis")
shortAxis = _method("shortAxis")
aspectRatio = _method("aspectRatio")
paris = _method("paris")
shapeFactor = _method("shapeFactor")
convexity = _method("convexity")
caliper = _method("caliper")
paror = _method("paror")
surfor = _method("surfor")
characteristicShape = _method("characteristicShape")
hull = _method("hull")
gradient = _method("gradient")
gradientX = _method("gradientX")
gradientY = _method("gradientY")
gradientZ = _method("gradientZ")
curvature = _method("curvature")
KAM = _method("KAM")
calcGROD = _method("calcGROD")
grainMean = _method("grainMean")
spatialProfile = _method("spatialProfile")
calcMisorientation = _method("calcMisorientation")
calcGND = _method("calcGND")
weightedBurgersVec = _method("weightedBurgersVec")
smooth = _method("smooth")
fill = _method("fill")
fillByGrainId = _method("fillByGrainId")
reduce = _method("reduce")
subsample = _method("subsample")
interp = _method("interp")
pos2ind = _method("pos2ind")
pixelSizeMatch = _method("pixelSizeMatch")
calcDistortion = _method("calcDistortion")
undistort = _method("undistort")
setOptions = _method("setOptions")

__all__ = ["gridify", "extent", "shift", "transform", "findByLocation", "findByOrientation", "inpolygon", "reduceBoundary", "flip", "simplifyBoundary", "refineBoundary", "smoothBoundary", "principalComponents", "fitEllipse", "longAxis", "shortAxis", "aspectRatio", "paris", "shapeFactor", "convexity", "caliper", "paror", "surfor", "characteristicShape", "hull", "gradient", "gradientX", "gradientY", "gradientZ", "curvature", "KAM", "calcGROD", "grainMean", "spatialProfile", "calcMisorientation", "calcGND", "weightedBurgersVec", "smooth", "fill", "fillByGrainId", "reduce", "subsample", "interp", "pos2ind",
           "pixelSizeMatch", "calcDistortion", "undistort", "setOptions"]
