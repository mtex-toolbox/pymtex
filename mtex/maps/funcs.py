# the verbs of the maps layer as functions: gridify(ebsd) and ebsd.gridify() are one call
from __future__ import annotations

from ..geometry.funcs import _method

gridify = _method("gridify")
gridify.__doc__ += """

Example
Arrange a small list of measurements into a grid, leaving one interior lattice site
unmeasured. The index mapping identifies where each original measurement ended up.

  import numpy as np
  from mtex import EBSD, crystalFrame, phaseList, rotation, vector3d, degree, gridify

  x, y = np.meshgrid(np.arange(5), np.arange(4))
  keep = ~((x == 2) & (y == 1))
  pos = np.c_[x[keep], y[keep]]
  rot = rotation.byAxisAngle(vector3d.Z, x[keep]*degree)
  ebsd = EBSD(pos, rot, np.ones(len(pos), dtype=int), phaseList(crystalFrame('m-3m')))
  grid, newId = gridify(ebsd, return_index=True)
  print('List shape:', ebsd.shape, 'Grid shape:', grid.shape)
  print('Unmeasured cells:', np.count_nonzero(grid.isPadding))
  print('Padding is indexed:', bool(np.any(grid.isIndexed[grid.isPadding])))
  print('Positions preserved:', np.allclose(grid.pos.data.reshape(-1, 3)[newId], ebsd.pos.data))

gridify arranges the data; it does not interpolate a measurement into the missing cell.
Use isPadding to distinguish that absent lattice site from an acquired but unindexed point.
"""
extent = _method("extent")
shift = _method("shift")
transform = _method("transform")
findByLocation = _method("findByLocation")
findByLocation.__doc__ += """

Example
Find the nearest measurement to a position and every measurement within a specified
radius. The small sample is downloaded and cached on first use.

  import numpy as np
  from mtex import mtexdata, findByLocation

  ebsd = mtexdata('small', verbose=False)['Forsterite']
  query = np.array([np.mean(ebsd.x), np.mean(ebsd.y)])
  nearest = findByLocation(ebsd, query)
  nearby = findByLocation(ebsd, query, radius=30)
  print('Query (um):', np.round(query, 2))
  print('Nearest position (um):', np.round(nearest.pos.data.reshape(-1, 3)[0, :2], 2))
  print('Measurements within 30 um:', nearby.size)

The result is an EBSD selection. Without radius the nearest measurement is returned even
when the query is outside the scan; specify a radius when distance should limit the selection.
"""
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
KAM.__doc__ += """

Example
Compute nearest-neighbour KAM within reconstructed Forsterite grains. The small
sample is downloaded and cached on first use; all displayed angles are in degrees.

  import numpy as np
  import matplotlib.pyplot as plt
  from mtex import mtexdata, calcGrains, KAM, degree, plot, hold, mtexColorbar

  ebsd = mtexdata('small', verbose=False)['Forsterite']
  grains = calcGrains(ebsd, angle=10*degree, minPixel=5)
  kam = KAM(ebsd, order=1, threshold=2.5*degree)/degree
  print('Median KAM (degrees):', round(float(np.nanmedian(kam)), 3))
  print('Pixels with no accepted neighbour:', np.count_nonzero(~np.isfinite(kam)))
  plot(ebsd, kam, colorRange=(0, 2.5))
  mtexColorbar(title='KAM (degrees)')
  hold(True)
  plot(grains.boundary, lineWidth=1)
  hold(False)
  plt.show()

calcGrains stores grain ids on the map, so comparisons across its grain boundaries are
excluded. The threshold additionally rejects individual neighbour pairs above 2.5 degrees.
KAM depends on scan spacing, neighbourhood and noise; it is not itself a strain measurement.
"""
calcGROD = _method("calcGROD")
calcGROD.__doc__ += """

Example
Compare each Forsterite orientation with the mean of its reconstructed grain.
The small sample is downloaded and cached on first use.

  import numpy as np
  import matplotlib.pyplot as plt
  from mtex import mtexdata, calcGrains, calcGROD, degree, plot, hold, mtexColorbar

  ebsd = mtexdata('small', verbose=False)['Forsterite']
  grains = calcGrains(ebsd, angle=10*degree, minPixel=5)
  grod = calcGROD(ebsd, grains)
  omega = grod.angle()/degree
  print('Median deviation from grain mean (degrees):', round(float(np.nanmedian(omega)), 3))
  plot(ebsd, omega, colorRange=(0, 5))
  mtexColorbar(title='GROD angle (degrees)')
  hold(True)
  plot(grains.boundary, lineWidth=1)
  hold(False)
  plt.show()

The result contains misorientations; angle extracts their magnitudes. GROD uses a grain
reference, while KAM uses local neighbours. Select one indexed phase before calculating GROD;
pixels without a reconstructed grain have no grain reference and give NaN.
"""
grainMean = _method("grainMean")
grainMean.__doc__ += """

Example
Average the GROD angle once per grain and map the same averages back to the pixels.
The small Forsterite sample is downloaded and cached on first use.

  import numpy as np
  import matplotlib.pyplot as plt
  from mtex import mtexdata, calcGrains, calcGROD, grainMean, degree, plot, mtexColorbar

  ebsd = mtexdata('small', verbose=False)['Forsterite']
  grains = calcGrains(ebsd, angle=10*degree, minPixel=5)
  omega = calcGROD(ebsd, grains).angle()/degree
  perGrain = grainMean(ebsd, omega, grains)
  perPixel = grainMean(ebsd, omega)
  print('Per-grain shape:', perGrain.shape, 'Per-pixel shape:', perPixel.shape)
  print('First five grain means (degrees):', np.round(perGrain[:5], 3))
  plot(grains, perGrain, colorRange=(0, 5))
  mtexColorbar(title='Mean GROD angle (degrees)')
  plt.show()

With grains supplied, the output follows grains.id; without it, each pixel receives its
grain average. Averaging the deviation angle gives an angular spread, not a mean orientation.
Use orientation.mean to average orientations with crystal symmetry.
"""
spatialProfile = _method("spatialProfile")
spatialProfile.__doc__ += """

Example
Extract a horizontal profile through the small Forsterite map and plot its
misorientation relative to the first selected orientation. The sample is cached after its first download.

  import numpy as np
  import matplotlib.pyplot as plt
  from mtex import mtexdata, spatialProfile, degree

  ebsd = mtexdata('small', verbose=False)['Forsterite']
  y = float(np.median(ebsd.y))
  line = [[float(np.min(ebsd.x)), y], [float(np.max(ebsd.x)), y]]
  profile, distance = spatialProfile(ebsd, line)
  omega = profile.orientations.angle(profile.orientations[0])/degree
  print('Selected measurements:', profile.size)
  print('Distance increases:', bool(np.all(np.diff(distance) >= 0)))
  plt.plot(distance, omega, '.')
  plt.xlabel('Distance along profile (um)')
  plt.ylabel('Misorientation to first orientation (degrees)')
  plt.show()

Distances use the scan unit and start at the first line endpoint. The default band extends
half a lattice step on either side of the line; radius changes that band. Large jumps can
mark different grains. This extracts measured pixels rather than interpolating an orientation field.
"""
calcMisorientation = _method("calcMisorientation")
calcGND = _method("calcGND")
weightedBurgersVec = _method("weightedBurgersVec")
smooth = _method("smooth")
smooth.__doc__ += """

Example
Denoise a synthetic map containing two grains and reproducible orientation noise.
Reconstruct the grains first so the filter respects their boundary.

  import numpy as np
  import matplotlib.pyplot as plt
  from mtex import (EBSD, crystalFrame, phaseList, rotation, vector3d, degree,
                    calcGrains, smooth, meanFilter, KAM, plot, mtexColorbar, mtexTitle, newMtexFigure)

  x, y = np.meshgrid(np.arange(20), np.arange(12))
  base = np.where(x < 10, 0.0, 20.0)
  noise = np.random.default_rng(7).normal(0, 0.4, x.shape)
  rot = rotation.byAxisAngle(vector3d.Z, (base + noise).ravel()*degree)
  cs = crystalFrame('m-3m')
  ebsd = EBSD(np.c_[x.ravel(), y.ravel()], rot, np.ones(x.size, dtype=int), phaseList(cs))
  grains = calcGrains(ebsd, angle=10*degree)
  filtered = smooth(ebsd, meanFilter(numNeighbours=1))
  for data, title in ((ebsd, 'Before smoothing'), (filtered, 'After smoothing')):
      newMtexFigure()
      kam = KAM(data)/degree
      print(title, 'median KAM (degrees):', round(float(np.nanmedian(kam)), 3))
      plot(data, kam, colorRange=(0, 1.5), micronbar=False)
      mtexColorbar(title='KAM (degrees)')
      mtexTitle(title)
  print('Grain ids unchanged:', np.array_equal(filtered.grainId, ebsd.grainId))
  plt.show()

The returned map has new orientations and retains the original segmentation. Lower KAM
shows reduced local variation in this synthetic example; it is not a general measure of
denoising quality. Choose a filter and length scale appropriate to the features to retain.
"""
fill = _method("fill")
fill.__doc__ += """

Example
Fill a three-by-three hole in a synthetic orientation gradient. The missing cells
exist on the grid and have phase zero and NaN rotations; no data file is needed.

  import numpy as np
  import matplotlib.pyplot as plt
  from mtex import (EBSD, crystalFrame, phaseList, rotation, vector3d, degree,
                    gridify, fill, plot, mtexColorbar, mtexTitle, newMtexFigure)

  x, y = np.meshgrid(np.arange(11), np.arange(11))
  rot = rotation.byAxisAngle(vector3d.Z, (0.3*x + 0.2*y).ravel()*degree)
  hole = ((x >= 4) & (x <= 6) & (y >= 4) & (y <= 6)).ravel()
  rot.data[hole] = np.nan
  phase = np.where(hole, 0, 1)
  ebsd = gridify(EBSD(np.c_[x.ravel(), y.ravel()], rot, phase, phaseList(crystalFrame('m-3m'))))
  filled = fill(ebsd)
  print('Indexed cells before and after:', np.count_nonzero(ebsd.isIndexed), np.count_nonzero(filled.isIndexed))
  measured = ebsd.isIndexed
  change = filled.rotations[measured].angle(ebsd.rotations[measured])/degree
  print('Largest change at measured cells (degrees):', round(float(np.max(change)), 6))
  for data, title in ((ebsd, 'Before filling'), (filled, 'After filling')):
      newMtexFigure()
      plot(data, data.rotations.angle()/degree, colorRange=(0, 5), micronbar=False)
      mtexColorbar(title='Rotation about z (degrees)')
      mtexTitle(title)
  plt.show()

Only missing orientations are solved for; the measured orientations stay fixed. The
default uses a variational continuation, whereas MATLAB fill uses nearest neighbours.
A gap connected to the outer edge remains empty unless extrapolate=True is requested.
"""
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
