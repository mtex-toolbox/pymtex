# # Properties of Three-Dimensional Grains
#
# [Three-Dimensional Grains](Grains3D.html) defines the `grain3d` collection and its
# polyhedral representation. Three-dimensional grains retain material properties such as
# `meanOrientation`, but their size and shape require volume and surface measures rather
# than planar area and perimeter.
#
# Start with size: how much material belongs to the large grains? Then distinguish
# elongated grains from compact ones, and identify shapes cut short by the measurement
# boundary. These questions need different measures and different statistical weights.


import matplotlib.pyplot as plt
import numpy as np

from mtex import (crystalFrame, degree, grain3d, histogram, hold, ipfColorKey, mtexColorbar, mtexdatafile, mtexTitle, newMtexFigure, nextAxis,
                  norm, plot, plotEllipsoid, plottingConvention, quiver, setCamera)

plottingConvention.default('y↑→x')

# ## Load the example microstructure
#
# The bundled data set is a Neper tessellation. The previous
# [Neper Interface](NeperInterface.html) page explains how such a collection is generated
# or imported. Loading the existing file needs no Neper installation. Its lengths are
# expressed in the tessellation coordinate units; assign a physical scale before comparing
# with an experiment.

# +
# assign trigonal quartz symmetry, as on the Neper Interface page
cs = crystalFrame.load('quartz.cif')
tessFile = mtexdatafile('my100grains')[0]
grains = grain3d.load(tessFile, CS=cs)
print(grains)

plot(grains, grains.meanOrientation, micronbar='off', edgeAlpha=0.1)
setCamera(plottingConvention.default3D)
# -

# Each colour represents one mean orientation. The faceted outlines are the boundary
# polygons from which the geometric properties are computed.

# ## Compare size measures
#
# Diameter, surface area, and volume answer different questions. The diameter spans the
# two most distant vertices. Surface sums the areas of all faces, while volume measures the
# enclosed polyhedron. Their units follow the mesh coordinates: length, length squared, and
# length cubed. A tessellation has no experimental length calibration merely because it
# can be plotted.
#
# Select by grain id so the choice remains meaningful after subsetting.

# +
grainId = 9
grain = grains['id', grainId]

print(grain.diameter)
# -

print(grain.surface())

print(grain.volume)

# Equal volumes need not mean equal elongation or equal surface area. The
# equivalent-sphere diameter provides a size measure independent of surface roughness;
# sphericity compares the measured surface with that of a sphere enclosing the same
# volume. A sphere has sphericity one.

equivalentDiameter = (6 * grains.volume / np.pi) ** (1 / 3)
sphericity = (36 * np.pi) ** (1 / 3) * grains.volume ** (2 / 3) / grains.surface()

# Face and neighbourhood counts provide complementary structural measures.

print(np.r_[grain.numFaces, grain.numNeighbors])

# A triangulated interface can contain many mesh faces but still separate only two
# grains. `numFaces` depends on the mesh; `numNeighbors` describes grain connectivity. A
# face on the outside has no neighbouring grain.

# ## Which grains are completely inside the measurement?
#
# A grain meeting the measurement hull is truncated: its measured volume is only the part
# inside the box. The zero grain id identifies the exterior and lets us find these grains
# without relying on a bounding-box tolerance. Make this selection on the full collection,
# before selecting a phase.

gB = grains.boundary
hullIds = np.unique(gB.grainId[np.any(gB.grainId == 0, axis=1)])
isInterior = ~np.isin(grains.id, hullIds)
print([len(grains), np.count_nonzero(isInterior)])

# Interior grains have complete measured shapes. Excluding hull grains also preferentially
# excludes large grains, which are more likely to meet the box. Report the selection with a
# size distribution. Below we retain the whole synthetic tessellation so that the two
# weightings use the same set.

# ## Number-weighted and volume-weighted distributions
#
# A plain numeric histogram gives every grain one count. This answers how many grains
# fall in a size interval. `histogram(grains, ...)` weights each grain by its volume and
# reports relative volume in percent. For questions about the fraction of material in an
# interval, this is the more realistic view.

# +
volumeEdges = np.linspace(0, grains.volume.max() + np.finfo(float).eps, 21)
newMtexFigure(layout=[1, 2])
histogram(grains.volume, volumeEdges, color=grains.color[0])
plt.xlabel('grain volume (length units$^3$)')
plt.ylabel('number of grains')
mtexTitle('Number weighted')

nextAxis()
histogram(grains, grains.volume, volumeEdges)
mtexTitle('Volume weighted')
# -

# Large grains are relatively inconspicuous in the count histogram but gain weight in the
# second panel. Neither weighting is universally preferable; the denominator must match the
# scientific question.
#
# The same distinction applies to any other property. Here the horizontal coordinate is
# surface area, while the weight remains grain volume.

# +
surfaceEdges = np.linspace(0, grains.surface().max() + np.finfo(float).eps, 16)
newMtexFigure(layout=[1, 2])
histogram(grains.surface(), surfaceEdges, color=grains.color[0])
plt.xlabel('surface area (length units$^2$)')
plt.ylabel('number of grains')
mtexTitle('Number weighted')

nextAxis()
histogram(grains, grains.surface(), surfaceEdges)
plt.xlabel('surface area (length units$^2$)')
mtexTitle('Volume weighted')
# -

# The right panel asks what fraction of the total material belongs to grains in each
# surface-area interval. It does not show a surface-area fraction.

# ## Relate diameter and volume
#
# A scatter plot tests how two measures covary. Taking the cube root of volume puts both
# axes in units of length. Similar grain shapes should lie near a common trend; elongated
# or irregular grains can depart from it.

# +
plt.figure()
plt.scatter(grains.volume ** (1 / 3), grains.diameter, 18, grains.color)
plt.xlabel('cube root of volume (length units)')
plt.ylabel('diameter (length units)')
# -

# The overall increase confirms that larger volumes usually have larger diameters. The
# vertical spread at a fixed cube-root volume records shape variation rather than a change
# of units.

# ## Ellipsoid-based shape
#
# `principalComponents` computes three orthogonal vectors `a`, `b`, and `c` from the
# grain's volume moments. Their directions are the principal directions, and their lengths
# are the half-axes of an ellipsoid scaled to the same volume as the grain.
# `plotEllipsoid` draws them.

# +
a, b, c = grains.principalComponents()

# compute one IPF colour for each ellipsoid
cKey = ipfColorKey(grains.CS)
color = cKey.orientation2color(grains.meanOrientation)

newMtexFigure()
plotEllipsoid(grains.centroid, a, b, c, faceColor=color)
setCamera(plottingConvention.default3D)
# -

# The ellipsoids preserve centroid, principal directions, and volume, while discarding
# individual facets. Long thin ellipsoids therefore identify anisotropic shape without
# reproducing every boundary face.

# ## Is shape anisotropy associated with grain size?
#
# The ratio of the longest to the shortest ellipsoid half-axis separates elongation from
# size. Colouring by sphericity adds a surface measure: grains can have similar aspect
# ratios yet differ in how faceted they are.

# +
axisLengths = np.column_stack([norm(a), norm(b), norm(c)])
aspectRatio = axisLengths.max(axis=1) / axisLengths.min(axis=1)
plt.figure()
h = plt.scatter(equivalentDiameter, aspectRatio, 24, sphericity)
plt.xlabel('equivalent-sphere diameter (length units)')
plt.ylabel('longest / shortest principal half-axis')
plt.colorbar(h, label='sphericity')
# -

# The ellipsoid reduces each grain to a few shape descriptors. It cannot resolve narrow
# necks or individual facets. For voxel reconstructions,
# [compare smoothing choices](Grains3DSmoothing.html) before interpreting sphericity,
# because staircase surfaces inflate the denominator.

# ## Vertices and three kinds of centre
#
# `grain.V` contains the vertices used by the selected grain. Its `centroid` is the centre
# of the enclosed volume. Each entry of `grain.boundary` is one face, and
# `grain.boundary.centroid` contains one area centroid per face.

# +
grain = grains['id', 5]

plot(grain, faceAlpha=0.5, edgeAlpha=0.2, micronbar='off')
hold(True)
plot(grain.centroid)
plot(grain.V)
plot(grain.boundary.centroid)
hold(False)
setCamera(plottingConvention.default3D)
# -

# The single interior marker is the volume centroid. Vertex markers lie on polygon corners,
# while the face-centroid markers lie within the boundary faces. These point sets describe
# different levels of the same geometry.

# ## Whole-grain property reference
#
# The main whole-grain properties are
#
# | Property | Purpose | Property | Purpose |
# | --- | --- | --- | --- |
# | `volume` | enclosed volume | `surface` | enclosing surface area |
# | `diameter` | largest vertex-to-vertex distance | `principalComponents` | volume-matched ellipsoid half-axes |
# | `centroid` | volume centroid | `V` | grain vertices |
# | `numPixel` | voxel count after reconstruction | `numFaces` | number of mesh faces |
# | `numNeighbors` | number of neighbouring grains | `boundary` | enclosing boundary faces |
#
# Several `grain2d` shape properties do not yet have three-dimensional counterparts:
# `caliper`, `equivalentRadius`, `equivalentPerimeter`, `shapeFactor`, `isBoundary`,
# `hasHole`, and `isInclusion`.

# ## Boundary-face properties
#
# A three-dimensional grain boundary stores one entry per polygonal face. Its principal
# geometric and crystallographic properties are
#
# | Property | Purpose | Property | Purpose |
# | --- | --- | --- | --- |
# | `area` | face area (length squared) | `N` | stored normal direction |
# | `diameter` | largest vertex-to-vertex distance | `perimeter` | length around the face |
# | `centroid` | area centroid | `grainId` | ids of adjacent grains |
# | `misorientation` | orientation difference across the face | `ebsdId` | adjacent voxel ids after reconstruction |
#
# The stored normal has one direction for a shared face. The
# [outward-normal example](Grains3D.html) explains how `I_GF` changes that sign for one
# chosen grain. Here `antipodal=True` deliberately treats opposite normal directions as
# equivalent and emphasizes the boundary-plane axes.

hold(True)
quiver(grain.boundary, grain.boundary.N, antipodal=True, lineWidth=2)
hold(False)

# The arrows are normal to the faces on which they start. Because they are plotted
# antipodally, this figure does not claim that every arrow points outwards.

# ## Misorientation across indexed faces
#
# An indexed face separates two grains whose mean orientations are known. Filtering with
# `'indexed'` excludes faces for which that crystallographic comparison is unavailable.
# The colour below is the misorientation angle across each remaining face. In this
# tessellation it comes from grain mean orientations; a voxel reconstruction instead stores
# the difference between the neighbouring voxel orientations.

# +
indexedBoundary = grains.boundary['indexed']
plot(indexedBoundary, indexedBoundary.misorientation.angle() / degree, micronbar='off', edgeAlpha=0.1)
setCamera(plottingConvention.default3D)
mtexColorbar(location='southoutside', title='misorientation angle (degrees)')
# -

# Faces with similar colours separate grain pairs with similar misorientation angles. The
# colour does not encode face orientation or face area, which are the separate properties
# `N` and `area`.

# ## References
#
# * R. Quey, P. R. Dawson and F. Barbe,
#   [Large-scale 3D random polycrystals for the finite element method: Generation, meshing and remeshing](https://doi.org/10.1016/j.cma.2011.01.002),
#   *Computer Methods in Applied Mechanics and Engineering* 200 (2011), 1729-1745,
#   describes the synthetic polycrystal generation behind the example tessellation.

# ## Next
#
# Continue with [Operations with Three-Dimensional Grains](Grains3DOperations.html) to
# cut, triangulate, and rotate the grains whose properties were measured here.

# ## Technical details
#
# `surface` is a method, `grains.surface()`, as `principalComponents` is; MATLAB's
# `[a, b, c] = principalComponents(grains)` is `a, b, c = grains.principalComponents()`.
# The ellipsoids are drawn on a grid of 24 by 12 rather than MATLAB's 50 by 20, which keeps
# the interactive page of a thousand ellipsoids small. MATLAB's plain `histogram`,
# `scatter` and `colorbar` are matplotlib's.
