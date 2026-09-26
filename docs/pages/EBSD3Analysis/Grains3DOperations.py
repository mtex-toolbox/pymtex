# # Operations with Three-Dimensional Grains
#
# The [Three-Dimensional EBSD Analysis](EBSD3Analysis.html) overview distinguishes volume
# measurements from the `grain3d` surface representation. This page shows how to cut those
# grains into a planar map, replace polygonal faces by triangles, and rotate geometry
# together with orientation.
#
# The examples use the synthetic tessellation introduced on the
# [Neper Interface](NeperInterface.html) page. The preceding
# [Properties of Three-Dimensional Grains](Grains3DProperties.html) page explains the
# volume, surface, and face properties used below.


import matplotlib.pyplot as plt
import numpy as np

from mtex import (crystalFrame, degree, grain3d, hold, mtexdatafile, mtexTitle, newMtexFigure, nextAxis, plot, plottingConvention, rotate,
                  rotation, setCamera, vector3d, zvector)

plottingConvention.default('y↑→x')

# ## Load the example microstructure

# +
# assign trigonal quartz symmetry, as on the Neper Interface page
cs = crystalFrame.load('quartz.cif')
tessFile = mtexdatafile('my100grains')[0]
grains = grain3d.load(tessFile, CS=cs)
print(grains)

plot(grains, grains.meanOrientation, micronbar='off', edgeAlpha=0.1)
setCamera(plottingConvention.default3D)
# -

# Each colour represents the mean orientation of one grain. The outer faces hide most of
# the grains inside the tessellated volume, which is why a planar section answers a
# different question from this surface view.

# ## Cut one planar section
#
# [Three-Dimensional Grains](Grains3D.html) introduces planar sectioning and the `grain2d`
# result. The direct form of `slice` used here specifies the plane by a normal `N` and any
# point `P0` in the plane.

# +
# point through which the plane passes
P0 = grains.midPoint

# plane normal
N = vector3d(1, -1, 1)

grainSlice = grains.slice(N, P0)
print(grainSlice)

plot(grainSlice, grainSlice.meanOrientation, micronbar='off')
setCamera(plottingConvention.default3D)
# -

# The slice is still drawn in the three-dimensional scene. From the default viewpoint it
# is seen obliquely. A plotting convention with `N` pointing out of the screen gives the
# face-on view a microscope would have.

sectionView = plottingConvention()
sectionView.outOfScreen = N
sectionView.north = zvector
setCamera(sectionView)

# The polygons are cuts through grains, not complete grains. A grain that is large in this
# section may occupy little volume, and a large grain can be missed when the plane does
# not intersect it.
#
# The returned `Id3d` property records the parent grain of each section polygon in the
# original collection. Use it to recover the full polyhedra that produced selected section
# polygons.

# +
parentIds = np.unique(grainSlice.Id3d)
parentGrains = grains['id', parentIds]

newMtexFigure(layout=[1, 2])
plot(grainSlice, grainSlice.meanOrientation, micronbar='off')
setCamera(sectionView)
mtexTitle('Planar sections')

nextAxis()
plot(parentGrains, parentGrains.meanOrientation, micronbar='off', edgeAlpha=0.1)
setCamera(plottingConvention.default3D)
mtexTitle('Parent grains')
# -

# The left panel contains only the section polygons. The right panel shows their parent
# polyhedra extending on both sides of the cutting plane.

# ## How representative is a grain's section size?
#
# Compare each section's equal-area-circle diameter with the equivalent-sphere diameter of
# its parent.

# +
dSection = 2 * np.sqrt(grainSlice.area / np.pi)
dParent = (6 * grains['id', grainSlice.Id3d].volume / np.pi) ** (1 / 3)
plt.figure()
plt.scatter(dParent, dSection, 24)
limit = max(dParent.max(), dSection.max())
plt.plot([0, limit], [0, limit], 'k--')
plt.axis('equal')
plt.xlabel('parent equivalent-sphere diameter (length units)')
plt.ylabel('section equal-area-circle diameter (length units)')
# -

# The dashed line marks equal diameters. The spread reflects both the cutting position and
# grain shape. It is not a calibration curve: a differently placed plane samples different
# grains, and elongated grains can have section diameters above the line.

# ## Compare several parallel sections
#
# Several slices require several calls to `slice`. Drawing horizontal cuts together shows
# how little of the volume any single section represents.

# +
newMtexFigure()
N = vector3d.Z
z = grains.V.z
zLevels = np.linspace(z.min(), z.max(), 7)
for k in zLevels[1:-1]:
  grainSlice = grains.slice(N, vector3d(0, 0, k))
  plot(grainSlice, grainSlice.meanOrientation)
  hold(True)
hold(False)
setCamera(plottingConvention.default3D)
# -

# Follow one colour from slice to slice. A grain that is large in one section may be absent
# from the next.

# ## Triangulate polygonal faces
#
# The faces of these grains are polygons with many vertices. Some computations are much
# faster on triangles. `triangulate` returns equivalent grains whose polygonal faces have
# been divided into triangles.

# +
selectedGrains = grains[19:21]
grainsTri = selectedGrains.triangulate()
print(grainsTri)

print([selectedGrains.numFaces.sum(), grainsTri.numFaces.sum()])

plot(grainsTri, grainsTri.meanOrientation, micronbar='off', edgeAlpha=0.1)
setCamera(plottingConvention.default3D)
# -

# The face count increases while the displayed shape is unchanged: the new triangles cover
# the same boundary polygons. Triangulation changes the mesh representation, not the
# physical grains or their stored mean orientations.

# ## Rotate geometry and orientation together
#
# `rotate` turns grains in space. By default it rotates both things a grain carries: its
# shape and its orientation. Rotating only one of them would describe a different specimen
# rather than the same specimen seen differently.

# +
rot = rotation.byAxisAngle(vector3d(1, 1, 1), 30 * degree)
grainsRotated = rot * grains

plot(grainsRotated, grainsRotated.meanOrientation, micronbar='off', edgeAlpha=0.1)
setCamera(plottingConvention.default3D)
# -

# The shape has turned about the coordinate origin, and the colours change. An IPF colour
# says which crystal direction points along a fixed specimen axis. The same rotation is
# applied to all orientations; symmetry-reduced orientation differences need not equal
# that rotation angle. A rigid rotation preserves the relation between grains, while
# changing their relation to the coordinate axes.
#
# The method form provides a `center` option when the spatial rotation should use a point
# other than the origin.

rotationCenter = grains.midPoint
grainsAboutCenter = rotate(grains, rot, center=rotationCenter)

# ## Rotate only one part of the data
#
# Two flags deliberately decouple geometry from orientation. The names say which values
# are kept fixed, not which values are rotated.
#
# | Flag | Geometry | Mean orientation |
# | --- | --- | --- |
# | `'keepEuler'` | rotated | unchanged |
# | `'keepXY'` | unchanged | rotated |
#
# For example, the first command below turns the vertices while retaining the orientation
# values. The second changes the orientations while retaining the vertices.

geometryOnly = rotate(grains, rot, 'keepEuler')
orientationOnly = rotate(grains, rot, 'keepXY')

# These flags are useful when geometry and orientation require separate corrections. They
# do not represent a rigid rotation of the whole specimen. The `center` option affects only
# a spatial rotation, so it has no effect when `'keepXY'` leaves the geometry unchanged.

# ## Function reference
#
# | Function | Purpose | Function | Purpose |
# | --- | --- | --- | --- |
# | `slice` | cut planar grain polygons | `intersected` | find grains crossed by a plane |
# | `triangulate` | divide polygonal faces into triangles | `rotate` | rotate geometry and orientations |
# | `neighbors` | list adjacent grain ids | `id2ind` | map persistent ids to array positions |

# ## References
#
# * R. Quey, P. R. Dawson and F. Barbe,
#   [Large-scale 3D random polycrystals for the finite element method: Generation, meshing and remeshing](https://doi.org/10.1016/j.cma.2011.01.002),
#   *Computer Methods in Applied Mechanics and Engineering* 200 (2011), 1729-1745,
#   describes the synthetic polycrystal construction used for the example tessellation.

# ## Technical details
#
# `Id3d` holds the ids of the parent grains, so the parents are `grains['id', Id3d]`;
# MATLAB's text calls it an array position, which in an unfiltered collection is the same
# number. The loop over parallel sections needs no `rmappdata`, the interactive selection
# of MATLAB's figure. MATLAB's `grains(20:21)` is `grains[19:21]`.
