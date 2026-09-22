# # Vector Operations
#
# Three-dimensional vectors can be added, scaled, compared, and combined. Every operation
# works on a whole list at once, so a loop over vectors is usually unnecessary in MTEX.
#
# This page assumes the construction methods from
# [Defining Three Dimensional Vectors](VectorDefinition.html). A reference frame is the
# coordinate system in which the vectors are expressed. The plotting convention below only
# lays that frame out on screen; it does not change the vectors.

import numpy as np

from mtex import angle, cat, cross, degree, dot, mean, mtexdatafile, norm, normalize, plottingConvention, scatter, vector3d

plottingConvention.default('y↑→x')

# ## Building New Vectors
#
# Sums and multiples of the specimen axes `vector3d.X`, `vector3d.Y`, and `vector3d.Z` are
# `vector3d` objects again.

v = vector3d.X + 2 * vector3d.Y
v

# [+](vector3d.plus.html) and [-](vector3d.minus.html) add or subtract Cartesian
# components. [*](vector3d.mtimes.html) scales a vector by a scalar. The inner product
# [dot](vector3d.dot.html) and cross product [cross](vector3d.cross.html) have their usual
# linear-algebra meanings. The order of the cross product matters.

u = dot(v, vector3d.Y) * vector3d.Y + 2 * cross(v, vector3d.Z)
u

# The first term is $2\vec Y$, while the second is $4\vec X-2\vec Y$. Their Y components
# cancel, leaving $\vec u=4\vec X$. MTEX draws the original vector in the XY plane and the
# result along positive X as arrows in space with `arrow3d`; the three dimensional plots
# are not ported yet, so the picture is left out here.

# ## Angles
#
# The [angle](vector3d.angle.html) between two directions is a central measurement in
# texture analysis. One example is the tilt of a lattice-plane normal away from the sheet
# normal. The angle between complete crystal orientations is a different operation,
# introduced in [Misorientations](MisorientationTheory.html).

angle(vector3d.X, vector3d.Y) / degree

# MTEX returns angles in radians, hence the division by `degree`. The angle between
# directed vectors lies between $0$ and $180^\circ$. If the data represent axes instead,
# the answer is never obtuse; see [Axes and Antipodal Symmetry](VectorsAxes.html).

# ## Length and Normalization
#
# [norm](vector3d.norm.html) returns the length of each vector.
# [normalize](vector3d.normalize.html) divides each vector by its length.

norm(u)

# ---

u = normalize(u)
u

# Normalization does not change where a vector points, so it does not move its position
# in a spherical plot. It matters in calculations: the `dot` of two unit vectors is the
# cosine of their angle.

dot(normalize(v), vector3d.Y)

# A zero vector has no direction. Consequently, `normalize(vector3d(0,0,0))` has undefined,
# NaN components.

# ## Lists of Vectors
#
# `cat` joins vectors into one list, and an index selects an entry, the first one being
# `w[0]`. The general rules are covered in [Lists and Indexing](ListsAndIndexing.html).

w = cat(v, u)
w[0]

# Arithmetic on a list is entry by entry. Adding the single vector `v` to the two-entry
# list `w` adds it to both entries.

w = w + v
w

# When both inputs are lists of the same size, binary operations pair their entries by
# position. Use [angle_outer](vector3d.angle_outer.html) or
# [dot_outer](vector3d.dot_outer.html) to compare every entry of one list with every entry
# of another.

# ## Selecting from a List
#
# The following file contains directions stored as polar and azimuth angles. Its displayed
# summary confirms that the resulting `vector3d` list has 1,000 entries.

fname = mtexdatafile('vectors')
v = vector3d.load(fname, columnNames=['polar angle', 'azimuth angle'])
v

# A logical condition retains only directions with a polar angle below $60^\circ$.

selected = v.theta < 60 * degree
scatter(v[selected], grid=True)

# The empty outer ring shows that every plotted direction lies within $60^\circ$ of the Z
# axis. Count the omitted entries directly.

numOmitted = np.sum(~selected)
numOmitted

# Thus, 236 of the 1,000 directions lie at least $60^\circ$ away from the Z axis.

# ## Averaging a List
#
# [mean](vector3d.mean.html) averages the Cartesian components. The result is a mean
# vector, which generally is not a unit vector.

m = mean(v)
m

# Normalizing it gives the mean direction.

meanDirection = normalize(m)
meanDirection

# Because every input vector has unit length, the length of `m` is the mean resultant
# length.

meanResultantLength = norm(m)
meanResultantLength

# Here it is about 0.719. Identical unit directions give one; dispersed or mutually
# cancelling directions give a shorter result. Normalize unequal input vectors first when
# this directional statistic is intended.
#
# Opposite directed observations cancel. For axes, where `v` and `-v` mean the same thing,
# use `mean(v, antipodal=True)` instead; see
# [Axes and Antipodal Symmetry](VectorsAxes.html).

# ## Operations at a Glance
#
# These methods operate elementwise unless their description says otherwise.
#
# | | |
# |---|---|
# | [angle(v1,v2)](vector3d.angle.html) | pointwise angle between vectors |
# | [angle_outer(v1,v2)](vector3d.angle_outer.html) | all pairwise angles |
# | [dot(v1,v2)](vector3d.dot.html) | pointwise inner product |
# | [dot_outer(v1,v2)](vector3d.dot_outer.html) | all pairwise inner products |
# | [cross(v1,v2)](vector3d.cross.html) | pointwise cross product |
# | [a*v](vector3d.mtimes.html) | multiplication by a scalar |
# | [a*v](vector3d.times.html) | componentwise scaling by an array of the same size |
# | [norm(v)](vector3d.norm.html) | length of every vector |
# | [normalize(v)](vector3d.normalize.html) | length scaled to one |
# | [orth(v)](vector3d.orth.html) | an arbitrary orthogonal unit vector |
# | [orthProj(v,N)](vector3d.orthProj.html) | component orthogonal to `N` |
# | [perp(v)](vector3d.perp.html) | best-fit direction orthogonal to a list |
# | [v.sum()](vector3d.sum.html) | componentwise sum over a list |
# | [mean(v)](vector3d.mean.html) | mean vector, or mean axis for antipodal data |
# | [polar(v)](vector3d.polar.html) | polar angle, azimuth, and length |
# | [rotate(v,rot)](vector3d.rotate.html) | turn by a rotation |

# ## Further Reading
#
# * N. I. Fisher, T. Lewis, and B. J. J. Embleton,
#   [Descriptive and ancillary methods, and sampling problems](https://doi.org/10.1017/CBO9780511623059.004),
#   in _Statistical Analysis of Spherical Data_, Cambridge University Press, 1987, treats
#   mean directions, resultant lengths, and the difference between vectorial and axial
#   data.

# ## Next
#
# [Density Estimation](VectorsDensityEstimation.html) replaces a long list of directions
# by a smooth function on the sphere. Read [Axes and Antipodal Symmetry](VectorsAxes.html)
# before analysing data whose two signs are physically equivalent. Turning a direction into
# another one is the job of a [rotation](Rotations.html).
