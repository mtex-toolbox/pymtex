# # Clustering directions and orientations
#
# A cluster is a group of observations that lie close together in the space of the data.
# Clustering assigns each observation to a group without requiring the groups to be named in
# advance.
#
# MTEX can cluster directions and orientations while respecting their geometry and symmetry.
# This is different from [density estimation](DensityEstimation.html), which constructs a
# continuous density from discrete observations. Here the result is one integer label per
# observation and one representative center per cluster.
#
# ## Cluster directional data
#
# We start with directions sampled from a squared smiley function. Squaring the function
# concentrates the sample around its eyes and mouth while retaining points between those
# features.

# +
import numpy as np

from mtex import *

plottingConvention.default('y↑→x')
rng = np.random.default_rng(1)
S2F = S2Fun.smiley() ** 2
v = discreteSample(S2F, 10000, rng=rng)

scatter(v, markerAlpha=0.2, markerSize=2)
# -

# ## Read the cluster result
#
# For directions, [calcCluster](vector3d.calcCluster.html) uses CLASSIX by default. The array
# `cInd` has one integer cluster label for every input direction, counted from zero. The
# entries of `center` are the corresponding mean directions.
#
# Cluster labels are identifiers, not ranks: cluster 0 is not intrinsically more important
# than cluster 1. We use the labels only to assign colors, `ind2color` counting from zero as
# well, and then mark each returned center.

# +
cInd, center = calcCluster(v)

plot(v, ind2color(cInd), markerAlpha=0.2, markerSize=2)
annotate(center)
# -

# ## Tune the grouping
#
# The colored regions follow the separated strokes of the smiley, while the centers sit inside
# the densest parts of those regions. CLASSIX first forms local groups and then merges nearby
# groups into clusters.
#
# Two options control how readily this happens. Increasing `radius` makes the local groups
# coarser and usually produces fewer clusters. The `minPoints` option sets the size below
# which tiny groups are merged into neighboring clusters. If the automatic result is too
# fragmented, increase `radius`; if isolated specks survive, increase `minPoints`. The port
# estimates the radius itself, from the distances to the tenth nearest neighbours, so no
# further package is needed.
#
# ## Respect antipodal symmetry
#
# Antipodal directions identify a direction `v` with its opposite `-v`. Setting the
# `antipodal` property tells MTEX to measure proximity with this equivalence in mind.
#
# The smiley sample on its own cannot show this. No feature of it lies opposite another, so
# identifying `v` with `-v` has nothing to merge. Add the opposite of every sampled direction,
# and each feature gains one.

# +
vSym = cat(v, -v)

cInd, center = calcCluster(vSym)

plot(vSym, ind2color(cInd))
annotate(center)
# -

# The two discs are the upper and the lower hemisphere. Read as ordinary directions, the
# sample falls into eight clusters. Every feature of the smiley and every one of their
# opposites carries a color of its own.

# +
vSym.antipodal = True
cInd, center = calcCluster(vSym)

plot(vSym, ind2color(cInd), markerAlpha=0.2, markerSize=2)
annotate(center)
# -

# One disc is left, and four clusters, each exactly twice the size of before. Every feature
# has been merged with the one opposite it. The two eyes stay apart, because one eye is not
# the opposite of the other.
#
# The blue cluster reaches the rim of the disc at the top and at the bottom. Those are
# opposite directions, and they now share a label. This is a change in the meaning of a
# direction, not merely a change in how the same labels are drawn.
#
# ## Build an orientation example
#
# The same idea applies in orientation space. We construct an orientation distribution
# function (ODF) with a broad fibre component and a compact unimodal component.
# [Kernel selection](OptimalKernel.html) explains how the halfwidth controls smoothing when an
# ODF is estimated from observations.

# +
cs = crystalFrame('432')
odf = (0.7 * fibreODF(fibre.gamma(cs), halfwidth=10 * degree)
       + 0.3 * unimodalODF(orientation.byEuler(30 * degree, 10 * degree, 60 * degree, cs)))

ori = discreteSample(odf, 10000, rng=rng)
# -

# ## Compare the model with its sample
#
# Filled contours show sections through the model ODF. The points are the sampled
# orientations drawn over every section.

plotSection(odf, contourf=True)
mtexColorMap('white2black')
plot(ori, add2all=True, markerSize=3, markerAlpha=0.25, all=True)

# Notice the narrow concentration of points around the compact peak and the elongated
# concentration along the fibre. These are the two structures the clustering step should
# distinguish.
#
# ## Cluster the orientations
#
# Orientation clustering does not use CLASSIX unless it is selected explicitly. The embedding
# used by this method accounts for crystal symmetry, so symmetrically equivalent
# representations of an orientation receive the same cluster label. The call supplies the
# radius and the minimum cluster size, as the text above says of this example.

# +
cId, center = calcCluster(ori, method='classix', radius=0.1, minPoints=100)

plotSection(ori, ind2color(cId), markerSize=3, markerAlpha=0.25, all=True)
annotate(center)
# -

# The dominant color follows the fibre and another color collects the compact unimodal
# portion. The annotated centers summarize the returned groups; they do not replace the full
# within-cluster spread visible here.
#
# ## References
#
# * X. Chen and S. Güttel, [Fast and explainable clustering based on
#   sorting](https://doi.org/10.1016/j.patcog.2024.110298), _Pattern Recognition_ 150 (2024),
#   110298, introduces the CLASSIX algorithm and its radius and minimum-size controls.
