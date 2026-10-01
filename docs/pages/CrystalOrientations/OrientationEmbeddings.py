# # Embeddings of Orientations
#
# An orientation embedding represents each physical orientation by a unique tensor in a
# Euclidean space. Symmetrically equivalent rotations therefore have the same embedding.
# This makes ordinary linear operations available without first choosing one rotation
# from each equivalence class.
#
# This page assumes the symmetry equivalence developed in
# [Orientation Symmetry](OrientationSymmetry.html) and the representative selection
# explained in [Fundamental Regions](OrientationFundamentalRegion.html). The
# [misorientation angle](MisorientationTheory.html) supplies the intrinsic distance used
# below.
#
# ## Why Fundamental-Region Coordinates Jump
#
# A rotation matrix is a tensorial representation of a rotation, but it is not a unique
# representation of an orientation with crystal symmetry. Restricting rotations to a
# fundamental region selects one representative from each equivalence class. Near the
# boundary, however, two nearby orientations can be assigned representatives on opposite
# sides of the region. Their coordinate vectors are then far apart even though their
# misorientation angle is small.
#
# The following experiment compares three distances for random pairs of cubic
# orientations.

# +
import matplotlib.pyplot as plt
import numpy as np

from mtex import *

# cubic proper rotation symmetry
cs = crystalFrame('432')

# select representatives in the fundamental region
ori1 = projectIntoFundamentalRegion(orientation.rand(20000, cs))
ori2 = projectIntoFundamentalRegion(orientation.rand(20000, cs))

# intrinsic misorientation angle in degrees
omega = angle(ori1, ori2) / degree

# Euclidean distances between matrix and Rodrigues representatives
distMat = norm(tensor(ori1) - tensor(ori2))
distRV = norm(Rodrigues(ori1) - Rodrigues(ori2))

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(omega, distMat, 3, marker='.')
plt.xlabel(r'$\omega(\mathtt{ori}_1,\mathtt{ori}_2)$ in degrees')
plt.ylabel(r'$\Vert\mathtt{tensor(ori_1)}-\mathtt{tensor(ori_2)}\Vert_2$')
plt.title('rotation matrices')

plt.subplot(1, 3, 2)
plt.scatter(omega, distRV, 3, marker='.')
plt.xlabel(r'$\omega(\mathtt{ori}_1,\mathtt{ori}_2)$ in degrees')
plt.ylabel(r'$\Vert\mathtt{R(ori_1)}-\mathtt{R(ori_2)}\Vert_2$')
plt.title('Rodrigues vectors')

plt.subplot(1, 3, 3)
plt.scatter(distMat, distRV, 3, marker='.')
plt.xlabel(r'$\Vert\mathtt{tensor(ori_1)}-\mathtt{tensor(ori_2)}\Vert_2$')
plt.ylabel(r'$\Vert\mathtt{R(ori_1)}-\mathtt{R(ori_2)}\Vert_2$')
plt.title('two coordinate distances')
# -

# In the first two panels, points near the left edge can still have a large coordinate
# distance. The third panel shows that matrix and
# [Rodrigues Frank](rotation.byRodrigues.html) coordinates jump differently. Neither
# Euclidean coordinate distance is the geometry of orientation space.
#
# This is why averaging fundamental-region coordinates is unsafe. Consider the
# orientations with Bunge Euler angles $(44^{\circ},0^{\circ},0^{\circ})$ and
# $(46^{\circ},0^{\circ},0^{\circ})$.

# +
ori = projectIntoFundamentalRegion(orientation.byEuler(np.array([44, 46]) * degree, 0, 0, cs))

# average the selected Rodrigues representatives
naiveMean = orientation.byRodrigues(mean(ori.Rodrigues()), cs)
naiveMean
# -

# The displayed result is the identity orientation. It is $45^{\circ}$ from the expected
# mean because the two selected Rodrigues vectors lie on opposite sides of a
# fundamental-region boundary. In contrast, [mean](orientation.mean.html) handles crystal
# symmetry.

intrinsicMean = mean(ori)
intrinsicMean

# The symmetry-aware mean is the physical orientation represented by
# $(45^{\circ},0^{\circ},0^{\circ})$. More generally, this coordinate jump affects any
# vector-space method applied directly to fundamental-region matrices, Rodrigues vectors,
# or Euler angles.
#
# ## Constructing an Embedding
#
# The `embedding` class replaces a chosen coordinate representative by a
# higher-dimensional tensor that is invariant under the crystal symmetry. Its Euclidean
# metric is locally isometric to the misorientation metric. It therefore agrees with the
# misorientation angle for small separations, but its chord distance is not the geodesic
# distance for arbitrary pairs.

e1 = embedding(ori1)
e2 = embedding(ori2)
e2

# The summary identifies the cubic symmetry, tensor rank, packed dimension, and number of
# embedded orientations. This output is useful because the required tensor rank and
# dimension depend on the crystal symmetry.
#
# Compare the Euclidean embedding distance with the misorientation angle. Division by
# `degree` expresses the locally isometric distance in degrees.

# +
distE = norm(e1 - e2) / degree

plt.close('all')
plt.scatter(omega, distE, 3, marker='.')
plt.xlabel(r'$\omega(\mathtt{ori}_1,\mathtt{ori}_2)$ in degrees')
plt.ylabel(r'$\Vert\mathcal{E}(\mathtt{ori}_1)-\mathcal{E}(\mathtt{ori}_2)\Vert_2$')
# -

# Near the origin the point cloud follows the diagonal: embedding distance closely
# approximates the misorientation angle. The increasing curvature at larger angles is the
# difference between an ambient chord and a geodesic on orientation space.
#
# ## Averaging in Embedding Space
#
# An arithmetic mean of embeddings lies in the ambient Euclidean space and generally is
# not itself the embedding of an orientation. Project it back with
# [orientation](embedding.orientation.html).

e = embedding(ori)
meanEmbedding = mean(e)
embeddingMean = orientation(meanEmbedding)
embeddingMean

# MTEX may print this cubic orientation with Euler angle $315^{\circ}$ instead of
# $45^{\circ}$. Those are symmetry-equivalent representatives, not different physical
# means. The intrinsic angular error confirms the result.

targetMean = orientation.byEuler(45 * degree, 0, 0, cs)
embeddingMeanError = angle(embeddingMean, targetMean) / degree
embeddingMeanError

# ## Constant Norm and Dispersion
#
# All orientations of one symmetry have the same embedding norm. The raw radius depends on
# the symmetry; the `normalized` option divides by that radius. The following output is
# therefore a row of ones.

normalizedNorms = norm(embedding(orientation.rand(5, cs)), normalized=True)
normalizedNorms

# Normalized embeddings lie on the unit sphere. Their arithmetic mean lies inside the unit
# ball, so its norm can summarize concentration. A value close to one indicates a tight
# cluster. A value near zero indicates strong cancellation; it can accompany orientations
# far apart in the orientation space, but does not by itself prove that a maximally
# separated pair is present.
#
# Write the normalized mean-embedding norm as
#
# $$ n=\frac{1}{\rho}\left\Vert\frac{1}{N}\sum_{i=1}^N
# \mathcal E(\mathtt{ori}_i)\right\Vert, \qquad
# \rho=\Vert\mathcal E(\mathtt{ori})\Vert.$$
#
# Compare it with the angular standard deviation
#
# $$ \sigma=\left(\frac{1}{N}\sum_{i=1}^N
# \omega(\mathtt{ori}_i,\mathtt{mori})^2\right)^{1/2},$$
#
# where $\mathtt{mori}$ is the mean orientation. The first samples come from one family of
# [unimodal de la Vallee Poussin distributions](RadialODFs.html#3) with varying halfwidth.

# +
n = []
sigma = []
for hw in np.logspace(-1, 1.75, 40) * degree:

  psi = SO3DeLaValleePoussinKernel(halfwidth=hw)
  odf = unimodalODF(orientation.rand(cs), psi)
  ori = discreteSample(odf, max(1000, round(1000 * (hw * 6) ** 3)))

  n.append(norm(mean(embedding(ori)), normalized=True))
  sigma.append(std(ori))

n, sigma = np.array(n), np.array(sigma)
plt.figure()
plt.plot(sigma, np.sqrt(np.maximum(0, 1 - n)), linewidth=2)
plt.xlabel(r'standard deviation $\sigma$')
plt.ylabel(r'$\sqrt{1-n}$')
# -

# The smooth curve can suggest that the mean-embedding norm determines the standard
# deviation. That apparent relationship comes from varying only one distribution family.
# [Bingham distributions](BinghamODFs.html) form a broader family and expose the
# ambiguity.

# +
n = []
sigma = []
for k in range(1, 601, 2):

  kappa = np.random.rand(4)
  kappa = k * kappa / np.sum(kappa)
  odf = BinghamODF(kappa, cs)
  ori = discreteSample(odf, 1000)

  n.append(norm(mean(embedding(ori)), normalized=True))
  sigma.append(std(ori))

n, sigma = np.array(n), np.array(sigma)
plt.scatter(sigma, np.sqrt(np.maximum(0, 1 - n)), 12)
plt.legend(['de la Vallee Poussin', 'Bingham'], loc='best')
# -

# The Bingham points do not collapse onto the first curve. Thus the norm of the mean
# embedding is a concentration summary, not a one-to-one proxy for angular standard
# deviation. The figure is also a warning against calibrating one dispersion measure from
# a single distribution family.
#
# ## Operations
#
# Embeddings support the following linear-space operations:
#
# * `+`, `-`, `*` and `/`
# * [mean](embedding.mean.html)
# * [norm](embedding.norm.html)
# * [dot](embedding.dot.html)
# * [rotate](embedding.rotate.html)
#
# ## Packed Numeric Coordinates
#
# The tensor representation stores repeated components. The
# [double](embedding.double.html) method packs the independent components into a numeric
# matrix while preserving Euclidean distances exactly. For cubic symmetry the following
# output compares the full tensor component count with the packed dimension.

fullDimension = double(e1, full=True).shape[1]
packedDimension = double(e1).shape[1]
fullDimension, packedDimension

# Each row of the packed matrix now represents one orientation and can be passed to
# numerical or machine-learning code. This packing is not a learned dimensionality
# reduction, and an arbitrary row in the ambient space need not correspond to a valid
# orientation.

distD = np.linalg.norm(double(e1) - double(e2), axis=1) / degree
packingError = np.max(np.abs(distE - distD))
packingError

# +
plt.close('all')
plt.scatter(omega, distD, 3, marker='.')
plt.xlabel(r'$\omega(\mathtt{ori}_1,\mathtt{ori}_2)$ in degrees')
plt.ylabel(r'$\Vert\mathtt{double}(\mathcal{E}_1)-\mathtt{double}(\mathcal{E}_2)\Vert_2$')
# -

# This plot reproduces the earlier embedding-distance plot, and the printed packing error
# is at floating-point roundoff. Packing removes redundant tensor entries without changing
# the metric.
#
# ## References
#
# * R. Arnold, P. E. Jupp and H. Schaeben, [Statistics of ambiguous rotations](https://doi.org/10.1016/j.jmva.2017.10.007),
#   _Journal of Multivariate Analysis_ 165, 73--85, 2018.
# * R. Hielscher and L. Lippert, [Locally isometric embeddings of quotients of the rotation group modulo finite symmetries](https://doi.org/10.1016/j.jmva.2021.104764),
#   _Journal of Multivariate Analysis_ 185, 104764, 2021.
# * M. Moakher, [Means and averaging in the group of rotations](https://doi.org/10.1137/S0895479801383877),
#   _SIAM Journal on Matrix Analysis and Applications_ 24, 1--16, 2002.
# * K. V. Mardia and P. E. Jupp, [Directional Statistics](https://doi.org/10.1002/9780470316979),
#   Wiley, 2000, gives the broader statistical background for directional data.
#
# ## Next
#
# Continue with [Misorientations](Misorientations.html) to study relative orientations and
# their symmetry. For distributions of whole orientation populations, continue with
# [Orientation Density Functions](ODFAnalysis.html).
