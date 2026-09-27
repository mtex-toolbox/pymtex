# # Vector Fields in Orientation Space
#
# A vector field on the rotation group assigns a tangent vector to every orientation.
# Evaluating an `SO3VectorField` at an orientation $R$ returns an `SO3TangentVector`
# attached to $R$.
#
# Read [The Tangent Space on the Rotation Group](RotationTangentSpace.html) first for the
# definition of tangent vectors and their left and right representations. Vector fields
# model orientation-dependent spin in Taylor and Sachs calculations. The gradient of an
# orientation distribution function (ODF) is another important example.

# +
import numpy as np

from mtex import *

plottingConvention.default('y↑→x')
# -

# ## A first vector field: the gradient of an ODF
#
# Consider the model ODF for a quartz specimen known as the Dubna texture. The gradient
# points in the direction of the fastest local increase of the ODF, and its norm gives that
# rate of increase.

odf = SO3Fun.dubna()
G = odf.grad()
G

# Evaluation at one sampled orientation returns the tangent vector attached to that
# orientation.

rng = np.random.default_rng(1)
ori = odf.discreteSample(1, rng=rng)
G.eval(ori)

# Plot the ODF in sigma sections and draw the gradient arrows on top.

plot(odf, 'sigma')
hold('on')
plot(G, linewidth=1.5, color='black', resolution=7.5 * degree)
hold('off')

# The arrows point uphill towards the nearest local maximum. Their lengths increase where the
# ODF changes more steeply. This ascent direction is the basis of the
# [steepest-descent algorithm](SO3Fun.steepestDescent.html) used by [max](SO3Fun.max.html)
# and [calcComponents](SO3Fun.calcComponents.html).
#
# ## Three representations
#
# MTEX provides three interchangeable representations. A harmonic or RBF field stores three
# scalar component functions using the array convention introduced in
# [Vector-Valued Orientation Functions](SO3FunVectorValued.html).
#
# | representation | class | when to use it |
# |---|---|---|
# | three harmonic component functions | [SO3VectorFieldHarmonic](SO3VectorFieldHarmonic.SO3VectorFieldHarmonic.html) | differentiation and global spectral approximation |
# | three radial-basis component functions | [SO3VectorFieldRBF](SO3VectorFieldRBF.SO3VectorFieldRBF.html) | approximation by local kernels |
# | an evaluation formula | [SO3VectorFieldHandle](SO3VectorFieldHandle.SO3VectorFieldHandle.html) | an explicit rule that can be evaluated at any orientation |
#
# All three representations support the common `SO3VectorField` operations.
#
# ## Left and right tangent-vector coordinates
#
# An `SO3VectorField` has a requested tangent-space representation and two associated
# symmetries. Left coordinates are the default. The [right](SO3VectorField.right.html) and
# [left](SO3VectorField.left.html) methods re-express the same geometric vectors; they do
# not change the vectors or their base orientations.

GR = right(G)
GR

# ---

v = GR.eval(ori)
vRight = right(G.eval(ori))
norm(v - vRight)

# The small residual shows that converting the field before evaluation and converting the
# evaluated tangent vector give the same result. Arithmetic also converts compatible fields
# to a common representation automatically.

G + GR

# ## Why the visible symmetries change
#
# A field's tangent vectors live in the frame of one side: left vectors in the specimen
# frame, right vectors in the crystal frame. Symmetry acts differently on the two coordinate
# choices. For a right-represented tangent vector, evaluations at symmetry-equivalent
# orientations are meaningful only with respect to the left symmetry. For a left-represented
# tangent vector, the reverse applies.

ori = orientation.rand(G.CS, G.SS, rng=rng)
G.eval(ori.symmetrise())

# ---

GR.eval(ori.symmetrise())

# The left vectors turn with the crystal symmetry operation that relates the equivalent
# orientations, while the right vectors, given in crystal coordinates, repeat. The field
# keeps the crystal and specimen frames of the ODF either way, and the component functions
# of each side carry the symmetry that side leaves them.
#
# ## Operations on vector fields
#
# The following operations apply to vector fields `VF`, `VF1` and `VF2`:
#
# * sums, differences, scaling and division
# * inner products with [dot(VF1, VF2)](SO3VectorField.dot.html)
# * cross products with [cross(VF1, VF2)](SO3VectorField.cross.html)
# * norms with [norm(VF)](SO3VectorField.norm.html)
# * normalization with [normalize(VF)](SO3VectorField.normalize.html)
# * rotation with [rotate(VF, rot)](SO3VectorField.rotate.html)
# * averages with [mean(VF)](SO3VectorField.mean.html)
#
# Since a gradient is itself a vector field, MTEX can also compute its divergence, curl and
# scalar antiderivative.
#
# ## Divergence and the Laplacian
#
# Treating `G` as an orientation-space velocity field gives an intuitive reading of its
# divergence. Negative divergence marks a sink where nearby orientations condense. Positive
# divergence marks a source where they spread apart.
#
# The divergence of a gradient equals the Laplacian of its scalar field. Plot the two
# calculations side by side at the same sigma section.

plot(G.div(), 'sigma', 60 * degree)
nextAxis()
plot(laplace(SO3FunHarmonic(odf)), 'sigma', 60 * degree)
mtexColorbar()

# The source and sink regions, contour shapes and colour scale agree in the two panels. The
# left panel was computed from the vector field, whereas the right panel was computed
# directly from the ODF.
#
# ## Curl and conservative fields
#
# Curl describes the axis of local circulation within orientation space. The curl of a
# gradient is zero, so the next plot should contain no nonzero arrows.

plot(G.curl(), 'sigma')

# Vanishing curl identifies a conservative field: a field that is the gradient of a scalar
# potential. The [antiderivative](SO3VectorField.antiderivative.html) method reconstructs
# that potential.

odf2 = G.antiderivative()
odf2

# A gradient loses the additive constant of its source function. Restoring the original
# mean makes the reconstructed potential coincide with `odf`.

odf2 = odf2 + mean(odf)
plot(odf2, 'sigma')

# ## Define a field by an evaluation formula
#
# A Python function is convenient when a vector formula is known. The following rule uses
# the rotation axis multiplied by the rotation angle, with cubic symmetry on both sides.

cs = crystalFrame('432')
cs

# ---

f = lambda mori: mori.axis() * mori.angle()
VF = SO3VectorFieldHandle(f, cs, cs)
VF

# Evaluating a $10^\circ$ rotation about $[1\;2\;3]$ and reducing the axis to small integers
# recovers the expected direction ratio $1:2:3$.

round(VF.eval(orientation.byAxisAngle(vector3d(1, 2, 3), 10 * degree, cs, cs)))

# The following axis-angle plot samples the formula throughout the cubic fundamental region.
# Every arrow points away from the identity at the origin, as a field of axis times angle
# must.

quiver3(VF, 'axisAngle', resolution=7.5 * degree, color='black', linewidth=2)

# ## Convert a field to harmonic form
#
# Passing any `SO3VectorField` to the harmonic constructor expands its three components by
# quadrature.

SO3VectorFieldHarmonic(VF)

# ## Fit harmonic components to sampled values
#
# A second construction starts from rotations and one `vector3d` value at each rotation. The
# first array dimension again corresponds to nodes.

nodes = equispacedSO3Grid(specimenFrame('1'), points=1e3).flatten()
y = vector3d.byPolar(np.sin(3 * nodes.angle()), nodes.phi2 + np.pi / 2)

# The approximation below produces a harmonic vector field with bandwidth 16.

SO3VF1 = SO3VectorFieldHarmonic.approximate(nodes, y, bandwidth=16)
SO3VF1

# ## Construct by quadrature of a function
#
# A function that accepts a rotation and returns a `vector3d` can also be passed directly to
# quadrature. Here the earlier cubic formula produces a harmonic vector field.

SO3VF2 = SO3VectorFieldHarmonic.quadrature(lambda v: f(v))
SO3VF2

# ## Construct from three scalar harmonic functions
#
# A three-component `SO3FunHarmonic` can be wrapped directly. Its first, second and third
# entries become the $x$, $y$ and $z$ components of the vector field.

SO3F = SO3FunHarmonic(rng.random((3, 1000)))
SO3F

# ---

SO3VF3 = SO3VectorFieldHarmonic(SO3F)
SO3VF3

# ## Application: orientation-dependent spin in the Taylor model
#
# Taylor theory accommodates a prescribed strain by activating slip systems in each crystal.
# The antisymmetric part of the resulting deformation describes the local lattice spin, and
# therefore the local misorientation predicted for each orientation. Without an input
# orientation, [calcTaylor](strainTensor.calcTaylor.html) returns this spin as an
# `SO3VectorField`.

cs = crystalFrame('432')
sS = slipSystem.bcc(cs)
sS

# Set plane strain with $q=0$ and calculate the spin field for the symmetrised
# body-centred-cubic slip systems.

q = 0
epsilon = strainTensor(np.diag([1, -q, -(1 - q)]))
epsilon

# ---

_, _, W = calcTaylor(epsilon, sS.symmetrise())
W

# Display the spin directions in four Euler-angle sections.

sP = phi1Sections(cs, specimenFrame('222'))
sP.phi1 = np.arange(10, 71, 20) * degree
plot(W, sP, resolution=7.5 * degree, layout=[2, 2])

# Direction and length vary with orientation, showing that the Taylor model predicts a
# different local misorientation across orientation space. The value at the copper
# orientation can be retrieved directly.

WCopper = W.eval(orientation.copper(cs))
WCopper

# ## The amount of spin
#
# The norm of the spin vector is the angle of local misorientation. Its maximum locates the
# orientation with the largest predicted rotation.

_, oriMax = max(norm(W))
oriMax

# Plot the norm at $0.5^\circ$ resolution and overlay the more coarsely sampled vector
# field. The background shows magnitude, the arrows show direction, and the annotation marks
# `oriMax`.

plot(norm(W), sP, resolution=0.5 * degree, layout=[2, 2])
mtexColorMap('LaboTeX')
hold('on')
plot(W, sP, resolution=7.5 * degree, color='black')
hold('off')
annotate(oriMax)

# ## Compare spin with a crystal direction
#
# Since `W` gives the rotation axis of the local misorientation, its inner product with a
# chosen direction measures signed alignment. Here the direction is crystal $[100]$.

plot(dot(W, Miller(1, 0, 0, cs)), sP, layout=[2, 2])
mtexColorMap('blue2red')
mtexColorbar()

# Positive and negative regions indicate parallel and antiparallel components along
# $[100]$. Values near zero indicate that the spin axis is locally perpendicular to that
# direction.
#
# ## Sources and sinks of the Taylor spin field
#
# Finally compute the divergence of `W`. As in the gradient example, negative values are
# sinks and positive values are sources in orientation space.

flux = W.div()
flux

# ---

plot(flux, sP, resolution=0.5 * degree, layout=[2, 2], faceAlpha=0.5)
mtexColorMap('blue2red')
mtexColorbar()

# The alternating red and blue regions show that the Taylor spin field moves orientations
# towards some parts of orientation space and away from others.
#
# ## References
#
# * A. Morawiec,
#   [Orientations and Rotations: Computations in Crystallographic Textures](https://doi.org/10.1007/978-3-662-09156-2),
#   Springer, 2004, develops the tangent-space geometry used for gradients and vector fields
#   on $\mathrm{SO}(3)$.
# * H.-J. Bunge,
#   [Some applications of the Taylor theory of polycrystal plasticity](https://doi.org/10.1002/crat.19700050112),
#   _Kristall und Technik_ 5 (1970), 145--175, gives the orientation-dependent Taylor
#   factors and spin fields used in the final example.
#
# ## Next
#
# Continue with [Rotational Kernel Functions](SO3Kernels.html) to understand the localized
# basis functions used by the RBF representation listed on this page.
#
# ## Technical Details
#
# A field in the port keeps its crystal and specimen frames on either side; MATLAB's hidden
# symmetries are the groups the component functions carry. A field is stored on the side it
# is given in, so `right(G)` is a field of its own at bandwidth 49 where MATLAB keeps the
# left one at 48 and converts on evaluation. The Taylor spin field is in left components, as
# the Julia port keeps it, where MATLAB shows the right spin tensor: the copper spin
# $(0, -0.463, 0)$ in specimen coordinates stands for MATLAB's tensor with entries
# $\pm 0.3215$ in crystal coordinates, a vector of length 0.455. `dot(W, Miller(1, 0, 0, cs))`
# of the left field takes its right components, since a crystal direction lives in the
# crystal frame. The largest spin is found at $(259^\circ, 42.4^\circ, 123^\circ)$ with 1.207,
# MATLAB's at $(268.7^\circ, 45.4^\circ, 63.3^\circ)$, where the port's field has 1.173, 9.4
# degrees away under the symmetry of the strain. MATLAB's cached Dubna ODF keeps the
# plotting convention y↓→x of an earlier session, the port's follows the page's y↑→x, so
# the sigma sections of the gradient are MATLAB's turned by 180 degrees about z. The sampled
# orientation and the random coefficients come from NumPy.
