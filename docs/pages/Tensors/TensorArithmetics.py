# # Tensor Arithmetic
#
# This page assumes the tensor ranks and physical classes introduced in
# [Defining Tensorial Properties](TensorDefinition.html). It shows how MTEX
# applies arithmetic to one tensor or a list without loops. The central
# operation is a *contraction*: a sum over selected component indices.
#
# The rotation examples also assume the active rotations introduced in
# [Defining Rotations](RotationDefinition.html).

import numpy as np

from mtex import *

plottingConvention.default('y↑→x')

# ## Basic algebra
#
# Addition, scalar multiplication, and elementwise operations follow the array
# notation. These variables each contain three random rank 2 tensors,
# so every line below acts on all three at once.

# +
T1 = tensor.rand(3, rank=2)
T2 = tensor.rand(3, rank=2)

# addition and scalar multiplication
T = T1 + 2 * T2

# componentwise multiplication
T = times(T1, T2)
# -

# The function `times`, MATLAB's operator `.*`, multiplies corresponding components. It is
# not an outer product and does not change the rank. The random tensors are algebraic
# examples rather than tensors with the symmetries of a physical property.

# ## Contractions with EinsteinSum
#
# Hooke's law is a useful model for reading a contraction. The following
# rank 4 stiffness tensor contains room-temperature measurements for San
# Carlos olivine from
# [Abramson et al. (1997)](https://doi.org/10.1029/97JB00682).
# [Importing Tensor Data](TensorImport.html) explains the loading step and
# the reference-frame information that must accompany published components.

C = stiffnessTensor.load(mtexdatafile('olivine1997'))

# The data are in GPa. Apply a strain that stretches along x and compresses
# along z.

eps = strainTensor(np.diag([1, 0, -1]))
eps

# Hooke's law contracts the last two indices of the stiffness with the two
# indices of the strain,
#
# $$\sigma_{ij} = \sum_{k,l} C_{ijkl} \epsilon_{kl}.$$
#
# [EinsteinSum](EinsteinSum.html) takes each tensor followed by one label
# for each of its indices. Equal negative labels mark indices to sum over.
# Positive labels survive, and their values set the index order in the
# result. The signs are MTEX syntax; the physical indices are not negative.

sigma = EinsteinSum(C, [1, 2, -1, -2], eps, [-1, -2])
sigma

# Labels 1 and 2 leave a rank 2 result. Its diagonal normal stresses are
# 248.9, -8.65, and -161.9 GPa; all shear components are zero.
#
# The same notation describes elementary products. The outer product
#
# $$ (a \otimes b)_{ij} = a_i b_j $$
#
# has two free indices and no summation.

a = tensor([1, 2, 3], rank=1)
b = tensor([0, 2, 1], rank=1)

# ---

ab = EinsteinSum(a, 1, b, 2)
ab

# The inner product
#
# $$ a \mathbin{\cdot} b = \sum_i a_i b_i $$
#
# instead repeats one negative label and leaves no free index. Its result is
# the scalar 7.

aDotB = EinsteinSum(a, -1, b, -1)
aDotB

# ## Stiffness is not compliance
#
# Linear compressibility is the relative shortening along a direction `v`
# under unit hydrostatic pressure. It is a contraction of the *compliance*
# $S=C^{-1}$,
#
# $$ \beta(v) = \sum_{i,j,k} S_{ijkk} v_i v_j.$$
#
# [inv](tensor.inv.html) turns the stiffness into a compliance tensor. The
# data file identifies x with [100], the crystallographic a direction.

v = xvector
S = inv(C)
beta = EinsteinSum(S, [-1, -2, -3, -3], v, -1, v, -2)
beta

# The result is 0.0018 GPa$^{-1}$. It is the same contraction used by
# [linearCompressibility](stiffnessTensor.linearCompressibility.html).

betaDirect = C.linearCompressibility(v)

# Contracting the stiffness in the same way is dimensionally wrong. The
# result is 460.25 GPa, not a compressibility.

wrongUnits = EinsteinSum(C, [-1, -2, -3, -3], v, -1, v, -2)
wrongUnits

# ## Named contractions
#
# Frequently used contractions have operators. For a stiffness and a
# strain, both `*` and `colon`, MATLAB's `:`, apply Hooke's law. They also preserve the
# physical result class, so the first command displays a `stressTensor` rather than
# the generic `tensor` returned by `EinsteinSum`.

sigmaTimes = C * eps
sigmaColon = colon(C, eps)
sigmaTimes

# Between two rank 2 tensors, the double dot product is their inner product.
# It is also the trace of either matrix product below. Each result here is a
# list of three scalars because `T1` and `T2` are lists.

inner = colon(T1, T2)
innerFromLeftTrace = trace(T1 * T2.T)
innerFromRightTrace = trace(T1.T * T2)
inner

# ## Rank 2 matrix operations
#
# [det](tensor.det.html) returns one determinant for each rank 2 tensor in
# the list.

d = det(T1)
d

# ## Rotating a tensor
#
# [rotate](tensor.rotate.html) actively rotates a physical property. The
# workshop example below starts with a second-rank thermal-conductivity
# tensor whose largest value, 10.5 W m$^{-1}$ K$^{-1}$, is along x. The
# rotation moves that preferred direction 45 degrees about z.

# +
K = tensor(np.diag([10.5, 6.2, 6.2]), rank=2, name='thermal conductivity', unit='W m^-1 K^-1')
r = rotation.byAxisAngle(zvector, 45 * degree)
Krot = rotate(K, r)

newMtexFigure(layout=[1, 2])
plot(K, complete=True, upper=True)
mtexTitle('before rotation')

nextAxis()
plot(Krot, complete=True, upper=True)
mtexTitle('after rotation')

setColorRange('equal')
mtexColorbar(title='thermal conductivity (W m$^{-1}$ K$^{-1}$)')
# -

# The high-conductivity region turns in the right plot, while its maximum
# stays 10.5. A rank 2 tensor receives one rotation matrix per index,
#
# $$ K'_{ij} = \sum_{p,q} R_{ip} R_{jq} K_{pq}. $$
#
# Rotating the property is not a *frame change*. A frame change re-expresses
# the same physical object in another reference frame. Use
# [transformReferenceFrame](tensor.transformReferenceFrame.html) for that
# operation; [Reference Frame Alignment](SymmetryAlignment.html) shows why
# the distinction matters for published crystal properties.

# ## A third-rank example
#
# Nye gives the following piezoelectric modulus for a crystal with one
# threefold axis. The two printed component tables show the same property
# before and after an active rotation of -45 degrees about z.

# +
P = np.array([[0, 0, 0, .17, 0,   0],
              [0, 0, 0, 0,   .17, 0],
              [0, 0, 0, 0,   0,   5.17]]) * 10**-11

P0 = tensor(P, rank=3, name='piezoelectric modulus')
P0
# -

# ---

rPiezo = rotation.byAxisAngle(zvector, -45 * degree)
P45 = rotate(P0, rPiezo)
P45

# Components that were zero before the rotation are nonzero afterwards.
# Nothing about the crystal changed internally: the component table changed
# because the property now points in different specimen directions.

# ## Next
#
# [Tensor Visualization](TensorVisualisation.html) explains the directional
# plot used above and the specialized plots for physical tensor classes.
# [Anisotropic Elasticity](AnisotropicTheory.html) applies these contractions
# to elastic moduli, energy, and wave propagation. The polycrystal step is
# [Tensor Averages](TensorAverage.html).

# ## Further reading
#
# * J. F. Nye, [Physical Properties of Crystals: Their Representation by Tensors and
#   Matrices](https://search.worldcat.org/title/11114089), Oxford University Press, 1985.
#   Pages 120-121 contain the third-rank rotation example above.
# * C. Hammond, [The physical properties of crystals and their description by
#   tensors](https://doi.org/10.1093/acprof:oso/9780198738671.003.0014), in _The Basics of
#   Crystallography and Diffraction_, 4th edition, 2015.
# * E. H. Abramson, J. M. Brown, L. J. Slutsky, and J. Zaug, [The elastic constants of San
#   Carlos olivine to 17 GPa](https://doi.org/10.1029/97JB00682), _Journal of Geophysical
#   Research_ 102(B6), 12253-12263, 1997.
