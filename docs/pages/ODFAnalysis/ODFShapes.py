# # Unimodal ODF Shapes
#
# A [unimodal ODF](RadialODFs.html) is a normalized peak around one orientation. Its
# [SO(3) kernel](SO3Kernels.html) sets the profile of that peak: how the density falls
# with angular distance from the centre.
#
# The *halfwidth* is the distance at which a kernel has fallen to half its maximum. It is
# a spread parameter, not normally a cutoff. Matching halfwidths aligns one visible
# feature, but the tails and peak heights may still differ.
#
# Kernel choice matters when measured orientations are turned into a density; see
# [Density Estimation](DensityEstimation.html). It also controls how many harmonic
# degrees are needed to represent the resulting ODF.
# [Series Expansion](SO3FunHarmonicRepresentation.html) introduces that representation
# and defines its bandwidth.
#
# This page compares kernels for point-centred radial components. Pass an `SO3Kernel`
# object to [unimodalODF](unimodalODF.html), or use its `halfwidth` option to select the
# default de la Vallee Poussin kernel. A [fibre ODF](FibreODFs.html) is different: current
# [fibreODF](fibreODF.html) accepts an `S2Kernel` because its density decays across a
# curve rather than from one orientation.

import matplotlib.pyplot as plt
import numpy as np

from mtex import *

plottingConvention.default('y↑→x')

# +
psi = [SO3AbelPoissonKernel(0.79), SO3DeLaValleePoussinKernel(13), SO3BumpKernel(35 * degree), SO3DirichletKernel(3),
       SO3vonMisesFisherKernel(7.5), SO3GaussWeierstrassKernel(0.07), fibreVonMisesFisherKernel(7.2), SO3SquareSingularityKernel(0.72)]

kernelName = ['Abel-Poisson', 'de la Vallee Poussin', 'bump', 'Dirichlet', 'von Mises-Fisher', 'Gauss-Weierstrass',
              'fibre von Mises-Fisher', 'square singularity']
# -

# ## Comparing Halfwidths
#
# The positional arguments above are the native parameters of the eight families. The
# labelled output converts them to the common halfwidth scale. The values were chosen to
# span the comparable range from about $15^\circ$ to $37^\circ$. The plots below
# therefore show the difference in *shape* rather than in width.

# +
halfwidthInDegree = [kernel.halfwidth() / degree for kernel in psi]
for name, hw in zip(kernelName, halfwidthInDegree):
  print(f'{name:24s} {hw:8.4f}')
# -

# ## Profiles in Orientation Space
#
# Plot each kernel against angular distance from its centre. Every kernel is normalized
# to mean 1 on orientation space, not to have maximum 1. Narrower profiles can therefore
# have higher peaks.

# +
plt.figure(figsize=(10, 4.5), layout='constrained')
hold(True)
for kernel, name in zip(psi, kernelName):
  plot(kernel, displayName=name)
hold(False)
plt.xlabel('angle from centre in degree')
plt.ylabel('kernel value')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# -

# The bump kernel is constant inside its halfwidth and exactly zero outside it. This
# special cutoff should not be inferred from the meaning of halfwidth in the other
# families. The Dirichlet kernel oscillates through zero, whereas the smooth families
# decay with different tails.
#
# A negative kernel is useful for some harmonic calculations, but it is not a probability
# density. A radial ODF made directly from the Dirichlet kernel can therefore take
# negative values. The labelled output confirms that this example does.

omega = np.linspace(0, np.pi, 1000)
dirichletMinimum = np.min(psi[3].eval(np.cos(omega / 2)))
dirichletMinimum

# ## Projected Profiles in a Pole Figure
#
# A pole figure is the crystallographic Radon transform of an ODF; see
# [Pole Figures of an ODF](ODFPoleFigure.html). Applying that transform to a kernel gives
# the profile contributed by one radial component to a pole figure peak. This is the
# curve a measured pole figure peak is compared against.

# +
plt.figure(figsize=(10, 4.5), layout='constrained')
hold(True)
for kernel, name in zip(psi, kernelName):
  plot(kernel.radon(), symmetric=True, displayName=name, lineWidth=2)
hold(False)
plt.ylim(-5, 20)
plt.xlabel('angular distance in degree')
plt.ylabel('projected kernel value')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# -

# Projection changes the profiles but does not erase their main distinctions. The bump
# remains compactly supported, and the Dirichlet curve retains negative side lobes. The
# other curves stay nonnegative but differ in how broadly their pole-figure intensity is
# spread.

# ## Harmonic Cost
#
# A radial kernel stores one coefficient for every harmonic degree from zero through its
# bandwidth. Faster coefficient decay permits a lower bandwidth and makes harmonic
# computations cheaper.

# +
plt.figure(figsize=(7, 4.5), layout='constrained')
hold(True)
for kernel, name in zip(psi, kernelName):
  plotSpectrum(kernel, bandwidth=32, lineWidth=2, displayName=name)
hold(False)
plt.xlabel('harmonic degree')
plt.ylabel('radial coefficient')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# -

# The plot shows only degrees 0 through 32 so that the low-degree decay can be compared.
# The table reports the full stored bandwidth of each kernel. Bandwidth 10 means degrees
# 0 through 10 and therefore 11 stored radial coefficients, not 10.

# +
kernelBandwidth = [kernel.bandwidth for kernel in psi]
for name, bw in zip(kernelName, kernelBandwidth):
  print(f'{name:24s} {bw:5d}')
# -

# The de la Vallee Poussin kernel ends at bandwidth 10. Both von Mises-Fisher variants
# and the Gauss-Weierstrass kernel end at 11, whereas the discontinuous bump kernel
# continues to bandwidth 1024. A sharp edge is expensive in a smooth harmonic basis.
#
# This combination of a finite exact expansion, nonnegative values, and smooth real-space
# shape is why MTEX uses the de la Vallee Poussin family by default for unimodal ODFs and
# orientation density estimation.

# ## References
#
# * [Schaeben (1999)](https://doi.org/10.1155/TSM.33.365) derives the de la Vallee
#   Poussin orientation density function and its finite harmonic expansion.
# * [Hielscher (2013)](https://doi.org/10.1016/j.jmva.2013.03.014) compares kernel
#   families for density estimation on the rotation group.

# ## Next
#
# Build point-centred peaks in [Radial ODFs](RadialODFs.html) and tubular components in
# [Fibre ODFs](FibreODFs.html). The formulas and constructor details for every family are
# collected in [SO(3) Kernels](SO3Kernels.html).

# ## Technical Details
#
# MATLAB's tables are printed rows here, and its figure commands are matplotlib's:
# `figure('position', ...)` is `plt.figure(figsize=...)`, `legend` is `plt.legend`. The
# halfwidth is the method `kernel.halfwidth()`, the bandwidth the property
# `kernel.bandwidth`. The port cuts the bump kernel at the bandwidth its halfwidth
# needs rather than at 1024, so that number differs.
