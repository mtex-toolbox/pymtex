# # Grain Reconstruction by Markovian Clustering
#
# This page builds on the three reconstruction steps introduced in
# [Grain Reconstruction](GrainReconstruction.html) and the weighted criteria from
# [Advanced Grain Reconstruction](GrainReconstructionAdvanced.html). A pixel graph has one
# node for each EBSD measurement and an edge between neighbouring spatial cells. The edge
# weight measures how strongly the criterion connects the two measurements.
#
# A phase change always receives zero connectivity. The reconstructed grains therefore
# remain phase-homogeneous, whichever clustering method reads the graph.
#
# By default, `calcGrains` takes the connected components of the pairs with positive
# connectivity. Two pixels then end up in the same grain whenever any chain of connected
# pairs leads from one to the other. That chain may be long and only weakly connected to
# the regions at either end.
#
# Markov clustering (MCL) reads the weighted graph differently. It simulates random walks
# by alternating two operations. Expansion squares the transition matrix, so it considers
# two-step walks. Inflation raises every transition probability to a power `p` and
# renormalises each column. This favours paths on which flow is already strong and
# suppresses weak links. Repetition leaves flow circulating within regions that the walk
# rarely leaves, and those regions become grains.
#
# The keyword `mcl` selects this clustering step. Its value is the pair
# `(inflationPower, maxIt)`. MCL does not accept a requested number of grains; the
# inflation power controls the granularity of the result.

from mtex import *

# ## A criterion with something to say
#
# MCL acts on edge weights, so a graded criterion gives it more evidence than a hard
# yes-or-no threshold. An unweighted graph can still be split by its topology, but equal
# nonzero weights do not describe which orientation links are weaker than others.
#
# We demonstrate the distinction on the deformed austenite in the `EMSphinx` data set. This
# is the same region used for the threshold and `gbcFMC` comparison on the
# [basic reconstruction page](GrainReconstruction.html).
# `calcGrains` marks the pixels of the grains that `minPixel` drops on the map it is given,
# so every reconstruction below starts from its own copy of the map.

# +
plottingConvention.default('y↓→x')
ebsd = mtexdata('EMSphinx', verbose=False)

ebsd = ebsd['Iron fcc']
ebsd = ebsd[inpolygon(ebsd, [40, 30, 80, 60])]

grainsHard = calcGrains(ebsd.copy(), angle=10 * degree, minPixel=10)
print(f'hard threshold: {grainsHard.size} grains')
grainsHard = smoothBoundary(grainsHard, 5)

plot(ebsd, ebsd.orientations, ipfDirection=zvector)
hold(True)
plot(grainsHard.boundary, lineWidth=4)
hold(False)
mtexTitle('Hard threshold')
# -

# The black lines are the baseline reconstruction. Notice the broad colour gradients
# inside grains and the low-angle boundaries that a 10 degree threshold cannot detect.

# +
grainsHardMCL = calcGrains(ebsd.copy(), mcl=(1.2, 4), angle=10 * degree, minPixel=10)
print(f'hard threshold + MCL: {grainsHardMCL.size} grains')
grainsHardMCL = smoothBoundary(grainsHardMCL, 5)

hold(True)
plot(grainsHardMCL.boundary, lineWidth=3, lineColor='white')
hold(False)
mtexTitle('Hard threshold + MCL')
# -

# The white MCL boundaries lie over the black baseline boundaries, and both methods return
# the same 19 grains on this map. Here the binary criterion has not given MCL useful
# boundary-strength information.
#
# The soft threshold [`gbcSoft`](gbcSoft.html) supplies that information. Its connectivity
# falls gradually around the threshold instead of jumping. It takes the threshold angle
# and the width of the transition.

# +
grainsSoft = calcGrains(ebsd.copy(), criterion=gbcSoft(1 * degree, 0.5 * degree), minPixel=10)
print(f'soft threshold: {grainsSoft.size} grains')
grainsSoft = smoothBoundary(grainsSoft, 5)

hold(True)
plot(grainsSoft.boundary, lineWidth=1, lineColor='gray')
hold(False)
mtexTitle('Soft threshold without MCL')
# -

# The gray lines show what the soft criterion does by itself. Connected components only
# ask whether an edge weight is positive, so this use of a soft threshold mainly shifts the
# effective cutoff. The gradual weights become informative when MCL uses their relative
# strengths. The shift alone raises the count from 19 to 40 grains.

# +
grainsSoftMCL = calcGrains(ebsd.copy(), mcl=(1.2, 4), criterion=gbcSoft(1 * degree, 0.5 * degree), minPixel=10)
print(f'soft threshold + MCL: {grainsSoftMCL.size} grains')
grainsSoftMCL = smoothBoundary(grainsSoftMCL, 5)

plot(ebsd, ebsd.orientations, ipfDirection=zvector)
hold(True)
plot(grainsSoftMCL.boundary, lineWidth=3)
hold(False)
mtexTitle('Soft threshold + MCL')
# -

# The low-angle boundaries of the deformed grains are now found, while the bent lattice
# inside them is not cut into dense contours. A hard threshold low enough to reach those
# boundaries does produce them: at half a degree the same region shatters into 214 grains,
# against the 45 returned here. The important change is not the nominal angle but the
# pairing of graded connectivity with global clustering.
#
# ## The inflation power
#
# Of the two MCL parameters, the inflation power is the one to tune. It sets how sharply
# the walk is forced onto its likely paths. A larger value generally breaks the graph into
# more and smaller clusters.

for p in [1.1, 1.4, 1.8]:
  grains = calcGrains(ebsd.copy(), mcl=(p, 4), criterion=gbcSoft(1 * degree, 0.5 * degree), minPixel=10)
  print(f'inflation {p:.1f}: {grains.size} grains')

# The count rises from 45 grains at `p = 1.1` to 52 at `p = 1.8`. This is a parameter
# check, not a target supplied to MCL. Inspect the boundaries as well as the count, because
# a finer partition is not automatically a more physical one.
#
# The second parameter limits the number of MCL iterations and defaults to 4. Raising it
# changes the result very little and costs a great deal. The transition matrix fills in as
# the walk expands, so on this map eight iterations take about twelve times as long as four
# and return two grains more.
#
# ## When to use it
#
# MCL is a global clustering step. It does not decide from one pixel pair in isolation,
# but from how the weighted graph hangs together. This is useful in deformed material,
# where no single threshold angle separates every physical boundary without also cutting
# through bent lattices.
#
# MCL is not the only global approach in MTEX. The criterion `gbcFMC` reaches a comparable
# segmentation of this map in a fraction of the time and needs no threshold angle. See
# [Grain Reconstruction](GrainReconstruction.html) for that route.
#
# MCL remains the more flexible option. It clusters the weights produced by
# [`gbcSoft`](gbcSoft.html), [`gbcCustom`](gbcCustom.html), or a criterion of your own
# without needing to know what those weights mean. Since expansion can make the sparse
# transition matrix much denser, test the method on a representative subregion before
# applying it to a large map.
#
# ## References
#
# * S. van Dongen, "Graph clustering via a discrete uncoupling process", *SIAM Journal on
#   Matrix Analysis and Applications* 30 (2008), 121-141,
#   [doi:10.1137/040608635](https://doi.org/10.1137/040608635). This paper defines the
#   expansion and inflation operations used by MCL.
#
# * F. Bachmann, R. Hielscher and H. Schaeben, "Grain detection from 2d and 3d EBSD data -
#   Specification of the MTEX algorithm", *Ultramicroscopy* 111 (2011), 1720-1733,
#   [doi:10.1016/j.ultramic.2011.08.002](https://doi.org/10.1016/j.ultramic.2011.08.002).
#   This paper describes the spatial graph from which MTEX reconstructs grains.
