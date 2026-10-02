# Parent grain reconstruction, timed 2026-10-02

`bench/reconstruction.py` runs the steps of the MaParentGrainReconstruction page on the
martensite map (343 602 pixels, 10 654 grains at 3 degree), `--routes` the vote and grain
graph routes on MATLAB's refined relationship, `bench/matlab/probe_reconstruction.m` and
`probe_votes.m` the same in MATLAB R2024b on frameSymmetry. Warm runs; a desktop MATLAB was
running beside, so the Python times vary by a quarter between runs.

| step | Python before | Python after | MATLAB |
| --- | --- | --- | --- |
| calcGrains | 1.0 s | 1.0 s | 6.4 s |
| calcGBFit | 0.34 s | 0.3 s | 1.3 s |
| calcParent2Child | 8.2 s | 3.0 s | 24.8 s |
| calcVariantGraph | 0.6 s | 0.6 s | 8.7 s |
| clusterVariantGraph | 4.1 s | 1.7 s | 9.4 s |
| growth (calcGBVotes, calcParentFromVote) | 0.4 s | 0.4 s | 1.7 s |
| calcParentEBSD | 1.3 s | 1.3 s | 2.0 s |
| calcGBVotes | 1.1 s | 0.5 s | 5.3 s |
| calcTPVotes | 15.8 s | 3.4 s | 30.7 s |
| calcParentFromGraph | 39 s | 6.1 s | not run |

What the time went into, and what took it out:

- `calcTPVotes`: a stable argsort of the 24^3 variant triples of every triple point to keep
  the best two; `_best` takes the best few of a finite row by repeated argmin, the first
  minimum on a tie as the stable sort.
- `calcParentFromGraph`: `calcParent` recomputed the parent variants on each of its 25 calls
  per cluster through a Python pair loop, and `stripSym` built the trivial group and searched
  the register on every call. The variants of a relationship are remembered
  (`_variantData`), `_distinctModulo` is one pair search, a frame's sibling by a named group
  is remembered in the session (`SESSION.siblings`, cleared by a reset), the trivial group is
  built once and `isLaue`, `isProper` are read once per group.
- `calcParent2Child`: the misfit took the reduced angle of every misorientation to all 23
  child to child misorientations, 115 times. `_nearestC2C` puts the images l c_k r of the
  classes of the c_k in a k-d tree over the double cover; equivalent c_k share their images
  and the first names them, where the angles decided by rounding before. The refined
  relationship moved by 0.0005 degree (3.4985 to 3.4980 degree from Kurdjumov-Sachs).
- `clusterVariantGraph`: scipy's sparse product, one thread and a counting pass;
  `mclExpandNumba` in `maps/kernels.py` is the product with the inflation and the pruning
  fused, over row chunks on the threads, bit for bit scipy's matrix.

## A map eleven times larger

The page's steps (`calcVariantGraph` at 3.5 degree with tolerance 3.5 degree as the page has it)
on `15um_ECD_dataset_02.ctf` of the variant graph paper
(`../../publications/21_variantGraph/old/tuomo3`, 2072 x 1892 = 3 920 224 pixels, the same two
phases), 249 785 grains, 5 993 805 graph nodes; the results the same before and after.

| step | before | after | peak memory before, after |
| --- | --- | --- | --- |
| calcGrains | 8.2 s | 8 to 10 s | 2.4 GB |
| calcParent2Child | 4.3 s | 3.4 s | 2.4 GB |
| calcVariantGraph | 27 s | 11.5 s | 10.6 GB, 2.4 GB |
| clusterVariantGraph | 108 s | 104 s | 4.5 GB |
| calcParentFromVote | 0.09 s | 0.06 s | |
| growth (calcGBVotes, calcParentFromVote) | 12 s | 7 s | |
| mergeSimilar, mergeInclusions | 3.6 s, 6.5 s | 1.9 s, 6.3 s | |
| calcParentEBSD | 26 s | 2.1 s | 13.5 GB, 6.1 GB |

- `calcVariantGraph` held the table of the 24 x 24 variant angles of all 700 000 child
  boundaries and its probabilities at once, 3 GB each with the temporaries of `erf`;
  `_variantPairs` takes it in chunks of 32 768 boundaries and evaluates the probability only
  below the angle where it is 0.1. `grains.neighbors` found the unique pairs by rows; one key
  per pair and a table of the selection take a tenth of the time.
- `calcParentEBSD` spent 17 s in `chordAngle` over the 24 variants of every pixel.
  `closestAngle` in `geometry/kernels.py` turns the closest image into the angle inside the
  kernel, which `orientationAngle` and `symPairAngles` take, seven times faster on its own;
  `_nearestVariant` finds the variant of a child as the nearest of the 576 images
  inv(V_k) inv(r) to inv(P) c, without the table of the variants of every pixel.
- `clusterVariantGraph` is the MCL expansion itself: the graph grows to 59 million entries,
  each expansion 11 billion products, 0.7 s at the first and 16 s at the last of the ten
  steps on ten threads. A hash accumulator in place of the dense one changed nothing, so it
  is not the cache; the work is what MCL with the cutoff 1e-4 asks.

## The clustering's parameters, kept

The expansion of `clusterVariantGraph` is the work MCL asks, so a faster clustering changes
the result. The whole page reconstruction on the large map against the defaults, the parent
pixels whose orientation differs; a relationship turned by 0.08 degree, the difference the
tie-break of the vote and the gap to MATLAB's fit make, is the scale:

| run | clusterVariantGraph | parent grains | parent pixels off by more than 5 degree |
| --- | --- | --- | --- |
| numIter 10, cutOff 1e-4, the defaults | 99 s | 4743 | |
| the relationship turned 0.08 degree | 106 s | 4772 | 0.41 % |
| numIter 8 | 70 s | 4771 | 0.51 % |
| cutOff 2e-4 | 59 s | 4797 | 0.84 %, 1513 pixels no longer reconstructed |
| numIter 6 | 38 s | 4856 | 1.11 % |

On the martensite map 0.17, 0.67, 1.31 and 2.23 percent. cutOff 1e-3 keeps only 57 percent of
the votes, the 32 largest entries per row cost more than they save. The defaults stay
MATLAB's (2026-10-02).

## Open questions

- `calcParentFromGraph` spends what is left in the 24 candidate parents of every cluster,
  each with a robust `mean` and an `angle` of a few orientations: Python overhead per call,
  which one compiled loop over the clusters would remove.
- The vote of `calcParent2Child` at a c_k with a stabilizer: two equivalents of a
  misorientation lie equally near it and give different votes, the image order and rounding
  of `projectToRef` choose; the other choice ends 0.08 degree away, 0.0024 degree from
  MATLAB's, at an objective 0.07 percent higher. A vote averaged over the tied equivalents
  would not depend on the choice.
- `mergeInclusions` on the large map, 6 s in `inclusionHost`, a Python loop of reductions.
- `calcParentFromGraph` in MATLAB is untimed (the probe runs it only with `full` set).
