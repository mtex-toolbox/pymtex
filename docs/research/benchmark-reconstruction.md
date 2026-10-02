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

## Open questions

- `calcParentFromGraph` spends what is left in the 24 candidate parents of every cluster,
  each with a robust `mean` and an `angle` of a few orientations: Python overhead per call,
  which one compiled loop over the clusters would remove.
- `calcParentEBSD`: the variant id of every pixel through `orientationAngle` and
  `chordAngle` in NumPy, about 1 s.
- `calcParentFromGraph` in MATLAB is untimed (the probe runs it only with `full` set).
