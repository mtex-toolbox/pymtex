# the MaParentGrainReconstruction steps on the martensite map, timed, with the numbers
# bench/matlab/probe_reconstruction.m prints; --matlab-p2c takes MATLAB's refined
# relationship so that the steps after the fit compare on the same one
import sys
import time

import numpy as np

from mtex import calcGrains, degree, mtexdata, orientation, parentGrainReconstructor
from mtex.geometry.quaternion import rotation
from mtex.maps.reconstruction import _quantile

T = time.perf_counter


def step(name, f, show=lambda: ""):
  t = T()
  f()
  print(f"{name:20s} {T() - t:6.2f} s  {show()}")


# the quintiles as MTEX's quantile takes them
def q(x):
  return np.array2string(np.array([_quantile(x, p) for p in (0.2, 0.4, 0.6, 0.8)]), precision=4)


ebsd = mtexdata("martensite", silent=True)
out = {}
step("calcGrains", lambda: out.update(grains=calcGrains(ebsd, angle=3 * degree, minPixel=2, alpha=12)), lambda: len(out["grains"]))
job = parentGrainReconstructor(ebsd, out["grains"])
KS = orientation.KurdjumovSachs(job.csParent, job.csChild)
job.p2c = KS
step("calcGBFit", lambda: out.update(fit=job.calcGBFit()[0] / degree), lambda: q(out["fit"]))
if "--matlab-p2c" in sys.argv:
  job.p2c = orientation(rotation._fromData(np.array([0.9183690176, -0.01211658611, 0.3880635769, 0.07653885347])), job.csParent, job.csChild)
else:
  step("calcParent2Child", job.calcParent2Child, lambda: f"{float(job.p2c.angle(KS)) / degree:.4f}")
print(f"{'fit':20s} {'':8s}  {q(job.calcGBFit()[0] / degree)}")
step("calcVariantGraph", lambda: job.calcVariantGraph(threshold=3.5 * degree, tolerance=1.5 * degree), lambda: job.graph.nnz)
step("clusterVariantGraph", lambda: job.clusterVariantGraph(includeSimilar=True), lambda: q(job.votes.prob[:, 0]))
step("calcParentFromVote", lambda: job.calcParentFromVote(minProb=0.5), lambda: np.count_nonzero(job.isParent))
step("growth", lambda: (job.calcGBVotes(p2c=True, reconsiderAll=True, threshold=4 * degree, tolerance=1.5 * degree), job.calcParentFromVote()),
     lambda: np.count_nonzero(job.isParent))
step("mergeSimilar", lambda: job.mergeSimilar(threshold=7.5 * degree), lambda: len(job.grains))
step("mergeInclusions", lambda: job.mergeInclusions(maxSize=50), lambda: f"{len(job.grains)} {np.count_nonzero(job.isParent)}")
step("calcVariants", job.calcVariants, lambda: np.bincount(job.packetId[np.isfinite(job.packetId)].astype(int), minlength=5)[1:])
step("calcParentEBSD", lambda: out.update(pe=job.calcParentEBSD()), lambda: q(out["pe"].fit / degree))
print(job)

# the vote and grain graph routes, bench/matlab/probe_votes.m
if "--routes" in sys.argv:
  job = parentGrainReconstructor(ebsd, out["grains"], KS)
  job.p2c = orientation(rotation._fromData(np.array([0.9183690176, -0.01211658611, 0.3880635769, 0.07653885347])), job.csParent, job.csChild)
  step("calcGBVotes", job.calcGBVotes, lambda: q(job.votes.prob[:, 0]))
  step("calcTPVotes", job.calcTPVotes, lambda: q(job.votes.prob[:, 0]))
  step("calcGraph", lambda: job.calcGraph(threshold=2.5 * degree, tolerance=2.5 * degree), lambda: job.graph.nnz)
  step("clusterGraph", job.clusterGraph)
  step("calcParentFromGraph", job.calcParentFromGraph, lambda: np.count_nonzero(job.isParent))
