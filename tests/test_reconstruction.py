# M9: the parent grains a transformed microstructure came from, the Julia R1 to R7 on a
# synthetic map of two parent grains broken into three variants each
import numpy as np
import pytest

from mtex import EBSD, calcGrains, crystalFrame, degree, orientation, parentGrainReconstructor, rotation, vector3d
from mtex.geometry.quaternion import qmul
from mtex.maps.phaselist import phaseList
from mtex.maps.reconstruction import accumVotes

CSP = crystalFrame("m-3m", mineral="austenite")
CSC = crystalFrame("m-3m", mineral="martensite")
P2C = orientation.KurdjumovSachs(CSP, CSC)
P = orientation(rotation.byAxisAngle(vector3d([[0, 0, 1], [1, 1, 1]]), np.array([0, 40 * degree])), CSP)


# two parent grains side by side, each a block on the left and two stacked on the right, so
# that three grains of one parent meet at a triple point; every grain is tilted a little
# off its ideal variant, by a different amount, so that no two fits tie
def martensiteMap(n=18, inclusion=False, gradient=0.0):
  cv = P2C.variants(P).reshape(2, -1)

  def variantOf(i, j):
    if inclusion and 5 <= i < 8 and 3 <= j < 6:
      return 20
    return (2 if i < 4 else 7 if j < 9 else 14) if i < n // 2 else (5 if i < 13 else 9 if j < 7 else 13)

  pos, q = [], []
  for j in range(n):
    for i in range(n):
      side, v = (0 if i < n // 2 else 1), variantOf(i, j)
      k = 3 * side + v
      r = qmul(rotation.byAxisAngle(vector3d(1, 2 + k, 3 - k), 0.1 * degree * k).data.reshape(4), cv[side, v - 1].data.reshape(4))
      if gradient:
        r = qmul(rotation.byAxisAngle(vector3d(0, 0, 1), gradient * (i + j)).data.reshape(4), r)
      pos.append([i, j, 0.0])
      q.append(r)
  return EBSD(np.array(pos), rotation._fromData(np.array(q)), np.full(n * n, 2), phaseList(CSP, CSC))


def fitsParent(g):
  o = orientation._fromRotation(g.meanRotation.data, (CSP, g.frame))
  return np.min(np.stack([np.asarray(o.angle(P[k])).reshape(-1) for k in range(2)]), axis=0)


@pytest.fixture(scope="module")
def plain():
  e = martensiteMap()
  return e, calcGrains(e, threshold=2 * degree)


def test_martensiteMap_isWhatItWasBuiltAs(plain):
  e, g = plain
  assert len(g) == 6 and np.all(g.phaseId == 2) and g.numPixel.sum() == len(e)


def test_parentGrainReconstructor_startsWithEveryGrainAChild(plain):
  e, g = plain
  job = parentGrainReconstructor(e, g, P2C)
  assert job.csParent is CSP and job.csChild is CSC and job.parentPhaseId == 1 and job.childPhaseId == 2
  assert np.all(job.isChild) and not job.isParent.any() and len(job.parentGrains) == 0


# the phases are guessed from the map, the most frequent indexed one the child and the
# rarest the parent, even without a pixel of it
def test_parentGrainReconstructor_guessesThePhasesFromTheMap(plain):
  e, g = plain
  guess = parentGrainReconstructor(e, g)
  assert guess.csChild is CSC and guess.csParent is CSP and guess.p2c is None


def test_parentGrainReconstructor_singlePhaseMapTakesTheParentFromP2c(plain):
  e, g = plain
  es = EBSD(e.pos, e.rotations, np.ones(len(e)), phaseList(CSC))
  single = parentGrainReconstructor(es, calcGrains(es, threshold=2 * degree))
  assert single.csParent is None
  single.p2c = P2C
  assert single.csParent is CSP and single.parentPhaseId == 2 and single.grains.CSList[2] is CSP


# the grain at a centroid
def at(g, x, y):
  c = g.centroid.data[:, :2] if hasattr(g.centroid, "data") else np.asarray(g.centroid)
  return int(np.argmin(np.hypot(c[:, 0] - x, c[:, 1] - y)))


# a job on the plain map with the votes of its boundaries
def votedJob(plain):
  e, g = plain
  job = parentGrainReconstructor(e, g, P2C)
  job.calcGBVotes()
  return job


def test_calcGBVotes_areMatlabs(plain):
  e, g = plain
  v = votedJob(plain).votes
  assert v.kind == "prob" and v.parentId.shape == (6, 2) and np.all(v.prob[:, 0] >= v.prob[:, 1])
  # MATLAB's votes on this map, bench/matlab of 2026-09-23, every grain found by its centroid
  matlab = {(15, 3): (1, 4, 0.899322594999436, 0.017423629589798), (6, 4): (1, 24, 0.651237914888712, 0.328137992472640),
            (1.5, 8.5): (1, 11, 0.637438508846248, 0.341937398515102), (6, 13): (1, 15, 0.901683835098880, 0.008174851127677),
            (10.5, 8.5): (1, 13, 0.847578670101204, 0.016728922755818), (15, 12): (1, 2, 0.987843796190228, 0.0)}
  for (x, y), (i1, i2, p1, p2) in matlab.items():
    k = at(g, x, y)
    assert np.array_equal(v.parentId[k], [i1, i2]) and np.allclose(v.prob[k], [p1, p2], atol=1e-9)


def test_calcParentFromVote_turnsTheVotedGrainsIntoTheirParent(plain):
  job = votedJob(plain)
  job.calcParentFromVote(minProb=0.3)
  assert np.all(job.isParent) and np.all(np.isnan(job.votes.prob[:, 0]))
  assert np.max(fitsParent(job.grains)) < 2 * degree


# the grains of one parent are one grain
def test_mergeSimilar_joinsTheGrainsOfOneParent(plain):
  job = votedJob(plain)
  job.calcParentFromVote(minProb=0.3)
  job.mergeSimilar()
  assert len(job.grains) == 2 and np.max(fitsParent(job.grains)) < 2 * degree
  assert np.all(job.isTransformed) and np.array_equal(np.sort(job.numChildren), [3, 3])
  assert np.array_equal(np.sort(job.grains.clusterSize), [3, 3]) and np.all(job.isMerged)


# a job whose one sure parent has grown into the grains around it
def grownJob(plain):
  job = votedJob(plain)
  job.calcParentFromVote(minProb=0.95)
  assert np.count_nonzero(job.isParent) == 1
  for _ in range(2):
    job.calcGBVotes(p2c=True, threshold=3 * degree)
    job.calcParentFromVote(minProb=0.5)
  return job


def test_calcGBVotes_parentGrowsIntoItsChildren(plain):
  e, g = plain
  job = grownJob(plain)
  right = [at(g, 10.5, 8.5), at(g, 15, 3), at(g, 15, 12)]
  assert np.array_equal(np.flatnonzero(job.isParent), np.sort(right)) and np.max(fitsParent(job.grains)[right]) < 2 * degree


# the growth step again from the measured orientations, a grain with a child neighbour
# left undecided as MATLAB leaves it
def test_calcGBVotes_reconsiderAllLeavesAChildWithAChildNeighbourUndecided(plain):
  job = grownJob(plain)
  job.calcGBVotes(p2c=True, reconsiderAll=True, threshold=3 * degree)
  assert np.all(np.isnan(job.votes.prob[job.isChild, 0]))


def test_calcTPVotes_areSureOfOneParent(plain):
  e, g = plain
  job = parentGrainReconstructor(e, g, P2C)
  job.calcTPVotes()
  assert np.all(job.votes.prob[:, 0] > 0.8) and np.all(job.votes.prob[:, 1] == 0)


def test_calcTPVotes_findTheTwoParents(plain):
  e, g = plain
  job = parentGrainReconstructor(e, g, P2C)
  job.calcTPVotes()
  job.calcParentFromVote(minProb=0.5)
  job.mergeSimilar()
  assert len(job.grains) == 2 and np.max(fitsParent(job.grains)) < 2 * degree


# the boundary votes of the grain means and of the boundary pixels
def votesBothWays(e, g):
  a = parentGrainReconstructor(e, g, P2C)
  b = parentGrainReconstructor(e, g, P2C, useBoundaryOrientations=True)
  a.calcGBVotes()
  b.calcGBVotes()
  return a, b


# every pixel of a grain carries one orientation, so the boundary pixels and the grain
# mean are the same
def test_useBoundaryOrientations_isTheGrainMeanOnUniformGrains(plain):
  a, b = votesBothWays(*plain)
  assert np.allclose(a.votes.prob, b.votes.prob, atol=1e-9)


# with a gradient across the grains they differ, and the boundary pixels find every parent
def test_useBoundaryOrientations_findsEveryParentAcrossAGradient():
  eg = martensiteMap(gradient=0.05 * degree)
  a, b = votesBothWays(eg, calcGrains(eg, threshold=2 * degree))
  assert np.max(np.abs(np.nan_to_num(a.votes.prob) - np.nan_to_num(b.votes.prob))) > 1e-6
  b.calcParentFromVote(minProb=0.3)
  b.mergeSimilar()
  assert len(b.grains) == 2 and np.max(fitsParent(b.grains)) < 3 * degree


def test_calcGBFit_hasOneFitPerPairOfNeighbours(plain):
  e, g = plain
  fit, pairs = parentGrainReconstructor(e, g, P2C).calcGBFit()
  assert len(fit) == len(pairs) == len(g.neighbors())


def test_calcParent2Child_fitsTheRelationshipOfTheMap(plain):
  e, g = plain
  job = parentGrainReconstructor(e, g, P2C)
  job.calcParent2Child(local=True)
  assert job.p2c.angle(P2C) < 5 * degree
  assert np.array_equal(np.sort(job.variantMap), np.arange(1, 25))


# the grains go back to what they were measured as
def test_revert_givesTheGrainsBackAsMeasured(plain):
  e, g = plain
  job = votedJob(plain)
  job.calcParentFromVote(minProb=0.3)
  prior = g.meanRotation.data.copy()
  job.revert()
  assert np.all(job.isChild) and np.allclose(np.abs(np.sum(job.grains.meanRotation.data * prior, axis=1)), 1)


# a job reconstructed through to the two merged parent grains and their variants
def reconstructedJob(plain):
  job = votedJob(plain)
  job.calcParentFromVote(minProb=0.3)
  job.mergeSimilar()
  job.calcVariants()
  return job


def test_calcVariants_theChildrenOfOneParentAreDifferentVariants(plain):
  job = reconstructedJob(plain)
  vId = job.variantId
  assert np.all(np.isfinite(vId)) and np.all((job.packetId >= 1) & (job.packetId <= 4))
  # the three children of one parent are three different variants of it
  for p in np.unique(job.mergeId):
    ids = vId[job.mergeId == p]
    assert len(np.unique(ids)) == len(ids) == 3


def test_calcParentEBSD_isTheParentMap(plain):
  e, g = plain
  job = reconstructedJob(plain)
  pe = job.calcParentEBSD()
  assert np.all(pe.phaseId == 1) and np.array_equal(np.unique(pe.grainId), np.unique(job.mergeId))
  op = orientation._fromRotation(pe.rotations.data.reshape(-1, 4), (CSP, pe.frame))
  assert np.max(np.min(np.stack([np.asarray(op.angle(P[k])).reshape(-1) for k in range(2)]), axis=0)) < 2 * degree
  assert np.all(np.isfinite(pe.fit)) and np.max(pe.fit) < 2 * degree and np.all(np.isfinite(pe.variantId))
  assert len(job.parentEBSD) == len(e)


# a merged grain comes apart again into the prior grains it was made of
def test_revert_splitsAMergedGrainIntoItsPriorGrains(plain):
  job = reconstructedJob(plain)
  job.revert(np.array([True, False]))
  assert len(job.grains) == 4 and np.count_nonzero(job.isChild) == 3 and np.count_nonzero(job.isParent) == 1


# the map the job was made from is untouched
def test_parentGrainReconstructor_leavesItsMapUntouched(plain):
  e, g = plain
  job = reconstructedJob(plain)
  job.calcParentEBSD()
  job.revert(np.array([True, False]))
  assert np.allclose(np.abs(np.sum(e.rotations.data * job.ebsdPrior.rotations.data, axis=-1)), 1)


def test_mergeInclusions_absorbsAnInclusion():
  ei = martensiteMap(inclusion=True)
  gi = calcGrains(ei, threshold=2 * degree)
  assert len(gi) == 7
  job = parentGrainReconstructor(ei, gi, P2C)
  job.calcGBVotes()
  job.calcParentFromVote(minProb=0.3)
  # the inclusion stays a child and what is around it is merged
  inc = int(np.argmin(gi.numPixel))
  job.revert([inc])
  job.mergeSimilar()
  assert np.count_nonzero(job.isChild) == 1
  n0 = len(job.grains)
  job.mergeInclusions(maxSize=20)
  assert len(job.grains) == n0 - 1 and np.count_nonzero(job.isChild) == 0
  assert np.max(fitsParent(job.grains)) < 2 * degree


# two grains, grain 0 with three votes for variant 2 and one for 1, grain 1 with none
VOTE_GRAIN = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
VOTE_ID = np.array([[2, 1], [2, 3], [2, 3], [1, 2]])
VOTE_FIT = np.array([[0.5, 3], [0.5, 4], [1.0, 5], [0.5, 1]]) * degree


def test_accumVotes_asMatlabAccumulatesThem():
  v = accumVotes(VOTE_GRAIN, VOTE_ID, VOTE_FIT, 2)
  assert np.array_equal(v.parentId[0], [2, 1]) and v.prob[0, 0] > v.prob[0, 1] and np.all(np.isnan(v.prob[1]))


# minFit and maxFit select the votes on their best and second best fit
def test_accumVotes_minFitAndMaxFitSelectTheVotes():
  v = accumVotes(VOTE_GRAIN, VOTE_ID, VOTE_FIT, 2, minFit=0.7 * degree, maxFit=2 * degree)
  assert np.array_equal(v.parentId[0], [2, 1]) and v.prob[0, 1] == 0


def test_accumVotes_bestFitKeepsTheBestFit():
  b = accumVotes(VOTE_GRAIN, VOTE_ID, VOTE_FIT, 2, bestFit=True)
  assert b.kind == "fit" and b.parentId[0, 0] in (1, 2) and np.isclose(b.fit[0, 0], 0.5 * degree)


def test_accumVotes_countCountsTheVotes():
  c = accumVotes(VOTE_GRAIN, VOTE_ID, VOTE_FIT, 2, count=True)
  assert c.kind == "count" and c.parentId[0, 0] == 2 and c.count[0, 0] == 4
