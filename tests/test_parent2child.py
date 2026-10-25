# G7: parent to child orientation relationships, MATLAB's check_calcParent2Child and the
# Julia parent2child.jl numbers
import numpy as np
import pytest

from mtex import Miller, calcParent, calcParent2Child, calcVariantId, crystalFrame, degree, orientation, rotation, round2Miller, vector3d
from mtex.geometry.parent2child import (_best, _childToChild, _nearestC2C, _nearestVariant, _parentOfManyLoop, _robustMeans, _rowQuantile,
                                         allHKL, misfit)
from mtex.geometry.quaternion import quaternion
from mtex.geometry.quaternion import qmul

csP = crystalFrame("432", [3.66, 3.66, 3.66], mineral="fcc")
csC = crystalFrame("432", [2.87, 2.87, 2.87], mineral="bcc")
cP = crystalFrame("m-3m", [3.66, 3.66, 3.66], mineral="fcc")
cC = crystalFrame("m-3m", [2.87, 2.87, 2.87], mineral="bcc")

# the options of the global search
opt = dict(searchResolution=10 * degree, numLocal=1, maxSample=300, scanSample=100)


def perturb(o, ax, omega):
  return o._new(qmul(o.data, rotation.byAxisAngle(vector3d(ax), omega).data))


# child to child misorientations of n parent grains, each split into two variants, with
# noise and a share of outliers
def c2cData(rng, p2c, n, noise, outliers):
  V = p2c.variants().flatten()
  parent = orientation.rand(n, p2c.frameA, rng=rng)
  o1 = parent * V[rng.integers(0, V.size, n)].inv()
  o2 = parent * V[rng.integers(0, V.size, n)].inv()
  mori = o1.inv() * o2
  mori = mori._new(qmul(mori.data, rotation.byAxisAngle(vector3d.rand(n, rng=rng), noise * rng.standard_normal(n)).data))
  k = int(round(outliers * n))
  data = np.concatenate([orientation.rand(k, p2c.frameB, rng=rng).data.reshape(-1, 4), mori.data.reshape(-1, 4)[k:]])
  mori = orientation(rotation._fromData(data), p2c.frameB, p2c.frameB)
  return mori[mori.angle() >= 5 * degree]


# noisy Kurdjumov-Sachs misorientations with outliers, a start 3 degrees off, the local fit,
# a permutation of the data and pairs of child orientations
@pytest.fixture(scope="module")
def ks():
  rng = np.random.default_rng(0)
  p2c = orientation.KurdjumovSachs(csP, csC)
  mori = c2cData(rng, p2c, 1500, 1 * degree, 0.2)
  p2c0 = perturb(p2c, [1, 2, 3], 3 * degree)
  p = calcParent2Child(mori, p2c0, local=True)
  perm = rng.permutation(mori.size)
  parent = orientation.rand(200, csP, rng=rng)
  V = p2c.variants().flatten()
  pairs = orientation(rotation._fromData(np.stack([(parent * V[rng.integers(0, 24, 200)].inv()).data,
                                                  (parent * V[rng.integers(0, 24, 200)].inv()).data], axis=1)), csC)
  pg = calcParent2Child(mori, p2c0, **opt)
  return dict(p2c=p2c, mori=mori, p2c0=p2c0, p=p, perm=perm, pairs=pairs, pg=pg)


def test_calcParent2Child_localSearchRecoversTheRelationship(ks):
  p, p2c, p2c0 = ks["p"], ks["p2c"], ks["p2c0"]
  assert p.angle(p2c) < 0.3 * degree and p.angle(p2c) < p2c0.angle(p2c)


def test_misfit_mostWithinTheNoise(ks):
  # the misfit of every misorientation against the result, most within the noise
  fit = misfit(ks["mori"], ks["p"])
  assert len(fit) == ks["mori"].size and np.quantile(fit, 0.75) < 3 * degree


def test_misfit_isTheSmallestReducedAngleToAChildToChildMisorientation(ks):
  _, c2c = _childToChild(ks["p"])
  a = np.asarray(ks["mori"].reshape(-1, 1).angle(c2c.reshape(1, -1)))
  variant, omega = _nearestC2C(ks["mori"], c2c)
  assert np.allclose(omega, a.min(axis=1), atol=1e-12) and np.allclose(a[np.arange(len(a)), variant], omega, atol=1e-12)


def test_best_takesTheBestFewInTheOrderOfTheStableSort():
  fit = np.random.default_rng(3).integers(0, 5, (50, 30)).astype(float)
  ids, value = _best(fit, 3, (30,))
  order = np.argsort(fit, axis=1, kind="stable")[:, :3]
  assert np.array_equal(ids, order + 1) and np.array_equal(value, np.take_along_axis(fit, order, axis=1))


def test_calcParent2Child_subsampledFitAgreesAndGivesEveryMisfit(ks):
  q, fit = calcParent2Child(ks["mori"], ks["p2c0"], local=True, maxSample=300, fit=True)
  assert len(fit) == ks["mori"].size and q.angle(ks["p"]) < 0.3 * degree


def test_calcParent2Child_localSearchIsOrderIndependent(ks):
  assert ks["p"].angle(calcParent2Child(ks["mori"][ks["perm"]], ks["p2c0"], local=True)) < 1e-3 * degree


def test_calcParent2Child_globalSearchIsDeterministic(ks):
  assert ks["pg"].angle(calcParent2Child(ks["mori"], ks["p2c0"], **opt)) == 0


def test_calcParent2Child_globalSearchIsOrderIndependent(ks):
  assert ks["pg"].angle(calcParent2Child(ks["mori"][ks["perm"]], ks["p2c0"], **opt)) < 1e-3 * degree


def test_calcParent2Child_globalSearchFindsTheRelationshipFrom25DegreesOff(ks):
  pf = calcParent2Child(ks["mori"], perturb(ks["p2c"], [0, 0, 1], 25 * degree), **opt)
  assert pf.angle(ks["p2c"]) < 0.5 * degree


def test_calcParent2Child_fromPairsOfChildOrientations(ks):
  assert calcParent2Child(ks["pairs"], ks["p2c0"], local=True).angle(ks["p2c"]) < 0.1 * degree


def test_calcParent2Child_refusesARelationshipOfOtherFrames(ks):
  with pytest.raises(ValueError):
    calcParent2Child(ks["mori"], ks["p2c"].inv(), local=True)


# 40 parents with one child each of a known variant, and a grain of five children of one parent
# stored as arbitrary symmetric representatives
@pytest.fixture(scope="module")
def kids():
  rng = np.random.default_rng(1)
  p2c = orientation.KurdjumovSachs(cP, cC)
  V = p2c.variants().flatten()
  parent = orientation.rand(40, cP, rng=rng).projectIntoFundamentalRegion()
  ids = rng.integers(1, 25, 40)
  child = perturb(parent * V[ids - 1].inv(), vector3d.rand(40, rng=rng), 0.3 * degree)
  P = orientation.rand(cP, rng=rng).projectIntoFundamentalRegion()
  ks = np.array([1, 5, 9, 14, 20])
  kids = perturb(P * V[ks - 1].inv(), vector3d.rand(5, rng=rng), 0.5 * degree)
  G = cC.sym.properGroup().rotations().data.reshape(-1, 4)
  kids = kids._new(qmul(kids.data, G[rng.integers(0, len(G), 5)]))
  Pn = P._new(np.broadcast_to(P.data.reshape(1, 4), (5, 4)).copy())
  return dict(p2c=p2c, V=V, parent=parent, ids=ids, child=child, P=P, ks=ks, kids=kids, Pn=Pn)


def test_KurdjumovSachs_has24Variants(kids):
  assert kids["V"].size == 24


def test_calcVariantId_recognisesEveryChildWithItsVariant(kids):
  # one based as MATLAB counts from the parent's representative in the fundamental region
  r = calcVariantId(kids["parent"], kids["child"], kids["p2c"])
  assert np.array_equal(r.variantId, kids["ids"]) and np.max(r.fit) < 0.5 * degree


def test_calcVariantId_nearestVariantIsTheArgminOverTheTableOfAngles(kids):
  p2c, parent, child = kids["p2c"], kids["parent"], kids["child"]
  a = p2c.variants(parent).reshape(child.size, -1).angle(child.reshape(-1, 1))
  k, fit = _nearestVariant(parent, child, p2c)
  assert np.array_equal(k, np.argmin(a, axis=1)) and np.allclose(fit, np.min(a, axis=1), atol=1e-12)


def test_calcVariantId_packetAndBainIdsInRange(kids):
  r = calcVariantId(kids["parent"], kids["child"], kids["p2c"])
  assert np.all((1 <= r.packetId) & (r.packetId <= 4)) and np.all((1 <= r.bainId) & (r.bainId <= 3))


def test_calcVariantId_fourPacketsOfSixAndThreeBainGroupsOfEight(kids):
  p0, V = kids["parent"][0], kids["V"]
  pk = calcVariantId(p0, p0 * V.inv(), kids["p2c"])
  assert np.array_equal(np.bincount(pk.packetId)[1:], [6, 6, 6, 6]) and np.array_equal(np.bincount(pk.bainId)[1:], [8, 8, 8])
  assert np.array_equal(pk.variantId, np.arange(1, 25))


def test_calcVariantId_otherFamiliesNameTheSamePackets(kids):
  p0, V = kids["parent"][0], kids["V"]
  pk = calcVariantId(p0, p0 * V.inv(), kids["p2c"])
  pk2 = calcVariantId(p0, p0 * V.inv(), kids["p2c"], packet=(Miller([[1, 1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, -1]], cP), Miller(0, 1, 1, cC)))
  assert np.array_equal(pk2.packetId, pk.packetId)


def test_calcVariantId_refusesParentsAndChildrenOfOtherCounts(kids):
  with pytest.raises(ValueError):
    calcVariantId(kids["parent"][:3], kids["child"], kids["p2c"])


def test_calcParent_findsTheParentOfAGrainsChildren(kids):
  # whatever representatives the children are stored as
  found, fit = calcParent(kids["kids"], kids["p2c"])
  assert found.angle(kids["P"]) < 1 * degree and fit < 1 * degree and found.frameA is cP and found.frameB is kids["kids"].frameB


def test_calcParent_foundParentGivesTheChildrensVariants(kids):
  found, _ = calcParent(kids["kids"], kids["p2c"])
  r = calcVariantId(found, kids["kids"], kids["p2c"])
  assert np.array_equal(r.variantId, kids["ks"]) and np.max(r.fit) < 1 * degree


def test_calcParent_ofManyChildrenTriesTheCandidatesAsOneAtATime(kids):
  p2c, kids5 = kids["p2c"], kids["kids"]
  PV = p2c.variants(parent=True).flatten()
  w = np.array([1.0, 2, 1, 3, 1]) / 8
  cand = qmul(kids5.data.reshape(-1, 4)[3:4], PV.data.reshape(-1, 4))
  ids, f = calcParent(kids5, p2c, weights=w, id=True)
  ids0, f0 = _parentOfManyLoop(kids5, cand, PV, p2c, w, True)
  assert np.array_equal(ids, ids0) and abs(f - f0) < 1e-12


def test_robustMeans_isTheRobustMeanOfEveryList():
  rng = np.random.default_rng(7)
  q = rng.standard_normal((6, 9, 4)) * [1, 0.05, 0.05, 0.05] + [3, 0, 0, 0]
  q[:, :2] = rng.standard_normal((6, 2, 4))
  q /= np.linalg.norm(q, axis=-1, keepdims=True)
  w = rng.random(9)
  m = _robustMeans(q, w)
  ref = np.stack([quaternion(q[k]).mean(weights=w, robust=True).data for k in range(6)])
  assert np.allclose(np.abs(np.sum(m * ref, axis=1)), 1, atol=1e-12)


def test_rowQuantile_isTheNanQuantileOfEveryRow():
  x = np.random.default_rng(8).random((40, 7))
  x[np.random.default_rng(9).random((40, 7)) < 0.3] = np.nan
  x[0] = np.nan
  assert np.array_equal(_rowQuantile(x, 0.8), np.array([np.nanquantile(r, 0.8) if np.isfinite(r).any() else np.nan for r in x]), equal_nan=True)


def test_calcParent_weighted(kids):
  assert calcParent(kids["kids"], kids["p2c"], weights=[1, 1, 1, 1, 5])[0].angle(kids["P"]) < 1 * degree


def test_calcParent_idNamesTheParentVariantTakingTheChildToItsParent(kids):
  # child with parent: child * PV[id] is that parent
  pid, f = calcParent(kids["kids"], kids["Pn"], kids["p2c"], id=True)
  assert pid.shape == (5,) and np.all(f < 1 * degree)
  PV = kids["p2c"].variants(parent=True).flatten()
  back = orientation(rotation._fromData(qmul(kids["kids"].data, PV.data[pid - 1])), cP)
  assert np.all(back.angle(kids["P"]) < 1 * degree)


def test_calcParent_numFitGivesTheBestFitFirst(kids):
  pid, _ = calcParent(kids["kids"], kids["Pn"], kids["p2c"], id=True)
  ids2, f2 = calcParent(kids["kids"], kids["Pn"], kids["p2c"], numFit=2, id=True)
  assert ids2.shape == (5, 2) and f2.shape == (5, 2) and np.array_equal(ids2[:, 0], pid)


def test_calcParent_withoutIdGivesTheParents(kids):
  # as MATLAB returns them, one parent serving all
  _, f = calcParent(kids["kids"], kids["Pn"], kids["p2c"], id=True)
  parents, f1 = calcParent(kids["kids"], kids["P"], kids["p2c"])
  assert parents.frameA is cP and np.all(parents.angle(kids["P"]) < 1 * degree) and np.allclose(f1, f)


# pairs of children, as at grain boundaries
def childPairs(kids):
  k = kids["kids"]
  return orientation(rotation._fromData(np.stack([k.data[:4], k.data[1:5]], axis=1)), cC)


def test_calcParent_idsOfPairsOfChildren(kids):
  pid, f = calcParent(childPairs(kids), kids["p2c"], id=True)
  assert pid.shape == (4, 2) and np.all(f < 1.5 * degree) and np.all(pid >= 1)


def test_calcParent_parentsOfPairsOfChildren(kids):
  pp, fit = calcParent(childPairs(kids), kids["p2c"])
  assert pp.shape == (4, 2)
  np.testing.assert_allclose(pp[:, 0].angle(pp[:, 1]), fit, rtol=0, atol=1e-12)
  # the first child pair admits several equally good parents
  assert np.all(pp[1:].angle(kids["P"]) < 2 * degree)


def test_calcParent_numFitOfPairsOfChildren(kids):
  ids2, f2 = calcParent(childPairs(kids), kids["p2c"], numFit=2, id=True)
  assert ids2.shape == (4, 2, 2) and f2.shape == (4, 2)


def test_calcParent_idsOfTriplesOfChildren(kids):
  # as at triple points
  k = kids["kids"]
  triples = orientation(rotation._fromData(np.stack([k.data[:3], k.data[1:4], k.data[2:5]], axis=1)), cC)
  pid, f = calcParent(triples, kids["p2c"], id=True)
  assert pid.shape == (3, 3) and np.all(f < 1.5 * degree)


def test_calcParent_turnsARelationshipGivenTheOtherWayRound(kids):
  assert calcParent(kids["kids"], kids["p2c"].inv())[0].angle(kids["P"]) < 1 * degree


def test_round2Miller_orientationGivesItsPlaneAndDirection():
  cs = crystalFrame("m-3m")
  o = orientation.byMiller([1, 1, 2], [-1, -1, 1], cs)
  hkl, uvw = round2Miller(o)
  assert np.array_equal(np.round(hkl.hkl), [1, 1, 2]) and np.array_equal(np.round(uvw.uvw), [-1, -1, 1])
  assert hkl.frame.numSym() == 1 and hkl.dispStyle == "hkl" and uvw.dispStyle == "uvw"


def test_round2Miller_KurdjumovSachsPlanesAndDirections():
  n1, n2, d1, d2 = round2Miller(orientation.KurdjumovSachs(cP, cC))
  assert np.array_equal(np.abs(np.round(n1.hkl)), [1, 1, 1]) and sorted(np.abs(np.round(n2.hkl))) == [0, 1, 1]
  assert sorted(np.abs(np.round(d1.uvw))) == [0, 1, 1] and sorted(np.abs(np.round(d2.uvw))) == [1, 1, 1]
  assert n1.frame is not cP and n1.frame.numSym() == 1 and n2.frame.numSym() == 1


def test_round2Miller_indicesDefineTheRelationshipAgain():
  n1, n2, d1, d2 = round2Miller(orientation.KurdjumovSachs(cP, cC))
  assert orientation.map(Miller(n1, cP), Miller(n2, cC), Miller(d1, cP, "uvw"), Miller(d2, cC, "uvw")).angle(orientation.KurdjumovSachs(cP, cC)) < 1e-6


def test_round2Miller_NishiyamaWassermannPlanesAndDirections():
  n1, n2, d1, d2 = round2Miller(orientation.NishiyamaWassermann(cP, cC))
  assert np.array_equal(np.abs(np.round(n1.hkl)), [1, 1, 1]) and sorted(np.abs(np.round(d1.uvw))) == [0, 1, 1]


def test_round2Miller_ofAList():
  L = round2Miller(orientation.cat(orientation.KurdjumovSachs(cP, cC), orientation.Bain(cP, cC)))
  assert L[0].shape == (2,)


def test_round2Miller_hexagonalGivesFourIndices():
  hx = crystalFrame("6/mmm", [3, 3, 4.7])
  a, b = round2Miller(orientation.byEuler(30 * degree, 40 * degree, 0, hx))
  assert a.dispStyle == "hkil" and b.dispStyle == "UVTW"


def test_allHKL_isCoprimeAndComplete():
  H = allHKL(2)
  assert len(H) == 98 and not np.any(np.all(H == 0, axis=1))


# the fit reports the optimisation unasked, as MATLAB's, and is silent with verbose=False
def test_calcParent2Child_reportsUnlessSilenced(capsys):
  cs1, cs2 = crystalFrame("m-3m", mineral="parent"), crystalFrame("m-3m", mineral="child")
  p2c = orientation.byEuler(10 * degree, 20 * degree, 30 * degree, cs1, cs2)
  mori = orientation.rand(20, cs2, cs2)
  calcParent2Child(mori, p2c, local=True, verbose=False)
  assert capsys.readouterr().out == ""
  calcParent2Child(mori, p2c, local=True)
  assert "optimizing" in capsys.readouterr().out
