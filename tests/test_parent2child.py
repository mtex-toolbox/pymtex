# G7: parent to child orientation relationships, MATLAB's check_calcParent2Child and the
# Julia parent2child.jl numbers
import numpy as np
import pytest

from mtex import Miller, calcParent, calcParent2Child, calcVariantId, crystalFrame, degree, orientation, rotation, round2Miller, vector3d
from mtex.geometry.parent2child import allHKL, misfit
from mtex.geometry.quaternion import qmul

csP = crystalFrame("432", [3.66, 3.66, 3.66], mineral="fcc")
csC = crystalFrame("432", [2.87, 2.87, 2.87], mineral="bcc")


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


def test_calcParent2Child():
  rng = np.random.default_rng(0)
  p2c = orientation.KurdjumovSachs(csP, csC)
  mori = c2cData(rng, p2c, 1500, 1 * degree, 0.2)
  p2c0 = perturb(p2c, [1, 2, 3], 3 * degree)
  # recovery from a local start
  p = calcParent2Child(mori, p2c0, local=True)
  assert p.angle(p2c) < 0.3 * degree and p.angle(p2c) < p2c0.angle(p2c)
  # the misfit of every misorientation against the result, most within the noise, also
  # when the fit subsamples
  fit = misfit(mori, p)
  assert len(fit) == mori.size and np.quantile(fit, 0.75) < 3 * degree
  q, fit = calcParent2Child(mori, p2c0, local=True, maxSample=300, fit=True)
  assert len(fit) == mori.size and q.angle(p) < 0.3 * degree
  # independence of the order of the data
  perm = rng.permutation(mori.size)
  assert p.angle(calcParent2Child(mori[perm], p2c0, local=True)) < 1e-3 * degree
  # the global search is deterministic and order independent, and finds the relationship
  opt = dict(searchResolution=10 * degree, numLocal=1, maxSample=300, scanSample=100)
  pg = calcParent2Child(mori, p2c0, **opt)
  assert pg.angle(calcParent2Child(mori, p2c0, **opt)) == 0
  assert pg.angle(calcParent2Child(mori[perm], p2c0, **opt)) < 1e-3 * degree
  far = perturb(p2c, [0, 0, 1], 25 * degree)
  pf = calcParent2Child(mori, far, searchResolution=6 * degree, numLocal=3, maxSample=600, scanSample=150)
  assert pf.angle(p2c) < 0.5 * degree
  # pairs of child orientations instead of misorientations, and the frame check
  parent = orientation.rand(200, csP, rng=rng)
  V = p2c.variants().flatten()
  pairs = orientation(rotation._fromData(np.stack([(parent * V[rng.integers(0, 24, 200)].inv()).data,
                                                  (parent * V[rng.integers(0, 24, 200)].inv()).data], axis=1)), csC)
  assert calcParent2Child(pairs, p2c0, local=True).angle(p2c) < 0.1 * degree
  with pytest.raises(ValueError):
    calcParent2Child(mori, p2c.inv(), local=True)


def test_variant_ids_and_the_parent_of_children():
  rng = np.random.default_rng(1)
  cP = crystalFrame("m-3m", [3.66, 3.66, 3.66], mineral="fcc")
  cC = crystalFrame("m-3m", [2.87, 2.87, 2.87], mineral="bcc")
  p2c = orientation.KurdjumovSachs(cP, cC)
  V = p2c.variants().flatten()
  assert V.size == 24
  # every child of a parent is recognised with its variant, one based as MATLAB counts
  parent = orientation.rand(40, cP, rng=rng)
  ids = rng.integers(1, 25, 40)
  child = perturb(parent * V[ids - 1].inv(), vector3d.rand(40, rng=rng), 0.3 * degree)
  r = calcVariantId(parent, child, p2c)
  assert np.array_equal(r.variantId, ids) and np.max(r.fit) < 0.5 * degree
  assert np.all((1 <= r.packetId) & (r.packetId <= 4)) and np.all((1 <= r.bainId) & (r.bainId <= 3))
  # four packets of six variants and three Bain groups of eight, one parent for all
  pk = calcVariantId(parent[0], parent[0] * V.inv(), p2c)
  assert np.array_equal(np.bincount(pk.packetId)[1:], [6, 6, 6, 6]) and np.array_equal(np.bincount(pk.bainId)[1:], [8, 8, 8])
  assert np.array_equal(pk.variantId, np.arange(1, 25))
  # other families name the groups
  pk2 = calcVariantId(parent[0], parent[0] * V.inv(), p2c, packet=(Miller([[1, 1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, -1]], cP), Miller(0, 1, 1, cC)))
  assert np.array_equal(pk2.packetId, pk.packetId)
  with pytest.raises(ValueError):
    calcVariantId(parent[:3], child, p2c)
  # the parent of a grain's children, whatever representatives the children are stored as
  P = orientation.rand(cP, rng=rng)
  ks = np.array([1, 5, 9, 14, 20])
  kids = perturb(P * V[ks - 1].inv(), vector3d.rand(5, rng=rng), 0.5 * degree)
  G = cC.sym.properGroup().rotations().data.reshape(-1, 4)
  kids = kids._new(qmul(kids.data, G[rng.integers(0, len(G), 5)]))
  found, fit = calcParent(kids, p2c)
  assert found.angle(P) < 1 * degree and fit < 1 * degree and found.frameA is cP and found.frameB is kids.frameB
  r = calcVariantId(found, kids, p2c)
  assert np.array_equal(r.variantId, ks) and np.max(r.fit) < 1 * degree
  assert calcParent(kids, p2c, weights=[1, 1, 1, 1, 5])[0].angle(P) < 1 * degree
  # child with parent: the id names the parent variant the child as stored is taken to
  # its parent by, so child * PV[id] is that parent
  Pn = P._new(np.broadcast_to(P.data.reshape(1, 4), (5, 4)).copy())
  pid, f = calcParent(kids, Pn, p2c)
  assert pid.shape == (5,) and np.all(f < 1 * degree)
  PV = p2c.variants(parent=True).flatten()
  back = orientation(rotation._fromData(qmul(kids.data, PV.data[pid - 1])), cP)
  assert np.all(back.angle(P) < 1 * degree)
  ids2, f2 = calcParent(kids, Pn, p2c, numFit=2)
  assert ids2.shape == (5, 2) and f2.shape == (5, 2) and np.array_equal(ids2[:, 0], pid)
  # pairs of children, as at grain boundaries, and triples, as at triple points
  pairs = orientation(rotation._fromData(np.stack([kids.data[:4], kids.data[1:5]], axis=1)), cC)
  pid, f = calcParent(pairs, p2c)
  assert pid.shape == (4, 2) and np.all(f < 1.5 * degree) and np.all(pid >= 1)
  ids2, f2 = calcParent(pairs, p2c, numFit=2)
  assert ids2.shape == (4, 2, 2) and f2.shape == (4, 2)
  triples = orientation(rotation._fromData(np.stack([kids.data[:3], kids.data[1:4], kids.data[2:5]], axis=1)), cC)
  pid, f = calcParent(triples, p2c)
  assert pid.shape == (3, 3) and np.all(f < 1.5 * degree)
  # a relationship given the other way round is turned
  assert calcParent(kids, p2c.inv())[0].angle(P) < 1 * degree


def test_round2Miller():
  cs = crystalFrame("m-3m")
  o = orientation.byMiller([1, 1, 2], [-1, -1, 1], cs)
  hkl, uvw = round2Miller(o)
  assert np.array_equal(np.round(hkl.hkl), [1, 1, 2]) and np.array_equal(np.round(uvw.uvw), [-1, -1, 1])
  assert hkl.frame.numSym() == 1 and hkl.dispStyle == "hkl" and uvw.dispStyle == "uvw"
  cP = crystalFrame("m-3m", [3.66, 3.66, 3.66], mineral="fcc")
  cC = crystalFrame("m-3m", [2.87, 2.87, 2.87], mineral="bcc")
  n1, n2, d1, d2 = round2Miller(orientation.KurdjumovSachs(cP, cC))
  assert np.array_equal(np.abs(np.round(n1.hkl)), [1, 1, 1]) and sorted(np.abs(np.round(n2.hkl))) == [0, 1, 1]
  assert sorted(np.abs(np.round(d1.uvw))) == [0, 1, 1] and sorted(np.abs(np.round(d2.uvw))) == [1, 1, 1]
  assert n1.frame is not cP and n1.frame.numSym() == 1 and n2.frame.numSym() == 1
  # the rotation the four indices define is the relationship again
  assert orientation.map(Miller(n1, cP), Miller(n2, cC), Miller(d1, cP, "uvw"), Miller(d2, cC, "uvw")).angle(orientation.KurdjumovSachs(cP, cC)) < 1e-6
  n1, n2, d1, d2 = round2Miller(orientation.NishiyamaWassermann(cP, cC))
  assert np.array_equal(np.abs(np.round(n1.hkl)), [1, 1, 1]) and sorted(np.abs(np.round(d1.uvw))) == [0, 1, 1]
  # a list, and the four index styles of a hexagonal crystal
  L = round2Miller(orientation.cat(orientation.KurdjumovSachs(cP, cC), orientation.Bain(cP, cC)))
  assert L[0].shape == (2,)
  hx = crystalFrame("6/mmm", [3, 3, 4.7])
  a, b = round2Miller(orientation.byEuler(30 * degree, 40 * degree, 0, hx))
  assert a.dispStyle == "hkil" and b.dispStyle == "UVTW"
  # the index table is coprime and complete
  H = allHKL(2)
  assert len(H) == 98 and not np.any(np.all(H == 0, axis=1))
