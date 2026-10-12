# the symmetry reduced angle and dot: the batched kernel against the pair kernel
import numpy as np

from mtex import orientation, crystalFrame, angle, dot
from mtex.geometry import symreduce as sr
from mtex.geometry.kernels import closestImage


def test_batched_equals_pairwise():
  cs = crystalFrame("m-3m")
  rng = np.random.default_rng(11)
  ref = orientation.rand(cs, rng=rng)
  for n in (10, 300):
    oris = orientation.rand(n, cs, rng=rng)
    L, R = sr.reductionGroups(oris)
    es, frm = sr.referenceImages(ref.data, L, R, False)
    assert es.shape == (24, 4) and frm.shape == (24, 3)
    sd, k = sr.bestDot(oris.data, es)
    sa = sr.chordAngle(oris.data, es[k])
    assert np.max(np.abs(sr.symDots(oris.data, ref.data, L, R) - sd)) < 1e-12
    assert np.max(np.abs(sr.symAngles(oris.data, ref.data, L, R) - sa)) < 1e-12
    # the projected representative is equivalent to its orientation and closest to the reference
    p = oris.projectIntoFundamentalRegion(ref)
    assert np.max(np.abs(sr.chordAngle(p.data, ref.data) - sa)) < 1e-12
    assert np.min(sr.symReduce(oris.data, p.data, L, R)[0]) > 1 - 1e-10
    # list against list: the pair kernel and the scalar loop agree
    other = orientation.rand(n, cs, rng=rng)
    vals, li, rj, usedInv = sr.symReduce(oris.data, other.data, L, R)
    loop = np.array([sr.symReduce(oris.data[i], other.data[i], L, R)[0] for i in range(n)])
    assert np.max(np.abs(vals - loop)) < 1e-12
    assert np.max(np.abs(dot(oris, other) - vals)) < 1e-12
    img = closestImage(oris.data, other.data, L, R)[1]
    assert np.max(np.abs(angle(oris, other) - sr.chordAngle(oris.data, img))) < 1e-12
  # the identity keeps only the distinct images
  es, frm = sr.referenceImages(np.array([1.0, 0, 0, 0]), L, R, False)
  assert es.shape == (24, 4)
  es, frm = sr.referenceImages(np.array([1.0, 0, 0, 0]), L, R, True)
  assert es.shape == (24, 4)


# an element joins the earlier one it equals only when that one stays
def test_firstOfClusters_keepsTheFirstOfEveryChain():
  same = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)
  assert sr._firstOfClusters(same).tolist() == [True, False, True]


def test_firstOfClusters_ofNothingIsEmpty():
  assert sr._firstOfClusters(np.zeros((0, 0), dtype=bool)).shape == (0,)


def test_distinctUnitRows_dropsTheRepeatsOfEarlierRows():
  X = np.array([[1.0, 0, 0], [0, 1, 0], [1, 1e-9, 0], [0, 0, 1]])
  assert np.array_equal(sr._distinctUnitRows(X, 1e-6), X[[0, 1, 3]])


def test_isAboutZ_holdsTheIdentityAndTheTurnsAboutZ():
  q = np.array([[1.0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
  assert sr._isAboutZ(q).tolist() == [True, True, False]


def test_chordAngle_isImportableWhereOtherLayersTakeIt():
  from mtex.geometry.representations import chordAngle
  assert sr.chordAngle is chordAngle
