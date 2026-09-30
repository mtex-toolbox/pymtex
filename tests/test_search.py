# nearest neighbours in plain lists (check_find)
import numpy as np
import pytest
from scipy import sparse

from mtex import orientation, rotation, crystalFrame, specimenFrame, vector3d, degree, angle, find, isclose, inv

rng = np.random.default_rng(0)
N, M, tol = 120, 10, 1e-9


def checkK(v, w, dRef, k):
  ind, d = find(v, w, k)
  assert ind.shape == (M, k) and d.shape == (M, k)
  assert np.all((ind >= 0) & (ind < N)) and all(len(set(ind[j])) == k for j in range(M))
  assert np.max(np.abs(d - np.sort(dRef, axis=1)[:, :k])) < tol
  assert np.max(np.abs(dRef[np.arange(M)[:, None], ind] - d)) < tol


def checkEps(v, w, dRef, eps):
  I, D = find(v, w, eps)
  assert I.shape == (M, N) and D.shape == (M, N) and sparse.issparse(I) and sparse.issparse(D) and I.dtype == bool
  Id = I.toarray()
  assert np.all(Id[dRef < eps - 1e-6]) and not np.any(Id[dRef > eps + 1e-6]) and np.any(Id)
  assert np.max(np.abs(D.toarray()[Id] - dRef[Id])) < tol and D.nnz <= I.nnz


def radiusAt(ds, frac):
  k = max(0, min(ds.size - 2, int(np.round(frac * ds.size))))
  return (ds[k] + ds[k + 1]) / 2


@pytest.mark.parametrize("case", [("1", "1"), ("321", "1"), ("1", "mmm"), ("432", "mmm"), ("321", "mmm"), ("2", "mmm")])
def test_find_orientations(case):
  cs, ss = crystalFrame(case[0]), specimenFrame(case[1])
  v, w = orientation.rand(N, cs, ss, rng=rng), orientation.rand(M, cs, ss, rng=rng)
  dRef = angle(w[:, None], v[None, :])
  assert dRef.shape == (M, N)
  for k in (1, 3, 7):
    checkK(v, w, dRef, k)
  ind, d = find(v, w)
  assert ind.shape == (M,) and np.max(np.abs(d - np.min(dRef, axis=1))) < tol and np.max(np.abs(dRef[np.arange(M), ind] - d)) < tol
  ds = np.sort(dRef.ravel())
  checkEps(v, w, dRef, radiusAt(ds, 0.02))
  checkEps(v, w, dRef, radiusAt(ds, 0.20))


def test_find_special_cases():
  cs, ss = crystalFrame("321"), specimenFrame("mmm")
  v = orientation.rand(N, cs, ss, rng=rng)
  w = orientation(orientation.rand(M, cs, ss, rng=rng), crystalFrame("321", intern=False), specimenFrame("mmm", intern=False))
  assert v.CS is not w.CS and isclose(v.CS, w.CS)
  ind, d = find(v, w, 3)
  checkK(v, w, angle(w[:, None], v[None, :]), 3)
  with pytest.raises(ValueError):
    find(v, orientation.rand(M, crystalFrame("432"), ss, rng=rng), 3)
  # misorientations, with and without grain exchange
  v, w = orientation.rand(N, cs, cs, rng=rng), orientation.rand(M, cs, cs, rng=rng)
  dRef = angle(w[:, None], v[None, :])
  checkK(v, w, dRef, 3)
  checkEps(v, w, dRef, 15 * degree)
  v.antipodal, w.antipodal = True, True
  dRef = angle(w[:, None], v[None, :])
  vi = orientation(inv(v), cs, cs)
  v0, w0 = orientation(v), orientation(w)
  v0.antipodal, w0.antipodal = False, False
  assert np.max(np.abs(dRef - np.minimum(angle(w0[:, None], v0[None, :]), angle(w0[:, None], vi[None, :])))) < 1e-10
  checkK(v, w, dRef, 3)
  checkEps(v, w, dRef, 15 * degree)
  # exact hits, a rotation as the query, and the shapes of the epsilon form
  v = orientation.rand(N, cs, ss, rng=rng)
  I, D = find(v, v[:5], 5 * degree)
  assert all(I[j, j] for j in range(5)) and all(abs(D[j, j]) < 1e-6 for j in range(5))
  w = orientation.rand(M, cs, ss, rng=rng)
  checkK(v, w, angle(w[:, None], v[None, :]), 3)
  ind, d = find(v, rotation(w), 3)
  assert ind.shape == (M, 3)
  I, D = find(v, orientation.rand(cs, ss, rng=rng), 20 * degree)
  assert I.dtype == bool and I.shape == (1, N)


def test_find_vectors():
  v, w = vector3d.rand(N, rng=rng), vector3d.rand(M, rng=rng)
  dRef = angle(w[:, None], v[None, :])
  ind, d = find(v, w)
  assert np.allclose(d, np.min(dRef, axis=1)) and v.find(w, 3)[0].shape == (M, 3)
  I, D = find(v, w, 20 * degree)
  assert I.shape == (M, N) and np.all(I.toarray() == (dRef <= 20 * degree))
  dA = angle(w[:, None], v[None, :], antipodal=True)
  assert np.allclose(find(v, w, antipodal=True)[1], np.min(dA, axis=1))
