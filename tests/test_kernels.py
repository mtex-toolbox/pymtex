# the compiled kernels agree with their NumPy forms, on every shape the port uses them with
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import mtex
from mtex import crystalFrame, orientation, rotation, setMTEXpref, vector3d
from mtex.geometry import kernels
from mtex.geometry.kernels import (_bestDotNumpy, _closestImageNumpy, _projectDirectionsNumpy, _projectToRefNumpy, _qmulNumpy,
                                   _qrotNumpy, _symReduceNumpy)
from mtex.geometry.orientationregion import commonAxisCenter, commonRotations
from mtex.geometry.symreduce import reductionGroups, referenceImages

rng = np.random.default_rng(3)
numba = pytest.importorskip("numba")


# workqueue is the only threading backend available on some machines, including macOS
def test_parallel_kernels_from_two_threads_with_workqueue():
  code = textwrap.dedent('''
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import numpy as np
    import numba
    from mtex.geometry.kernels import eulerQuat
    from mtex.io.kernels import parseBlockNumba

    angles = np.zeros(50000)
    block = b"1 2 3\\n" * len(angles)
    def rotations():
      q = eulerQuat(angles, angles, angles)
      assert np.all(q[:, 0] == 1) and np.all(q[:, 1:] == 0)
    def table():
      assert np.all(parseBlockNumba(block, False, None) == [1, 2, 3])
    rotations()
    table()
    assert numba.threading_layer() == "workqueue"
    barrier = Barrier(2)
    def run(f):
      for _ in range(12):
        barrier.wait(timeout=30)
        f()
    with ThreadPoolExecutor(2) as pool:
      futures = [pool.submit(run, f) for f in (rotations, table)]
      for future in futures:
        future.result(timeout=60)
  ''')
  env = dict(os.environ, NUMBA_THREADING_LAYER="workqueue", NUMBA_NUM_THREADS="2")
  result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
  assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def numbaBackend():
  setMTEXpref("kernels", "numba")
  assert kernels.backend() == "numba"
  yield
  setMTEXpref("kernels", "numba")


def test_backend_follows_the_preference():
  setMTEXpref("kernels", "numpy")
  assert kernels.backend() == "numpy"
  setMTEXpref("kernels", "numba")
  assert kernels.backend() == "numba"


def test_product_and_rotation_agree_with_numpy(numbaBackend):
  p, q = rotation.rand(1000, rng=rng).data, rotation.rand(1000, rng=rng).data
  v = vector3d.rand(1000, rng=rng).data
  assert np.allclose(kernels.qmul(p, q), _qmulNumpy(p, q), atol=1e-15)
  assert np.allclose(kernels.qrot(p, v), _qrotNumpy(p, v), atol=1e-15)
  # broadcasting on the leading axes, a single element, a strided slice, an integer input
  assert np.allclose(kernels.qmul(p[:, None, :], q[None, :5, :]), _qmulNumpy(p[:, None, :], q[None, :5, :]))
  assert np.allclose(kernels.qrot(p[:, None, :], v[None, :7, :]), _qrotNumpy(p[:, None, :], v[None, :7, :]))
  assert np.allclose(kernels.qmul(p[0], q[0]), _qmulNumpy(p[0], q[0])) and kernels.qmul(p[0], q[0]).shape == (4,)
  wide = np.concatenate([p, np.ones((1000, 1))], axis=1)
  assert np.allclose(kernels.qmul(wide[:, :4], q), _qmulNumpy(p, q))
  assert np.allclose(kernels.qmul(np.array([1, 0, 0, 0]), q), q)
  assert kernels.qmul(p.astype(np.float32), q.astype(np.float32)).dtype == np.float32
  assert kernels.qmul(np.empty((0, 4)), np.empty((0, 4))).shape == (0, 4)
  # one quaternion times a map's worth, the constant left factor not copied out
  big = rotation.rand(70000, rng=rng).data.reshape(700, 100, 4)
  assert np.allclose(kernels.qmul(p[0], big), _qmulNumpy(p[0], big), atol=1e-15) and kernels.qmul(p[:1], big).shape == (700, 100, 4)
  assert kernels.qmul(p[0].astype(np.float32), big.astype(np.float32)).dtype == np.float32


def test_searches_agree_with_numpy(numbaBackend):
  cs, hx = crystalFrame("m-3m"), crystalFrame("6/mmm")
  for frames, exchange in [((cs,), False), ((cs, hx), False), ((cs, cs), True)]:
    o1, o2 = orientation.rand(500, *frames, rng=rng), orientation.rand(500, *frames, rng=rng)
    L, R = reductionGroups(o1)
    es, _ = referenceImages(o2[0].data, L, R, exchange)
    val, idx = kernels.bestDot(o1.data, es)
    valN, idxN = _bestDotNumpy(o1.data, es)
    assert np.allclose(val, valN, atol=1e-14) and np.array_equal(idx, idxN)
    got = kernels.symReduce(o1.data, o2.data, L, R, exchange)
    want = _symReduceNumpy(o1.data, o2.data, L, R, exchange)
    assert np.allclose(got[0], want[0], atol=1e-14)
    assert all(np.array_equal(g, w) for g, w in zip(got[1:], want[1:]))
    # one reference against a list, and a single pair
    got = kernels.symReduce(o2[0].data, o1.data, L, R, exchange)
    want = _symReduceNumpy(np.broadcast_to(o2[0].data, o1.data.shape), o1.data, L, R, exchange)
    assert np.allclose(got[0], want[0], atol=1e-14) and got[0].shape == (500,)
    assert kernels.symReduce(o1[0].data, o2[0].data, L, R, exchange)[0].shape == ()
    # the pair search with its image, against a list and against one reference
    vals, img = kernels.closestImage(o1.data, o2.data, L, R, exchange)
    valsN, imgN = _closestImageNumpy(o1.data, o2.data, L, R, exchange)
    assert np.allclose(vals, valsN, atol=1e-14) and np.allclose(img, imgN, atol=1e-14)
    assert np.all(np.sum(img * o1.data, axis=-1) >= 0)
    assert kernels.closestImage(o2[0].data, o1.data, L, R, exchange)[1].shape == (500, 4)
    # the projection onto one reference, and onto the identity where the conjugates tie
    for ref in (o2[0].data, np.array([1.0, 0, 0, 0])):
      es, frm = referenceImages(ref, L, R, exchange)
      common = commonRotations(L, R) if ref[0] == 1 else None
      cimgs = kernels.qrot(common, commonAxisCenter(common)[None]) if common is not None and len(common) > 1 else None
      got = kernels.projectToRef(o1.data, es, frm, L, R, ref, common, cimgs)
      want = _projectToRefNumpy(o1.data, es, frm, L, R, ref, np.empty((0, 4)) if cimgs is None else common, np.empty((0, 3)) if cimgs is None else cimgs)
      assert np.allclose(got, want, atol=1e-13)


# the searches split over the threads give the serial result to the bit
def test_searches_over_threads_equal_the_serial_loop(numbaBackend, monkeypatch):
  cs, hx = crystalFrame("m-3m"), crystalFrame("6/mmm")
  o1, o2 = orientation.rand(3000, cs, hx, rng=rng), orientation.rand(3000, cs, hx, rng=rng)
  L, R = reductionGroups(o1)
  es, frm = referenceImages(o2[0].data, L, R, False)
  v = vector3d.rand(3000, rng=rng).data
  o = cs.rot.data.reshape(-1, 5)
  imgs = kernels.qrot(o[:, :4], np.array([[0.3, 0.2, 0.9]])) * o[:, 4:5]
  calls = lambda: [*kernels.symReduce(o1.data, o2.data, L, R), *kernels.closestImage(o1.data, o2.data, L, R),
                   *kernels.bestDot(o1.data, es), kernels.projectToRef(o1.data, es, frm, L, R, o2[0].data),
                   kernels.projectDirections(v, imgs, o[:, :4], o[:, 4]), kernels.variantAngles(o1.data[:50], L[:6], R)]
  monkeypatch.setattr(kernels, "PARALLEL_WORK", 1 << 62)
  serial = calls()
  monkeypatch.setattr(kernels, "PARALLEL_WORK", 0)
  monkeypatch.setattr(kernels, "threads", lambda: 4)
  assert all(np.array_equal(a, b) for a, b in zip(calls(), serial))


def test_direction_projection_agrees_with_numpy(numbaBackend):
  for name in ("m-3m", "-3m", "4/m", "-6m2"):
    fr = crystalFrame(name)
    o = fr.rot.data.reshape(-1, 5)
    v = vector3d.rand(2000, rng=rng).data
    imgs = kernels.qrot(o[:, :4], np.array([[0.3, 0.2, 0.9]])) * o[:, 4:5]
    got = kernels.projectDirections(v, imgs, o[:, :4], o[:, 4])
    assert np.allclose(got, _projectDirectionsNumpy(v, imgs, o[:, :4], o[:, 4]), atol=1e-14)
    assert kernels.projectDirections(v[0], imgs, o[:, :4], o[:, 4]).shape == (3,)


def test_the_public_operations_agree_between_backends():
  cs, hx = crystalFrame("m-3m"), crystalFrame("6/mmm")
  mo, mo2 = orientation.rand(300, cs, hx, rng=rng), orientation.rand(300, cs, hx, rng=rng)
  r, v = rotation.rand(300, rng=rng), vector3d.rand(300, rng=rng)
  out = {}
  for name in ("numpy", "numba"):
    setMTEXpref("kernels", name)
    out[name] = [(r * r).data, (r * v).data, mtex.dot(mo, mo2), mtex.dot(mo, mo[0]), mtex.angle(mo, mo2),
                 mo.projectIntoFundamentalRegion(mo2[0]).data, mo.projectIntoFundamentalRegion(mo2).data, mtex.mean(mo).data,
                 mo.projectIntoFundamentalRegion().data, orientation(mo.data, cs, cs).projectIntoFundamentalRegion().data,
                 mtex.angle(mo), mtex.Miller(v, cs).projectIntoFundamentalRegion().data]
  setMTEXpref("kernels", "numba")
  for a, b in zip(out["numpy"], out["numba"]):
    assert np.allclose(a, b, atol=1e-12)


def test_euler_angles_agree_with_numpy(numbaBackend):
  from mtex.geometry.kernels import eulerQuat
  from mtex.geometry.representations import matthies2quat
  rng = np.random.default_rng(3)
  a, b, g = (rng.uniform(-7, 7, size=(40, 60)) for _ in range(3))
  q = eulerQuat(a, b, g)
  setMTEXpref("kernels", "numpy")
  assert np.allclose(q, matthies2quat(a, b, g), atol=1e-15)


# blasThreads sets the thread count of every loaded OpenBLAS for a block and restores it
def test_blas_threads_are_set_and_restored():
  import scipy.linalg  # noqa: F401 - SciPy's own OpenBLAS beside NumPy's
  from mtex.geometry.kernels import _openblasLibraries, blasThreads
  libs = _openblasLibraries()
  before = [g() for g, _ in libs]
  with blasThreads(1):
    assert all(g() == 1 for g, _ in libs)
  assert [g() for g, _ in libs] == before


def test_region_membership_agrees_with_numpy(numbaBackend):
  # the planes of a fundamental region turned by the rotations about z, as the SO(3) grid tests
  from mtex import fundamentalRegion
  from mtex.geometry.kernels import _regionMembershipNumpy
  N = fundamentalRegion(crystalFrame("m-3m")).N
  z = rotation.byAxisAngle(vector3d(0, 0, 1), np.arange(4) * np.pi / 2).data * np.array([1, -1, -1, -1])
  NB = kernels.qmul(z[:, None, :], N[None]).reshape(-1, 4)
  q = rotation.rand(5000, rng=rng).data
  keep, near = kernels.regionMembership(q, NB, 4, 1e-3, 0.03)
  k0, n0 = _regionMembershipNumpy(q, NB, 4, 1e-3, 0.03)
  assert np.array_equal(keep, k0) and np.array_equal(near, n0) and keep.any() and near.any() and not near.all()
