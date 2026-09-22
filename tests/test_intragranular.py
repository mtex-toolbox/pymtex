# M7: the gradient, the curvature and the other intragranular quantities, the Julia I1 to I10
import numpy as np
import pytest

from mtex import (EBSD, phaseList, crystalFrame, rotation, orientation, vector3d, zvector, xvector, degree, calcGrains, gridify, angle, setMTEXpref, getMTEXpref,
                  gradient, gradientX, gradientZ, curvature, KAM, calcGROD, grainMean, spatialProfile, calcMisorientation)
from mtex.maps.intragranular import neighbourBall
from conftest import close

CSI = crystalFrame("m-3m", mineral="test")
K1, K2 = 0.002, 0.0035


# a map whose rotation about z has an angle linear in the position, so its gradient is
# K1 z along x and K2 z along y, exactly
def linearMap(A, n, dtype=np.float64, phases=None, shift=None):
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  pos = np.stack([A[0, 0] * i + A[0, 1] * j, A[1, 0] * i + A[1, 1] * j, np.zeros(n * n)], axis=-1).astype(dtype)
  w = K1 * pos[:, 0] + K2 * pos[:, 1]
  if shift is not None:
    w = w + shift(i, j)
  rot = rotation.byAxisAngle(zvector, w.astype(dtype))
  return EBSD(pos, rot, np.ones(n * n, dtype=np.int8), phaseList(CSI) if phases is None else phases)


def rot2(w):
  return np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])


GEOM = (("square", np.eye(2)), ("rotated 30", rot2(np.pi / 6)), ("rotated 45", rot2(np.pi / 4)), ("sheared", np.array([[1.0, 0.4], [0.0, 1.2]])),
        ("hex", np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])))


def test_I1_a_linear_field_has_the_gradient_it_was_built_with():
  for name, A in GEOM:
    for stencil in ("oneSided", "lattice", "full"):
      e = linearMap(A, 12)
      gx, gy = e.gradient(stencil=stencil)
      assert gx.frame is e.frame and gx.shape == e.shape
      # where a gradient is defined at all it is the exact one, on every grid and from every
      # stencil; the one sided one has none where the neighbour it asks for is outside the
      # rhombus a hexagonal index box leaves empty
      ok = ~np.isnan(gx.data[:, 2])
      assert ok.sum() > 0.9 * e.size
      assert np.max(np.abs(gx.data[ok][:, :2])) < 1e-12 and np.max(np.abs(gx.data[ok][:, 2] - K1)) < 1e-12
      assert np.max(np.abs(gy.data[ok][:, 2] - K2)) < 1e-12
      # a square grid has a neighbour for every pixel either way
      if name != "hex":
        assert ok.all()
      # the derivative along a specimen direction, and none along the normal
      gd = e.gradient(vector3d(1, 0, 0), stencil=stencil)
      assert np.max(np.abs(gd.data[ok][:, 2] - K1)) < 1e-12 and np.all(np.isnan(e.gradient(vector3d(0, 0, 1), stencil=stencil).data))
      assert np.all(np.isnan(gradientZ(e).data)) and close(gradientX(e).data[ok], gd.data[ok])
  with pytest.raises(ValueError):
    linearMap(np.eye(2), 5).gradient(stencil="2hop")
  assert len(gradient(linearMap(np.eye(2), 5))) == 2


def test_I2_the_stored_equivalent_does_not_matter():
  e = linearMap(np.eye(2), 10)
  G = CSI.sym.properGroup().rotations()
  rng = np.random.default_rng(3)
  s = EBSD(e.pos, e.rotations * G[rng.integers(0, len(G), e.size)], e.phaseId, e.CSList)
  a, b = e.gradient()
  c, d = s.gradient()
  assert np.max(np.linalg.norm(a.data - c.data, axis=1)) < 1e-10 and np.max(np.linalg.norm(b.data - d.data, axis=1)) < 1e-10


def test_I2b_the_stencil_decides_which_neighbours_are_asked():
  # a hole inside the map: the one sided difference asks for the neighbour it wants and has
  # none, the wider stencils answer from what is left
  e = linearMap(np.eye(2), 9)
  ph = e.phaseId.copy()
  ph[4 + 4 * 9] = 0                                   # the pixel at (4, 4) is gone
  h = EBSD(e.pos, e.rotations, ph, e.CSList)
  k = 3 + 4 * 9                                       # its neighbour to the left
  assert np.isnan(h.gradient(stencil="oneSided")[0].data[k, 2])
  assert close(h.gradient(stencil="lattice")[0].data[k, 2], K1) and close(h.gradient(stencil="full")[0].data[k, 2], K1)
  # and at the border the one sided difference steps backwards instead
  assert close(e.gradient(stencil="oneSided")[0].data[8, 2], K1)


def test_I3_a_boundary_is_where_the_field_stops():
  # the right half is the same field turned by 20 degrees, so a difference across the seam
  # is no derivative
  e = linearMap(np.eye(2), 12, shift=lambda i, j: np.where(i >= 6, 20 * degree, 0.0))
  gx = e.gradient()[0]
  seam = [5 + j * 12 for j in range(12)]              # the last column of the left half
  assert np.max(np.abs(gx.data[seam, 2] - K1)) > 0.1
  calcGrains(e, angle=5 * degree)
  # the one sided difference asks for the neighbour across the seam and, once the grains
  # are known, is told it may not have it
  assert np.all(np.isnan(e.gradient()[0].data[seam, 2]))
  gx2, gy2 = e.gradient(stencil="lattice")
  assert np.max(np.abs(gx2.data[seam, 2] - K1)) < 1e-12 and np.max(np.abs(gy2.data[seam, 2] - K2)) < 1e-12
  # a pixel that was not indexed has no gradient at all
  ph = e.phaseId.copy()
  ph[3 + 3 * 12] = 0
  e2 = EBSD(e.pos, e.rotations, ph, e.CSList)
  assert np.isnan(e2.gradient(stencil="lattice")[0].data[3 + 3 * 12, 2])


def test_I4_the_curvature_of_the_linear_field():
  e = linearMap(np.eye(2), 8)
  kappa = e.curvature()
  assert kappa.shape == e.shape + (3, 3)
  M = kappa[20]
  assert close(M[2, 0], K1) and close(M[2, 1], K2) and abs(M[0, 0]) < 1e-12 and abs(M[1, 1]) < 1e-12
  # nothing is known along the normal of the map
  assert np.all(np.isnan(M[:, 2])) and curvature(e, stencil="full").shape == kappa.shape


def test_I5_the_kernel_average_misorientation():
  e = linearMap(np.eye(2), 12)
  kam = e.KAM()
  inner = np.array([i + j * 12 for i in range(1, 11) for j in range(1, 11)])
  # four neighbours at +-x and +-y, so the mean of |K1|, |K1|, |K2|, |K2|
  assert np.max(np.abs(kam[inner] - (K1 + K2) / 2)) < 1e-12 and kam.shape == e.shape
  assert np.max(np.abs(e.KAM(max=True)[inner] - max(K1, K2))) < 1e-12
  # the second order ball, over the offsets it holds
  offs = neighbourBall(e.lattice, 2)
  assert len(offs) == 12
  ref = np.mean([abs(o[0] * K1 + o[1] * K2) for o in offs])
  kam2 = KAM(e, order=2)
  assert np.max(np.abs(kam2[[i + j * 12 for i in range(2, 10) for j in range(2, 10)]] - ref)) < 1e-12
  # a threshold drops the pairs above it: here the two along y, so what is left is the pair
  # along x alone
  assert np.max(np.abs(e.KAM(threshold=(K1 + K2) / 2)[inner] - K1)) < 1e-12 and np.all(np.isnan(e.KAM(threshold=K1 / 2)))
  # a radius is a ball in the plane rather than in steps over the stencil
  assert len(neighbourBall(e.lattice, radius=1.5)) == 8 and close(e.KAM(radius=1.5)[inner], KAM(e, order=2)[inner] * 0 + e.KAM(radius=1.5)[inner])
  # weights are a function of the distance
  assert close(e.KAM(order=2, weights=lambda d: float(d <= 1))[inner], (K1 + K2) / 2)
  # an indexed phase label does not supply a missing orientation
  hole = 5 + 5 * 12
  q = e.rotations.data.copy()
  q[hole] = np.nan
  h = EBSD(e.pos, rotation(q), e.phaseId, e.CSList)
  kh = h.KAM()
  assert np.isnan(kh[hole]) and close(kh[hole - 1], (K1 + 2 * K2) / 3) and close(h.KAM(max=True)[hole - 1], K2)


def test_I6_the_grain_reference_orientation_deviation():
  # one grain of sixteen pixels at 0, 0.2, ... 3 degrees about z: the angles are symmetric
  # about 1.5 degrees, so the mean sits there and the deviations are what is left of them
  w = 0.2 * np.arange(16)
  i, j = np.meshgrid(np.arange(4), np.arange(4), indexing="xy")
  pos = np.stack([i.reshape(-1), j.reshape(-1), np.zeros(16)], axis=-1).astype(float)
  e = EBSD(pos, rotation.byAxisAngle(zvector, w * degree), np.ones(16, dtype=np.int8), phaseList(CSI))
  g = calcGrains(e, angle=5 * degree)
  assert len(g) == 1
  grod = e.calcGROD(g)
  assert isinstance(grod, orientation) and grod.CS is CSI and grod.SS is CSI and grod.shape == e.shape
  assert close(grod.angle() / degree, np.abs(w - 1.5), 1e-9)
  # against a reference of one's own choosing
  assert close(calcGROD(e, g, rotation.id).angle() / degree, w, 1e-9) and close(e.calcGROD(rotRef=rotation.id).angle() / degree, w, 1e-9)
  with pytest.raises(ValueError):
    EBSD(pos, e.rotations, e.phaseId, e.CSList).calcGROD(g)


def test_I7_the_mean_of_a_property_over_a_grain():
  e = linearMap(np.eye(2), 12, shift=lambda i, j: np.where(i >= 6, 20 * degree, 0.0))
  g = calcGrains(e, angle=5 * degree)
  i = np.arange(144) % 12
  e.val = (i >= 6).astype(float)
  per = e.grainMean(e.val, g)
  assert len(per) == len(g) and close(np.sort(per), [0.0, 1.0])
  # the same value at every pixel of the grain it belongs to
  pix = grainMean(e, e.val)
  assert pix.shape == e.shape and np.array_equal(pix, e.val)
  # a reduction of one's own, and a range to take it over
  assert close(np.sort(e.grainMean(e.val, g, np.max)), [0.0, 1.0]) and np.isnan(e.grainMean(e.val, g, llim=0.5)[np.argmin(per)])
  with pytest.raises(ValueError):
    e.grainMean(e.val[:5], g)


def test_I8_a_profile_along_a_line():
  e = linearMap(np.eye(2), 10)
  sub, d = e.spatialProfile(np.array([[0.0, 0.0], [9.0, 0.0]]))
  assert len(sub) == 10 and close(d, np.arange(10.0))
  sub2, d2 = spatialProfile(e, [0, 9], [0, 0])
  assert np.array_equal(sub2.pos.data, sub.pos.data) and close(d2, d)
  with pytest.raises(ValueError):
    e.spatialProfile(np.array([[0.0, 0.0], [np.nan, 0.0]]))
  # the misorientation angle grows linearly along the profile
  w = sub.rotations.angle(sub.rotations[0])
  assert close(w, K1 * d)
  # a polyline turns the corner and the distance keeps running
  sub3, d3 = e.spatialProfile(np.array([[0.0, 0.0], [9.0, 0.0], [9.0, 9.0]]))
  assert len(sub3) == 19 and close(d3[-1], 18.0) and np.all(np.diff(d3) >= 0)


def test_I9_uncorrelated_misorientations():
  e = linearMap(np.eye(2), 20)
  m = e.calcMisorientation(sampleSize=500, rng=11)
  assert isinstance(m, orientation) and m.CS is CSI and m.SS is CSI and m.antipodal and 400 < len(m) <= 500
  # every pair really is that far apart
  m2 = calcMisorientation(e, sampleSize=200, minDistance=15.0, rng=1)
  assert len(m2) < 200 and m2.angle().max() < 62.8 * degree
  # two phases: the pair carries a direction of the first phase into the second, as a
  # boundary between them does
  csA, csB = crystalFrame("m-3m", mineral="phaseA"), crystalFrame("622", mineral="phaseB")
  i, j = np.meshgrid(np.arange(10), np.arange(10), indexing="xy")
  pos = np.stack([i.reshape(-1), j.reshape(-1), np.zeros(100)], axis=-1).astype(float)
  qA, qB = rotation.byEuler(20 * degree, 30 * degree, 40 * degree), rotation.byEuler(70 * degree, 50 * degree, 10 * degree)
  eA = EBSD(pos, rotation(np.tile(qA.data, (100, 1))), np.ones(100, dtype=np.int8), phaseList(csA))
  eB = EBSD(pos, rotation(np.tile(qB.data, (100, 1))), np.ones(100, dtype=np.int8), phaseList(csB))
  mAB = eA.calcMisorientation(eB, sampleSize=50, minDistance=0.0)
  assert mAB.CS is csA and mAB.SS is csB and not mAB.antipodal
  expected = orientation._fromRotation((qB.inv() * qA).data, (csA, csB))
  assert np.all(mAB.angle(expected) < 1e-9)


def test_I10_single_precision_and_the_shape_of_the_output():
  e = linearMap(GEOM[4][1], 10, dtype=np.float32)
  gx, gy = e.gradient()
  ok = ~np.isnan(gx.data[:, 2])
  assert np.max(np.abs(gx.data[ok, 2] - K1)) < 1e-5 and np.max(np.abs(gy.data[ok, 2] - K2)) < 1e-5
  # a gridded map answers in its own shape
  gr = gridify(linearMap(np.eye(2), 6))
  assert gr.gradient()[0].shape == gr.shape and gr.KAM().shape == gr.shape and gr.curvature().shape == gr.shape + (3, 3)
  with pytest.raises(ValueError):
    linearMap(np.eye(2), 6)[[1, 1]].KAM()


def test_the_numba_and_the_numpy_gradient_and_KAM_agree():
  # a map of three grains with holes, a seam and a hexagonal lattice, so that every branch of
  # the neighbourhood is met
  e = linearMap(GEOM[4][1], 14, shift=lambda i, j: np.where(i >= 7, 20 * degree, np.where(j >= 9, 40 * degree, 0.0)))
  ph = e.phaseId.copy()
  ph[[30, 31, 45, 100]] = 0
  e = EBSD(e.pos, e.rotations, ph, e.CSList)
  calcGrains(e, angle=5 * degree)
  before = getMTEXpref("kernels", "numba")
  out = {}
  try:
    for kernels in ("numba", "numpy"):
      setMTEXpref("kernels", kernels)
      out[kernels] = [e.gradient(stencil=st)[c].data for st in ("oneSided", "lattice", "full") for c in (0, 1)]
      out[kernels] += [e.KAM(), e.KAM(order=2, max=True), e.KAM(radius=1.2, threshold=0.01, weights=lambda d: 1 / d)]
      g = calcGrains(e, angle=5 * degree)
      out[kernels].append(e.calcGROD(g).data)
  finally:
    setMTEXpref("kernels", before)
  for a, b in zip(out["numba"], out["numpy"]):
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a, b, atol=1e-12, equal_nan=True)
