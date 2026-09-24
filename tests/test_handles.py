# handles and sums (docs/plans/functions.md F5: H5a to H5c, check_S1Fun, the handle and
# composition cases of check_radonOptions)
import numpy as np
import pytest
from numpy import pi

from mtex import (Miller, S1FunHandle, S1FunHarmonic, S2DeLaValleePoussinKernel, S2FunHandle, S2FunHarmonic, S2FunSum,
                  SO3FunComposition, SO3FunHandle, SO3FunHarmonic, SO3FunRBF, angle, crystalFrame, degree, fibre, fibreODF,
                  mean, orientation, pdf, rotate, rotation, specimenFrame, unimodalODF, vector3d, zvector)

rng = np.random.default_rng(51)


def test_H5a_SO3FunHandle():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  # check_radonOptions's handle: 1 + cos(angle), averaged over the groups when asked
  f = SO3FunHandle(lambda o: 1 + np.cos(angle(o, noSymmetry=True)), cs, ss, bandwidth=16)
  o = orientation(rotation.rand(7, rng=rng), cs, ss)
  assert np.allclose(f(o), 1 + np.cos(angle(o, noSymmetry=True))) and f(o[0]) == f(o)[0] and f(o[0]) == f(rotation(o[0]))
  # another frame is refused; the same frames without their groups are not another frame
  with pytest.raises(ValueError):
    f(orientation(rotation.rand(2, rng=rng), cs, specimenFrame.rolling))
  g = SO3FunHandle(lambda o: 1 + np.cos(angle(o, noSymmetry=True)), cs, ss, bandwidth=16, symmetrise=True)
  E, S = cs.sym.rot.flatten().rot, ss.sym.rot.flatten().rot
  for i in range(E.size):
    for j in range(S.size):
      assert np.isclose(g(S[j] * rotation(o[0]) * E[i]), g(o[0]), rtol=1e-10)
  # the harmonic form and the operations through it
  h = SO3FunHarmonic(g)
  assert h.bandwidth == 16 and h.isReal
  rs = rotation.rand(20, rng=rng)
  assert np.max(np.abs(h(rs) - g(rs))) < 1e-6 * np.max(np.abs(g(rs)))
  assert isinstance(pdf(g, Miller(1, 0, 0, cs)), S2FunHarmonic)
  assert np.isclose(mean(g), mean(h))
  assert np.allclose((2 * g)(rs), 2 * g(rs))
  assert "SO3FunHandle" in repr(g)
  # exchange symmetry averages with the inverses
  fx = SO3FunHandle(lambda o: o.a + o.b, cs, cs, antipodal=True, symmetrise=True)
  assert np.isclose(fx(rs[0]), fx(rs[0].inv()))


def test_H5b_S2FunHandle_and_S1FunHandle():
  cs = crystalFrame("6/mmm")
  f = S2FunHandle(lambda v: v.z ** 2 + 0.3 * v.x, cs, bandwidth=16)
  vs = vector3d(vector3d.rand(6, rng=rng), frame=cs)
  assert np.allclose(f(vs), vs.z ** 2 + 0.3 * vs.x) and f(vs[0]) == f(vs)[0]
  with pytest.raises(ValueError):
    f(Miller(1, 0, 0, crystalFrame("m-3m")))
  g = S2FunHandle(lambda v: v.z ** 2 + 0.3 * v.x, cs, bandwidth=16, symmetrise=True)
  v0 = vector3d(vs[0].data)
  E = cs.sym.rot.flatten()
  for i in range(E.size):
    assert np.isclose(g(E[i] * v0), g(v0))
  a = S2FunHandle(lambda v: v.z ** 2 + 0.3 * v.x, bandwidth=16, antipodal=True)
  assert a.antipodal and np.isclose(a(v0), a(-v0)) and np.isclose(a(v0), v0.z ** 2)
  h = S2FunHarmonic(g, bandwidth=8)
  ds = vector3d.rand(20, rng=rng)
  assert np.max(np.abs(h(ds) - g(ds))) < 1e-7
  assert S2FunHarmonic(a, bandwidth=8).antipodal
  assert np.isclose(mean(g), mean(h)) and np.allclose((2 * g)(ds), 2 * g(ds))
  s = S1FunHandle(lambda x: np.cos(2 * x) + 0.5 * np.sin(x) + 0.3, bandwidth=8)
  xs = np.linspace(0, 2 * pi, 13)
  assert np.allclose(s(xs), np.cos(2 * xs) + 0.5 * np.sin(xs) + 0.3)
  S = S1FunHarmonic(s, bandwidth=4)
  # check_S1Fun
  assert np.max(np.abs((S + 3)(xs) - (s(xs) + 3))) < 1e-10
  assert np.max(np.abs((3 + S)(xs) - (s(xs) + 3))) < 1e-10
  assert np.max(np.abs((S - 1.5)(xs) - (s(xs) - 1.5))) < 1e-10
  assert np.isclose(mean(S + 3), mean(S) + 3) and S.isReal
  assert np.allclose((S1FunHarmonic([2.0]) + 5)(xs), 7.0)


def test_H5c_sums():
  cs, ss = crystalFrame("432"), specimenFrame("1")
  u = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss))
  fb = fibreODF(fibre(orientation(rotation.rand(rng=rng), cs, ss), Miller(1, 1, 1, cs)))
  h = SO3FunHandle(lambda o: 1 + np.cos(angle(o, noSymmetry=True)), cs, ss, bandwidth=16, symmetrise=True)
  s = 0.5 * u + 0.5 * fb
  assert isinstance(s, SO3FunComposition) and len(s.components) == 2 and np.isclose(mean(s), 1)
  rs = rotation.rand(10, rng=rng)
  assert np.allclose(s(rs), 0.5 * u(rs) + 0.5 * fb(rs))
  t = s + h - u
  assert isinstance(t, SO3FunComposition) and len(t.components) == 4 and np.allclose(t(rs), s(rs) + h(rs) - u(rs))
  # the absolute value of a difference is a handle, MATLAB's mean(abs(f - g))
  a = abs(h - u)
  assert isinstance(a, SO3FunHandle) and np.allclose(a(rs), np.abs(h(rs) - u(rs)))
  assert np.allclose((s + 2)(rs), s(rs) + 2) and np.isclose(mean(s + 2), 3)
  H = SO3FunHarmonic(s)
  assert H.isReal and H.CS is cs
  assert np.max(np.abs(H(rs) - s(rs))) < 3e-2 * np.max(s(rs))
  # check_radonOptions on the composition: the pole figure exists and is axis valued for a Laue crystal
  assert isinstance(pdf(s, Miller(1, 0, 0, cs)), S2FunHarmonic)
  csL = crystalFrame("m-3m")
  sL = 0.5 * unimodalODF(orientation(rotation.rand(rng=rng), csL, ss)) + 0.5 * fibreODF(fibre(orientation(rotation.rand(rng=rng), csL, ss), Miller(1, 1, 1, csL)))
  assert pdf(sL, Miller(1, 0, 0, csL)).antipodal
  assert pdf(s, vector3d(1, 0, 0, antipodal=True)).antipodal
  # two RBFs with the same kernel stay one RBF, with different kernels a sum
  u2 = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss))
  assert isinstance(u + u2, SO3FunRBF) and np.isclose(mean(u + u2), 2)
  assert isinstance(u + unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=5 * degree), SO3FunComposition)
  with pytest.raises(ValueError):
    u + unimodalODF(orientation(rotation.rand(rng=rng), csL, ss))
  # a harmonic function plus a kernel form is a composition too
  assert isinstance(SO3FunHarmonic(u) + fb, SO3FunComposition) and isinstance(fb + SO3FunHarmonic(u), SO3FunComposition)
  # a turn of the specimen carries every kind of term with it, so the rotated function takes
  # at q g the value the original takes at g; a triclinic pair, so no group folds the two
  # sides differently
  cs1, ss1 = crystalFrame("1", [1, 1, 1]), specimenFrame("1")
  q = rotation.byAxisAngle(zvector, 30 * degree)
  gs = orientation(rotation.rand(20, rng=rng), cs1, ss1)
  ub = unimodalODF(orientation(rotation.rand(rng=rng), cs1, ss1))
  fbb = fibreODF(fibre(orientation(rotation.rand(rng=rng), cs1, ss1), Miller(1, 1, 1, cs1)), halfwidth=10 * degree)
  for f in (ub, fbb, 0.4 * ub + 0.6 * fbb):
    rf = rotate(f, q)
    assert np.allclose(rf(orientation(q * rotation(gs), rf.CS, rf.SS)), f(gs))
    rr = rotate(f, q, "right")
    assert np.allclose(rr(orientation(rotation(gs) * q, rr.CS, rr.SS)), f(gs))
  # on the sphere
  a = S2FunHandle(lambda v: v.z ** 2, bandwidth=16)
  b = S2FunHarmonic(S2DeLaValleePoussinKernel(20))
  ab = a + b
  ds = vector3d.rand(10, rng=rng)
  assert isinstance(ab, S2FunSum) and np.allclose(ab(ds), a(ds) + b(ds)) and np.isclose(mean(ab), mean(a) + mean(b))
  assert np.max(np.abs(S2FunHarmonic(ab)(ds) - ab(ds))) < 1e-6
  assert isinstance(b + a, S2FunSum) and np.allclose((a / b)(ds), a(ds) / b(ds))
