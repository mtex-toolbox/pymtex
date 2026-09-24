# the radial basis and fibre representations (docs/plans/functions.md F4: R1, R3, R4, R5 -
# check_odfGrad, check_radonOptions, check_calcPoleFigureSuperposition)
import numpy as np
import pytest

from mtex import (FourierODF, Miller, S2DeLaValleePoussinKernel, SO3DeLaValleePoussinKernel, SO3FunCBF, SO3FunHarmonic, SO3FunRBF,
                  calcPDF, crystalFrame, degree, fibre, fibreODF, grad, ipdf, mean, orientation, pdf, radon, rotation,
                  specimenFrame, uniformODF, unimodalODF, vector3d, xvector, yvector, zvector)
from mtex.geometry.symreduce import reductionGroups

rng = np.random.default_rng(41)


def test_R1_uniform_and_unimodal_ODFs():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  u = uniformODF(cs, ss)
  assert np.isclose(mean(u), 1) and u(rotation.rand(rng=rng)) == 1 and u.CS is cs
  assert SO3FunHarmonic(u).isclose(SO3FunHarmonic([1.0], cs, ss))
  o = orientation(rotation.rand(5, rng=rng), cs, ss)
  f = unimodalODF(o, halfwidth=15 * degree)
  assert np.isclose(mean(f), 1) and f.center.size == 5
  psi = SO3DeLaValleePoussinKernel(halfwidth=15 * degree)
  # the value is the mean of the kernel over the equivalents of the centres
  L, R = reductionGroups(o)
  L, R = rotation(L), rotation(R)
  for r in rotation.rand(4, rng=rng):
    direct = np.sum([psi(L[j] * o[k] * R[i], r) for k in range(5) for i in range(R.size) for j in range(L.size)]) / (5 * L.size * R.size * psi.mean())
    assert np.isclose(f(r), direct, rtol=1e-4)
    assert f(orientation(r, cs, ss)) == f(r)
  # the frames stripped of their groups are the same frames, another frame is not
  assert np.isclose(f(orientation(rotation.id, cs.stripSym(), ss.stripSym())), f(rotation.id))
  with pytest.raises(ValueError):
    f(orientation(rotation.id, cs, specimenFrame.rolling))
  # the harmonic form agrees away from the truncation, is symmetric and has mean one
  h = SO3FunHarmonic(f)
  assert np.isclose(mean(h), 1) and h.CS is cs and h.isReal
  rs = rotation.rand(200, rng=rng)
  assert np.max(np.abs(h(rs) - f(rs))) < 2e-2 * np.max(f(rs))
  assert f(rs.reshape(20, 10)).shape == (20, 10)
  # weights scale the mean
  g = unimodalODF(o, halfwidth=15 * degree, weights=[1, 2, 3, 4, 5])
  assert np.isclose(mean(g), 1) and np.isclose(g.weights[4], 5 * g.weights[0])
  assert "SO3FunRBF" in repr(f)
  # exchange symmetry averages with the inverses
  fx = unimodalODF(orientation(rotation.rand(3, rng=rng), cs, cs, antipodal=True), halfwidth=15 * degree)
  for r in rotation.rand(3, rng=rng):
    assert np.isclose(fx(r), fx(r.inv()), rtol=1e-8)
  assert SO3FunHarmonic(fx).antipodal
  # a kernel positional, the identity as the default centre, the algebra
  assert isinstance(unimodalODF(o, psi).psi, SO3DeLaValleePoussinKernel)
  assert unimodalODF(cs, ss).center.size == 1
  assert np.isclose((2 * f + 1)(rs[0]), 2 * f(rs[0]) + 1)
  assert np.isclose(f.inv()(rs[0].inv()), f(rs[0])) and f.inv().CS is ss


def test_R3_gradients_of_the_kernel_forms():
  d = 0.005 * degree

  def fd(f, o, v, side):
    r = rotation(o)
    a = rotation.byAxisAngle(v, d) * r if side == "left" else r * rotation.byAxisAngle(v, d)
    b = rotation.byAxisAngle(v, -d) * r if side == "left" else r * rotation.byAxisAngle(v, -d)
    return (f(a) - f(b)) / (2 * d)

  # check_odfGrad: the unimodal ODF, both tangent spaces
  cs, ss = crystalFrame("-3m1"), crystalFrame("m-3m")
  odf = unimodalODF(orientation(rotation.rand(20, rng=rng), cs, ss))
  os = orientation(rotation.rand(30, rng=rng), cs, ss)
  for side in ("left", "right"):
    ts = grad(odf, os, side)
    assert ts.shape == (30,) and ts.side == side
    worst = 0.0
    for i in range(30):
      g = np.array([fd(odf, os[i], v, side) for v in (xvector, yvector, zvector)])
      worst = max(worst, np.linalg.norm(ts[i].data - g) / (1 + np.linalg.norm(g)))
    assert worst < 1e-3
    assert np.max(np.linalg.norm(ts.data, axis=-1)) > 1e-3
  # the fibre ODF
  cs2, ss2 = crystalFrame("432"), specimenFrame("222")
  fodf = fibreODF(Miller(vector3d.rand(rng=rng), cs2), vector3d.rand(rng=rng), ss2)
  os2 = orientation(rotation.rand(30, rng=rng), cs2, ss2)
  for side in ("left", "right"):
    worst = 0.0
    for i in range(30):
      t = grad(fodf, os2[i], side)
      g = np.array([fd(fodf, os2[i], v, side) for v in (xvector, yvector, zvector)])
      worst = max(worst, np.linalg.norm(t.data - g) / (1 + np.linalg.norm(g)))
    assert worst < 1e-3


def test_R4_fibre_ODFs():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  h, r = Miller(1, 1, 0, cs), vector3d.rand(rng=rng)
  f = fibreODF(h, r, ss, halfwidth=12 * degree)
  assert np.isclose(mean(f), 1) and f.CS is cs and f.SS is ss
  psi = S2DeLaValleePoussinKernel(halfwidth=12 * degree)
  L, R = reductionGroups(orientation(rotation.id, cs, ss))
  L, R = rotation(L), rotation(R)
  hn = vector3d(h.data).normalize()
  for g in rotation.rand(4, rng=rng):
    direct = np.sum([psi((L[j] * g * R[i] * hn).dot(r)) for i in range(R.size) for j in range(L.size)]) / (L.size * R.size * psi.mean())
    assert np.isclose(f(g), direct, rtol=1e-8)
    for i in range(0, R.size, 5):
      for j in range(0, L.size, 3):
        assert np.isclose(f(L[j] * g * R[i]), f(g), rtol=1e-8)
  # the harmonic form
  H = SO3FunHarmonic(f)
  assert np.isclose(mean(H), 1) and H.isReal
  # MATLAB's FourierODF is the same
  assert isinstance(FourierODF(f), SO3FunHarmonic) and np.allclose(FourierODF(f).fhat, H.fhat)
  rs = rotation.rand(100, rng=rng)
  assert np.max(np.abs(H(rs) - f(rs))) < 2e-2 * np.max(f(rs))
  # from a fibre of P1, and several fibres
  fb = fibre(orientation(rotation.rand(rng=rng), cs, ss), h)
  f2 = fibreODF(fb, halfwidth=12 * degree)
  assert np.isclose(mean(f2), 1) and f2.SS is ss
  f3 = fibreODF(Miller(vector3d.cat(h, Miller(1, 0, 0, cs)), cs), vector3d.cat(r, vector3d.rand(rng=rng)), ss, weights=[1, 3])
  assert np.isclose(mean(f3), 1) and np.isclose(f3.weights[1], 3 * f3.weights[0])
  assert "SO3FunCBF" in repr(f)
  assert isinstance(SO3FunCBF(fb, psi), SO3FunCBF)


def test_R5_pole_figures_of_the_kernel_forms():
  cs, ss = crystalFrame("432"), specimenFrame("1")
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=15 * degree)
  h = Miller(1, 0, 0, cs)
  P = pdf(odf, h)
  Ph = pdf(SO3FunHarmonic(odf), h)
  vs = vector3d.rand(20, rng=rng)
  assert np.max(np.abs(P(vs) - Ph(vs))) < 2e-2 * np.max(P(vs))
  assert P.antipodal and P.frame is ss   # a two-fold axis of 432 sends (100) to (-100)
  assert pdf(unimodalODF(orientation(rotation.rand(rng=rng), crystalFrame("m-3m"), ss)), Miller(1, 0, 0, crystalFrame("m-3m"))).antipodal
  assert odf.pdf(h, bandwidth=4).bandwidth <= 4
  # the pole figure is the S2 kernel at the images of h, and the uniform ODF gives one
  assert np.allclose(pdf(uniformODF(cs, ss), h)(vs), 1)
  # ipdf, radon and the mean of a pole figure
  I = ipdf(odf, vs[0])
  assert I.frame is cs and np.isclose(mean(P), 1) and np.isclose(mean(I), 1)
  assert np.isclose(radon(odf, h, vs[0]), P(vs[0]))
  # check_calcPoleFigureSuperposition
  cs3 = crystalFrame("321")
  odf3 = unimodalODF(orientation(rotation.rand(rng=rng), cs3), halfwidth=15 * degree)
  hs = Miller(vector3d.cat(Miller(1, 0, 0, cs3), Miller(0, 0, 1, cs3)), cs3)
  ref = 0.5 * pdf(odf3, hs[0])(vs) + 0.5 * pdf(odf3, hs[1])(vs)
  assert np.max(np.abs(calcPDF(odf3, hs, vs) - ref)) < 1e-10
  ref2 = 0.52 * pdf(odf3, hs[0])(vs) + 1.23 * pdf(odf3, hs[1])(vs)
  assert np.max(np.abs(calcPDF(odf3, hs, vs, superposition=[0.52, 1.23]) - ref2)) < 1e-10
  assert np.allclose(calcPDF(odf3, hs[0], vs), pdf(odf3, hs[0])(vs))
  assert np.max(np.abs(calcPDF(odf3, hs, superposition=[0.52, 1.23])(vs) - ref2)) < 1e-10
  assert np.max(np.abs(calcPDF(SO3FunHarmonic(odf3), hs, vs, superposition=[0.52, 1.23]) - ref2)) < 2e-2 * np.max(ref2)
  with pytest.raises(ValueError):
    calcPDF(odf3, hs, vs, superposition=[1.0])
  with pytest.raises(ValueError):
    pdf(odf3, hs)
  # the fibre ODF goes through its harmonic form
  fodf = fibreODF(Miller(1, 1, 1, cs), zvector, ss)
  assert np.allclose(pdf(fodf, h)(vs), pdf(SO3FunHarmonic(fodf), h)(vs))
