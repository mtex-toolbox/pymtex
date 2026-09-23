# vector fields on the sphere (docs/plans/functions.md F7: V7, V8)
import numpy as np
import pytest

from mtex import (Miller, S2FunHarmonic, S2VectorFieldHandle, S2VectorFieldHarmonic, crystalFrame, cross, degree, dot, grad,
                  interpolate, laplace, mean, norm, orientation, quadratureS2Grid, rhoVector, rotate, rotation, specimenFrame,
                  symmetrise, thetaVector, vector3d, xvector, yvector, zvector)

rng = np.random.default_rng(77)


def randomS2(n, frame=None):
  return S2FunHarmonic(rng.standard_normal(n) + 1j * rng.standard_normal(n), frame, real=True)


def test_V7_the_gradient_on_the_sphere():
  s = randomS2(49)
  d = 1e-5
  for v in vector3d.rand(3, rng=rng):
    g = grad(s, v)
    assert abs(g.dot(v)) < 1e-10
    t = v.cross(xvector).normalize()
    fd = (s((v + d * t).normalize()) - s((v - d * t).normalize())) / (2 * d)
    assert abs(g.dot(t) - fd) < 1e-6 * (1 + abs(fd))
  G = grad(s)
  assert isinstance(G, S2VectorFieldHarmonic) and G.bandwidth == s.bandwidth + 1
  vs = vector3d.rand(8, rng=rng)
  scale = np.max(np.linalg.norm(grad(s, vs).data, axis=-1))
  assert np.max(np.abs(G(vs).data - grad(s, vs).data)) < 1e-5 * scale
  assert np.max(np.abs(G(vs).data - np.stack([G(vs[i]).data for i in range(8)]))) < 1e-12
  # the mean of a surface gradient is twice the first moment of the function, and Green's
  # identity int |grad s|^2 = -int s Laplace s holds
  moment = np.array([mean(S2FunHarmonic(lambda v: s(v) * v.data[..., i], bandwidth=s.bandwidth + 1)) for i in range(3)])
  assert np.allclose(G.mean().data, 2 * moment, atol=1e-5)
  assert np.isclose(mean(dot(G, G)), -dot(s, laplace(s)), rtol=1e-5)
  # a symmetric function has an equivariant gradient, including the improper elements
  cs = crystalFrame("-3m")
  sc = randomS2(49, cs)
  Gc = grad(sc)
  assert Gc.frame is cs
  E = cs.sym.rot.flatten()
  for i in range(E.size):
    M = E[i].matrix()
    for j in range(2):
      v = vs[j].data
      assert np.linalg.norm(Gc(vector3d(M @ v)).data - M @ Gc(vector3d(v)).data) < 1e-5 * scale
  assert np.linalg.norm(symmetrise(Gc).fhat - Gc.fhat) < 1e-5 * np.linalg.norm(Gc.fhat)
  # rotate commutes with grad
  q = rotation.byAxisAngle(vector3d(1, 2, 3).normalize(), 37 * degree)
  Gr = rotate(G, q)
  ref = grad(rotate(s, q))
  assert np.max(np.abs(Gr(vs).data - ref(vs).data)) < 1e-5 * scale
  assert np.max(np.abs(Gr(vs).data - (q * vector3d(G(q.inv() * vs).data)).data)) < 1e-12
  with pytest.raises(ValueError):
    rotate(Gc, orientation(q, specimenFrame("1"), specimenFrame("1")))
  Go = rotate(Gc, orientation(q, cs, specimenFrame("222")))
  assert Go.frame is specimenFrame("222")


def test_V8_algebra_quadrature_handles():
  s = randomS2(36)
  G = grad(s)
  H = S2VectorFieldHarmonic(rng.standard_normal((25, 3)) + 1j * rng.standard_normal((25, 3)))
  vs = vector3d.rand(6, rng=rng)
  # the field from a callable and from samples on the rule
  L = G.bandwidth
  assert S2VectorFieldHarmonic(lambda v: G(v), bandwidth=L).isclose(G, 1e-5)
  g = quadratureS2Grid(L)
  assert S2VectorFieldHarmonic(g.nodes(), G(g.nodes()), bandwidth=L, weights=g.weights).isclose(G, 1e-5)
  assert interpolate(g.nodes(), G(g.nodes()), bandwidth=L, weights=g.weights).isclose(G, 1e-4)
  assert S2VectorFieldHarmonic([G.component(i) for i in range(3)]).isclose(G, 1e-12)
  # the pointwise algebra
  d = dot(G, H)
  assert np.allclose(d(vs), np.sum(G(vs).data * H(vs).data, axis=-1), rtol=1e-5)
  assert np.allclose(cross(G, H)(vs).data, np.cross(G(vs).data, H(vs).data), rtol=1e-5)
  assert np.allclose((s * H)(vs).data, s(vs)[:, None] * H(vs).data, rtol=1e-5)
  assert np.allclose(abs(H)(vs), np.linalg.norm(H(vs).data, axis=-1), rtol=1e-1)
  assert np.allclose(np.linalg.norm(H.normalize()(vs).data, axis=-1), 1, rtol=1e-1)
  assert np.allclose(dot(H, yvector)(vs), H(vs).y)
  assert np.allclose(cross(H, yvector)(vs).data, np.cross(H(vs).data, yvector.data))
  assert np.allclose(H.mean().data, np.real(H.fhat[0]) / np.sqrt(4 * np.pi))
  assert np.allclose((H + xvector)(vs[0]).data, H(vs[0]).data + xvector.data)
  assert np.allclose(H.antipode()(vs[0]).data, H(-vs[0]).data)
  # a handle, wrapped and converted
  hd = S2VectorFieldHandle(lambda v: v.cross(zvector), bandwidth=4)
  assert np.allclose(hd(vs).data, np.cross(vs.data, zvector.data))
  hh = S2VectorFieldHarmonic(hd)
  assert hh.bandwidth == 4 and np.max(np.abs(hh(vs).data - hd(vs).data)) < 1e-5
  q = rotation.byAxisAngle(xvector, 20 * degree)
  assert np.max(np.abs(rotate(hd, q)(vs).data - rotate(hh, q)(vs).data)) < 1e-5
  assert np.max(np.abs(S2VectorFieldHandle(H)(vs).data - H(vs).data)) < 1e-12
  # the polar basis is orthonormal and tangential
  for i in range(6):
    v = vs[i]
    et, er = thetaVector(v), rhoVector(v)
    assert abs(et.dot(v)) < 1e-12 and abs(er.dot(v)) < 1e-12 and abs(et.dot(er)) < 1e-12
    assert np.isclose(norm(et.norm()), 1) and np.isclose(norm(er.norm()), 1)
    assert np.allclose(et.cross(er).data, v.data)
  # a framed field evaluates framed directions of its frame only
  cs = crystalFrame("m-3m")
  Hc = S2VectorFieldHarmonic(rng.standard_normal((25, 3)) + 1j * rng.standard_normal((25, 3)), cs)
  assert isinstance(Hc(Miller(1, 1, 0, cs)), vector3d)
  with pytest.raises(ValueError):
    Hc(vector3d(xvector, frame=specimenFrame("1")))
  M = cs.sym.rot.flatten()[2].matrix()
  assert np.linalg.norm(Hc(vector3d(M @ xvector.data)).data - M @ Hc(xvector).data) < 1e-10


def test_the_normal_field():
  # MATLAB's S2VectorFieldHarmonic.normal: v at v, exact at bandwidth two
  f = S2VectorFieldHarmonic.normal()
  v = vector3d.rand(20)
  assert f.bandwidth == 2 and np.allclose(f.eval(v).data, v.normalize().data, atol=1e-10)
