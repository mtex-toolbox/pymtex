# S2FunHarmonic (docs/plans/functions.md F3d)
import numpy as np
import pytest

from mtex import (Miller, S2DeLaValleePoussinKernel, S2FunHarmonic, crystalFrame, degree, dot, mean, norm, quadratureS2Grid,
                  vector3d, zvector, symmetrise)

rng = np.random.default_rng(34)


def randomCoefficients(n):
  return rng.standard_normal(n) + 1j * rng.standard_normal(n)


def test_F3d_S2FunHarmonic():
  cs = crystalFrame("6/mmm")
  psi = S2DeLaValleePoussinKernel(halfwidth=10 * degree)
  f = S2FunHarmonic(psi)
  assert f.frame is None and f.isReal and not f.antipodal and np.isclose(mean(f), 1)
  for v in vector3d.rand(5, rng=rng):
    assert abs(f(v) - psi(v.z)) < 1e-2 * psi(1.0)
  vs = vector3d.rand(3, 4, rng=rng)
  assert f(vs).shape == (3, 4) and np.allclose(f(vs), [[f(vs[i, j]) for j in range(4)] for i in range(3)])
  assert np.isclose(f(vector3d(0, 0, 2)), f(zvector))
  # in a frame: framed arguments must match, symmetrisation makes it invariant
  g = S2FunHarmonic(randomCoefficients(49), cs)
  with pytest.raises(ValueError):
    g(Miller(0, 0, 1, crystalFrame("m-3m")))
  m = Miller(1, 2, 3, cs)
  assert np.isclose(g(m), g(vector3d(m.data)))
  E = cs.sym.rot.flatten()
  for i in range(E.size):
    assert np.isclose(g(E[i] * vector3d(m.data)), g(vector3d(m.data)))
  assert symmetrise(g).isclose(g)
  # a subgroup the frame does not name
  rots = crystalFrame("6").sym.rotations().flatten()
  gs = S2FunHarmonic(randomCoefficients(49)).symmetrise(rots)
  v = vector3d.rand(rng=rng)
  assert np.isclose(gs(rots[1] * v), gs(v))
  # axes: even degrees only, invariant under the antipode
  a = S2FunHarmonic(randomCoefficients(36), antipodal=True)
  assert a.antipodal and np.all(a.fhat[1:4] == 0) and np.all(a.fhat[9:16] == 0)
  for v in vector3d.rand(3, rng=rng):
    assert np.isclose(a(-v), a(v))
  d = S2FunHarmonic(randomCoefficients(36))
  assert np.isclose(d.even()(v), (d(v) + d(-v)) / 2)
  # from a callable and from samples
  h = S2FunHarmonic(lambda v: g(v), cs, bandwidth=6)
  assert h.isclose(g, 1e-6)
  q = quadratureS2Grid(6)
  assert S2FunHarmonic.quadrature(vector3d(q.nodes(), frame=cs), g(q.nodes()), bandwidth=6, weights=q.weights).isclose(g, 1e-6)
  assert S2FunHarmonic.quadrature(q, g(q.nodes()), cs, bandwidth=6).isclose(g, 1e-6)
  # the algebra and the moments by quadrature
  f2 = S2FunHarmonic(randomCoefficients(25))
  ns, w = q.nodes(), q.weights
  assert np.allclose((f + f2)(ns), f(ns) + f2(ns))
  assert np.allclose((2 * f2 - 1)(ns), 2 * f2(ns) - 1)
  assert abs(mean(f2) - np.sum(w * f2(ns)) / (4 * np.pi)) < 1e-6 * abs(mean(f2))
  assert abs(norm(f2) ** 2 - np.sum(w * np.abs(f2(ns)) ** 2) / (4 * np.pi)) < 1e-6 * norm(f2) ** 2
  qd = quadratureS2Grid(f.bandwidth + f2.bandwidth)
  assert abs(dot(f, f2) - np.sum(qd.weights * np.conj(f(qd.nodes())) * f2(qd.nodes())) / (4 * np.pi)) < 1e-6 * abs(dot(f, f2))
  assert np.allclose(f2.conj()(ns), np.conj(f2(ns)))
  assert "S2FunHarmonic" in repr(g)
