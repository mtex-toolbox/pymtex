# axis fields on the sphere (docs/plans/functions.md F7: V9)
import numpy as np
import pytest

from mtex import (Miller, S2AxisFieldHarmonic, S2FunHarmonic, angle, crystalFrame, dot, grad, quadratureS2Grid, specimenFrame,
                  vector3d, zvector)

rng = np.random.default_rng(79)


def test_V9_axis_fields_on_the_sphere():
  # a smooth axis field, recovered from the outer products
  a = lambda v: vector3d(np.ones(v.shape), v.z, v.x ** 2, antipodal=True).normalize()
  A = S2AxisFieldHarmonic(a, bandwidth=32)
  vs = vector3d.rand(10, rng=rng)
  assert isinstance(A, S2AxisFieldHarmonic) and A.bandwidth == 32
  assert A(vs).antipodal
  assert np.max(angle(A(vs), a(vs))) < 1e-3 and angle(A(vs[0]), a(vs[0])) < 1e-3
  # from samples, and from a vector field
  g = quadratureS2Grid(32)
  B = S2AxisFieldHarmonic(g.nodes(), a(g.nodes()), bandwidth=32, weights=g.weights)
  assert np.max(angle(B(vs), A(vs))) < 1e-8
  G = grad(S2FunHarmonic(rng.standard_normal(16) + 1j * rng.standard_normal(16), real=True))
  C = S2AxisFieldHarmonic(G, bandwidth=24)
  assert np.max(angle(C(vs), vector3d(G(vs).data, antipodal=True))) < 1e-2
  # the cosine to a constant axis and to another field
  d = dot(A, zvector)
  assert np.allclose(d(vs), np.abs(a(vs).dot(zvector)), atol=2e-3)
  assert np.allclose(A.angle(zvector)(vs), angle(a(vs), vector3d(zvector, antipodal=True)), atol=2e-3)
  assert np.allclose(dot(A, B)(vs), 1, atol=1e-3)
  assert np.allclose(dot(A, G)(vs), np.abs(np.sum(a(vs).data * G(vs).data, axis=-1)), atol=3e-2)
  # a framed axis field is equivariant under the group, including improper elements
  cs = crystalFrame("4/mmm")
  E = S2AxisFieldHarmonic(rng.standard_normal((36, 6)) + 1j * rng.standard_normal((36, 6)), cs)
  els = cs.sym.rot.flatten()
  for i in range(els.size):
    M = els[i].matrix()
    moved = E(vector3d(vs.data @ M.T))
    ref = vector3d(E(vs).data @ M.T, antipodal=True)
    assert np.max(angle(moved, ref)) < 1e-6
  assert E(Miller(1, 0, 0, cs)).antipodal
  with pytest.raises(ValueError):
    E(vector3d(vector3d(1, 0, 0), frame=specimenFrame("1")))
