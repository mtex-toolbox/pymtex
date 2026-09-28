# axis fields on the sphere (docs/plans/functions.md F7: V9)
import numpy as np
import pytest

from mtex import (Miller, S2AxisFieldHarmonic, S2FunHarmonic, angle, crystalFrame, dot, grad, quadratureS2Grid, specimenFrame,
                  vector3d, zvector)

rng = np.random.default_rng(79)

# a smooth axis field, its harmonic expansion and the directions to evaluate at
a = lambda v: vector3d(np.ones(v.shape), v.z, v.x ** 2, antipodal=True).normalize()
A = S2AxisFieldHarmonic(a, bandwidth=32)
vs = vector3d.rand(10, rng=rng)


# the same field from samples on a quadrature grid
def sampledField():
  g = quadratureS2Grid(32)
  return S2AxisFieldHarmonic(g.nodes(), a(g.nodes()), bandwidth=32, weights=g.weights)


# a gradient field of a random real function
def gradientField():
  return grad(S2FunHarmonic(rng.standard_normal(16) + 1j * rng.standard_normal(16), real=True))


# a smooth axis field, recovered from the outer products
def test_S2AxisFieldHarmonic_fromACallableRecoversTheAxes():
  assert isinstance(A, S2AxisFieldHarmonic) and A.bandwidth == 32
  assert A(vs).antipodal
  assert np.max(angle(A(vs), a(vs))) < 1e-3 and angle(A(vs[0]), a(vs[0])) < 1e-3


def test_S2AxisFieldHarmonic_fromSamplesIsTheExpansion():
  assert np.max(angle(sampledField()(vs), A(vs))) < 1e-8


def test_S2AxisFieldHarmonic_fromAVectorFieldTakesItsAxes():
  G = gradientField()
  C = S2AxisFieldHarmonic(G, bandwidth=24)
  assert np.max(angle(C(vs), vector3d(G(vs).data, antipodal=True))) < 1e-2


# the cosine to a constant axis
def test_dot_withAConstantAxisIsTheAbsoluteCosine():
  d = dot(A, zvector)
  assert np.allclose(d(vs), np.abs(a(vs).dot(zvector)), atol=2e-3)
  assert np.allclose(A.angle(zvector)(vs), angle(a(vs), vector3d(zvector, antipodal=True)), atol=2e-3)


# the cosine to another field
def test_dot_withAnotherFieldIsTheAbsoluteCosine():
  G = gradientField()
  assert np.allclose(dot(A, sampledField())(vs), 1, atol=1e-3)
  assert np.allclose(dot(A, G)(vs), np.abs(np.sum(a(vs).data * G(vs).data, axis=-1)), atol=3e-2)


# a framed axis field is equivariant under the group, including improper elements
def test_S2AxisFieldHarmonic_framedIsEquivariantUnderTheGroup():
  cs = crystalFrame("4/mmm")
  E = S2AxisFieldHarmonic(rng.standard_normal((36, 6)) + 1j * rng.standard_normal((36, 6)), cs)
  els = cs.sym.rot.flatten()
  for i in range(els.size):
    M = els[i].matrix()
    moved = E(vector3d(vs.data @ M.T))
    ref = vector3d(E(vs).data @ M.T, antipodal=True)
    assert np.max(angle(moved, ref)) < 1e-6
  assert E(Miller(1, 0, 0, cs)).antipodal


def test_S2AxisFieldHarmonic_framedRefusesDirectionsOfAnotherFrame():
  cs = crystalFrame("4/mmm")
  E = S2AxisFieldHarmonic(rng.standard_normal((36, 6)) + 1j * rng.standard_normal((36, 6)), cs)
  with pytest.raises(ValueError):
    E(vector3d(vector3d(1, 0, 0), frame=specimenFrame("1")))
