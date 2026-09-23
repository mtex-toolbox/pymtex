# SO3FunHarmonic: construction, evaluation, symmetry and algebra (docs/plans/functions.md F3a to F3c)
import numpy as np
import pytest

from mtex import (SO3DeLaValleePoussinKernel, SO3FunHarmonic, crystalFrame, deg2dim, degree, dot, isRealWigner, mean, norm,
                  orientation, quadratureSO3Grid, rotation, specimenFrame, symmetrise, inv)

rng = np.random.default_rng(31)


def randomCoefficients(L):
  return rng.standard_normal(deg2dim(L + 1)) + 1j * rng.standard_normal(deg2dim(L + 1))


def test_F3a_construction_and_evaluation():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  psi = SO3DeLaValleePoussinKernel(halfwidth=10 * degree)
  f = SO3FunHarmonic(psi, cs, ss)
  assert f.isReal and not f.antipodal and f.bandwidth == psi.bandwidth
  assert f.CS is cs and f.SS is ss
  assert np.isclose(f.mean(), 1) and np.isclose(mean(f), 1)
  # the kernel centred at the identity, symmetrised, is the mean of the kernel over the
  # orbit of the identity, to its truncation
  A, B = cs.sym.properGroup().rotations().flatten(), ss.sym.properGroup().rotations().flatten()
  for r in rotation.rand(5, rng=rng):
    orbit = np.mean([psi(B[j] * r * A[i]) for i in range(A.size) for j in range(B.size)])
    assert abs(f(orientation(r, cs, ss)) - orbit) < 1e-2 * psi(1.0)
    assert np.isclose(f(r), f(orientation(r, cs, ss)))
  # unsymmetrised, it is the kernel itself
  f0 = SO3FunHarmonic(psi, cs, ss, symmetrise=False)
  for r in rotation.rand(3, rng=rng):
    assert abs(f0(r) - psi(r)) < 1e-2 * psi(1.0)
  with pytest.raises(ValueError):
    f(orientation(rotation.id, cs, specimenFrame("1")))
  # arrays and containers keep their shape and value type
  os = orientation(rotation.rand(4, 5, rng=rng), cs, ss)
  assert f(os).shape == (4, 5) and f(os).dtype == np.float64
  assert np.allclose(f(os), [[f(os[i, j]) for j in range(5)] for i in range(4)])
  # coefficients in, coefficients out, real or complex
  fhat = randomCoefficients(5)
  g = SO3FunHarmonic(fhat, cs, ss, symmetrise=False)
  assert not g.isReal and np.array_equal(g.fhat, fhat)
  assert SO3FunHarmonic(fhat, cs, ss).isclose(symmetrise(g))
  gr = SO3FunHarmonic(fhat, cs, ss, real=True, symmetrise=False)
  assert gr.isReal and isRealWigner(gr._fhat) and SO3FunHarmonic(g, real=True, symmetrise=False).isclose(gr)
  assert np.array_equal(SO3FunHarmonic(gr, real=False, symmetrise=False).fhat, gr.fhat)
  # a vector ending inside a degree block is padded to its end, as MATLAB does
  assert SO3FunHarmonic(rng.standard_normal(7), cs, ss).bandwidth == 1
  # from a callable by quadrature, and from samples on the quadrature nodes
  h = SO3FunHarmonic(lambda o: f(o), cs, ss, bandwidth=f.bandwidth)
  assert h.isclose(f, 1e-6)
  q = quadratureSO3Grid(f.bandwidth, cs, ss)
  o = q.nodes()
  h2 = SO3FunHarmonic.quadrature(o, f(o), bandwidth=f.bandwidth, weights=q.weights)
  assert h2.isclose(f, 1e-6)
  assert SO3FunHarmonic.quadrature(q, f(q), bandwidth=f.bandwidth).isclose(f, 1e-6)
  assert f.truncate(10).isclose(SO3FunHarmonic(f.fhat[:deg2dim(11)], cs, ss, symmetrise=False)) and f.truncate(40).bandwidth == 40
  assert SO3FunHarmonic(f, bandwidth=10).bandwidth == 10
  assert "bandwidth" in repr(f)


def test_F3b_symmetry():
  # the projection makes a random function invariant under every element of both groups,
  # in closure order, and is idempotent: MTEX's dead test in symmetrise.m for every group
  for name in ("1", "-1", "2/m", "mm2", "4", "-4", "4/mmm", "3", "-3m", "6", "-62m", "6/mmm", "23", "m-3m", "432"):
    cs = crystalFrame(name)
    for sname in ("1", "222", "mmm"):
      ss = specimenFrame(sname)
      f = SO3FunHarmonic(randomCoefficients(6), cs, ss)
      assert f.symmetrise().isclose(f)
      r = rotation.rand(rng=rng)
      E, S = cs.sym.rot.flatten().rot, ss.sym.rot.flatten().rot
      for i in range(E.size):
        for j in range(S.size):
          assert np.isclose(f(S[j] * r * E[i]), f(r))
  # inversion swaps the frames and the argument
  cs, ss = crystalFrame("-3m"), specimenFrame("222")
  f = SO3FunHarmonic(randomCoefficients(5), cs, ss)
  g = inv(f)
  assert g.CS is ss and g.SS is cs
  for r in rotation.rand(5, rng=rng):
    assert np.isclose(g(r.inv()), f(r))
  assert g.inv().isclose(f)
  # exchange symmetry needs one frame on both sides and makes f(g) = f(g^-1)
  with pytest.raises(ValueError):
    SO3FunHarmonic(f.fhat, cs, ss, antipodal=True)
  h = SO3FunHarmonic(randomCoefficients(5), cs, cs, antipodal=True)
  assert h.antipodal
  for r in rotation.rand(5, rng=rng):
    assert np.isclose(h(r.inv()), h(r))
  assert np.array_equal(h.withExchange(False).fhat, h.fhat) and not h.withExchange(False).antipodal
  assert h.withExchange(False).withExchange().isclose(h)
  # a real function stays real under everything
  fr = SO3FunHarmonic(randomCoefficients(5), cs, ss, real=True)
  assert isRealWigner(fr._fhat) and isRealWigner(fr.inv()._fhat) and isRealWigner(fr.conj()._fhat)


def test_F3c_algebra():
  cs, ss = crystalFrame("4/mmm"), specimenFrame("1")
  f = SO3FunHarmonic(randomCoefficients(5), cs, ss)
  g = SO3FunHarmonic(randomCoefficients(8), cs, ss)
  rs = rotation.rand(6, rng=rng)
  assert np.allclose((f + g)(rs), f(rs) + g(rs))
  assert np.allclose((f - g)(rs), f(rs) - g(rs))
  assert np.allclose((2.5 * f)(rs), 2.5 * f(rs)) and np.allclose((f / 2)(rs), f(rs) / 2)
  assert np.allclose((f + 3)(rs), f(rs) + 3) and np.allclose((3 - f)(rs), 3 - f(rs))
  assert np.allclose(f.conj()(rs), np.conj(f(rs)))
  assert (f + g).bandwidth == 8
  with pytest.raises(ValueError):
    f + SO3FunHarmonic(g.fhat, crystalFrame("m-3m"), ss)
  # mean, norm and the inner product are those of the orthonormal basis: check by quadrature
  q = quadratureSO3Grid(8)
  rots, w = q.nodes(), q.weights
  assert abs(g.mean() - np.sum(w * g(rots))) < 1e-6 * abs(g.mean())
  assert abs(norm(g) ** 2 - np.sum(w * np.abs(g(rots)) ** 2)) < 1e-6 * norm(g) ** 2
  assert abs(dot(f, g) - np.sum(w * np.conj(f(rots)) * g(rots))) < 1e-6 * abs(dot(f, g))
  # real functions give real numbers
  fr = SO3FunHarmonic(randomCoefficients(5), cs, ss, real=True)
  assert isinstance(fr.mean(), float) and isinstance(dot(fr, fr), float)
  assert (fr + fr).isReal and (2 * fr).isReal and not (fr + f).isReal and not (1j * fr).isReal
  # the harmonic power per degree
  assert len(g.power) == 9 and np.isclose(np.linalg.norm(g.power), norm(g))
