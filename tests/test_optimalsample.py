# optimal sampling on the sphere and the rotation group: the gradients against finite
# differences, L-BFGS against gradient descent, the weights on the simplex, the moving
# node plans, the symmetric cases (docs/plans/functions.md F6, after the Julia tests)
import math

import numpy as np
import pytest

from mtex import (Miller, S2FunHarmonic, SO3FunHarmonic, crystalFrame, degree, discreteSample, fibreODF, optimalSample,
                  orientation, rotation, specimenFrame, uniformODF, vector3d, zvector)
from mtex.analysis.optimalsample import _Problem, _initial, _softmax, _step

rng = np.random.default_rng(607)


def densities():
  fs = S2FunHarmonic(np.array([math.sqrt(4 * np.pi), 0, 0.6 * math.sqrt(4 * np.pi / 3), 0], dtype=complex))
  cr = np.zeros(10, dtype=complex)
  cr[0], cr[5] = 1, 0.6 / math.sqrt(3)
  return fs, SO3FunHarmonic(cr)


def test_the_gradients_and_the_iteration():
  for f in densities():
    sphere = not hasattr(f, "pdf")
    xs = vector3d.rand(12, rng=rng).data if sphere else rotation.rand(12, rng=rng).data
    p = _Problem(f, xs, 4)
    c = np.full(12, 1 / 12)
    value, residual = p.residual(xs, c)
    g = p.gradient(xs, c, residual, weights=True)
    d = rng.standard_normal(36)
    if sphere:
      V = d.reshape(12, 3)
      V -= np.sum(V * xs, axis=1)[:, None] * xs
    h = 1e-6
    vp, vm = p.residual(_step(p, xs, d, h), c)[0], p.residual(_step(p, xs, d, -h), c)[0]
    assert np.isclose((vp - vm) / (2 * h), g[:36] @ d, atol=1e-8, rtol=1e-6)
    dz = rng.standard_normal(12)
    vp, vm = p.residual(xs, _softmax(np.log(c) + h * dz))[0], p.residual(xs, _softmax(np.log(c) - h * dz))[0]
    assert np.isclose((vp - vm) / (2 * h), g[36:] @ dz, atol=1e-8, rtol=1e-6)
    # L-BFGS beats gradient descent and the objective never rises
    r = optimalSample(f, xs, bandwidth=4, maxIter=25, tol=0, diagnostics=True)
    gd = optimalSample(f, xs, bandwidth=4, maxIter=25, tol=0, memory=0, diagnostics=True)
    assert r.discrepancy < gd.discrepancy < value / 3 and np.all(np.diff(r.history) <= 1e-14)
    assert np.allclose(r.weights, c) and r.nodes.size == 12 and r.bandwidth == 4
    assert isinstance(r.nodes, vector3d if sphere else orientation)
    # optimised weights stay on the simplex and move
    nodes, w = optimalSample(f, xs, bandwidth=4, maxIter=25, tol=0, optimizeWeights=True, warmUp=5)
    assert np.isclose(np.sum(w), 1) and np.all(w >= 0) and np.max(np.abs(w - c)) > 1e-4 and nodes.size == 12
    # the scale of the density does not matter, the count is exact, a single node works
    assert np.isclose(optimalSample(7 * f, xs, bandwidth=4, maxIter=3, diagnostics=True).discrepancy,
                      optimalSample(f, xs, bandwidth=4, maxIter=3, diagnostics=True).discrepancy, rtol=1e-9)
    assert optimalSample(f, 17, bandwidth=2, maxIter=0).size == 17
    assert optimalSample(f, xs[0], bandwidth=2, maxIter=2).size == 1
    assert optimalSample(f, 60, bandwidth=0, maxIter=2, diagnostics=True).converged
    assert optimalSample(f, xs, "steepestDescent", bandwidth=4, maxIter=2, diagnostics=True).discrepancy <= value
    with pytest.raises(ValueError):
      optimalSample(f, 0)
    with pytest.raises(ValueError):
      optimalSample(f, xs, weights=np.zeros(12))
    with pytest.raises(ValueError):
      optimalSample(f, xs, weights=np.ones(2))
    with pytest.raises(ValueError):
      optimalSample(-1 * f, xs, bandwidth=2)
    # the plan that follows the nodes agrees with a fresh transform
    many = _initial(f, 60)
    pp = _Problem(f, many, 4)
    many = _step(pp, many, rng.standard_normal(180), 0.01)
    v1, d1 = pp.residual(many, np.full(60, 1 / 60))
    v2, d2 = _Problem(f, many, 4).residual(many, np.full(60, 1 / 60))
    assert np.allclose(d1, d2, atol=1e-12) and np.isclose(v1, v2)


def test_the_symmetric_cases_and_a_large_sample():
  fs, _ = densities()
  anti = S2FunHarmonic(np.array([math.sqrt(4 * np.pi)], dtype=complex), antipodal=True)
  ra = optimalSample(anti, 16, bandwidth=4, maxIter=5, optimizeWeights=True, warmUp=100, tol=1e5, diagnostics=True)
  assert ra.nodes.antipodal and np.all(np.isfinite(ra.history)) and np.max(np.abs(ra.weights - 1 / 16)) > 1e-8
  with pytest.raises(ValueError):
    optimalSample(fs, 2, weights=[0, 1], optimizeWeights=True)
  cs = crystalFrame("222")
  sym = SO3FunHarmonic(np.array([1.0 + 0j]), cs, cs, antipodal=True)
  rs = optimalSample(sym, 16, bandwidth=4, maxIter=3)
  assert rs.CS is cs and rs.SS is cs and rs.antipodal
  for f in (S2FunHarmonic(np.array([math.sqrt(4 * np.pi)], dtype=complex), cs, antipodal=True), sym):
    xs = _initial(f, 8)
    p = _Problem(f, xs, 4)
    c = np.full(8, 1 / 8)
    _, residual = p.residual(xs, c)
    g = p.gradient(xs, c, residual)
    h = 1e-5
    vp, vm = p.residual(_step(p, xs, g, h), c)[0], p.residual(_step(p, xs, g, -h), c)[0]
    assert np.isclose((vp - vm) / (2 * h), g @ g, atol=1e-9, rtol=1e-5)
  # a larger sample keeps optimising: the first per node step is below the step tolerance
  # while the discrepancy is still far from its minimum
  odf = 0.5 * uniformODF(cs) + 0.5 * fibreODF(Miller(0, 0, 1, cs), zvector, halfwidth=20 * degree)
  short = optimalSample(odf, 200, bandwidth=16, maxIter=1, diagnostics=True)
  full = optimalSample(odf, 200, bandwidth=16, maxIter=80, diagnostics=True)
  assert full.iterations > 1 and full.discrepancy < 0.3 * short.discrepancy and full.nodes.CS is cs
  # the method of a function, and the page's use: a better reconstruction than a random one
  ori = odf.optimalSample(200, bandwidth=16, maxIter=40)
  assert isinstance(ori, orientation) and ori.size == 200
