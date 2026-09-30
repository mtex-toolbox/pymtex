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


# one density with twelve random nodes, their equal weights and the discrepancy they start from
@pytest.fixture(scope="module", params=[0, 1], ids=["S2", "SO3"])
def start(request):
  f = densities()[request.param]
  sphere = not hasattr(f, "_poleFigure")
  xs = vector3d.rand(12, rng=rng).data if sphere else rotation.rand(12, rng=rng).data
  p = _Problem(f, xs, 4)
  c = np.full(12, 1 / 12)
  value, residual = p.residual(xs, c)
  return f, sphere, xs, p, c, value, residual


def test_gradient_inTheNodesIsTheFiniteDifference(start):
  f, sphere, xs, p, c, value, residual = start
  g = p.gradient(xs, c, residual, weights=True)
  d = rng.standard_normal(36)
  if sphere:
    V = d.reshape(12, 3)
    V -= np.sum(V * xs, axis=1)[:, None] * xs
  h = 1e-6
  vp, vm = p.residual(_step(p, xs, d, h), c)[0], p.residual(_step(p, xs, d, -h), c)[0]
  assert np.isclose((vp - vm) / (2 * h), g[:36] @ d, atol=1e-8, rtol=1e-6)


def test_gradient_inTheWeightsIsTheFiniteDifference(start):
  f, sphere, xs, p, c, value, residual = start
  g = p.gradient(xs, c, residual, weights=True)
  h = 1e-6
  dz = rng.standard_normal(12)
  vp, vm = p.residual(xs, _softmax(np.log(c) + h * dz))[0], p.residual(xs, _softmax(np.log(c) - h * dz))[0]
  assert np.isclose((vp - vm) / (2 * h), g[36:] @ dz, atol=1e-8, rtol=1e-6)


def test_optimalSample_LBFGSBeatsGradientDescentMonotonically(start):
  # L-BFGS beats gradient descent and the objective never rises
  f, sphere, xs, p, c, value, residual = start
  r = optimalSample(f, xs, bandwidth=4, maxIter=25, tol=0, diagnostics=True)
  gd = optimalSample(f, xs, bandwidth=4, maxIter=25, tol=0, memory=0, diagnostics=True)
  assert r.discrepancy < gd.discrepancy < value / 3 and np.all(np.diff(r.history) <= 1e-14)
  assert np.allclose(r.weights, c) and r.nodes.size == 12 and r.bandwidth == 4
  assert isinstance(r.nodes, vector3d if sphere else orientation)


def test_optimalSample_optimisedWeightsStayOnTheSimplex(start):
  # optimised weights stay on the simplex and move
  f, sphere, xs, p, c, value, residual = start
  nodes, w = optimalSample(f, xs, bandwidth=4, maxIter=25, tol=0, optimizeWeights=True, warmUp=5)
  assert np.isclose(np.sum(w), 1) and np.all(w >= 0) and np.max(np.abs(w - c)) > 1e-4 and nodes.size == 12


def test_optimalSample_ignoresTheScaleOfTheDensity(start):
  # the scale of the density does not matter
  f, sphere, xs, p, c, value, residual = start
  assert np.isclose(optimalSample(7 * f, xs, bandwidth=4, maxIter=3, diagnostics=True).discrepancy,
                    optimalSample(f, xs, bandwidth=4, maxIter=3, diagnostics=True).discrepancy, rtol=1e-9)


def test_optimalSample_givesTheCountAskedFor(start):
  # the count is exact, a single node works
  f, sphere, xs, p, c, value, residual = start
  assert optimalSample(f, 17, bandwidth=2, maxIter=0).size == 17
  assert optimalSample(f, xs[0], bandwidth=2, maxIter=2).size == 1
  assert optimalSample(f, 60, bandwidth=0, maxIter=2, diagnostics=True).converged
  assert optimalSample(f, xs, "steepestDescent", bandwidth=4, maxIter=2, diagnostics=True).discrepancy <= value


def test_optimalSample_refusesWhatItCannotSample(start):
  f, sphere, xs, p, c, value, residual = start
  with pytest.raises(ValueError):
    optimalSample(f, 0)
  with pytest.raises(ValueError):
    optimalSample(f, xs, weights=np.zeros(12))
  with pytest.raises(ValueError):
    optimalSample(f, xs, weights=np.ones(2))
  with pytest.raises(ValueError):
    optimalSample(-1 * f, xs, bandwidth=2)


def test_Problem_planFollowingTheNodesIsAFreshTransform(start):
  # the plan that follows the nodes agrees with a fresh transform
  f = start[0]
  many = _initial(f, 60)
  pp = _Problem(f, many, 4)
  many = _step(pp, many, rng.standard_normal(180), 0.01)
  v1, d1 = pp.residual(many, np.full(60, 1 / 60))
  v2, d2 = _Problem(f, many, 4).residual(many, np.full(60, 1 / 60))
  assert np.allclose(d1, d2, atol=1e-12) and np.isclose(v1, v2)


def test_optimalSample_ofAnAntipodalDensityOptimisesWeights():
  anti = S2FunHarmonic(np.array([math.sqrt(4 * np.pi)], dtype=complex), antipodal=True)
  ra = optimalSample(anti, 16, bandwidth=4, maxIter=5, optimizeWeights=True, warmUp=100, tol=1e5, diagnostics=True)
  assert ra.nodes.antipodal and np.all(np.isfinite(ra.history)) and np.max(np.abs(ra.weights - 1 / 16)) > 1e-8


def test_optimalSample_refusesZeroWeightsToOptimise():
  fs, _ = densities()
  with pytest.raises(ValueError):
    optimalSample(fs, 2, weights=[0, 1], optimizeWeights=True)


def test_optimalSample_keepsTheGroupsOfASymmetricFunction():
  cs = crystalFrame("222")
  sym = SO3FunHarmonic(np.array([1.0 + 0j]), cs, cs, antipodal=True)
  rs = optimalSample(sym, 16, bandwidth=4, maxIter=3)
  assert rs.CS is cs and rs.SS is cs and rs.antipodal


@pytest.mark.parametrize("sphere", [True, False], ids=["S2", "SO3"])
def test_gradient_ofASymmetricFunctionIsTheFiniteDifference(sphere):
  cs = crystalFrame("222")
  if sphere:
    f = S2FunHarmonic(np.array([math.sqrt(4 * np.pi)], dtype=complex), cs, antipodal=True)
  else:
    f = SO3FunHarmonic(np.array([1.0 + 0j]), cs, cs, antipodal=True)
  xs = _initial(f, 8)
  p = _Problem(f, xs, 4)
  c = np.full(8, 1 / 8)
  _, residual = p.residual(xs, c)
  g = p.gradient(xs, c, residual)
  h = 1e-5
  vp, vm = p.residual(_step(p, xs, g, h), c)[0], p.residual(_step(p, xs, g, -h), c)[0]
  assert np.isclose((vp - vm) / (2 * h), g @ g, atol=1e-9, rtol=1e-5)


def basalFibre():
  hexa = crystalFrame("622")
  return hexa, fibreODF(hexa.cAxis, zvector, specimenFrame("mmm"))


def test_gradient_throughTheFoldedPlanIsTheDirectSums():
  # a specimen group with a two-fold about z: the gradient through the folded plan is the
  # gradient of the direct sums
  import mtex.analysis.optimalsample as osm
  _, fib = basalFibre()
  xs, c = _initial(fib, osm.DIRECT_SUM_NODES), np.full(osm.DIRECT_SUM_NODES, 1 / osm.DIRECT_SUM_NODES)
  planned = _Problem(fib, xs, 16)
  assert planned.plan is not None
  old, osm.DIRECT_SUM_NODES = osm.DIRECT_SUM_NODES, 10 ** 9
  try:
    direct = _Problem(fib, xs, 16)
  finally:
    osm.DIRECT_SUM_NODES = old
  gp = planned.gradient(xs, c, planned.residual(xs, c)[1])
  gd = direct.gradient(xs, c, direct.residual(xs, c)[1])
  assert np.linalg.norm(gp - gd) < 1e-5 * np.linalg.norm(gd)


def test_optimalSample_ofTheBasalFibrePutsTheCAxesAlongZ():
  # and the basal fibre sample puts the c-axes along Z
  hexa, fib = basalFibre()
  basal = optimalSample(fib, 300, bandwidth=16, maxIter=40)
  assert np.mean(np.abs((basal * Miller(0, 0, 0, 1, hexa)).normalize().data[..., 2])) > 0.9


def fibreAndUniform():
  cs = crystalFrame("222")
  return cs, 0.5 * uniformODF(cs) + 0.5 * fibreODF(Miller(0, 0, 1, cs), zvector, halfwidth=20 * degree)


def test_optimalSample_ofALargerSampleKeepsOptimising():
  # a larger sample keeps optimising: the first per node step is below the step tolerance
  # while the discrepancy is still far from its minimum
  cs, odf = fibreAndUniform()
  short = optimalSample(odf, 200, bandwidth=16, maxIter=1, diagnostics=True)
  full = optimalSample(odf, 200, bandwidth=16, maxIter=80, diagnostics=True)
  assert full.iterations > 1 and full.discrepancy < 0.3 * short.discrepancy and full.nodes.CS is cs


def test_optimalSample_isAMethodOfTheFunction():
  # the method of a function is the same call
  _, odf = fibreAndUniform()
  ori = odf.optimalSample(200, bandwidth=16, maxIter=1)
  assert isinstance(ori, orientation) and ori.size == 200
