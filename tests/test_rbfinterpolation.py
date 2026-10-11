# the fit of a sum of radial kernels to values at orientations
import numpy as np

from mtex import (SO3DeLaValleePoussinKernel, SO3FunRBF, crystalFrame, degree, equispacedSO3Grid, interp, mean, orientation,
                  rotation, unimodalODF)
from mtex.functions.rbfinterpolation import mlsq, summationMatrix

rng = np.random.default_rng(11)
cs = crystalFrame("222")


def test_summationMatrix_isTheMeanOverTheEquivalents():
  psi = SO3DeLaValleePoussinKernel(halfwidth=15 * degree)
  centres = orientation(rotation.rand(30, rng=rng), cs)
  nodes = orientation(rotation.rand(40, rng=rng), cs)
  dense = summationMatrix(psi, centres, nodes, exact=True)
  f = SO3FunRBF(centres, psi, np.eye(30)[:, 3])
  assert np.allclose(dense[:, 3], np.real(f.eval(nodes)))


def test_mlsq_keepsTheWeightsPositiveAndFits():
  A = np.abs(rng.standard_normal((60, 20)))
  c = np.abs(rng.standard_normal(20))
  y = A @ c
  chat, it = mlsq(A, y, np.full(20, np.sum(c) / 20), maxIter=500, tol=1e-12)
  assert np.all(chat > 0) and np.linalg.norm(A @ chat - y) < 0.05 * np.linalg.norm(y)


# a density fitted to the values of a unimodal ODF
def densityFit():
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs), halfwidth=15 * degree)
  nodes = orientation(rotation.rand(2000, rng=rng), cs)
  return odf, interp(nodes, odf.eval(nodes), density=True, halfwidth=15 * degree)


def test_interp_densityIsPositiveOfMeanOne():
  _, f = densityFit()
  assert np.isclose(mean(f), 1) and np.all(f.weights > 0)


# the centres of negligible weight are dropped, as MATLAB's unimodalODF thins
def test_interp_densityDropsTheCentresOfNegligibleWeight():
  _, f = densityFit()
  assert f.center.size < equispacedSO3Grid(cs, resolution=15 * degree).size


def test_interp_densityFitsTheFunction():
  odf, f = densityFit()
  test = orientation(rotation.rand(200, rng=rng), cs)
  assert np.linalg.norm(np.real(f.eval(test) - odf.eval(test))) < 0.2 * np.linalg.norm(np.real(odf.eval(test)))


def test_SO3FunRBF_interpolateReportsTheIterationsOfLsqr():
  nodes = orientation(rotation.rand(300, rng=rng), cs)
  y = rng.standard_normal(300)
  f, it = SO3FunRBF.interpolate(nodes, y, withIterations=True, halfwidth=20 * degree)
  assert 0 < it <= 30


def test_SO3FunRBF_interpolateExactFitsEveryNode():
  nodes = orientation(rotation.rand(300, rng=rng), cs)
  y = rng.standard_normal(300)
  g = SO3FunRBF.interpolate(nodes, y, exact=True, halfwidth=5 * degree, tol=1e-12, maxIter=500)
  assert g.center.size == 300
  assert np.linalg.norm(np.real(g.eval(nodes)) - y) < 1e-3 * np.linalg.norm(y)
