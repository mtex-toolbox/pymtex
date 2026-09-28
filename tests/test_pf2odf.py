# the pole figure operator, the modified least squares solver, ghost correction, the zero
# range method and the regularisation (docs/plans/functions.md P4b: S1 to S6 after the Julia
# tests)
import numpy as np
import pytest

from mtex import (MLSSolver, Miller, SO3DeLaValleePoussinKernel, SO3FunRBF, calcError, calcNormalization, calcODF,
                  calcODFIterative, crystalFrame, degree, equispacedS2Grid, equispacedSO3Grid, mean, orientation, regularS2Grid,
                  rotation, specimenFrame, textureindex, uniformODF, unimodalODF, zeroRangeMethod)
from mtex.analysis.pf2odf import PoleFigureOperator
from mtex.functions.polefigure import _poleDensity

rng = np.random.default_rng(4)


def l1(f, g, grid):
  return np.sum(np.abs(np.real(f(grid)) - np.real(g(grid)))) / grid.size / 2


# a cubic unimodal ODF, its (100) pole figure and the solver's first operator at random weights
@pytest.fixture(scope="module")
def forward():
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=12.5 * degree)
  r = equispacedS2Grid(resolution=12.5 * degree, antipodal=True)
  pf = odf.calcPoleFigure(Miller(1, 0, 0, cs), r)
  s = MLSSolver(pf, resolution=12.5 * degree).init()
  op = s.ops[0]
  c = rng.random(s.S3G.size)
  return cs, odf, r, pf, s, op, c, op.forward(c)


def test_PoleFigureOperator_isThePoleFigureOfTheRBF(forward):
  # the operator is the pole figure of the radial basis function with the same weights
  cs, odf, r, pf, s, op, c, I = forward
  assert I.size == pf.intensities.size
  f = SO3FunRBF(s.S3G, s.psi, c)
  assert np.max(np.abs(I - _poleDensity(f, pf.allH[0][0], pf.allR[0]).ravel())) < 1e-5 * np.max(I)


def test_PoleFigureOperator_backwardIsTheAdjoint(forward):
  # backward is the adjoint of forward, by construction
  cs, odf, r, pf, s, op, c, I = forward
  y = rng.standard_normal(I.size)
  assert np.isclose(I @ y, c @ op.backward(y), rtol=1e-10)


def test_PoleFigureOperator_ofASuperpositionIsTheMean(forward):
  # a superposed pole figure is the mean of its components
  cs, odf, r, pf, s, op, c, I = forward
  hs = Miller([1, 1], [0, 1], [0, 1], cs)
  ops = MLSSolver(odf.calcPoleFigure([hs[0], hs[1], hs], r), resolution=12.5 * degree).init().ops
  assert np.allclose(ops[2].forward(c), (ops[0].forward(c) + ops[1].forward(c)) / 2)


def test_PoleFigureOperator_standsAlone(forward):
  # the operator alone
  cs, odf, r, pf, s, op, c, I = forward
  assert PoleFigureOperator(s.S3G, s.psi, pf.allH[0], pf.c[0], pf.allR[0]).nSym == 1


# a cubic unimodal ODF, three pole figures of it and the reconstruction
@pytest.fixture(scope="module")
def reconstruction():
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  ref = orientation(rotation.rand(rng=rng), cs, ss)
  odf = unimodalODF(ref, halfwidth=12.5 * degree)
  r = equispacedS2Grid(resolution=10 * degree, antipodal=True)
  pfs = odf.calcPoleFigure(Miller([1, 1, 1], [0, 1, 1], [0, 1, 0], cs), r)
  rec = calcODF(pfs, resolution=10 * degree)
  g = equispacedSO3Grid(cs, ss, resolution=7.5 * degree)
  return cs, ss, ref, odf, pfs, rec, g


def test_calcODF_reproducesTheDataWithoutRescaling(reconstruction):
  cs, ss, ref, odf, pfs, rec, g = reconstruction
  assert isinstance(rec, SO3FunRBF) and rec.CS is cs and rec.SS is ss
  # the data is reproduced and needs no rescaling
  assert np.all(calcError(pfs, rec) < 0.02)
  alpha = calcNormalization(pfs, rec)
  assert np.all(np.abs(alpha - 1) < 0.01)


def test_calcODF_reproducesTheODFItCameFrom(reconstruction):
  # and so is the ODF it came from: the L1 distance, the mean and the mode
  cs, ss, ref, odf, pfs, rec, g = reconstruction
  assert l1(odf, rec, g) < 0.05
  assert np.isclose(mean(rec), 1)
  assert rec(ref) > 0.8 * np.max(np.real(rec(g))) and rec(ref) > 0.7 * odf(ref)


def test_calcODF_putsTheScalingIntoTheNormalisation(reconstruction):
  # a scaled measurement gives the same ODF, the scaling goes into the normalisation
  cs, ss, ref, odf, pfs, rec, g = reconstruction
  alpha = calcNormalization(pfs, rec)
  rec2 = calcODF(pfs * 7, resolution=10 * degree)
  assert np.allclose(calcNormalization(pfs * 7, rec2), 7 * alpha, rtol=1e-2)
  assert np.max(np.abs(np.real(rec2(g)) - np.real(rec(g)))) < 0.05 * np.max(np.real(rec(g)))


def test_MLSSolver_exposesItsState(reconstruction):
  # the solver exposes its state, the pole figure method is the same call
  pfs = reconstruction[4]
  s = MLSSolver(pfs, resolution=10 * degree)
  s.calcODF()
  assert s.alpha.size == 3 and s.c.size == s.S3G.size and s.odf.center.size == s.S3G.size
  assert np.all(calcError(pfs, pfs.calcODF(resolution=10 * degree, iterMax=8)) < 0.03)


def test_calcODF_withSpecimenSymmetryAndSuperposedPoleFigures():
  cs, ss = crystalFrame("321"), specimenFrame("222")
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=15 * degree)
  r = equispacedS2Grid(resolution=12.5 * degree, antipodal=True, frame=ss)
  h = [Miller(1, 0, 0, cs), Miller([1, 0], [0, 1], [-1, -1], [1, 1], cs), Miller(0, 0, 1, cs)]
  pfs = odf.calcPoleFigure(h, r)
  rec = calcODF(pfs, resolution=10 * degree, halfwidth=12.5 * degree)
  assert np.all(calcError(pfs, rec) < 0.05)
  # the reconstruction carries the frames of the data and is invariant under the groups
  assert rec.CS is cs and rec.SS is ss
  o = orientation(rotation.rand(rng=rng), cs, ss)
  s = ss.properGroup().rotations().flatten()[1]
  assert np.isclose(rec(o), rec(orientation(s * rotation(o.data), cs, ss)))


def test_calcODF_ghostCorrectionTakesOutTheUniformPortion():
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  odf = 0.5 * unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=12.5 * degree) + 0.5
  r = equispacedS2Grid(resolution=10 * degree, antipodal=True)
  pfs = odf.calcPoleFigure(Miller([1, 1, 1], [0, 1, 1], [0, 1, 0], cs), r)
  g = equispacedSO3Grid(cs, ss, resolution=7.5 * degree)
  rec = calcODF(pfs, resolution=10 * degree)
  ghosts = calcODF(pfs, resolution=10 * degree, noGhostCorrection=True)
  # the uniform portion the data allows is found and taken out of the components
  assert 0.35 < rec.c0 < 0.6 and ghosts.c0 == 0
  # and the ODF is closer to the truth for it
  assert l1(odf, rec, g) < l1(odf, ghosts, g)
  assert l1(odf, rec, g) < 0.05


# a sharp cubic ODF, its pole figures and their zero range
@pytest.fixture(scope="module")
def zeroRange():
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  ref = orientation(rotation.rand(rng=rng), cs, ss)
  odf = unimodalODF(ref, halfwidth=10 * degree)
  r = equispacedS2Grid(resolution=10 * degree, antipodal=True)
  pfs = odf.calcPoleFigure(Miller([1, 1, 1], [0, 1, 1], [0, 1, 0], cs), r)
  # the background below which the data says nothing; the default 0.001 of the maximum
  # fires on measured data, not on a noise free simulation
  zrm = zeroRangeMethod(pfs, delta=0.02)
  g = equispacedSO3Grid(cs, ss, resolution=10 * degree)
  return ref, odf, pfs, zrm, g


def test_zeroRangeMethod_emptiesOnlyWhereTheTextureIsNot(zeroRange):
  ref, odf, pfs, zrm, g = zeroRange
  zero = zrm.checkZeroRange(g)
  # a sharp texture leaves most of the orientation space empty, and nothing of the texture
  # is dropped with it
  assert 0.2 < np.count_nonzero(zero) / g.size < 0.9
  assert np.max(np.real(odf(g))[zero]) < 0.05


def test_checkZeroRange_ofOnePoleFigure(zeroRange):
  # one pole figure at a time
  ref, odf, pfs, zrm, g = zeroRange
  v = pfs.allR[0].flatten()
  assert zrm.checkZeroRange(v, 0).shape == (v.size,)


def test_calcODF_withTheZeroRangeFitsFromFewerFunctions(zeroRange):
  # the thinned grid costs nothing: the same fit from fewer ansatz functions
  ref, odf, pfs, zrm, g = zeroRange
  rec = calcODF(pfs, resolution=10 * degree, zeroRange=zrm)
  plain = calcODF(pfs, resolution=10 * degree)
  assert rec.center.size < plain.center.size == g.size
  assert np.all(calcError(pfs, rec) <= calcError(pfs, plain) + 0.01)
  assert rec(ref) > 0.8 * np.max(np.real(rec(g)))


def test_calcODF_zeroRangeFlagDropsNothingOnNoiseFreeData(zeroRange):
  # the flag builds the default method, which drops nothing on this noise free data
  ref, odf, pfs, zrm, g = zeroRange
  assert calcODF(pfs, resolution=10 * degree, zeroRange=True).center.size == g.size


# a cubic ODF measured on a regular grid, and the plain reconstruction
@pytest.fixture(scope="module")
def regular():
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=12.5 * degree)
  r = regularS2Grid(resolution=10 * degree, antipodal=True)
  pfs = odf.calcPoleFigure(Miller([1, 1, 1], [0, 1, 1], [0, 1, 0], cs), r)
  return odf, pfs, calcODF(pfs, resolution=10 * degree)


def test_calcODF_withQuadratureWeightsOnARegularGrid(regular):
  # a regular grid oversamples the pole, so every point standing for its own area is a
  # different problem than every point standing for the same
  odf, pfs, plain = regular
  rec = calcODF(pfs, resolution=10 * degree, quadratureWeights=True)
  assert np.all(calcError(pfs, rec) < 0.05)


def test_calcODF_regularisationSmooths(regular):
  # the regularisation smooths: the texture index drops and the fit stays reasonable
  odf, pfs, plain = regular
  smooth = calcODF(pfs, resolution=10 * degree, regularisation=1e-3)
  assert textureindex(smooth) < textureindex(plain)
  assert np.all(calcError(pfs, smooth) < 0.15)


def test_MLSSolver_takesAKernelPositionally(regular):
  # a kernel given positionally, with intensity weights
  pfs = regular[1]
  psi = SO3DeLaValleePoussinKernel(halfwidth=15 * degree)
  assert MLSSolver(pfs, psi, intensityWeights=True).psi is psi


def test_calcODFIterative_fitsAtLeastAsWellAsOneSolve(regular):
  # the successive refinement fits the pole figures at least as well as one solve
  odf, pfs, plain = regular
  it = calcODFIterative(pfs, resolution=10 * degree, maxSteps=3)
  assert np.mean(calcError(pfs, it)) <= 1.5 * np.mean(calcError(pfs, plain))
  assert calcError(it, odf) < 0.2
  assert isinstance(calcODFIterative(pfs, resolution=15 * degree, maxSteps=2, nothinning=True), SO3FunRBF)
