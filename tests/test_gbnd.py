# the grain boundary normal distribution from the traces of a planar section
import numpy as np
import pytest

from mtex import calcGBND, calcGBNDKernel, calcGrains, degree
from mtex.io.sampledata import testGrainsMap as grainsMap


@pytest.fixture(scope="module")
def section():
  ebsd = grainsMap()
  grains = calcGrains(ebsd, 5 * degree)
  return ebsd, grains, grains.boundary["indexed"]


def test_the_kernel_is_nonnegative():
  Psi, psi = calcGBNDKernel(5 * degree, nonneg=True)
  t = np.linspace(-1, 1, 101)
  assert np.all(Psi.eval(t) >= 0) and Psi.eval(0.0) > Psi.eval(0.5)
  assert psi.bandwidth == 62 and psi.antipodal


def test_the_default_kernel_inverts_the_integral_equation():
  # MATLAB's calcGBNDKernel(5*degree) with chebfun, to its printed digits
  Psi, psi = calcGBNDKernel(5 * degree)
  assert np.allclose(Psi.eval(np.array([0, 0.05, 0.1, 0.3, 1.0])),
                     [276.147304114098, 167.866113593382, -9.03704578585202, -20.5916324566095, 2.91254597144238], rtol=1e-9)
  assert np.isclose(Psi.A[6], -55.4416483062927, rtol=1e-9) and np.all(Psi.A[1::2] == 0)
  assert psi.bandwidth == 62 and psi.antipodal


def test_calcGBND_isADensityOfAxesInTheCrystalFrame(section):
  ebsd, grains, gB = section
  f = calcGBND(gB, ebsd, halfwidth=10 * degree, nonneg=True)
  assert f.frame is gB.CSList[int(gB.phaseId[0, 0])] and f.antipodal
  assert np.isclose(f.mean(), 1, atol=1e-3)


def test_calcGBND_withTheDefaultKernelHasMeanOne(section):
  # the default kernel, a density of mean one too
  ebsd, _, gB = section
  assert np.isclose(calcGBND(gB, ebsd, halfwidth=10 * degree).mean(), 1, atol=1e-3)


def test_calcGBND_fromTheGrainsHasMeanOne(section):
  # the grains give the orientations of their means, the same boundaries
  _, grains, gB = section
  g = calcGBND(gB, grains, halfwidth=10 * degree, nonneg=True)
  assert np.isclose(g.mean(), 1, atol=1e-3)


def test_calcGBND_isAMethodOfTheBoundaries(section):
  # the method form, MATLAB's gB.calcGBND(ebsd)
  ebsd, _, gB = section
  f = calcGBND(gB, ebsd, halfwidth=10 * degree, nonneg=True)
  assert np.allclose(gB.calcGBND(ebsd, halfwidth=10 * degree, nonneg=True)._fhat, f._fhat)


def test_the_character_distribution_of_one_misorientation(section):
  ebsd, grains, gB = section
  # the misorientation across the first boundary, conditioned on, keeps its segments
  mori = gB[:1].misorientation[0]
  f = calcGBND(gB, grains, mori, halfwidth=10 * degree, nonneg=True)
  assert f.antipodal and np.isclose(f.mean(), 1, atol=1e-3) and f.frame.numSym() == 1
