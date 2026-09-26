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


def test_the_kernel_is_nonnegative_and_the_default_waits():
  Psi, psi = calcGBNDKernel(5 * degree, nonneg=True)
  t = np.linspace(-1, 1, 101)
  assert np.all(Psi.eval(t) >= 0) and Psi.eval(0.0) > Psi.eval(0.5)
  assert psi.bandwidth == 62 and psi.antipodal
  with pytest.raises(NotImplementedError):
    calcGBNDKernel(5 * degree)


def test_a_density_of_axes_in_the_crystal_frame(section):
  ebsd, grains, gB = section
  f = calcGBND(gB, ebsd, halfwidth=10 * degree, nonneg=True)
  assert f.frame is gB.CSList[int(gB.phaseId[0, 0])] and f.antipodal
  assert np.isclose(f.mean(), 1, atol=1e-3)
  # the grains give the orientations of their means, the same boundaries
  g = calcGBND(gB, grains, halfwidth=10 * degree, nonneg=True)
  assert np.isclose(g.mean(), 1, atol=1e-3)
  # the method form, MATLAB's gB.calcGBND(ebsd)
  assert np.allclose(gB.calcGBND(ebsd, halfwidth=10 * degree, nonneg=True)._fhat, f._fhat)


def test_the_character_distribution_of_one_misorientation(section):
  ebsd, grains, gB = section
  # the misorientation across the first boundary, conditioned on, keeps its segments
  mori = gB[:1].misorientation[0]
  f = calcGBND(gB, grains, mori, halfwidth=10 * degree, nonneg=True)
  assert f.antipodal and np.isclose(f.mean(), 1, atol=1e-3) and f.frame.numSym() == 1
