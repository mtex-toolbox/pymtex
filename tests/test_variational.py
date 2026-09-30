# M8: the one energy over the graph, the Julia D9b and the calibration of its length
import numpy as np
import pytest

from mtex import degree, splineFilter, halfQuadraticFilter, l1TVFilter, infimalConvolutionFilter
from mtex.analysis.variational import graphEigenvalue, regularisation
from conftest import close
from test_denoising import angleMapD, angles, HEX


def wave(Lambda, A, n=41):
  return angleMapD(lambda i, j: 0.5 * np.sin(2 * np.pi * (A[0, 0] * i + A[0, 1] * j) / Lambda), n, A=A)


def gain(e, s, n=41):
  mid = np.array([i + 20 * n for i in range(12, 29)])
  return np.max(np.abs(angles(s)[mid])) / np.max(np.abs(angles(e)[mid]))


@pytest.mark.parametrize("Lambda", [4.0, 8.0])
@pytest.mark.parametrize("A", [np.eye(2), HEX], ids=["square", "hexagonal"])
def test_splineFilter_halvesAWaveOfItsSmoothingLength(Lambda, A):
  # a plane wave of that wavelength comes out at half amplitude, which is what the length is
  # defined to mean, on either lattice and at either order of the penalty
  e = wave(Lambda, A)
  assert abs(gain(e, e.smooth(splineFilter(Lambda))) - 0.5) < 5e-3


def test_halfQuadraticFilter_halvesAWaveOfItsSmoothingLength():
  e = wave(8.0, np.eye(2))
  assert abs(gain(e, e.smooth(halfQuadraticFilter(8.0, l1DataFit=False, l1TV=False))) - 0.5) < 5e-3


def test_splineFilter_letsALongerWaveThroughBetter():
  # and a longer wave survives a given filter better than a shorter one
  F = splineFilter()
  assert gain(wave(4.0, np.eye(2)), wave(4.0, np.eye(2)).smooth(F)) < gain(wave(12.0, np.eye(2)), wave(12.0, np.eye(2)).smooth(F))


def test_regularization_isTheWeightOfALength():
  e = wave(8.0, np.eye(2))
  l = e.lattice
  # the eigenvalue of the square lattice at a wavelength of four steps: two offsets along the
  # wave at 1 - cos(pi / 2) and two across it at zero
  assert close(graphEigenvalue(l, 4.0), 2.0) and close(graphEigenvalue(l, 1.0), graphEigenvalue(l, 2.0))
  assert close(regularisation(splineFilter(4.0), l, 2), 0.25) and close(regularisation(splineFilter(alpha=3), l, 2), 3)
  assert close(regularisation(splineFilter(), l, 2), regularisation(splineFilter(4.0), l, 2))


def test_infimalConvolutionFilter_keepsASlope():
  # the infimal convolution keeps a slope where the first order filter alone flattens it
  ramp = angleMapD(lambda i, j: 0.3 * i, 15)
  ic = ramp.smooth(infimalConvolutionFilter(alpha=2.0, beta=2.0, threshold=np.pi))
  assert np.max(np.abs(angles(ic) - angles(ramp))) < 0.5
