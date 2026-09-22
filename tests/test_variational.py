# M8: the one energy over the graph, the Julia D9b and the calibration of its length
import numpy as np

from mtex import degree, splineFilter, halfQuadraticFilter, l1TVFilter, infimalConvolutionFilter
from mtex.analysis.variational import graphEigenvalue, regularization
from conftest import close
from test_denoising import angleMapD, angles, HEX


def wave(Lambda, A, n=41):
  return angleMapD(lambda i, j: 0.5 * np.sin(2 * np.pi * (A[0, 0] * i + A[0, 1] * j) / Lambda), n, A=A)


def gain(e, s, n=41):
  mid = np.array([i + 20 * n for i in range(12, 29)])
  return np.max(np.abs(angles(s)[mid])) / np.max(np.abs(angles(e)[mid]))


def test_D9b_the_smoothing_length_is_the_wavelength_it_halves():
  # a plane wave of that wavelength comes out at half amplitude, which is what the length is
  # defined to mean, on either lattice and at either order of the penalty
  for Lambda in (4.0, 8.0):
    for A in (np.eye(2), HEX):
      e = wave(Lambda, A)
      assert abs(gain(e, e.smooth(splineFilter(Lambda))) - 0.5) < 5e-3
  e = wave(8.0, np.eye(2))
  assert abs(gain(e, e.smooth(halfQuadraticFilter(8.0, l1DataFit=False, l1TV=False))) - 0.5) < 5e-3
  # and a longer wave survives a given filter better than a shorter one
  F = splineFilter()
  assert gain(wave(4.0, np.eye(2)), wave(4.0, np.eye(2)).smooth(F)) < gain(wave(12.0, np.eye(2)), wave(12.0, np.eye(2)).smooth(F))


def test_the_weight_of_a_length():
  e = wave(8.0, np.eye(2))
  l = e.lattice
  # the eigenvalue of the square lattice at a wavelength of four steps: two offsets along the
  # wave at 1 - cos(pi / 2) and two across it at zero
  assert close(graphEigenvalue(l, 4.0), 2.0) and close(graphEigenvalue(l, 1.0), graphEigenvalue(l, 2.0))
  assert close(regularization(splineFilter(4.0), l, 2), 0.25) and close(regularization(splineFilter(alpha=3), l, 2), 3)
  assert close(regularization(splineFilter(), l, 2), regularization(splineFilter(4.0), l, 2))
  # the infimal convolution keeps a slope where the first order filter alone flattens it
  ramp = angleMapD(lambda i, j: 0.3 * i, 15)
  ic = ramp.smooth(infimalConvolutionFilter(alpha=2.0, beta=2.0, threshold=np.pi))
  assert np.max(np.abs(angles(ic) - angles(ramp))) < 0.5
