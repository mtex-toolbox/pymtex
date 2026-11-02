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


# a random graph of three components over a vector field, some pixels measured
def randomGraphField(seed=3, n=60, d=3):
  from mtex.analysis.filters.variational import graphCSR
  rng = np.random.default_rng(seed)
  comp = np.repeat([1, 2, 3], n // 3)
  a, b = rng.integers(0, n, 400), rng.integers(0, n, 400)
  keep = (comp[a] == comp[b]) & (a < b)
  edges = np.unique(np.stack([a[keep], b[keep]], axis=1), axis=0)
  return graphCSR(edges, n), rng.normal(size=(n, d)), rng.normal(size=(n, d)), rng.random(n) > 0.3, comp


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("pin", [False, True])
def test_the_compiled_energy_is_the_numpy_energy(order, pin):
  from mtex.analysis.filters.variational import _energyOf, _componentEnergy
  G, f, x, isData, comp = randomGraphField()
  active = comp != 2
  args = (f, isData, pin, True, True, 0.8, 1e-2, 0.7, order, comp, 3)
  E = _energyOf(G, *args)(x, active)
  ref = _componentEnergy(G, G.laplacian() if order == 2 else None, *args, x)
  assert np.allclose(E[[1, 3]], ref[[1, 3]], rtol=1e-12) and E[2] == 0


def test_the_history_updated_step_by_step_is_anderson_rebuilt_from_the_last_steps():
  # the extrapolation from the ring of differences and its Gram matrix, against the least
  # squares over the last five iterates and steps stacked anew
  from mtex.analysis.filters.variational import _andersonHistory
  rng = np.random.default_rng(5)
  h, prev, Y, R = _andersonHistory((20, 3), 5), rng.normal(size=(20, 3)), [], []
  for _ in range(8):
    x = prev + rng.normal(size=(20, 3)) * 0.1
    xa = h.extrapolate(x, prev)
    Y, R = (Y + [x.ravel()])[-5:], (R + [(x - prev).ravel()])[-5:]
    if len(Y) > 1:
      dY, dR = np.diff(np.array(Y), axis=0).T, np.diff(np.array(R), axis=0).T
      gam = np.linalg.solve(dR.T @ dR + 1e-12 * np.eye(dR.shape[1]), dR.T @ R[-1])
      assert np.allclose(xa.ravel(), x.ravel() - dY @ gam, atol=1e-10)
    prev = x


def test_a_stopped_pixel_leaves_the_history():
  # once dropped, the rows of a pixel frozen at its value contribute nothing to the extrapolation
  from mtex.analysis.filters.variational import _andersonHistory
  rng = np.random.default_rng(6)
  h, prev = _andersonHistory((10, 3), 5), rng.normal(size=(10, 3))
  for _ in range(4):
    x = prev + rng.normal(size=(10, 3)) * 0.1
    h.extrapolate(x, prev)
    prev = x
  h.drop(np.array([0, 1]), prev)
  x = prev.copy()
  x[2:] += rng.normal(size=(8, 3)) * 0.1
  h.extrapolate(x, prev)
  assert np.all(h.DY[:h.count, :6] == 0) and np.all(h.DR[:h.count, :6] == 0)
  assert np.allclose(h.gram[:h.count, :h.count], h.DR[:h.count] @ h.DR[:h.count].T)

