# the tile wise phase correlation, MATLAB tools/registration_tools
import numpy as np
import pytest
from scipy.ndimage import fourier_shift, gaussian_filter

from mtex import mapImage, pairShifts, spatialTransformShift, vector3d, xcfShift
from mtex.maps.registration import xcfFilters, xcfROIGrid


# a smooth random picture and its copy moved by (2.37, -1.61) pixels, the first twenty rows of
# the copy padding; the pair bench/matlab/probe_trueebsd.m correlated
def pair():
  rng = np.random.default_rng(1)
  A = gaussian_filter(rng.standard_normal((256, 320)), 3)
  B = np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(A), (-1.61, 2.37))))
  B[:20] = np.nan
  return A, B


def test_a_known_sub_pixel_shift_is_recovered():
  A, B = pair()
  u, peak, pos = xcfShift(A, B, ROISize=64, numROI=8)
  ux, uy = np.asarray(u.x), np.asarray(u.y)
  assert u.size == 48 and np.isfinite(peak).all()
  # the window pulls every tile a little towards no shift, as it does MATLAB's
  assert abs(np.median(ux) - 2.37) < 0.12 and abs(np.median(uy) + 1.61) < 0.1
  assert np.all(np.abs(ux - 2.37) < 0.15) and np.all(np.abs(uy + 1.61) < 0.15)


def test_the_tiles_are_matlabs_tile_for_tile():
  A, B = pair()
  u, peak, pos = xcfShift(A, B, ROISize=64, numROI=8)
  # MATLAB counts pixels from one, so its tile centres are one further on
  for k, ux, uy, pk, px, py in ((0, 2.304, -1.552, 4037311.615173989, 34, 54), (7, 2.268, -1.54, 4816370.623516851, 70, 88),
                                (20, 2.296, -1.568, 4004815.155961514, 143, 122), (47, 2.276, -1.532, 3991756.6033212147, 288, 224)):
    assert abs(u.x[k] - ux) < 1e-12 and abs(u.y[k] - uy) < 1e-12
    assert abs(peak[k] / pk - 1) < 1e-12
    assert pos.x[k] == px - 1 and pos.y[k] == py - 1


def test_the_padding_is_left_out_and_a_flat_tile_is_nan():
  A, B = pair()
  _, _, pos = xcfShift(A, B, ROISize=64, numROI=8)
  # no tile reaches into the twenty padded rows
  assert np.asarray(pos.y).min() - 32 >= 20
  C = A.copy()
  C[100:180, 120:200] = 1.0
  D = C.copy()
  u, peak, pos = xcfShift(C, D, ROISize=32, numROI=12)
  flat = (np.abs(np.asarray(pos.x) - 159.5) < 20) & (np.abs(np.asarray(pos.y) - 139.5) < 20)
  assert flat.any() and np.isnan(peak[flat]).all() and np.isnan(np.asarray(u.x)[flat]).all()
  assert np.nanmax(np.abs(np.asarray(u.x))) < 1e-9


def test_the_peak_is_the_weight_of_a_fit():
  A, B = pair()
  u, peak, pos = xcfShift(A, B, ROISize=64, numROI=8)
  T = spatialTransformShift.fit(pos, pos + u, weights=peak)
  assert abs(T.M[0, 2] - 2.37) < 0.15 and abs(T.M[1, 2] + 1.61) < 0.15


def test_two_images_answer_in_specimen_units():
  A, B = pair()
  mA = mapImage(A, dxy=0.5, origin=vector3d(10, 20, 0))
  mB = mapImage(B, dxy=0.5, origin=vector3d(10, 20, 0))
  u, peak, pos = xcfShift(mA, mB, ROISize=64, numROI=8)
  u0, _, pos0 = xcfShift(A, B, ROISize=64, numROI=8)
  assert np.allclose(u.data, 0.5 * u0.data)
  assert np.allclose(pos.data, [10, 20, 0] + 0.5 * pos0.data)
  with pytest.raises(ValueError):
    xcfShift(A, B[:, :100])


def test_the_tile_grid_and_the_filters():
  cx, cy = xcfROIGrid(64, (4, 3), (2, 2, 300, 200))
  assert cx.shape == (3, 4) and cx.min() - 32 >= 2 and cx.max() + 32 <= 302
  with pytest.raises(ValueError):
    xcfROIGrid(64, (4, 3), (2, 2, 40, 200))
  band, window = xcfFilters(64, (0, 0, 32, 16))
  assert band.shape == (64, 64) and band[0, 0] == 1 and band[32, 32] < 0.01 and band[0, 16] == 1
  assert np.allclose(window, window.T) and window.max() < 1


def test_the_mean_shift_of_a_pass():
  ps = pairShifts(vector3d(np.zeros((3, 3))), vector3d([[2, 0, 0], [0, 2, 0], [np.nan, 0, 0]]), [1, 1, np.nan], 32)
  assert np.allclose(ps.meanShift(2.0), [1, 0.5, 0.5])
  assert np.allclose(pairShifts().meanShift(1.0), 0)
