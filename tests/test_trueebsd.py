# the TrueEBSD job, MATLAB EBSDAnalysis/@trueEbsd
import os

import numpy as np
import pytest

from mtex import (gridLayout, mapImage, mapImageList, spatialTransformId, spatialTransformProjective, spatialTransformShift,
                  spatialTransformTilt, transformReferenceFrame, trueEbsd, vector3d)
from mtex.io.sampledata import mtexDataPath

rng = np.random.default_rng(11)
K = rng.uniform(2 * np.pi / 40, 2 * np.pi / 10, 40) * np.exp(1j * rng.uniform(0, 2 * np.pi, 40))
PHI = rng.uniform(0, 2 * np.pi, 40)


# a smooth picture defined everywhere, so that a distorted copy is exact
def picture(x, y):
  return np.sum(np.cos(K.real[:, None, None] * x + K.imag[:, None, None] * y + PHI[:, None, None]), axis=0)


A = np.array([[1.004, 0.012, 3.2], [-0.006, 0.997, -2.1], [0, 0, 1]])
H = np.array([[1.0, 0.0, 1.5], [0.0, 0.985, 0.5], [1e-4, 1.5e-4, 1.0]])


def moved(M, x, y):
  d = M[2, 0] * x + M[2, 1] * y + M[2, 2]
  return (M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d, (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d


# a chain of three: sheared and moved, tilted, the reference, on grids that pixelSizeMatch
# centres, so that the imposed transforms are stated in the coordinates the job fits in
def chain(r=200, c=240):
  y, x = np.mgrid[0:r, 0:c].astype(float)
  x, y = x - (c - 1) / 2, y - (r - 1) / 2
  ref = picture(x, y)
  one = picture(*moved(H, x, y))
  zero = picture(*moved(H, *moved(A, x, y)))
  return mapImageList([mapImage(zero, name="zero"), mapImage(one, name="one"), mapImage(ref, name="ref")])


def test_the_fitted_hops_recover_the_imposed_transforms():
  job = trueEbsd(chain(), [spatialTransformShift(), spatialTransformProjective()])
  job.pixelSizeMatch()
  job.setOptions(numROI=8, registerOn="raw")
  job.calcDistortion(fitErr=True, verbose=False)
  assert isinstance(job.T[0], spatialTransformShift) and isinstance(job.T[1], spatialTransformProjective)
  p = vector3d(np.column_stack([rng.uniform(-90, 90, 50), rng.uniform(-70, 70, 50), np.zeros(50)]))
  for T, M in zip(job.T, (A, H)):
    x, y = moved(M, np.asarray(p.x), np.asarray(p.y))
    q = T.eval(p)
    assert np.max(np.hypot(np.asarray(q.x) - x, np.asarray(q.y) - y)) < 0.25
  for ps in job.fitError:
    assert ps.meanShift(1.0)[0] < 0.3
  assert len(job.shifts) == 2 and all(len(s) == 1 for s in job.shifts)


def test_undistort_puts_every_map_on_the_reference_grid():
  job = trueEbsd(chain(), [spatialTransformShift(), spatialTransformProjective()])
  job.pixelSizeMatch()
  job.setOptions(numROI=8, registerOn="raw")
  job.calcDistortion(verbose=False)
  job.undistort()
  ref = job.undistortedList[2]
  assert np.array_equal(ref.img, job.resizedList[2].img)
  for m in job.undistortedList[:2]:
    assert m.gridSize() == ref.gridSize() and np.allclose(m.pos.data, ref.pos.data)
    ok = np.isfinite(m.img)
    assert ok.mean() > 0.9
    # nearest pixel resampling of a picture of about 1.5 per pixel in gradient
    assert np.median(np.abs(m.img[ok] - ref.img[ok])) < 0.6 * np.std(ref.img)
    assert np.corrcoef(m.img[ok], ref.img[ok])[0, 1] > 0.9


def test_the_job_checks_its_sequence():
  L = chain(40, 50)
  with pytest.raises(ValueError):
    trueEbsd(L, [spatialTransformShift()])
  turned = mapImageList([L[0], transformReferenceFrame(L[1], gridLayout.rowMajor()), L[2]])
  with pytest.raises(ValueError):
    trueEbsd(turned)
  L[1].scanUnit = "nm"
  with pytest.raises(ValueError):
    trueEbsd(L)
  job = trueEbsd(chain(40, 50))
  job.setOptions(1, registerOn="RAW", roiSize=[2, 4])
  assert job.opt[1]["registerOn"] == "raw" and job.opt[0]["registerOn"] == "edge" and np.allclose(job.opt[1]["roiSize"], [2, 4])
  for bad in ({"registerOn": "edges"}, {"roiSize": -1}, {"numROI": 0}, {"colour": 1}):
    with pytest.raises(ValueError):
      job.setOptions(**bad)
  with pytest.raises(ValueError):
    job.calcDistortion()
  assert "as imported" in repr(job)


DATA = os.path.join(mtexDataPath(), "EBSD", "trueEbsdWCCoSmall.h5")


# the EBSDTrueEbsd page against the MATLAB bridge (bench/matlab/probe_trueebsd.m 'page'):
# every tile, image and pixel of the undistorted map agreed, here the table of the run
@pytest.mark.skipif(not os.path.isfile(DATA), reason="the trueEbsdWCCoSmall file is written by bench/matlab/export_trueebsd.m")
def test_the_wc_co_sequence_as_matlab_runs_it():
  from mtex import mtexdata, rescale, imboxfilt, spatialTransformDrift
  ebsd = mtexdata("trueEbsdWCCoSmall")
  img = ebsd.meta["trueEbsdImgs"]
  imgList = mapImageList([mapImage(ebsd.bc, ebsd, name="bcImg")] +
                         [mapImage(getattr(img, n), dxy=img.pixSzImg, name=n) for n in ("fsdB3", "fsdT3", "fsdT1", "fsdT10")])
  imgList[1:] = rescale(imboxfilt(imgList[1:], 3))
  T = [spatialTransformShift() + spatialTransformDrift(), spatialTransformId(), spatialTransformShift(), spatialTransformTilt()]
  job = trueEbsd(imgList, T)
  job.pixelSizeMatch()
  job.setOptions(numROI=16)
  job.calcDistortion(fitErr=True, verbose=False)
  job.undistort()
  px = job.resizedList[0].dx
  assert [int(round(float(np.max(job.opt[n]["roiSize"])) / px)) for n in range(4)] == [64, 64, 64, 64]
  for got, want in ((job.shifts[0][0], (3.64, 2.20, 1.84)), (job.shifts[0][1], (2.04, 0.23, 0.49)), (job.fitError[0], (1.53, 0.07, 0.09)),
                    (job.fitError[1], (2.03, -0.14, -1.50)), (job.shifts[2][0], (3.25, -2.69, -1.81)), (job.fitError[2], (0.27, 0.16, 0.15)),
                    (job.shifts[3][0], (3.61, 2.86, -2.19)), (job.fitError[3], (0.26, 0.06, -0.10))):
    assert np.allclose(got.meanShift(px), want, atol=0.005)
  out = job.undistortedList[0].ebsd
  assert out.shape == (192, 256) and np.bincount(out.phaseId.reshape(-1)).tolist() == [1803, 43403, 3840, 106]
  assert out.fsdB3.shape == (192, 256, 3) and np.array_equal(out.fsdT10, job.resizedList[4].img)
  assert isinstance(job.T[3], spatialTransformTilt)
