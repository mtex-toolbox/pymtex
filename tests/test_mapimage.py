# images on the specimen, MATLAB EBSDAnalysis/@mapImage
import numpy as np
import pytest

from mtex import (EBSDsquare, crystalFrame, edgeMap, gridLayout, imboxfilt, mapImage, mapImageList, orientation, plottingConvention,
                  rescale, rotation, specimenFrame, transformReferenceFrame, vector3d)

rng = np.random.default_rng(3)


def test_mapImage_gridIsDerivedFromOriginAndSteps():
  mg = mapImage(rng.random((4, 6)), dxy=(0.5, 2.0), origin=vector3d(1, 2, 0))
  P = mg.pos.data
  assert P.shape == (4, 6, 3)
  assert np.allclose(P[0, 0], [1, 2, 0]) and np.allclose(P[0, 1], [1.5, 2, 0]) and np.allclose(P[1, 0], [1, 4, 0])
  assert mg.dx == 0.5 and mg.dy == 2.0 and mg.nChannel == 1
  assert mg.layout == gridLayout.columnMajor()
  assert np.allclose(mg.extent(), [1, 3.5, 2, 8, 0, 0])


# an image of its own has a frame of its own, with the image's axes, drawn as SEM images are
def test_mapImage_ofItsOwnHasAFrameOfItsOwn():
  mg = mapImage(rng.random((4, 6)), dxy=(0.5, 2.0), origin=vector3d(1, 2, 0))
  assert mg.frame.axesNames == ("iX", "iY", "iZ") and mg.how2plot == plottingConvention.ij
  assert mapImage(np.zeros((2, 2))).frame is not mapImage(np.zeros((2, 2))).frame


def test_mapImage_integerImagesAreScaledToOne():
  mg = mapImage(np.array([[0, 255], [128, 64]], dtype=np.uint8))
  assert mg.img.dtype == float and mg.img.max() == 1.0


def test_mapImage_refusesANameThatIsNoIdentifier():
  with pytest.raises(ValueError):
    mapImage(np.zeros((2, 2)), name="not a name")


# a three channel image of 5 x 7 pixels of 0.25 from (-1, 3)
def colourImage():
  img = rng.random((5, 7, 3))
  return img, mapImage(img, dxy=0.25, origin=vector3d(-1, 3, 0))


def test_interp_isExactAtThePixelCentres():
  img, mg = colourImage()
  v = mg.interp(mg.pos)
  assert v.shape == (5, 7, 3) and np.allclose(v, img)


def test_interp_isBilinearBetweenThePixelCentres():
  img, mg = colourImage()
  q = vector3d(-1 + 0.25 * 2.5, 3 + 0.25 * 1.25, 0)
  a = mg.interp(q)
  i, t = 1, 0.25
  expect = (1 - t) * 0.5 * (img[i, 2] + img[i, 3]) + t * 0.5 * (img[i + 1, 2] + img[i + 1, 3])
  assert np.allclose(a, expect)
  assert np.allclose(mg.interp(q, method="nearest"), img[1, 3])


def test_interp_outsideIsNanUnlessExtrapolated():
  img, mg = colourImage()
  out = vector3d(-2, 3, 0)
  assert np.isnan(mg.interp(out)).all()
  assert np.allclose(mg.interp(out, extrapolationMethod="nearest"), img[0, 0])


def test_pos2ind_roundTrips():
  mg = mapImage(rng.random((6, 9)), dxy=0.3, origin=vector3d(2, 1, 0))
  i, j, inside = mg.pos2ind(mg.pos)
  assert inside.all() and np.array_equal(i, np.arange(6)[:, None] + 0 * j) and np.array_equal(j, np.arange(9)[None, :] + 0 * i)
  lin = mg.pos2ind(vector3d([[2, 1, 0], [2 + 0.3 * 8, 1 + 0.3 * 5, 0], [0, 0, 0]]), linear=True)
  assert list(lin) == [0, 53, -1]


def ebsdMap(ny=8, nx=12):
  cs = crystalFrame("m-3m", mineral="Iron")
  rot = rotation.rand(ny, nx)
  return EBSDsquare(rot, np.ones((ny, nx)), ["notIndexed", cs], dxy=(0.5, 0.5), prop={"bc": rng.random((ny, nx))})


def test_mapImage_channelOfAMapTakesItsGeometry():
  e = ebsdMap()
  mg = mapImage(e.bc, e, name="bcImg")
  assert mg.ebsd is e and mg.frame is e.frame and mg.scanUnit == e.scanUnit
  assert np.allclose(mg.pos.data, e.pos.data)
  assert mg.layout == e.layout and mg.dx == 0.5
  assert np.allclose(mg.interp(e.pos), e.bc)


def test_mapImage_refusesAChannelOfAnotherShapeThanTheMap():
  e = ebsdMap()
  with pytest.raises(ValueError):
    mapImage(e.bc[:, :5], e)


def test_subGrid_cropsTheMapAndMovesTheOrigin():
  e = ebsdMap()
  mg = mapImage(e.bc, e)
  s = mg.subGrid(np.arange(2, 5), np.arange(3, 9))
  assert s.gridSize() == (3, 6) and s.ebsd.shape == (3, 6)
  assert np.allclose(s.pos.data, e.pos.data[2:5, 3:9])
  assert np.allclose(s.img, e.bc[2:5, 3:9])


# a mask crops to the box around it
def test_subGrid_byAMaskCropsToItsBox():
  e = ebsdMap()
  mg = mapImage(e.bc, e)
  mask = np.zeros(e.shape, dtype=bool)
  mask[1, 2] = mask[3, 6] = True
  assert mg.subGrid(mask).gridSize() == (3, 5)


def test_subGrid_refusesIndicesThatAreNoRange():
  e = ebsdMap()
  mg = mapImage(e.bc, e)
  with pytest.raises(ValueError):
    mg.subGrid([0, 2], [0, 1])


# an image with a missing value
def holedImage():
  img = rng.random((6, 8))
  img[2, 3] = np.nan
  return img


def test_rescale_mapsEveryImageOntoZeroToOne():
  img = holedImage()
  L = mapImageList([mapImage(img), mapImage(3 * img + 1)])
  R = rescale(L)
  assert isinstance(R, mapImageList)
  assert np.nanmin(R[1].img) == 0 and np.nanmax(R[1].img) == 1 and np.isnan(R[1].img[2, 3])
  assert np.allclose(R[0].img, R[1].img, equal_nan=True)


def test_imboxfilt_isMatlabsMovingMean():
  img = holedImage()
  f = imboxfilt(mapImage(img), 3).img
  # MATLAB's movmean down the columns, then along the rows, the window shrinking at the border
  # and around the missing value
  col = np.array([[np.nanmean(img[max(i - 1, 0):i + 2, j]) for j in range(8)] for i in range(6)])
  both = np.array([[np.nanmean(col[i, max(j - 1, 0):j + 2]) for j in range(8)] for i in range(6)])
  assert np.allclose(f, both) and np.isfinite(f).all()


def test_imboxfilt_refusesAnEvenWindow():
  with pytest.raises(ValueError):
    imboxfilt(mapImage(holedImage()), 4)


def test_edgeMap_lightsUpTheBoundariesAndTheFrame():
  img = np.zeros((6, 7))
  img[:, 4:] = 1.0
  e = edgeMap(mapImage(img))
  # the boundary between the columns 3 and 4 lights up, and the frame of the image, which is
  # differenced against zero
  assert e[3, 1] == 0 and e[3, 3] == 3 and e[3, 4] == 3 and e[3, 5] == 0
  assert e[0, 5] == 3 and e[0, 1] == 0
  e2 = edgeMap(mapImage(np.stack([img, img, img], axis=2)))
  assert np.allclose(e2, np.sqrt(3) * e)
  assert edgeMap(mapImage(img), 2)[3, 2] == 3


def test_transformReferenceFrame_layoutReindexesAndThePositionsFollow():
  img = rng.random((4, 6))
  mg = mapImage(img, dxy=0.5, origin=vector3d(1, 1, 0))
  gL = gridLayout(-vector3d.X, vector3d.Y)
  t = transformReferenceFrame(mg, gL)
  assert t.gridSize() == (6, 4) and t.layout == gL and t.frame is mg.frame
  # every pixel keeps its value where it was
  assert np.allclose(t.interp(mg.pos), img)
  assert np.allclose(mg.interp(t.pos), t.img)
  back = transformReferenceFrame(t, mg)
  assert np.array_equal(back.img, img) and np.allclose(back.pos.data, mg.pos.data)


SCREEN_FRAME = specimenFrame("map", how2plot="x↑→y")


def fsdList():
  return mapImageList([mapImage(rng.random((10, 16)), dxy=0.25, name="fsd")])


def test_transformReferenceFrame_byScreenAlignmentStatesTheImagesInTheFrameOfTheMap():
  L = fsdList()
  t = transformReferenceFrame(L, SCREEN_FRAME, "byScreenAlignment")
  assert t[0].frame is SCREEN_FRAME
  # the picture is the same way up on screen, x up and y to the right, so the array is unmoved
  assert np.array_equal(t[0].img, L[0].img)
  assert t[0].layout == gridLayout(-vector3d.X, vector3d.Y)


def test_transformReferenceFrame_byScreenAlignmentNeedsAFrame():
  with pytest.raises(ValueError):
    transformReferenceFrame(fsdList(), gridLayout.rowMajor(), "byScreenAlignment")


def test_transformReferenceFrame_byTheScreenAlignmentOrientationIsByScreenAlignment():
  L = fsdList()
  t = transformReferenceFrame(L, SCREEN_FRAME, "byScreenAlignment")
  o = orientation.byScreenAlignment(L[0].frame, SCREEN_FRAME)
  assert np.array_equal(transformReferenceFrame(L, SCREEN_FRAME, o)[0].img, t[0].img)


def test_gridLayout_assumedForIsTheLayoutAFrameIsDrawnIn():
  gL = gridLayout.assumedFor(specimenFrame("a", how2plot=plottingConvention.ij))
  assert gL == gridLayout.columnMajor() and gL.name == "assumed from plot"
  assert gridLayout.assumedFor(specimenFrame("b", how2plot="x↑→y")) == gridLayout(-vector3d.X, vector3d.Y)


def imageSequence():
  e = ebsdMap()
  return mapImageList([mapImage(e.bc, e, name="bcImg"), mapImage(rng.random((3, 5, 3)), dxy=0.1, name="bse")])


def test_mapImageList_reprIsTheTableOfTheSequence():
  s = repr(imageSequence())
  assert "bcImg" in s and "3 × 5 × 3" in s and "0.1 µm" in s and "Iron" in s and "iY" in s


def test_mapImageList_sliceIsAList():
  assert isinstance(imageSequence()[:1], mapImageList)
