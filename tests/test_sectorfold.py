# the fold of a sector and the polar coordinates a colour key is built on (docs/plans/plotting.md
# S3): MTEX's own values read through the bridge, as ../mtexJulia/test/sectorfold.jl pins them
import numpy as np
import pytest

from mtex import conventionalSector, crystalFrame, degree, reflection, specimenFrame, vector3d
from mtex.geometry.symmetry import _isin
from mtex.plotting.sectorfold import foldsSector, hasSmoothKey, polarCoordinates, regionCenter, sectorFolds

POLAR = {
  "m-3m": [((0.4282, 0.1925, 0.7615), 1.0, 0.0), ((0.0, 0.0, 1.0), 0.0, 0.0), ((0.5774, 0.5774, 0.5774), 0.0, 240.113),
           ((0.7071, 0.0, 0.7071), 0.0, 119.685), ((0.2349, 0.1056, 0.9663), 0.47144, 0.0),
           ((0.5456, 0.4177, 0.7265), 0.47165, 240.113), ((0.6083, 0.1031, 0.7870), 0.47189, 119.685)],
  "-3m": [((0.5, -0.2887, 0.3333), 1.0, 0.0), ((0.0, 0.0, 1.0), 0.0, 0.0), ((0.5, -0.8660, 0.0), 0.0, 120.125),
          ((1.0, 0.0, 0.0), 0.0, 240.320), ((0.3441, -0.1987, 0.9177), 0.39022, 0.0),
          ((0.6396, -0.7385, 0.2132), 0.39561, 120.125), ((0.9594, -0.1846, 0.2132), 0.39561, 240.320)],
  "622": [((0.5774, 0.0, 0.3333), 1.0, 0.0), ((0.0, 0.0, 1.0), 0.0, 0.0), ((0.8660, -0.5, 0.0), 0.0, 120.125),
          ((0.8660, 0.5, 0.0), 0.0, 240.320)],
  "mmm": [((0.3333, 0.3333, 0.3333), 1.0, 0.0), ((0.0, 0.0, 1.0), 0.0, 0.0), ((0.0, 1.0, 0.0), 0.0, 238.607),
          ((1.0, 0.0, 0.0), 0.0, 121.843)]}

# the fold MTEX's HSVDirectionKey takes; a plane and its opposite normal are the same fold
R3 = np.sqrt(3) / 2
FOLDS = {"2": (-1, 0, 0), "2/m": (1, 0, 0), "222": (1, 0, 0), "4": (1, -1, 0), "-4": (1, 0, 0), "4/m": (1, -1, 0),
         "422": (1, -1, 0), "-42m": (0, 1, 0), "3": (0.5, -R3, 0), "-3": (0.5, -R3, 0), "-3m": (0.5, R3, 0), "6": (0, 1, 0),
         "-6": (0.5, -R3, 0), "6/m": (0, 1, 0), "622": (0, 1, 0), "23": (1, 0, -1), "m-3": (1, -1, 0), "432": (1, -1, 0)}


def unit(x):
  x = np.asarray(x, dtype=float)
  return x / np.linalg.norm(x)


@pytest.mark.parametrize("name", list(POLAR))
def test_polarCoordinates_areMTEXs(name):
  cf = crystalFrame(name)
  sR = conventionalSector(cf)
  c = unit(POLAR[name][0][0])
  pts = np.array([unit(x[0]) for x in POLAR[name]])
  r, rho = polarCoordinates(sR, pts, c, cf.how2plot.east)
  for k, (_, wantR, wantRho) in enumerate(POLAR[name]):
    assert abs(r[k] - wantR) < 1e-3
    d = np.degrees(rho[k]) - wantRho
    assert min(abs(np.mod(d, 360)), abs(np.mod(-d, 360))) < 2


@pytest.mark.parametrize("name", list(POLAR))
def test_polarCoordinates_radiusIsOneAtTheCentreAndZeroAtTheVertices(name):
  cf = crystalFrame(name)
  sR = conventionalSector(cf)
  c = unit(POLAR[name][0][0])
  # the centre is the only place the radius reaches one, and the boundary the only zero
  assert np.isclose(polarCoordinates(sR, c, c, cf.how2plot.east)[0][0], 1)
  for v in sR.vertices().data:
    assert polarCoordinates(sR, v, c, cf.how2plot.east)[0][0] < 1e-8


@pytest.mark.parametrize("name", ["m", "mm2", "mmm", "3m", "4mm", "4/mmm", "6mm", "-6m2", "6/mmm", "-43m", "m-3m"])
def test_sectorFolds_ofGroupsBoundedByTheirOwnMirrorsIsEmpty(name):
  pg = crystalFrame(name).sym
  assert sectorFolds(pg) == [] and hasSmoothKey(pg)


@pytest.mark.parametrize("name", list(FOLDS))
def test_sectorFolds_isTheFoldMTEXTakes(name):
  folds = sectorFolds(crystalFrame(name).sym)
  assert len(folds) == 1 and abs(folds[0] @ unit(FOLDS[name])) > 0.99


@pytest.mark.parametrize("name", list(FOLDS))
def test_sectorFolds_foldIsAMirrorOfTheSectorButNotOfTheGroup(name):
  pg = crystalFrame(name).sym
  folds = sectorFolds(pg)
  sR = conventionalSector(pg)
  assert foldsSector(folds[0], sR) and not _isin(reflection(vector3d(folds[0])), pg.rot.flatten())[0]


def test_hasSmoothKey_isFalseWhereTheKeyJumps():
  assert not hasSmoothKey(crystalFrame("-3").sym) and not hasSmoothKey(crystalFrame("-4").sym)
  assert hasSmoothKey(crystalFrame("432").sym)


def test_sectorFolds_ofTheWholeSphereIsTheEquatorOfTheConvention():
  # the whole sphere folds at the equator of the frame's plotting convention
  ss = specimenFrame("1")
  folds = sectorFolds(ss.sym, conventionalSector(ss))
  assert len(folds) == 1 and abs(folds[0] @ ss.how2plot.outOfScreen.data) > 1 - 1e-9


def test_regionCenter_ofATriangleIsItsBarycentre():
  c = regionCenter(conventionalSector(crystalFrame("m-3m")))
  assert np.allclose(c, unit(np.array([0, 0, 1.0]) + unit([1, 0, 1]) + unit([1, 1, 1])))
  sR = conventionalSector(crystalFrame("4/mmm"))
  assert np.allclose(regionCenter(sR), unit(np.sum(sR.vertices().data, axis=0)))


def test_regionCenter_ofALuneIsItsBisector():
  lune = conventionalSector(crystalFrame("2/m"))
  assert lune.checkInside(regionCenter(lune)) and np.allclose(regionCenter(lune), unit(np.sum(lune.N, axis=0)))


@pytest.mark.parametrize("name", ["m-3m", "-3m", "mmm", "2/m"])
def test_polarCoordinates_compiledRadiusIsTheNumpyOne(name):
  from mtex import setMTEXpref
  sR = conventionalSector(crystalFrame(name))
  c = regionCenter(sR).reshape(3)
  x = np.random.default_rng(1).standard_normal((5000, 3))
  x = np.vstack([x / np.linalg.norm(x, axis=1, keepdims=True), c])
  out = {}
  for backend in ("numba", "numpy"):
    setMTEXpref("kernels", backend)
    try:
      out[backend] = polarCoordinates(sR, x, c)[0]
    finally:
      setMTEXpref("kernels", "numba")
  assert np.allclose(out["numba"], out["numpy"], atol=1e-12)


def test_polarCoordinates_putTheVerticesOfATriangleOnThirdsOfTheTurn():
  # the vertices are knots of the equalisation, so they land on 0, 1 and 2 thirds exactly
  from mtex.plotting.directionkeys import HSVDirectionKey
  for pg in ("m-3m", "6/mmm", "mmm", "-3m", "-3"):
    sR = HSVDirectionKey(crystalFrame(pg)).sR
    _, rho = polarCoordinates(sR, sR.vertices(), regionCenter(sR))
    t = np.mod(np.asarray(rho).ravel() / (2 * np.pi / 3), 3)
    assert np.max(np.min(np.abs(t[:, None] - np.arange(4)), axis=1)) < 1e-12
