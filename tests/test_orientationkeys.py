# the orientation colour keys and the key on a grid (docs/plans/plotting.md S3): the inverse
# pole figure key against MTEX's ipfHSVKey, as ../mtexJulia/test/orientationkeys.jl pins it
import os

import numpy as np
import pytest

from mtex import (BungeColorKey, HSVDirectionKey, SO3DeLaValleePoussinKernel, axisAngleColorKey, crystalFrame, degree,
                  ipfColorKey, ipfColors, ipfHKLKey, ipfHSVKey, ipfSpotKey, ipfTSLKey, loadEBSD, orientation, rotation,
                  setMTEXpref, specimenFrame, spotColorKey, vector3d, xvector, zvector, Miller)
from mtex.plotting.orientationkeys import COLOR_NAN, colorGrid, keyColorGrid

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd", "Forsterite_cut.ctf")

ANGLES = [(0, 0, 0), (30, 40, 50), (10, 20, 30), (120, 60, 240), (200, 100, 350), (45, 45, 45)]
REF = {
  "m-3m": [(1.0, 0.0103, 0.0), (0.6553, 0.3821, 0.9627), (0.9433, 0.5593, 0.7073), (0.6001, 0.3281, 0.9684),
           (0.9987, 0.0139, 0.5280), (0.4282, 0.0, 1.0)],
  "622": [(1.0, 0.0103, 0.0), (0.9448, 0.7219, 0.5459), (1.0, 0.0, 0.3943), (0.9037, 1.0, 0.0), (0.1133, 0.5362, 0.3881),
          (0.9189, 0.7623, 0.7607)],
  "-3m": [(1.0, 0.0103, 0.0), (0.4626, 0.2811, 0.1379), (1.0, 0.0, 0.3943), (0.9037, 1.0, 0.0), (0.1133, 0.5362, 0.3881),
          (0.3150, 0.2039, 0.2028)]}


@pytest.mark.parametrize("name", list(REF))
def test_ipfHSVKey_coloursAreMTEXs(name):
  cs, ss = crystalFrame(name), specimenFrame("1")
  key = ipfHSVKey(cs)
  a = np.array(ANGLES, dtype=float) * degree
  os_ = orientation.byEuler(a[:, 0], a[:, 1], a[:, 2], cs, ss)
  got = key.orientation2color(os_)
  assert got.shape == (6, 3)
  for k in range(6):
    assert np.max(np.abs(got[k] - REF[name][k])) < 5 / 255


# a cubic key and an orientation of it
def _cubicKey():
  cs, ss = crystalFrame("m-3m"), specimenFrame("1")
  return cs, ipfHSVKey(cs), orientation.byEuler(0.3, 0.7, 1.1, cs, ss)


def test_ipfHSVKey_isAnIpfColorKeyOfTheFrameOverADirectionKey():
  cs, key, _ = _cubicKey()
  assert key.frameA is cs and key.CS is cs and isinstance(key, ipfColorKey) and isinstance(key.dirMap, HSVDirectionKey)


def test_ipfHSVKey_equivalentOrientationsShareAColour():
  _, key, o = _cubicKey()
  # a symmetrically equivalent orientation is the same orientation and takes the same colour
  assert np.max(np.abs(key.orientation2color(o.symmetrise()) - key.orientation2color(o))) < 1e-9


def test_ipfHSVKey_ipfDirectionIsAnySpecimenDirection():
  cs, key, o = _cubicKey()
  # the direction the inverse pole figure is of can be any specimen direction
  kx = ipfHSVKey(cs, ipfDirection=xvector)
  assert not np.allclose(kx.orientation2color(o), key.orientation2color(o))
  d = kx.crystalDirection(o)
  assert d.frame is cs and np.allclose(d.data, (rotation(o).inv() * xvector).data)


def test_ipfColorKey_ofOrientationsTakesAnIpfDirectionSetAfterwards():
  cs, _, o = _cubicKey()
  kx = ipfHSVKey(cs, ipfDirection=xvector)
  # the key of a list of orientations, of an ipf direction set afterwards
  k2 = ipfColorKey(o)
  k2.ipfDirection = xvector
  assert np.allclose(k2.orientation2color(o), kx.orientation2color(o))


def test_ipfTSLKey_and_ipfHKLKey_colourAnOrientation():
  cs, _, o = _cubicKey()
  # the vendors' keys seen through an inverse pole figure
  assert ipfTSLKey(cs).orientation2color(o).shape == (3,) and ipfHKLKey(cs).orientation2color(o).shape == (3,)


def test_ipfSpotKey_isItsColourAtTheCentre():
  cs, ss = crystalFrame("m-3m"), specimenFrame("1")
  spot = ipfSpotKey(cs, center=Miller(0, 0, 1, cs), color=(0, 0, 1))
  assert np.allclose(spot.orientation2color(orientation(rotation.id, cs, ss)), (0, 0, 1))


def test_ipfHSVKey_reprNamesTheKey():
  _, key, _ = _cubicKey()
  assert "ipfHSVKey" in repr(key)


# twenty orthorhombic orientations, the first the reference
def _orthorhombic():
  cs = crystalFrame("mmm")
  os_ = orientation(rotation.rand(20, rng=np.random.default_rng(3)), cs)
  return cs, os_, os_[0]


def test_axisAngleColorKey_coloursTheReferenceGrey():
  cs, os_, ref = _orthorhombic()
  # the axis angle key: the reference itself has no rotation and is grey
  k = axisAngleColorKey(cs, oriRef=ref, maxAngle=10 * degree)
  c = k.orientation2color(os_)
  assert c.shape == (20, 3) and np.all(np.abs(c[0] - c[0, 0]) < 1e-6)


def test_axisAngleColorKey_takesTheReferenceAsArgument():
  cs, os_, ref = _orthorhombic()
  c = axisAngleColorKey(cs, oriRef=ref, maxAngle=10 * degree).orientation2color(os_)
  assert np.allclose(axisAngleColorKey(cs, maxAngle=10 * degree).orientation2color(os_, ref), c)


def test_axisAngleColorKey_aboveTheThresholdHasNoColour():
  cs, os_, ref = _orthorhombic()
  # above the threshold the colour is missing
  k2 = axisAngleColorKey(cs, oriRef=ref, thresholdAngle=1 * degree)
  c2 = k2.orientation2color(os_)
  assert np.all(np.isfinite(c2[0])) and np.all(np.isnan(c2[1]))


def test_axisAngleColorKey_withoutAReferenceTakesTheIdentity():
  cs, os_, _ = _orthorhombic()
  # without a reference the identity is the reference
  assert np.all(np.isfinite(axisAngleColorKey(cs).orientation2color(os_)))


def test_axisAngleColorKey_refusesReferencesNotMatchingTheOrientations():
  cs, os_, ref = _orthorhombic()
  k = axisAngleColorKey(cs, oriRef=ref, maxAngle=10 * degree)
  with pytest.raises(ValueError):
    k.orientation2color(os_, os_[:3])


def test_BungeColorKey_coloursTheIdentityBlackAndStaysInTheBox():
  cs, os_, _ = _orthorhombic()
  # the Bunge key: identity is black, every colour inside the box
  b = BungeColorKey(cs)
  assert np.allclose(b.orientation2color(orientation(rotation.id, cs)), 0)
  cb = b.orientation2color(os_)
  assert cb.shape == (20, 3) and np.all((cb >= 0) & (cb <= 1))


def test_BungeColorKey_rangesAreTheOrthorhombicEulerBox():
  b = BungeColorKey(crystalFrame("mmm"))
  assert np.isclose(b.phi1Range, 2 * np.pi) and np.isclose(b.PhiRange, np.pi / 2) and np.isclose(b.phi2Range, np.pi)


def test_spotColorKey_isItsColourAtTheCentre():
  _, os_, ref = _orthorhombic()
  # the spot key: its colour at the centre, white far away
  sp = spotColorKey(ref, (0.0, 0.0, 1.0), psi=SO3DeLaValleePoussinKernel(halfwidth=10 * degree))
  assert sp.orientation2color(ref)[2] > 0.99 and sp.orientation2color(ref)[0] < 0.01
  assert sp.orientation2color(os_).shape == (20, 3)


def test_ipfHSVKey_sharpKeyKeepsItsMaxAngle():
  cs, os_, _ = _orthorhombic()
  # the sharp inverse pole figure key centred on the mean
  ipf = ipfHSVKey(cs, maxAngle=7.5 * degree, ipfDirection=os_.mean() * HSVDirectionKey(cs).whiteCenter)
  assert ipf.orientation2color(os_).shape == (20, 3) and np.isclose(ipf.maxAngle, 7.5 * degree)


# an olivine-like refractive index and eight orientations of it
def _refraction():
  from mtex import refractiveIndexTensor
  cs = crystalFrame("mmm")
  rI = refractiveIndexTensor(np.diag([1.670, 1.635, 1.651]), cs)
  return cs, rI, orientation(rotation.rand(8, rng=np.random.default_rng(3)), cs)


def test_spectralTransmissionColorKey_isOfTheFrameAtNinetyDegrees():
  from mtex import spectralTransmissionColorKey
  cs, rI, _ = _refraction()
  key = spectralTransmissionColorKey(rI, 22800)
  assert key.frameA is cs and np.isclose(key.phi, 90 * degree)


def test_spectralTransmissionColorKey_isTheTransmissionAlongZPolarisedAlongX():
  from mtex import spectralTransmission, spectralTransmissionColorKey
  _, rI, os_ = _refraction()
  key = spectralTransmissionColorKey(rI, 22800)
  # the default setup: light along z, the polariser along x of the specimen
  direct = spectralTransmission(rI, os_.inv() * zvector, 22800, polarizationDirection=os_.inv() * xvector)
  assert np.allclose(key.orientation2color(os_), direct)


def test_spectralTransmissionColorKey_withoutPolariserIsTheCircularBranch():
  from mtex import spectralTransmission, spectralTransmissionColorKey
  _, rI, os_ = _refraction()
  key = spectralTransmissionColorKey(rI, 22800)
  # no polariser is the circular branch, the fast wave at 45 degrees
  key.polarizer = None
  assert np.allclose(key.orientation2color(os_), spectralTransmission(rI, os_.inv() * zvector, 22800))


# the cubic key, its colour grid and three thousand directions coloured by both
@pytest.fixture(scope="module")
def grid432():
  cs = crystalFrame("432")
  key = HSVDirectionKey(cs)
  cg = colorGrid(key)
  vs = vector3d.rand(3000, rng=np.random.default_rng(4))
  return cs, key, cg, vs, key.direction2color(vs)


def test_colorGrid_isTheKeyToItsResolution(grid432):
  _, _, cg, vs, exact = grid432
  # the grid is the key to the accuracy its resolution allows, and no better
  assert np.max(np.abs(exact - cg.direction2color(vs))) < 20 / 255


def test_keyColorGrid_isOneCachedGridPerFrame(grid432):
  cs, _, cg, _, _ = grid432
  assert len(cg) == len(keyColorGrid(cs)) and keyColorGrid(cs) is keyColorGrid(cs)


def test_colorGrid_nodeTakesExactlyTheKeysColour(grid432):
  _, key, cg, _, _ = grid432
  # a node of the grid takes exactly the colour of the key
  v = vector3d(cg.grid.data[17])
  assert np.allclose(cg.direction2color(v), key.direction2color(v), atol=1e-12)


def test_colorGrid_colourNanPoleWhite(grid432):
  _, _, cg, _, _ = grid432
  # a NaN or zero pole is coloured white
  assert np.allclose(cg.direction2color(vector3d(np.nan, np.nan, np.nan)), COLOR_NAN)
  assert np.allclose(cg.direction2color(vector3d([[1.0, 0, 0], [np.nan] * 3]))[1], COLOR_NAN)


def test_colorGrid_finerIsNearerTheKey(grid432):
  _, key, cg, vs, exact = grid432
  # a finer grid is nearer the key
  err = lambda g: np.max(np.abs(exact - g.direction2color(vs)))
  assert err(cg) < err(colorGrid(key, resolution=1 * degree)) < err(colorGrid(key, resolution=2 * degree))


# the Forsterite map and its colours on the grid of the key
@pytest.fixture(scope="module")
def forsteriteColours():
  ebsd = loadEBSD(FIXTURE)
  return ebsd, ipfColors(ebsd)


def test_ipfColors_givesAColourPerMeasurement(forsteriteColours):
  ebsd, c = forsteriteColours
  assert c.shape == ebsd.rotations.shape + (3,) and np.all((c >= -1e-9) & (c <= 1 + 1e-9))


def test_ipfColors_notIndexedIsWhite(forsteriteColours):
  ebsd, c = forsteriteColours
  # what carries no orientation carries no colour of the key
  notIndexed = ~ebsd.isIndexed
  assert np.all(c[notIndexed] == 1)


def test_ipfColors_isThePhasesKeyExactlyOrWithinTheGridResolution(forsteriteColours):
  ebsd, c = forsteriteColours
  # a measurement takes the colour its own phase's key gives its orientation: exactly when
  # the map is coloured exactly, and within the grid's resolution by default
  exact = ipfColors(ebsd, exact=True)
  for i in ebsd.CSList.indexedPhaseId:
    at = np.flatnonzero(np.asarray(ebsd.phaseId).reshape(-1) == i)
    if len(at) == 0:
      continue
    at = at[0]
    key = ipfHSVKey(ebsd.CSList[i])
    o = orientation._fromRotation(ebsd.rotations.data.reshape(-1, 4)[at], (ebsd.CSList[i], ebsd.frame))
    assert np.allclose(exact.reshape(-1, 3)[at], key.orientation2color(o))
    assert np.max(np.abs(c.reshape(-1, 3)[at] - exact.reshape(-1, 3)[at])) < 20 / 255
  assert np.max(np.abs(c - exact)) < 20 / 255


def test_ipfColors_takesTheIpfDirection(forsteriteColours):
  ebsd, c = forsteriteColours
  # the direction the key is of reaches through
  assert not np.allclose(ipfColors(ebsd, ipfDirection=xvector), c)


def test_an_orientation_never_measured_has_no_colour():
  cs = crystalFrame("m-3m")
  q = np.array([[1.0, 0, 0, 0], [np.nan] * 4])
  rgb = ipfColorKey(cs).orientation2color(orientation(q, cs))
  assert np.all(np.isfinite(rgb[0])) and np.isnan(rgb[1]).all()


@pytest.mark.parametrize("name", ["m-3m", "mmm", "-3m", "2/m"])
def test_precompute_colourIsTheNearestNodesWithinTheGridResolution(name):
  cs = crystalFrame(name)
  ori = orientation(rotation.rand(20000, rng=np.random.default_rng(3)), cs)
  exact = ipfHSVKey(cs).orientation2color(ori)
  grid = ipfHSVKey(cs).precompute().orientation2color(ori)
  # the colours of the node nearest each direction, half a degree apart
  assert np.mean(np.abs(grid - exact).max(axis=1)) < 0.02 and np.max(np.abs(grid - exact)) < 0.1


@pytest.mark.parametrize("name", ["m-3m", "mmm", "-3m", "2/m"])
def test_precompute_numpyLookupIsTheCompiledOne(name):
  cs = crystalFrame(name)
  ori = orientation(rotation.rand(20000, rng=np.random.default_rng(3)), cs)
  key = ipfHSVKey(cs).precompute()
  grid = key.orientation2color(ori)
  setMTEXpref("kernels", "numpy")
  try:
    assert np.allclose(key.orientation2color(ori), grid)
  finally:
    setMTEXpref("kernels", "numba")


def test_precompute_followsAChangedMaxAngle():
  cs = crystalFrame("m-3m")
  ori = orientation(rotation.rand(2000, rng=np.random.default_rng(4)), cs)
  key = ipfHSVKey(cs).precompute(2 * degree)
  shared = key._grid[1]
  key.maxAngle = 10 * degree
  sharp = ipfHSVKey(cs, maxAngle=10 * degree).orientation2color(ori)
  assert key._grid[1] is shared and np.mean(np.abs(key.orientation2color(ori) - sharp).max(axis=1)) < 0.05
  assert key._grid[1] is not shared


def test_precompute_leavesAnUnmeasuredOrientationBlank():
  cs = crystalFrame("m-3m")
  ori = orientation(rotation.rand(2000, rng=np.random.default_rng(4)), cs)
  key = ipfHSVKey(cs).precompute(2 * degree)
  key.maxAngle = 10 * degree
  d = ori.data.copy()
  d[5] = np.nan
  rgb = key.orientation2color(orientation(rotation(d), cs))
  assert np.all(np.isnan(rgb[5])) and np.all(np.isfinite(rgb[:5]))
