# the orientation colour keys and the key on a grid (docs/plans/plotting.md S3): the inverse
# pole figure key against MTEX's ipfHSVKey, as ../mtexJulia/test/orientationkeys.jl pins it
import os

import numpy as np
import pytest

from mtex import (BungeColorKey, HSVDirectionKey, SO3DeLaValleePoussinKernel, axisAngleColorKey, crystalFrame, degree,
                  ipfColorKey, ipfColors, ipfHKLKey, ipfHSVKey, ipfSpotKey, ipfTSLKey, loadEBSD, orientation, rotation,
                  specimenFrame, spotColorKey, vector3d, xvector, zvector, Miller)
from mtex.plotting.orientationkeys import COLOR_NAN, colorGrid, keyColorGrid

rng = np.random.default_rng(3)
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
def test_P7_the_inverse_pole_figure_key(name):
  cs, ss = crystalFrame(name), specimenFrame("1")
  key = ipfHSVKey(cs)
  a = np.array(ANGLES, dtype=float) * degree
  os_ = orientation.byEuler(a[:, 0], a[:, 1], a[:, 2], cs, ss)
  got = key.orientation2color(os_)
  assert got.shape == (6, 3)
  for k in range(6):
    assert np.max(np.abs(got[k] - REF[name][k])) < 5 / 255


def test_P7_the_key_is_of_the_orientation():
  cs, ss = crystalFrame("m-3m"), specimenFrame("1")
  key = ipfHSVKey(cs)
  assert key.CS1 is cs and key.CS is cs and isinstance(key, ipfColorKey) and isinstance(key.dirMap, HSVDirectionKey)
  o = orientation.byEuler(0.3, 0.7, 1.1, cs, ss)
  # a symmetrically equivalent orientation is the same orientation and takes the same colour
  assert np.max(np.abs(key.orientation2color(o.symmetrise()) - key.orientation2color(o))) < 1e-9
  # the direction the inverse pole figure is of can be any specimen direction
  kx = ipfHSVKey(cs, ipfDirection=xvector)
  assert not np.allclose(kx.orientation2color(o), key.orientation2color(o))
  d = kx.crystalDirection(o)
  assert d.frame is cs and np.allclose(d.data, (rotation(o).inv() * xvector).data)
  # the key of a list of orientations, of an ipf direction set afterwards
  k2 = ipfColorKey(o)
  k2.ipfDirection = xvector
  assert np.allclose(k2.orientation2color(o), kx.orientation2color(o))
  # the vendors' keys and the spot key seen through an inverse pole figure
  assert ipfTSLKey(cs).orientation2color(o).shape == (3,) and ipfHKLKey(cs).orientation2color(o).shape == (3,)
  spot = ipfSpotKey(cs, center=Miller(0, 0, 1, cs), color=(0, 0, 1))
  assert np.allclose(spot.orientation2color(orientation(rotation.id, cs, ss)), (0, 0, 1))
  assert "ipfHSVKey" in repr(key)


def test_T2_orientation_keys():
  cs = crystalFrame("mmm")
  os_ = orientation(rotation.rand(20, rng=rng), cs)
  ref = os_[0]
  # the axis angle key: the reference itself has no rotation and is grey
  k = axisAngleColorKey(cs, oriRef=ref, maxAngle=10 * degree)
  c = k.orientation2color(os_)
  assert c.shape == (20, 3) and np.all(np.abs(c[0] - c[0, 0]) < 1e-6)
  assert np.allclose(axisAngleColorKey(cs, maxAngle=10 * degree).orientation2color(os_, ref), c)
  # above the threshold the colour is missing
  k2 = axisAngleColorKey(cs, oriRef=ref, thresholdAngle=1 * degree)
  c2 = k2.orientation2color(os_)
  assert np.all(np.isfinite(c2[0])) and np.all(np.isnan(c2[1]))
  # without a reference the identity is the reference
  assert np.all(np.isfinite(axisAngleColorKey(cs).orientation2color(os_)))
  with pytest.raises(ValueError):
    k.orientation2color(os_, os_[:3])
  # the Bunge key: identity is black, every colour inside the box
  b = BungeColorKey(cs)
  assert np.allclose(b.orientation2color(orientation(rotation.id, cs)), 0)
  cb = b.orientation2color(os_)
  assert cb.shape == (20, 3) and np.all((cb >= 0) & (cb <= 1))
  assert np.isclose(b.phi1Range, 2 * np.pi) and np.isclose(b.PhiRange, np.pi / 2) and np.isclose(b.phi2Range, np.pi)
  # the spot key: its colour at the centre, white far away
  sp = spotColorKey(ref, (0.0, 0.0, 1.0), psi=SO3DeLaValleePoussinKernel(halfwidth=10 * degree))
  assert sp.orientation2color(ref)[2] > 0.99 and sp.orientation2color(ref)[0] < 0.01
  assert sp.orientation2color(os_).shape == (20, 3)
  # the sharp inverse pole figure key centred on the mean
  ipf = ipfHSVKey(cs, maxAngle=7.5 * degree, ipfDirection=os_.mean() * HSVDirectionKey(cs).whiteCenter)
  assert ipf.orientation2color(os_).shape == (20, 3) and np.isclose(ipf.maxAngle, 7.5 * degree)


def test_the_spectral_transmission_key():
  from mtex import refractiveIndexTensor, spectralTransmission, spectralTransmissionColorKey, xvector, zvector
  cs = crystalFrame("mmm")
  rI = refractiveIndexTensor(np.diag([1.670, 1.635, 1.651]), cs)
  os_ = orientation(rotation.rand(8, rng=rng), cs)
  key = spectralTransmissionColorKey(rI, 22800)
  assert key.CS1 is cs and np.isclose(key.phi, 90 * degree)
  # the default setup: light along z, the polariser along x of the specimen
  direct = spectralTransmission(rI, os_.inv() * zvector, 22800, polarizationDirection=os_.inv() * xvector)
  assert np.allclose(key.orientation2color(os_), direct)
  # no polariser is the circular branch, the fast wave at 45 degrees
  key.polarizer = None
  assert np.allclose(key.orientation2color(os_), spectralTransmission(rI, os_.inv() * zvector, 22800))


def test_P9_the_colour_key_on_a_grid():
  cs = crystalFrame("432")
  key = HSVDirectionKey(cs)
  cg = colorGrid(key)
  vs = vector3d.rand(3000, rng=np.random.default_rng(4))
  exact, onGrid = key.direction2color(vs), cg.direction2color(vs)
  # the grid is the key to the accuracy its resolution allows, and no better
  assert np.max(np.abs(exact - onGrid)) < 20 / 255
  assert len(cg) == len(keyColorGrid(cs)) and keyColorGrid(cs) is keyColorGrid(cs)
  # a node of the grid takes exactly the colour of the key
  v = vector3d(cg.grid.data[17])
  assert np.allclose(cg.direction2color(v), key.direction2color(v), atol=1e-12)
  # a NaN or zero pole is coloured white
  assert np.allclose(cg.direction2color(vector3d(np.nan, np.nan, np.nan)), COLOR_NAN)
  assert np.allclose(cg.direction2color(vector3d([[1.0, 0, 0], [np.nan] * 3]))[1], COLOR_NAN)
  # a finer grid is nearer the key
  err = lambda g: np.max(np.abs(exact - g.direction2color(vs)))
  assert err(colorGrid(key, resolution=0.25 * degree)) < err(cg) < err(colorGrid(key, resolution=2 * degree))


def test_P8_a_map_by_its_keys():
  ebsd = loadEBSD(FIXTURE)
  c = ipfColors(ebsd)
  assert c.shape == ebsd.rotations.shape + (3,) and np.all((c >= -1e-9) & (c <= 1 + 1e-9))
  # what carries no orientation carries no colour of the key
  notIndexed = ~ebsd.isIndexed
  assert np.all(c[notIndexed] == 1)
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
  # the direction the key is of reaches through
  assert not np.allclose(ipfColors(ebsd, ipfDirection=xvector), c)


def test_the_compiled_grid_route_agrees_with_the_numpy_one():
  import numpy as np
  from mtex import crystalFrame, orientation, xvector
  from mtex.geometry.session import getMTEXpref, setMTEXpref
  from mtex.maps.ebsd import EBSD
  from mtex.plotting.orientationkeys import ipfColors
  pytest.importorskip("numba")
  rng = np.random.default_rng(3)
  n = 5000
  cs = crystalFrame("m-3m")
  ori = orientation.rand(n, cs, rng=rng)
  pos = np.stack([np.arange(n) % 100, np.arange(n) // 100, np.zeros(n)], axis=-1).astype(float)
  ebsd = EBSD(pos, ori, np.ones(n, dtype=int), ["notIndexed", cs])
  old = getMTEXpref("kernels", "numba")
  try:
    setMTEXpref("kernels", "numba")
    fast = ipfColors(ebsd, ipfDirection=xvector)
    setMTEXpref("kernels", "numpy")
    slow = ipfColors(ebsd, ipfDirection=xvector)
  finally:
    setMTEXpref("kernels", old)
  # the same node for all but the directions half way between two nodes
  same = np.all(np.abs(fast - slow) < 1e-12, axis=-1)
  assert np.mean(same) > 0.99
  assert np.max(np.abs(fast - slow)) < 0.2


def test_an_orientation_never_measured_has_no_colour():
  cs = crystalFrame("m-3m")
  q = np.array([[1.0, 0, 0, 0], [np.nan] * 4])
  rgb = ipfColorKey(cs).orientation2color(orientation(q, cs))
  assert np.all(np.isfinite(rgb[0])) and np.isnan(rgb[1]).all()
