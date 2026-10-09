# the direction colour keys (docs/plans/plotting.md S3): the colours against MTEX's own, read
# through the bridge, as ../mtexJulia/test/directionkeys.jl pins them
import numpy as np
import pytest

from mtex import (HKLDirectionKey, HSVDirectionKey, Miller, S2DeLaValleePoussinKernel, TSLDirectionKey, crystalFrame, degree,
                  rotation, specimenFrame, spotDirectionKey, vector3d, xvector, yvector, zvector)
from mtex.plotting.directionkeys import hsl2hsv, hsv2rgb, oklab2rgb, rgb2hsv, rgb2oklab, sphereColor

rng = np.random.default_rng(2)

# MATLAB's colours since both pin a triangle's vertices as exact knots of the equalised azimuth
# (MATLAB b23d9cc59); the rest of the grid differs, which leaves up to 5e-4 at the corners
COLORS = {
  "m-3m": [((0.4786, 0.2151, 0.8513), (0.9000, 0.9000, 0.8999)), ((0.0, 0.0, 1.0), (1.0000, 0.0103, 0.0000)),
           ((0.5774, 0.5774, 0.5774), (0.0039, 0.0000, 1.0000)), ((0.7071, 0.0, 0.7071), (0.0000, 1.0000, 0.0071)),
           ((0.2673, 0.5345, 0.8018), (0.7561, 0.8786, 0.9195)), ((-0.2673, 0.8018, 0.5345), (0.7561, 0.8786, 0.9195)),
           ((0.3651, -0.1826, 0.9129), (0.9220, 0.7361, 0.7942)), ((0.1098, 0.1098, 0.9879), (1.0000, 0.0000, 0.3349))],
  "-3m": [((0.75, -0.4330, 0.5), (0.8999, 1.0000, 0.0001)), ((0.0, 0.0, 1.0), (1.0000, 0.0103, 0.0000)),
          ((0.5, -0.8660, 0.0), (0.0038, 0.0000, 1.0000)), ((1.0, 0.0, 0.0), (0.0039, 0.0000, 1.0000)),
          ((0.2673, 0.5345, 0.8018), (0.7817, 0.0472, 0.5031)), ((-0.2673, 0.8018, 0.5345), (0.9180, 0.7675, 0.8908)),
          ((0.3651, -0.1826, 0.9129), (0.9876, 0.4845, 0.1330)), ((0.1098, 0.1098, 0.9879), (0.7886, 0.0532, 0.0456))],
  "622": [((0.8660, 0.0, 0.5), (0.9035, 0.0000, 1.0000)), ((0.0, 0.0, 1.0), (1.0000, 0.0103, 0.0000)),
          ((0.8660, -0.5, 0.0), (0.0000, 1.0000, 0.0072)), ((0.8660, 0.5, 0.0), (0.0000, 1.0000, 0.0072)),
          ((0.2673, 0.5345, 0.8018), (0.7817, 0.0472, 0.5031))]}

# the white centres MTEX's HSVDirectionKey reports; the port keeps the same half for all but
# 2, 2/m and 23, where MTEX's half is an artefact of the sign its table gives the sector
# normal, and there the port's centre is the mirror image in the fold
WHITE = {"2": (-0.7071, 0.0, 0.7071), "2/m": (-0.5774, 0.5774, 0.5774), "23": (0.4723, 0.0, 0.8814), "32": (0.8660, 0.0, 0.5),
         "222": (0.5774, 0.5774, 0.5774), "4": (0.9239, 0.3827, 0.0), "6": (0.9659, -0.2588, 0.0),
         "m-3": (0.4786, 0.2151, 0.8513), "432": (0.4786, 0.2151, 0.8513), "-3": (0.8660, 0.0, 0.5), "3": (1.0, 0.0, 0.0)}
MIRRORED = ("2", "2/m", "23")


@pytest.mark.parametrize("name", list(COLORS))
def test_HSVDirectionKey_coloursAreMTEXs(name):
  key = HSVDirectionKey(crystalFrame(name))
  for p, want in COLORS[name]:
    assert np.max(np.abs(key.direction2color(vector3d(*p)) - want)) < 2e-3


def test_HSVDirectionKey_equivalentDirectionsShareAColour():
  cs = crystalFrame("m-3m")
  key = HSVDirectionKey(cs)
  # symmetrically equivalent directions share a colour, which is the point of the key
  colors = key.direction2color(Miller(1, 2, 3, cs).symmetrise())
  assert colors.shape[-1] == 3 and np.max(np.abs(colors - colors.reshape(-1, 3)[0])) < 1e-9


def test_HSVDirectionKey_centreIsPaleAndCornersArePrimaries():
  key = HSVDirectionKey(crystalFrame("m-3m"))
  # the centre of the sector is the pale one and the corners carry the primaries
  assert np.all(np.abs(key.direction2color(key.whiteCenter) - 0.9) < 0.02)
  corners = key.direction2color(key.sR.vertices())
  assert np.all(corners.max(axis=-1) > 0.99) and np.all(corners.sum(axis=-1) < 1.1)


def test_HSVDirectionKey_keepsTheShapeAndGivesColours():
  key = HSVDirectionKey(crystalFrame("m-3m"))
  # every colour is a colour, and the shape of the list is kept
  vs = vector3d.rand(20, 10, rng=rng)
  c = key.direction2color(vs)
  assert c.shape == (20, 10, 3) and np.all(c >= -1e-9) and np.all(c <= 1 + 1e-9)


def test_HSVDirectionKey_nanOrZeroDirectionColoursNeutral():
  # a NaN or zero direction colours neutral rather than failing
  sk = HSVDirectionKey(specimenFrame())
  field = vector3d(np.array([[[1.0, 0, 0], [0, 1.0, 0]], [[np.nan] * 3, [0.3, 0.2, 0.9]]]))
  cols = sk.direction2color(field)
  assert cols.shape == (2, 2, 3) and np.all(np.isfinite(cols))
  assert np.allclose(cols[1, 0], sk.direction2color(vector3d(0, 0, 0)))


def test_HSVDirectionKey_reprNamesTheKey():
  assert "HSVDirectionKey" in repr(HSVDirectionKey(crystalFrame("m-3m")))


def test_HSVDirectionKey_maxAngleIsWhiteAtTheCentreAndBlackBeyondTheCap():
  # the sharp key: white at the centre, black beyond the cap
  sharp = HSVDirectionKey(crystalFrame("m-3m"), maxAngle=5 * degree)
  assert np.all(sharp.direction2color(sharp.whiteCenter) > 0.85)
  far = rotation.byAxisAngle(xvector, 20 * degree) * sharp.whiteCenter
  assert np.all(sharp.direction2color(far) < 0.3)


def test_TSLDirectionKey_givesOneColourPerDirection():
  tsl = TSLDirectionKey(crystalFrame("m-3m"))
  assert tsl.direction2color(vector3d([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).shape == (3, 3)


def test_HKLDirectionKey_colours001Red():
  hkl = HKLDirectionKey(crystalFrame("m-3m"))
  assert hkl.direction2color(zvector)[0] > 0.99 and hkl.direction2color(zvector)[1] < 0.01


def test_HKLDirectionKey_staysBelowOne():
  hkl = HKLDirectionKey(crystalFrame("m-3m"))
  assert np.all(hkl.direction2color(vector3d.rand(50, rng=rng)) <= 1)


def test_HKLDirectionKey_ofTriclinicGivesOneColour():
  assert HKLDirectionKey(crystalFrame("1")).direction2color(vector3d(0, 0, 1)).shape == (3,)


def test_spotDirectionKey_isItsColourAtTheCentreAndWhiteFarAway():
  cs = crystalFrame("m-3m")
  spot = spotDirectionKey(cs, center=Miller(1, 1, 1, cs), color=(0.0, 0.0, 1.0), psi=S2DeLaValleePoussinKernel(halfwidth=7.5 * degree))
  c0 = spot.direction2color(Miller(1, 1, 1, cs))
  assert c0[2] > 0.99 and c0[0] < 0.05
  assert np.all(spot.direction2color(Miller(1, 0, 0, cs)) > 0.95)


def test_spotDirectionKey_refusesSeveralCentres():
  cs = crystalFrame("m-3m")
  with pytest.raises(ValueError):
    spotDirectionKey(cs, center=Miller([1, 1], [1, 0], [1, 0], cs), color=(1, 0, 0))


def test_HSVDirectionKey_ofSpecimenFoldsAtTheEquatorWithTheViewerPoleWhite():
  # the whole sphere folds at the equator of the convention: the pole towards the viewer,
  # the centre of the chart that is drawn, is the white centre
  ss = specimenFrame("1")
  key = HSVDirectionKey(ss)
  pole = ss.how2plot.outOfScreen.data
  assert len(key.folds) == 1 and abs(key.folds[0] @ zvector.data) > 1 - 1e-9
  assert key.whiteCenter.data @ pole > 1 - 1e-9


def test_HSVDirectionKey_ofTriclinicCrystalIsWhiteTowardsTheViewer():
  assert HSVDirectionKey(crystalFrame("1")).whiteCenter.data @ crystalFrame("1").how2plot.outOfScreen.data > 1 - 1e-9


def test_HSVDirectionKey_ofSymmetricSpecimenFrameColoursAxes():
  # a key of a specimen frame with symmetry colours axes
  k2 = HSVDirectionKey(specimenFrame("222"))
  c = k2.direction2color(vector3d.rand(20, antipodal=True, rng=rng))
  assert np.all((c >= 0) & (c <= 1))


@pytest.mark.parametrize("name", list(WHITE))
def test_HSVDirectionKey_whiteCentreIsMTEXsOrItsMirror(name):
  key = HSVDirectionKey(crystalFrame(name))
  c = key.whiteCenter.data.reshape(3)
  m = np.asarray(WHITE[name]) / np.linalg.norm(WHITE[name])
  if name in MIRRORED:
    n = key.folds[-1]
    assert (m - 2 * (m @ n) * n) @ c > 0.999
  else:
    assert m @ c > 0.999


def test_rgb2hsv_givesTheHueAndSaturationOfPrimariesAndGrey():
  rgb = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0.5, 0.5, 0.5]])
  hsv = rgb2hsv(rgb)
  assert np.allclose(hsv[:, 0], [0, 1 / 3, 2 / 3, 0]) and np.allclose(hsv[:, 1], [1, 1, 1, 0])


def test_hsv2rgb_invertsRgb2hsv():
  rgb = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0.5, 0.5, 0.5]])
  hsv = rgb2hsv(rgb)
  assert np.allclose(hsv2rgb(hsv[:, 0], hsv[:, 1], hsv[:, 2]), rgb)


def test_hsl2hsv_ofFullSaturationAtHalfLightnessIsFullValue():
  h, s, v = hsl2hsv(0.0, 1.0, 0.5)
  assert np.isclose(s, 1) and np.isclose(v, 1)


def test_sphereColor_isPaleAtTheWhiteCentreDarkAtTheBlackAndSaturatedAtTheRim():
  # the white centre of the colour sphere is pale, the black one dark, the rim saturated
  assert np.all(np.abs(sphereColor(1.0, 0.0) - 0.9) < 0.02) and np.all(np.abs(sphereColor(0.0, 0.0) - 0.25) < 0.02)
  assert sphereColor(0.5, 0.0).max() > 0.99 and sphereColor(0.5, 2 * np.pi / 3).argmax() == 1


# the rate at which the colour of a key changes, in OKLab per degree, at random directions
# of its sector and along two perpendicular steps
def _colourSpeed(key, n=4000, d=0.15 * degree):
  v = vector3d.rand(6 * n, rng=np.random.default_rng(3)).data.reshape(-1, 3)
  v = v[key.sR.checkInside(vector3d(v)).reshape(-1)][:n]
  e1 = np.cross(v, [0.3, -0.5, 0.8])
  e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
  lab = rgb2oklab(key.direction2color(vector3d(v)))
  g = []
  for e in (e1, np.cross(v, e1)):
    w = v + d * e
    out = ~key.sR.checkInside(vector3d(w)).reshape(-1)
    w[out] = v[out] - d * e[out]
    w /= np.linalg.norm(w, axis=-1, keepdims=True)
    g.append(np.linalg.norm(rgb2oklab(key.direction2color(vector3d(w))) - lab, axis=-1) / (d / degree))
  return np.hypot(*g)


# the ratio of the fast to the slow colour changes of a key over its sector
def _colourSpread(name, **kw):
  g = _colourSpeed(HSVDirectionKey(crystalFrame(name), **kw))
  return np.percentile(g, 90) / np.percentile(g, 10)


def test_oklab2rgb_invertsRgb2oklab():
  rgb = np.array([[1.0, 0, 0], [0, 1.0, 0], [0.2, 0.4, 0.9], [0.5, 0.5, 0.5]])
  assert np.allclose(oklab2rgb(rgb2oklab(rgb)), rgb, atol=1e-6)


@pytest.mark.parametrize("name", ["m-3m", "6/mmm", "-3m", "4mm"])
def test_HSVDirectionKey_perceptualChangesItsColourAtASteadierRate(name):
  # MTEX's key changes its colour three times as fast in some parts of the sector as in
  # others; the perceptual one at a steadier rate, and less steady as its lightness follows the hue
  even, follows, mtex = _colourSpread(name, perceptual=True), _colourSpread(name, perceptual=True, hueLightness=1), _colourSpread(name)
  assert even < 2.4 < 2.9 < mtex and even < follows < mtex


def test_HSVDirectionKey_perceptualCentreIsPale():
  key = HSVDirectionKey(crystalFrame("m-3m"), perceptual=True)
  assert rgb2oklab(key.direction2color(key.whiteCenter))[0] > 0.94


@pytest.mark.parametrize("hueLightness", [0, 0.5, 1])
def test_HSVDirectionKey_perceptualCubicCornersAreRedGreenBlue(hueLightness):
  # the vertices of the cubic triangle red, green and blue
  key = HSVDirectionKey(crystalFrame("m-3m"), perceptual=True, hueLightness=hueLightness)
  corners = key.direction2color(vector3d([[0, 0, 1], [1, 0, 1], [1, 1, 1]]))
  assert list(np.argmax(corners, axis=-1)) == [0, 1, 2]
  assert np.all((corners >= 0) & (corners <= 1))


@pytest.mark.parametrize("name", ["-3m", "32", "6/mmm", "4/mmm", "m-3m", "mmm", "-43m"])
def test_HSVDirectionKey_coloursACrystalDirectionWhateverTheAlignment(name):
  # the key is a function of the crystal direction: turning the lattice against the axes,
  # c* off z, changes no colour (MATLAB a3e47891a)
  rng2 = np.random.default_rng(3)
  hkl = rng2.integers(-3, 4, size=(40, 3))
  hkl = hkl[np.any(hkl != 0, axis=1)]
  abc = [5, 6, 7] if name == "mmm" else [5, 5, 5] if name in ("m-3m", "-43m") else [5, 5, 7]
  ref = crystalFrame(name, abc, "X||a", "Z||c*")
  c0 = HSVDirectionKey(ref).direction2color(Miller(hkl, ref))
  for al in [("X||b*", "Z||a"), ("X||b", "Z||a*")]:
    cs = crystalFrame(name, abc, *al)
    # a direction on a boundary of the sector, as [-3 -1 0] of -43m, is folded to either side of it,
    # where the key differs by 2e-3 in one channel at any resolution of the equalisation
    assert np.allclose(HSVDirectionKey(cs).direction2color(Miller(hkl, cs)), c0, atol=3e-3)
