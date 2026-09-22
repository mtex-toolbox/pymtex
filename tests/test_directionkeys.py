# the direction colour keys (docs/plans/plotting.md S3): the colours against MTEX's own, read
# through the bridge, as ../mtexJulia/test/directionkeys.jl pins them
import numpy as np
import pytest

from mtex import (HKLDirectionKey, HSVDirectionKey, Miller, S2DeLaValleePoussinKernel, TSLDirectionKey, crystalFrame, degree,
                  rotation, specimenFrame, spotDirectionKey, vector3d, xvector, yvector, zvector)
from mtex.plotting.directionkeys import hsl2hsv, hsv2rgb, rgb2hsv, sphereColor

rng = np.random.default_rng(2)

# the residual sits at the corners of the sector, where the azimuth is equalised on a grid
# of a thousand steps on both sides, so the tolerance is ten of 255 and not one
COLORS = {
  "m-3m": [((0.4786, 0.2151, 0.8513), (0.9000, 0.9000, 0.9000)), ((0.0, 0.0, 1.0), (1.0000, 0.0103, 0.0000)),
           ((0.5774, 0.5774, 0.5774), (0.0071, 0.0000, 1.0000)), ((0.7071, 0.0, 0.7071), (0.0019, 1.0000, 0.0000)),
           ((0.2673, 0.5345, 0.8018), (0.7560, 0.8785, 0.9195)), ((-0.2673, 0.8018, 0.5345), (0.7560, 0.8785, 0.9195)),
           ((0.3651, -0.1826, 0.9129), (0.9220, 0.7363, 0.7938)), ((0.1098, 0.1098, 0.9879), (1.0000, 0.0000, 0.3325))],
  "-3m": [((0.75, -0.4330, 0.5), (0.9037, 1.0000, 0.0000)), ((0.0, 0.0, 1.0), (1.0000, 0.0103, 0.0000)),
          ((0.5, -0.8660, 0.0), (0.0455, 0.0000, 1.0000)), ((1.0, 0.0, 0.0), (0.0455, 0.0000, 1.0000)),
          ((0.2673, 0.5345, 0.8018), (0.7815, 0.0473, 0.4971)), ((-0.2673, 0.8018, 0.5345), (0.9180, 0.7675, 0.8896)),
          ((0.3651, -0.1826, 0.9129), (0.9876, 0.4841, 0.1332)), ((0.1098, 0.1098, 0.9879), (0.7887, 0.0532, 0.0456))],
  "622": [((0.8660, 0.0, 0.5), (0.9139, 0.0000, 1.0000)), ((0.0, 0.0, 1.0), (1.0000, 0.0103, 0.0000)),
          ((0.8660, -0.5, 0.0), (0.0224, 1.0000, 0.0000)), ((0.8660, 0.5, 0.0), (0.0224, 1.0000, 0.0000)),
          ((0.2673, 0.5345, 0.8018), (0.7815, 0.0473, 0.4971))]}

# the white centres MTEX's HSVDirectionKey reports; the port keeps the same half for all but
# 2, 2/m and 23, where MTEX's half is an artefact of the sign its table gives the sector
# normal, and there the port's centre is the mirror image in the fold
WHITE = {"2": (-0.7071, 0.0, 0.7071), "2/m": (-0.5774, 0.5774, 0.5774), "23": (0.4723, 0.0, 0.8814), "32": (0.8660, 0.0, 0.5),
         "222": (0.5774, 0.5774, 0.5774), "4": (0.9239, 0.3827, 0.0), "6": (0.9659, -0.2588, 0.0),
         "m-3": (0.4786, 0.2151, 0.8513), "432": (0.4786, 0.2151, 0.8513), "-3": (0.8660, 0.0, 0.5), "3": (1.0, 0.0, 0.0)}
MIRRORED = ("2", "2/m", "23")


@pytest.mark.parametrize("name", list(COLORS))
def test_P6_the_direction_colour_key(name):
  key = HSVDirectionKey(crystalFrame(name))
  for p, want in COLORS[name]:
    assert np.max(np.abs(key.direction2color(vector3d(*p)) - want)) < 10 / 255


def test_P6_the_key_is_a_key():
  cs = crystalFrame("m-3m")
  key = HSVDirectionKey(cs)
  # symmetrically equivalent directions share a colour, which is the point of the key
  colors = key.direction2color(Miller(1, 2, 3, cs).symmetrise())
  assert colors.shape[-1] == 3 and np.max(np.abs(colors - colors.reshape(-1, 3)[0])) < 1e-9
  # the centre of the sector is the pale one and the corners carry the primaries
  assert np.all(np.abs(key.direction2color(key.whiteCenter) - 0.9) < 0.02)
  corners = key.direction2color(key.sR.vertices())
  assert np.all(corners.max(axis=-1) > 0.99) and np.all(corners.sum(axis=-1) < 1.1)
  # every colour is a colour, and the shape of the list is kept
  vs = vector3d.rand(20, 10, rng=rng)
  c = key.direction2color(vs)
  assert c.shape == (20, 10, 3) and np.all(c >= -1e-9) and np.all(c <= 1 + 1e-9)
  # a NaN or zero direction colours neutral rather than failing
  sk = HSVDirectionKey(specimenFrame())
  field = vector3d(np.array([[[1.0, 0, 0], [0, 1.0, 0]], [[np.nan] * 3, [0.3, 0.2, 0.9]]]))
  cols = sk.direction2color(field)
  assert cols.shape == (2, 2, 3) and np.all(np.isfinite(cols))
  assert np.allclose(cols[1, 0], sk.direction2color(vector3d(0, 0, 0)))
  assert "HSVDirectionKey" in repr(key)


def test_T1_the_other_direction_keys():
  cs = crystalFrame("m-3m")
  key = HSVDirectionKey(cs)
  assert np.all(key.direction2color(key.whiteCenter) > 0.85)
  # the sharp key: white at the centre, black beyond the cap
  sharp = HSVDirectionKey(cs, maxAngle=5 * degree)
  assert np.all(sharp.direction2color(sharp.whiteCenter) > 0.85)
  far = rotation.byAxisAngle(xvector, 20 * degree) * sharp.whiteCenter
  assert np.all(sharp.direction2color(far) < 0.3)
  # the TSL and HKL keys colour the sector, the corner [001] red for HKL
  tsl, hkl = TSLDirectionKey(cs), HKLDirectionKey(cs)
  assert tsl.direction2color(vector3d([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).shape == (3, 3)
  assert hkl.direction2color(zvector)[0] > 0.99 and hkl.direction2color(zvector)[1] < 0.01
  assert np.all(hkl.direction2color(vector3d.rand(50, rng=rng)) <= 1)
  assert HKLDirectionKey(crystalFrame("1")).direction2color(vector3d(0, 0, 1)).shape == (3,)
  # a spot key is its colour at the centre and white far away
  spot = spotDirectionKey(cs, center=Miller(1, 1, 1, cs), color=(0.0, 0.0, 1.0), psi=S2DeLaValleePoussinKernel(halfwidth=7.5 * degree))
  c0 = spot.direction2color(Miller(1, 1, 1, cs))
  assert c0[2] > 0.99 and c0[0] < 0.05
  assert np.all(spot.direction2color(Miller(1, 0, 0, cs)) > 0.95)
  with pytest.raises(ValueError):
    spotDirectionKey(cs, center=Miller([1, 1], [1, 0], [1, 0], cs), color=(1, 0, 0))


def test_S3c_the_specimen_key_and_the_white_halves():
  # the whole sphere folds at the equator of the convention: the pole towards the viewer,
  # the centre of the chart that is drawn, is the white centre
  ss = specimenFrame("1")
  key = HSVDirectionKey(ss)
  pole = ss.how2plot.outOfScreen.data
  assert len(key.folds) == 1 and abs(key.folds[0] @ zvector.data) > 1 - 1e-9
  assert key.whiteCenter.data @ pole > 1 - 1e-9
  assert HSVDirectionKey(crystalFrame("1")).whiteCenter.data @ crystalFrame("1").how2plot.outOfScreen.data > 1 - 1e-9
  # a key of a specimen frame with symmetry colours axes
  k2 = HSVDirectionKey(specimenFrame("222"))
  c = k2.direction2color(vector3d.rand(20, antipodal=True, rng=rng))
  assert np.all((c >= 0) & (c <= 1))
  for name, w in WHITE.items():
    key = HSVDirectionKey(crystalFrame(name))
    c = key.whiteCenter.data.reshape(3)
    m = np.asarray(w) / np.linalg.norm(w)
    if name in MIRRORED:
      n = key.folds[-1]
      assert (m - 2 * (m @ n) * n) @ c > 0.999
    else:
      assert m @ c > 0.999


def test_the_colour_arithmetic():
  rgb = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0.5, 0.5, 0.5]])
  hsv = rgb2hsv(rgb)
  assert np.allclose(hsv[:, 0], [0, 1 / 3, 2 / 3, 0]) and np.allclose(hsv[:, 1], [1, 1, 1, 0])
  assert np.allclose(hsv2rgb(hsv[:, 0], hsv[:, 1], hsv[:, 2]), rgb)
  h, s, v = hsl2hsv(0.0, 1.0, 0.5)
  assert np.isclose(s, 1) and np.isclose(v, 1)
  # the white centre of the colour sphere is pale, the black one dark, the rim saturated
  assert np.all(np.abs(sphereColor(1.0, 0.0) - 0.9) < 0.02) and np.all(np.abs(sphereColor(0.0, 0.0) - 0.25) < 0.02)
  assert sphereColor(0.5, 0.0).max() > 0.99 and sphereColor(0.5, 2 * np.pi / 3).argmax() == 1
