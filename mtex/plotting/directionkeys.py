# direction colour keys: a direction of the fundamental sector becomes a colour (MATLAB
# plotting/directionColorKeys; docs/plans/plotting.md S3)
#
# MTEX's colour sphere (HSVDirectionKey/direction2color with ar2rgb): the radius in the
# folded sector becomes the polar angle, pi at the white centre and 0 at the black one, the
# two halves meeting at pi/2 on the boundary they share; the equalised azimuth becomes the
# hue. Lightness follows the polar angle through the gray gradient, and the saturation is
# taken in HSL, which holds the lightness steady around the circle where HSV does not.
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..geometry.frames import referenceFrame
from ..geometry.kernels import qrot
from ..geometry.plottingconvention import plottingConvention
from ..geometry.sphericalregion import conventionalSector, projectIntoRegion, sphericalRegion
from ..geometry.units import degree
from ..geometry.vector3d import vector3d, xvector, asVector3d
from ..geometry.representations import unitRows
from .sectorfold import _angle, azimuthAbout, polarCoordinates, regionCenter, sectorFolds, uniqueNormals
from ..geometry.display import aligned

# ------------------------------------------------------------------ colour arithmetic

def rgb2hsv(rgb):
  """MATLAB rgb2hsv, hsv2rgb and hsl2hsv on arrays of shape (..., 3) or three arrays"""
  rgb = np.asarray(rgb, dtype=float)
  r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
  mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
  d = mx - mn
  safe = np.where(d < 1e-12, 1.0, d)
  h = np.where(mx == r, np.mod((g - b) / safe, 6), np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4)) / 6
  h = np.where(d < 1e-12, 0.0, h)
  s = np.where(mx < 1e-12, 0.0, d / np.where(mx < 1e-12, 1.0, mx))
  return np.stack([h, s, mx], axis=-1)


def hsl2hsv(h, s, l):
  """the HSV coordinates of colours given in HSL, as three arrays h, s, v"""
  ll = 2 * np.asarray(l, dtype=float)
  ss = np.where(ll <= 1, s * ll, s * (2 - ll))
  v = (ll + ss) / 2
  sv = np.where(ll + ss < 1e-12, 0.0, 2 * ss / np.where(ll + ss < 1e-12, 1.0, ll + ss))
  return h, sv, v


def hsv2rgb(h, s, v):
  """the RGB colours of HSV coordinates, three arrays of any common shape, as (..., 3) in [0, 1]"""
  h, s, v = np.broadcast_arrays(np.asarray(h, dtype=float), np.asarray(s, dtype=float), np.asarray(v, dtype=float))
  h6 = 6 * np.mod(h, 1)
  i = np.floor(h6).astype(int) % 6
  f = h6 - np.floor(h6)
  p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
  r = np.choose(i, [v, q, p, p, t, v])
  g = np.choose(i, [t, v, v, q, p, p])
  b = np.choose(i, [p, p, t, v, v, q])
  return np.stack([r, g, b], axis=-1)


# plain HSV gives yellow and cyan more of the circle than red, green and blue, so the hue is
# passed through the cumulative of three Gaussians sitting on the primaries (MATLAB ar2rgb)
def _hueCorrection(steps=1000):
  z = np.linspace(0, 1, steps)
  f = 0.5 + sum(np.exp(-200 * (np.mod(z - c + 0.5, 1) - 0.5) ** 2) for c in (0.0, 1 / 3, 2 / 3))
  return np.cumsum(f / np.sum(f))


HUE_CORRECTION = _hueCorrection()


def correctedHue(h):
  """the hue passed through the correction that gives red, green and blue as much of the colour
  circle as yellow, cyan and magenta, MATLAB's ar2rgb"""
  n = len(HUE_CORRECTION)
  t = np.mod(np.asarray(h, dtype=float), 1) * (n - 1)
  i = np.clip(np.floor(t).astype(int), 0, n - 2)
  return HUE_CORRECTION[i] + (t - i) * (HUE_CORRECTION[i + 1] - HUE_CORRECTION[i])


def angleRadiusColor(hue, radius, gray, hueCorrection=True):
  """the hue from the angle, the lightness from the radius and the saturation from the gray"""
  hue, radius, gray = np.broadcast_arrays(np.asarray(hue, dtype=float), np.asarray(radius, dtype=float), np.asarray(gray, dtype=float))
  L = (radius - 0.5) * gray + 0.5
  d = 1 - np.abs(2 * L - 1)
  S = np.where(d < 1e-12, 0.0, gray * (1 - np.abs(2 * radius - 1)) / np.where(d < 1e-12, 1.0, d))
  h, s, v = hsl2hsv(hue, S, L)
  return hsv2rgb(correctedHue(h) if hueCorrection else h, s, v)


KEY_GRAY_VALUE = (0.2, 0.5)      # the gray of the white and of the black centre
KEY_GRAY_GRADIENT = (0.5, 0.5)   # how fast the gray fades away from a centre


def sphereColor(radius, rho, grayValue=KEY_GRAY_VALUE, grayGradient=KEY_GRAY_GRADIENT, colorStretching=1.0, gray=None,
                hueCorrection=True):
  """the colour of a point of the colour sphere: the radius, 1 at the white centre and 0 at
  the black one, and the azimuth as the hue"""
  radius, rho = np.broadcast_arrays(np.asarray(radius, dtype=float), np.asarray(rho, dtype=float))
  theta = radius * np.pi
  white = theta > np.pi / 2
  g = np.where(white, grayGradient[0], grayGradient[1])
  t = ((2 * g * theta / np.pi + (1 - g) * (1 - np.cos(theta))) / 2) ** colorStretching
  gv = np.where(white, grayValue[0], grayValue[1])
  gr = 1 - 2 * gv * np.abs(t - 0.5) if gray is None else np.asarray(gray, dtype=float)
  return angleRadiusColor(np.mod(rho / (2 * np.pi), 1), t, gr, hueCorrection)


# OKLab (Ottosson 2020) to and from gamma encoded sRGB, over arrays of shape (..., 3)
_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                 [0.0883024619, 0.2817188376, 0.6299787005]])
_LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                 [0.0259040371, 0.7827717662, -0.8086757660]])
_LMS_INV, _LAB_INV = np.linalg.inv(_LMS), np.linalg.inv(_LAB)


def rgb2oklab(rgb):
  """the OKLab coordinates of gamma encoded sRGB colours, over arrays of shape (..., 3)"""
  c = np.asarray(rgb, dtype=float)
  lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
  return np.cbrt(lin @ _LMS.T) @ _LAB.T


def oklab2rgb(lab):
  """the gamma encoded sRGB colours of OKLab coordinates, clipped to the gamut, over (..., 3)"""
  lin = ((np.asarray(lab, dtype=float) @ _LAB_INV.T) ** 3) @ _LMS_INV.T
  lin = np.clip(lin, 0, 1)
  return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


# the perceptual colour sphere: a double cone in OKLab over the hue circle, the lightness
# running linearly from the white centre to the boundary and on to the black centre, the
# chroma growing linearly away from both centres, the azimuth turning the hue. Equal steps
# along the radius and along the arc are then equal colour differences for every hue.
# hueLightness moves the lightness of the boundary from one value for all hues (0, the most
# even key) to the lightness at which each hue is most saturated in sRGB (1, bright yellow
# and cyan beside dark blue, close to MTEX's key); the chroma of the boundary is at each hue
# the largest the whole cone keeps inside sRGB
PERCEPTUAL_CONE = (0.95, 0.62, 0.30)   # the lightness at the white centre, on the boundary and at the black centre
PERCEPTUAL_HUE_SMOOTHING = 12 * degree
_HUES = np.linspace(0, 2 * np.pi, 360, endpoint=False)


# a periodic table convolved with a Gaussian of the given width in radians
def _circularSmooth(values, width):
  n = len(values)
  x = (np.arange(n) - n // 2) * 2 * np.pi / n
  w = np.exp(-x ** 2 / (2 * width ** 2))
  return np.real(np.fft.ifft(np.fft.fft(values) * np.fft.fft(np.fft.ifftshift(w / w.sum()))))


# OKLab spends 81 degrees of hue from red to yellow and 33 from yellow to green, so an evenly
# turning hue gives red two thirds of that side of the sector. The azimuth is taken instead
# through the hues of red, yellow, green, cyan, blue and magenta at every sixth of the turn,
# the slope between them smoothed so the steps leave no seam in the key
def _perceptualHue(steps=3600, smoothing=PERCEPTUAL_HUE_SMOOTHING):
  anchors = np.array([np.arctan2(*rgb2oklab(c)[[2, 1]]) for c in
                      ([1.0, 0, 0], [1.0, 1, 0], [0, 1.0, 0], [0, 1.0, 1], [0, 0, 1.0], [1.0, 0, 1])])
  gaps = np.mod(np.diff(np.append(anchors, anchors[0])), 2 * np.pi)
  slope = _circularSmooth(np.repeat(gaps / (np.pi / 3), steps // 6), smoothing)
  return anchors[0] + np.concatenate([[0.0], np.cumsum(slope)]) * 2 * np.pi / steps


PERCEPTUAL_HUE = _perceptualHue()


def _inGamut(L, C, h):
  lin = ((np.stack(np.broadcast_arrays(L, C * np.cos(h), C * np.sin(h)), axis=-1) @ _LAB_INV.T) ** 3) @ _LMS_INV.T
  return np.all((lin >= -1e-9) & (lin <= 1 + 1e-9), axis=-1)


# the largest c for which the chroma c t at the lightness L stays in sRGB, over every hue at
# once: L and t are (hues, samples), the result one value per hue
def _largestChroma(L, t, steps=26):
  lo, hi = np.zeros(len(_HUES)), np.full(len(_HUES), 0.45)
  for _ in range(steps):
    m = (lo + hi) / 2
    ok = np.all(_inGamut(L, m[:, None] * t, _HUES[:, None]), axis=-1)
    lo, hi = np.where(ok, m, lo), np.where(ok, hi, m)
  return lo


# the lightness at which each hue reaches its largest chroma in sRGB, the cusp of the gamut
@lru_cache(maxsize=1)
def _cuspLightness():
  Ls = np.linspace(0.3, 0.99, 139)
  C = np.stack([_largestChroma(np.full((len(_HUES), 1), L), np.ones(1)) for L in Ls], axis=-1)
  return Ls[np.argmax(C, axis=-1)]


# the boundary lightness and chroma of the cone at every hue of _HUES; the chroma is
# smoothed and then lowered until it is nowhere above the largest one
@lru_cache(maxsize=8)
def _perceptualCone(hueLightness):
  L0 = PERCEPTUAL_CONE[1]
  Lb = L0 if hueLightness == 0 else _circularSmooth(L0 + hueLightness * (_cuspLightness() - L0), 10 * degree)
  Lb = np.broadcast_to(Lb, _HUES.shape)
  s = np.linspace(0.01, 1.99, 199)
  L = _coneLightness(s[None, :], Lb[:, None])
  Cb = _largestChroma(L, 1 - np.abs(1 - s)[None, :])
  smooth = _circularSmooth(Cb, 8 * degree)
  return Lb, smooth - max(0.0, np.max(smooth - Cb))


# the lightness at s, 0 at the white centre, 1 on the boundary of lightness Lb, 2 at the black centre
def _coneLightness(s, Lb):
  Lw, _, Ld = PERCEPTUAL_CONE
  return np.where(s <= 1, Lw + (Lb - Lw) * s, Lb + (Ld - Lb) * (s - 1))


def perceptualSphereColor(radius, rho, gray=None, hueLightness=0.0):
  """the colour of a point of the perceptual colour sphere, a double cone in OKLab: the radius, 1
  at the white centre and 0 at the black one, the azimuth as the hue; hueLightness moves the
  boundary towards the lightness each hue is brightest at"""
  radius, rho = np.broadcast_arrays(np.asarray(radius, dtype=float), np.asarray(rho, dtype=float))
  Lbs, Cbs = _perceptualCone(float(hueLightness))
  h = np.interp(np.mod(rho, 2 * np.pi), np.linspace(0, 2 * np.pi, len(PERCEPTUAL_HUE)), PERCEPTUAL_HUE)
  Lb = np.interp(np.mod(h, 2 * np.pi), _HUES, Lbs, period=2 * np.pi)
  Cb = np.interp(np.mod(h, 2 * np.pi), _HUES, Cbs, period=2 * np.pi)
  s = 2 * (1 - radius)
  C = Cb * (1 - np.abs(1 - s)) * (1.0 if gray is None else np.asarray(gray, dtype=float))
  return oklab2rgb(np.stack([_coneLightness(s, Lb), C * np.cos(h), C * np.sin(h)], axis=-1))


# ------------------------------------------------------------------ the keys

def _asDirections(h):
  v = asVector3d(h)
  return v.data.reshape(-1, 3), v.shape


def _sym(CS):
  return CS.sym if isinstance(CS, referenceFrame) else CS


class directionColorKey:
  """a colour key turning directions into colours

    rgb = key.direction2color(h)     an (..., 3) array in [0, 1]
    key.sym, key.sR                  the group and the sector the key is defined on
    key.plot()                       the key drawn on its sector
  """

  def __init__(self, CS):
    self.CS = CS
    self.sym = _sym(CS)
    self.sR = conventionalSector(CS)

  @property
  def keySector(self):
    """a sector the key is drawn over"""
    return self.sR

  def direction2color(self, h):
    """the colours of directions, an (..., 3) array in [0, 1]; defined by every key"""
    raise NotImplementedError

  def plot(self, **kwargs):
    """the key drawn on its sector"""
    from .sphericalplot import plotKey
    return plotKey(self, **kwargs)

  def __repr__(self):
    return aligned(f"{type(self).__name__} ({self.CS.char()})")


class HSVDirectionKey(directionColorKey):
  """MTEX's default direction key: white at the centre of the fundamental sector, the primary
  colours at its vertices

    HSVDirectionKey(cs)
    HSVDirectionKey(cs, whiteCenter=v, maxAngle=5*degree)
    HSVDirectionKey(cs, perceptual=True, hueLightness=0.5)

  A finite maxAngle makes the key sharp: the whole colour circle is spent on the cap of that
  opening about the white centre and everything outside it is black. perceptual=True takes
  the colours from a double cone in OKLab instead of MTEX's HSL sphere: the colour changes
  at a nearly steady rate over the sector. hueLightness, from 0 to 1, lets the lightness of
  the boundary follow the hue towards where each hue is brightest: yellow comes back as
  yellow, and the key comes closer to MTEX's and less even.
  """

  def __init__(self, CS=None, whiteCenter=None, maxAngle=np.inf, sR=None, colorStretching=1.0, grayValue=KEY_GRAY_VALUE,
               grayGradient=KEY_GRAY_GRADIENT, antipodal=False, perceptual=False, hueLightness=0.0):
    # HSVDirectionKey() colours the whole sphere of the specimen; antipodal=True the key of axes
    from ..geometry.frames import specimenFrame
    from ..geometry.vector3d import vector3d as _v
    # HSVDirectionKey(v): the key of the frame the directions live in, MATLAB's call
    if isinstance(CS, _v):
      antipodal = antipodal or CS.antipodal
      CS = specimenFrame.default if CS.frame is None else CS.frame
    CS = specimenFrame.default if CS is None else CS
    if antipodal and hasattr(CS, "Laue"):
      CS = CS.Laue()
    super().__init__(CS)
    sector = self.sR if sR is None else sR
    N = uniqueNormals(sector)
    self.folds = sectorFolds(self.sym, sector)
    self.sR = sphericalRegion(np.array(N).reshape(-1, 3), np.zeros(len(N)), frame=sector.frame)
    self.reduced = sphericalRegion(np.array(N + self.folds).reshape(-1, 3), np.zeros(len(N) + len(self.folds)), frame=sector.frame)
    self.whiteCenter = vector3d(regionCenter(self.reduced) if whiteCenter is None else unitRows(_asDirections(whiteCenter)[0][0]), isNormalized=True)
    pC = CS.how2plot if isinstance(CS, referenceFrame) and CS.how2plot is not None else plottingConvention.default
    self.reference = pC.east
    self.maxAngle = float(maxAngle)
    self.colorStretching, self.grayValue, self.grayGradient = colorStretching, grayValue, grayGradient
    self.perceptual, self.hueLightness = bool(perceptual), float(hueLightness)

  def _sphereColor(self, radius, rho, gray=None):
    if self.perceptual:
      return perceptualSphereColor(radius, rho, gray, self.hueLightness)
    return sphereColor(radius, rho, self.grayValue, self.grayGradient, self.colorStretching, gray)

  def direction2color(self, h, gray=None):
    """the colour of directions: each is brought into the sector, then into the white half, the
    fold telling which of the two centres its colours run from. A sharp key instead brings
    each to the equivalent nearest the white centre and reads the radius off the angle to it"""
    xyz, shape = _asDirections(h)
    n = len(xyz)
    norm = np.linalg.norm(xyz, axis=-1)
    finite = np.all(np.isfinite(xyz), axis=-1) & (norm > 0)
    unit = np.where(finite[:, None], xyz / np.where(finite, norm, 1.0)[:, None], self.reference.data)
    gray = None if gray is None else np.asarray(gray, dtype=float).reshape(-1)
    wC = self.whiteCenter.data.reshape(3)
    if np.isfinite(self.maxAngle):
      owC = vector3d(wC).orth().data.reshape(3) if _angle(np.array([0.0, 0.0, 1.0]), wC) < 1 * degree else np.array([0.0, 0.0, 1.0])
      els = self.sym.rot.flatten().data[:, :4]
      imgs = qrot(els[None, :, :], unit[:, None, :]) * self.sym.rot.flatten().data[None, :, 4:5]
      best = np.argmax(imgs @ wC, axis=1)
      hs = imgs[np.arange(n), best]
      rho = azimuthAbout2(hs, wC, owC)
      r = np.maximum(0.0, 1 - _angle(hs, wC) / self.maxAngle)
      rgb = self._sphereColor(np.where(finite, r, 0.0), np.where(finite, rho, 0.0), gray)
      return rgb.reshape(shape + (3,))
    hs = projectIntoRegion(vector3d(unit, frame=self.sR.frame, isNormalized=True), self.sR, self.sym).data.reshape(-1, 3)
    white = np.ones(n, dtype=bool)
    for fold in self.folds:
      flip = hs @ fold < 1e-5
      hs = np.where(flip[:, None], hs - 2 * (hs @ fold)[:, None] * fold, hs)
      white &= ~flip
    r, rho = polarCoordinates(self.reduced, hs, wC, self.reference)
    radius = np.where(white, 0.5 + r / 2, (1 - r) / 2)
    rgb = self._sphereColor(np.where(finite, radius, 0.0), np.where(finite, rho, 0.0), gray)
    return rgb.reshape(shape + (3,))


def azimuthAbout2(h, center, ref):
  """the azimuth of h, one direction or rows of them, about the centre, measured from the
  reference direction"""
  h = np.asarray(h, dtype=float)
  p = h - (h @ center)[..., None] * center
  r = ref - (ref @ center) * center
  return np.arctan2(np.cross(r, p) @ center, p @ r)


class TSLDirectionKey(directionColorKey):
  """the vendors' keys, both on the Laue group whatever the frame carries (MATLAB
  TSLDirectionKey, HKLDirectionKey): TSL reads the polar coordinates of the sector from its
  centre, unfolded and without the hue correction; HKL blends the three primaries linearly
  over the polar angle and the azimuth of the sector"""

  def __init__(self, CS):
    super().__init__(CS.Laue() if isinstance(CS, referenceFrame) else CS.Laue())
    V = self.sR.vertices().data.reshape(-1, 3)
    self.center = regionCenter(self.sR)
    self.reference = V[0] if len(V) else np.array([1.0, 0.0, 0.0])

  @property
  def whiteCenter(self):
    """the centre of the sector the colours run from"""
    return vector3d(self.center, isNormalized=True)

  def direction2color(self, h, gray=None):
    """the colours of directions, the polar coordinates in the sector about its centre turned into
    hue and lightness without the hue correction; gray lowers the saturation"""
    xyz, shape = _asDirections(h)
    unit = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    hs = projectIntoRegion(vector3d(unit, frame=self.sR.frame, isNormalized=True), self.sR, self.sym).data.reshape(-1, 3)
    r, rho = polarCoordinates(self.sR, hs, self.center, self.reference)
    g = 1.0 if gray is None else np.asarray(gray, dtype=float).reshape(-1)
    return angleRadiusColor(np.mod(rho / (2 * np.pi), 1), 0.5 + r / 2, g, hueCorrection=False).reshape(shape + (3,))


class HKLDirectionKey(directionColorKey):
  """the HKL vendor key on the Laue group: the three primaries blended linearly over the polar
  angle and the azimuth of the fundamental sector

    key = HKLDirectionKey(cs)"""

  def __init__(self, CS):
    super().__init__(CS.Laue())
    V = self.sR.vertices().data.reshape(-1, 3)
    theta = np.arccos(np.clip(V[:, 2], -1, 1)) if len(V) else np.zeros(0)
    rho = np.array([np.arctan2(v[1], v[0]) for v in V if abs(v[2]) < 1 - 1e-6])
    self.maxTheta = float(np.max(theta)) if len(theta) else np.pi / 2
    self.rhoRange = (float(np.min(rho)), float(np.max(rho))) if len(rho) else (0.0, 2 * np.pi)

  @property
  def whiteCenter(self):
    """the centre of the fundamental sector"""
    return vector3d(regionCenter(self.sR), isNormalized=True)

  def direction2color(self, h, gray=None):
    """the colours of directions, red at the pole of the sector, green and blue along its two edges"""
    xyz, shape = _asDirections(h)
    unit = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    hs = projectIntoRegion(vector3d(unit, frame=self.sR.frame, isNormalized=True), self.sR, self.sym).data.reshape(-1, 3)
    theta, rho = np.arccos(np.clip(hs[:, 2], -1, 1)), np.arctan2(hs[:, 1], hs[:, 0])
    lo, hi = self.rhoRange
    if (self.sym.numSym() <= 2 and len(self.sR.N) == 0) or (self.maxTheta >= np.pi / 2 - 1e-9 and hi - lo >= 2 * np.pi - 1e-9):
      # the triclinic Laue group: three lobes of the upper hemisphere, as MTEX draws it
      m = 2 * np.pi / 3
      rr = np.mod(rho + m + np.pi, 2 * np.pi) - np.pi
      br = np.mod(rho - m + np.pi, 2 * np.pi) - np.pi
      nt = np.abs(np.pi / 2 - theta) / (np.pi / 2)
      rgb = np.stack([np.maximum(0, (1 - nt) * (m - np.abs(rr)) / m), np.maximum(0, (1 - nt) * (m - np.abs(rho)) / m),
                      np.maximum(0, (1 - nt) * (m - np.abs(br)) / m)], axis=-1)
    else:
      t = theta / self.maxTheta
      rgb = np.stack([np.clip(1 - t, 0, 1), np.clip(t * (hi - rho) / (hi - lo), 0, 1), np.clip(t * (rho - lo) / (hi - lo), 0, 1)], axis=-1)
    return rgb.reshape(shape + (3,))


class spotDirectionKey(directionColorKey):
  """each centre, a crystal direction, given its colour, fading to white with the kernel of
  the angular distance to the nearest symmetric equivalent; several centres multiply

    spotDirectionKey(cs, center=Miller(1, 1, 1, cs), color=(0, 0, 1), psi=S2DeLaValleePoussinKernel(halfwidth=7.5*degree))
  """

  def __init__(self, CS, center=None, color=(1.0, 0.0, 0.0), psi=None):
    super().__init__(CS)
    from ..functions.kernelfunctions import S2DeLaValleePoussinKernel
    self.psi = S2DeLaValleePoussinKernel(halfwidth=10 * degree) if psi is None else psi
    self.color = np.asarray(color, dtype=float).reshape(-1, 3)
    self.frame = CS if isinstance(CS, referenceFrame) else None
    self.center = vector3d(0, 0, 1) if center is None else center
    if center is not None and len(self.color) != len(self.equivalents):
      raise ValueError("one colour per centre")

  @property
  def center(self):
    """the centres, with the equivalents under the group kept beside them; set as in MATLAB"""
    return self._center

  @center.setter
  def center(self, value):
    """sets the centres, the equivalents under the group computed beside them"""
    xyz = _asDirections(value)[0]
    self._center = vector3d(xyz, frame=self.frame)
    els = self.sym.rot.flatten().data
    self.equivalents = [qrot(els[:, :4], (c / np.linalg.norm(c))[None]) * els[:, 4:5] for c in xyz]

  def direction2color(self, h, gray=None):
    """the colours of directions: white, multiplied for every centre by its colour faded with the
    kernel of the angle to the nearest equivalent of the centre"""
    xyz, shape = _asDirections(h)
    unit = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    psi0 = float(self.psi(1.0))
    rgb = np.ones((len(unit), 3))
    color = np.asarray(self.color, dtype=float).reshape(-1, 3)
    if len(color) != len(self.equivalents):
      raise ValueError("one colour per centre")
    for eq, c in zip(self.equivalents, color):
      w = np.asarray(self.psi(np.clip(np.max(unit @ eq.T, axis=1), -1, 1)), dtype=float) / psi0
      hsv = rgb2hsv(c)
      # black has no hue to keep, so it fades through the grey of its weight
      f = np.stack([1 - w] * 3, axis=-1) if np.all(c == 0) else hsv2rgb(hsv[0], w * hsv[1], 1.0)
      rgb = rgb * f
    return rgb.reshape(shape + (3,))

  @property
  def whiteCenter(self):
    """the first centre of the key"""
    return self.center[0] if self.center.size > 1 else self.center


__all__ = ["directionColorKey", "HSVDirectionKey", "TSLDirectionKey", "HKLDirectionKey", "spotDirectionKey", "sphereColor",
           "hsv2rgb", "rgb2hsv", "hsl2hsv"]
