# orientation colour keys: through an inverse pole figure an orientation becomes a colour
# (MATLAB plotting/orientationColorKeys; docs/plans/plotting.md S3)
#
# The crystal direction that points along a fixed specimen direction is what a direction key
# colours, so the colour says which crystal axis lies along that direction of the specimen;
# the keys of the whole orientation colour its axis and angle from a reference, its Euler
# angles, or its distance to a few centres. A map holds a million orientations, so the key
# of its crystal frame is evaluated once on a grid and looked up afterwards.
from __future__ import annotations

import threading

import numpy as np

from ..geometry.frames import crystalFrame, referenceFrame, specimenFrame
from ..geometry.grids.s2grid import equispacedS2Grid
from ..geometry.kernels import qrot
from ..geometry.orientation import orientation
from ..geometry.orientationregion import eulerRegion
from ..geometry.quaternion import rotation
from ..geometry.representations import quat2euler
from ..geometry.units import degree
from ..geometry.vector3d import vector3d, zvector
from .directionkeys import HKLDirectionKey, HSVDirectionKey, TSLDirectionKey, hsv2rgb, rgb2hsv, spotDirectionKey


def _framesOf(x, CS2=None):
  if isinstance(x, orientation):
    return x.frameA, x.frameB, x.antipodal
  if hasattr(x, "CSList") and hasattr(x, "phaseId"):
    return x.CS, x.frame, False
  return x, specimenFrame.default if CS2 is None else CS2, False


def _conj(q):
  return q * np.array([1.0, -1.0, -1.0, -1.0])


class orientationColorKey:
  """a colour key turning orientations into colours

    rgb = key.orientation2color(ori)    an (..., 3) array in [0, 1]
    key.CS1, key.CS2, key.antipodal
  """

  def __init__(self, x, CS2=None):
    self.CS1, self.CS2, self.antipodal = _framesOf(x, CS2)

  @property
  def CS(self):
    return self.CS1

  @property
  def SS(self):
    return self.CS2

  def orientation2color(self, ori):
    raise NotImplementedError

  def __repr__(self):
    return f"{type(self).__name__} ({self.CS1.char()} → {self.CS2.char()})"


# ------------------------------------------------------------------ the inverse pole figure keys

class ipfColorKey(orientationColorKey):
  """an orientation coloured by the crystal direction it carries along a specimen direction

    ipfColorKey(cs), ipfColorKey(ebsd), ipfColorKey(ori)
    key.ipfDirection = xvector
    rgb = key.orientation2color(ori)

  dirMap is the direction key doing the colouring, HSVDirectionKey unless given.
  """

  def __init__(self, x, ipfDirection=None, dirMap=None):
    super().__init__(x)
    self.ipfDirection = vector3d(zvector if ipfDirection is None else ipfDirection)
    self.dirMap = HSVDirectionKey(self.CS1) if dirMap is None else dirMap

  # the crystal directions along the specimen direction of every orientation
  def crystalDirection(self, ori):
    q = ori.data.reshape(-1, 4)
    d = self.ipfDirection.normalize().data.reshape(3)
    return vector3d(qrot(_conj(q), np.tile(d, (len(q), 1))).reshape(ori.shape + (3,)), frame=self.CS1, isNormalized=True)

  # an orientation that was never measured has no colour, and a map shows nothing for it
  def orientation2color(self, ori):
    rgb = self.dirMap.direction2color(self.crystalDirection(ori))
    rgb[~np.all(np.isfinite(ori.data[..., :4]), axis=-1)] = np.nan
    return rgb

  @property
  def whiteCenter(self):
    return self.dirMap.whiteCenter

  def plot(self, **kwargs):
    from .sphericalplot import plotKey
    return plotKey(self.dirMap, **kwargs)


class ipfHSVKey(ipfColorKey):
  """the MTEX default inverse pole figure key, white at the centre of the sector

    ipfHSVKey(cs), ipfHSVKey(cs, maxAngle=7.5*degree, whiteCenter=v, ipfDirection=xvector)
  """

  def __init__(self, x, ipfDirection=None, **kwargs):
    CS1 = _framesOf(x)[0]
    super().__init__(x, ipfDirection, HSVDirectionKey(CS1, **kwargs))

  # the sharpness and the white centre live on the direction key and are set through it
  @property
  def maxAngle(self):
    return self.dirMap.maxAngle

  @maxAngle.setter
  def maxAngle(self, value):
    self.dirMap.maxAngle = float(value)

  @property
  def whiteCenter(self):
    return self.dirMap.whiteCenter

  @whiteCenter.setter
  def whiteCenter(self, value):
    self.dirMap.whiteCenter = vector3d(value).normalize()


class ipfTSLKey(ipfColorKey):

  def __init__(self, x, ipfDirection=None):
    super().__init__(x, ipfDirection, TSLDirectionKey(_framesOf(x)[0]))


class ipfHKLKey(ipfColorKey):

  def __init__(self, x, ipfDirection=None):
    super().__init__(x, ipfDirection, HKLDirectionKey(_framesOf(x)[0]))


class ipfSpotKey(ipfColorKey):
  """single spots of the inverse pole figure in their own colours

    ipfSpotKey(cs, center=Miller(1, 0, 0, cs), color=(0, 0, 1), psi=S2DeLaValleePoussinKernel(halfwidth=7.5*degree))
  """

  def __init__(self, x, ipfDirection=None, center=None, color=(1.0, 0.0, 0.0), psi=None):
    super().__init__(x, ipfDirection, spotDirectionKey(_framesOf(x)[0], center, color, psi))

  @property
  def center(self):
    return self.dirMap.center

  @center.setter
  def center(self, value):
    self.dirMap.center = value

  @property
  def color(self):
    return self.dirMap.color

  @color.setter
  def color(self, value):
    self.dirMap.color = np.asarray(value, dtype=float).reshape(-1, 3)

  @property
  def psi(self):
    return self.dirMap.psi

  @psi.setter
  def psi(self, value):
    self.dirMap.psi = value


# ------------------------------------------------------------------ keys of the whole orientation

class axisAngleColorKey(orientationColorKey):
  """the rotation from a reference orientation as a colour: its axis in the specimen frame
  through the key of the whole sphere, its angle through the saturation (Thomsen et al.)

    axisAngleColorKey(CS1, CS2, oriRef=..., maxAngle=..., thresholdAngle=...)
    rgb = key.orientation2color(ori)
    rgb = key.orientation2color(ori, oriRef)

  The saturation is full at maxAngle, by default the 80th percentile of the angles, and the
  colour is NaN above thresholdAngle. Without a reference the identity is the reference.
  """

  def __init__(self, x=None, CS2=None, oriRef=None, maxAngle=None, thresholdAngle=np.inf):
    # without a frame the key reads the axes in the specimen and needs no crystal
    super().__init__(crystalFrame("1") if x is None else x, CS2)
    self.oriRef = oriRef
    self.maxAngle = None if maxAngle is None else float(maxAngle)
    self.thresholdAngle = float(thresholdAngle)
    self.dirMapping = HSVDirectionKey(self.CS2)

  def orientation2color(self, ori, oriRef=None):
    ref = self.oriRef if oriRef is None else oriRef
    if ref is None:
      omega = np.asarray(ori.angle()).reshape(-1)
      axes = ori.axis().data.reshape(-1, 3)
    else:
      if ref.size not in (1, ori.size):
        raise ValueError("one reference orientation, or one per orientation")
      r = ref if ref.size == ori.size else ref._new(np.broadcast_to(ref.data.reshape(4), ori.shape + (4,)).copy())
      omega = np.asarray(ori.angle(r)).reshape(-1)
      axes = ori.axis(r).data.reshape(-1, 3)
    finite = np.isfinite(omega)
    maxAngle = self.maxAngle if self.maxAngle is not None else (float(np.quantile(omega[finite], 0.8)) if np.any(finite) else 1.0)
    gray = np.minimum(1.0, omega / max(maxAngle, 1e-12))
    rgb = self.dirMapping.direction2color(vector3d(axes), gray=gray)
    out = np.where(((omega > self.thresholdAngle) | ~finite)[:, None], np.nan, rgb)
    return out.reshape(ori.shape + (3,))


class BungeColorKey(orientationColorKey):
  """the three Euler angles of the representative inside the Euler box, scaled to red, green
  and blue

    BungeColorKey(cs, ss)
  """

  def __init__(self, x, CS2=None):
    super().__init__(x, CS2)
    self.phi1Range, self.PhiRange, self.phi2Range = eulerRegion(self.CS1, self.CS2)

  def orientation2color(self, ori):
    es = ori.flatten().symmetrise()
    q = es.data.reshape(es.shape[0], -1, 4)
    a, b, c = quat2euler(q, "Bunge")
    m1, m2, m3 = self.phi1Range, self.PhiRange, self.phi2Range
    excess = np.maximum(a - m1, 0) + np.maximum(b - m2, 0) + np.maximum(c - m3, 0)
    k = np.argmin(excess, axis=0)
    idx = (k, np.arange(q.shape[1]))
    rgb = np.stack([np.clip(a[idx] / m1, 0, 1), np.clip(b[idx] / m2, 0, 1), np.clip(c[idx] / m3, 0, 1)], axis=-1)
    return rgb.reshape(ori.shape + (3,))


class spotColorKey(orientationColorKey):
  """each centre orientation given its colour, fading to white with the kernel of the
  disorientation to it; several centres multiply

    spotColorKey(center=ori, color=(0, 0, 1), psi=SO3DeLaValleePoussinKernel(halfwidth=10*degree))
  """

  # spotColorKey(ori) centres the key on the orientations given; spotColorKey(ebsd) names
  # the phase, the centre, the colour and the kernel being set afterwards as in MATLAB
  def __init__(self, x, center=None, color=None, psi=None):
    super().__init__(x)
    from ..functions.kernelfunctions import SO3DeLaValleePoussinKernel
    # spotColorKey(ori, color): the second argument is the colour when it is no orientation
    if center is not None and not isinstance(center, orientation):
      center, color = None, center
    if center is None and isinstance(x, orientation):
      center = x
    self.center = None if center is None else center.flatten()
    self.color = np.asarray((1.0, 0.0, 0.0) if color is None else color, dtype=float).reshape(-1, 3)
    self.psi = SO3DeLaValleePoussinKernel(halfwidth=10 * degree) if psi is None else psi

  def orientation2color(self, ori):
    if self.center is None:
      raise ValueError("the key has no centre yet; set key.center to an orientation")
    center = self.center.flatten()
    color = np.asarray(self.color, dtype=float).reshape(-1, 3)
    if len(color) != center.size:
      raise ValueError("one colour per centre")
    flat = ori.flatten()
    psi0 = float(self.psi(1.0))
    rgb = np.ones((flat.size, 3))
    for j in range(center.size):
      omega = np.asarray(flat.angle(center[j])).reshape(-1)
      w = np.asarray(self.psi(np.cos(omega / 2)), dtype=float).reshape(-1) / psi0
      hsv = rgb2hsv(color[j])
      rgb = rgb * hsv2rgb(hsv[0], w * hsv[1], 1.0)
    return rgb.reshape(ori.shape + (3,))


class spectralTransmissionColorKey(orientationColorKey):
  """the colour of a thin section of an anisotropic crystal between polariser and analyser:
  light along the propagation direction of the specimen through a plate of the given
  thickness in nanometres; no polariser is circular polarisation

    key = spectralTransmissionColorKey(rI, thickness)
    key.propagationDirection, key.polarizer, key.phi
  """

  def __init__(self, rI, thickness):
    super().__init__(rI.frame)
    self.rI = rI
    self.thickness = thickness
    self.propagationDirection = vector3d(0, 0, 1)
    self.polarizer = vector3d(1, 0, 0)
    self.phi = 90 * degree

  def orientation2color(self, ori):
    from ..analysis.optics import spectralTransmission
    back = ori.inv()
    prop = back * self.propagationDirection
    if self.polarizer is None:
      return spectralTransmission(self.rI, prop, self.thickness, phi=self.phi)
    return spectralTransmission(self.rI, prop, self.thickness, polarizationDirection=back * self.polarizer, phi=self.phi)


# ------------------------------------------------------------------ the key on a grid

# the colour of a direction changes smoothly over the sphere, so the key can be evaluated
# once on an equispaced grid and looked up afterwards; the lookup absorbs the symmetry,
# where the exact route expands every direction over the group to bring it into the sector
COLOR_GRID_RESOLUTION = 0.5 * degree
COLOR_NAN = np.array([1.0, 1.0, 1.0])


class colorGrid:

  def __init__(self, key, resolution=COLOR_GRID_RESOLUTION):
    self.key = key.dirMap if isinstance(key, ipfColorKey) else key
    self.grid = equispacedS2Grid(resolution=resolution)
    self.colors = self.key.direction2color(vector3d(self.grid.data, isNormalized=True)).reshape(-1, 3)
    self.stepwise = self.grid.hasUniformSteps
    self.resolution = resolution
    # the first node, step and count of every ring, what the arithmetic look-up needs
    g = self.grid
    rings = g.rhoGrid
    self.ladders = (float(g.thetaGrid.points[0]), float(g.thetaGrid.step), np.array([r.points[0] for r in rings], dtype=float),
                    np.array([r.step if len(r) > 1 else 1.0 for r in rings], dtype=float), np.array([len(r) for r in rings], dtype=np.int64),
                    np.asarray(g.offset, dtype=np.int64)) if self.stepwise else None

  def __len__(self):
    return len(self.colors)

  # a NaN or zero direction, the pole of an unmeasured orientation, has no grid node and is
  # coloured white, as MTEX greys such a cell rather than failing the map
  def direction2color(self, h):
    xyz = (h.data if isinstance(h, vector3d) else np.asarray(h, dtype=float)).reshape(-1, 3)
    ok = np.all(np.isfinite(xyz), axis=-1) & (np.linalg.norm(xyz, axis=-1) > 0)
    out = np.tile(COLOR_NAN, (len(xyz), 1))
    if np.any(ok):
      v = vector3d(xyz[ok], isNormalized=False)
      ind = self.grid.nearestStepwise(v) if self.stepwise else self.grid.find(v)[0]
      out[ok] = self.colors[np.asarray(ind).reshape(-1)]
    return out.reshape(np.shape(h)[:-1] + (3,) if not isinstance(h, vector3d) else h.shape + (3,))


_GRIDS = {}
_GRID_LOCK = threading.RLock()


# the grid held by the frame's identity: frames are interned, so the same frame brings the
# same grid, and the grid depends on the sector alone, not on the specimen direction an
# inverse pole figure is taken of
def keyColorGrid(cs, resolution=COLOR_GRID_RESOLUTION):
  with _GRID_LOCK:
    cg = _GRIDS.get(id(cs))
    if cg is None or cg[0] is not cs or abs(cg[1].resolution - resolution) > 1e-6 * resolution:
      cg = (cs, colorGrid(HSVDirectionKey(cs), resolution))
      _GRIDS[id(cs)] = cg
    return cg[1]


# the colour of every measurement of a map, or of every grain, each phase by its own inverse
# pole figure key; what is not indexed takes the background colour. The map goes through the
# grid of the key unless exact=True colours every measurement through the key itself; the
# grid route is one compiled loop over the quaternions where numba is there
def ipfColors(x, ipfDirection=None, unindexed=(1.0, 1.0, 1.0), exact=False, resolution=COLOR_GRID_RESOLUTION, at=None):
  from .kernels import ipfGridColors
  d = vector3d(zvector if ipfDirection is None else ipfDirection).normalize().data.reshape(3)
  rots = x.meanRotation if hasattr(x, "meanRotation") else x.rotations
  q = rots.data.reshape(-1, 4)
  phaseId = np.asarray(x.phaseId).reshape(-1)
  # the measurements a picture shows are fewer than the map holds, and colouring those alone
  # is what a map far larger than its picture costs
  if at is not None:
    at = np.asarray(at).reshape(-1)
    q, phaseId = np.ascontiguousarray(q[at]), phaseId[at]
  out = np.empty((len(q), 3))
  out[:] = np.asarray(unindexed, dtype=float)
  for i in x.CSList.indexedPhaseId:
    sub = np.flatnonzero(phaseId == i)
    if len(sub) == 0:
      continue
    cs = x.CSList[i]
    key = HSVDirectionKey(cs) if exact else keyColorGrid(cs, resolution)
    if exact or key.ladders is None or not ipfGridColors(q, sub, d, key.ladders, key.colors, out):
      dirs = qrot(_conj(q[sub]), np.tile(d, (len(sub), 1)))
      out[sub] = key.direction2color(vector3d(dirs, frame=cs)).reshape(-1, 3)
  return out if at is not None else out.reshape(rots.shape + (3,))


__all__ = ["orientationColorKey", "ipfColorKey", "ipfHSVKey", "ipfTSLKey", "ipfHKLKey", "ipfSpotKey", "axisAngleColorKey",
           "BungeColorKey", "spotColorKey", "spectralTransmissionColorKey", "colorGrid", "ipfColors"]
