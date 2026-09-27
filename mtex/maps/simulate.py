# a map whose answer is known (MATLAB EBSDAnalysis/@simulateEBSD, Julia simulate.jl): a job
# holds the settings and the map at the current stage of the simulation, and every verb
# rewrites that map
#
#   eS = simulateEBSD()
#   eS.xdim, eS.ydim, eS.stepSize         the map over 1:stepSize:xdim by 1:stepSize:ydim
#   eS.CS, eS.ori0                        the crystal frame and the orientation of every measurement
#   eS.makeMap()                          the map of ori0, eS.EBSDsim
#   eS.addnoise()                         noise of eS.noiseFun ('uniform', 'logn') up to eS.noiseMax
#   eS.addFeature_singleStep()            a step of eS.mori_angle about eS.axS
#   eS.addFeature_simpleGradient()        a gradient of eS.mori_angle per step along eS.gradDir
#   eS.addFeature_circularSubgrain()      a disc turned by eS.mori_angle about eS.axS
from __future__ import annotations

import numpy as np

from ..geometry.frames import crystalFrame
from ..geometry.kernels import qmul
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from ..geometry.display import aligned


# the rotation of every measurement of a mask turned on the specimen side by the axis angle
# vectors v, one or one per measurement: exp(ori, v, leftVector)
def _turn(ebsd, mask, v):
  q = ebsd.rotations.data.reshape(-1, 4)
  k = np.flatnonzero(np.asarray(mask).reshape(-1))
  v = np.broadcast_to(np.asarray(v, dtype=float).reshape(-1, 3), (len(k), 3))
  w = np.linalg.norm(v, axis=1)
  axis = np.where(w[:, None] > 0, v / np.where(w > 0, w, 1)[:, None], [0.0, 0.0, 1.0])
  r = np.concatenate([np.cos(w / 2)[:, None], np.sin(w / 2)[:, None] * axis], axis=1)
  q[k] = qmul(r, q[k])


class simulateEBSD:
  """a simulated map whose answer is known: a job holding the settings, the grid xdim, ydim and
  stepSize, the orientations ori0, the noise and the features, and the map EBSDsim at the current
  stage, which every verb rewrites

    eS = simulateEBSD()
    eS.makeMap()
    eS.addFeature_simpleGradient()
    eS.addnoise()"""

  def __init__(self):
    self.xdim = 100
    self.ydim = 100
    self.stepSize = 1
    self.domainID = None
    self.noiseFun = None
    self.noiseMax = 0.05 * degree
    self.EBSDsim = None
    self.axS = vector3d(0.0, 1.0, 0.0)
    self.gradDir = vector3d(1.0, 0.0, 0.0)
    self.mori_angle = 0.0017
    self.ori0 = orientation.id(crystalFrame("1", mineral="kryptonite"))
    # the generator of the noise; None draws a fresh one, as MATLAB's rand does
    self.rng = None

  def __repr__(self):
    fmt = lambda v: "(" + ", ".join(f"{x:g}" for x in np.round(np.asarray(v.data, dtype=float).reshape(-1)[:3], 4) + 0.0) + ")"
    rows = [("xdim", f"{self.xdim:g}"), ("ydim", f"{self.ydim:g}"), ("stepSize", f"{self.stepSize:g}"),
            ("CS", f"{self.CS.pointGroup} ({self.CS.mineral})"), ("noiseFun", repr(self.noiseFun)),
            ("noiseMax", f"{self.noiseMax / degree:g}°"), ("axS", fmt(self.axS)), ("gradDir", fmt(self.gradDir)),
            ("mori_angle", f"{self.mori_angle / degree:g}°"),
            ("EBSDsim", "none yet" if self.EBSDsim is None else f"{self.EBSDsim.size} measurements")]
    w = max(len(k) for k, _ in rows)
    return aligned("simulateEBSD\n" + "\n".join(f"  {k:<{w}}  {v}" for k, v in rows))

  @property
  def CS(self):
    """the crystal frame is the one of ori0"""
    return self.ori0.CS

  @CS.setter
  def CS(self, cs):
    """sets the crystal frame, ori0 kept as rotations in the new frame"""
    self.ori0 = orientation(rotation(self.ori0.data), cs)

  def makeMap(self):
    """the map of ori0 over the grid, every measurement of the one phase, gridded; one ori0
    per pixel is taken in MATLAB's order of the list, column by column"""
    from .ebsd import EBSD
    from .phaselist import phaseList
    xs = np.arange(1, self.xdim + self.stepSize / 2, self.stepSize, dtype=float)
    ys = np.arange(1, self.ydim + self.stepSize / 2, self.stepSize, dtype=float)
    x, y = np.meshgrid(xs, ys)
    n = x.size
    q = np.asarray(self.ori0.data, dtype=float).reshape(-1, 4)
    if len(q) == 1:
      q = np.tile(q, (n, 1))
    elif len(q) != n:
      raise ValueError(f"ori0 is one orientation or one for each of the {n} pixels")
    pos = np.stack([x.reshape(-1, order="F"), y.reshape(-1, order="F"), np.zeros(n)], axis=1)
    self.EBSDsim = EBSD(pos, rotation(q), np.ones(n, dtype=np.int8), phaseList(self.CS),
                        prop={"emptyProp": np.ones(n)}).gridify()
    self.domainID = None
    return self

  def addnoise(self):
    """every orientation turned on the specimen side about a random axis by a random angle:
    uniform up to noiseMax, or lognormal scaled so that the largest is noiseMax"""
    e = self.EBSDsim
    n = e.size
    rng = np.random.default_rng() if self.rng is None else self.rng
    axes = vector3d.rand(n, rng=rng).data.reshape(-1, 3)
    if self.noiseFun in ("uniform", "uniForm"):
      w = self.noiseMax * rng.random(n)
    elif self.noiseFun in ("lognorm", "logn"):
      w = np.exp(rng.standard_normal(n))
      w = w / w.max() * self.noiseMax
    else:
      raise ValueError(f"the noise is 'uniform' or 'logn', not {self.noiseFun!r}")
    _turn(e, np.ones(n, dtype=bool), axes * w[:, None])
    return self

  def _axis(self):
    return np.asarray(self.axS.normalize().data, dtype=float).reshape(3)

  def addFeature_singleStep(self):
    """the orientations of the right third and the upper half of the middle third turned by
    mori_angle about axS: a known low angle boundary along the edge of that domain"""
    X, Y = self.xdim, self.ydim
    poly = np.array([[X / 3, 0], [X / 3, Y / 2], [2 * X / 3, Y / 2], [2 * X / 3, Y], [X, Y], [X, 0]], dtype=float)
    self.domainID = self.EBSDsim.inpolygon(poly)
    _turn(self.EBSDsim, self.domainID, self._axis() * self.mori_angle)
    return self

  def addFeature_circularSubgrain(self):
    """the orientations of a disc at the centre, of a quarter of the smaller extent, turned by
    mori_angle about axS"""
    c = np.array([self.xdim / 2, self.ydim / 2])
    r = c.min() / 2
    t = np.linspace(0, 2 * np.pi, 360)
    self.domainID = self.EBSDsim.inpolygon(np.stack([np.cos(t) * r, np.sin(t) * r], axis=1) + c)
    _turn(self.EBSDsim, self.domainID, self._axis() * self.mori_angle)
    return self

  def addFeature_simpleGradient(self):
    """every orientation turned about axS by mori_angle for every step of its position along
    gradDir, the direction squared componentwise as MATLAB's updateOri takes it; an axis
    given as a crystal direction is carried into the specimen by every orientation"""
    e = self.EBSDsim
    q = e.rotations.data.reshape(-1, 4)
    if isinstance(getattr(self.axS, "frame", None), crystalFrame):
      ax = (orientation(rotation(q), self.CS, e.frame) * self.axS.normalize()).data.reshape(-1, 3)
    else:
      ax = np.broadcast_to(self._axis(), (len(q), 3))
    g = np.asarray(self.gradDir.normalize().data, dtype=float).reshape(3)
    P = e.pos.data.reshape(-1, 3) / e.dPos
    step = np.linalg.norm(P * g ** 2, axis=1)
    _turn(e, np.ones(len(q), dtype=bool), -step[:, None] * ax * self.mori_angle)
    return self


__all__ = ["simulateEBSD"]
