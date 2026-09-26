# the interface to Neper, the tessellation program (MATLAB @neper, ../master)
#
#   np = neper.init()                             the one interface, with its settings
#   np.geometry = 'cube(100,100,100)'; np.morpho = 'graingrowth'; np.iterMax = 1000
#   grains = np.simulateGrains(ori)               one grain per orientation
#   grains = np.simulateGrains(1000, odf)         the orientations sampled from an ODF
#   grains = np.simulateTwinGrains(n, mori, odf, bndA, width=w)   lamellae with twins
#   grains = np.simulateChildGrains(n, p2c, odfParent, habitPlane=h, width=w)
#
# The tessellation is written to np.filePath and read back with grain3d.load. The command is
# the preference 'neperCommand' ('neper' by default); np.env adds to its environment, for
# instance LD_LIBRARY_PATH for a Neper built against another library than the one installed.
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from ..geometry.session import getMTEXpref


class neper:

  _instance = None

  def __init__(self):
    self.id = 1
    self.geometry = "cube(1,1,1)"
    self.morpho = "graingrowth"
    self.iterMax = 1000
    self.varNeperopts = ""
    self.fileName2d = "2dslice"
    self.fileName3d = "allgrains"
    self.filePath = os.path.join(tempfile.gettempdir(), "neper")
    self.env = {}

  @classmethod
  def instance(cls):
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  @classmethod
  def init(cls):
    return cls.instance()

  @property
  def command(self):
    return getMTEXpref("neperCommand", "neper")

  # whether the command runs
  def available(self):
    try:
      return self._run([self.command, "--version"], check=False).returncode == 0
    except OSError:
      return False

  def _run(self, args, silent=True, check=True):
    os.makedirs(self.filePath, exist_ok=True)
    env = dict(os.environ, **{k: str(v) for k, v in self.env.items()})
    out = subprocess.run(args, env=env, capture_output=True, text=True)
    with open(os.path.join(self.filePath, "neper.log"), "a") as fh:
      fh.write(out.stdout + out.stderr)
    if not silent:
      print(out.stdout)
    if check and out.returncode != 0:
      raise RuntimeError(f"neper failed: {out.stderr.strip()[-500:]}")
    return out

  def _morpho(self, aspectRatio=None):
    m = str(self.morpho)
    if aspectRatio is not None:
      m += ",aspratio (" + ",".join(f"{a:g}" for a in np.atleast_1d(aspectRatio)) + ")"
    return m

  def _tessellate(self, n, morpho, silent):
    base = os.path.join(self.filePath, self.fileName3d)
    args = [self.command, "-T", "-n", str(n), "-id", str(self.id), "-morpho", morpho, "-morphooptistop", f"iter={self.iterMax}",
            "-domain", str(self.geometry), "-o", base]
    if self.varNeperopts:
      args += str(self.varNeperopts).split()
    self._run(args, silent)
    return base + ".tess"

  # the grains of a tessellation with one grain per orientation, or n orientations of an ODF
  def simulateGrains(self, ori, odf=None, silent=True, aspectRatio=None):
    from .grain3d import grain3d
    if np.isscalar(ori):
      ori = odf.discreteSample(int(ori))
    tess = self._tessellate(ori.size, self._morpho(aspectRatio), silent)
    grains = grain3d.load(tess, CS=ori.CS)
    grains.meanOrientation = ori.flatten()
    return grains

  # the lamellae of a tessellation: every cell cut into slabs normal to the directions given
  def _lamellae(self, n, normals, width, silent, aspectRatio):
    lam = os.path.join(self.filePath, "lamnormals.txt")
    os.makedirs(self.filePath, exist_ok=True)
    xyz = normals.data.reshape(-1, 3)
    np.savetxt(lam, np.column_stack([np.arange(1, n + 1), xyz]), fmt=["%d", "%f", "%f", "%f"])
    base = os.path.join(self.filePath, self.fileName3d)
    morpho = self._morpho(aspectRatio) + f"::lamellar(w={width:g},v=msfile({lam}))"
    args = [self.command, "-T", "-n", f"{n}::from_morpho", "-id", str(self.id), "-morpho", morpho, "-morphooptistop",
            f"iter={self.iterMax}", "-domain", str(self.geometry), "-o", base]
    self._run(args, silent)
    return base + ".tess"

  # how many cells every host was cut into: the cells come host by host, and the cells of a
  # host end with the first one none of whose faces is parallel to the host's lamellae, the
  # last host taking what is left. MATLAB searches in chunks of 50 cells and fails for a host
  # of more lamellae; this search has no such window
  @staticmethod
  def _cellsPerHost(grains, normals, nHosts):
    N = grains.boundary.N.data
    n = normals.data.reshape(-1, 3)
    I = grains.I_GF.tocsr()
    counts = np.zeros(nHosts, dtype=np.int64)
    host, start = 0, 0
    for k in range(len(grains)):
      if host == nHosts - 1:
        break
      faces = I.indices[I.indptr[k]:I.indptr[k + 1]]
      if np.all(np.abs(N[faces] @ n[host]) < np.cos(1e-3)):
        counts[host] = k + 1 - start
        start, host = k + 1, host + 1
    counts[host] = len(grains) - start
    return counts

  # twinned lamellae: every second lamella of a host turned by the twin misorientation
  def simulateTwinGrains(self, numGrains, mori, odf, bndA, width=None, silent=True, aspectRatio=None, rng=None):
    from ..geometry.quaternion import rotation
    from .grain3d import grain3d
    rng = np.random.default_rng() if rng is None else rng
    ori = odf.discreteSample(numGrains)
    nA = ori * bndA.discreteSample(numGrains)
    if width is None:
      size = _cubeSize(self.geometry)
      width = (np.prod(size) / numGrains) ** (1 / 3) / 4
    grains = grain3d.load(self._lamellae(numGrains, nA, width, silent, aspectRatio))
    hostId = np.repeat(np.arange(numGrains), self._cellsPerHost(grains, nA, numGrains))
    from ..geometry.kernels import qmul
    q = ori.data.reshape(-1, 4)[hostId].copy()
    q[1::2] = qmul(q[1::2], np.broadcast_to(mori.data.reshape(4), q[1::2].shape))
    q = qmul(rotation.rand(len(q), maxAngle=2 * np.pi / 180, rng=rng).data.reshape(-1, 4), q)
    o = ori.flatten()[np.zeros(len(q), dtype=np.int64)]
    o.data[...] = q.reshape(o.data.shape)
    grains.meanOrientation = o
    return grains

  # child lamellae in parent grains: the children of a parent in variants that alternate
  def simulateChildGrains(self, numParents, p2c, odfParent, habitPlane=None, width=0.015, silent=True, aspectRatio=None, rng=None):
    from ..geometry.miller import Miller
    from ..geometry.quaternion import rotation
    from .grain3d import grain3d
    rng = np.random.default_rng() if rng is None else rng
    if p2c.size == 1:
      p2c = p2c.variants()[:6]
    oriP = odfParent.discreteSample(numParents)
    h = Miller(1, 1, 1, p2c.CS) if habitPlane is None else habitPlane
    nA = oriP * h.normalize()
    grains = grain3d.load(self._lamellae(numParents, nA, width, silent, aspectRatio))
    pId = np.repeat(np.arange(numParents), self._cellsPerHost(grains, nA, numParents))
    N = p2c.size
    vId = np.zeros(len(grains), dtype=np.int64)
    vId[0] = rng.integers(N)
    for k in range(1, len(grains)):
      r = rng.integers(N - 1)
      vId[k] = r + (r >= vId[k - 1])
    ori = rotation.rand(len(grains), maxAngle=2 * np.pi / 180, rng=rng) * oriP.flatten()[pId] * p2c.flatten()[vId].inv()
    grains.meanOrientation = ori
    return grains


def _cubeSize(geometry):
  import re
  m = re.fullmatch(r"\s*cube\(([^)]*)\)\s*", str(geometry))
  return np.array([float(x) for x in m.group(1).split(",")]) if m else np.ones(3)


__all__ = ["neper"]
