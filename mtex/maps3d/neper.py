# the interface to Neper, the tessellation program (MATLAB @neper, ../master)
#
#   np = neper.init()                             the one interface, with its settings
#   np.geometry = 'cube(100,100,100)'; np.morpho = 'graingrowth'; np.maxIter = 1000
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
from ..geometry.display import display


class neper:
  """the interface to Neper, the tessellation program, one per session with its settings: the
  geometry, the morphology, the iterations and the file path the tessellation is written to

    nep = neper.init()
    grains = nep.simulateGrains(1000, odf)"""

  _instance = None

  def __init__(self):
    self.id = 1
    self.geometry = "cube(1,1,1)"
    self.morpho = "graingrowth"
    self.maxIter = 1000
    self.varNeperopts = ""
    self.fileName2d = "2dslice"
    self.fileName3d = "allgrains"
    self.filePath = os.path.join(tempfile.gettempdir(), "neper")
    self.env = {}

  @classmethod
  def instance(cls):
    """the one interface of the session, created on first call"""
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  @classmethod
  def init(cls):
    """the one interface of the session, neper.instance()"""
    return cls.instance()

  @property
  def command(self):
    """the command that runs Neper, the preference neperCommand, 'neper' by default"""
    return getMTEXpref("neperCommand", "neper")

  def available(self):
    """whether the command runs"""
    try:
      return self._run([self.command, "--version"], check=False).returncode == 0
    except OSError:
      return False

  def _run(self, args, verbose=False, check=True):
    os.makedirs(self.filePath, exist_ok=True)
    env = dict(os.environ, **{k: str(v) for k, v in self.env.items()})
    out = subprocess.run(args, env=env, capture_output=True, text=True)
    with open(os.path.join(self.filePath, "neper.log"), "a") as fh:
      fh.write(out.stdout + out.stderr)
    if verbose:
      print(out.stdout)
    if check and out.returncode != 0:
      raise RuntimeError(f"neper failed: {out.stderr.strip()[-500:]}")
    return out

  def _morpho(self, aspectRatio=None):
    m = str(self.morpho)
    if aspectRatio is not None:
      m += ",aspratio (" + ",".join(f"{a:g}" for a in np.atleast_1d(aspectRatio)) + ")"
    return m

  def _tessellate(self, n, morpho, verbose):
    base = os.path.join(self.filePath, self.fileName3d)
    args = [self.command, "-T", "-n", str(n), "-id", str(self.id), "-morpho", morpho, "-morphooptistop", f"iter={self.maxIter}",
            "-domain", str(self.geometry), "-o", base]
    if self.varNeperopts:
      args += str(self.varNeperopts).split()
    self._run(args, verbose)
    return base + ".tess"

  def simulateGrains(self, ori, odf=None, verbose=False, aspectRatio=None):
    """the grains of a Neper tessellation of the domain, one grain per orientation; MATLAB's
    neper.simulateGrains

      nep = neper.init()                             # the interface, its geometry and morphology set before
      grains = nep.simulateGrains(100, odf)          # 100 grains of orientations sampled from an ODF
      grains = nep.simulateGrains(ori)               # one grain per orientation
      grains = nep.simulateGrains(ori, aspectRatio=[2, 1, 1])  # grains elongated along x

    Neper tessellates the domain nep.geometry (a cube by default) into as many cells as
    there are orientations, with the morphology nep.morpho optimised over nep.maxIter
    iterations from the seed nep.id; the tessellation is written into nep.filePath, read
    back by grain3d.load, and the grains take the orientations in their order. Neper's
    output goes to neper.log in nep.filePath; the command is the preference neperCommand.

    MATLAB calls it on the class, neper.simulateGrains; here it is a method of the interface
    neper.init() returns, and MATLAB's flag 'silent' is verbose=False, the default.

    Input
      ori - orientation, one per grain, or the number of grains
      odf - SO3Fun, with a number of grains: the distribution the orientations are sampled
          from
    Output
      grains - grain3d, with meanOrientation
    Keywords
      verbose - print Neper's output
      aspectRatio - three lengths, the aspect ratio of the grains along x, y and z, added to
          the morphology as Neper's aspratio
    See also
      neper.simulateTwinGrains, neper.simulateChildGrains, grain3d, NeperInterface"""
    from .grain3d import grain3d
    if np.isscalar(ori):
      ori = odf.discreteSample(int(ori))
    tess = self._tessellate(ori.size, self._morpho(aspectRatio), verbose)
    grains = grain3d.load(tess, CS=ori.CS)
    grains.meanOrientation = ori.flatten()
    return grains

  # the lamellae of a tessellation: every cell cut into slabs normal to the directions given
  def _lamellae(self, n, normals, width, verbose, aspectRatio):
    lam = os.path.join(self.filePath, "lamnormals.txt")
    os.makedirs(self.filePath, exist_ok=True)
    xyz = normals.data.reshape(-1, 3)
    np.savetxt(lam, np.column_stack([np.arange(1, n + 1), xyz]), fmt=["%d", "%f", "%f", "%f"])
    base = os.path.join(self.filePath, self.fileName3d)
    morpho = self._morpho(aspectRatio) + f"::lamellar(w={width:g},v=msfile({lam}))"
    args = [self.command, "-T", "-n", f"{n}::from_morpho", "-id", str(self.id), "-morpho", morpho, "-morphooptistop",
            f"iter={self.maxIter}", "-domain", str(self.geometry), "-o", base]
    self._run(args, verbose)
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

  def simulateTwinGrains(self, numGrains, mori, odf, bndA, width=None, verbose=False, aspectRatio=None, rng=None):
    """a microstructure of twinned grains: Neper's grains cut into lamellae, every second
    lamella in twin orientation; MATLAB's neper.simulateTwinGrains

      grains = nep.simulateTwinGrains(100, mori, odf, bndA)            # 100 twinned grains
      grains = nep.simulateTwinGrains(100, mori, odf, bndA, width=0.05)  # lamellae of width 0.05
      grains = nep.simulateTwinGrains(100, mori, odf, bndA, rng=np.random.default_rng(0))  # the same random turns every run

    The orientations of the host grains are sampled from odf, and every host gets a
    lamella normal, a crystal direction sampled from bndA turned by the host's orientation.
    Neper tessellates the domain into numGrains cells and cuts every cell into lamellae of
    the given width normal to its direction. The lamellae take their host's orientation and
    every second one the twin orientation, the host's orientation times mori; every lamella
    is turned at random by up to 2 degree.

    MATLAB calls it on the class, neper.simulateTwinGrains; here it is a method of the
    interface neper.init() returns, and MATLAB's flag 'silent' is verbose=False, the default.

    Input
      numGrains - the number of host grains
      mori - orientation, the twin misorientation from the host to the twin
      odf - SO3Fun, the distribution of the host orientations
      bndA - S2Fun on crystal directions, the distribution of the lamella normals
    Output
      grains - grain3d, every lamella one grain, with meanOrientation
    Keywords
      width - the width of the lamellae, by default a quarter of the edge of a cube of the
          mean grain volume of a cube(...) geometry
      verbose - print Neper's output
      aspectRatio - three lengths, the aspect ratio of the host grains along x, y and z
      rng - numpy.random.Generator of the random turns of the lamellae
    See also
      neper.simulateGrains, neper.simulateChildGrains, NeperInterface"""
    from ..geometry.quaternion import rotation
    from .grain3d import grain3d
    rng = np.random.default_rng() if rng is None else rng
    ori = odf.discreteSample(numGrains)
    nA = ori * bndA.discreteSample(numGrains)
    if width is None:
      size = _cubeSize(self.geometry)
      width = (np.prod(size) / numGrains) ** (1 / 3) / 4
    grains = grain3d.load(self._lamellae(numGrains, nA, width, verbose, aspectRatio))
    hostId = np.repeat(np.arange(numGrains), self._cellsPerHost(grains, nA, numGrains))
    from ..geometry.kernels import qmul
    q = ori.data.reshape(-1, 4)[hostId].copy()
    q[1::2] = qmul(q[1::2], np.broadcast_to(mori.data.reshape(4), q[1::2].shape))
    q = qmul(rotation.rand(len(q), maxAngle=2 * np.pi / 180, rng=rng).data.reshape(-1, 4), q)
    o = ori.flatten()[np.zeros(len(q), dtype=np.int64)]
    o.data[...] = q.reshape(o.data.shape)
    grains.meanOrientation = o
    return grains

  def simulateChildGrains(self, numParents, p2c, odfParent, habitPlane=None, width=0.015, verbose=False, aspectRatio=None, rng=None):
    """a microstructure after a phase transformation: Neper's parent grains cut into lamellae of
    child variants; MATLAB's neper.simulateChildGrains

      grains = nep.simulateChildGrains(10, p2c, odfParent)       # 10 parent grains, habit plane {111}
      grains = nep.simulateChildGrains(10, p2c, odfParent, habitPlane=Miller(1, 1, 0, p2c.CS), width=0.05)  # another habit plane and width
      grains = nep.simulateChildGrains(10, p2c.variants()[:12], odfParent, rng=np.random.default_rng(0))  # the variants to choose from

    The parent orientations are sampled from odfParent, and every parent grain of the
    tessellation is cut into lamellae normal to its habit plane, the plane turned by the
    parent orientation. Every lamella takes a child orientation of its parent, a variant
    chosen at random from those of p2c, but never the variant of the lamella before it, and
    is turned at random by up to 2 degree. A single relationship gives its first six
    variants to choose from.

    MATLAB calls it on the class, neper.simulateChildGrains; here it is a method of the
    interface neper.init() returns, and MATLAB's flag 'silent' is verbose=False, the default.

    Input
      numParents - the number of parent grains
      p2c - orientation, the parent to child orientation relationship, or the variants to
          choose from
      odfParent - SO3Fun, the distribution of the parent orientations
    Output
      grains - grain3d of the child lamellae, with meanOrientation
    Keywords
      habitPlane - Miller of the parent, the plane the lamellae lie in, by default {111}
      width - the width of the lamellae
      verbose - print Neper's output
      aspectRatio - three lengths, the aspect ratio of the parent grains along x, y and z
      rng - numpy.random.Generator of the variants and the random turns
    See also
      neper.simulateGrains, neper.simulateTwinGrains, orientation.variants, NeperInterface"""
    from ..geometry.miller import Miller
    from ..geometry.quaternion import rotation
    from .grain3d import grain3d
    rng = np.random.default_rng() if rng is None else rng
    if p2c.size == 1:
      p2c = p2c.variants()[:6]
    oriP = odfParent.discreteSample(numParents)
    h = Miller(1, 1, 1, p2c.CS) if habitPlane is None else habitPlane
    nA = oriP * h.normalize()
    grains = grain3d.load(self._lamellae(numParents, nA, width, verbose, aspectRatio))
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

  def __repr__(self):
    return display("neper", [("geometry", self.geometry), ("morpho", self.morpho), ("id", self.id), ("filePath", self.filePath)])


def _cubeSize(geometry):
  import re
  m = re.fullmatch(r"\s*cube\(([^)]*)\)\s*", str(geometry))
  return np.array([float(x) for x in m.group(1).split(",")]) if m else np.ones(3)


__all__ = ["neper"]
