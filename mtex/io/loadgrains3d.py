# the import of grain meshes (MATLAB grain3d.load, loadGrains_Dream3d, loadNeperTess, ../master)
#
#   grains = loadGrains3d('grains.dream3d')              the surface mesh of a DREAM.3D file
#   grains = loadGrains3d('grains.tess', CS=cs)          a Neper tessellation of polyhedra
#   tess = loadNeperTess('slice.tess')                   the parts of a tessellation, 2D or 3D
#
# DREAM.3D writes its feature quaternions as (x, y, z, w) of the passive rotation: the rotation
# of a grain is (w, -x, -y, -z), as the Euler angles of its voxels give it. MATLAB's
# loadGrains_Dream3d reads them as (w, x, y, z) in the order (x, y, z, w) and so turns every
# grain away from its voxels; this reader takes the voxels' side.
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

from ..geometry.frames import crystalFrame
from ..geometry.quaternion import rotation
from ..maps.phaselist import phaseList
from .load3d import DREAM3D_SYMMETRY

# the crystal systems Neper names, as point groups
NEPER_SYMMETRY = {"triclinic": "1", "cubic": "m-3m", "hexagonal": "6/mmm", "-1": "1"}


def loadGrains3d(fname, *args, interface=None, CS=None, orientFaces=True, **kwargs):
  """the 3D grains of a file, MATLAB's grain3d.load: the surface mesh of a DREAM.3D file or the
  polyhedra of a 3D Neper tessellation; the faces are oriented unless orientFaces=False

    loadGrains3d('grains.dream3d')
    loadGrains3d('grains.tess', CS=cs)"""
  ext = os.path.splitext(fname)[1].lower()
  interface = interface or {".tess": "neper", ".dream3d": "dream3d"}.get(ext)
  if interface == "neper":
    t = loadNeperTess(fname)
    if t.dimension != 3:
      raise ValueError("a 2D tessellation; read it with loadNeperTess or neper.getSlice")
    from ..maps3d.grain3d import grain3d
    cs = CS if CS is not None else crystalFrame(NEPER_SYMMETRY.get(t.crysym, t.crysym))
    cs = cs if isinstance(cs, phaseList) else phaseList([cs])
    return grain3d.fromFaces(t.V, t.F, t.I_GF, t.ori, cs, np.ones(t.I_GF.shape[0], dtype=np.int8))
  if interface == "dream3d":
    return loadGrains_Dream3d(fname, CS=CS, orientFaces=orientFaces)
  raise ValueError(f"no grain interface for {fname}")


def loadGrains_Dream3d(fname, CS=None, orientFaces=True):
  """the 3D grains of the surface mesh of a DREAM.3D file, the rotations those of the voxels, where
  MATLAB's reader turns every grain away from them; the faces oriented unless noOrientFaces=True"""
  import h5py
  from scipy.sparse import csr_matrix
  from ..maps3d.grain3d import grain3d
  with h5py.File(fname, "r") as f:
    tri = "DataStructure/TriangleDataContainer"
    feat = "DataStructure/DataContainer/CellFeatureData"
    if f"{tri}/FaceData/FaceLabels" not in f:
      raise ValueError("the file does not contain any grain information")
    labels = np.asarray(f[f"{tri}/FaceData/FaceLabels"][...], dtype=np.int64).reshape(-1, 2)
    xyzw = np.asarray(f[f"{feat}/AvgQuats"][...], dtype=float).reshape(-1, 4)
    active = np.asarray(f[f"{feat}/Active"][...]).reshape(-1).astype(bool) if f"{feat}/Active" in f else np.ones(len(xyzw), dtype=bool)
    phases = np.asarray(f[f"{feat}/Phases"][...]).reshape(-1)
    structures = np.asarray(f["DataStructure/DataContainer/CellEnsembleData/CrystalStructures"][...]).reshape(-1)
    V = np.asarray(f[f"{tri}/SharedVertexList"][...], dtype=float).reshape(-1, 3)
    F = np.asarray(f[f"{tri}/SharedTriList"][...], dtype=np.int64).reshape(-1, 3)
  q = np.column_stack([xyzw[:, 3], -xyzw[:, :3]])[active]
  if CS is None:
    CS = ["notIndexed"] + [crystalFrame(DREAM3D_SYMMETRY[s], mineral="unknown") if 0 <= s < len(DREAM3D_SYMMETRY) else "notIndexed"
                           for s in structures[1:]]
  # the lower feature first, the outside -1 first of all; the first side +1, the second -1
  labels = np.sort(labels, axis=1)
  pos, neg = labels[:, 1] > 0, labels[:, 0] > 0
  n = int(labels.max())
  rows = np.concatenate([labels[neg, 0] - 1, labels[pos, 1] - 1])
  cols = np.concatenate([np.flatnonzero(neg), np.flatnonzero(pos)])
  vals = np.concatenate([np.ones(int(neg.sum())), -np.ones(int(pos.sum()))])
  I = csr_matrix((vals, (rows, cols)), shape=(n, len(F)))
  grains = grain3d.fromFaces(V, F, I, q, CS, phases[active].astype(np.int8))
  return grains if not orientFaces else grains.orientFaces()


def loadNeperTess(fname):
  """the parts of a Neper tessellation: the dimension, the vertices, the faces as polygons of
  vertex indices, the orientations, the crystal system, and the incidence of the polyhedra
  and the faces (3D) or the cell ids (2D)"""
  from scipy.sparse import csr_matrix
  with open(fname) as fh:
    lines = [ln.rstrip("\n") for ln in fh]
  k = 0

  def nxt():
    nonlocal k
    while k < len(lines) and not lines[k].strip():
      k += 1
    k += 1
    return lines[k - 1].strip()

  if nxt() != "***tess" or nxt() != "**format":
    raise ValueError(f"{fname} is not a Neper tessellation")
  nxt()
  if nxt() != "**general":
    raise ValueError(f"{fname}: **general expected")
  dimension = int(nxt().split()[0])
  if nxt() != "**cell":
    raise ValueError(f"{fname}: **cell expected")
  numCells = int(nxt())
  crysym, ori, cellIds = "1", None, None
  line = nxt()
  while line != "**vertex":
    if line == "*id":
      vals = []
      while not lines[k].strip().startswith("*"):
        vals += [int(v) for v in lines[k].split()]
        k += 1
      cellIds = np.array(vals)
    elif line == "*crysym":
      crysym = nxt()
    elif line.startswith("*ori"):
      kind = nxt().split(":")[0].strip().lower()
      width = 4 if kind == "quaternion" else 3
      vals = np.array([float(v) for i in range(numCells) for v in lines[k + i].split()]).reshape(-1, width)
      k += numCells
      ori = rotation(vals) if kind == "quaternion" else rotation.byRodrigues(vals)
    elif line.startswith("*"):
      while not lines[k].strip().startswith("*"):
        k += 1
    line = nxt()
  numV = int(nxt())
  V = np.array([[float(v) for v in lines[k + i].split()[1:4]] for i in range(numV)])
  k += numV
  if nxt() != "**edge":
    raise ValueError(f"{fname}: **edge expected")
  numE = int(nxt())
  k += numE
  if nxt() != "**face":
    raise ValueError(f"{fname}: **face expected")
  numF = int(nxt())
  F = []
  for i in range(numF):
    vals = [int(v) for v in lines[k + 4 * i].split()]
    F.append(np.array(vals[2:], dtype=np.int64) - 1)
  k += 4 * numF
  out = SimpleNamespace(dimension=dimension, V=V, F=F, ori=ori, crysym=crysym, cellIds=cellIds, I_GF=None)
  if dimension == 3:
    if nxt() != "**polyhedron":
      raise ValueError(f"{fname}: **polyhedron expected")
    n = int(nxt())
    rows, cols, vals = [], [], []
    for i in range(n):
      v = [int(x) for x in lines[k + i].split()]
      f = np.array(v[2:])
      rows += [v[0] - 1] * len(f)
      cols += list(np.abs(f) - 1)
      vals += list(np.sign(f))
    out.I_GF = csr_matrix((np.array(vals, dtype=float), (rows, cols)), shape=(max(rows) + 1, numF))
  return out


__all__ = ["loadGrains3d", "loadGrains_Dream3d", "loadNeperTess"]
