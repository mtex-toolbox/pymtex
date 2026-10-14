# the import of grain meshes (MATLAB grain3d.load, loadGrains_Dream3d, loadNeperTess, ../master)
#
#   grains = loadGrains3d('grains.dream3d')              the surface mesh of a DREAM.3D file
#   grains = loadGrains3d('grains.tess', CS=cs)          a Neper tessellation of polyhedra
#   tess = loadNeperTess('slice.tess')                   the parts of a tessellation, 2D or 3D
#
# DREAM.3D writes its feature quaternions as (x, y, z, w) of the passive rotation: the rotation
# of a grain is (w, -x, -y, -z), as the Euler angles of its voxels give it. The mesh and the
# grain data are located by the names of their data sets, as the voxels are (load3d.py).
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

from ..geometry.frames import crystalFrame
from ..geometry.quaternion import rotation
from ..maps.phaselist import phaseList
from .load3d import _dream3dEnsemble, _loadEBSD_dream3d, _named, _parent, dream3dDatasets

# the crystal systems Neper names, as point groups
NEPER_SYMMETRY = {"triclinic": "1", "cubic": "m-3m", "hexagonal": "6/mmm", "-1": "1"}


def loadGrains3d(fname, *args, interface=None, CS=None, orientFaces=True, **kwargs):
  """the 3D grains read from a file, the surface mesh of a DREAM.3D file or the polyhedra of a
  3D Neper tessellation; MATLAB's grain3d.load and loadGrains_Dream3d

    grains = loadGrains3d('grains.dream3d')                     # the surface mesh of DREAM.3D
    grains = loadGrains3d('grains.dream3d', orientFaces=False)  # the faces as stored
    grains = loadGrains3d('grains.tess', CS=cs)                 # a Neper tessellation

  The format is told by the extension:

  - '.dream3d': the surface mesh of DREAM.3D 6 or DREAM3D-NX, whatever its data containers are
  called. A file without a surface mesh gives the grains calcGrains forms from the feature ids
  of its voxels.
  - '.tess': the polyhedra of a 3D Neper tessellation with the orientations it holds. A 2D
  tessellation is refused; read it with loadNeperTess.

  DREAM.3D stores the faces with random winding, so their normals point into or out of the
  grain at random. The faces are therefore oriented on import by grain3d.orientFaces, which
  makes the face normals, the grain volumes and boundary.grainId meaningful;
  orientFaces=False keeps the winding of the file. grain3d.load is the same function.

  Input
    fname - the file name, .dream3d or .tess
  Output
    grains - grain3d
  Keywords
    interface - the format, 'dream3d' or 'neper'; by default told by the extension
    CS - a crystalFrame or a phaseList replacing the phases of the file; for a Neper file by
        default the crystal system it states
    orientFaces - orient the faces of a DREAM.3D mesh consistently, MATLAB's 'noOrientFaces'
        when False
  See also
    grain3d.orientFaces, loadNeperTess, loadEBSD3, calcGrains, Dream3dGrains, NeperInterface"""
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
    return _loadGrains_Dream3d(fname, CS=CS, orientFaces=orientFaces)
  raise ValueError(f"no grain interface for {fname}")


# the 3D grains of the surface mesh of a DREAM.3D file, the faces oriented unless
# orientFaces=False; a file without a mesh gives the grains calcGrains forms from the feature
# ids of its voxels. The triangle geometry is the data container of the face labels, the grain
# data the attribute matrix of the mean orientations
def _loadGrains_Dream3d(fname, CS=None, orientFaces=True):
  import h5py
  from scipy.sparse import csr_matrix
  from ..maps3d.grain3d import grain3d
  from .h5array import readArray
  with h5py.File(fname, "r") as f:
    paths = dream3dDatasets(f)
    faceLabels = _named(paths, "FaceData/FaceLabels", "Face Data/FaceLabels")
    if faceLabels:
      mesh = [p for p in paths if p.startswith(_parent(_parent(faceLabels[0])) + "/")]
      ori = _named(paths, "AvgQuats", "AvgEulerAngles")
      if not ori:
        raise ValueError(f"{fname} does not contain any grain orientations")
      feature = [p for p in paths if _parent(p) == _parent(ori[0])]
      labels = np.asarray(readArray(f[faceLabels[0]]), dtype=np.int64).reshape(-1, 2)
      V = np.asarray(readArray(f[_named(mesh, "SharedVertexList")[0]]), dtype=float).reshape(-1, 3)
      F = np.asarray(readArray(f[_named(mesh, "SharedTriList")[0]]), dtype=np.int64).reshape(-1, 3)
      mean = np.asarray(f[ori[0]][...], dtype=float)
      phases = np.asarray(f[_named(feature, "Phases")[0]][...]).reshape(-1)
      structures = _named(paths, "CrystalStructures")[0]
      ensemble = _dream3dEnsemble(f, structures) if CS is None else CS
  if not faceLabels:
    from ..maps3d.calcgrains3d import calcGrains
    return calcGrains(_loadEBSD_dream3d(fname, CS=CS), grainId=True)
  if ori[0].endswith("AvgQuats"):
    q = np.column_stack([mean[:, 3], -mean[:, :3]])
  else:
    q = rotation.byEuler(mean[:, 0], mean[:, 1], mean[:, 2], "ZXZ").data
  # the first feature is the one no voxel belongs to
  q, phases = q[1:], phases[1:]
  CS = ensemble
  # the lower feature first, the outside -1 first of all; the first side +1, the second -1
  labels = np.sort(labels, axis=1)
  pos, neg = labels[:, 1] > 0, labels[:, 0] > 0
  n = int(labels.max())
  rows = np.concatenate([labels[neg, 0] - 1, labels[pos, 1] - 1])
  cols = np.concatenate([np.flatnonzero(neg), np.flatnonzero(pos)])
  vals = np.concatenate([np.ones(int(neg.sum())), -np.ones(int(pos.sum()))])
  I = csr_matrix((vals, (rows, cols)), shape=(n, len(F)))
  grains = grain3d.fromFaces(V, F, I, q, CS, phases.astype(np.int8))
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
      vals = np.array(" ".join(lines[k:k + numCells]).split(), dtype=float).reshape(-1, width)
      k += numCells
      ori = rotation(vals) if kind == "quaternion" else rotation.byRodrigues(vals)
    elif line.startswith("*"):
      while not lines[k].strip().startswith("*"):
        k += 1
    line = nxt()
  numV = int(nxt())
  V = np.array([l.split()[1:4] for l in lines[k:k + numV]], dtype=float)
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
    # a polyhedron line is its id, the number of its faces and the faces, signed by orientation
    rows, faces = [], []
    for line in lines[k:k + n]:
      v = np.array(line.split(), dtype=np.int64)
      rows.append(np.full(len(v) - 2, v[0] - 1))
      faces.append(v[2:])
    rows, faces = np.concatenate(rows), np.concatenate(faces)
    out.I_GF = csr_matrix((np.sign(faces).astype(float), (rows, np.abs(faces) - 1)), shape=(rows.max() + 1, numF))
  return out


__all__ = ["loadGrains3d", "loadNeperTess"]
