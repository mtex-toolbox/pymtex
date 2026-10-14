# the import of volumes (MATLAB EBSD3.load, loadEBSD_dream3d, loadEBSD_xnovo, ../master)
#
#   ebsd = loadEBSD3('volume.dream3d')                   the voxel data of a DREAM.3D file
#   ebsd = loadEBSD3('volume.h5')                        the LabDCT volume of an xnovo file
#   ebsd = loadEBSD3(['s0.ang', 's1.ang'], z=[0, 1])     a stack of 2D maps at their heights
#
# HDF5 keeps its arrays in C order, the slowest axis first, where MATLAB's h5read returns them
# reversed; the arrays are turned to (x, y, z) here, so that voxel [i, j, k] of the volume is
# the one MATLAB addresses as (i+1, j+1, k+1). A data set is read over the cores (h5array.py)
# and turned and converted in one compiled pass (turnVolume, turnBunge and turnRodrigues of kernels.py).
#
# Neither the data containers of a DREAM.3D file nor their attribute matrices have fixed names
# (DREAM.3D 6 writes DataContainers/.../CellData, DREAM3D-NX DataStructure/.../Cell Data, a
# pipeline whatever it is told), so the data are located by the name of the data set.
from __future__ import annotations

import os
import re

import numpy as np

from ..geometry.frames import crystalFrame
from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..maps.phaselist import distinctMinerals, phaseList
from ..maps3d.ebsd3 import EBSD3, EBSD3square
from .h5array import readArray
from .kernels import turnBunge, turnRodrigues, turnVolume

# DREAM.3D's crystal structures, by their number; 999 and anything else is not indexed
DREAM3D_SYMMETRY = ("622", "432", "6", "23", "1", "121", "222", "4", "422", "3", "322", "1")
# DREAM.3D puts the a axis of a hexagonal or trigonal cell along x: the voxels its IPF colours
# green, which its key labels <2-1-10>, are 6.6 degrees from <2-1-10> under X||a and 28 under
# X||a* (the Zircaloy slab of zenodo 3785085, 2026-10-14)
DREAM3D_ALIGNMENT = {"622": "X||a", "6": "X||a", "3": "X||a", "322": "X||a"}
# the data sets that tell the voxel data of a DREAM.3D file, the first found deciding
DREAM3D_VOXELS = ("EulerAngles", "Quats", "FeatureIds")


def loadEBSD3(fname, *args, interface=None, z=None, CS=None, **kwargs):
  """the EBSD map of a volume read from a file, the voxel data of a DREAM.3D file, the LabDCT
  volume of an xnovo file, or a stack of 2D maps; MATLAB's EBSD3.load

    ebsd = loadEBSD3('volume.dream3d')                     # the format from the file
    ebsd = loadEBSD3('volume.h5', interface='xnovo')       # the format given
    ebsd = loadEBSD3('volume.dream3d', CS=cs)              # other phases than the file's
    ebsd = loadEBSD3(['s0.ang', 's1.ang'], z=[0, 1])       # a stack of 2D maps at their heights

  The format is told by the extension and, for an HDF5 file, by what it holds: a LabDCT group
  makes it an xnovo file, the cell data of a grid a DREAM.3D file, whatever the data
  containers are called. The voxels of a DREAM.3D or xnovo file form a voxelLattice, so
  ebsd[i, j, k] addresses one of them; every further scalar array of the file becomes a
  property. A list of files is read by loadEBSD, every file a layer at its height z, as a
  list of voxels without a lattice; the phases and the properties are those of the first
  file. EBSD3.load is the same function.

  Input
    fname - the file name, or a list of file names of 2D maps
  Output
    ebsd - EBSD3
  Keywords
    interface - the format, 'dream3d' or 'xnovo'; by default told by the file
    z - the height of every map of a stack, by default 0, 1, 2, ...
    CS - a crystalFrame or a phaseList replacing the phases of the file
  See also
    loadEBSD, loadGrains3d, calcGrains, EBSD3Analysis, EBSD3Plotting"""
  if isinstance(fname, (list, tuple)):
    return _stack(fname, z, *args, **kwargs)
  ext = os.path.splitext(fname)[1].lower().lstrip(".")
  import h5py
  if interface is None and ext in ("h5", "hdf5"):
    with h5py.File(fname, "r") as f:
      interface = "xnovo" if "LabDCT" in f else "dream3d" if dream3dVolume(f) is not None else None
  if interface == "dream3d" or (interface is None and ext == "dream3d"):
    return _loadEBSD_dream3d(fname, CS=CS)
  if interface == "xnovo":
    return _loadEBSD_xnovo(fname, CS=CS)
  raise ValueError(f"{fname} is not a volume this port reads: DREAM.3D, xnovo, or a list of 2D maps")


def dream3dDatasets(f):
  """the paths of all data sets of an open DREAM.3D file, those of a group before those of its
  subgroups; the readers locate their data by the name of the data set"""
  import h5py
  paths = []

  def walk(group, prefix):
    below = []
    for name in group:
      obj = group.get(name)
      if isinstance(obj, h5py.Dataset):
        paths.append(prefix + name)
      elif isinstance(obj, h5py.Group):
        below.append((obj, prefix + name + "/"))
    for obj, p in below:
      walk(obj, p)

  walk(f, "")
  return paths


# the paths ending in one of the names, in the order of the names
def _named(paths, *names):
  return [p for name in names for p in paths if p.endswith("/" + name)]


# the group of a path
def _parent(path):
  return path.rpartition("/")[0]


# the attribute matrix of the voxels, the group of the first of EulerAngles, Quats and FeatureIds
def _voxelData(paths):
  hit = _named(paths, *DREAM3D_VOXELS)
  return _parent(hit[0]) if hit else None


# the phases of DREAM.3D's crystal structures, the first the not indexed one, a number outside
# the table not indexed too; named by the materials and on the cells (a, b, c, alpha, beta, gamma
# in degrees) the file states, unknown and the cell of the group where it states none, the
# hexagonal and trigonal ones X||a; phases of one name take a letter each
def _dream3dPhases(structures, materials=None, cells=None):
  structures = np.asarray(structures).reshape(-1)
  phases = ["notIndexed"]
  for k in range(1, len(structures)):
    s = structures[k]
    if not 0 <= s < len(DREAM3D_SYMMETRY):
      phases.append("notIndexed")
      continue
    name = _material(materials[k]) if materials is not None and k < len(materials) else ""
    cell = np.asarray(cells[k], dtype=float).reshape(-1) if cells is not None and k < len(cells) else np.zeros(0)
    lattice = (cell[:3], cell[3:] * degree) if cell.size == 6 and np.all(cell > 0) else ()
    group = DREAM3D_SYMMETRY[s]
    phases.append(crystalFrame(group, *lattice, *([DREAM3D_ALIGNMENT[group]] if group in DREAM3D_ALIGNMENT else []),
                               mineral=name or "unknown"))
  return distinctMinerals(phases)


# a material name as DREAM.3D stores it, bytes or text, with the blanks around it
def _material(name):
  return (name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name)).strip()


# the phases of a DREAM.3D file from the ensemble data beside its crystal structures
def _dream3dEnsemble(f, structures):
  ensemble = [p for p in dream3dDatasets(f) if _parent(p) == _parent(structures)]
  materials = _named(ensemble, "MaterialName", "Material Name", "PhaseName", "Phase Name")
  cells = _named(ensemble, "LatticeConstants", "Lattice Constants")
  return _dream3dPhases(f[structures][...], f[materials[0]][...].reshape(-1) if materials else None,
                        f[cells[0]][...].reshape(-1, 6) if cells and f[cells[0]].size % 6 == 0 else None)


# the name a cell array has as a property: the feature ids are the grain ids, a letter after a
# blank is a capital and whatever else no name holds an underscore, as MATLAB's makeValidName
def _propertyName(name):
  name = re.sub(r"\s+(\S)", lambda m: m.group(1).upper(), name.replace("FeatureIds", "grainId").strip())
  name = re.sub(r"\W", "_", name)
  return name if name[:1].isalpha() else "x" + name


# the grid of an image geometry as (cells along x, y, z), spacing, origin: data sets of
# _SIMPL_GEOMETRY in DREAM.3D 6, attributes of the data container in DREAM3D-NX
def _dream3dGrid(f, root):
  if root + "/_SIMPL_GEOMETRY/DIMENSIONS" in f:
    part = {name: f[f"{root}/_SIMPL_GEOMETRY/{name}"][...] for name in ("DIMENSIONS", "SPACING", "ORIGIN")}
  else:
    part = {name: f[root].attrs["_" + name] for name in ("DIMENSIONS", "SPACING", "ORIGIN")}
  dims, spacing, origin = (np.asarray(part[name], dtype=float).reshape(3) for name in ("DIMENSIONS", "SPACING", "ORIGIN"))
  return tuple(int(n) for n in dims), spacing, origin


# an array of a file turned to (x, y, z) in the type the file stores it in: a property of a
# volume stays float32 or int32, half of what float64 takes on 49 million voxels
def _turned(a):
  return turnVolume(a, a.dtype)


def dream3dVolume(f):
  """what an open DREAM.3D file holds as a volume, None where it holds no voxels on a grid: the
  attribute matrix of the voxels (cellData) in its data container (root), the cells along
  x, y, z (shape), spacing and origin, the names of the cell arrays, the properties as
  {name of the property: name of the array} and the path of the crystal structures"""
  from types import SimpleNamespace
  paths = dream3dDatasets(f)
  cellData = _voxelData(paths)
  if cellData is None:
    return None
  root = _parent(cellData)
  if root + "/_SIMPL_GEOMETRY/DIMENSIONS" not in f and "_DIMENSIONS" not in f[root or "/"].attrs:
    return None
  shape, spacing, origin = _dream3dGrid(f, root)
  n = shape[0] * shape[1] * shape[2]
  names = [p[len(cellData) + 1:] for p in paths if _parent(p) == cellData]
  # every scalar cell array but the phases and the mask is a property
  scalar = [name for name in names if name not in ("Phases", "Mask") and f[cellData + "/" + name].size == n
            and f[cellData + "/" + name].dtype.kind in "iufb"]
  structures = [p for p in paths if p.startswith(root + "/") and p.endswith("/CrystalStructures")]
  return SimpleNamespace(paths=paths, cellData=cellData, root=root, shape=shape, spacing=spacing, origin=origin, names=names,
                         properties={_propertyName(name): name for name in scalar}, structures=structures[0] if structures else None)


# the phases a DREAM.3D volume declares, one unknown triclinic phase where it declares none
def _dream3dCS(f, vol):
  return _dream3dEnsemble(f, vol.structures) if vol.structures else ["notIndexed", crystalFrame("1")]


# the 3D EBSD map of the voxel data of a DREAM.3D file: the Euler angles or else the quaternions,
# the phases with the mask, every further scalar cell array as a property, the feature ids as
# grainId; without orientations the rotations are NaN, without phases the voxels are one phase.
# Only the arrays that enter the map are read
def _loadEBSD_dream3d(fname, CS=None):
  import h5py
  with h5py.File(fname, "r") as f:
    vol = dream3dVolume(f)
    if vol is None:
      raise ValueError(f"{fname} holds no voxel data: none of the data sets {', '.join(DREAM3D_VOXELS)} on a grid")
    cellData, shape, names = vol.cellData, vol.shape, vol.names
    nx, ny, nz = shape

    # a cell array turned to (x, y, z), its components last
    def read(name, dtype=None):
      a = readArray(f[cellData + "/" + name]).reshape(nz, ny, nx, -1)
      return _turned(a) if dtype is None else turnVolume(a, dtype)

    if "EulerAngles" in names:
      rot = rotation._fromData(turnBunge(readArray(f[cellData + "/EulerAngles"]).reshape(nz, ny, nx, 3)))
    elif "Quats" in names:
      # the passive rotation as (x, y, z, w)
      q = read("Quats", np.float64)
      rot = rotation(np.concatenate([q[..., 3:], -q[..., :3]], axis=-1))
    else:
      rot = rotation.nan(shape)
    phases = read("Phases").reshape(shape) if "Phases" in names else np.ones(shape, dtype=np.int8)
    if "Mask" in names:
      phases[read("Mask", np.uint8).reshape(shape) == 0] = 0
    if CS is None:
      CS = _dream3dCS(f, vol)
    prop = {key: read(name).reshape(shape) for key, name in vol.properties.items()}
  cs = CS if isinstance(CS, phaseList) else phaseList(CS)
  return EBSD3square(rot, phases, cs, vol.spacing, origin=vol.origin, prop=prop)


# the 3D EBSD map of the LabDCT volume of an xnovo HDF5 file, the orientations from Rodrigues
# vectors, quaternions or Euler angles, the completeness and grain ids as properties, in mm
def _loadEBSD_xnovo(fname, CS=None, map="LabDCT"):
  import h5py
  with h5py.File(fname, "r") as f:
    g = f[map]
    data = g["Data"]
    phase = turnVolume(readArray(data["PhaseId"]), np.int64)
    shape = phase.shape
    if "Rodrigues" in data:
      rot = rotation._fromData(turnRodrigues(readArray(data["Rodrigues"])))
    elif "Quaternion" in data:
      rot = rotation(turnVolume(readArray(data["Quaternion"])))
    else:
      convention = next((c for c in ("ZXZ", "ZYZ") if "Euler" + c in data), None)
      if convention is None:
        raise ValueError(f"{fname} holds no orientations in {map}/Data")
      e = turnVolume(readArray(data["Euler" + convention]))
      rot = rotation.byEuler(e[..., 0], e[..., 1], e[..., 2], convention)
    q = rot.data.reshape(-1, 4)
    q[~np.all(np.isfinite(q), axis=1)] = [1, 0, 0, 0]
    prop = {}
    if "Completeness" in data:
      prop["Completeness"] = _turned(readArray(data["Completeness"]))
    if "GrainId" in data:
      prop["grainId"] = _turned(readArray(data["GrainId"]))
    center = np.asarray(g["Center"][...], dtype=float)
    spacing = np.asarray(g["Spacing"][...], dtype=float)
    if isinstance(CS, crystalFrame):
      cs = ["notIndexed", CS]
    else:
      cs = _xnovoPhases(f) if CS is None else CS
  origin = center - (np.array(shape) - 1) * spacing / 2
  e = EBSD3square(rot, phase, cs if isinstance(cs, phaseList) else phaseList(cs), spacing, origin=origin, prop=prop)
  e.scanUnit = "mm"
  return e


# the phases of an xnovo file: the space group and the cell of every phase, its Laue group
def _xnovoPhases(f):
  cs = ["notIndexed"]
  for name in sorted(f["PhaseInfo"]):
    p = f["PhaseInfo"][name]
    uc = np.asarray(p["UnitCell"][...], dtype=float).reshape(-1)
    mineral = p["Name"][0].decode() if "Name" in p else f"Phase{len(cs)}"
    color = None
    if "Color" in p:
      rgba = np.array([p["Color"][0]], dtype=np.int32).view(np.uint8)
      color = tuple(float(c) / 255 for c in rgba[:3])
    sg = int(p["SpaceGroup"][0])
    # a trigonal or hexagonal group takes the angles of the hexagonal cell, as MATLAB does with
    # the rhombohedral cell of Corundum
    angles = np.array([90.0, 90.0, 120.0]) if 143 <= sg <= 194 else uc[3:6]
    fr = crystalFrame(uc[:3], angles * np.pi / 180, "X||a", "Z||c*", spaceId=sg, mineral=mineral, color=color)
    cs.append(fr.Laue())
  return distinctMinerals(cs)


# a stack of 2D maps at the heights z, by default 0, 1, 2, ..., as MATLAB's EBSD3.load(files, '3d', Z)
def _stack(files, z, *args, **kwargs):
  from .load import loadEBSD
  maps = [loadEBSD(f, *args, grid=False, **kwargs) for f in files]
  z = np.arange(len(files), dtype=float) if z is None else np.asarray(z, dtype=float).reshape(-1)
  if len(z) != len(maps):
    raise ValueError("one height per file")
  first = maps[0]
  pos = np.concatenate([np.column_stack([m.pos.data.reshape(-1, 3)[:, :2], np.full(m.size, zk)]) for m, zk in zip(maps, z)])
  rot = np.concatenate([m.rotations.data.reshape(-1, 4) for m in maps])
  ph = np.concatenate([m.phaseId.reshape(-1) for m in maps])
  prop = {k: np.concatenate([np.asarray(m.prop[k]).reshape(-1) for m in maps]) for k in first.prop if all(k in m.prop for m in maps)}
  return EBSD3(pos, rotation(rot), ph, first.CSList, prop, scanUnit=first.scanUnit, frame=first.frame)


__all__ = ["loadEBSD3"]
