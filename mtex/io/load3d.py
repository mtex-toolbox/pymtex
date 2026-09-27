# the import of volumes (MATLAB EBSD3.load, loadEBSD_dream3d, loadEBSD_xnovo, ../master)
#
#   ebsd = loadEBSD3('volume.dream3d')                   the cell data of a DREAM.3D image geometry
#   ebsd = loadEBSD3('volume.h5')                        the LabDCT volume of an xnovo file
#   ebsd = loadEBSD3(['s0.ang', 's1.ang'], z=[0, 1])     a stack of 2D maps at their heights
#
# HDF5 keeps its arrays in C order, the slowest axis first, where MATLAB's h5read returns them
# reversed; the arrays are turned to (x, y, z) here, so that voxel [i, j, k] of the volume is
# the one MATLAB addresses as (i+1, j+1, k+1).
from __future__ import annotations

import os

import numpy as np

from ..geometry.frames import crystalFrame
from ..geometry.quaternion import rotation
from ..maps.phaselist import phaseList
from ..maps3d.ebsd3 import EBSD3, EBSD3square

# DREAM.3D's crystal structures, by their number; 999 and anything else is not indexed
DREAM3D_SYMMETRY = ("622", "432", "6", "23", "1", "121", "222", "4", "422", "3", "322", "1")


def loadEBSD3(fname, *args, interface=None, z=None, CS=None, **kwargs):
  """the 3D EBSD map of a volume file, MATLAB's EBSD3.load: the cell data of a DREAM.3D file, the
  LabDCT volume of an xnovo file, or a stack of 2D maps at the heights z; CS replaces the
  phases of the file

    loadEBSD3('volume.dream3d')
    loadEBSD3('volume.h5', interface='xnovo')
    loadEBSD3(['s0.ang', 's1.ang'], z=[0, 1])"""
  if isinstance(fname, (list, tuple)):
    return _stack(fname, z, *args, **kwargs)
  ext = os.path.splitext(fname)[1].lower().lstrip(".")
  import h5py
  if interface == "dream3d" or (interface is None and ext == "dream3d"):
    return loadEBSD_dream3d(fname, CS=CS)
  if interface == "xnovo" or interface is None and ext in ("h5", "hdf5"):
    with h5py.File(fname, "r") as f:
      if "LabDCT" in f:
        return loadEBSD_xnovo(fname, CS=CS)
      if "DataStructure" in f:
        return loadEBSD_dream3d(fname, CS=CS)
  raise ValueError(f"{fname} is not a volume this port reads: DREAM.3D, xnovo, or a list of 2D maps")


# (z, y, x, ...) of the file to (x, y, z, ...)
def _xyz(a):
  a = np.asarray(a)
  return np.ascontiguousarray(np.swapaxes(a, 0, 2))


def loadEBSD_dream3d(fname, CS=None):
  """the 3D EBSD map of the image geometry of a DREAM.3D file: the Euler angles, the phases with the
  mask, every further scalar cell array as a property, the feature ids as grainId"""
  import h5py
  root = "DataStructure/DataContainer"
  with h5py.File(fname, "r") as f:
    g = f[root]
    spacing = np.asarray(g.attrs["_SPACING"], dtype=float)
    origin = np.asarray(g.attrs["_ORIGIN"], dtype=float)
    cell = g["CellData"]
    eul = _xyz(cell["EulerAngles"][...]).astype(float)
    shape = eul.shape[:3]
    phases = _xyz(cell["Phases"][...]).reshape(shape).astype(np.int64)
    if "Mask" in cell:
      phases[_xyz(cell["Mask"][...]).reshape(shape) == 0] = 0
    if CS is None:
      structures = np.asarray(g["CellEnsembleData/CrystalStructures"][...]).reshape(-1)
      CS = ["notIndexed"] + [crystalFrame(DREAM3D_SYMMETRY[s], mineral="unknown") if 0 <= s < len(DREAM3D_SYMMETRY) else "notIndexed"
                             for s in structures[1:]]
    # every further scalar cell array is a property, the feature ids the grain id
    prop = {}
    for name, ds in cell.items():
      if name in ("Phases", "Mask") or ds.ndim != 4 or ds.shape[-1] != 1:
        continue
      key = "grainId" if name == "FeatureIds" else name.replace(" ", "")
      prop[key] = _xyz(ds[...]).reshape(shape).astype(float)
  rot = rotation.byEuler(eul[..., 0], eul[..., 1], eul[..., 2], "ZXZ")
  cs = CS if isinstance(CS, phaseList) else phaseList(CS)
  return EBSD3square(rot, phases, cs, spacing, origin=origin, prop=prop)


def loadEBSD_xnovo(fname, CS=None, map="LabDCT"):
  """the 3D EBSD map of the LabDCT volume of an xnovo HDF5 file, the orientations from Rodrigues
  vectors, quaternions or Euler angles, the completeness and grain ids as properties, in mm"""
  import h5py
  with h5py.File(fname, "r") as f:
    g = f[map]
    data = g["Data"]
    phase = _xyz(data["PhaseId"][...]).astype(np.int64)
    shape = phase.shape
    if "Rodrigues" in data:
      from ..geometry.vector3d import vector3d
      rot = rotation.byRodrigues(vector3d(_xyz(data["Rodrigues"][...]).astype(float)))
    elif "Quaternion" in data:
      rot = rotation(_xyz(data["Quaternion"][...]).astype(float))
    elif "EulerZXZ" in data:
      e = _xyz(data["EulerZXZ"][...]).astype(float)
      rot = rotation.byEuler(e[..., 0], e[..., 1], e[..., 2], "ZXZ")
    elif "EulerZYZ" in data:
      e = _xyz(data["EulerZYZ"][...]).astype(float)
      rot = rotation.byEuler(e[..., 0], e[..., 1], e[..., 2], "ZYZ")
    else:
      raise ValueError(f"{fname} holds no orientations in {map}/Data")
    q = rot.data.reshape(-1, 4)
    q[~np.all(np.isfinite(q), axis=1)] = [1, 0, 0, 0]
    prop = {}
    if "Completeness" in data:
      prop["Completeness"] = _xyz(data["Completeness"][...]).astype(float)
    if "GrainId" in data:
      prop["grainId"] = _xyz(data["GrainId"][...]).astype(float)
    center = np.asarray(g["Center"][...], dtype=float)
    spacing = np.asarray(g["Spacing"][...], dtype=float)
    cs = ["notIndexed", CS] if isinstance(CS, crystalFrame) else CS if CS is not None else _xnovoPhases(f)
  origin = center - (np.array(shape) - 1) * spacing / 2
  e = EBSD3square(rot, phase, cs if isinstance(cs, phaseList) else phaseList(cs), spacing, origin=origin, prop=prop)
  object.__setattr__(e, "scanUnit", "mm")
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
  return cs


# a stack of 2D maps at the heights z, by default 0, 1, 2, ..., as MATLAB's EBSD3.load(files, '3d', Z)
def _stack(files, z, *args, **kwargs):
  from .load import loadEBSD
  maps = [loadEBSD(f, *args, noGrid=True, **kwargs) for f in files]
  z = np.arange(len(files), dtype=float) if z is None else np.asarray(z, dtype=float).reshape(-1)
  if len(z) != len(maps):
    raise ValueError("one height per file")
  first = maps[0]
  pos = np.concatenate([np.column_stack([m.pos.data.reshape(-1, 3)[:, :2], np.full(m.size, zk)]) for m, zk in zip(maps, z)])
  rot = np.concatenate([m.rotations.data.reshape(-1, 4) for m in maps])
  ph = np.concatenate([m.phaseId.reshape(-1) for m in maps])
  prop = {k: np.concatenate([np.asarray(m.prop[k]).reshape(-1) for m in maps]) for k in first.prop if all(k in m.prop for m in maps)}
  return EBSD3(pos, rotation(rot), ph, first.CSList, prop, scanUnit=first.scanUnit, frame=first.frame)


__all__ = ["loadEBSD3", "loadEBSD_dream3d", "loadEBSD_xnovo"]
