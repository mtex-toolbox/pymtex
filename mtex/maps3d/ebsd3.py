# EBSD3: measurements of a volume (MATLAB EBSD3 and EBSD3square, ../master)
#
#   ebsd = EBSD3(pos, rot, phaseId, CSList)          a list of voxels, its lattice inferred
#   ebsd = EBSD3square(rot, phaseId, CSList, dxyz)   a volume on its (n1, n2, n3) array
#   ebsd.gridify()                                   a list back onto its array, the gaps padded
#   ebsd[i, j, k], ebsd['Iron'], ebsd[mask]          voxels, phases, selections
#   ebsd.d1, ebsd.d2, ebsd.d3, ebsd.dx, ebsd.dy, ebsd.dz
#   ebsd.slice(plane3d(N, P0)), ebsd.slice(N, P0)    the section through a plane, a 2D EBSD
#   ebsd.interp(pos)                                 the voxel holding every position
#
# A gridded volume is an (n1, n2, n3) array whose axes run along d1, d2, d3 as MATLAB's
# ndgrid order, x fastest in MATLAB's memory and first in the array here. A list keeps the
# lattice it came from, so a selection can be put back onto its array. The padding gridify
# adds is notIndexed and marked in isPadding, as in a 2D map.
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from ..geometry.kernels import columnMax, columnMin
from ..geometry.plane import plane3d
from ..geometry.vector3d import vector3d
from ..maps.ebsd import EBSD, _distinct, _rotations, _vectors
from .voxellattice import boxLattice, voxelLattice
from ..geometry.display import aligned, frameLabel


class EBSD3(EBSD):
  """the measurements of a volume: positions, rotations and phase ids of voxels with the list of
  the phases' frames and the properties of every measurement

    ebsd = EBSD3(pos, rot, phaseId, CSList)                     # the voxels as a list
    ebsd = EBSD3(pos, rot, phaseId, CSList, prop={'ci': ci})    # with properties
    ebsd = EBSD3(pos, rot, phaseId, CSList, D=D)                # the steps of the voxel lattice stated
    ebsd = EBSD3square(rot, phaseId, CSList, dxyz=(dx, dy, dz))  # an (n1, n2, n3) array of voxels

  A volume is the three dimensional EBSD: a list of voxels, or arrays of shape (n1, n2, n3)
  with ebsd[i, j, k] one voxel, where orientation gradients are available along all three
  axes. phaseId indexes CSList, 0 being the not indexed phase. EBSD3.load reads the volumes of
  DREAM.3D and Neper.

  The voxel lattice, the origin and the three steps d1, d2, d3, is inferred from the
  positions unless it is given: the steps are the shortest independent differences between
  neighbours, or the rows of D. Positions on no lattice leave ebsd.lattice None.

  Input
    pos - vector3d, or an (..., 3) array, the centres of the voxels
    rotations - rotation, the rotation of every voxel
    phaseId - int array, the phase of every voxel as the index into CSList
    CSList - phaseList, or a list of crystalFrame with 'notIndexed' first
    prop - dict, a property per voxel of the shape of the volume
  Output
    ebsd - EBSD3
  Keywords
    scanUnit - the unit of the positions
    frame - specimenFrame of the positions, by default that of pos or the default one
    lattice - voxelLattice of the positions, when it is known, instead of inferring it
    meta - dict, what a file states besides the measurements
    phaseMap - the phase number the file gives each entry of CSList
    id - the id of every voxel, by default its position in the list
    isPadding - the mask of the cells a grid adds to close it
    D - 3 x 3 array, the steps of the voxel lattice as rows, instead of inferring them
  Properties
    pos - vector3d voxel-centre positions
    rotations - measured rotation at each voxel
    phaseId - phase-list index at each voxel, zero for notIndexed
    CSList - phase metadata and crystal frames
    shape - voxel-array shape or shape of a scattered selection
    lattice - voxel grid geometry and spacing
    isPadding - mask of unmeasured cells introduced by gridding
    scanUnit - unit of spatial positions
  See also
    EBSD3square, EBSD, voxelLattice, calcGrains, grain3d, EBSD3Analysis"""

  def __init__(self, pos, rotations, phaseId, CSList, prop=None, scanUnit="um", frame=None, lattice=None, meta=None,
               phaseMap=None, id=None, isPadding=None, D=None):
    super().__init__(pos, rotations, phaseId, CSList, prop, scanUnit=scanUnit, frame=frame, lattice=False, meta=meta,
                     phaseMap=phaseMap, id=id, isPadding=isPadding)
    if lattice is None and self.size >= 4:
      try:
        lattice = voxelLattice.fromPositions(self.pos.data, D)
      except ValueError:
        lattice = None
    self.lattice = lattice or None

  # ------------------------------------------------------------------ the lattice

  @property
  def isGridded(self):
    """whether the voxels are held as an (n1, n2, n3) array"""
    return self.ndim == 3

  def _vec(self, k):
    if self.lattice is None:
      raise ValueError("the positions are not on a lattice of voxels")
    return vector3d(self.lattice.D[k], frame=self.frame)

  @property
  def d1(self):
    """the first step vector of the voxel lattice"""
    return self._vec(0)

  @property
  def d2(self):
    """the second step vector of the voxel lattice"""
    return self._vec(1)

  @property
  def d3(self):
    """the third step vector of the voxel lattice"""
    return self._vec(2)

  @property
  def dx(self):
    """the length of the first step of the voxel lattice"""
    return float(self.lattice.spacing[0])

  @property
  def dy(self):
    """the length of the second step of the voxel lattice"""
    return float(self.lattice.spacing[1])

  @property
  def dz(self):
    """the length of the third step of the voxel lattice"""
    return float(self.lattice.spacing[2])

  @property
  def dPos(self):
    """the shortest step of the voxel lattice, NaN without a lattice"""
    return np.nan if self.lattice is None else float(self.lattice.spacing.min())

  @property
  def N(self):
    """a volume has no measurement plane"""
    return vector3d(np.full(3, np.nan), frame=self.frame)

  def extent(self):
    """the bounding box of the voxels, [xmin, xmax, ymin, ymax, zmin, zmax], the padding left out"""
    P = self.pos.data.reshape(-1, 3)
    if self._pad is not None:
      P = P[~self._pad.reshape(-1)]
    return np.column_stack([columnMin(P), columnMax(P)]).reshape(-1)

  # ------------------------------------------------------------------ subsets

  # the voxels of the flat indices, in the given shape; the lattice follows the voxels
  def _take(self, flat, shape):
    pos = _vectors(self.pos.data.reshape(-1, 3)[flat].reshape(shape + (3,)), self.frame)
    rot = _rotations(self.rotations.data.reshape(-1, 4)[flat].reshape(shape + (4,)))
    ph = self.phaseId.reshape(-1)[flat].reshape(shape)
    prop = {k: v.reshape((-1,) + v.shape[self.ndim:])[flat].reshape(shape + v.shape[self.ndim:]) for k, v in self.prop.items()}
    l = self.lattice.subset(flat) if self.lattice is not None and len(flat) and _distinct(flat, self.size) else None
    pad = None if self._pad is None else self._pad.reshape(-1)[flat].reshape(shape)
    return EBSD3(pos, rot, ph, self.CSList, prop, scanUnit=self.scanUnit, frame=self.frame, lattice=l or False,
                 meta=self.meta, phaseMap=self.phaseMap, id=self.id.reshape(-1)[flat].reshape(shape), isPadding=pad)

  # the volume with its positions and rotations moved; a lattice given, or the old one carried
  # through the rigid motion (R, v) the positions went, else one inferred from where they land
  def _moved(self, pos, rot, lattice=None, motion=None):
    if lattice is None and motion is not None and self.lattice is not None:
      R, v = motion
      lattice = voxelLattice(self.lattice.origin @ R.T + v, self.lattice.D @ R.T, self.lattice.ijk)
    return EBSD3(_vectors(pos, self.frame), _rotations(rot), self.phaseId, self.CSList, self.prop, scanUnit=self.scanUnit,
                 frame=self.frame, lattice=lattice, meta=self.meta, phaseMap=self.phaseMap, id=self.id, isPadding=self._pad)

  def copy(self):
    """a copy of the volume with its own arrays, the frames shared"""
    return EBSD3(self.pos.copy(), self.rotations.copy(), self.phaseId.copy(), self.CSList,
                 {k: v.copy() for k, v in self.prop.items()}, scanUnit=self.scanUnit, frame=self.frame,
                 lattice=self.lattice or False, meta=dict(self.meta), phaseMap=self.phaseMap, id=self.id.copy(),
                 isPadding=self._pad)

  def flatten(self):
    """the voxels as a list, the volume itself when it is one"""
    return self if self.ndim == 1 else self._take(np.arange(self.size), (self.size,))

  # ------------------------------------------------------------------ the array

  def gridify(self, return_index=False):
    """a list of voxels put back onto the (n1, n2, n3) array of its lattice; MATLAB's gridify of
    EBSD3

      g = ebsd3['Iron'].gridify()                              # the selection on its lattice
      g, newId = ebsd3['Iron'].gridify(return_index=True)      # and where every voxel went

    A selection from a volume, ebsd3['Iron'] or ebsd3[condition], is a list of voxels that
    keeps the lattice of the grid it came from. gridify rebuilds that grid: every voxel goes
    to its lattice cell, and the cells no voxel occupies are padding, notIndexed with a NaN
    rotation, NaN (or 0) properties and ebsd.isPadding set. A volume that is gridded already
    is returned as it is.

    Input
      ebsd - EBSD3 whose positions lie on a lattice of voxels
    Output
      g - EBSD3square, the gridded volume, g.pos.flat[newId] the positions of the list
      newId - the flat (C order) array index of every voxel of the list, with return_index
    Keywords
      return_index - return newId as well
    See also
      EBSD3square, EBSD.gridify, calcGrains, voxelLattice"""
    if self.isGridded:
      return (self, np.arange(self.size)) if return_index else self
    l = self.lattice
    if l is None:
      raise ValueError("gridify needs the positions on a lattice of voxels")
    shape = l.shape
    newId = np.ravel_multi_index(l.ijk.T, shape)
    n = int(np.prod(shape))
    full, pos = boxLattice(l.origin, l.D, shape)
    rot = np.full((n, 4), np.nan, dtype=self.dtype)
    rot[newId] = self.rotations.data.reshape(-1, 4)
    ph = np.zeros(n, dtype=np.int8)
    ph[newId] = self.phaseId.reshape(-1)
    pad = np.ones(n, dtype=bool)
    pad[newId] = self.isPadding.reshape(-1)
    prop = {}
    for k, v in self.prop.items():
      v = np.asarray(v)
      tail = v.shape[self.ndim:]
      out = np.full((n,) + tail, np.nan if np.issubdtype(v.dtype, np.floating) else 0, dtype=v.dtype)
      out[newId] = v.reshape((-1,) + tail)
      prop[k] = out.reshape(shape + tail)
    ids = np.full(n, -1, dtype=np.int64)
    ids[newId] = self.id.reshape(-1)
    g = EBSD3(_vectors(pos.astype(self.dtype), self.frame), _rotations(rot.reshape(shape + (4,))), ph.reshape(shape),
              self.CSList, prop, scanUnit=self.scanUnit, frame=self.frame, lattice=full, meta=self.meta,
              phaseMap=self.phaseMap, id=ids.reshape(shape), isPadding=pad.reshape(shape))
    return (g, newId) if return_index else g

  # ------------------------------------------------------------------ sections

  def interp(self, pos):
    """the voxel holding every position, or none outside the box of voxel centres, as MATLAB's
    nearest griddedInterpolant without extrapolation; a 2D array of positions gives a
    gridded 2D map, a 3D one a volume"""
    P = pos.data if isinstance(pos, vector3d) else np.asarray(pos, dtype=float)
    shape = P.shape[:-1]
    P = P.reshape(-1, 3)
    g = self.gridify()
    l = g.lattice
    # the index within the grid, whose first voxel a part of a volume need not have at the origin
    shp = np.array(g.shape)
    t = l.index(P) - (columnMin(l.ijk) if len(l.ijk) else 0)
    inside = np.all((t > -1e-9) & (t < shp - 1 + 1e-9), axis=1)
    # a position halfway between two voxels takes the upper one, as MATLAB's nearest interpolant
    ijk = np.clip(np.floor(t + 0.5 + 1e-9).astype(np.int64), 0, np.maximum(shp - 1, 0))
    src = np.ravel_multi_index(ijk.T, g.shape) if g.size else np.zeros(len(P), dtype=np.int64)
    inside &= g.size > 0
    inside &= ~g.isPadding.reshape(-1)[src]
    n = len(P)
    rot = np.full((n, 4), np.nan)
    rot[inside] = g.rotations.data.reshape(-1, 4)[src[inside]]
    ph = np.zeros(n, dtype=np.int8)
    ph[inside] = g.phaseId.reshape(-1)[src[inside]]
    prop = {}
    for k, v in g.prop.items():
      tail = v.shape[3:]
      out = np.full((n,) + tail, np.nan)
      out[inside] = v.reshape((-1,) + tail)[src[inside]]
      prop[k] = out.reshape(shape + tail)
    args = (_vectors(P.reshape(shape + (3,)), self.frame), _rotations(rot.reshape(shape + (4,))), ph.reshape(shape), self.CSList, prop)
    kw = dict(scanUnit=self.scanUnit, frame=self.frame, phaseMap=self.phaseMap)
    return EBSD3(*args, **kw) if len(shape) == 3 else EBSD(*args, **kw)

  def slice(self, plane, P0=None):
    """the 2D map of the section of a volume through a plane; MATLAB's slice of EBSD3

      ebsd = ebsd3.slice(plane3d(vector3d.Z, vector3d(0, 0, 2)))  # a plane3d
      ebsd = ebsd3.slice(vector3d(1, 1, 1), vector3d(1, 1, 2))    # normal and a point on the plane
      ebsd = ebsd3.slice(vector3d.Z, 2.5)                         # normal and distance from the origin

    The volume is sampled on a square grid of the voxel spacing in the plane, spanned by the
    edge of the bounding box that lies most in the plane and the direction across it, the
    grid covering the section of the box. Every grid point takes the voxel holding it, as
    interp does; the points outside the volume are not indexed. The result is a gridded EBSD
    map whose positions lie in the plane.

    Input
      ebsd3 - EBSD3
      plane - plane3d, or the normal vector3d of the plane
    Output
      ebsd - EBSD, a gridded map in the plane
    Keywords
      P0 - with a normal: a point on the plane, a vector3d, or its distance from the origin;
          by default the plane through the origin
    See also
      EBSD3.interp, grain3d.slice, plane3d, EBSD3Plotting"""
    if not isinstance(plane, plane3d):
      plane = plane3d(plane, P0 if P0 is not None else 0)
    N = plane.N.normalize().data.reshape(3)
    ext = self.extent().reshape(3, 2)
    corners = np.array([[ext[0, a], ext[1, b], ext[2, c]] for c in (0, 1) for b in (0, 1) for a in (0, 1)])
    p0 = plane.project(vector3d(corners[0], frame=self.frame)).data.reshape(3)
    box = np.diag(ext[:, 1] - ext[:, 0])
    inPlane = box - (box @ N)[:, None] * N
    u = inPlane[np.argmax(np.linalg.norm(inPlane, axis=1))]
    u = u / np.linalg.norm(u)
    v = np.cross(N, u)
    step = self.dPos
    uv = (corners - p0) @ np.stack([u, v], axis=1) / step
    lo, hi = np.floor(columnMin(uv)) - 1, np.ceil(columnMax(uv)) + 1
    a, b = np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1)
    grid = p0 + step * (a[None, :, None] * u + b[:, None, None] * v)
    return self.interp(grid)

  # ------------------------------------------------------------------ display

  def __repr__(self):
    from ..geometry.display import sizeString
    from ..maps.phaselist import phaseTable
    lines = [f"EBSD3 ({frameLabel(self.frame)})"]
    if self.size == 0:
      return aligned(lines[0] + "\n  the volume is empty")
    lines.append(f"  size: {sizeString(self.shape)}" + (" grid" if self.isGridded else ""))
    # a count per phase in one pass, np.unique would sort every voxel; the shares are of the
    # measured voxels, the padding left out, as a map's
    perPhase = np.bincount(self.phaseId[~self.isPadding].astype(np.intp), minlength=1)
    ids = np.flatnonzero(perPhase)
    counts = perPhase[ids]
    share = [f"{c} ({float(f'{100 * c / (counts.sum() or 1):.2g}'):g}%)" for c in counts]
    lines.append(phaseTable(self.CSList.frames, ids, share))
    if self.prop:
      lines.append("  properties: " + ", ".join(sorted(self.prop)))
    lines.append(f"  scan unit: {self.scanUnit}")
    e = self.extent()
    lines.append(f"  X × Y × Z: [{e[0]:.4g} → {e[1]:.4g}] × [{e[2]:.4g} → {e[3]:.4g}] × [{e[4]:.4g} → {e[5]:.4g}]")
    if self.lattice is not None:
      lines.append(f"  voxel: {' × '.join(f'{s:.4g}' for s in self.lattice.spacing)}")
    return aligned("\n".join(lines))

  @classmethod
  def load(cls, fname, *args, **kwargs):
    """EBSD3.load('volume.dream3d'), EBSD3.load(['s0.ang', 's1.ang'], z=[0, 1])"""
    from ..io.load3d import loadEBSD3
    return loadEBSD3(fname, *args, **kwargs)

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from .intragranular3d import gradientX, gradientY, gradientZ, curvature, KAM, fill, spatialProfile, gradientDim
    from ..plotting.volumeplot import _plotEBSD3 as plot


def EBSD3square(rotations, phaseId=None, CSList=None, dxyz=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), scanUnit="um", frame=None,
                prop=None, D=None):
  """a volume on its (n1, n2, n3) array of voxels, on a rectangular lattice

    ebsd = EBSD3square(rot, phaseId, CSList, dxyz=(dx, dy, dz))           # the voxel size
    ebsd = EBSD3square(rot, phaseId, CSList, dxyz=0.5, origin=(10, 0, 0))  # a cube voxel, moved
    ebsd = EBSD3square(rot, phaseId, CSList, D=D)                          # any steps, as rows of D

  The voxel [i, j, k] sits at origin + i d1 + j d2 + k d3, the steps (dx, 0, 0), (0, dy, 0),
  (0, 0, dz) or the rows of D; the positions are built from them, as MATLAB's EBSD3square
  does from its voxel size. MATLAB's class EBSD3square is an EBSD3 of three dimensions here,
  ebsd.isGridded; MATLAB passes the positions and the phaseMap too.

  Input
    rotations - rotation of shape (n1, n2, n3)
    phaseId - int array of the same shape, the phase as the index into CSList, all 1 by default
    CSList - phaseList, or a list of crystalFrame with 'notIndexed' first
  Output
    ebsd - EBSD3
  Keywords
    dxyz - the voxel size (dx, dy, dz), or one number for all three
    origin - the position of the voxel [0, 0, 0]
    scanUnit - the unit of the positions
    frame - specimenFrame of the positions, by default the default one
    prop - dict, a property per voxel of the shape of the volume
    D - 3 x 3 array, the steps of the lattice as rows, instead of dxyz
  See also
    EBSD3, EBSDsquare, voxelLattice"""
  from ..geometry.quaternion import rotation
  from ..geometry._array import MtexArray
  rot = rotations if isinstance(rotations, MtexArray) else rotation(rotations)
  if rot.ndim != 3:
    raise ValueError("a volume is an (n1, n2, n3) array of rotations")
  D = np.diag(np.broadcast_to(np.asarray(dxyz, dtype=float), (3,))) if D is None else np.asarray(D, dtype=float)
  shape = rot.shape
  lat, pos = boxLattice(origin, D, shape)
  ph = np.ones(shape, dtype=np.int8) if phaseId is None else np.asarray(phaseId).reshape(shape)
  return EBSD3(pos, rot, ph, CSList, prop, scanUnit=scanUnit, frame=frame, lattice=lat)


__all__ = ["EBSD3", "EBSD3square"]
