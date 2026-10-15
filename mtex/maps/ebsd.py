# an EBSD map: a table of measurements with a geometry (MATLAB @EBSD, Julia ebsd.jl)
#
#   EBSD(pos, rotations, phaseId, CSList, prop)     a scattered list, one dimensional arrays
#   EBSD(..., unitCell=uC)                           with the lattice read off a cell polygon
#   gridify(ebsd)                                    the same map with its arrays in map shape
#   ebsd['Forsterite'], ebsd['indexed'], ebsd[mask], ebsd[i, j]   sub-maps
#   ebsd.orientations                                the framed orientations of the one phase present
#   ebsd.mad                                         a property, one value per measurement
#   ebsd.isPadding                                   the cells of a grid that hold no measurement
#
# The storage rank is the whole difference between MTEX's @EBSD and its @EBSDsquare and
# @EBSDhex: a gridded map holds its arrays in map shape, so ebsd[i, j] is a scan position,
# and every method is written once. What a map is measured on is the lattice, a statement
# about the positions and not about the storage. A map is not a container: indexing
# returns a map, and one measurement is a map of one.
#
# A grid turned against the axes of the plane is closed into a rectangle by padding, cells
# that were never measured: MATLAB gives them the phase NaN, here they are notIndexed and
# marked in isPadding. The extent, the phase counts, the phase selections and the plots
# leave them out, and nothing fills them.
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from ..geometry._array import MtexArray
from ..geometry.display import frameLabel, sizeString
from ..geometry.frames import crystalFrame, referenceFrame, specimenFrame
from ..geometry.kernels import qmul, qrot
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.vector3d import vector3d
from .dynprop import dynProp
from .lattice import lattice, latticeBasis, mapPlane
from .phaselist import phaseList, phaseTable
from ..geometry.display import aligned

FIELDS = ("pos", "rotations", "phaseId", "CSList", "lattice", "prop", "meta", "scanUnit", "phaseMap", "id", "_pad")


class EBSD:
  """an EBSD map: positions, rotations and phase ids with the list of the phases' frames and the
  properties of every measurement; MATLAB's EBSD

    ebsd = EBSD(pos, rot, phaseId, CSList)                      # the measurements as a list
    ebsd = EBSD(pos, rot, phaseId, CSList, {'mad': mad})        # with properties
    ebsd = EBSD(pos, rot, phaseId, CSList, prop, unitCell=uC)   # the shape of a pixel stated

  A map is a table of measurements, a row per pixel, or, gridified, the same arrays in map
  shape, ebsd[i, j] the pixel of row i and column j. phaseId indexes CSList, 0 the not indexed
  phase. A position that is NaN is dropped, as MATLAB drops it. The lattice of the positions,
  square or hexagonal, is read off the positions, or off the unit cell when it is given;
  loadEBSD, mtexdata and the simulations build maps the same way.

  Input
    pos - vector3d, or an (n, 2) or (n, 3) array, the centres of the pixels
    rotations - rotation, the rotation of every pixel
    phaseId - int array, the phase of every pixel as the index into CSList
    CSList - phaseList, or a list of crystalFrame with 'notIndexed' first
    prop - dict, a property per pixel of the shape of the map, ebsd.prop['mad'] and ebsd.mad
  Output
    ebsd - EBSD
  Keywords
    unitCell - vector3d or (k, 2) array, the corners of a pixel about its centre, from which
        the lattice is built
    scanUnit - the unit of the positions
    frame - specimenFrame of the positions, by default that of pos or the default one
    lattice - the lattice of the positions, when it is known, instead of reading it off them
    meta - dict, what a file states besides the measurements
    phaseMap - the phase number the file gives each entry of CSList
    id - the id of every pixel, by default its position in the list
    isPadding - the mask of the cells a grid adds to close it, MATLAB's NaN phase
  Properties
    pos - vector3d measurement positions in the specimen frame
    rotations - rotation coefficients of the measured orientations
    phaseId - zero-based index into CSList for each measurement; zero is notIndexed
    CSList - phaseList containing the phase frames and mineral metadata
    orientations - orientation data for a selection containing one indexed phase
    prop - additional arrays such as band contrast or grainId, also accessible as attributes
    id - measurement ids, distinct from positional array indices after selection
    lattice - geometry and spacing of a regular scan, or None for scattered data
    shape - measurement-array shape: a list or a spatially gridded array
    isPadding - mask of grid cells without an original measurement
    scanUnit - unit of the spatial coordinates
    grainId - grain assignments, available after calcGrains has written them to the map
  See also
    loadEBSD, EBSDsquare, gridify, simulateEBSD, EBSDImport, EBSDSelect, EBSDPlotting,
    GrainReconstruction"""

  def __init__(self, pos, rotations, phaseId, CSList, prop=None, unitCell=None, scanUnit="um", frame=None,
               lattice=None, meta=None, phaseMap=None, id=None, isPadding=None):
    if not isinstance(rotations, MtexArray):
      rotations = rotation(rotations)
    # single precision when either array is
    posDtype = pos.data.dtype if isinstance(pos, vector3d) else np.asarray(pos).dtype
    dtype = np.float32 if np.float32 in (rotations.dtype, posDtype) else np.float64
    if rotations.dtype != dtype:
      rotations = rotations.astype(dtype)
    fr = frame if frame is not None else (pos.frame if isinstance(pos, vector3d) and pos.frame is not None else specimenFrame.default)
    if not isinstance(fr, specimenFrame):
      raise TypeError("the positions of a map are in a specimen frame")
    xyz = pos.data if isinstance(pos, vector3d) else np.asarray(pos)
    if xyz.ndim == 2 and xyz.shape[1] == 2:
      xyz = np.concatenate([xyz, np.zeros(xyz.shape[:-1] + (1,))], axis=-1)
    xyz = np.ascontiguousarray(xyz, dtype=dtype)
    shape = xyz.shape[:-1]
    if rotations.shape != shape:
      raise ValueError(f"positions of shape {shape} and rotations of shape {rotations.shape} do not match")
    ph = np.asarray(phaseId)
    if ph.shape != shape:
      raise ValueError(f"the phase ids have shape {ph.shape}, the map {shape}")
    cs = CSList if isinstance(CSList, phaseList) else phaseList(CSList)
    if ph.size and (ph.min() < 0 or ph.max() >= len(cs)):
      raise ValueError(f"a phase id indexes CSList, so it runs from 0 to {len(cs) - 1}")
    self.pos = _vectors(xyz, fr)
    self.rotations = _rotations(np.ascontiguousarray(rotations.data))
    self.phaseId = ph.astype(np.int8)
    # the phase ids were checked against the list above; __setattr__ would check them again
    object.__setattr__(self, "CSList", cs)
    self.prop = dynProp()
    self.meta = {} if meta is None else dict(meta)
    self.scanUnit = scanUnit
    self.phaseMap = np.arange(len(cs)) if phaseMap is None else np.asarray(phaseMap)
    self.id = np.arange(ph.size).reshape(shape) if id is None else np.asarray(id).reshape(shape)
    self.lattice = None
    pad = None if isPadding is None else np.asarray(isPadding, dtype=bool).reshape(shape)
    self._pad = pad if pad is not None and pad.any() else None
    for k, v in (prop or {}).items():
      self._setProp(k, v)
    # the nan positions are dropped, as MATLAB does; a test over the whole array first, since
    # one along its short last axis costs ten times as much
    if np.isnan(xyz).any():
      bad = np.isnan(xyz).any(axis=-1)
      sub = self._subset(~bad)
      for name in FIELDS:
        object.__setattr__(self, name, getattr(sub, name))
      return
    if lattice is None and unitCell is not None:
      A, stencil, _ = latticeBasis(unitCell)
      lattice = _tryLattice(self.pos, basis=(A, "hex" if len(stencil) == 6 else "square"))
    self.lattice = _tryLattice(self.pos) if lattice is None else lattice

  # ------------------------------------------------------------------ the table

  @property
  def shape(self):
    """the shape of the map, (n,) for a list and (rows, columns) for a grid"""
    return self.rotations.shape

  @property
  def ndim(self):
    """the number of array axes, 1 for a list and 2 for a grid"""
    return self.rotations.ndim

  @property
  def size(self):
    """the number of measurements, the padding of a grid included"""
    return self.rotations.size

  def __len__(self):
    return self.size

  @property
  def isempty(self):
    """whether the map holds no measurement"""
    return self.size == 0

  @property
  def isGridded(self):
    """whether the arrays are in map shape, two dimensional"""
    return self.ndim == 2

  @property
  def dtype(self):
    """the floating point type of the rotations, single or double precision"""
    return self.rotations.dtype

  @property
  def frame(self):
    """the specimen frame of the positions"""
    return self.pos.frame

  @property
  def x(self):
    """the x coordinates of the positions"""
    return self.pos.x

  @property
  def y(self):
    """the y coordinates of the positions"""
    return self.pos.y

  @property
  def z(self):
    """the z coordinates of the positions"""
    return self.pos.z

  @property
  def xyz(self):
    """the positions as one array of shape shape + (3,)"""
    return self.pos.data

  @property
  def phase(self):
    """the phase numbers of the file, the phase ids translated through phaseMap"""
    return self.phaseMap[self.phaseId]

  def projectIntoFundamentalRegion(self):
    """the map with every orientation replaced by its representative in the fundamental
    region of its phase (MATLAB's EBSD.project2FundamentalRegion)"""
    out = self.copy()
    rot = out.rotations.data.reshape(-1, 4)
    for p in self.indexedPhaseId:
      sel = (self.phaseId == p).reshape(-1)
      if not np.any(sel):
        continue
      o = orientation._fromRotation(rot[sel], (self.CSList[int(p)], self.frame))
      rot[sel] = o.projectIntoFundamentalRegion().data.reshape(-1, 4)
    return out

  @property
  def how2plot(self):
    """the plotting convention of the specimen frame the map is written in"""
    from ..geometry.plottingconvention import plottingConvention
    pC = getattr(self.frame, "how2plot", None)
    return plottingConvention.default if pC is None else pC

  @property
  def isPadding(self):
    """the cells of a grid that hold no measurement"""
    return np.zeros(self.shape, dtype=bool) if self._pad is None else self._pad

  @property
  def isIndexed(self):
    """whether a measurement belongs to an indexed phase"""
    return self.CSList._indexedMask[self.phaseId]

  @property
  def indexedPhaseId(self):
    """the indexed phases at least one measurement belongs to"""
    present = np.unique(self.phaseId)
    return np.array([p for p in present if self.CSList.isIndexed(p)], dtype=int)

  @property
  def mineralList(self):
    """the mineral names of the phases, in the order of CSList"""
    return self.CSList.mineralList

  @property
  def colorList(self):
    """the colours of the phases, in the order of CSList"""
    return self.CSList.colorList

  @property
  def CS(self):
    """the crystal frame of the one indexed phase present"""
    return self.CSList[self._onePhase()]

  @CS.setter
  def CS(self, value):
    """ebsd['olivine'].CS = cs.Laue(): the frame of the one phase replaced, for every map sharing the list"""
    self.CSList.frames[self._onePhase()] = value

  def _onePhase(self):
    ids = self.indexedPhaseId
    if len(ids) != 1:
      raise ValueError(f"the map has {len(ids)} indexed phases, so name one: ebsd['phase']")
    return int(ids[0])

  @property
  # the orientations of the one indexed phase, one per measurement in the shape of the map:
  # a site of another phase or an empty one is NaN, as MATLAB keeps it
  def orientations(self):
    """the orientations of the one indexed phase present, framed by it and by the specimen"""
    p = self._onePhase()
    q = np.array(self.rotations.data, dtype=float)
    q[self.phaseId != p] = np.nan
    o = orientation._fromRotation(q, (self.CSList[p], self.frame))
    # neighbouring pixels repeat the orientation of their grain: not independent draws
    o.notIID = True
    return o

  # the properties, reachable as ebsd.mad next to the fields themselves
  def __getattr__(self, name):
    prop = self.__dict__.get("prop", {})
    if name in prop:
      return prop[name]
    if name.startswith("_") or not prop:
      raise AttributeError(name)
    raise AttributeError(f"an EBSD map has no property {name!r}; it carries " + ", ".join(repr(k) for k in prop))

  def __setattr__(self, name, value):
    # ebsd.CSList = [cs1, cs2]: a list of frames becomes the phase list, which every id must index
    if name == "CSList":
      value = value if isinstance(value, phaseList) else phaseList(value)
      if self.size and int(np.max(self.phaseId)) >= len(value):
        raise ValueError(f"a phase id indexes CSList, so it runs from 0 to {len(value) - 1}")
    if name in FIELDS or name.startswith("_") or isinstance(getattr(type(self), name, None), property):
      object.__setattr__(self, name, value)
    else:
      self._setProp(name, value)

  # a property is one value, or one row, per measurement; a container, a vector3d per
  # measurement say, is kept as it is
  def _setProp(self, name, value):
    v = value if isinstance(value, MtexArray) else np.asarray(value)
    if v.shape[:self.ndim] != self.shape:
      if v.ndim >= 1 and v.shape[0] == self.size:
        v = v.reshape(self.shape + v.shape[1:])
      else:
        raise ValueError(f"the property {name!r} has shape {v.shape}, the map {self.shape}")
    self.prop[name] = v

  # ------------------------------------------------------------------ selection

  # the names ebsd['...'] takes, which IPython completes inside the brackets
  def _ipython_key_completions_(self):
    return self.CSList.selectionNames()

  def __getitem__(self, key):
    if isinstance(key, str):
      return self._subset(self._phaseMask([key]))
    if isinstance(key, referenceFrame):
      return self._subset(self.phaseId == self.CSList.phaseId(key))
    if hasattr(key, "numPixel"):
      if "grainId" not in self.prop:
        raise ValueError("reconstruct grains with calcGrains(ebsd) before selecting their measurements")
      # the measurements of the grains, not the padding a filled hole gave a grain id
      return self._subset(np.isin(self.grainId, key.id) & ~self.isPadding)
    if isinstance(key, (list, tuple)) and key and isinstance(key[0], str):
      if key[0] == "xy":
        return self.findByLocation(np.asarray(key[1:], dtype=float))
      if key[0] == "id":
        return self._byId(key[1] if len(key) == 2 else np.asarray(key[1:]))
      return self._subset(self._phaseMask(key))
    if isinstance(key, tuple) and len(key) == 2 and self.ndim == 1:
      raise IndexError("ebsd[i, j] addresses a pixel of a gridded map; on a list ask for a position with ebsd['xy', x, y]")
    return self._subset(key)

  # ebsd[ind] = ori: the orientations of the measurements named and the phase of their frame,
  # MATLAB's ebsd(ind).orientations = ori
  def __setitem__(self, key, ori):
    flat = isinstance(key, np.ndarray) and key.ndim == 1 and self.ndim > 1
    ind = (np.arange(self.size)[key] if flat else np.arange(self.size).reshape(self.shape)[key]).reshape(-1)
    q = self.rotations.data.reshape(-1, 4)
    q[ind] = np.broadcast_to(ori.data.reshape(-1, 4), (len(ind), 4))
    self.phaseId.reshape(-1)[ind] = self.CSList.phaseId(ori.CS)

  def name2id(self, name):
    """the phase id of a mineral name, MATLAB's name2id"""
    return self.CSList.phaseId(name)

  def _phaseMask(self, names):
    return self.CSList._selectionMask(names, self.phaseId) & ~self.isPadding

  # the sub-map of a NumPy selection: every array sliced, the phases and the frame kept,
  # the lattice restricted to the sites kept
  # ebsd['id', M]: one measurement per entry of M, in its shape, as MATLAB returns them;
  # a single id gives one measurement
  def _byId(self, q):
    q = np.asarray(q)
    ids = np.asarray(self.id).reshape(-1)
    order = np.argsort(ids)
    pos = np.searchsorted(ids[order], q.reshape(-1))
    found = pos < len(ids)
    place = np.where(found, order[np.minimum(pos, len(ids) - 1)], 0)
    if not np.all(found) or not np.array_equal(ids[place], q.reshape(-1)):
      raise ValueError("the map holds no measurement of every id asked for")
    if self.ndim > 1:
      sel = tuple(i.reshape(q.shape) for i in np.unravel_index(place, self.shape))
    else:
      sel = place.reshape(q.shape) if q.ndim else int(place[0])
    return self._subset(sel)

  def _subset(self, sel):
    # a flat mask or a flat list of positions addresses a gridded map by its linear order
    if isinstance(sel, np.ndarray) and sel.ndim == 1 and self.ndim > 1:
      if sel.dtype == bool and sel.size == self.size:
        sel = sel.reshape(self.shape)
      elif np.issubdtype(sel.dtype, np.integer):
        sel = np.unravel_index(sel, self.shape)
    ids = np.arange(self.size).reshape(self.shape)[sel]
    if ids.ndim == 0:
      ids = ids.reshape(1)
    return self._take(ids.reshape(-1), ids.shape)

  # the measurements of the flat indices, in the given shape; the lattice follows them where
  # no measurement is taken twice
  def _take(self, flat, shape):
    pos = _vectors(self.pos.data.reshape(-1, 3)[flat].reshape(shape + (3,)), self.frame)
    rot = _rotations(self.rotations.data.reshape(-1, 4)[flat].reshape(shape + (4,)))
    ph = self.phaseId.reshape(-1)[flat].reshape(shape)
    prop = {k: v.reshape((-1,) + v.shape[self.ndim:])[flat].reshape(shape + v.shape[self.ndim:]) for k, v in self.prop.items()}
    l = self.lattice
    if l is not None:
      l = l.subset(flat) if len(flat) and _distinct(flat, self.size) else None
    pad = None if self._pad is None else self._pad.reshape(-1)[flat].reshape(shape)
    return EBSD(pos, rot, ph, self.CSList, prop, scanUnit=self.scanUnit, frame=self.frame, lattice=l, meta=self.meta, phaseMap=self.phaseMap,
                id=self.id.reshape(-1)[flat].reshape(shape), isPadding=pad)

  def copy(self):
    """a copy of the map with its own arrays; the phase list and the lattice are shared"""
    return EBSD(self.pos.copy(), self.rotations.copy(), self.phaseId.copy(), self.CSList,
                {k: v.copy() for k, v in self.prop.items()}, scanUnit=self.scanUnit, frame=self.frame,
                lattice=self.lattice, meta=dict(self.meta), phaseMap=self.phaseMap, id=self.id.copy(), isPadding=self._pad)

  @classmethod
  def cat(cls, *maps):
    """the list of several maps, the phases of the first, MATLAB's [ebsd1, ebsd2]

      ebsd = EBSD.cat(ebsd1, ebsd2, ebsd3)"""
    first = maps[0]
    for m in maps[1:]:
      if m.frame is not first.frame:
        raise ValueError("maps in different specimen frames cannot be joined")
      if set(m.prop) != set(first.prop):
        raise ValueError("maps with different properties cannot be joined")
    pos = np.concatenate([m.pos.data.reshape(-1, 3) for m in maps])
    rot = np.concatenate([m.rotations.data.reshape(-1, 4) for m in maps])
    ph = np.concatenate([m.phaseId.reshape(-1) for m in maps])
    prop = {k: np.concatenate([m.prop[k].reshape((-1,) + m.prop[k].shape[m.ndim:]) for m in maps]) for k in first.prop}
    pad = np.concatenate([m.isPadding.reshape(-1) for m in maps])
    return cls(_vectors(pos, first.frame), _rotations(rot), ph, first.CSList, prop, scanUnit=first.scanUnit,
               meta=first.meta, phaseMap=first.phaseMap, isPadding=pad)

  def flatten(self):
    """the map as a list, one dimensional arrays in the order of the grid"""
    return self if self.ndim == 1 else self._subset(np.unravel_index(np.arange(self.size), self.shape))

  # ------------------------------------------------------------------ finding measurements

  def findByLocation(self, xy, radius=None):
    """the measurements closest to positions, or within a radius of them; MATLAB's
    EBSD.findByLocation

      e = ebsd.findByLocation([10000, 5000])                       # the nearest measurement
      e = ebsd.findByLocation([[10000, 5000], [20000, 7000]])      # one per position
      e = ebsd.findByLocation([10000, 5000], radius=100)           # all within 100 map units

    A position is (x, y), (x, y, z) or a vector3d. Without a radius every position gives its
    nearest measurement, in the order of the positions; with one the measurements within it
    of any position are returned once each, in the order of the map.

    Input
      ebsd - EBSD
      xy - the positions, (n, 2), (n, 3) or vector3d
    Output
      e - EBSD, the measurements found as a list
    Keywords
      radius - the distance within which every measurement is taken; the nearest one only
        when not given

    Example
    Find the nearest measurement to a position and every measurement within a specified
    radius. The small sample is downloaded and cached on first use.

      import numpy as np
      from mtex import mtexdata, findByLocation

      ebsd = mtexdata('small', verbose=False)['Forsterite']
      query = np.array([np.mean(ebsd.x), np.mean(ebsd.y)])
      nearest = findByLocation(ebsd, query)
      nearby = findByLocation(ebsd, query, radius=30)
      print('Query (um):', np.round(query, 2))
      print('Nearest position (um):', np.round(nearest.pos.data.reshape(-1, 3)[0, :2], 2))
      print('Measurements within 30 um:', nearby.size)

    The result is an EBSD selection. Without radius the nearest measurement is returned even
    when the query is outside the scan; specify a radius when distance should limit the selection.

    See also
      EBSD.findByOrientation, EBSD.inpolygon"""
    q = xy.data if isinstance(xy, vector3d) else np.asarray(xy, dtype=float)
    q = q.reshape(-1, q.shape[-1])
    if q.shape[1] == 2:
      q = np.concatenate([q, np.zeros((len(q), 1))], axis=1)
    P = self.pos.data.reshape(-1, 3)
    if radius is None:
      ids = np.array([np.argmin(np.linalg.norm(P - p, axis=1)) for p in q])
    else:
      ids = np.unique(np.concatenate([np.flatnonzero(np.linalg.norm(P - p, axis=1) <= radius) for p in q]))
    return self._subset(ids)

  def findByOrientation(self, ori, epsilon):
    """select measurements close to a reference orientation or an orientation fibre

      selected = ebsd.findByOrientation(ori, 5*degree)
      selected = ebsd.findByOrientation(f, 5*degree)

    Only the reference's crystal phase is considered. Orientations within epsilon of the
    reference, or of any orientation on the fibre, are returned as a selected EBSD map.
    Distances account for crystal and specimen symmetry. The threshold is inclusive.

    Input
      ori - one reference orientation, or fibre, carrying the crystal phase
      epsilon - angular tolerance in radians
    Output
      selected - EBSD measurements satisfying the phase and angle criteria
    See also
      orientation.angle, fibre.angle, EBSD.inpolygon"""
    from ..geometry.fibre import fibre
    cs = ori.CS
    sel = self.phaseId == self.CSList.phaseId(cs)
    ids = np.flatnonzero(sel.reshape(-1))
    o = orientation._fromRotation(self.rotations.data.reshape(-1, 4)[ids], (cs, self.frame))
    d = ori.angle(o) if isinstance(ori, fibre) else o.angle(ori)
    return self._subset(ids[d <= epsilon])

  def inpolygon(self, region):
    """the mask of the measurements inside a rectangle [x, y, w, h] or a polygon (N, 2), edges included

      ind = ebsd.inpolygon([xmin, ymin, dx, dy])                # by a rectangle
      ind = inpolygon(ebsd, [[x1, y1], [x2, y2], [x3, y3]])     # by a polygon
      ebsd = ebsd[ind]"""
    return _inRegion(self.pos.x, self.pos.y, region)

  # ------------------------------------------------------------------ moving the map

  def rotate(self, rot, keepXY=False, keepEuler=False):
    """a rigid motion of the map; keepXY moves only the orientations, keepEuler only the positions;
    MATLAB's EBSD.rotate

      ebsd = ebsd.rotate(10*degree)                                 # about the z axis
      ebsd = rotate(ebsd, rotation.byAxisAngle(vector3d.X, 180*degree))  # about the x axis
      ebsd = rotate(ebsd, 180*degree, keepEuler=True)               # only the positions
      ebsd = ebsd.rotate(90*degree, keepXY=True)                    # only the orientations

    The positions and the orientations turn together, about the origin; an angle is a
    rotation about z. A gridded map keeps the order of its arrays: the positions move and the
    lattice turns with them, where MATLAB stores the map anew. rot * ebsd is the same as
    rotate(ebsd, rot). MATLAB's center= option, a centre other than the origin, is not taken.

    Input
      ebsd - EBSD
      rot - rotation, or an angle about the z axis
    Output
      ebsd - EBSD
    Keywords
      keepXY - turn only the orientations, the Euler angles
      keepEuler - turn only the positions
    See also
      EBSD.shift, EBSD.transform, transformReferenceFrame"""
    if keepXY and keepEuler:
      raise ValueError("choose either keepXY or keepEuler, not both")
    if not isinstance(rot, MtexArray):
      rot = rotation.byAxisAngle(vector3d.Z, rot)
    pos = self.pos.data if keepXY else qrot(rot.data, self.pos.data)
    r = self.rotations.data if keepEuler else qmul(rot.data, self.rotations.data)
    if keepXY:
      return self._moved(pos, r, self.lattice)
    return self._moved(pos, r, motion=(rotation(rot).matrix().reshape(3, 3), np.zeros(3)) if rot.size == 1 else None)

  # MATLAB's rot * ebsd: the positions and the orientations turned together
  def __rmul__(self, rot):
    return self.rotate(rot) if isinstance(rot, rotation) else NotImplemented

  def shift(self, v):
    """the map with every position moved by the vector v, the arrays not reordered"""
    v = v.data if isinstance(v, vector3d) else np.asarray(v, dtype=float)
    return self._moved(self.pos.data + v, self.rotations.data, motion=(np.eye(3), v) if np.size(v) == 3 else None)

  def transform(self, f):
    """any map of the positions; the lattice is inferred again from where they land"""
    out = f(self.pos)
    pos = out.data if isinstance(out, vector3d) else np.asarray(out, dtype=float)
    return self._moved(pos, self.rotations.data)

  # the map with its positions and rotations moved; a lattice given, or the old one carried
  # through the rigid motion (R, v) the positions went, else one inferred from where they land
  def _moved(self, pos, rot, lattice=None, motion=None):
    if lattice is None and motion is not None and self.lattice is not None:
      lattice = self.lattice.moved(*motion, mapPlane(pos.reshape(-1, 3)))
    return EBSD(_vectors(pos, self.frame), _rotations(rot), self.phaseId, self.CSList, self.prop,
                scanUnit=self.scanUnit, meta=self.meta, phaseMap=self.phaseMap, id=self.id, isPadding=self._pad, lattice=lattice)

  def extent(self):
    """the bounds of the measured positions, [xmin, xmax, ymin, ymax]; the padding of a grid is
    no measurement, and on a turned grid it reaches far beyond the map"""
    P = self.pos.data.reshape(-1, 3)
    if self._pad is not None:
      P = P[~self._pad.reshape(-1)]
    return np.array([P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()])

  # ------------------------------------------------------------------ the lattice

  @property
  def unitCell(self):
    """the cell of the lattice about a site, as its corners"""
    if self.lattice is None:
      raise ValueError("the positions are not on a lattice, so the map has no unit cell")
    return vector3d(self.lattice.unitCell, frame=self.frame)

  @property
  def N(self):
    """the normal of the measurement plane"""
    n = self.lattice.plane.normal if self.lattice is not None else np.array([0.0, 0.0, 1.0])
    return vector3d(n, frame=self.frame, isNormalized=True)

  @property
  def dPos(self):
    """MATLAB's spacing of the positions: the shortest distance from the first corner of the
    unit cell to the others, the step on a square lattice and the step over sqrt(3) on a
    hexagonal one, where lattice.dxy is the step"""
    if self.lattice is None:
      return np.nan
    c = self.lattice.unitCell
    return float(np.linalg.norm(c[1:] - c[0], axis=1).min())

  def gridify(self, layout=None, unitCell=None, return_index=False):
    """the same map with its arrays in map shape, the missing cells padding; MATLAB's gridify

      grid = ebsd.gridify()                                # the rows and columns of the scan
      grid, newId = ebsd.gridify(return_index=True)        # and where every measurement went
      grid = ebsd.gridify('rowMajor')                      # transposed, one scan column per array row
      grid = ebsd.gridify(gridLayout.rowMajor())           # in the order a layout states
      grid = ebsd.gridify(unitCell=2 * ebsd.unitCell)      # resampled onto a coarser lattice

    The grid closes the map to a full rectangle: a cell with no measurement, a pixel of another
    phase after ebsd['Forsterite'] or one outside a region, is padding, not indexed, with a NaN
    rotation, marked in ebsd.isPadding and left out of the plots, the counts and the grains.
    So the grid may hold more cells than the map had measurements. Nothing is interpolated.

    In the default, column major, layout the first index runs along the grid direction closest
    to y and the second along the one closest to x, both with increasing coordinates, ebsd[0, 0]
    the corner of the smallest coordinates. Another layout reorders the arrays by a transpose and
    flips only; a hexagonal grid is always stored column major. The gradients, the curvature,
    the GND and KAM do not need a grid, they run on the lattice of a list too.

    Input
      ebsd - EBSD, a list or a grid
      layout - gridLayout, or 'columnMajor' or 'rowMajor', the order of the arrays
    Output
      grid - EBSD in map shape
      newId - the flat index of every measurement in the grid, grid.pos.flatten()[newId] is
          ebsd.pos
    Keywords
      unitCell - the unit cell of another lattice, onto which the map is resampled by the
          nearest measurement
      return_index - return newId as well

    Example
    Arrange a small list of measurements into a grid, leaving one interior lattice site
    unmeasured. The index mapping identifies where each original measurement ended up.

      import numpy as np
      from mtex import EBSD, crystalFrame, phaseList, rotation, vector3d, degree, gridify

      x, y = np.meshgrid(np.arange(5), np.arange(4))
      keep = ~((x == 2) & (y == 1))
      pos = np.c_[x[keep], y[keep]]
      rot = rotation.byAxisAngle(vector3d.Z, x[keep]*degree)
      ebsd = EBSD(pos, rot, np.ones(len(pos), dtype=int), phaseList(crystalFrame('m-3m')))
      grid, newId = gridify(ebsd, return_index=True)
      print('List shape:', ebsd.shape, 'Grid shape:', grid.shape)
      print('Unmeasured cells:', np.count_nonzero(grid.isPadding))
      print('Padding is indexed:', bool(np.any(grid.isIndexed[grid.isPadding])))
      print('Positions preserved:', np.allclose(grid.pos.data.reshape(-1, 3)[newId], ebsd.pos.data))

    gridify arranges the data; it does not interpolate a measurement into the missing cell.
    Use isPadding to distinguish that absent lattice site from an acquired but unindexed point.

    See also
      gridLayout, EBSD.layout, EBSDGrid, transformReferenceFrame"""
    from .gridify import gridify
    return gridify(self, layout, unitCell, return_index)

  @property
  def layout(self):
    """the directions the indices of a gridded map run along (MATLAB ebsd.layout); a list has none"""
    if not self.isGridded:
      return None
    from ..geometry.frames.gridlayout import gridLayout
    from .gridify import arrayDirections
    d1, d2 = arrayDirections(self)
    gL = gridLayout(d1, d2)
    for name in ("columnMajor", "rowMajor"):
      if gL == getattr(gridLayout, name)():
        gL.name = name
    return gL

  @classmethod
  def load(cls, fname, *args, **kwargs):
    """read an EBSD map from a supported vendor file or a text table

      ebsd = EBSD.load('map.ang')
      ebsd = EBSD.load('map.ctf', grid=False)

    This calls loadEBSD with the same arguments. The file format is selected from the extension
    unless interface is given. Phase information, orientations, positions and available pixel
    properties are read together; complete scans are returned on their grid by default.
    For text-table column definitions and import corrections, consult loadEBSD. Use inspectEBSD
    to inspect file metadata before importing.

    Input
      fname - input filename
      args - positional options accepted by loadEBSD
      kwargs - keyword options accepted by loadEBSD
    Output
      ebsd - EBSD map

    Example
    Read the Forsterite vendor file directly and draw its phase map. The sample file is
    downloaded and cached on first use. The Euler correction shown here is specific to this file.

      import matplotlib.pyplot as plt
      from mtex import EBSD, mtexdatafile, rotation, vector3d, degree, plot

      filename = mtexdatafile('forsterite', verbose=False)
      correction = rotation.byAxisAngle(vector3d.X, 180*degree)
      ebsd = EBSD.load(filename, eulerCorrection=correction)
      print(ebsd)
      plot(ebsd, refFrame='on')
      plt.show()

    Inspect the phase list and the reference-frame indicator after import. For another file,
    choose the format options and Euler correction from its acquisition convention.

    See also
      loadEBSD, inspectEBSD, EBSD.export"""
    from ..io.load import loadEBSD
    return loadEBSD(fname, *args, **kwargs)

  @property
  def EulerCorrection(self):
    """the frame change the import applied to the Euler angles, the identity when none"""
    r = self.meta.get("EulerCorrection")
    return rotation.id if r is None else r

  @property
  def color(self):
    """the colour of the one phase of the map, read and set on its frame: ebsd['Diopside'].color"""
    from .phaselist import colorTriplet, phaseColorNames
    return colorTriplet(phaseColorNames(self.CSList.frames)[self._onePhasePresent()])

  @color.setter
  def color(self, value):
    """sets the colour of the one phase of the map on its frame"""
    self.CSList.frames[self._onePhasePresent()].color = value

  # the one phase, indexed or not, the measurements belong to
  def _onePhasePresent(self):
    ids = np.unique(self.phaseId)
    if len(ids) != 1:
      raise ValueError(f"the map has {len(ids)} phases, so name one: ebsd['phase']")
    return int(ids[0])

  def export(self, fname, *args, **kwargs):
    """write the EBSD map to a vendor format or a text table

      ebsd.export('map.ang')
      ebsd.export('map.ctf')
      ebsd.export('map.txt')

    The filename extension chooses the exporter. Orientations, positions, phase information
    and supported pixel properties are written together. Export reverses the import correction
    for the target vendor convention. HDF5 export uses a copy of the original vendor file;
    consult export for supported formats, required source metadata and options.

    Input
      fname - output filename
      args - positional export options
      kwargs - keyword export options
    See also
      export, EBSD.load, loadEBSD"""
    from ..io.exportebsd import export
    return export(self, fname, *args, **kwargs)

  # ------------------------------------------------------------------ display

  # one row per measurement: id, phase, the Euler angles in degree, the properties
  def _rowTable(self):
    from ..geometry.representations import quat2euler
    q = self.rotations.data.reshape(-1, 4)
    e = np.degrees(np.column_stack(quat2euler(q, "Bunge")))
    labels = ["Id", "Phase", "orientation"] + sorted(self.prop)
    rows = []
    for k in range(self.size):
      ori = "-" if not np.all(np.isfinite(e[k])) else "(" + ",".join(f"{a:.1f}°" for a in e[k]) + ")"
      row = [str(int(self.id.reshape(-1)[k])), str(int(self.phaseId.reshape(-1)[k])), ori]
      for key in sorted(self.prop):
        v = np.asarray(self.prop[key]).reshape(-1)[k]
        row.append(f"{v:.4g}" if np.issubdtype(np.asarray(v).dtype, np.floating) else str(v))
      rows.append(row)
    widths = [max(len(l), *(len(r[j]) for r in rows)) for j, l in enumerate(labels)]
    def line(r):
      return "  " + "  ".join(x.rjust(w) for x, w in zip(r, widths))
    return "\n".join([line(labels)] + [line(r) for r in rows])

  def __repr__(self):
    lines = [f"EBSD ({frameLabel(self.frame)})"]
    if self.size == 0:
      return aligned(lines[0] + "\n  the map is empty")
    lines.append(f"  size: {sizeString(self.shape)}" + (" grid" if self.isGridded else ""))
    # the shares are of all measurements, the padding left out, to two significant digits,
    # as MATLAB writes them
    ids, counts = np.unique(self.phaseId[~self.isPadding], return_counts=True)
    share = [f"{c} ({float(f'{100 * c / (counts.sum() or 1):.2g}'):g}%)" for c in counts]
    lines.append(phaseTable(self.CSList.frames, ids, share))
    # a short list shows its measurements, as MATLAB does up to twenty
    if self.size <= 20:
      lines.append(self._rowTable())
    elif self.prop:
      lines.append("  properties: " + ", ".join(sorted(self.prop)))
    lines.append(f"  scan unit: {self.scanUnit}")
    ext = self.extent()
    lines.append(f"  X × Y: [{ext[0]:.4g} → {ext[1]:.4g}] × [{ext[2]:.4g} → {ext[3]:.4g}]")
    if self.lattice is not None:
      lines.append(f"  {self.lattice.shape} lattice: spacing {self.lattice.dxy:.4g}")
    sets = self.meta.get("dataSets")
    if sets and len(sets) > 1:
      lines.append(f"  data set: {self.meta.get('dataSet')} of {len(sets)}")
    return aligned("\n".join(lines))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from .traces import calcTraces
    from .intragranular import (gradient, gradientX, gradientY, gradientZ, curvature, KAM, calcGROD, grainMean,
                                spatialProfile, calcMisorientation, weightedBurgersVec)
    from .hexgrid import hex2cube, cube2hex
    from ..analysis.dislocations import calcGND
    from ..analysis.denoising import smooth, fill, fillByGrainId, reduce, reduce as subsample, interp
    from ..plotting.mapplot import plotEBSD as plot, quiverMap as quiver


# a vector3d and a rotation on arrays as they are, without the cast of the constructors, so
# that a map of millions of measurements is not copied and a float32 map stays float32
def _vectors(xyz, frame):
  v = object.__new__(vector3d)
  v.data, v.antipodal, v.isNormalized, v._frame, v.dispStyle = np.asarray(xyz), False, False, frame, "xyz"
  return v


def _rotations(q):
  r = object.__new__(rotation)
  r.data = np.asarray(q)
  return r


# whether a list of positions into n measurements names none twice: one mark per measurement,
# since hashing the list costs a hundred times as much
def _distinct(flat, n):
  if len(flat) > n:
    return False
  if len(flat) < 2 or np.all(flat[1:] > flat[:-1]):
    return True
  seen = np.zeros(n, dtype=bool)
  seen[flat] = True
  return int(np.count_nonzero(seen)) == len(flat)


# the lattice of the positions, or None where they are not on one
def _tryLattice(pos, basis=None):
  if pos.size < 3:
    return None
  try:
    return lattice(pos, basis=basis)
  except (ValueError, np.linalg.LinAlgError):
    return None


# points inside a polygon, and on its edge unless told otherwise, by the crossing number
# the mask of the points (x, y) inside a rectangle [x, y, w, h] or a polygon (N, 2), edges included
def _inRegion(x, y, region):
  region = np.asarray(region, dtype=float)
  if region.ndim == 1:
    if len(region) != 4:
      raise ValueError("a rectangle is [xmin, ymin, width, height]")
    x0, y0, w, h = region
    if not np.all(np.isfinite(region)) or w < 0 or h < 0:
      raise ValueError("rectangle coordinates must be finite and its size nonnegative")
    return (x0 <= x) & (x <= x0 + w) & (y0 <= y) & (y <= y0 + h)
  if region.ndim != 2 or region.shape[1] != 2 or region.shape[0] < 3 or not np.all(np.isfinite(region)):
    raise ValueError("polygon vertices must be a finite N x 2 matrix with at least three rows")
  return _inPolygon(x, y, region)


def _inPolygon(x, y, V, includeEdge=True):
  x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
  inside = np.zeros(x.shape, dtype=bool)
  onEdge = np.zeros(x.shape, dtype=bool)
  n = len(V)
  # the extent sets the tolerance, at least eight ulps of the largest coordinate, so that a
  # polygon far from the origin, a map in nm say, still has an edge (MATLAB's insidepoly)
  tol = max(1e-12 * max(1.0, np.ptp(V, axis=0).max()), 8 * np.finfo(float).eps * np.abs(V).max()) if n else 0.0
  for i in range(n):
    (ax, ay), (bx, by) = V[i], V[(i + 1) % n]
    cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
    onEdge |= (np.abs(cross) <= tol * max(1.0, abs(bx - ax) + abs(by - ay))) & (np.minimum(ax, bx) - tol <= x) & (x <= np.maximum(ax, bx) + tol) \
      & (np.minimum(ay, by) - tol <= y) & (y <= np.maximum(ay, by) + tol)
    crosses = ((ay > y) != (by > y)) & (x < ax + (y - ay) * (bx - ax) / (by - ay + (by == ay)))
    inside ^= crosses
  return (inside | onEdge) if includeEdge else (inside & ~onEdge)


def EBSDsquare(rotations, phaseId=None, CSList=None, dxy=(1.0, 1.0), scanUnit="um", frame=None, prop=None):
  """a map on a square grid from arrays in map shape, the rows of the array its rows of pixels;
  MATLAB's EBSDsquare, used to build a synthetic map

    ebsd = EBSDsquare(rot, phaseId, CSList)                  # pixels of unit step
    ebsd = EBSDsquare(rot, CSList=CSList, dxy=0.5)           # one phase, another step
    ebsd = EBSDsquare(rot, phaseId, CSList, dxy=(1, 2))      # rectangular pixels

  Pixel (i, j) sits at x = j dx, y = i dy. The map is gridified, as a gridded map read from a
  file is.

  Input
    rotations - rotation of shape (rows, columns)
    phaseId - int array of the same shape, the index into CSList; phase 1 everywhere by
        default
    CSList - phaseList, or a list of crystalFrame with 'notIndexed' first
  Output
    ebsd - EBSD in map shape
  Keywords
    dxy - the step of the grid, one number or (dx, dy)
    scanUnit - the unit of the positions
    frame - specimenFrame of the positions
    prop - dict of the properties, each of the shape of the map
  See also
    EBSD, gridify, simulateEBSD, QuadruplePoints"""
  from .gridify import gridify
  rot = rotations if isinstance(rotations, MtexArray) else rotation(rotations)
  ny, nx = rot.shape
  dx, dy = (float(dxy), float(dxy)) if np.isscalar(dxy) else (float(dxy[0]), float(dxy[1]))
  y, x = np.mgrid[0:ny, 0:nx]
  pos = np.stack([x * dx, y * dy, np.zeros_like(x, dtype=float)], axis=-1)
  ph = np.ones(rot.shape, dtype=np.int8) if phaseId is None else np.asarray(phaseId).reshape(rot.shape)
  return gridify(EBSD(pos, rot, ph, CSList, prop, scanUnit=scanUnit, frame=frame))


__all__ = ["EBSD", "EBSDsquare"]
