# what a three dimensional plot shows, as NumPy arrays and without a renderer: polygon meshes
# with a colour per face, orthogonal slices of a voxel volume, and the camera of a plotting
# convention. PyVista, matplotlib and the HTML writer of render3d.py draw a scene.
import numpy as np

__all__ = ["scene3d", "mesh3d", "slices3d", "picture3d", "lines3d", "arrows3d", "grainMesh", "boundaryMesh", "volumeSlices"]


class mesh3d:
  """polygons over vertices: V (n, 3), the faces as ptr/idx (face k is idx[ptr[k]:ptr[k+1]]),
  a colour per face (m, 3) in [0, 1] or one colour for all, the face and edge transparency"""

  def __init__(self, V, ptr, idx, faceColor, faceAlpha=1.0, edgeColor=(0.0, 0.0, 0.0), edgeAlpha=1.0, lineWidth=0.5,
               values=None, colormap=None, colorRange=None):
    # only the vertices a face uses are kept
    used, idx = np.unique(np.asarray(idx, dtype=np.int64), return_inverse=True)
    self.V = np.asarray(V, dtype=float).reshape(-1, 3)[used]
    self.ptr, self.idx = np.asarray(ptr, dtype=np.int64), idx.astype(np.int64)
    fc = np.asarray(faceColor, dtype=float)
    self.faceColor = np.clip(fc.reshape(-1, 3) if fc.size != 3 else np.tile(fc.reshape(3), (len(self), 1)), 0, 1)
    self.faceAlpha, self.edgeAlpha, self.lineWidth = float(faceAlpha), float(edgeAlpha), float(lineWidth)
    self.edgeColor = None if edgeColor is None or (isinstance(edgeColor, str) and edgeColor == "none") else _rgb(edgeColor)
    # the values the colours came from, for a colour bar
    self.values, self.colormap, self.colorRange = values, colormap, colorRange

  def __len__(self):
    return len(self.ptr) - 1

  @property
  def numCorners(self):
    return np.diff(self.ptr)

  # the fan triangles of the faces, (k, 3), and the face each belongs to
  def triangles(self):
    n = self.numCorners - 2
    face = np.repeat(np.arange(len(self)), n)
    start = np.repeat(self.ptr[:-1], n)
    j = np.arange(len(face)) - np.repeat(np.cumsum(n) - n, n)
    return np.stack([self.idx[start], self.idx[start + j + 1], self.idx[start + j + 2]], axis=1), face

  # the edges of the polygons, every one once, (e, 2)
  def edges(self):
    k = np.arange(len(self.idx)) + 1
    k[self.ptr[1:] - 1] = self.ptr[:-1]
    E = np.sort(np.stack([self.idx, self.idx[k]], axis=1), axis=1)
    return np.unique(E, axis=0)

  def bounds(self):
    return self.V.min(axis=0), self.V.max(axis=0)


class slices3d:
  """three orthogonal slices of a voxel volume: the colours (nx, ny, nz, 3) in [0, 1], the
  voxels shown (nx, ny, nz), the centre of the first voxel, the steps of the three voxel axes
  as the rows of D, and the voxel index of each slice, None for a slice not shown"""

  def __init__(self, rgb, visible, origin, D, at=None):
    self.rgb = np.clip(np.asarray(rgb, dtype=float), 0, 1)
    self.visible = np.asarray(visible, dtype=bool)
    self.origin, self.D = np.asarray(origin, dtype=float).reshape(3), np.asarray(D, dtype=float).reshape(3, 3)
    self.at = tuple(s // 2 for s in self.visible.shape) if at is None else tuple(at)

  # the slice through voxel index i along axis a: its colours (rows, cols, 3) over the other
  # two axes in increasing index, the voxels shown, and the corners of its rectangle, the
  # first at index (-1/2, -1/2), then along the rows, the columns, both
  def plane(self, a):
    i = self.at[a]
    rgb, vis = np.take(self.rgb, i, axis=a), np.take(self.visible, i, axis=a)
    b, c = [k for k in range(3) if k != a]
    n = self.visible.shape
    corners = np.zeros((4, 3))
    for k, (u, v) in enumerate(((-0.5, -0.5), (n[b] - 0.5, -0.5), (-0.5, n[c] - 0.5), (n[b] - 0.5, n[c] - 0.5))):
      corners[k, a], corners[k, b], corners[k, c] = i, u, v
    return rgb, vis, self.origin + corners @ self.D

  def bounds(self):
    n = np.array(self.visible.shape)
    box = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) * n - 0.5
    P = self.origin + box @ self.D
    return P.min(axis=0), P.max(axis=0)


class picture3d:
  """a picture on a rectangle in space: RGBA (rows, cols, 4) in [0, 1] with its first row at
  the bottom, and the corners of the rectangle at the bottom left, the bottom right, the top
  left and the top right; a two dimensional map drawn in the plane it lies in"""

  def __init__(self, rgba, corners):
    self.rgba = np.clip(np.asarray(rgba, dtype=float), 0, 1)
    self.corners = np.asarray(corners, dtype=float).reshape(4, 3)

  def bounds(self):
    return self.corners.min(axis=0), self.corners.max(axis=0)


class lines3d:
  """line segments (k, 2, 3), a colour per segment (k, 3) or one for all, their width"""

  def __init__(self, segments, color=(0.0, 0.0, 0.0), lineWidth=1.0, alpha=1.0):
    self.segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    c = np.asarray(_rgb(color) if isinstance(color, str) else color, dtype=float)
    self.color = np.clip(np.tile(c.reshape(3), (len(self.segments), 1)) if c.size == 3 else c.reshape(-1, 3), 0, 1)
    self.lineWidth, self.alpha = float(lineWidth), float(alpha)

  def bounds(self):
    P = self.segments.reshape(-1, 3)
    return P.min(axis=0), P.max(axis=0)


# arrows from P along D as their shafts and cones for their heads, a quarter of their length
# long, as quiver3 draws them; antipodal arrows point both ways and have no heads
def arrows3d(P, D, color=(0.0, 0.0, 1.0), lineWidth=1.0, antipodal=False, headSize=0.25, sides=12):
  P, D = np.asarray(P, dtype=float).reshape(-1, 3), np.asarray(D, dtype=float).reshape(-1, 3)
  c = np.asarray(_rgb(color) if isinstance(color, str) else color, dtype=float)
  c = np.tile(c.reshape(3), (len(P), 1)) if c.size == 3 else c.reshape(-1, 3)
  ok = np.all(np.isfinite(D), axis=1) & (np.linalg.norm(D, axis=1) > 0)
  P, D, c = P[ok], D[ok], c[ok]
  if antipodal:
    return [lines3d(np.stack([P - D, P + D], axis=1), c, lineWidth)]
  L = np.linalg.norm(D, axis=1)
  u = D / L[:, None]
  base = P + D * (1 - headSize)
  shafts = lines3d(np.stack([P, base], axis=1), c, lineWidth)
  # two directions across every arrow, and the rim of its cone
  a = np.where(np.abs(u[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
  e1 = np.cross(u, a)
  e1 /= np.linalg.norm(e1, axis=1)[:, None]
  e2 = np.cross(u, e1)
  t = 2 * np.pi * np.arange(sides) / sides
  r = (0.35 * headSize * L)[:, None, None]
  rim = base[:, None, :] + r * (np.cos(t)[None, :, None] * e1[:, None, :] + np.sin(t)[None, :, None] * e2[:, None, :])
  n = len(P)
  V = np.concatenate([rim.reshape(-1, 3), P + D, base])
  k = np.arange(sides)
  tip, centre = n * sides + np.arange(n), n * sides + n + np.arange(n)
  side = np.stack([np.arange(n)[:, None] * sides + k, np.arange(n)[:, None] * sides + (k + 1) % sides, np.repeat(tip[:, None], sides, 1)], axis=-1)
  bottom = np.stack([np.arange(n)[:, None] * sides + (k + 1) % sides, np.arange(n)[:, None] * sides + k, np.repeat(centre[:, None], sides, 1)], axis=-1)
  T = np.concatenate([side, bottom], axis=1).reshape(-1, 3)
  heads = mesh3d(V, np.arange(0, 3 * len(T) + 1, 3), T.reshape(-1), np.repeat(c, 2 * sides, axis=0), edgeColor=None)
  return [shafts, heads]


class scene3d:
  """the items of a three dimensional plot and the plotting convention it is seen by"""

  def __init__(self, how2plot=None, scanUnit="um"):
    self.items, self.how2plot, self.scanUnit = [], how2plot, scanUnit

  def add(self, item):
    self.items.append(item)
    return item

  def bounds(self):
    if not self.items:
      return np.zeros(3), np.ones(3)
    b = [it.bounds() for it in self.items]
    return np.min([lo for lo, _ in b], axis=0), np.max([hi for _, hi in b], axis=0)

  # the camera: the direction out of the screen and the one up, unit vectors
  def camera(self):
    from ..geometry.plottingconvention import plottingConvention
    pC = plottingConvention.default3D if self.how2plot is None else self.how2plot
    o = pC.outOfScreen.normalize().data.reshape(3)
    up = pC.north.normalize().data.reshape(3)
    return o, up


def _rgb(c):
  from matplotlib.colors import to_rgb
  return np.array(to_rgb(c) if isinstance(c, str) else np.asarray(c, dtype=float).reshape(3)[:3], dtype=float)


# the colour of every element: the values mapped by the colour map, or the colours as they are
def _faceColors(values, isScalar, colormap, colorRange):
  if not isScalar:
    return values, None, None
  from .colormaps import mtexColormap
  v = np.asarray(values, dtype=float)
  finite = v[np.isfinite(v)]
  lo, hi = (finite.min(), finite.max()) if colorRange is None and finite.size else (0.0, 1.0) if colorRange is None else colorRange
  if not hi > lo:
    hi = lo + max(abs(lo), 1.0) * 1e-6
  rgb = mtexColormap(colormap)(np.clip((v - lo) / (hi - lo), 0, 1))[:, :3]
  rgb[~np.isfinite(v)] = np.nan
  return rgb, (lo, hi), colormap


def grainMesh(grains, color=None, faceColor=None, faceAlpha=1.0, edgeColor="k", edgeAlpha=1.0, lineWidth=0.5, colormap=None,
              colorRange=None, ipfDirection=None):
  """the outer faces of the grains, each in the colour of the grain it bounds, as MATLAB's
  grain3d/plot; a face between two of the grains is inside and drawn grey only when the faces
  are translucent"""
  from .mapplot import mapValues
  gB = grains.boundary
  I = grains.I_GF.tocsc()
  # a face is outer when it belongs to one grain of the set, inner when to two
  owner = np.full(I.shape[1], -1, dtype=np.int64)
  C = I.tocoo()
  count = np.bincount(C.col, minlength=I.shape[1])
  owner[C.col] = C.row
  outer = count == 1
  inner = count == 2
  if faceColor is not None and color is None:
    rgb, rng = np.tile(_rgb(faceColor), (len(grains), 1)), None
    values = None
  else:
    values, isScalar = mapValues(grains, color, ipfDirection=ipfDirection)
    rgb, rng, _ = _faceColors(values, isScalar, colormap, colorRange)
    values = values if isScalar else None
  keep = outer | (inner if faceAlpha < 1 else False)
  faces = np.flatnonzero(keep)
  fc = np.full((len(faces), 3), 0.5)
  isOuter = outer[faces]
  fc[isOuter] = rgb[owner[faces[isOuter]]]
  # a face without a colour, a value NaN, is not drawn
  drawn = np.all(np.isfinite(fc), axis=1)
  faces, fc = faces[drawn], fc[drawn]
  ptr, idx = _subFaces(gB, faces)
  return mesh3d(gB.allV, ptr, idx, fc, faceAlpha, edgeColor, edgeAlpha, lineWidth,
                values=None if values is None else values, colormap=colormap, colorRange=rng)


def boundaryMesh(gB, color=None, faceColor=None, faceAlpha=1.0, edgeColor="k", edgeAlpha=1.0, lineWidth=0.5, colormap=None, colorRange=None):
  """the faces of a 3D grain boundary, light blue, in one colour, or coloured by a value or a
  colour per face, as MATLAB's grain3Boundary/plot"""
  m = len(gB)
  if color is None:
    rgb, rng, values = np.tile(_rgb("lightblue" if faceColor is None else faceColor), (m, 1)), None, None
  else:
    c = np.asarray(color, dtype=float)
    isScalar = c.size == m
    if not isScalar and c.size != 3 * m and c.size != 3:
      raise ValueError(f"one value or one colour per face is needed, {m} of them")
    values = c.reshape(-1) if isScalar else (np.tile(c.reshape(3), (m, 1)) if c.size == 3 else c.reshape(-1, 3))
    rgb, rng, _ = _faceColors(values, isScalar, colormap, colorRange)
    values = values if isScalar else None
  faces = np.flatnonzero(np.all(np.isfinite(rgb), axis=1))
  ptr, idx = _subFaces(gB, faces)
  return mesh3d(gB.allV, ptr, idx, rgb[faces], faceAlpha, edgeColor, edgeAlpha, lineWidth, values=values, colormap=colormap, colorRange=rng)


def volumeSlices(ebsd, color=None, faceColor=None, colormap=None, colorRange=None, ipfDirection=None, at=None):
  """three orthogonal slices through the voxels of an EBSD volume, coloured by phase, by a
  value, a colour or an orientation per voxel, as MATLAB's EBSD3/plot shows the volume in
  its slice planes; voxels not indexed or without a value are white"""
  from .mapplot import mapValues
  g, newId = ebsd.gridify(return_index=True)
  shape = g.lattice.shape
  n = int(np.prod(shape))
  values, isScalar = mapValues(ebsd, color, faceColor=faceColor, ipfDirection=ipfDirection)
  rgb, rng, _ = _faceColors(values, isScalar, colormap, colorRange)
  full = np.ones((n, 3))
  vis = np.zeros(n, dtype=bool)
  full[newId] = np.where(np.isfinite(rgb), rgb, 1.0)
  isIdx = np.asarray(ebsd.isIndexed).reshape(-1)
  vis[newId] = isIdx & np.all(np.isfinite(rgb), axis=1)
  # a phase plot shows the notIndexed voxels in their own colour, as a map does
  if color is None and faceColor is None:
    vis[newId] = True
  full[~vis] = 1.0
  l = g.lattice
  s = slices3d(full.reshape(shape + (3,)), vis.reshape(shape), l.origin, l.D, at)
  s.values, s.colormap, s.colorRange = (values if isScalar else None), colormap, rng
  return s


# the polygons of some faces of a boundary as ptr/idx
def _subFaces(gB, faces):
  counts = gB.numCorners[faces]
  ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
  start = np.repeat(gB._ptr[faces], counts)
  j = np.arange(ptr[-1]) - np.repeat(ptr[:-1], counts)
  return ptr, gB._idx[start + j]
