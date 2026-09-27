# the compiled kernels of the drawing: the colour of every orientation of a map read off the
# cube map of its key, and the polar coordinates of a sector fold that the key is built from,
# each a loop compiled by numba on its first call. Where numba is not there, or the
# preference 'kernels' asks for NumPy, the loop reports that it did nothing and the caller
# takes its own array route, which is the oracle of the tests. The pattern is the
# geometry's, mtex/geometry/kernels.py
from __future__ import annotations

import numpy as np

from ..geometry.kernels import _jit, backend, threads

_COMPILED = {}
prange = range


# the loops compiled over the cores, run on the threads of the preference: a map holds
# millions of measurements and each one's colour, or polar radius, is read on its own, so
# the loop splits over them
def _parallel(name, f):
  nb = _jit()
  if name not in _COMPILED:
    global prange
    prange = nb.prange
    _COMPILED[name] = nb.njit(cache=True, parallel=True)(f)
  nb.set_num_threads(min(threads(), nb.config.NUMBA_NUM_THREADS))
  return _COMPILED[name]


# ------------------------------------------------------------------ the colour of a map

def cubeMapNodes(n):
  """the key of a map is read off a cube map of the sphere: a direction falls on the face of its
  largest component, and its place there is the other two components over that one, in
  [-1, 1] and cut into n cells a side. The node is two divisions away, where a grid of rings
  needs an arccos and an arctan2. Face f is the axis f // 2, positive for even f; its cells
  run over the other two axes in increasing order"""
  t = (np.arange(n) + 0.5) / n * 2 - 1
  U, V = np.meshgrid(t, t, indexing="ij")
  faces = []
  for axis in range(3):
    u, v = [a for a in range(3) if a != axis]
    for sign in (1.0, -1.0):
      p = np.empty((n, n, 3))
      p[..., axis], p[..., u], p[..., v] = sign, U, V
      faces.append(p.reshape(-1, 3))
  p = np.concatenate(faces)
  return p / np.linalg.norm(p, axis=1, keepdims=True)


def cubeMapIndex(xyz, n):
  """the node of every direction, -1 for a zero or NaN one; the array form of the loop below"""
  xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
  a = np.abs(xyz)
  axis = np.argmax(a, axis=1)
  rows = np.arange(len(xyz))
  m = a[rows, axis]
  ok = np.isfinite(m) & (m > 0)
  face = 2 * axis + (xyz[rows, axis] <= 0)
  u = np.where(ok, xyz[rows, np.where(axis == 0, 1, 0)], 0.0)
  v = np.where(ok, xyz[rows, np.where(axis == 2, 1, 2)], 0.0)
  mm = np.where(ok, m, 1.0)
  iu = np.clip(((u / mm + 1) * 0.5 * n).astype(np.int64), 0, n - 1)
  iv = np.clip(((v / mm + 1) * 0.5 * n).astype(np.int64), 0, n - 1)
  return np.where(ok, (face * n + iu) * n + iv, -1)


# the colour of every measurement of one phase: the crystal direction along the specimen
# direction, then its node of the cube map. A measurement without an orientation takes the
# colour `blank`, white on a map.
def _ipfGridColorsLoop(q, sub, d, n, colors, blank, out):
  for t in prange(sub.shape[0]):
    i = sub[t]
    a, b, c, e = q[i, 0], -q[i, 1], -q[i, 2], -q[i, 3]
    s = 2 * (d[0] * b + d[1] * c + d[2] * e)
    a2 = 2 * a
    an = a * a - (b * b + c * c + e * e)
    x = a2 * (c * d[2] - d[1] * e) + s * b + an * d[0]
    y = a2 * (e * d[0] - d[2] * b) + s * c + an * d[1]
    z = a2 * (b * d[1] - d[0] * c) + s * e + an * d[2]
    ax, ay, az = abs(x), abs(y), abs(z)
    if ax >= ay and ax >= az:
      f, m, u, v = (0 if x > 0 else 1), ax, y, z
    elif ay >= az:
      f, m, u, v = (2 if y > 0 else 3), ay, x, z
    else:
      f, m, u, v = (4 if z > 0 else 5), az, x, y
    if not (m > 0) or not np.isfinite(m):
      out[i, 0], out[i, 1], out[i, 2] = blank[0], blank[1], blank[2]
      continue
    iu = min(max(int((u / m + 1) * 0.5 * n), 0), n - 1)
    iv = min(max(int((v / m + 1) * 0.5 * n), 0), n - 1)
    node = (f * n + iu) * n + iv
    out[i, 0], out[i, 1], out[i, 2] = colors[node, 0], colors[node, 1], colors[node, 2]


def ipfGridColors(q, sub, d, n, colors, out, blank=(1.0, 1.0, 1.0)):
  """whether the compiled route ran; the caller colours through the grid itself when it did not"""
  if backend() == "numpy":
    return False
  q = np.ascontiguousarray(q, dtype=np.float64).reshape(-1, 4)
  sub = np.ascontiguousarray(sub, dtype=np.int64).reshape(-1)
  d = np.ascontiguousarray(d, dtype=np.float64).reshape(3)
  colors = np.ascontiguousarray(colors, dtype=np.float64).reshape(-1, 3)
  _parallel("ipfGridColors", _ipfGridColorsLoop)(q, sub, d, int(n), colors, np.asarray(blank, dtype=np.float64).reshape(3), out)
  return True


# ------------------------------------------------------------------ the polar radius

# one over the geodesic through the centre: the boundary crossing of that circle sets the
# scale, so the radius is one at the centre and zero where the circle leaves the region.
# Every quantity here is a cross product and an angle of one direction, which as a loop
# needs none of the (n, 3) temporaries the array form builds per boundary.
def _polarRadiusLoop(N, xyz, c, out):
  for i in prange(xyz.shape[0]):
    x, y, z = xyz[i, 0], xyz[i, 1], xyz[i, 2]
    # the pole of the geodesic through the direction and the centre
    wx = y * c[2] - z * c[1]
    wy = z * c[0] - x * c[2]
    wz = x * c[1] - y * c[0]
    nw = np.sqrt(wx * wx + wy * wy + wz * wz)
    if nw < 1e-10:
      out[i] = 1.0
      continue
    wx, wy, wz = wx / nw, wy / nw, wz / nw
    best = np.inf
    for k in range(N.shape[0]):
      # where that geodesic crosses the boundary
      bx = wy * N[k, 2] - wz * N[k, 1]
      by = wz * N[k, 0] - wx * N[k, 2]
      bz = wx * N[k, 1] - wy * N[k, 0]
      nb = np.sqrt(bx * bx + by * by + bz * bz)
      if nb <= 1e-10:
        continue
      bx, by, bz = bx / nb, by / nb, bz / nb
      # the angle from the crossing to the direction, against the one to the centre
      cx, cy, cz = y * bz - z * by, z * bx - x * bz, x * by - y * bx
      a1 = np.arctan2(np.sqrt(cx * cx + cy * cy + cz * cz), x * bx + y * by + z * bz)
      cx, cy, cz = c[1] * bz - c[2] * by, c[2] * bx - c[0] * bz, c[0] * by - c[1] * bx
      a2 = np.arctan2(np.sqrt(cx * cx + cy * cy + cz * cz), c[0] * bx + c[1] * by + c[2] * bz)
      bottom = np.pi - a2
      d = 1.0 if bottom == 0.0 else (np.pi - a1) / bottom
      if d != d:
        d = 1.0
      if d < best:
        best = d
    out[i] = best if np.isfinite(best) else 1.0


def _polarRadiusNumpy(N, xyz, c):
  angle = lambda u, v: np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), np.sum(u * v, axis=-1))
  w = np.cross(xyz, c)
  nw = np.linalg.norm(w, axis=-1)
  atCenter = nw < 1e-10
  vxc = w / np.where(atCenter, 1.0, nw)[:, None]
  r = np.full(len(xyz), np.inf)
  for n in N:
    b = np.cross(vxc, n)
    nb = np.linalg.norm(b, axis=-1)
    ok = nb > 1e-10
    if not np.any(ok):
      continue
    bc = b[ok] / nb[ok][:, None]
    d = (np.pi - angle(xyz[ok], bc)) / (np.pi - angle(c, bc))
    d = np.where(np.isnan(d), 1.0, d)
    r[ok] = np.minimum(r[ok], d)
  r = np.where(np.isfinite(r), r, 1.0)
  return np.where(atCenter, 1.0, r)


def polarRadius(N, xyz, c):
  xyz = np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1, 3)
  c = np.ascontiguousarray(c, dtype=np.float64).reshape(3)
  N = np.ascontiguousarray(np.asarray(N, dtype=float).reshape(-1, 3))
  if backend() == "numpy":
    return _polarRadiusNumpy(N, xyz, c)
  out = np.empty(len(xyz))
  _parallel("polarRadius", _polarRadiusLoop)(N, xyz, c, out)
  return out


__all__ = []
