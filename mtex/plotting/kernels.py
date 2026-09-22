# the compiled kernels of the drawing: the colour of every orientation of a map read off the
# grid of its key, and the polar coordinates of a sector fold that the key is built from,
# each a loop compiled by numba on its first call. Where numba is not there, or the
# preference 'kernels' asks for NumPy, the loop reports that it did nothing and the caller
# takes its own array route, which is the oracle of the tests. The pattern is the
# geometry's, mtex/geometry/kernels.py
from __future__ import annotations

import numpy as np

from ..geometry.kernels import _jit, _loop, backend

_COMPILED = {}
prange = range


# the colour loop compiled over the cores: a map holds millions of measurements and each
# one's colour is read on its own, so the loop splits over them; `_loop` compiles the
# sector fold, which runs once per key and is short
def _parallel(name, f):
  if name not in _COMPILED:
    nb = _jit()
    global prange
    prange = nb.prange
    _COMPILED[name] = nb.njit(cache=True, parallel=True)(f)
  return _COMPILED[name]


# ------------------------------------------------------------------ the colour of a map

# the colour of every measurement of one phase: the crystal direction along the specimen
# direction, then the node of the key's grid it falls in, which is arithmetic on a grid of
# uniform steps - the ring from the polar angle, the place in the ring from the azimuth.
# A measurement without an orientation is white, as the grid's own route colours it.
def _ipfGridColorsLoop(q, sub, d, theta0, dtheta, first, step, count, offset, colors, out):
  nTheta = first.shape[0]
  for t in prange(sub.shape[0]):
    i = sub[t]
    a, b, c, e = q[i, 0], -q[i, 1], -q[i, 2], -q[i, 3]
    s = 2 * (d[0] * b + d[1] * c + d[2] * e)
    a2 = 2 * a
    an = a * a - (b * b + c * c + e * e)
    x = a2 * (c * d[2] - d[1] * e) + s * b + an * d[0]
    y = a2 * (e * d[0] - d[2] * b) + s * c + an * d[1]
    z = a2 * (b * d[1] - d[0] * c) + s * e + an * d[2]
    r = np.sqrt(x * x + y * y + z * z)
    if not (r > 0) or not np.isfinite(r):
      out[i, 0], out[i, 1], out[i, 2] = 1.0, 1.0, 1.0
      continue
    w = z / r
    if w > 1.0:
      w = 1.0
    elif w < -1.0:
      w = -1.0
    k = int(np.round((np.arccos(w) - theta0) / dtheta))
    if k < 0:
      k = 0
    elif k > nTheta - 1:
      k = nTheta - 1
    if count[k] == 1:
      j = 0
    else:
      j = int(np.round((np.arctan2(y, x) - first[k]) / step[k])) % count[k]
    node = offset[k] + j
    out[i, 0], out[i, 1], out[i, 2] = colors[node, 0], colors[node, 1], colors[node, 2]


# whether the compiled route ran; the caller colours through the grid itself when it did not
def ipfGridColors(q, sub, d, ladders, colors, out):
  if backend() == "numpy" or ladders is None:
    return False
  theta0, dtheta, first, step, count, offset = ladders
  if not (dtheta > 0):
    return False
  q = np.ascontiguousarray(q, dtype=np.float64).reshape(-1, 4)
  sub = np.ascontiguousarray(sub, dtype=np.int64).reshape(-1)
  d = np.ascontiguousarray(d, dtype=np.float64).reshape(3)
  colors = np.ascontiguousarray(colors, dtype=np.float64).reshape(-1, 3)
  _parallel("ipfGridColors", _ipfGridColorsLoop)(q, sub, d, float(theta0), float(dtheta),
                                                np.ascontiguousarray(first, dtype=np.float64),
                                                np.ascontiguousarray(step, dtype=np.float64),
                                                np.ascontiguousarray(count, dtype=np.int64),
                                                np.ascontiguousarray(offset, dtype=np.int64), colors, out)
  return True


# ------------------------------------------------------------------ the polar radius

# one over the geodesic through the centre: the boundary crossing of that circle sets the
# scale, so the radius is one at the centre and zero where the circle leaves the region.
# Every quantity here is a cross product and an angle of one direction, which as a loop
# needs none of the (n, 3) temporaries the array form builds per boundary.
def _polarRadiusLoop(N, xyz, c, out):
  for i in range(xyz.shape[0]):
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
  _loop("polarRadius", _polarRadiusLoop)(N, xyz, c, out)
  return out


__all__ = []
