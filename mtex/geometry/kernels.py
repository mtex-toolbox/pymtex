# the compiled kernels: the Hamilton product, the rotation of vectors, the searches of the
# symmetry reduction and the projections built on them, each a loop over elements compiled
# by numba on its first call, with a NumPy form that is the fallback without numba and the
# oracle of the tests. numba is imported on the first call, never at import; the preference
# 'kernels' ('numba' or 'numpy') picks the backend
from __future__ import annotations

import numpy as np

from .session import getMTEXpref

_numba = None        # the module once imported, False when it is not installed
_compiled = {}       # the loops by name, compiled on first use


def _jit():
  global _numba
  if _numba is None:
    try:
      import numba
      _numba = numba
    except ImportError:
      _numba = False
  return _numba


def backend():
  return "numba" if getMTEXpref("kernels", "numba") == "numba" and _jit() else "numpy"


def _loop(name, f):
  if name not in _compiled:
    _compiled[name] = _numba.njit(cache=True, nogil=True)(f)
  return _compiled[name]


# the floating type two operands share, float64 for anything else
def _floatType(*arrays):
  dtypes = {a.dtype for a in arrays}
  return dtypes.pop() if len(dtypes) == 1 and next(iter({a.dtype.kind for a in arrays})) == "f" else np.float64


# ------------------------------------------------------------------ the Hamilton product

def _qmulLoop(p, q, out):
  for i in range(p.shape[0]):
    a0, a1, a2, a3 = p[i, 0], p[i, 1], p[i, 2], p[i, 3]
    b0, b1, b2, b3 = q[i, 0], q[i, 1], q[i, 2], q[i, 3]
    out[i, 0] = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
    out[i, 1] = a1 * b0 + a0 * b1 - a3 * b2 + a2 * b3
    out[i, 2] = a2 * b0 + a3 * b1 + a0 * b2 - a1 * b3
    out[i, 3] = a3 * b0 - a2 * b1 + a1 * b2 + a0 * b3


def _qmulNumpy(p, q):
  a1, b1, c1, d1 = np.moveaxis(p, -1, 0)
  a2, b2, c2, d2 = np.moveaxis(q, -1, 0)
  return np.stack([
    a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
    b1 * a2 + a1 * b2 - d1 * c2 + c1 * d2,
    c1 * a2 + d1 * b2 + a1 * c2 - b1 * d2,
    d1 * a2 - c1 * b2 + b1 * c2 + a1 * d2], axis=-1)


# the Hamilton product on trailing axes of four, broadcast on the leading ones (quaternion/times)
def qmul(p, q):
  p, q = np.asarray(p), np.asarray(q)
  if backend() == "numpy":
    return _qmulNumpy(p, q)
  dtype = _floatType(p, q)
  p, q = np.broadcast_arrays(p, q)
  shape = p.shape
  p = np.ascontiguousarray(p, dtype=dtype).reshape(-1, 4)
  q = np.ascontiguousarray(q, dtype=dtype).reshape(-1, 4)
  out = np.empty_like(p)
  _loop("qmul", _qmulLoop)(p, q, out)
  return out.reshape(shape)


# ------------------------------------------------------------------ Euler angles to quaternions

# Rz(alpha) Ry(beta) Rz(gamma) as a quaternion, one row per triple
def _eulerQuatLoop(alpha, beta, gamma, out):
  for i in range(alpha.shape[0]):
    hP = beta[i] / 2
    hp = (alpha[i] + gamma[i]) / 2
    hm = (alpha[i] - gamma[i]) / 2
    cP = np.cos(hP)
    sP = np.sin(hP)
    out[i, 0] = cP * np.cos(hp)
    out[i, 1] = -sP * np.sin(hm)
    out[i, 2] = sP * np.cos(hm)
    out[i, 3] = cP * np.sin(hp)


# the quaternions of Matthies angles of one shape, or None on the NumPy backend, whose form
# is representations.matthies2quat
def eulerQuat(alpha, beta, gamma):
  if backend() == "numpy":
    return None
  shape = alpha.shape
  a = np.ascontiguousarray(alpha, dtype=np.float64).reshape(-1)
  b = np.ascontiguousarray(beta, dtype=np.float64).reshape(-1)
  g = np.ascontiguousarray(gamma, dtype=np.float64).reshape(-1)
  out = np.empty((a.shape[0], 4))
  _loop("eulerQuat", _eulerQuatLoop)(a, b, g, out)
  return out.reshape(shape + (4,))


# ------------------------------------------------------------------ the rotation of vectors

def _qrotLoop(q, v, out):
  for i in range(q.shape[0]):
    a, b, c, d = q[i, 0], q[i, 1], q[i, 2], q[i, 3]
    x, y, z = v[i, 0], v[i, 1], v[i, 2]
    s = 2 * (x * b + y * c + z * d)
    a2 = 2 * a
    an = a * a - (b * b + c * c + d * d)
    out[i, 0] = a2 * (c * z - y * d) + s * b + an * x
    out[i, 1] = a2 * (d * x - z * b) + s * c + an * y
    out[i, 2] = a2 * (b * y - x * c) + s * d + an * z


def _qrotNumpy(q, xyz):
  a, b, c, d = np.moveaxis(q, -1, 0)
  x, y, z = np.moveaxis(xyz, -1, 0)
  n = b * b + c * c + d * d
  s = 2 * (x * b + y * c + z * d)
  a2, an = 2 * a, a * a - n
  return np.stack([
    a2 * (c * z - y * d) + s * b + an * x,
    a2 * (d * x - z * b) + s * c + an * y,
    a2 * (b * y - x * c) + s * d + an * z], axis=-1)


# q v q* on trailing axes of four and three, broadcast on the leading ones (vector3d/rotate)
def qrot(q, xyz):
  q, xyz = np.asarray(q), np.asarray(xyz)
  if backend() == "numpy":
    return _qrotNumpy(q, xyz)
  dtype = _floatType(q, xyz)
  shape = np.broadcast_shapes(q.shape[:-1], xyz.shape[:-1])
  q = np.ascontiguousarray(np.broadcast_to(q, shape + (4,)), dtype=dtype).reshape(-1, 4)
  xyz = np.ascontiguousarray(np.broadcast_to(xyz, shape + (3,)), dtype=dtype).reshape(-1, 3)
  out = np.empty_like(xyz)
  _loop("qrot", _qrotLoop)(q, xyz, out)
  return out.reshape(shape + (3,))


# ------------------------------------------------------------------ the image search

def _bestDotLoop(qs, es, val, idx):
  for i in range(qs.shape[0]):
    q0, q1, q2, q3 = qs[i, 0], qs[i, 1], qs[i, 2], qs[i, 3]
    best = -1.0
    k = 0
    for j in range(es.shape[0]):
      d = abs(q0 * es[j, 0] + q1 * es[j, 1] + q2 * es[j, 2] + q3 * es[j, 3])
      if d > best:
        best = d
        k = j
    val[i] = best
    idx[i] = k


def _bestDotNumpy(qs, es):
  d = np.abs(qs.reshape(-1, 4) @ es.T)
  k = np.argmax(d, axis=1)
  return np.take_along_axis(d, k[:, None], axis=1)[:, 0].reshape(qs.shape[:-1]), k.reshape(qs.shape[:-1])


# the best |<q, e>| over the images e and where it is, for every q of a list, the first
# maximum on a tie
def bestDot(qs, es):
  if backend() == "numpy":
    return _bestDotNumpy(qs, es)
  shape = qs.shape[:-1]
  qs = np.ascontiguousarray(qs, dtype=np.float64).reshape(-1, 4)
  es = np.ascontiguousarray(es, dtype=np.float64)
  val, idx = np.empty(len(qs)), np.empty(len(qs), dtype=np.int64)
  _loop("bestDot", _bestDotLoop)(qs, es, val, idx)
  return val.reshape(shape), idx.reshape(shape)


# ------------------------------------------------------------------ the pair search

# for each pair the reduced |<q1, l q2 r>| maximised over L and R (and inv(q2) under
# exchange) with the (l, r, usedInverse) attaining it: <q1, l q2 r> = <inv(q2) inv(l) q1, r>,
# two products per l and the dots against R; the (useInv, l, r) scan keeps the first maximum
def _symReduceLoop(q1, q2, L, R, exchange, vals, li, rj, usedInv):
  for i in range(q1.shape[0]):
    a0, a1, a2, a3 = q1[i, 0], q1[i, 1], q1[i, 2], q1[i, 3]
    best = -1.0
    bl, br, binv = 0, 0, False
    for useInv in range(2 if exchange else 1):
      # inv(q2), or of its inverse under exchange
      s = -1.0 if useInv else 1.0
      c0, c1, c2, c3 = q2[i, 0], -s * q2[i, 1], -s * q2[i, 2], -s * q2[i, 3]
      for l in range(L.shape[0]):
        l0, l1, l2, l3 = L[l, 0], -L[l, 1], -L[l, 2], -L[l, 3]
        # t = inv(l) q1
        t0 = l0 * a0 - l1 * a1 - l2 * a2 - l3 * a3
        t1 = l1 * a0 + l0 * a1 - l3 * a2 + l2 * a3
        t2 = l2 * a0 + l3 * a1 + l0 * a2 - l1 * a3
        t3 = l3 * a0 - l2 * a1 + l1 * a2 + l0 * a3
        # m = inv(q2) t
        m0 = c0 * t0 - c1 * t1 - c2 * t2 - c3 * t3
        m1 = c1 * t0 + c0 * t1 - c3 * t2 + c2 * t3
        m2 = c2 * t0 + c3 * t1 + c0 * t2 - c1 * t3
        m3 = c3 * t0 - c2 * t1 + c1 * t2 + c0 * t3
        for r in range(R.shape[0]):
          d = abs(m0 * R[r, 0] + m1 * R[r, 1] + m2 * R[r, 2] + m3 * R[r, 3])
          if d > best:
            best = d
            bl, br, binv = l, r, useInv == 1
    vals[i] = best
    li[i] = bl
    rj[i] = br
    usedInv[i] = binv


def _symReduceNumpy(q1, q2, L, R, exchange):
  n = len(q1)
  vals = np.full(n, -np.inf)
  li, rj, usedInv = np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
  for useInv in ((False, True) if exchange else (False,)):
    b = q2 * [1, -1, -1, -1] if useInv else q2
    bi = b * [1, -1, -1, -1]
    for l in range(len(L)):
      m = _qmulNumpy(bi, _qmulNumpy((L[l] * [1, -1, -1, -1])[None, :], q1))
      d = np.abs(m @ R.T)
      r = np.argmax(d, axis=1)
      best = np.take_along_axis(d, r[:, None], axis=1)[:, 0]
      better = best > vals
      vals[better], li[better], rj[better], usedInv[better] = best[better], l, r[better], useInv
  return vals, li, rj, usedInv


def symReduce(q1, q2, L, R, exchange=False):
  q1, q2 = np.broadcast_arrays(q1, q2)
  shape = q1.shape[:-1]
  q1 = np.ascontiguousarray(q1, dtype=np.float64).reshape(-1, 4)
  q2 = np.ascontiguousarray(q2, dtype=np.float64).reshape(-1, 4)
  L, R = np.ascontiguousarray(L, dtype=np.float64), np.ascontiguousarray(R, dtype=np.float64)
  if backend() == "numpy":
    vals, li, rj, usedInv = _symReduceNumpy(q1, q2, L, R, exchange)
  else:
    n = len(q1)
    vals, li, rj, usedInv = np.empty(n), np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64), np.empty(n, dtype=bool)
    _loop("symReduce", _symReduceLoop)(q1, q2, L, R, bool(exchange), vals, li, rj, usedInv)
  return vals.reshape(shape), li.reshape(shape), rj.reshape(shape), usedInv.reshape(shape)


# ------------------------------------------------------------------ the pair search with its image

# the pair search, and the image l q2 r (of inv(q2) under exchange) it chose, with the sign
# that makes <q1, image> nonnegative: the step the projection onto a list of references,
# the pair angle and the mean take
def _closestImageLoop(q1, q2, L, R, exchange, vals, out):
  for i in range(q1.shape[0]):
    a0, a1, a2, a3 = q1[i, 0], q1[i, 1], q1[i, 2], q1[i, 3]
    best = -1.0
    bl, br, binv = 0, 0, False
    for useInv in range(2 if exchange else 1):
      s = -1.0 if useInv else 1.0
      c0, c1, c2, c3 = q2[i, 0], -s * q2[i, 1], -s * q2[i, 2], -s * q2[i, 3]
      for l in range(L.shape[0]):
        l0, l1, l2, l3 = L[l, 0], -L[l, 1], -L[l, 2], -L[l, 3]
        t0 = l0 * a0 - l1 * a1 - l2 * a2 - l3 * a3
        t1 = l1 * a0 + l0 * a1 - l3 * a2 + l2 * a3
        t2 = l2 * a0 + l3 * a1 + l0 * a2 - l1 * a3
        t3 = l3 * a0 - l2 * a1 + l1 * a2 + l0 * a3
        m0 = c0 * t0 - c1 * t1 - c2 * t2 - c3 * t3
        m1 = c1 * t0 + c0 * t1 - c3 * t2 + c2 * t3
        m2 = c2 * t0 + c3 * t1 + c0 * t2 - c1 * t3
        m3 = c3 * t0 - c2 * t1 + c1 * t2 + c0 * t3
        for r in range(R.shape[0]):
          d = abs(m0 * R[r, 0] + m1 * R[r, 1] + m2 * R[r, 2] + m3 * R[r, 3])
          if d > best:
            best = d
            bl, br, binv = l, r, useInv == 1
    vals[i] = best
    # the image l b r
    s = -1.0 if binv else 1.0
    b0, b1, b2, b3 = q2[i, 0], s * q2[i, 1], s * q2[i, 2], s * q2[i, 3]
    l0, l1, l2, l3 = L[bl, 0], L[bl, 1], L[bl, 2], L[bl, 3]
    t0 = l0 * b0 - l1 * b1 - l2 * b2 - l3 * b3
    t1 = l1 * b0 + l0 * b1 - l3 * b2 + l2 * b3
    t2 = l2 * b0 + l3 * b1 + l0 * b2 - l1 * b3
    t3 = l3 * b0 - l2 * b1 + l1 * b2 + l0 * b3
    r0, r1, r2, r3 = R[br, 0], R[br, 1], R[br, 2], R[br, 3]
    p0 = t0 * r0 - t1 * r1 - t2 * r2 - t3 * r3
    p1 = t1 * r0 + t0 * r1 - t3 * r2 + t2 * r3
    p2 = t2 * r0 + t3 * r1 + t0 * r2 - t1 * r3
    p3 = t3 * r0 - t2 * r1 + t1 * r2 + t0 * r3
    if p0 * a0 + p1 * a1 + p2 * a2 + p3 * a3 < 0:
      p0, p1, p2, p3 = -p0, -p1, -p2, -p3
    out[i, 0], out[i, 1], out[i, 2], out[i, 3] = p0, p1, p2, p3


def _closestImageNumpy(q1, q2, L, R, exchange):
  vals, li, rj, usedInv = _symReduceNumpy(q1, q2, L, R, exchange)
  b = np.where(usedInv[:, None], q2 * [1, -1, -1, -1], q2)
  p = _qmulNumpy(_qmulNumpy(L[li], b), R[rj])
  return vals, np.where((np.sum(p * q1, axis=-1) < 0)[:, None], -p, p)


def closestImage(q1, q2, L, R, exchange=False):
  q1, q2 = np.broadcast_arrays(q1, q2)
  shape = q1.shape[:-1]
  q1 = np.ascontiguousarray(q1, dtype=np.float64).reshape(-1, 4)
  q2 = np.ascontiguousarray(q2, dtype=np.float64).reshape(-1, 4)
  L, R = np.ascontiguousarray(L, dtype=np.float64), np.ascontiguousarray(R, dtype=np.float64)
  if backend() == "numpy":
    vals, out = _closestImageNumpy(q1, q2, L, R, exchange)
  else:
    vals, out = np.empty(len(q1)), np.empty_like(q1)
    _loop("closestImage", _closestImageLoop)(q1, q2, L, R, bool(exchange), vals, out)
  return vals.reshape(shape), out.reshape(shape + (4,))


# ------------------------------------------------------------------ the projection onto one reference

# every rotation moved to its equivalent closest to one reference: the image e of the
# reference closest to q, taken from the table frm of the (l, r, usedInverse) each image came
# from, gives the representative inv(l) q inv(r), signed towards the reference; around the
# identity the conjugate inv(c) p c by the shared rotation c whose image of the axis sector
# centre is closest to the axis of p decides between the tied conjugates
def _projectToRefLoop(qs, es, frm, L, R, ref, common, cimgs, out):
  for i in range(qs.shape[0]):
    q0, q1, q2, q3 = qs[i, 0], qs[i, 1], qs[i, 2], qs[i, 3]
    best = -1.0
    k = 0
    for j in range(es.shape[0]):
      d = abs(q0 * es[j, 0] + q1 * es[j, 1] + q2 * es[j, 2] + q3 * es[j, 3])
      if d > best:
        best = d
        k = j
    li, rj = frm[k, 0], frm[k, 1]
    if frm[k, 2] != 0:
      q1, q2, q3 = -q1, -q2, -q3
    l0, l1, l2, l3 = L[li, 0], -L[li, 1], -L[li, 2], -L[li, 3]
    t0 = l0 * q0 - l1 * q1 - l2 * q2 - l3 * q3
    t1 = l1 * q0 + l0 * q1 - l3 * q2 + l2 * q3
    t2 = l2 * q0 + l3 * q1 + l0 * q2 - l1 * q3
    t3 = l3 * q0 - l2 * q1 + l1 * q2 + l0 * q3
    r0, r1, r2, r3 = R[rj, 0], -R[rj, 1], -R[rj, 2], -R[rj, 3]
    p0 = t0 * r0 - t1 * r1 - t2 * r2 - t3 * r3
    p1 = t1 * r0 + t0 * r1 - t3 * r2 + t2 * r3
    p2 = t2 * r0 + t3 * r1 + t0 * r2 - t1 * r3
    p3 = t3 * r0 - t2 * r1 + t1 * r2 + t0 * r3
    if p0 * ref[0] + p1 * ref[1] + p2 * ref[2] + p3 * ref[3] < 0:
      p0, p1, p2, p3 = -p0, -p1, -p2, -p3
    if common.shape[0] > 1:
      nv = np.sqrt(p1 * p1 + p2 * p2 + p3 * p3)
      if 2 * np.arctan2(nv, abs(p0)) >= 1e-8:
        s = -1.0 if p0 < 0 else 1.0
        x, y, z = s * p1 / nv, s * p2 / nv, s * p3 / nv
        best = -2.0
        k = 0
        for j in range(cimgs.shape[0]):
          d = x * cimgs[j, 0] + y * cimgs[j, 1] + z * cimgs[j, 2]
          if d > best:
            best = d
            k = j
        c0, c1, c2, c3 = common[k, 0], common[k, 1], common[k, 2], common[k, 3]
        # inv(c) p c
        t0 = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3
        t1 = -c1 * p0 + c0 * p1 + c3 * p2 - c2 * p3
        t2 = -c2 * p0 - c3 * p1 + c0 * p2 + c1 * p3
        t3 = -c3 * p0 + c2 * p1 - c1 * p2 + c0 * p3
        p0 = t0 * c0 - t1 * c1 - t2 * c2 - t3 * c3
        p1 = t1 * c0 + t0 * c1 - t3 * c2 + t2 * c3
        p2 = t2 * c0 + t3 * c1 + t0 * c2 - t1 * c3
        p3 = t3 * c0 - t2 * c1 + t1 * c2 + t0 * c3
    out[i, 0], out[i, 1], out[i, 2], out[i, 3] = p0, p1, p2, p3


def _projectToRefNumpy(qs, es, frm, L, R, ref, common, cimgs):
  _, k = _bestDotNumpy(qs, es)
  li, rj, useInv = frm[k, 0], frm[k, 1], frm[k, 2].astype(bool)
  q = np.where(useInv[:, None], qs * [1, -1, -1, -1], qs)
  p = _qmulNumpy(_qmulNumpy(L[li] * [1, -1, -1, -1], q), R[rj] * [1, -1, -1, -1])
  p = np.where((np.sum(p * ref, axis=-1) < 0)[:, None], -p, p)
  if len(common) > 1:
    small = 2 * np.arctan2(np.linalg.norm(p[:, 1:], axis=-1), np.abs(p[:, 0])) < 1e-8
    a = np.where(p[:, :1] < 0, -1.0, 1.0) * p[:, 1:]
    a = a / np.where(small, 1.0, np.linalg.norm(a, axis=-1))[:, None]
    k = np.argmax(a @ cimgs.T, axis=1)
    conj = _qmulNumpy(_qmulNumpy(common[k] * [1, -1, -1, -1], p), common[k])
    p = np.where(small[:, None], p, conj)
  return p


def projectToRef(qs, es, frm, L, R, ref, common=None, cimgs=None):
  shape = qs.shape[:-1]
  qs = np.ascontiguousarray(qs, dtype=np.float64).reshape(-1, 4)
  es, frm = np.ascontiguousarray(es, dtype=np.float64), np.ascontiguousarray(frm, dtype=np.int64)
  L, R = np.ascontiguousarray(L, dtype=np.float64), np.ascontiguousarray(R, dtype=np.float64)
  ref = np.ascontiguousarray(ref, dtype=np.float64)
  common = np.empty((0, 4)) if common is None else np.ascontiguousarray(common, dtype=np.float64)
  cimgs = np.empty((0, 3)) if cimgs is None else np.ascontiguousarray(cimgs, dtype=np.float64)
  if backend() == "numpy":
    out = _projectToRefNumpy(qs, es, frm, L, R, ref, common, cimgs)
  else:
    out = np.empty_like(qs)
    _loop("projectToRef", _projectToRefLoop)(qs, es, frm, L, R, ref, common, cimgs, out)
  return out.reshape(shape + (4,))


# ------------------------------------------------------------------ the projection of directions

# every direction moved into the sector: the group element whose image of the centre is
# closest to the direction, applied inversely, with its parity
def _projectDirectionsLoop(xyz, imgs, q, parity, out):
  for i in range(xyz.shape[0]):
    x, y, z = xyz[i, 0], xyz[i, 1], xyz[i, 2]
    best = -2.0
    k = 0
    for j in range(imgs.shape[0]):
      d = x * imgs[j, 0] + y * imgs[j, 1] + z * imgs[j, 2]
      if d > best:
        best = d
        k = j
    a, b, c, d = q[k, 0], -q[k, 1], -q[k, 2], -q[k, 3]
    s = 2 * (x * b + y * c + z * d)
    a2 = 2 * a
    an = a * a - (b * b + c * c + d * d)
    out[i, 0] = (a2 * (c * z - y * d) + s * b + an * x) * parity[k]
    out[i, 1] = (a2 * (d * x - z * b) + s * c + an * y) * parity[k]
    out[i, 2] = (a2 * (b * y - x * c) + s * d + an * z) * parity[k]


def _projectDirectionsNumpy(xyz, imgs, q, parity):
  k = np.argmax(xyz @ imgs.T, axis=1)
  return _qrotNumpy(q[k] * [1, -1, -1, -1], xyz) * parity[k][:, None]


# ------------------------------------------------------------------ the projection into a region

# every direction moved into a region by the group: the first image of it that lies inside,
# the direction itself when none does. A region is a few planes, so the test is a dot
# product against each and the loop leaves as soon as one image is in - which is why this
# is a loop and not the (n, |G|, 3) block of images the NumPy form has to build.
def _projectIntoSectorLoop(xyz, els, N, alpha, regionAnti, vAnti, tol, out):
  m, p = els.shape[0], N.shape[0]
  for i in range(xyz.shape[0]):
    x, y, z = xyz[i, 0], xyz[i, 1], xyz[i, 2]
    out[i, 0], out[i, 1], out[i, 2] = x, y, z
    n = np.sqrt(x * x + y * y + z * z)
    done = False
    for side in range(2):
      if done or (side == 1 and not vAnti):
        break
      sign = 1.0 if side == 0 else -1.0
      for j in range(m):
        a, b, c, d, par = els[j, 0], els[j, 1], els[j, 2], els[j, 3], els[j, 4]
        s = 2 * (x * b + y * c + z * d)
        a2 = 2 * a
        an = a * a - (b * b + c * c + d * d)
        tx = sign * par * (a2 * (c * z - y * d) + s * b + an * x)
        ty = sign * par * (a2 * (d * x - z * b) + s * c + an * y)
        tz = sign * par * (a2 * (b * y - x * c) + s * d + an * z)
        ok = True
        for k in range(p):
          if tx * N[k, 0] + ty * N[k, 1] + tz * N[k, 2] < (alpha[k] - tol) * n:
            ok = False
            break
        if not ok and regionAnti:
          ok = True
          for k in range(p):
            if -(tx * N[k, 0] + ty * N[k, 1] + tz * N[k, 2]) < (alpha[k] - tol) * n:
              ok = False
              break
        if ok:
          out[i, 0], out[i, 1], out[i, 2] = tx, ty, tz
          done = True
          break


def _projectIntoSectorNumpy(xyz, els, N, alpha, regionAnti, vAnti, tol):
  out = xyz.copy()
  inRegion = lambda w: (np.ones(w.shape[:-1], dtype=bool) if len(N) == 0 else
                        np.all(w @ N.T >= (alpha - tol) * np.linalg.norm(w, axis=-1)[..., None], axis=-1))
  # the images in chunks, so that the block of every image of every direction stays small
  step = max(1, 200000 // max(len(els), 1))
  for k in range(0, len(xyz), step):
    x = xyz[k:k + step]
    imgs = _qrotNumpy(els[None, :, :4], x[:, None, :]) * els[None, :, 4:5]
    inside = inRegion(imgs)
    if regionAnti:
      inside |= inRegion(-imgs)
    hit = np.any(inside, axis=1)
    chosen = imgs[np.arange(len(x)), np.argmax(inside, axis=1)]
    if vAnti:
      insideNeg = inRegion(-imgs)
      if regionAnti:
        insideNeg |= inside
      hitNeg = ~hit & np.any(insideNeg, axis=1)
      chosen = np.where(hitNeg[:, None], -imgs[np.arange(len(x)), np.argmax(insideNeg, axis=1)], chosen)
      hit |= hitNeg
    out[k:k + step] = np.where(hit[:, None], chosen, x)
  return out


def projectIntoSector(xyz, els, N, alpha, regionAnti=False, vAnti=False, tol=1e-4):
  shape = xyz.shape[:-1]
  xyz = np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1, 3)
  els = np.ascontiguousarray(els, dtype=np.float64).reshape(-1, 5)
  N = np.ascontiguousarray(N, dtype=np.float64).reshape(-1, 3)
  alpha = np.ascontiguousarray(alpha, dtype=np.float64).reshape(-1)
  if backend() == "numpy":
    out = _projectIntoSectorNumpy(xyz, els, N, alpha, bool(regionAnti), bool(vAnti), float(tol))
  else:
    out = np.empty_like(xyz)
    _loop("projectIntoSector", _projectIntoSectorLoop)(xyz, els, N, alpha, bool(regionAnti), bool(vAnti), float(tol), out)
  return out.reshape(shape + (3,))


def projectDirections(xyz, imgs, q, parity):
  shape = xyz.shape[:-1]
  xyz = np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1, 3)
  imgs, q = np.ascontiguousarray(imgs, dtype=np.float64), np.ascontiguousarray(q, dtype=np.float64)
  parity = np.ascontiguousarray(parity, dtype=np.float64).reshape(-1)
  if backend() == "numpy":
    out = _projectDirectionsNumpy(xyz, imgs, q, parity)
  else:
    out = np.empty_like(xyz)
    _loop("projectDirections", _projectDirectionsLoop)(xyz, imgs, q, parity, out)
  return out.reshape(shape + (3,))


__all__ = []
