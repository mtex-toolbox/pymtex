# spatial transforms: a map of the plane of a scan onto itself, the distortion between two
# frames of positions (MATLAB geometry/spatialTransforms)
#
#   T * pos                    the positions moved by T; T1 * T2 moves by T2 first
#   T1 + T2                    the stages chained, T1 first
#   inv(T), T.displacement(pos), T.discretize(pos), transform(ebsd, T)
#   spatialTransformId()                         the identity
#   spatialTransformShift(M)                     an affine map, a 3 x 3 homogeneous matrix
#   spatialTransformRigid(u)                     a translation by the vector3d u
#   spatialTransformProjective(H)                a homography; .byTilt(theta, wd, centre)
#   spatialTransformPoly(c, degree)              a polynomial displacement of degree 1 or 2
#   spatialTransformDrift(s, u, slowScan=)       a displacement along the slow scan direction
#   spatialTransformField(pos, u)                a displacement interpolated between points
#   spatialTransformHandle(fun, funInv)          a function of the positions
#   spatialTransformComposite(T1, T2, ...)       stages applied in order
#   spatialTransformTilt()                       projective, affine and quadratic stages
#   spatialTransformInverse(T)                   the inverse of T by fixed point iteration
#   T = spatialTransformProjective.fit(posA, posB, weights=w)   and so for every class
#   spatialTransformList([T1, T2])               a list of hops, displayed as MATLAB's table
#
# A transform moves x and y and leaves z; it keeps the frame and the shape of the positions.
from __future__ import annotations

import numpy as np

from .vector3d import vector3d
from .display import aligned


def _xyz(pos):
  return np.asarray(pos.data, dtype=float).reshape(-1, 3)


# the positions with new x and y
def _moved(pos, x, y):
  d = np.array(pos.data, dtype=float)
  flat = d.reshape(-1, 3)
  flat[:, 0], flat[:, 1] = x, y
  return pos._new(d)


# ------------------------------------------------------------------ the fits

def fitData(posA, posB, weights=None):
  """the displacement, the weights and the positions a fit works from: a point in frame A, where
  it went in frame B and how much it is worth; a missing or unweighted point is dropped"""
  if not (isinstance(posA, vector3d) and isinstance(posB, vector3d)):
    raise TypeError("a transform is fitted to two vector3d point sets")
  A, B = _xyz(posA), _xyz(posB)
  if len(A) != len(B):
    raise ValueError(f"the two point sets have to be the same length, got {len(A)} and {len(B)}")
  x, y = A[:, 0], A[:, 1]
  dx, dy = B[:, 0] - x, B[:, 1] - y
  w = np.ones(len(x)) if weights is None else np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0)
  if len(w) != len(x):
    raise ValueError(f"expected {len(x)} weights, got {len(w)}")
  keep = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(w) & np.isfinite(x) & np.isfinite(y)
  if not keep.any():
    raise ValueError("every measurement handed to the fit is missing or unweighted")
  return dx[keep], dy[keep], w[keep], x[keep], y[keep]


def robustLsq(A, b, w, robust=True, maxIter=50):
  """weighted least squares, reweighted against outliers with Tukey's bisquare at 95 percent
  efficiency: the residuals corrected for leverage, one scale from their median absolute
  deviation, and the least favourable weight of the right hand sides for every point"""
  b = np.asarray(b, dtype=float).reshape(len(A), -1)
  w = np.asarray(w, dtype=float).reshape(-1)
  sw = np.sqrt(w)[:, None]
  p = np.linalg.lstsq(A * sw, b * sw, rcond=None)[0]
  if not robust:
    return p
  tune = 4.685
  Q = np.linalg.qr(A * sw)[0]
  h = np.minimum(np.sum(Q ** 2, axis=1), 1 - np.finfo(float).eps)
  adj = 1 / np.sqrt(1 - h)[:, None]
  for _ in range(maxIter):
    r = (b - A @ p) * adj
    s = np.median(np.abs(r - np.median(r, axis=0)), axis=0) / 0.6745
    s = np.maximum(s, np.finfo(float).eps)
    u = r / (tune * s)
    bi = (np.abs(u) < 1) * (1 - u ** 2) ** 2
    sw = np.sqrt(w * bi.min(axis=1))[:, None]
    pNew = np.linalg.lstsq(A * sw, b * sw, rcond=None)[0]
    done = np.linalg.norm(pNew - p) <= 1e-8 * max(1.0, np.linalg.norm(p))
    p = pNew
    if done:
      break
  return p


def polyBasis(x, y, degree):
  """the design matrix of a bivariate polynomial: [1 x y] or [1 x y x^2 xy y^2]"""
  x, y = np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)
  if degree == 1:
    return np.stack([np.ones_like(x), x, y], axis=1)
  if degree == 2:
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
  raise ValueError(f"a polynomial displacement is fitted at degree 1 or 2, got {degree}")


# ------------------------------------------------------------------ the transforms

class spatialTransform:
  """the base of the spatial transforms, maps of the plane of a scan onto itself that move x and y
  and leave z; T * pos moves positions, T1 * T2 is T2 followed by T1, T1 + T2 chains T1 first

    pos2 = T * pos
    T.inv(), T.displacement(pos), T.discretize(pos), transform(ebsd, T)"""

  def eval(self, pos):
    """the positions moved by the transform, T * pos"""
    raise NotImplementedError

  def inv(self):
    """the inverse transform"""
    raise NotImplementedError

  def paramChar(self):
    """the parameters of the transform as a line of text for its display"""
    return ""

  def __call__(self, pos):
    return self.eval(pos)

  # T * pos moves the positions; T1 * T2 is T2 followed by T1, one matrix where both have one
  def __mul__(self, other):
    if isinstance(other, vector3d):
      return self.eval(other)
    if not isinstance(other, spatialTransform):
      return NotImplemented
    out = self.absorb(other)
    return out if out is not None else spatialTransformComposite(other, self)

  # T1 + T2 chains the stages, T1 first
  def __add__(self, other):
    if not isinstance(other, spatialTransform):
      return NotImplemented
    if isinstance(self, spatialTransformId):
      return other
    if isinstance(other, spatialTransformId):
      return self
    return spatialTransformComposite(self, other)

  def shortChar(self):
    """the short name of the transform in its display, the class name without spatialTransform"""
    return type(self).__name__.replace("spatialTransform", "").lower()

  def char(self):
    """the name and the parameters of the transform as one line of text"""
    return f"{self.shortChar()}  {self.paramChar()}"

  def fitOptions(self):
    """the keywords a refit of this transform takes to keep its kind, as a dict"""
    return {}

  def matrix(self):
    """the 3 x 3 homogeneous matrix of the transform, None where it has none"""
    return None

  def stageList(self):
    """the stages the transform is applied in, a list"""
    return [self]

  @property
  def isid(self):
    """whether the transform is the identity"""
    return False

  def displacement(self, pos):
    """the displacement of the positions, T * pos - pos"""
    return self.eval(pos) - pos

  def norm(self, pos):
    """the length of the displacement at every position"""
    return self.displacement(pos).norm()

  def discretize(self, pos):
    """the transform as a field of displacements at the given points"""
    p = vector3d(_xyz(pos))
    return spatialTransformField(p, self.displacement(p))

  def absorb(self, other):
    """one matrix for two transforms that each have one: affine where the product keeps the
    last row, projective otherwise"""
    if self.isid:
      return other
    if other.isid:
      return self
    H1, H2 = self.matrix(), other.matrix()
    if H1 is None or H2 is None:
      return None
    H = H1 @ H2
    return spatialTransformShift(H) if np.linalg.norm(H[2] - [0, 0, 1]) < 1e-12 else spatialTransformProjective(H)

  def __repr__(self):
    stages = self.stageList()
    rows = [(self.shortChar(), "·", self.paramChar())] if len(stages) <= 1 else \
           [(self.shortChar() if k == 0 else "", s.shortChar(), s.paramChar()) for k, s in enumerate(stages)]
    w0 = max(len("model"), *(len(r[0]) for r in rows))
    w1 = max(len("stage"), *(len(r[1]) for r in rows))
    lines = [f"  {'model':<{w0}}  {'stage':<{w1}}  parameters"] + [f"  {a:<{w0}}  {b:<{w1}}  {c}" for a, b, c in rows]
    return aligned(f"{type(self).__name__}\n" + "\n".join(lines))


class spatialTransformId(spatialTransform):
  """the identity transform, which leaves every position where it is"""

  def eval(self, pos):
    """the positions unchanged"""
    return pos

  def inv(self):
    """the identity itself"""
    return self

  @property
  def isid(self):
    """always True"""
    return True

  def shortChar(self):
    """'identity', the name in the display"""
    return "identity"

  def paramChar(self):
    """the empty parameter entry of the display"""
    return "·"

  def stageList(self):
    """the stages of the identity, none"""
    return []

  def matrix(self):
    """the 3 x 3 identity matrix"""
    return np.eye(3)


class spatialTransformShift(spatialTransform):
  """an affine transform of the plane, given by a 3 x 3 homogeneous matrix with last row [0 0 1]

    T = spatialTransformShift(M)
    T = spatialTransformShift.fit(posA, posB, weights=w)"""

  def __init__(self, M=None):
    M = np.eye(3) if M is None else np.asarray(M, dtype=float)
    if M.shape != (3, 3):
      raise ValueError("an affine transform is given by a 3x3 homogeneous matrix")
    if np.linalg.norm(M[2] - [0, 0, 1]) >= 1e-12:
      raise ValueError("the last row of an affine homogeneous matrix is [0 0 1]; a projective one is spatialTransformProjective")
    self.M = M

  def eval(self, pos):
    """the positions moved by the affine map"""
    P = _xyz(pos)
    M = self.M
    return _moved(pos, M[0, 0] * P[:, 0] + M[0, 1] * P[:, 1] + M[0, 2], M[1, 0] * P[:, 0] + M[1, 1] * P[:, 1] + M[1, 2])

  def inv(self):
    """the affine map of the inverse matrix"""
    return spatialTransformShift(np.linalg.inv(self.M))

  @property
  def isid(self):
    """whether the matrix is the identity"""
    return np.linalg.norm(self.M - np.eye(3)) < 1e-12

  def matrix(self):
    """the 3 x 3 homogeneous matrix of the affine map"""
    return self.M

  def parts(self):
    """the scales, the rotation and the shear of the linear part, from its QR decomposition"""
    Q, R = np.linalg.qr(self.M[:2, :2])
    d = np.sign(np.diag(R))
    d[d == 0] = 1
    Q, R = Q * d, d[:, None] * R
    if np.linalg.det(Q) < 0:
      Q[:, 1], R[1] = -Q[:, 1], -R[1]
    rot = np.degrees(np.arctan2(Q[1, 0], Q[0, 0]))
    shear = np.degrees(np.arctan(R[0, 1] / R[1, 1]))
    return R[0, 0], R[1, 1], 0.0 if abs(rot) < 1e-9 else rot, 0.0 if abs(shear) < 1e-9 else shear

  def paramChar(self):
    """the scales, the rotation, the shear and the translation of the map as a line of text"""
    sx, sy, rot, shear = self.parts()
    return f"scale {sx:.4g} × {sy:.4g}, rotate {rot:.3g}°, shear {shear:.3g}°, move ({self.M[0, 2]:.4g}, {self.M[1, 2]:.4g})"

  @staticmethod
  def fit(posA, posB, weights=None, **kwargs):
    """the affine map fitted to point pairs by weighted least squares, the positions posA in frame A
    and where they went, posB"""
    return spatialTransformPoly.fit(posA, posB, weights=weights, degree=1, robust=False).shiftMatrix()


class spatialTransformRigid(spatialTransformShift):
  """a translation of the plane by the vector3d u

    T = spatialTransformRigid(vector3d(1, 2, 0))
    T = spatialTransformRigid.fit(posA, posB, weights=w)"""

  def __init__(self, u=None):
    super().__init__()
    if u is not None:
      if not isinstance(u, vector3d) or u.size != 1:
        raise ValueError("a rigid transform is given by one vector3d displacement")
      d = _xyz(u)[0]
      self.M[:2, 2] = d[:2]

  @property
  def u(self):
    """the translation vector"""
    return vector3d(self.M[0, 2], self.M[1, 2], 0.0)

  def inv(self):
    """the translation by -u"""
    return spatialTransformRigid(-self.u)

  def paramChar(self):
    """the translation as a line of text"""
    return f"move ({self.M[0, 2]:.4g}, {self.M[1, 2]:.4g})"

  @staticmethod
  def fit(posA, posB, weights=None, **kwargs):
    """the translation fitted to point pairs, the weighted mean of their displacements"""
    dx, dy, w, _, _ = fitData(posA, posB, weights)
    return spatialTransformRigid(vector3d(np.sum(w * dx) / np.sum(w), np.sum(w * dy) / np.sum(w), 0.0))


class spatialTransformProjective(spatialTransform):
  """a projective transform of the plane, a homography given by a 3 x 3 matrix normalised to
  H[2, 2] = 1; straight lines stay straight

    T = spatialTransformProjective(H)
    T = spatialTransformProjective.byTilt(theta, wd, centre)
    T = spatialTransformProjective.fit(posA, posB, weights=w)"""

  def __init__(self, H=None):
    H = np.eye(3) if H is None else np.asarray(H, dtype=float)
    if H.shape != (3, 3):
      raise ValueError("a projective transform is given by a 3x3 homography")
    if abs(H[2, 2]) <= 1e-12:
      raise ValueError("a homography with H[2, 2] = 0 sends the origin to infinity")
    self.H = H / H[2, 2]

  def eval(self, pos):
    """the positions moved by the homography"""
    P = _xyz(pos)
    H = self.H
    d = H[2, 0] * P[:, 0] + H[2, 1] * P[:, 1] + H[2, 2]
    return _moved(pos, (H[0, 0] * P[:, 0] + H[0, 1] * P[:, 1] + H[0, 2]) / d, (H[1, 0] * P[:, 0] + H[1, 1] * P[:, 1] + H[1, 2]) / d)

  def inv(self):
    """the homography of the inverse matrix"""
    return spatialTransformProjective(np.linalg.inv(self.H))

  @property
  def isid(self):
    """whether the homography is the identity"""
    return np.linalg.norm(self.H - np.eye(3)) < 1e-12

  def matrix(self):
    """the 3 x 3 homography"""
    return self.H

  def paramChar(self):
    """the perspective terms of the homography as a line of text"""
    return f"perspective ({self.H[2, 0]:.3g}, {self.H[2, 1]:.3g})"

  @staticmethod
  def byTilt(theta, wd, centre=None):
    """the specimen tilted by theta about an axis along x through centre, seen from the working
    distance wd: the far side shrinks and straight lines stay straight"""
    c = np.zeros(3) if centre is None else _xyz(centre)[0]
    if wd <= 0:
      raise ValueError(f"a working distance is positive, got {wd}")
    cx, cy = c[0], c[1]
    s, k = np.sin(theta) / wd, np.cos(theta)
    return spatialTransformProjective([[1, -cx * s, cx * s * cy], [0, k - cy * s, cy * (1 + cy * s - k)], [0, -s, 1 + s * cy]])

  @staticmethod
  def fit(posA, posB, weights=None, robust=True, maxIter=50):
    """the homography of the direct linear transform, robustly reweighted. Both point sets are
    first moved to their centroid and scaled to a mean distance of sqrt(2), Hartley's
    normalisation: without it the algebraic residual weighs the points by their distance from
    the origin, and ten percent of outliers can hold the reweighting in a wrong basin"""
    dx, dy, w, x, y = fitData(posA, posB, weights)
    if len(x) < 4:
      raise ValueError(f"a homography needs at least four points, got {len(x)}")

    def normalised(u, v):
      cu, cv = u.mean(), v.mean()
      r = np.mean(np.hypot(u - cu, v - cv))
      k = np.sqrt(2) / r if r > 0 else 1.0
      return (u - cu) * k, (v - cv) * k, np.array([[k, 0, -k * cu], [0, k, -k * cv], [0, 0, 1]])
    xa, ya, Na = normalised(x, y)
    xb, yb, Nb = normalised(x + dx, y + dy)
    o, z = np.ones_like(x), np.zeros_like(x)
    A = np.concatenate([np.stack([xa, ya, o, z, z, z, -xa * xb, -ya * xb], axis=1),
                        np.stack([z, z, z, xa, ya, o, -xa * yb, -ya * yb], axis=1)])
    h = robustLsq(A, np.concatenate([xb, yb]), np.concatenate([w, w]), robust, maxIter)
    return spatialTransformProjective(np.linalg.inv(Nb) @ np.append(h.reshape(-1), 1).reshape(3, 3) @ Na)


class spatialTransformPoly(spatialTransform):
  """a polynomial displacement of degree 1 or 2, the coefficients c one column per coordinate over
  the basis [1 x y] or [1 x y x^2 xy y^2]

    T = spatialTransformPoly(c, 2)
    T = spatialTransformPoly.fit(posA, posB, degree=2, weights=w)"""

  def __init__(self, c=None, degree=None):
    c = np.zeros((3, 2)) if c is None else np.asarray(c, dtype=float)
    degree = (1 + (c.shape[0] > 3)) if degree is None else int(degree)
    m = polyBasis(0, 0, degree).shape[1]
    if c.shape != (m, 2):
      raise ValueError(f"a degree {degree} polynomial displacement has {m} x 2 coefficients, got {c.shape}")
    self.c, self.degree = c, degree

  def eval(self, pos):
    """the positions moved by the polynomial displacement"""
    P = _xyz(pos)
    B = polyBasis(P[:, 0], P[:, 1], self.degree)
    return _moved(pos, P[:, 0] + B @ self.c[:, 0], P[:, 1] + B @ self.c[:, 1])

  def inv(self):
    """the inverse: the inverse affine map at degree 1, the fixed point inverse at degree 2"""
    return self.shiftMatrix().inv() if self.degree == 1 else spatialTransformInverse(self)

  @property
  def isid(self):
    """whether every coefficient vanishes"""
    return np.linalg.norm(self.c) < 1e-12

  def shiftMatrix(self):
    """the affine map a degree 1 displacement is"""
    if self.degree != 1:
      raise ValueError("only a degree 1 polynomial displacement is an affine map")
    c = self.c
    return spatialTransformShift([[c[1, 0] + 1, c[2, 0], c[0, 0]], [c[1, 1], c[2, 1] + 1, c[0, 1]], [0, 0, 1]])

  def shortChar(self):
    """the name in the display, poly11 or poly22 by the degree"""
    return f"poly{self.degree}{self.degree}"

  def paramChar(self):
    """the size of the coefficients as a line of text"""
    return f"|c| = {np.linalg.norm(self.c):.4g}"

  def fitOptions(self):
    """the degree, the keyword a refit keeps"""
    return {"degree": self.degree}

  @staticmethod
  def fit(posA, posB, weights=None, degree=1, robust=True, maxIter=50):
    """the polynomial displacement fitted to point pairs by least squares, reweighted against outliers
    with Tukey's bisquare unless robust=False"""
    dx, dy, w, x, y = fitData(posA, posB, weights)
    B = polyBasis(x, y, degree)
    if B.shape[0] < B.shape[1]:
      raise ValueError(f"a degree {degree} polynomial needs at least {B.shape[1]} points, got {B.shape[0]}")
    return spatialTransformPoly(robustLsq(B, np.stack([dx, dy], axis=1), w, robust, maxIter), degree)


class spatialTransformDrift(spatialTransform):
  """a displacement along the slow scan direction: linear between the knots s, the displacements u
  one row each, NaN beyond them unless extrapolate=True continues the end segments

    T = spatialTransformDrift(s, u, slowScan=vector3d(0, 1, 0))
    T = spatialTransformDrift.fit(posA, posB)          the median displacement of every scan line"""

  def __init__(self, s=None, u=None, slowScan=None, extrapolate=False):
    s = np.zeros(0) if s is None else np.asarray(s, dtype=float).reshape(-1)
    u = np.zeros((0, 2)) if u is None else np.asarray(u, dtype=float)
    if u.shape != (len(s), 2):
      raise ValueError(f"expected {len(s)} x 2 displacements to go with {len(s)} knots, got {u.shape}")
    order = np.argsort(s, kind="stable")
    self.s, self.u = s[order], u[order]
    sc = np.array([0.0, 1.0, 0.0]) if slowScan is None else _xyz(slowScan)[0]
    self.slowScan = sc / np.linalg.norm(sc)
    self.extrapolate = bool(extrapolate)

  def eval(self, pos):
    """the positions moved by the drift interpolated at their coordinate along the slow scan direction"""
    if len(self.s) == 0:
      return pos
    P = _xyz(pos)
    if len(self.s) == 1:
      return _moved(pos, P[:, 0] + self.u[0, 0], P[:, 1] + self.u[0, 1])
    t = P @ self.slowScan
    d = np.stack([self._interp(t, self.u[:, k]) for k in range(2)], axis=1)
    return _moved(pos, P[:, 0] + d[:, 0], P[:, 1] + d[:, 1])

  # linear between the knots; beyond them NaN, or the end segments continued
  def _interp(self, t, v):
    out = np.interp(t, self.s, v)
    lo, hi = t < self.s[0], t > self.s[-1]
    if self.extrapolate:
      out[lo] = v[0] + (t[lo] - self.s[0]) * (v[1] - v[0]) / (self.s[1] - self.s[0])
      out[hi] = v[-1] + (t[hi] - self.s[-1]) * (v[-1] - v[-2]) / (self.s[-1] - self.s[-2])
    else:
      out[lo | hi] = np.nan
    return out

  def inv(self):
    """the inverse by fixed point iteration"""
    return spatialTransformInverse(self)

  @property
  def isid(self):
    """whether the drift has no knots or displaces by nothing"""
    return len(self.s) == 0 or np.linalg.norm(self.u) < 1e-12

  def paramChar(self):
    """the knots and the largest displacement as a line of text"""
    if len(self.s) == 0:
      return "(no knots)"
    return f"{len(self.s)} knots over {self.s[0]:.4g} .. {self.s[-1]:.4g}, |u| <= {np.hypot(self.u[:, 0], self.u[:, 1]).max():.4g}"

  def fitOptions(self):
    """the slow scan direction, the keyword a refit keeps"""
    return {"slowScan": vector3d(self.slowScan)}

  @staticmethod
  def fit(posA, posB, weights=None, slowScan=None, extrapolate=False, **kwargs):
    """a linear spline through the median displacement of every scan line"""
    dx, dy, _, x, y = fitData(posA, posB, weights)
    sc = np.array([0.0, 1.0, 0.0]) if slowScan is None else _xyz(slowScan)[0]
    sc = sc / np.linalg.norm(sc)
    t = x * sc[0] + y * sc[1]
    span = max(1.0, t.max() - t.min())
    order = np.argsort(t, kind="stable")
    ts = t[order]
    start = np.concatenate([[True], np.diff(ts) > 1e-10 * span])
    grp = np.empty(len(t), dtype=np.int64)
    grp[order] = np.cumsum(start) - 1
    knots = ts[start]
    u = np.stack([_groupMedian(dx, grp, len(knots)), _groupMedian(dy, grp, len(knots))], axis=1)
    return spatialTransformDrift(knots, u, slowScan=vector3d(sc), extrapolate=extrapolate)


# the median of the values of every group 0 .. k-1, each group's values sorted together
def _groupMedian(values, grp, k):
  order = np.lexsort((values, grp))
  a, c = values[order], np.bincount(grp, minlength=k)
  s = np.concatenate([[0], np.cumsum(c)[:-1]])
  return (a[s + (c - 1) // 2] + a[s + c // 2]) / 2

class spatialTransformField(spatialTransform):
  """a displacement interpolated between points: linear over their triangulation, the nearest point's
  beyond it, where MATLAB takes natural neighbours and extrapolates linearly

    T = spatialTransformField(pos, u)
    T = spatialTransformField.fit(posA, posB)"""

  def __init__(self, pos=None, u=None):
    if (pos is None) != (u is None):
      raise ValueError("a field is given by the points and the displacement")
    if pos is not None:
      if not (isinstance(pos, vector3d) and isinstance(u, vector3d)):
        raise TypeError("a field is given by two vector3d, the points and the displacement")
      if pos.size != u.size:
        raise ValueError(f"expected one displacement per point, got {pos.size} and {u.size}")
    self.pos = pos.flatten() if pos is not None else None
    self.u = u.flatten() if u is not None else None
    self._interpolants = None

  def eval(self, pos):
    """linear over the triangulation of the points in coordinates of unit spread, the nearest
    point beyond it; MATLAB takes natural neighbours inside and linear extrapolation outside"""
    if self.pos is None or self.pos.size == 0:
      return pos
    P = _xyz(pos)
    if self._interpolants is None:
      from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
      X = _xyz(self.pos)[:, :2]
      self._c, self._s = X.mean(axis=0), np.where(X.std(axis=0) < np.finfo(float).eps, 1.0, X.std(axis=0))
      Xs = (X - self._c) / self._s
      U = _xyz(self.u)[:, :2]
      self._interpolants = (LinearNDInterpolator(Xs, U), NearestNDInterpolator(Xs, U))
    Q = (P[:, :2] - self._c) / self._s
    d = self._interpolants[0](Q)
    out = ~np.isfinite(d[:, 0])
    if out.any():
      d[out] = self._interpolants[1](Q[out])
    return _moved(pos, P[:, 0] + d[:, 0], P[:, 1] + d[:, 1])

  def inv(self):
    """the inverse by fixed point iteration"""
    return spatialTransformInverse(self)

  @property
  def isid(self):
    """whether the field has no points or displaces by nothing"""
    return self.pos is None or self.pos.size == 0 or np.max(np.linalg.norm(_xyz(self.u), axis=1)) < 1e-12

  def paramChar(self):
    """the number of points and the largest displacement as a line of text"""
    if self.pos is None or self.pos.size == 0:
      return "(empty)"
    return f"{self.pos.size} points, |u| <= {np.max(np.linalg.norm(_xyz(self.u), axis=1)):.4g}"

  @staticmethod
  def fit(posA, posB, weights=None, **kwargs):
    """the field of the displacements of the point pairs, at the points of frame A"""
    dx, dy, _, x, y = fitData(posA, posB, weights)
    z = np.zeros_like(x)
    return spatialTransformField(vector3d(np.stack([x, y, z], axis=1)), vector3d(np.stack([dx, dy, z], axis=1)))


class spatialTransformHandle(spatialTransform):
  """a transform given by a function of the positions and, for the inverse, a second one

    T = spatialTransformHandle(fun, funInv)"""

  def __init__(self, fun, funInv=None):
    if not callable(fun):
      raise TypeError("a spatialTransformHandle is built from a function")
    self.fun, self.funInv = fun, funInv

  def eval(self, pos):
    """the positions the function returns"""
    return self.fun(pos)

  def inv(self):
    """the transform of the inverse function, an error where none was given"""
    if self.funInv is None:
      raise ValueError("this transform was built from a function with no inverse; supply one as "
                       "spatialTransformHandle(fun, funInv), or fit a transform class that knows its own inverse")
    return spatialTransformHandle(self.funInv, self.fun)

  def paramChar(self):
    """the name of the function as a line of text"""
    name = getattr(self.fun, "__name__", repr(self.fun))
    return name + ("" if self.funInv is not None else "  (no inverse)")


class spatialTransformComposite(spatialTransform):
  """stages applied in order, the first given first; nested composites are flattened

    T = spatialTransformComposite(T1, T2, T3)"""

  def __init__(self, *stages):
    self.stages = []
    for s in stages:
      for t in (s if isinstance(s, (list, tuple)) else [s]):
        if not isinstance(t, spatialTransform):
          raise TypeError(f"a composite is built from spatial transforms, got {type(t).__name__}")
        self.stages += t.stages if isinstance(t, spatialTransformComposite) else [t]

  def eval(self, pos):
    """the positions moved through every stage in order"""
    for s in self.stages:
      pos = s.eval(pos)
    return pos

  def inv(self):
    """the inverses of the stages in reverse order"""
    return spatialTransformComposite(*[s.inv() for s in reversed(self.stages)])

  @property
  def isid(self):
    """whether every stage is the identity"""
    return all(s.isid for s in self.stages)

  def shortChar(self):
    """the short names of the stages joined by dashes"""
    return "-".join(s.shortChar() for s in self.stages) if self.stages else "empty"

  def paramChar(self):
    """the stages with their parameters as one line of text"""
    return " -> ".join(s.char() for s in self.stages) if self.stages else "(empty)"

  def stageList(self):
    """the stages, a list"""
    return list(self.stages)

  def fitStage(self, s, posA, posB, **kwargs):
    """the stage s, counted from zero, fitted to a residual measured after the ones before it"""
    if not 0 <= s < len(self.stages):
      raise IndexError(f"this transform has {len(self.stages)} stages, asked for stage {s}")
    p = self.stages[s]
    out = type(self).__new__(type(self))
    out.stages = list(self.stages)
    out.stages[s] = type(p).fit(posA, posB, **{**p.fitOptions(), **kwargs})
    return out


class spatialTransformTilt(spatialTransformComposite):
  """specimen tilt, fitted in stages against a re-measured residual: a projective stage takes the
  bulk, then an affine and a quadratic polynomial what is left; an unfitted stage is the
  identity, so a partly fitted tilt evaluates as far as it has got"""

  def __init__(self, *stages):
    if not stages:
      stages = (spatialTransformProjective(), spatialTransformPoly(np.zeros((3, 2)), 1), spatialTransformPoly(np.zeros((6, 2)), 2))
    super().__init__(*stages)

  def inv(self):
    """the inverses of the stages in reverse order, a plain composite"""
    return spatialTransformComposite(*self.stages).inv()

  def shortChar(self):
    """'tilt', the name in the display"""
    return "tilt"


class spatialTransformInverse(spatialTransform):
  """the inverse of a transform that knows none in closed form, each preimage found by fixed point
  iteration to the tolerance tol relative to the extent of the positions

    Tinv = spatialTransformInverse(T, tol=1e-10, maxIter=100)"""

  def __init__(self, T, tol=1e-10, maxIter=100):
    if not isinstance(T, spatialTransform):
      raise TypeError("the inverse wraps a spatial transform")
    self.T, self.tol, self.maxIter = T, tol, maxIter

  def eval(self, pos):
    """the preimage by fixed point iteration: the position moved against the error its image makes"""
    target = _xyz(pos)[:, :2]
    cur = pos
    finite = target[np.isfinite(target)]
    scale = max(1.0, np.abs(finite).max()) if finite.size else 1.0
    prev = np.inf
    resid = np.inf
    for k in range(1, self.maxIter + 1):
      e = _xyz(self.T.eval(cur))[:, :2] - target
      P = _xyz(cur)
      cur = _moved(cur, P[:, 0] - e[:, 0], P[:, 1] - e[:, 1])
      r = np.hypot(e[:, 0], e[:, 1])
      resid = np.nanmax(r) if np.isfinite(r).any() else 0.0
      if resid <= self.tol * scale:
        break
      if k > 3 and resid > 0.9 * prev:
        break
      prev = resid
    if resid > self.tol * scale:
      raise ValueError(f"the displacement field did not invert - it stalled at {resid:g} after {k} iterations, against a "
                       f"tolerance of {self.tol * scale:g}; either it folds, or it is asked for preimages well outside the "
                       "region it was fitted on")
    return cur

  def inv(self):
    """the transform it inverts"""
    return self.T

  @property
  def isid(self):
    """whether the inverted transform is the identity"""
    return self.T.isid

  def shortChar(self):
    """the short name of the inverted transform with the prefix inv-"""
    return "inv-" + self.T.shortChar()

  def paramChar(self):
    """the parameters of the inverted transform as a line of text"""
    return self.T.paramChar()


class spatialTransformList(list):
  """a list of transforms, one hop each, displayed as MATLAB's table of a spatialTransform array"""

  def __getitem__(self, k):
    out = super().__getitem__(k)
    return spatialTransformList(out) if isinstance(k, slice) else out

  def __repr__(self):
    rows = []
    for n, T in enumerate(self):
      stages = T.stageList()
      if len(stages) <= 1:
        rows.append((str(n), T.shortChar(), "·", T.paramChar()))
      else:
        rows += [(str(n) if k == 0 else "", T.shortChar() if k == 0 else "", S.shortChar(), S.paramChar()) for k, S in enumerate(stages)]
    labels = ("", "model", "stage", "parameters")
    w = [max(len(labels[j]), *(len(r[j]) for r in rows)) if rows else len(labels[j]) for j in range(4)]
    line = lambda r: "  " + "  ".join(x.ljust(q) for x, q in zip(r, w)).rstrip()
    return aligned("\n".join([f"spatialTransform\n size: 1 × {len(self)}", "", line(labels)] + [line(r) for r in rows]))


__all__ = ["spatialTransformList", "spatialTransform", "spatialTransformId", "spatialTransformShift", "spatialTransformRigid", "spatialTransformProjective",
           "spatialTransformPoly", "spatialTransformDrift", "spatialTransformField", "spatialTransformHandle", "spatialTransformComposite",
           "spatialTransformTilt", "spatialTransformInverse"]
