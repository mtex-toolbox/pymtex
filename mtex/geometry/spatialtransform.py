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
  it went in frame B and how much it is worth; a point whose position, displacement or weight
  is not a number is dropped, a negative weight counts as zero"""
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
    raise ValueError("every measurement handed to the fit is missing")
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
  """a transformation of scan positions in the xy plane

    pos2 = T * pos
    pos2 = T(pos)
    u = T.displacement(pos)
    Ti = T.inv()

  Transforms map vector3d positions to positions of the same shape. The supplied transform
  classes change x and y and preserve z and the coordinate frame. Positions and displacements
  use the same spatial units as the input data.

  Composition follows matrix order: `T1 * T2` applies T2 first, then T1. The expression
  T1 + T2 builds a sequence applying T1 first, then T2. Transforming EBSD data with
  transform(ebsd, T) also updates its spatial representation.

  This is the base class. Construct one of the affine, projective, polynomial, drift or
  interpolated-field subclasses, or wrap a callable with spatialTransformHandle.

  See also
    spatialTransformShift, spatialTransformProjective, spatialTransformPoly,
    spatialTransformDrift, spatialTransformField, spatialTransformComposite"""

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
    """the length of the displacement at every position

      n = norm(T, pos)"""
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
  """an affine map of scan positions, including translation, rotation, scale and shear

    T = spatialTransformShift(M)
    T = spatialTransformShift.fit(posA, posB, weights=w)
    pos2 = T * pos

  The homogeneous matrix acts on the column [x, y, 1]. Thus its upper-left 2-by-2 block
  sets the linear map and M[:2, 2] sets the translation. The z coordinate is preserved.
  The default matrix is the identity. Despite the class name, this represents a full affine
  map; spatialTransformRigid provides a pure translation.

  Input
    M - 3-by-3 matrix with last row [0, 0, 1], or None for the identity
  Output
    T - spatialTransformShift
  Properties
    M - homogeneous affine matrix
  See also
    spatialTransformShift.fit, spatialTransformShift.parts, spatialTransformRigid,
    spatialTransformProjective"""

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
    """decompose the affine linear part into scales, rotation and shear

      sx, sy, rotationAngle, shearAngle = T.parts()

    A QR decomposition separates a rotation from an upper-triangular scale and shear.
    Both angles are returned in degrees. A reflection is represented by a signed second
    scale. Translation is not included in these four values; it is T.M[:2, 2].

    Output
      sx - first scale factor
      sy - second scale factor, signed for reflections
      rotationAngle - rotation in degrees
      shearAngle - shear angle in degrees"""
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
  def fit(posA, posB, weights=None):
    """the affine map fitted to point pairs by weighted least squares, the positions posA in frame A
    and where they went, posB

      T = spatialTransformShift.fit(posA, posB)              # posB = T(posA) as nearly as can be
      T = spatialTransformShift.fit(posA, posB, weights=w)   # the points weighted

    The degree one polynomial fit of spatialTransformPoly.fit, plain least squares without
    the reweighting against outliers, as MATLAB's. A point with a missing position or weight
    is dropped.

    Input
      posA - vector3d, the positions in frame A
      posB - vector3d, the same points in frame B
    Output
      T - spatialTransformShift
    Keywords
      weights - one weight per point
    See also
      spatialTransformPoly.fit, spatialTransformRigid.fit, spatialTransformProjective.fit"""
    return spatialTransformPoly.fit(posA, posB, weights=weights, degree=1, robust=False).shiftMatrix()


class spatialTransformRigid(spatialTransformShift):
  """a translation of the plane by the vector3d u

    T = spatialTransformRigid(vector3d(1, 2, 0))
    T = spatialTransformRigid.fit(posA, posB, weights=w)

  Input
    u - vector3d translation; only x and y are used; None gives zero displacement"""

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
  def fit(posA, posB, weights=None):
    """the translation fitted to point pairs, the weighted mean of their displacements

      T = spatialTransformRigid.fit(posA, posB)              # the mean displacement
      T = spatialTransformRigid.fit(posA, posB, weights=w)   # the weighted mean

    A point with a missing position or weight is dropped.

    Input
      posA - vector3d, the positions in frame A
      posB - vector3d, the same points in frame B
    Output
      T - spatialTransformRigid
    Keywords
      weights - one weight per point
    See also
      spatialTransformShift.fit, spatialTransformField.fit"""
    dx, dy, w, _, _ = fitData(posA, posB, weights)
    return spatialTransformRigid(vector3d(np.sum(w * dx) / np.sum(w), np.sum(w * dy) / np.sum(w), 0.0))


class spatialTransformProjective(spatialTransform):
  """a projective transform of the plane, a homography given by a 3 x 3 matrix normalised to
  H[2, 2] = 1; straight lines stay straight

    T = spatialTransformProjective(H)
    T = spatialTransformProjective.byTilt(theta, wd, centre)
    T = spatialTransformProjective.fit(posA, posB, weights=w)

  Input
    H - 3-by-3 homogeneous projective matrix, normalised by H[2,2]; None gives identity"""

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
    distance wd: the far side shrinks and straight lines stay straight

      T = spatialTransformProjective.byTilt(70*degree, wd)          # about the x axis
      T = spatialTransformProjective.byTilt(70*degree, wd, centre)  # through a point left in place

    cos(theta) foreshortens across the tilt axis and wd sets how much the scale varies over
    the frame: as wd grows the perspective vanishes and only the foreshortening is left. This
    is a tilt one knows; one that was measured is fitted by spatialTransformTilt.

    Input
      theta - the tilt angle about the x axis
      wd - the working distance the surface is seen from, positive
      centre - vector3d, the point the tilt leaves in place, by default the origin
    Output
      T - spatialTransformProjective
    See also
      spatialTransformProjective.fit, spatialTransformTilt"""
    c = np.zeros(3) if centre is None else _xyz(centre)[0]
    if wd <= 0:
      raise ValueError(f"a working distance is positive, got {wd}")
    cx, cy = c[0], c[1]
    s, k = np.sin(theta) / wd, np.cos(theta)
    return spatialTransformProjective([[1, -cx * s, cx * s * cy], [0, k - cy * s, cy * (1 + cy * s - k)], [0, -s, 1 + s * cy]])

  @staticmethod
  def fit(posA, posB, weights=None, robust=True, maxIter=50):
    """the homography best mapping the positions posA onto posB, by the direct linear transform

      T = spatialTransformProjective.fit(posA, posB)               # robust against outliers
      T = spatialTransformProjective.fit(posA, posB, weights=w)    # the points weighted
      T = spatialTransformProjective.fit(posA, posB, robust=False) # plain least squares

    The equations x' (h31 x + h32 y + 1) = h11 x + h12 y + h13, and likewise for y', are solved
    by least squares, reweighted against outliers with Tukey's bisquare unless robust=False.
    Both point sets are first centred and scaled, Hartley's normalisation, so that points far
    from the origin do not dominate the fit.

    A point with a missing position or no weight is dropped; at least four points are needed.

    Input
      posA - vector3d, the positions in the first frame
      posB - vector3d, where they went in the second, one for each
    Output
      T - spatialTransformProjective, T * posA close to posB
    Keywords
      weights - array, the weight of every point pair
      robust - reweight against outliers with Tukey's bisquare
      maxIter - the largest number of reweighting steps
    References
      R. Hartley, In defense of the eight-point algorithm, IEEE Transactions on Pattern
          Analysis and Machine Intelligence 19 (1997) 580-593
    See also
      spatialTransformPoly.fit, spatialTransformProjective.byTilt, spatialTransformTilt,
      transform"""
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
  """a scan distortion described by a polynomial displacement in x and y

    T = spatialTransformPoly(c, 2)
    T = spatialTransformPoly.fit(posA, posB, degree=2)
    pos2 = T * pos

  The output is the input position plus the polynomial displacement. Rows of c multiply
  the terms [1, x, y] for degree 1, or `[1, x, y, x**2, x*y, y**2]` for degree 2.
  Its two columns give the x and y displacements. The z coordinate is unchanged.
  A degree-1 displacement is an affine map and has an exact matrix inverse; degree 2
  uses a numerical inverse. With no coefficients the transform is the identity.

  Input
    c - coefficient array of shape (3, 2) or (6, 2)
    degree - 1 or 2; inferred from the number of coefficient rows when omitted
  Output
    T - spatialTransformPoly
  Properties
    c - displacement coefficients
    degree - polynomial degree
  See also
    spatialTransformPoly.fit, spatialTransformPoly.shiftMatrix, spatialTransformInverse"""

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
    """the polynomial displacement best mapping the positions posA onto posB

      T = spatialTransformPoly.fit(posA, posB)                     # an affine map
      T = spatialTransformPoly.fit(posA, posB, degree=2)           # a quadratic displacement
      T = spatialTransformPoly.fit(posA, posB, degree=2, weights=w, robust=False)  # weighted least squares

    The displacement posB - posA is fitted over the basis [1 x y], or [1 x y x^2 xy y^2] for
    degree 2, by least squares, reweighted against outliers with Tukey's bisquare. A point with
    a missing position or no weight is dropped; degree 1 needs at least three points, degree 2
    six.

    Input
      posA - vector3d, the positions in the first frame
      posB - vector3d, where they went in the second, one for each
    Output
      T - spatialTransformPoly, T * posA close to posB
    Keywords
      weights - array, the weight of every point pair
      degree - 1 or 2, the degree of the polynomial
      robust - reweight against outliers with Tukey's bisquare
      maxIter - the largest number of reweighting steps
    See also
      spatialTransformProjective.fit, spatialTransformDrift.fit, spatialTransformTilt,
      transform"""
    dx, dy, w, x, y = fitData(posA, posB, weights)
    B = polyBasis(x, y, degree)
    if B.shape[0] < B.shape[1]:
      raise ValueError(f"a degree {degree} polynomial needs at least {B.shape[1]} points, got {B.shape[0]}")
    return spatialTransformPoly(robustLsq(B, np.stack([dx, dy], axis=1), w, robust, maxIter), degree)


class spatialTransformDrift(spatialTransform):
  """a displacement that varies along the slow scan direction

    T = spatialTransformDrift(s, u, slowScan=vector3d(0, 1, 0))
    T = spatialTransformDrift.fit(posA, posB)

  Project each position onto the unit slow-scan direction, then interpolate the x and y
  displacement at that coordinate. The displacement varies linearly between supplied knots;
  knots and their displacements are sorted together. Outside their range, positions become
  NaN unless extrapolate=True continues the end segments. No knots gives the identity.

  Input
    s - one-dimensional array of knot coordinates in spatial units
    u - array of shape (len(s), 2), giving x and y displacements at the knots
    slowScan - vector3d direction, normalised internally; default is the y direction
    extrapolate - continue the end segments outside the knot range; default False
  Output
    T - spatialTransformDrift
  See also
    spatialTransformDrift.fit, spatialTransformField"""

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
  def fit(posA, posB, weights=None, slowScan=None, extrapolate=False):
    """a drift along the slow scan direction, the linear spline through the median displacement
    of every scan line

      T = spatialTransformDrift.fit(posA, posB)                      # scan lines along x
      T = spatialTransformDrift.fit(posA, posB, slowScan=xvector)    # scan lines along y
      T = spatialTransformDrift.fit(posA, posB, extrapolate=True)    # continued beyond the lines

    The points of one scan line share their coordinate along the slow scan direction, up to
    rounding; each line becomes a knot with the median displacement of its points. The median
    rather than the mean: a line crossing a badly correlated patch is not pulled by it. The
    weights only drop the points whose weight is not a number, since one number per line
    leaves nothing for them to outvote; a weight zero is kept, as MATLAB keeps it.

    Input
      posA - vector3d, the positions in the first frame
      posB - vector3d, where they went in the second, one for each
    Output
      T - spatialTransformDrift, a knot per scan line
    Keywords
      weights - array, a weight per point pair; not a number drops a point
      slowScan - vector3d, the slow scan direction, by default y
      extrapolate - continue the end segments beyond the first and last line, where the
          drift is otherwise NaN
    See also
      spatialTransformPoly.fit, spatialTransformField, transform"""
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
  """a scan distortion interpolated from measured displacement vectors

    T = spatialTransformField(pos, u)
    T = spatialTransformField.fit(posA, posB)
    moved = T * query

  Displacements are interpolated linearly over a triangulation of the sample xy positions,
  with coordinates rescaled for interpolation. Queries outside the convex hull use the nearest
  sample's displacement. The interpolated x and y displacement is added to the query;
  z is preserved. This differs from MATLAB's natural-neighbour interpolation and linear
  extrapolation. With no samples the transform is the identity.

  Input
    pos - vector3d sample positions
    u - vector3d displacements, one per sample position
  Output
    T - spatialTransformField
  See also
    spatialTransformField.fit, spatialTransformDrift, spatialTransformInverse"""

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
  def fit(posA, posB, weights=None):
    """the field of the displacements of the point pairs, at the points of frame A

      T = spatialTransformField.fit(posA, posB)              # the displacements as measured

    There is nothing to fit: the field interpolates its samples exactly. The weights only
    drop the points whose weight is not a number, as the points with a missing position are
    dropped, so the same call works on the output of a failed correlation; a weight zero is
    kept, as MATLAB keeps it.

    Input
      posA - vector3d, the positions in frame A
      posB - vector3d, the same points in frame B
    Output
      T - spatialTransformField
    Keywords
      weights - one weight per point
    See also
      spatialTransformField, spatialTransformPoly.fit"""
    dx, dy, _, x, y = fitData(posA, posB, weights)
    z = np.zeros_like(x)
    return spatialTransformField(vector3d(np.stack([x, y, z], axis=1)), vector3d(np.stack([dx, dy, z], axis=1)))


class spatialTransformHandle(spatialTransform):
  """a spatial transform defined by a callable and an optional inverse callable

    T = spatialTransformHandle(fun, funInv)
    moved = T * pos

  The callable receives vector3d positions and must return their transformed positions,
  rather than their displacements. It is responsible for preserving the required shape,
  frame and z coordinates. Supply funInv to enable T.inv(); without it, requesting an
  inverse raises ValueError. The inverse callable receives positions in the output space.

  Input
    fun - callable mapping input positions to output positions
    funInv - inverse callable, or None if no inverse is provided
  Output
    T - spatialTransformHandle
  See also
    spatialTransform, spatialTransformInverse"""

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
  """a sequence of spatial transforms applied in the supplied order

    T = spatialTransformComposite(T1, T2, T3)
    moved = T * pos

  This applies T1, then T2, then T3. Each argument may also be a list or tuple of stages;
  nested composites are flattened. An empty sequence is the identity. Inversion reverses
  the order and inverts each stage. T1 + T2 constructs the same ordering, while `T1 * T2`
  uses matrix order and applies T2 first.

  Input
    stages - spatialTransform objects, or lists or tuples of them
  Output
    T - spatialTransformComposite
  Properties
    stages - ordered list of transforms
  See also
    spatialTransformComposite.fitStage, spatialTransformTilt"""

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
    """return a composite with one stage fitted to supplied point pairs

      fitted = T.fitStage(s, posA, posB, weights=w)

    The zero-based stage s is replaced by a fit of the same transform class. Existing model
    options, such as polynomial degree, are retained unless overridden by keywords. The other
    stages are kept. The input composite is unchanged.

    Point pairs are passed directly to that stage's fit method. When fitting residual
    misalignment, supply coordinates already adjusted for the other stages; this method does
    not calculate residual point pairs automatically.

    Input
      s - zero-based stage index
      posA - vector3d input positions for the selected stage
      posB - corresponding target positions
      kwargs - options accepted by the selected stage's fit method
    Output
      fitted - composite with the fitted stage
    See also
      spatialTransformPoly.fit, spatialTransformProjective.fit"""
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
  identity, so a partly fitted tilt evaluates as far as it has got

  Input
    stages - optional spatialTransform stages in application order; by default projective, affine and quadratic stages, all initially identity"""

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
  """a numerical inverse of a spatial transform without a closed-form inverse

    Ti = spatialTransformInverse(T, tol=1e-10, maxIter=100)
    original = Ti * transformed

  Starting at each target position, fixed-point iteration subtracts the error between its
  current image and the target. This works for sufficiently small, well-behaved distortions;
  a folded map or a poorly converging iteration raises ValueError.

  Convergence requires the largest xy error to be at most tol times max(1, the largest
  absolute finite target coordinate). Iteration also stops if the error ceases to decrease
  sufficiently after the first three steps. The z coordinate is preserved.

  Input
    T - spatialTransform to invert
    tol - relative position tolerance; default 1e-10
    maxIter - maximum iteration count; default 100
  Output
    Ti - spatialTransformInverse
  See also
    spatialTransform.inv, spatialTransformField, spatialTransformPoly"""

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
  """a list of transforms, one hop each, displayed as MATLAB's table of a spatialTransform array

  Input
    iterable - spatialTransform objects in hop order; an empty iterable gives an empty list"""

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
    def line(r):
      return "  " + "  ".join(x.ljust(q) for x, q in zip(r, w)).rstrip()
    return aligned("\n".join([f"spatialTransform\n size: 1 × {len(self)}", "", line(labels)] + [line(r) for r in rows]))


__all__ = ["spatialTransformList", "spatialTransform", "spatialTransformId", "spatialTransformShift", "spatialTransformRigid", "spatialTransformProjective",
           "spatialTransformPoly", "spatialTransformDrift", "spatialTransformField", "spatialTransformHandle", "spatialTransformComposite",
           "spatialTransformTilt", "spatialTransformInverse"]
