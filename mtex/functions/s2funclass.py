# the name of the family of functions on the sphere, MATLAB's S2Fun, for what is asked of the
# family rather than of one representation
#
#   S2Fun.smiley(bandwidth=256)        the smiley test function, by quadrature
#   S2Fun.smiley(v)                    its values at directions
#   S2Fun.smiley(exact=True)           the function itself, as a handle
#   S2Fun.unimodal(v, psi)             the kernel psi about v, by default a de la Vallee Poussin bump
#   S2Fun.interpolate(v, values)       a function through values
from __future__ import annotations

import numpy as np

from ..geometry.units import degree
from ..geometry.vector3d import vector3d

# the smiley: nine caps (z - h)^2 / (1 - h)^2 above the height h about their centres,
# weighted: the face, the two eyes, the five points of the mouth and the nose
_SMILEY_POLAR = np.array([[np.pi / 2, 0], [0.6, -0.6], [0.6, 0.6], [-0.5, -1], [-0.5, -0.5], [-0.5, 0], [-0.5, 0.5],
                          [-0.5, 1], [0, 0]])
_SMILEY_HEIGHT = np.array([0.7, 0.96, 0.96, 0.93, 0.93, 0.93, 0.93, 0.93, 0.96])
_SMILEY_WEIGHT = np.array([0.5, -0.5, -0.5, -0.25, -0.25, -0.25, -0.25, -0.25, 0.4])


def _smileyValues(v, even=False):
  from ..geometry.quaternion import rotation
  centers = rotation.byAxisAngle(vector3d(0, 0, 1), 90 * degree) * vector3d.byPolar(_SMILEY_POLAR[:, 0], _SMILEY_POLAR[:, 1])
  C = centers.data.reshape(-1, 3)
  xyz = v.normalize().data.reshape(-1, 3)
  z = xyz @ C.T

  def cap(z):
    return np.where(z > _SMILEY_HEIGHT, (z - _SMILEY_HEIGHT) ** 2 / (1 - _SMILEY_HEIGHT) ** 2, 0.0)

  f = cap(z) + (cap(-z) if even else 0)
  return (f @ _SMILEY_WEIGHT).reshape(v.shape)


class S2Fun:
  """the functions on the sphere: S2FunHarmonic, S2FunHandle, S2FunBingham"""

  @staticmethod
  def smiley(v=None, exact=False, even=False, **kwargs):
    """the smiley, a test function on the sphere whose face shows where a plot puts the axes

      sF = S2Fun.smiley()                    # its harmonic expansion by quadrature
      sF = S2Fun.smiley(bandwidth=256)       # of a given bandwidth
      values = S2Fun.smiley(v)               # its values at the directions v
      sF = S2Fun.smiley(exact=True)          # the function itself, as a handle
      sF = S2Fun.smiley(even=True)           # on both hemispheres, an antipodal function

    The face is a sum of quadratic splines of the cosine about nine centres, the head, two eyes,
    the mouth and the nose; it reads upright in the convention 'y↑→x'. The even smiley adds
    its reflection through the origin, so it takes the same value at v and -v. Further
    keywords go to S2FunHarmonic.quadrature.

    Input
      v - vector3d, directions to evaluate at
    Output
      sF - S2FunHarmonic, or S2FunHandle with exact=True
      values - array of the shape of v
    Keywords
      exact - return the function as an S2FunHandle instead of its expansion
      even - the antipodal smiley, the face also on the lower hemisphere
    See also
      S2FunHarmonic.quadrature, S2Fun.unimodal, S2FunHarmonicRepresentation"""
    from .handles import S2FunHandle
    from .s2fun import S2FunHarmonic
    if isinstance(v, vector3d):
      return _smileyValues(v, even)
    # the even smiley takes the same value at v and -v
    if exact:
      return S2FunHandle(lambda w: _smileyValues(w, even), antipodal=even, symmetrise=False)
    return S2FunHarmonic.quadrature(lambda w: _smileyValues(w, even), antipodal=even or kwargs.pop("antipodal", False), **kwargs)

  @staticmethod
  def unimodal(v=None, psi=None, *, halfwidth=25 * degree):
    """a unimodal function on the sphere, a radial kernel about one direction, as a harmonic
    expansion

      sF = S2Fun.unimodal()                          # about the north pole, halfwidth 25 degree
      sF = S2Fun.unimodal(halfwidth=10*degree)       # another halfwidth
      sF = S2Fun.unimodal(v, psi)                    # the kernel psi about the direction v

    The function is the kernel psi, by default the de la Vallee Poussin kernel of the
    halfwidth, turned from the north pole to v.

    Input
      v - vector3d, the direction of the maximum, by default the north pole
      psi - S2Kernel, by default the de la Vallee Poussin kernel of the halfwidth
    Output
      sF - S2FunHarmonic
    Keywords
      halfwidth - the halfwidth of the default de la Vallee Poussin kernel
    See also
      S2DeLaValleePoussinKernel, S2Fun.smiley, S2Kernels"""
    from ..geometry.quaternion import rotation
    from .kernelfunctions import S2DeLaValleePoussinKernel
    from .s2fun import S2FunHarmonic
    if isinstance(psi, (int, float, np.number)):
      raise TypeError("S2Fun.unimodal(v, psi) takes a kernel; a halfwidth is the keyword halfwidth=")
    psi = S2DeLaValleePoussinKernel(halfwidth=halfwidth) if psi is None else psi
    sF = S2FunHarmonic(psi)
    # a bump away from the north pole is turned there
    if v is not None and v.normalize().data.reshape(3)[2] < 1:
      sF = rotation.map(vector3d(0, 0, 1), vector3d(v.data.reshape(3))) * sF
    return sF

  @staticmethod
  def interpolate(nodes, values=None, **kwargs):
    """a function on the sphere through the values at the nodes, a regularised least squares fit;
    MATLAB's S2FunHarmonic.interpolate

      sF = S2Fun.interpolate(v, y)                              # a function through values at directions
      sF = S2Fun.interpolate(v, y, bandwidth=16, lambda_=1e-4)  # of a given degree, regularised
      sVF = S2Fun.interpolate(v, w)                             # a vector field through vectors w

    The harmonic coefficients up to the bandwidth minimise the weighted squared misfit at the
    nodes plus lambda_ times the Sobolev norm, solved by LSQR; the keywords, bandwidth, lambda_,
    s, weights, maxIter, tol and withParameters, are those of interpolate, where they are
    explained.

    Input
      nodes - vector3d, the directions
    Output
      sF - S2FunHarmonic, or S2VectorFieldHarmonic for vector values
    Keywords
      values - ndarray of one value per node, or vector3d for a vector field
    See also
      interpolate, S2FunHarmonic.adjoint, calcDensity"""
    from .density import interpolate
    return interpolate(nodes, values, **kwargs)


__all__ = ["S2Fun"]
