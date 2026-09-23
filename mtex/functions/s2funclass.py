# the name of the family of functions on the sphere, MATLAB's S2Fun, for what is asked of the
# family rather than of one representation
#
#   S2Fun.smiley(bandwidth=256)        the smiley test function, by quadrature
#   S2Fun.smiley(v)                    its values at directions
#   S2Fun.smiley(exact=True)           the function itself, as a handle
#   S2Fun.unimodal(v, halfwidth=w)     a de la Vallee Poussin bump about v
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
    from .handles import S2FunHandle
    from .s2fun import S2FunHarmonic
    if isinstance(v, vector3d):
      return _smileyValues(v, even)
    if exact:
      return S2FunHandle(lambda w: _smileyValues(w, even))
    return S2FunHarmonic.quadrature(lambda w: _smileyValues(w, even), **kwargs)

  @staticmethod
  def unimodal(v=None, halfwidth=25 * degree, psi=None):
    from ..geometry.quaternion import rotation
    from .kernelfunctions import S2DeLaValleePoussinKernel
    from .s2fun import S2FunHarmonic
    psi = S2DeLaValleePoussinKernel(halfwidth=halfwidth) if psi is None else psi
    sF = S2FunHarmonic(psi)
    if v is not None and float(np.arccos(np.clip(v.normalize().data.reshape(3) @ [0, 0, 1], -1, 1))) > 0:
      sF = rotation.map(vector3d(0, 0, 1), vector3d(v.data.reshape(3))) * sF
    return sF

  @staticmethod
  def interpolate(nodes, values=None, **kwargs):
    from .density import interpolate
    return interpolate(nodes, values, **kwargs)


__all__ = ["S2Fun"]
