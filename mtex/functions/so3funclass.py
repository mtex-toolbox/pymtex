# the name of the family of functions on SO(3), MATLAB's SO3Fun, for what is asked of the
# family rather than of one representation
#
#   SO3Fun.load('odf.txt', CS=cs, 'Bunge', 'interp', columnNames=[...])   an ODF from a file
#   SO3Fun.interpolate(ori, values)                                        a function through values
from __future__ import annotations


class SO3Fun:
  """the functions on SO(3): SO3FunHarmonic, SO3FunRBF, SO3FunCBF, SO3FunBingham, SO3FunHandle, SO3FunComposition"""

  @staticmethod
  def load(fname, *args, **kwargs):
    from ..io.loadodf import loadODF
    return loadODF(fname, *args, **kwargs)

  @staticmethod
  def dubna(ori=None):
    """the ODF of the Dubna quartz pole figures, MTEX's example function; its values at
    orientations when given"""
    from ..io.sampledata import mtexdata
    f = mtexdata("dubnaODF")
    return f if ori is None else f.eval(ori)

  @staticmethod
  def interpolate(nodes, values=None, **kwargs):
    from .density import interpolate
    return interpolate(nodes, values, **kwargs)


__all__ = ["SO3Fun"]
