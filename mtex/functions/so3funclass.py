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
    """an ODF read from a file, loadODF

      odf = SO3Fun.load('odf.txt', CS=cs, columnNames=[...])"""
    from ..io.loadodf import loadODF
    return loadODF(fname, *args, **kwargs)

  @staticmethod
  def dubna(ori=None):
    """the ODF of the Dubna quartz pole figures, MTEX's example function; its values at
    orientations when given

      odf = SO3Fun.dubna()          # the example ODF, an SO3FunHarmonic of trigonal quartz
      values = SO3Fun.dubna(ori)    # its values at orientations

    The function is read from the sample data, mtexdata('dubnaODF').

    Input
      ori - orientation of quartz, where to evaluate the function
    Output
      odf - SO3FunHarmonic
      values - array, the values at ori
    See also
      mtexdata, SO3FunHarmonic"""
    from ..io.sampledata import mtexdata
    f = mtexdata("dubnaODF")
    return f if ori is None else f.eval(ori)

  @staticmethod
  def interpolate(nodes, values=None, **kwargs):
    """the function on SO(3) through the values at the orientations, interpolate(nodes, values)

      SO3F = SO3Fun.interpolate(ori, values)                # an SO3FunHarmonic through the values
      SO3F = SO3Fun.interpolate(ori, values, bandwidth=4)   # of a given degree

    The keywords of interpolate, bandwidth, lambda_, s, weights, maxIter, tol and withParameters,
    are passed on: the coefficients fit the values by least squares with a Sobolev
    regularisation.

    Input
      nodes - orientation, or SO3TangentVector for a vector field
      values - ndarray of one value per node
    Output
      SO3F - SO3FunHarmonic, or SO3VectorFieldHarmonic for tangent vectors
    See also
      interpolate, SO3FunHarmonic.interpolate, calcDensity"""
    from .density import interpolate
    return interpolate(nodes, values, **kwargs)


__all__ = ["SO3Fun"]
