# the HEALPix grid: the centres of the hierarchical equal area iso-latitude pixels of
# Gorski et al. 2005, in the ring scheme (MATLAB geometry/HEALPixS2Grid.m)
#
#   HEALPixS2Grid(resolution=5*degree)   Nside the power of two nearest to 1/resolution
#   HEALPixS2Grid(nside=8)                12 nside^2 nodes
import numpy as np

from ..units import degree
from ..vector3d import vector3d


def HEALPixS2Grid(resolution=None, nside=None):
  """the HEALPix nodes of the sphere as a list of directions"""
  if nside is None:
    resolution = 2.5 * degree if resolution is None else resolution
    nside = 2 ** int(np.round(np.log2(1 / resolution)))
  nside = int(nside)
  # the north polar cap: ring i holds 4i pixels
  capZ, capPhi = [], []
  for i in range(1, nside):
    j = np.arange(1, 4 * i + 1)
    capZ.append(np.full(j.size, 1 - i ** 2 / (3 * nside ** 2)))
    capPhi.append(np.pi / (2 * i) * (j - 0.5))
  # the equatorial belt: every ring holds 4 nside pixels, staggered ring by ring
  beltZ, beltPhi = [], []
  j = np.arange(1, 4 * nside + 1)
  for i in range(nside, 3 * nside + 1):
    s = (i - nside + 1) % 2
    beltZ.append(np.full(j.size, 4 / 3 - 2 * i / (3 * nside)))
    beltPhi.append(np.pi / (2 * nside) * (j - s / 2))
  # the south polar cap mirrors the north one, its rings from the belt to the pole
  z = np.concatenate(capZ + beltZ + [-x for x in reversed(capZ)])
  phi = np.concatenate(capPhi + beltPhi + capPhi[::-1])
  r = np.sqrt(1 - z ** 2)
  return vector3d(np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1), isNormalized=True)


__all__ = ["HEALPixS2Grid"]
