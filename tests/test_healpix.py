# the HEALPix nodes: their count, their equal area rings, MATLAB's resolution rule
import numpy as np

from mtex import HEALPixS2Grid, degree


def test_count_and_symmetry():
  g = HEALPixS2Grid(nside=4)
  assert g.size == 12 * 16
  assert np.allclose(np.linalg.norm(g.data, axis=-1), 1)
  # the nodes come in antipodal pairs and the mean is the origin
  assert np.allclose(g.data.mean(axis=0), 0, atol=1e-12)
  # every ring lies at one of the HEALPix latitudes, 4 nside - 1 rings
  assert len(np.unique(np.round(g.z, 12))) == 4 * 4 - 1


def test_resolution_rule():
  # MATLAB's rule: nside = 2^round(log2(1/res)); 7 degree gives nside 8 and 768 nodes
  assert HEALPixS2Grid(resolution=7 * degree).size == 768
  assert HEALPixS2Grid(resolution=2.5 * degree).size == 12 * 32 ** 2
