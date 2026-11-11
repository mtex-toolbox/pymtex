# the shortest boundary network through the filled cells, against MATLAB develop
import os

import numpy as np
import pytest

from mtex import calcGrains

LOCAL = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data")


@pytest.mark.skipif(not os.path.isdir(os.path.join(LOCAL, "EBSD")), reason="the MTEX data folder is not here")
def test_the_chain_census_of_small_is_matlabs():
  # MATLAB develop's check_boundaryChains on the indexed pixels of small: 1017 segments in 168
  # chains, the longest of 65 segments, 34 of one, 97 junctions, 67 triple points
  from mtex import mtexdata
  ebsd = mtexdata("small", root=LOCAL, download=False)
  gB = calcGrains(ebsd[ebsd.isIndexed]).boundary
  size = np.bincount(gB.chainId)
  assert (len(gB), len(size), size.max(), np.sum(size == 1), len(gB.junctionId), len(gB.triplePoints)) == (1017, 168, 65, 34, 97, 67)
