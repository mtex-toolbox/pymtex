# orientFaces: the faces of polyhedral grains turned consistently, the normals outward
import numpy as np
from scipy.sparse import csr_matrix

from mtex import crystalFrame, grain3d, phaseList


# two unit cubes side by side along x, their shared face and some others given the wrong sign
def twoCubes():
  V = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1, 2)], dtype=float)
  i = lambda x, y, z: x + 3 * y + 6 * z
  F, I = [], []

  def cube(g, x0):
    for q in ([(x0, 0, 0), (x0, 1, 0), (x0 + 1, 1, 0), (x0 + 1, 0, 0)], [(x0, 0, 1), (x0 + 1, 0, 1), (x0 + 1, 1, 1), (x0, 1, 1)],
              [(x0, 0, 0), (x0 + 1, 0, 0), (x0 + 1, 0, 1), (x0, 0, 1)], [(x0, 1, 0), (x0, 1, 1), (x0 + 1, 1, 1), (x0 + 1, 1, 0)]):
      F.append([i(*p) for p in q])
      I.append((g, len(F) - 1, 1))
  cube(0, 0)
  cube(1, 1)
  F.append([i(0, 0, 0), i(0, 0, 1), i(0, 1, 1), i(0, 1, 0)]); I.append((0, len(F) - 1, 1))
  F.append([i(2, 0, 0), i(2, 1, 0), i(2, 1, 1), i(2, 0, 1)]); I.append((1, len(F) - 1, 1))
  F.append([i(1, 0, 0), i(1, 1, 0), i(1, 1, 1), i(1, 0, 1)]); I.append((0, len(F) - 1, 1)); I.append((1, len(F) - 1, 1))
  rows, cols, vals = zip(*I)
  return grain3d.fromFaces(V, F, csr_matrix((vals, (rows, cols)), shape=(2, len(F))), np.tile([1.0, 0, 0, 0], (2, 1)),
                           phaseList([crystalFrame("m-3m", mineral="Iron")]))


def test_the_signs_follow_the_corners():
  g = twoCubes()
  # all +1 as given: some faces point inward, the shared face counts for both grains
  assert not np.allclose(g.volume, 1)
  o = g.orientFaces()
  assert np.allclose(o.volume, 1) and np.allclose(o.centroid.data, [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]])
  # the shared face is +1 for one grain and -1 for the other, its sides the two grains
  s = np.asarray(o.I_GF.sum(axis=0)).reshape(-1)
  assert s[-1] == 0 and np.all(s[:-1] == 1)
  assert sorted(o.boundary.grainId[-1]) == [1, 2] and o.boundary.grainId[-1, 0] == o.id[o.I_GF[:, len(s) - 1].toarray().ravel() > 0][0]
