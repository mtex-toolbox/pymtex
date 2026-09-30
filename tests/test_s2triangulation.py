import numpy as np

from mtex import S2Triangulation, degree, equispacedS2Grid, vector3d

rng = np.random.default_rng(3)


def test_the_triangles_and_the_walk():
  v = vector3d.rand(300, rng=rng)
  sT = S2Triangulation(v)
  V = sT.vertices.data.reshape(-1, 3)
  # a closed triangulation of the sphere: 2n - 4 triangles, every one outward and counterclockwise
  assert len(sT) == 2 * v.size - 4
  n = np.cross(V[sT.T[:, 1]] - V[sT.T[:, 0]], V[sT.T[:, 2]] - V[sT.T[:, 0]])
  assert np.all(np.einsum("ij,ij->i", n, V[sT.T[:, 0]]) > 0)
  # the neighbour opposite a vertex shares the other two
  for k in range(3):
    other = sT.T[:, [j for j in range(3) if j != k]]
    assert all(set(o) <= set(sT.T[t]) for o, t in zip(other, sT.neighbours[:, k]))
  # the walk ends in the triangle the ray meets first, as the search over all of them says
  w = vector3d.rand(500, rng=rng)
  t, lam = sT.locate(w)
  W = w.data.reshape(-1, 3)
  hit = np.einsum("ij,ij->i", n, V[sT.T[:, 0]])[None, :] / (W @ n.T)
  hit[hit <= 0] = np.inf
  assert np.array_equal(t, np.argmin(hit, axis=1))
  assert np.allclose(lam.sum(axis=1), 1) and np.all(lam >= 0)
  # the coordinates rebuild the point on the plane of the triangle, along the ray
  P = np.einsum("ni,nij->nj", lam, V[sT.T[t]])
  assert np.allclose(np.cross(P, W), 0, atol=1e-12)


def test_the_barycentric_matrix():
  g = equispacedS2Grid(resolution=10 * degree)
  sT = S2Triangulation(g)
  B = sT.barycentric(sT.vertices)
  assert np.allclose(B.toarray(), np.eye(sT.vertices.size))
  assert np.allclose(sT.barycentric(vector3d.rand(50, rng=rng)).sum(axis=1), 1)
  assert sT.checkInside(sT.midPoints, np.arange(len(sT))).all()
