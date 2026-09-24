# M2: the spatial decomposition, the Julia D1 to D6, on the NumPy form and the compiled loop
import numpy as np
import pytest

from mtex import EBSD, phaseList, crystalFrame, rotation, vector3d, zvector, degree, shift, decompose
from mtex.maps.decomposition import alphaClose, dilate, localVoronoi, meshVertices, NO_PIXEL
from mtex.geometry.kernels import _jit

CSTEST = crystalFrame("m-3m", mineral="test")
SQ2 = np.eye(2)
HEX2 = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
RECT2 = np.diag([0.8, 1.4])
BACKENDS = ["numpy"] + (["numba"] if _jit() else [])


def rot2d(w):
  return np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])


# an n x n map on the lattice A, with round holes of the given radii
def decompMap(A, n, holes=(), dtype=np.float64):
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  pos = np.stack([A[0, 0] * i + A[0, 1] * j, A[1, 0] * i + A[1, 1] * j, np.zeros(n * n)], axis=-1).astype(dtype)
  rot = rotation.byAxisAngle(zvector, ((i // 5 + (j // 5) * 3) * 20 * degree).astype(dtype))
  ph = np.ones(n * n, dtype=np.int8)
  for ci, cj, r in holes:
    ph[np.hypot(i - ci, j - cj) <= r] = 0
  return EBSD(pos, rot, ph, phaseList(CSTEST))


# the index offsets each site sees across its faces
def neighbourOffsets(c):
  nb = [set() for _ in range(c.nSites)]
  for a, b in c.cells:
    nb[a].add(tuple(c.siteIJ[b] - c.siteIJ[a]))
    nb[b].add(tuple(c.siteIJ[a] - c.siteIJ[b]))
  return nb


# the sites every one of whose lattice neighbours within k steps is a site too
def innerSites(c, k):
  key = {tuple(ij) for ij in c.siteIJ}
  return [s for s in range(c.nSites) if all((c.siteIJ[s, 0] + di, c.siteIJ[s, 1] + dj) in key for di in range(-k, k + 1) for dj in range(-k, k + 1))]


def test_dilation_uses_zero_padding():
  rng = np.random.default_rng(82)
  for dims in ((7, 11), (0, 5), (5, 0)):
    mask = rng.random(dims) < 0.5
    for offsets in ([(0, 0), (1, -2), (-3, 1), (20, 0)], []):
      expected = np.array([[any(0 <= i - di < dims[0] and 0 <= j - dj < dims[1] and mask[i - di, j - dj] for di, dj in offsets)
                            for j in range(dims[1])] for i in range(dims[0])], dtype=bool).reshape(dims)
      assert np.array_equal(dilate(mask, np.array(offsets, dtype=int).reshape(-1, 2)), expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_D1_an_interior_cell_is_the_unit_cell(backend):
  # every interior site sees exactly its stencil, four neighbours on a square lattice and
  # six on a hexagonal one, no diagonals: the case a general Voronoi code has to tie break
  for A in (SQ2, RECT2, HEX2):
    for w in (0.0, 0.4):
      e = decompMap(rot2d(w) @ A, 25 if backend == "numba" else 14)
      c = decompose(e, backend=backend)
      st = {tuple(o) for o in e.lattice.stencil}
      nb = neighbourOffsets(c)
      inner = innerSites(c, 3)
      assert inner and all(nb[s] == st for s in inner)


@pytest.mark.parametrize("backend", BACKENDS)
def test_D2_the_complex_is_a_partition(backend):
  e = decompMap(SQ2, 20)
  c = decompose(e, backend=backend)
  # every face separates two sites and is met once
  assert np.all(c.cells >= 0) and np.all(c.cells[:, 0] < c.cells[:, 1])
  assert len(np.unique(c.cells, axis=0)) == len(c.cells)
  # a face has two distinct vertices
  assert np.all(c.F[:, 0] != c.F[:, 1]) and c.F.max() < len(c.V) and c.V.shape[1] == 2
  # every measurement is a site, and the rings carry none
  assert np.sum(c.site2pixel >= 0) == len(e) and np.array_equal(np.sort(c.site2pixel[c.site2pixel >= 0]), np.arange(len(e)))
  assert c.nSites > len(e) and len(c) == len(c.cells) == len(c.F)
  # the cells of a full square map are unit squares: two faces and one vertex per cell
  inner = innerSites(c, 2)
  faces = np.isin(c.cells, inner).all(axis=1)
  lengths = np.linalg.norm(c.V[c.F[faces, 0]] - c.V[c.F[faces, 1]], axis=1)
  assert np.allclose(lengths, 1.0)
  # the site whose cell covers a point is the nearest one
  s = c.siteAt([3.2, 4.4], [3, 4])
  assert c.site2pixel[s] == 3 + 4 * 20
  assert c.unitCell2d().shape == (4, 2)
  assert "faces" in repr(c)


@pytest.mark.parametrize("backend", BACKENDS)
def test_D3_the_closing_fills_what_is_narrower_than_2alpha(backend):
  # a hole narrower than twice the closing radius is filled and its cells lose their site,
  # so the neighbours grow in; a wider one keeps a core of sites, which becomes a notIndexed grain
  # the Python form is the oracle and slow, so it runs on a smaller map
  n = 40 if backend == "numba" else 26
  for r, filled in ((1.5, True), (2.0, True), (9.0, False)):
    e = decompMap(SQ2, n, holes=((n // 2, n // 2, r),))
    c = decompose(e, alpha=3.1, backend=backend)
    hasSite = np.zeros(len(e), dtype=bool)
    hasSite[c.site2pixel[c.site2pixel >= 0]] = True
    ind = e.isIndexed
    assert hasSite[ind].all()                       # a measurement is always a site
    kept = np.sum(~ind & hasSite)
    assert (kept == 0) == filled
  # a larger alpha fills more
  e = decompMap(SQ2, n, holes=((n // 2, n // 2, 4.0),))
  assert decompose(e, alpha=1.0, backend=backend).nSites > decompose(e, alpha=6.0, backend=backend).nSites


@pytest.mark.parametrize("backend", BACKENDS)
def test_D4_the_decomposition_follows_the_map(backend):
  # a rigid motion of the map moves the diagram with it and changes nothing else
  e = decompMap(SQ2, 20 if backend == "numba" else 12, holes=((8, 8, 2.0),))
  c0 = decompose(e, backend=backend)
  for e2 in (e.rotate(rotation.byAxisAngle(zvector, 37 * degree)), shift(e, vector3d(11.0, -5.0, 0.0))):
    c = decompose(e2, backend=backend)
    assert c.nSites == c0.nSites and len(c) == len(c0) and len(c.V) == len(c0.V)


def test_D5_the_alpha_closing_is_the_morphological_one():
  # a slit narrower than the disk is closed, a wider one stays open
  m = np.zeros((41, 41), dtype=bool)
  m[5:36, 5:36] = True
  m[19:21, 5:26] = False                          # a two cell wide slit
  closed = alphaClose(m, np.eye(2), 3.1)
  assert closed[19:21, 7:24].all() and closed[9, 9]   # filled, and extensive
  m2 = m.copy()
  m2[15:25, 5:26] = False                          # a ten cell wide slit
  assert not alphaClose(m2, np.eye(2), 3.1)[19, 14]


@pytest.mark.parametrize("backend", BACKENDS)
def test_D6_single_precision(backend):
  e = decompMap(HEX2, 20 if backend == "numba" else 12, dtype=np.float32)
  c = decompose(e, backend=backend)
  assert c.V.dtype == np.float32
  st = {tuple(o) for o in e.lattice.stencil}
  nb = neighbourOffsets(c)
  assert all(nb[s] == st for s in innerSites(c, 3))


@pytest.mark.skipif(len(BACKENDS) < 2, reason="numba is not installed")
def test_the_compiled_loop_matches_the_numpy_form():
  for A, holes in ((SQ2, ((8, 8, 2.0), (25, 30, 5.0))), (HEX2, ((15, 15, 1.5),))):
    e = decompMap(A, 34, holes=holes)
    a, b = decompose(e, backend="numpy"), decompose(e, backend="numba")
    assert a.nSites == b.nSites and len(a) == len(b) and len(a.V) == len(b.V)
    ka = np.lexsort((a.cells[:, 1], a.cells[:, 0]))
    kb = np.lexsort((b.cells[:, 1], b.cells[:, 0]))
    assert np.array_equal(a.cells[ka], b.cells[kb])
    assert np.allclose(np.sort(a.V, axis=0), np.sort(b.V, axis=0), atol=1e-9)
  # a map that is not on a lattice cannot be decomposed
  n = 10
  line = EBSD(np.stack([np.arange(n) * 0.3, np.zeros(n), np.zeros(n)], axis=-1), rotation.rand(n), np.ones(n, dtype=int), [CSTEST])
  with pytest.raises(ValueError):
    decompose(line)


@pytest.mark.skipif(not _jit(), reason="numba missing")
def test_the_compiled_weld_is_the_numpy_one():
  # the faces of a hexagonal map with holes, their vertices moved by rounding noise, and a
  # square one far from the origin: the hashed weld groups every vertex as the sorts do
  rng = np.random.default_rng(3)
  for A, holes, off in ((HEX2, ((15, 15, 1.5), (5, 22, 3.0)), 0.0), (SQ2, ((8, 8, 2.0),), 1e4)):
    e = shift(decompMap(A, 30, holes=holes), vector3d(off, -off, 0))
    l = e.lattice
    c = decompose(e, backend="numpy")
    cells, verts, _ = localVoronoi(c.sites, l.A, backend="numpy")
    verts = verts * (1 + 1e-15 * rng.standard_normal(verts.shape))
    tol = l.dxy / 100
    a, b = meshVertices(cells, verts, tol, backend="numpy"), meshVertices(cells, verts, tol, backend="numba")
    assert all(np.array_equal(x, y) and x.dtype == y.dtype for x, y in zip(a, b))
