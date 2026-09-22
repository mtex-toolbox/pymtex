# the isometric embedding of orientations (docs/plans/functions.md P4f: T10 and T11 after
# the Julia tests; check_embedding for every Laue class)
import numpy as np
import pytest

from mtex import crystalFrame, degree, double, embedding, mean, orientation, rotation, setDouble, specimenFrame, vector3d
from mtex.analysis.embedding import embeddingSpec

rng = np.random.default_rng(25)


def test_T10_the_isometric_embedding():
  # check_embedding: for every Laue class the coordinates are isometric and invertible
  for name in ("-1", "2/m", "mmm", "-3", "-3m", "4/m", "4/mmm", "6/m", "6/mmm", "m-3", "m-3m"):
    cs = crystalFrame(name)
    oris = orientation.rand(12, cs, specimenFrame.default, rng=rng)
    es = embedding(oris)
    d = double(es)
    assert es.size == 12 and d.shape == (12, es.dim)
    assert max(abs((es[i] - es[j]).norm() - np.linalg.norm(d[i] - d[j])) for i in range(12) for j in range(12)) < 1e-10
    assert max((setDouble(es[i], d[i]) - es[i]).norm() for i in range(12)) < 1e-10
    # symmetrically equivalent orientations have the same image
    for s in cs.rotations():
      assert (embedding(orientation(rotation(oris[0].data) * s, cs, oris.SS)) - es[0]).norm() < 1e-10
  # the embedding of a rotated orientation is the rotated embedding, and the norm is fixed
  cs = crystalFrame("m-3m")
  o = orientation.rand(cs, specimenFrame.default, rng=rng)
  r = rotation.rand(rng=rng)
  assert (embedding(orientation(r * rotation(o.data), cs, o.SS)) - r * embedding(o)).norm() < 1e-10
  assert np.isclose(embedding(o).norm(), embedding(orientation.id(cs, o.SS)).norm())
  # the dimensions are the ones the paper gives, and the cubic one is nine
  assert embedding(o).dim == 9 and embeddingSpec(crystalFrame("m-3")).dim == 7 and embedding(crystalFrame("-3m")).dim == 12
  with pytest.raises(ValueError):
    embedding(orientation.rand(cs, rng=rng).inv())
  assert "embedding" in repr(embedding(o)) and "dimension: 9" in repr(embedding(o))


def test_T11_back_from_the_embedding():
  for name in ("-1", "222", "321", "4/mmm", "m-3m"):
    cs = crystalFrame(name)
    o = orientation.rand(cs, specimenFrame.default, rng=rng)
    # an orientation is recovered from its own image, up to the symmetry
    assert embedding(o).orientation().angle(o) < 1e-6
    # and the mean of a tight cluster lands where the mean orientation is
    os = orientation(rotation.byAxisAngle(vector3d.rand(100, rng=rng), 3 * degree) * rotation(o.data), cs, o.SS)
    es = embedding(os)
    assert isinstance(es, embedding) and len(es) == 100 and es.shape == (100,)
    assert es[7].isclose(embedding(os[7]))                    # the container and the element agree
    assert mean(es).isclose(embedding(os[0]) * 0 + sum((embedding(x) for x in os), embedding(os[0]) * 0) / 100)
    assert np.allclose(double(es)[7], double(es[7]))
    assert mean(es).orientation().angle(mean(os)) < 0.01 * degree
    assert mean(es, weights=np.eye(100)[3]).isclose(es[3])
