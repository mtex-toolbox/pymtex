# the container base, tested through a dummy subclass
import numpy as np
import pytest

from mtex.geometry._array import MtexArray, cat


class Pair(MtexArray):
  _ncoord = 2
  _carried = ("flag",)

  def __init__(self, data, flag=False):
    super().__init__(data)
    self.flag = flag


def test_construction_checks_the_trailing_axis():
  p = Pair(np.zeros((4, 2)))
  assert p.shape == (4,) and p.ndim == 1 and p.size == 4 and len(p) == 4
  assert p.dtype == np.float64
  assert Pair(np.zeros((3, 2), dtype=np.float32)).dtype == np.float32
  assert Pair([[1, 2]]).dtype == np.float64
  with pytest.raises(ValueError):
    Pair(np.zeros((4, 3)))
  with pytest.raises(ValueError):
    Pair(1.0)


def test_indexing_returns_the_class_and_keeps_attributes():
  p = Pair(np.arange(12.0).reshape(3, 2, 2), flag=True)
  assert p.shape == (3, 2)
  e = p[1, 0]
  assert isinstance(e, Pair) and e.shape == () and e.ndim == 0 and len(e) == 1 and e.flag
  assert e.data.tolist() == [4.0, 5.0]
  assert p[1].shape == (2,) and p[1].flag
  assert p[..., 0].shape == (3,) and p[..., 0].data[:, 0].tolist() == [0.0, 4.0, 8.0]
  assert p[-1].data.tolist() == [[8.0, 9.0], [10.0, 11.0]]
  assert p[:, None].shape == (3, 1, 2)
  mask = np.array([True, False, True])
  assert p[mask].shape == (2, 2)
  assert p[0:2].shape == (2, 2)


def test_slices_are_views_and_setitem_writes_through():
  p = Pair(np.zeros((4, 2)))
  q = p[1:3]
  q.data[:] = 1.0
  assert p.data[1:3].tolist() == [[1.0, 1.0], [1.0, 1.0]]
  p[0] = Pair([5.0, 6.0])
  assert p.data[0].tolist() == [5.0, 6.0]
  p[2:4] = Pair([[7.0, 7.0], [8.0, 8.0]])
  assert p.data[3].tolist() == [8.0, 8.0]
  with pytest.raises(TypeError):
    p[0] = [1.0, 2.0]
  c = p.copy()
  c.data[:] = 0.0
  assert p.data[0].tolist() == [5.0, 6.0]


def test_iteration_reshape_flatten_transpose():
  p = Pair(np.arange(12.0).reshape(3, 2, 2))
  assert [e.shape for e in p] == [(2,), (2,), (2,)]
  assert p.reshape(6).shape == (6,) and p.reshape((2, 3)).shape == (2, 3) and p.reshape(-1).shape == (6,)
  assert p.flatten().shape == (6,)
  assert p.T.shape == (2, 3) and p.T.data[1, 2].tolist() == [10.0, 11.0]
  e = p[0, 0]
  assert e.T.shape == () and e.flatten().shape == (1,)
  with pytest.raises(TypeError):
    iter(e)


def test_cat_promotes_single_elements_and_checks_attributes():
  p = Pair(np.ones((2, 2)))
  q = Pair(np.zeros((3, 2)))
  assert cat(p, q).shape == (5,) and Pair.cat(p, q).shape == (5,)
  assert cat(p[0], q[1]).shape == (2,)
  m = Pair(np.ones((2, 3, 2)))
  assert cat(m, m, axis=1).shape == (2, 6) and cat(m, m, axis=-1).shape == (2, 6)
  assert cat(p, q, axis=-1).shape == (5,)
  with pytest.raises(ValueError):
    cat(p, Pair(np.zeros((1, 2)), flag=True))
  assert cat(Pair(np.ones((1, 2)), flag=True), Pair(np.ones((1, 2)), flag=True)).flag


def test_new_rejects_unknown_attributes_and_astype_keeps_them():
  p = Pair(np.ones((2, 2)), flag=True)
  assert p.astype(np.float32).dtype == np.float32 and p.astype(np.float32).flag
  with pytest.raises(TypeError):
    p._new(p.data, frame=None)
  assert repr(p) == "Pair (2)" and repr(p[0]) == "Pair (1)" and repr(p.reshape(1, 2)) == "Pair (1 × 2)"
