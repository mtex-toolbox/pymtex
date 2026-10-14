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


def test_numpy_sees_the_coordinates_and_leaves_the_operators_to_the_container():
  p = Pair(np.arange(6.0).reshape(3, 2))
  assert np.asarray(p) is p.data
  assert np.array_equal(np.array(p), p.data) and not np.shares_memory(np.array(p), p.data)
  assert np.asarray(p, dtype=np.float32).dtype == np.float32
  with pytest.raises(ValueError):
    np.asarray(p, dtype=np.float32, copy=False)
  with pytest.raises(TypeError):
    np.ones(3) + p


# two operands of every class of the flat namespace that has a cat, by the class defining it
def _catOperands():
  from mtex import (EBSD, PoleFigure, S2FunHarmonic, SO3FunHarmonic, crystalFrame, specimenFrame, orientation,
                    rotation, vector3d, tensor, fibre, slipSystem, phaseList, Miller, log, degree)
  rng = np.random.default_rng(0)
  cs, ss = crystalFrame("321"), specimenFrame("222")
  o = orientation.rand(3, cs, ss, rng=rng)
  ebsd = EBSD(np.c_[np.arange(6.0), np.zeros(6), np.zeros(6)], rotation.rand(6, rng=rng), np.ones(6, dtype=np.int8),
              phaseList(crystalFrame("m-3m", mineral="test")), {"mad": np.arange(6.0)})
  r = vector3d.rand(5, rng=rng)
  pf = PoleFigure(Miller(1, 0, 0, cs), r, np.ones(5))
  return {
    "MtexArray": (vector3d.rand(2, rng=rng), vector3d.rand(3, rng=rng)),
    "rotation": (rotation.rand(2, rng=rng), rotation.rand(3, rng=rng)),
    "tensor": (tensor.rand(2, rank=2, rng=rng), tensor.rand(3, rank=2, rng=rng)),
    "SO3TangentVector": (log(o, o[0]), log(o[1:], o[0])),
    "fibre": (fibre(o[:2], o[1:]), fibre(o[:1], o[2:])),
    "slipSystem": (slipSystem.basal(crystalFrame("6/mmm")), slipSystem.prismaticA(crystalFrame("6/mmm"))),
    "S2FunHarmonic": (S2FunHarmonic(rng.standard_normal(9)), S2FunHarmonic(rng.standard_normal(16))),
    "SO3FunHarmonic": (SO3FunHarmonic(rng.standard_normal(10), cs, ss), SO3FunHarmonic(rng.standard_normal(35), cs, ss)),
    "PoleFigure": (pf, pf),
    "EBSD": (ebsd[:2], ebsd[3:]),
  }


# the same object: the class, and the attributes equal, arrays by their entries
def _same(a, b):
  if isinstance(a, (np.ndarray, MtexArray)):
    return type(a) is type(b) and np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True) and \
      (not isinstance(a, MtexArray) or _same(vars(a), vars(b)))
  if isinstance(a, dict):
    return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
  if isinstance(a, (list, tuple)):
    return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
  if hasattr(a, "__dict__") and type(a).__module__.startswith("mtex") and not isinstance(a, type):
    return type(a) is type(b) and (a is b or _same(vars(a), vars(b)))
  return a is b or a == b


def test_cat_flat_joinsEveryClassAsItsOwnCat():
  import inspect
  import mtex
  owners = {getattr(mtex, n).cat.__qualname__.split(".")[0] for n in dir(mtex)
            if inspect.isclass(getattr(mtex, n)) and hasattr(getattr(mtex, n), "cat")}
  operands = _catOperands()
  assert owners == set(operands)
  for a, b in operands.values():
    assert _same(cat(a, b), type(a).cat(a, b))
