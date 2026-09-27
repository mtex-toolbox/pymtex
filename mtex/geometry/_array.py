# the container base: one coordinate array of shape (..., k) plus carried attributes
#
# Every geometric class holds many entities in one object, as in MTEX: a vector3d is a
# point cloud, a rotation a list of rotations. The base gives them the array interface -
# shape, indexing that returns the same class, views, reshape, cat, iteration - and
# carries the attributes a subclass declares (flags, frames) through every one of these.
# A single element is the zero-dimensional container, never a separate scalar class.
#
# A subclass sets `_ncoord`, the trailing axis, and `_carried`, the attribute names copied
# from an object to every object derived from it; combining two operands is the
# subclass's business, through `_new`.
from __future__ import annotations

import numpy as np
from .display import aligned

_slice = slice(None)


class MtexArray:
  """the container base: one coordinate array of shape (..., k) plus the attributes a subclass
  carries through indexing, reshape and cat; a single element is the zero-dimensional container"""
  _ncoord: int = 1
  _carried: tuple[str, ...] = ()
  # a NumPy array on the left of an operator hands over to the container's reflected method
  __array_ufunc__ = None

  def __init__(self, data, dtype=None):
    data = np.asarray(data, dtype=dtype)
    if not np.issubdtype(data.dtype, np.floating):
      data = data.astype(np.float64)
    if data.ndim == 0 or data.shape[-1] != self._ncoord:
      raise ValueError(f"{type(self).__name__} needs a trailing axis of length {self._ncoord}, got shape {data.shape}")
    self.data = data

  # ------------------------------------------------------------------ derived objects

  # the same class with other data; carried attributes are copied unless overridden
  def _new(self, data, **attrs):
    obj = object.__new__(type(self))
    obj.data = data
    for name in self._carried:
      setattr(obj, name, attrs.pop(name, getattr(self, name)))
    if attrs:
      raise TypeError(f"{type(self).__name__} carries no attribute {', '.join(attrs)}")
    return obj

  def copy(self):
    """a copy of the container with its own coordinate array, the carried attributes kept"""
    return self._new(self.data.copy())

  def astype(self, dtype):
    """the same container over its coordinates converted to another dtype"""
    return self._new(self.data.astype(dtype))

  # ------------------------------------------------------------------ the array interface

  @property
  def shape(self):
    """the shape of the container, the leading axes without the coordinate axis"""
    return self.data.shape[:-1]

  @property
  def ndim(self):
    """the number of leading axes, zero for a single element"""
    return self.data.ndim - 1

  @property
  def size(self):
    """the number of elements, MATLAB's numel"""
    return self.data.size // self._ncoord

  @property
  def dtype(self):
    """the dtype of the coordinate array"""
    return self.data.dtype

  # the number of elements, MATLAB's numel, so that a single element has length one
  def __len__(self):
    return self.size

  @property
  def isempty(self):
    """whether the container holds no element"""
    return self.size == 0

  # the index applies to the leading axes; the coordinate axis is always taken whole
  def __getitem__(self, idx):
    idx = idx if isinstance(idx, tuple) else (idx,)
    return self._new(self.data[idx + (_slice,)])

  # storing an element must not change what the list carries: the frames have to join
  def __setitem__(self, idx, value):
    idx = idx if isinstance(idx, tuple) else (idx,)
    value = self._coerce(value)
    if isinstance(value, type(self)):
      self._join(value)
    self.data[idx + (_slice,)] = value.data

  def __iter__(self):
    if self.ndim == 0:
      raise TypeError(f"iteration over a single {type(self).__name__}")
    return (self[i] for i in range(self.shape[0]))

  def reshape(self, *shape):
    """the same elements in another shape of the leading axes, the coordinate axis kept

      v.reshape(2, 3)
      v.reshape((2, 3))"""
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return self._new(self.data.reshape(tuple(shape) + (self._ncoord,)))

  def flatten(self):
    """the elements as a list, reshape(-1)"""
    return self.reshape(-1)

  @property
  def T(self):
    """the transpose of the leading axes; a single element and a list are their own"""
    axes = tuple(range(self.ndim))[::-1] + (self.ndim,)
    return self._new(self.data.transpose(axes))

  # ------------------------------------------------------------------ combining

  # an operand of the same class, or something the subclass knows to convert
  def _coerce(self, other):
    if isinstance(other, type(self)):
      return other
    raise TypeError(f"expected {type(self).__name__}, got {type(other).__name__}")

  # the carried attributes two operands share; a subclass with a join rule overrides it
  def _join(self, other):
    attrs = {}
    for name in self._carried:
      a, b = getattr(self, name), getattr(other, name)
      if a is not b and a != b:
        raise ValueError(f"cannot combine {type(self).__name__} with different {name}")
      attrs[name] = a
    return attrs

  @classmethod
  def cat(cls, *arrays, axis=0):
    """the list of the operands, MATLAB's [a, b]; single elements are lists of one"""
    arrays = [a if a.ndim > 0 else a.reshape(1) for a in arrays]
    first = arrays[0]
    attrs = first._join(first)
    for a in arrays[1:]:
      attrs = first._new(first.data, **attrs)._join(first._coerce(a))
    axis = axis if axis >= 0 else axis - 1
    return first._new(np.concatenate([a.data for a in arrays], axis=axis), **attrs)

  # ------------------------------------------------------------------ display

  def __repr__(self):
    from .display import header, sizeString
    return aligned(header(type(self).__name__, sizeString(self.shape)))


def cat(*arrays, axis=0):
  """cat(a, b, ...) is type(a).cat(a, b, ...): the list of the operands, MATLAB's [a, b], the
  carried attributes joined; axis= the axis to concatenate along"""
  return type(arrays[0]).cat(*arrays, axis=axis)
