# the layout of an array of grid cells, MATLAB gridLayout and layoutIndex
import numpy as np
import pytest

from mtex import gridLayout, layoutIndex, xvector, yvector, zvector


def test_the_named_layouts():
  cm, rm = gridLayout.columnMajor(), gridLayout.rowMajor()
  assert cm.name == "columnMajor" and np.allclose(cm.row.data, [0, 1, 0]) and np.allclose(cm.col.data, [1, 0, 0])
  assert rm.name == "rowMajor" and np.allclose(rm.row.data, [1, 0, 0])
  assert gridLayout() == cm and gridLayout(xvector, yvector) == rm and cm != rm
  assert np.allclose(cm.basis.data[2], [0, 0, -1])
  with pytest.raises(ValueError):
    gridLayout(xvector, xvector + yvector)


def test_a_layout_is_reached_by_a_transpose_and_flips():
  cm = gridLayout.columnMajor()
  A = np.arange(6).reshape(2, 3)
  # the array stored column major; asked for row major it is transposed
  lin = layoutIndex(gridLayout.rowMajor(), (yvector, xvector), A.shape)
  assert np.array_equal(A.reshape(-1)[lin], A.T)
  # rows along -x and columns along y is a quarter turn
  lin = layoutIndex(gridLayout(-xvector, yvector), (yvector, xvector), A.shape)
  assert np.array_equal(A.reshape(-1)[lin], np.rot90(A))
  # the layout it is in leaves it
  assert np.array_equal(layoutIndex(cm, (yvector, xvector), A.shape), np.arange(6).reshape(2, 3))
  # a layout a permutation does not reach wants the nearest one asked for
  turned = (np.cos(0.3) * yvector + np.sin(0.3) * xvector, np.cos(0.3) * xvector - np.sin(0.3) * yvector)
  with pytest.raises(ValueError):
    layoutIndex(cm, turned, A.shape)
  assert np.array_equal(layoutIndex(cm, turned, A.shape, nearest=True), np.arange(6).reshape(2, 3))
  # a tie at 45 degrees keeps the order
  d = ((xvector + yvector) / np.sqrt(2), (yvector - xvector) / np.sqrt(2))
  assert layoutIndex(cm, d, A.shape, nearest=True).shape == (2, 3)
