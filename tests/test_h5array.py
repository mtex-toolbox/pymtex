# a whole HDF5 data set read over the cores, against h5py's own read
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mtex import setMTEXpref
from mtex.io.h5array import _pieces, readArray

# the ways a file stores an array: the keywords of create_dataset, and whether the threads read it
STORED = {
  "contiguous": ({}, True),
  "chunks": ({"chunks": (3, 5, 4, 2)}, True),
  "gzip": ({"chunks": (1, 7, 9, 2), "compression": "gzip"}, True),
  "gzip over the edge": ({"chunks": (4, 4, 4, 2), "compression": "gzip"}, True),
  "shuffle and gzip": ({"chunks": (2, 7, 5, 2), "compression": "gzip", "shuffle": True}, True),
  "lzf": ({"chunks": (2, 7, 9, 2), "compression": "lzf"}, False),
  "checksum": ({"chunks": (2, 7, 9, 2), "fletcher32": True}, False),
}


@pytest.mark.parametrize("stored", STORED)
@pytest.mark.parametrize("dtype", ["<f4", "<f8", "<i4", "u1", ">f4"])
def test_readArray_isWhatH5pyReads(tmp_path, stored, dtype):
  a = (np.random.default_rng(0).random((6, 7, 9, 2)) * 200).astype(dtype)
  with h5py.File(tmp_path / "a.h5", "w") as f:
    f.create_dataset("a", data=a, **STORED[stored][0])
  with h5py.File(tmp_path / "a.h5", "r") as f:
    got = readArray(f["a"], least=0)
    assert got.dtype == f["a"].dtype and np.array_equal(got, f["a"][...]) and np.array_equal(got, a)


@pytest.mark.parametrize("stored", STORED)
def test_pieces_areReadByTheThreadsWhereTheBytesAreKnown(tmp_path, stored):
  from mtex.io.h5array import _filters
  with h5py.File(tmp_path / "a.h5", "w") as f:
    f.create_dataset("a", data=np.arange(6 * 7 * 9 * 2, dtype=np.float32).reshape(6, 7, 9, 2), **STORED[stored][0])
  with h5py.File(tmp_path / "a.h5", "r") as f:
    assert (_pieces(f["a"]) is not None and _filters(f["a"]) is not None) == (STORED[stored][1] and f.driver == "sec2")


def test_readArray_fillsTheChunksNeverWritten(tmp_path):
  with h5py.File(tmp_path / "a.h5", "w") as f:
    d = f.create_dataset("a", shape=(8, 6), dtype="f4", chunks=(2, 6), compression="gzip", fillvalue=7)
    d[2:4] = 1
  with h5py.File(tmp_path / "a.h5", "r") as f:
    assert np.array_equal(readArray(f["a"], least=0), f["a"][...])


def test_readArray_onOneThreadIsH5pys(tmp_path):
  a = np.arange(60.0).reshape(3, 4, 5)
  with h5py.File(tmp_path / "a.h5", "w") as f:
    f.create_dataset("a", data=a, chunks=(1, 4, 5), compression="gzip")
  setMTEXpref("threads", 1)
  try:
    with h5py.File(tmp_path / "a.h5", "r") as f:
      assert np.array_equal(readArray(f["a"], least=0), a)
  finally:
    setMTEXpref("threads", None)
