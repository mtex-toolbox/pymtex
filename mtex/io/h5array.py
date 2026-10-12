# a whole HDF5 data set read over the cores
#
#   a = readArray(ds)          the array ds[...] gives
#
# HDF5 reads and inflates under one lock, a chunk after the other: the 13 million voxels of a
# gzip compressed volume took 2.8 s an array. The bytes of a data set lie in the file where its
# chunk index says, so the threads read them there themselves (pread) and inflate them (zlib
# releases the GIL), 0.3 s. That holds for a contiguous data set and for chunks stored plain,
# deflated, or shuffled and deflated, in the machine's byte order; whatever else a file does
# (another filter, external storage, a user block, a driver other than the plain file) is read
# by h5py, the fallback and the oracle of the tests.
from __future__ import annotations

import os
import zlib

import numpy as np

from ..geometry.kernels import threads

# below this many bytes one h5py read is faster than the start of the threads
PARALLEL_BYTES = 1 << 22
# a contiguous data set is read in blocks of this many bytes, one per task
BLOCK_BYTES = 1 << 24
# HDF5's filter ids
DEFLATE, SHUFFLE = 1, 2


# the stored pieces of a data set as (offset in the file, bytes there, index of the first cell,
# filter mask), or None where h5py has to read it
def _pieces(ds):
  import h5py
  if ds.dtype.kind not in "iuf" or not ds.dtype.isnative or ds.file.driver != "sec2" or ds.file.userblock_size:
    return None
  plist = ds.id.get_create_plist()
  layout = plist.get_layout()
  if layout == h5py.h5d.CONTIGUOUS:
    offset = ds.id.get_offset()
    if offset is None or plist.get_external_count():
      return None
    return [(offset, ds.size * ds.dtype.itemsize, None, 0)]
  if layout != h5py.h5d.CHUNKED or not hasattr(ds.id, "chunk_iter"):
    return None
  pieces = []
  ds.id.chunk_iter(lambda c: pieces.append((c.byte_offset, c.size, c.chunk_offset, c.filter_mask)))
  return pieces


# the filters of a data set as (shuffled, deflated), or None for a pipeline read by h5py
def _filters(ds):
  plist = ds.id.get_create_plist()
  ids = [plist.get_filter(k)[0] for k in range(plist.get_nfilters())]
  if ids in ([], [DEFLATE], [SHUFFLE, DEFLATE]):
    return SHUFFLE in ids, DEFLATE in ids
  return None


# bytes read at an offset into a writable buffer, whatever the system returns per call
def _readInto(fd, buf, offset):
  view = memoryview(buf).cast("B")
  done = 0
  while done < len(view):
    n = os.preadv(fd, [view[done:]], offset + done)
    if n <= 0:
      raise OSError("the file ends inside a data set")
    done += n


def readArray(ds, least=PARALLEL_BYTES):
  """the whole of an HDF5 data set as h5py's ds[...] gives it, its stored bytes read and inflated
  over the threads of the preference where the data set holds at least `least` bytes"""
  nbytes = ds.size * ds.dtype.itemsize
  pieces = _pieces(ds) if nbytes >= least and threads() > 1 and hasattr(os, "preadv") else None
  filters = _filters(ds) if pieces is not None else None
  # an unwritten chunk is the fill value, a chunk some filter skipped is not what the pipeline says
  if filters is None or any(p[3] for p in pieces) or (pieces[0][2] is not None and _cells(ds, pieces) != ds.size):
    return ds[...]
  shuffled, deflated = filters
  out = np.empty(ds.shape, dtype=ds.dtype)
  flat = out.reshape(-1).view(np.uint8)
  item = ds.dtype.itemsize
  shape, chunks, dtype = ds.shape, ds.chunks, ds.dtype
  chunkBytes = int(np.prod(chunks)) * item if chunks else 0
  fd = os.open(ds.file.filename, os.O_RDONLY)

  # a block of a contiguous data set, read where it belongs
  def block(a, b):
    _readInto(fd, flat[a:b], pieces[0][0] + a)

  # a chunk read, inflated and unshuffled, and the part of it inside the data set put in its place
  def chunk(piece):
    offset, size, first, _ = piece
    raw = bytearray(size)
    _readInto(fd, raw, offset)
    if deflated:
      raw = zlib.decompress(raw, bufsize=chunkBytes)
    cells = np.frombuffer(raw, dtype=np.uint8)
    if shuffled and item > 1:
      cells = np.ascontiguousarray(cells.reshape(item, -1).T).reshape(-1)
    cells = cells.view(dtype).reshape(chunks)
    inside = tuple(slice(o, min(o + c, n)) for o, c, n in zip(first, chunks, shape))
    out[inside] = cells[tuple(slice(0, s.stop - s.start) for s in inside)]

  from concurrent.futures import ThreadPoolExecutor
  try:
    with ThreadPoolExecutor(threads()) as pool:
      if pieces[0][2] is None:
        edges = list(range(0, nbytes, BLOCK_BYTES)) + [nbytes]
        tasks = [pool.submit(block, a, b) for a, b in zip(edges[:-1], edges[1:])]
      else:
        tasks = [pool.submit(chunk, p) for p in pieces]
      for t in tasks:
        t.result()
  finally:
    os.close(fd)
  return out


# the cells the stored chunks cover inside the data set
def _cells(ds, pieces):
  first = np.array([p[2] for p in pieces], dtype=np.int64)
  return int(np.prod(np.minimum(first + ds.chunks, ds.shape) - first, axis=1).sum())


__all__ = ["readArray"]
