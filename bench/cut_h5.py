# a small HDF5 fixture out of a vendor file: the whole tree, the per pixel data sets cut to
# the first rows of the scan, and the big arrays that are not per pixel left out
#
#   .venv/bin/python bench/cut_h5.py source.h5 target.h5 nPixels
#
# A file cut this way keeps every header, every phase and every name a config resolves, so
# the reader is tested on the vendor's real layout at a few hundred kilobytes.
import sys

import h5py
import numpy as np


def cut(source, target, n, rows=None):
  with h5py.File(source, "r") as src, h5py.File(target, "w") as dst:
    for k, v in src.attrs.items():
      dst.attrs[k] = v
    total = None

    def visit(name, obj):
      nonlocal total
      if isinstance(obj, h5py.Group):
        g = dst.create_group(name)
        for k, v in obj.attrs.items():
          g.attrs[k] = v
        return
      shape = obj.shape
      if total is None and len(shape) >= 1 and shape[0] > 1000 and obj.dtype.kind in "fiuV" and (len(shape) == 1 or shape[1] <= 32):
        total = shape[0]
      if len(shape) >= 1 and total is not None and shape[0] == total:
        data = obj[:n]
      elif rows is not None and len(shape) >= 2 and shape[0] == rows[0] and shape[1] == rows[1]:
        data = obj[:max(1, n // rows[1])]
      elif obj.size * obj.dtype.itemsize > 50_000:
        return
      else:
        data = obj[()]
      if hasattr(data, "size") and data.size * data.dtype.itemsize > 500_000:
        return
      try:
        d = dst.create_dataset(name, data=data) if obj.dtype.names else dst.create_dataset(name, data=data, dtype=obj.dtype)
      except (OSError, TypeError, ValueError):
        try:
          d = dst.create_dataset(name, data=data)
        except (OSError, TypeError, ValueError) as e:
          print("left out", name, obj.dtype, e)
          return
      for k, v in obj.attrs.items():
        d.attrs[k] = v

    src.visititems(visit)


if __name__ == "__main__":
  cut(sys.argv[1], sys.argv[2], int(sys.argv[3]), tuple(int(v) for v in sys.argv[4].split("x")) if len(sys.argv) > 4 else None)
