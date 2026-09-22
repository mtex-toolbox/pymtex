# reading tensors from the text files of the sample data (MATLAB tensor.load,
# interfaces/loadTensor_generic.m, loadTensor_P.m; docs/plans/io.md I7)
#
#   C = stiffnessTensor.load('Olivine1997PC.GPa', cs)      the trailing 6 x 6 block of numbers, the density where stated
#   P = tensor.load('Single_RH_quartz_poly.P', cs, rank=3, doubleConvention=True)
#   T = loadTensor(fname, cs, rank=2)
#
# A tensor file is text: lines of prose, a line of lattice parameters, a density where
# the file gives one, and the coefficients as the last block of numeric rows of one
# width. The block is the tensor, read as the class asks: a 6 x 6 block a Voigt matrix, a
# 3 x 6 block the Voigt table of a rank three, a 3 x 3 block a rank two.
from __future__ import annotations

import os
import re

import numpy as np

from ..tensors.tensor import complianceTensor, stiffnessTensor, tensor
from ..geometry.frames import crystalFrame
from .table import readFile, text

_NUMBERS = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# the numeric rows of the file, the fixed width columns of a .P file split at their width
def _numericRows(fname):
  rows = []
  for line in text(readFile(fname)).split("\n"):
    s = line.strip()
    if not s:
      continue
    nums = _NUMBERS.findall(s)
    if nums and re.fullmatch(r"[\s\d.eE+-]+", s):
      rows.append([float(x) for x in nums])
  return rows


# the trailing rows of one width: six rows of six for a rank four, three of six for a rank
# three, three of three for a rank two; the lattice line above a block has the same width
# and is not part of it
def _trailingBlock(rows, rank):
  if not rows:
    raise ValueError("the file holds no row of numbers")
  width = len(rows[-1])
  block = []
  for r in reversed(rows):
    if len(r) != width:
      break
    block.append(r)
  block = block[::-1]
  if width == 6:
    n = 3 if rank == 3 or len(block) < 6 else 6
  elif width == 3:
    n = 3
  else:
    n = len(block)
  return np.array(block[-n:])


def loadTensor(fname, *args, cls=None, **kwargs):
  """the tensor a text file holds: the last block of numeric rows, the density on the line
  after the word density, the crystal frame given positionally"""
  fname = os.fspath(fname)
  rows = _numericRows(fname)
  cls = tensor if cls is None else cls
  M = _trailingBlock(rows, kwargs.get("rank", cls._rank))
  lines = [l.strip().lower() for l in text(readFile(fname)).split("\n")]
  density = None
  for k, l in enumerate(lines):
    if l.startswith("density") and k + 1 < len(lines):
      nums = _NUMBERS.findall(lines[k + 1])
      if nums:
        density = float(nums[0])
        if "kg" in l:
          density = density / 1000
  if density is not None and "density" not in kwargs:
    kwargs["density"] = density
  if M.shape == (6, 6) and cls is tensor and "rank" not in kwargs:
    kwargs["rank"] = 4
  if M.shape == (3, 6) and cls is tensor and "rank" not in kwargs:
    kwargs["rank"] = 3
  return cls(M, *args, **kwargs)


def _install():
  tensor.load = classmethod(lambda cls, fname, *a, **kw: loadTensor(fname, *a, cls=cls, **kw))


_install()

__all__ = ["loadTensor"]
