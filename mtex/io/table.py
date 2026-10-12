# the text of a vendor file: its bytes, its lines, its numeric block and its header pairs
#
#   data = readFile(path)                         the bytes, read once
#   lines, offset = leadingLines(data, stop)     the header lines, and where the block begins
#   lines, offset = linesFrom(data, offset, k)    the next k lines
#   M = readTable(data, offset, comments='#')     the numeric block as an (n, k) float array
#   header = headerDict(pairs)                    the flat header MATLAB keeps in ebsd.opt.header
#   writeTable(fname, names, cols)                columns under their names, as MATLAB's cprintf
#   cprintf(d, '-Lc', names)                      the same table printed
#
# The numeric block goes to the row parser of kernels.py as it stands: a decimal comma is
# decided once for the block, and a row of the wrong width names itself. Nothing here
# knows a format.
from __future__ import annotations

import itertools
import re

import numpy as np

from .kernels import _firstRows, parseBlock


def readFile(path):
  """the bytes of a file, read once"""
  with open(path, "rb") as f:
    return f.read()


def text(line):
  """a vendor file is ascii; a mineral name in latin-1 is read as such rather than failing"""
  return line.decode("ascii", errors="replace") if isinstance(line, bytes) else line


def readTable(data, offset=0, comments=None, delimiter=None):
  """the numeric block that begins at offset, one row per line, as an (n, k) array"""
  block = data[offset:] if offset else data
  if delimiter is not None and delimiter.strip():
    block = block.replace(delimiter.encode(), b" ")
  comments = comments.encode() if isinstance(comments, str) else comments
  return parseBlock(block, _decimalComma(block, comments), comments)


# whether the block writes the decimal comma: read off its first eight rows, not per field
def _decimalComma(block, comments):
  return any(b"," in row for row in itertools.islice(_firstRows(block, comments), 8))


def leadingLines(data, stop):
  """the lines before the first one that stop accepts, and the offset of that line; the file
  is not cut into lines beyond the header, which for a map of millions is the whole cost"""
  lines = []
  pos, n = 0, len(data)
  while pos < n:
    nl = data.find(b"\n", pos)
    end = n if nl < 0 else nl
    line = data[pos:end].removesuffix(b"\r")
    if stop(line):
      return lines, pos
    lines.append(line)
    pos = end + 1
  return lines, n


def linesFrom(data, offset, count):
  """count lines from offset on, and the offset after them"""
  lines = []
  pos, n = offset, len(data)
  for _ in range(count):
    if pos >= n:
      break
    nl = data.find(b"\n", pos)
    end = n if nl < 0 else nl
    lines.append(data[pos:end].removesuffix(b"\r"))
    pos = min(end + 1, n)
  return lines, pos


# ------------------------------------------------------------------ the header

def coerceNumeric(value):
  """a header value as MATLAB's coerceNumeric reads it: one number where it is one, the text otherwise"""
  s = value.strip()
  if not s:
    return s
  try:
    return float(s)
  except ValueError:
    return s


def validName(key):
  """the field name MATLAB's makeValidName gives a header key"""
  name = re.sub(r"[^A-Za-z0-9_]", "_", key.strip())
  if not name:
    return ""
  if not re.match(r"[A-Za-z]", name):
    name = "x" + name
  return name


def headerDict(pairs):
  """the flat header of the loaders, MATLAB's buildHeaderStruct: valid names, numbers where they are"""
  header = {}
  for key, value in pairs:
    name = validName(key)
    if name:
      header[name] = coerceNumeric(value)
  return header


# whether a line is a row of numbers
_NUMBER = re.compile(rb"^[\s]*[-+]?(\d+[.,]?\d*|[.,]\d+)([eE][-+]?\d+)?([\s,;]+[-+]?(\d+[.,]?\d*|[.,]\d+)([eE][-+]?\d+)?)*[\s]*$")


def isNumericRow(line):
  """whether a line is a row of numbers, a decimal comma allowed"""
  return bool(line.strip()) and _NUMBER.match(line) is not None


# the lines above the numeric block of a table and the offset of the block: the first header lines
# where their number is given, else the lines before the first row of numbers, a delimiter read as
# blank space; None for the offset where no row of numbers follows
def _tableStart(data, header=None, delimiter=None):
  if header is not None:
    return linesFrom(data, 0, int(header))
  sep = delimiter.encode() if delimiter is not None and delimiter.strip() else None
  head, offset = leadingLines(data, lambda l: isNumericRow(l if sep is None else l.replace(sep, b" ")))
  return head, None if offset >= len(data) else offset


# a column name in lower case, its runs of blank space one blank
def _canonicalName(name):
  return re.sub(r"\s+", " ", str(name).strip().lower())


def tableLines(names, cols, rowNames=None, uniform=False):
  """columns under a line of their names, six significant digits, every column as wide as its
  widest entry, as MATLAB's cprintf writes a file; row names in a first column, one width
  for all columns as cprintf prints"""
  cells = [[f"{x:g}" for x in np.asarray(c, dtype=float).reshape(-1)] for c in cols]
  if names is None:
    names = [""] * len(cells)
  if rowNames is not None:
    names, cells = [""] + list(names), [list(rowNames)] + cells
  widths = [max(len(n), *(len(s) for s in c)) for n, c in zip(names, cells)]
  if uniform:
    widths = widths[:1] + [max(widths[1:])] * (len(widths) - 1) if rowNames is not None else [max(widths)] * len(widths)
  lines = [" ".join(n.rjust(w) for n, w in zip(names, widths))] if any(names) else []
  for row in zip(*cells):
    lines.append(" ".join(s.rjust(w) for s, w in zip(row, widths)))
  return lines


def writeTable(fname, names, cols):
  """writes columns under a line of their names to a text file, as MATLAB's cprintf writes a table"""
  with open(fname, "w", encoding="utf-8") as f:
    f.write("\n".join(tableLines(names, cols)) + "\n")


def cprintf(d, *args, Lc=None, Lr=None):
  """MATLAB's cprintf(d, '-Lc', labels, '-Lr', rows): a matrix printed as a table, a vector as
  one row"""
  opts = dict(zip(args[::2], args[1::2]))
  Lc, Lr = opts.get("-Lc", Lc), opts.get("-Lr", Lr)
  d = np.atleast_2d(np.asarray(d, dtype=float))
  print("\n".join(tableLines(Lc, d.T, Lr, uniform=True)))


__all__ = ["cprintf"]
