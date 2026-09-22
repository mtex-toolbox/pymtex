# the text of a vendor file: its bytes, its lines, its numeric block and its header pairs
#
#   data = readFile(path)                         the bytes, read once
#   lines, offset = leadingLines(data, stop)     the header lines, and where the block begins
#   lines, offset = linesFrom(data, offset, k)    the next k lines
#   M = readTable(data, offset, comments='#')     the numeric block as an (n, k) float array
#   header = headerDict(pairs)                    the flat header MATLAB keeps in ebsd.opt.header
#   writeTable(fname, names, cols)                columns under their names, as MATLAB's cprintf
#
# The numeric block goes to the row parser of kernels.py as it stands: a decimal comma is
# decided once for the block, and a row of the wrong width names itself. Nothing here
# knows a format.
from __future__ import annotations

import re

import numpy as np

from .kernels import parseBlock


def readFile(path):
  with open(path, "rb") as f:
    return f.read()


# a vendor file is ascii; a mineral name in latin-1 is read as such rather than failing
def text(line):
  return line.decode("ascii", errors="replace") if isinstance(line, bytes) else line


# the numeric block that begins at offset, one row per line, as an (n, k) array
def readTable(data, offset=0, comments=None, delimiter=None):
  block = data[offset:] if offset else data
  if delimiter is not None and delimiter.strip():
    block = block.replace(delimiter.encode(), b" ")
  comments = comments.encode() if isinstance(comments, str) else comments
  return parseBlock(block, _decimalComma(block, comments), comments)


# whether the block writes the decimal comma: read off its first rows, not per field
def _decimalComma(block, comments):
  n = 0
  for line in block[:65536].split(b"\n"):
    s = line.strip()
    if not s or (comments is not None and s.startswith(comments)):
      continue
    if b"," in s:
      return True
    n += 1
    if n >= 8:
      break
  return False


# the lines before the first one that stop accepts, and the offset of that line; the file
# is not cut into lines beyond the header, which for a map of millions is the whole cost
def leadingLines(data, stop):
  lines = []
  pos, n = 0, len(data)
  while pos < n:
    nl = data.find(b"\n", pos)
    end = n if nl < 0 else nl
    line = data[pos:end]
    if line.endswith(b"\r"):
      line = line[:-1]
    if stop(line):
      return lines, pos
    lines.append(line)
    pos = end + 1
  return lines, n


# count lines from offset on, and the offset after them
def linesFrom(data, offset, count):
  lines = []
  pos, n = offset, len(data)
  for _ in range(count):
    if pos >= n:
      break
    nl = data.find(b"\n", pos)
    end = n if nl < 0 else nl
    line = data[pos:end]
    lines.append(line[:-1] if line.endswith(b"\r") else line)
    pos = min(end + 1, n)
  return lines, pos


# ------------------------------------------------------------------ the header

# a header value as MATLAB's coerceNumeric reads it: one number where it is one, the text otherwise
def coerceNumeric(value):
  s = value.strip()
  if not s:
    return s
  try:
    return float(s)
  except ValueError:
    return s


# the field name MATLAB's makeValidName gives a header key
def validName(key):
  name = re.sub(r"[^A-Za-z0-9_]", "_", key.strip())
  if not name:
    return ""
  if not re.match(r"[A-Za-z]", name):
    name = "x" + name
  return name


# the flat header of the loaders, MATLAB's buildHeaderStruct: valid names, numbers where they are
def headerDict(pairs):
  header = {}
  for key, value in pairs:
    name = validName(key)
    if name:
      header[name] = coerceNumeric(value)
  return header


# whether a line is a row of numbers
_NUMBER = re.compile(rb"^[\s]*[-+]?(\d+[.,]?\d*|[.,]\d+)([eE][-+]?\d+)?([\s,;]+[-+]?(\d+[.,]?\d*|[.,]\d+)([eE][-+]?\d+)?)*[\s]*$")


def isNumericRow(line):
  return bool(line.strip()) and _NUMBER.match(line) is not None


__all__ = []


# columns under a line of their names, six significant digits, every column as wide as its
# widest entry, as MATLAB's cprintf writes a table
def writeTable(fname, names, cols):
  cells = [[f"{x:g}" for x in np.asarray(c, dtype=float).reshape(-1)] for c in cols]
  widths = [max(len(n), *(len(s) for s in c)) for n, c in zip(names, cells)]
  lines = [" ".join(n.rjust(w) for n, w in zip(names, widths))]
  for row in zip(*cells):
    lines.append(" ".join(s.rjust(w) for s, w in zip(row, widths)))
  with open(fname, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
