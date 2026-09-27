# the text layout every class displays with, MATLAB's displayClass: the class name with the
# frames in parentheses, a block of aligned key: value lines, then the data as a table
import numpy as np


def sizeString(shape):
  """a size as MATLAB writes it, 2 × 3, and 1 for a single element"""
  return " × ".join(str(n) for n in shape) if len(shape) else "1"


def numString(x, digits=3, tiny=1e-10, decimals=None):
  """a number at three significant digits, integers without a decimal point, tiny as zero"""
  x = float(x)
  if x != x:
    return "NaN"
  if abs(x) < tiny:
    x = 0.0
  # a fixed number of decimals, trailing zeros dropped, where MATLAB rounds rather
  # than counts significant digits
  if decimals is not None:
    x = round(x, decimals)
  if x == int(x) and abs(x) < 1e15:
    return str(int(x))
  if decimals is not None:
    return f"{x:.{decimals}f}".rstrip("0")
  return f"{x:.{digits}g}"


def xnum2str(x):
  """a number as MATLAB's xnum2str writes it: the decimals follow the size, trailing zeros
  and a trailing point dropped, 27532, 1.3, 0.25"""
  x = float(x)
  if x != x:
    return "NaN"
  m = abs(x)
  if m > 1e6 or m <= 1e-4:
    s = f"{x:.2g}"
  else:
    decimals = 0 if m >= 10 else 1 if m >= 1 else 2 if m > 0.1 else 3 if m > 0.01 else 4 if m > 0.001 else 5
    # half away from zero, as MATLAB rounds
    s = f"{np.copysign(np.floor(m * 10 ** decimals + 0.5) / 10 ** decimals, x):.{decimals}f}"
    if decimals:
      s = s.rstrip("0").rstrip(".")
  return "0" if s == "-0" else s


def shownRows(n):
  """the rows of n a display shows: all up to the preference displayRows, else the first and
  the last quarter of it, None standing for the ellipsis between"""
  from .session import getMTEXpref
  limit = getMTEXpref("displayRows", 20)
  if n <= limit:
    return list(range(n))
  k = max(limit // 4, 1)
  return list(range(k)) + [None] + list(range(n - k, n))


def takeRows(n):
  """the indices of the rows a display of n shows, and their positions in the table with
  None for the ellipsis: a long array computes the coordinates of those rows only"""
  rows = shownRows(n)
  take = [k for k in rows if k is not None]
  pos, j = [], 0
  for k in rows:
    pos.append(None if k is None else j)
    j += k is not None
  return take, pos


def table(labels, columns, digits=3, tiny=1e-10, decimals=None, rows=None):
  """columns under their labels, right aligned, two spaces of indent; a long table keeps its
  first and last rows around a vertical ellipsis"""
  columns = [np.asarray(c).ravel() for c in columns]
  n = len(columns[0]) if columns else 0
  rows = shownRows(n) if rows is None else rows
  cells = [[numString(c[k], digits, tiny, decimals) if k is not None else "⋮" for k in rows] for c in columns]
  return textTable(labels, cells)


def textTable(labels, cells):
  """columns of text under their labels, right aligned, two spaces of indent"""
  widths = [max([len(l)] + [len(s) for s in col]) for l, col in zip(labels, cells)]
  lines = ["  " + "  ".join(l.rjust(w) for l, w in zip(labels, widths))]
  for row in zip(*cells):
    lines.append("  " + "  ".join(s.rjust(w) for s, w in zip(row, widths)))
  return "\n".join(line.rstrip() for line in lines)


def fields(pairs, indent="  "):
  """key: value lines, the colons aligned as MATLAB aligns them; a pair whose value is None is
  left out"""
  pairs = [(k, v) for k, v in pairs if v is not None]
  if not pairs:
    return []
  w = max(len(k) for k, _ in pairs)
  return [f"{indent}{k.ljust(w)}: {v}" for k, v in pairs]


_FIELD = None


def aligned(text):
  """a display with every run of key: value lines at one indent aligned on its colons"""
  global _FIELD
  if _FIELD is None:
    import re
    _FIELD = re.compile(r"^( +)([A-Za-z][^:=\n]{0,40}?) *: (.*)$")
  lines = text.split("\n")
  out, k = [], 0
  while k < len(lines):
    m = _FIELD.match(lines[k])
    if not m:
      out.append(lines[k])
      k += 1
      continue
    run = [m]
    while k + len(run) < len(lines):
      n = _FIELD.match(lines[k + len(run)])
      if not n or n.group(1) != m.group(1):
        break
      run.append(n)
    w = max(len(r.group(2)) for r in run)
    out += [f"{r.group(1)}{r.group(2).ljust(w)}: {r.group(3)}" for r in run]
    k += len(run)
  return "\n".join(out)


def frameLabel(fr):
  """a frame as a header names it: a crystal frame by its mineral, else its group, a specimen
  frame by the convention it is drawn in, which names its axes"""
  if fr is None:
    return ""
  from .frames import crystalFrame
  if isinstance(fr, crystalFrame):
    return fr.char()
  return fr.conventionChar() or fr.char()


def framesLabel(a, b):
  """two frames as a header names them, crystal → specimen"""
  return f"{frameLabel(a)} → {frameLabel(b)}"


def header(name, *info):
  """the first line of a display: the class name and, in parentheses, what it lives in"""
  info = [i for i in info if i]
  return f"{name} ({', '.join(info)})" if info else name


def display(head, pairs=(), *blocks):
  """a whole display: the header, the aligned fields, then the blocks, a table say"""
  return "\n".join([head] + fields(pairs) + [b for b in blocks if b])


__all__ = []
