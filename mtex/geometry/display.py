# the text layout every class displays with: MATLAB's displayClass and cprintf
import numpy as np


# a size as MATLAB writes it, 1 x 3, and 1 for a single element
def sizeString(shape):
  return " x ".join(str(n) for n in shape) if len(shape) else "1"


# a number at three significant digits, integers without a decimal point, tiny as zero
def numString(x, digits=3, tiny=1e-10, decimals=None):
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


# columns under their labels, right aligned, two spaces of indent
def table(labels, columns, digits=3, tiny=1e-10, decimals=None):
  cells = [[numString(x, digits, tiny, decimals) for x in np.asarray(c).ravel()] for c in columns]
  widths = [max(len(l), *(len(s) for s in col)) for l, col in zip(labels, cells)]
  lines = ["  " + "  ".join(l.rjust(w) for l, w in zip(labels, widths))]
  for row in zip(*cells):
    lines.append("  " + "  ".join(s.rjust(w) for s, w in zip(row, widths)))
  return "\n".join(lines)


# the header of a display: the class name and, when given, a note on the same line
def header(obj, info=""):
  name = type(obj).__name__
  return f"{name} ({info})" if info else name


__all__ = []
