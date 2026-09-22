# the numeric block of a text file parsed by rows in parallel, the io hot loop
#
#   M = parseBlock(block, comma, comments)    an (n, k) float array, one row per line of numbers
#
# The numba loop scans every row over its bytes and reads each field where it lies, so no
# string is made for a field and the rows run in parallel; a field is an integer mantissa
# and a power of ten, one rounding up to fifteen significant digits, as strtod is, and an
# ulp or two beyond. The NumPy form is np.loadtxt on the same
# bytes, the fallback without numba and the oracle of the tests. The decimal comma is
# decided by the caller, once for the block.
from __future__ import annotations

import io

import numpy as np

from ..geometry.kernels import _jit, backend

prange = range
_COMPILED = {}
# the powers of ten as the nearest doubles, exact up to 1e22
POW10 = np.array([float(f"1e{k}") for k in range(309)])
NEGPOW10 = np.array([float(f"1e-{k}") for k in range(309)])


def _parseLoop(buf, starts, ends, ncols, comma, comment, out, got):
  for r in prange(starts.shape[0]):
    k = starts[r]
    e = ends[r]
    j = 0
    bad = False
    while k < e:
      c = buf[k]
      if c == 32 or c == 9 or c == 13:
        k += 1
        continue
      if c == comment and j == 0:
        break
      # nan spelled out, as a reader that could not index writes it
      if (c == 110 or c == 78) and k + 2 < e and (buf[k + 1] == 97 or buf[k + 1] == 65) and (buf[k + 2] == 110 or buf[k + 2] == 78):
        k += 3
        if k < e and not (buf[k] == 32 or buf[k] == 9 or buf[k] == 13):
          bad = True
          break
        if j < ncols:
          out[r, j] = np.nan
        j += 1
        continue
      sign = 1.0
      if c == 45:
        sign = -1.0
        k += 1
      elif c == 43:
        k += 1
      mant = 0
      scale = 0
      ndig = 0
      seen = False
      # eighteen significant digits, what an int64 mantissa holds; the value is one rounding
      # up to fifteen of them, since the mantissa is then below 2^53
      while k < e and 48 <= buf[k] <= 57:
        d = buf[k] - 48
        if mant == 0 and d == 0:
          pass
        elif ndig < 18:
          mant = mant * 10 + d
          ndig += 1
        else:
          scale += 1
        k += 1
        seen = True
      if k < e and (buf[k] == 46 or (comma and buf[k] == 44)):
        k += 1
        while k < e and 48 <= buf[k] <= 57:
          d = buf[k] - 48
          if mant == 0 and d == 0:
            scale -= 1
          elif ndig < 18:
            mant = mant * 10 + d
            ndig += 1
            scale -= 1
          k += 1
          seen = True
      if not seen:
        bad = True
        break
      if k < e and (buf[k] == 101 or buf[k] == 69):
        k += 1
        es = 1
        ex = 0
        eseen = False
        if k < e and buf[k] == 45:
          es = -1
          k += 1
        elif k < e and buf[k] == 43:
          k += 1
        while k < e and 48 <= buf[k] <= 57:
          ex = ex * 10 + (buf[k] - 48)
          k += 1
          eseen = True
        if not eseen:
          bad = True
          break
        scale += es * ex
      if k < e and not (buf[k] == 32 or buf[k] == 9 or buf[k] == 13):
        bad = True
        break
      v = float(mant)
      if scale < 0:
        v = v / POW10[-scale] if scale > -23 else (v * NEGPOW10[-scale] if scale > -309 else 0.0)
      elif scale > 0:
        v = v * POW10[scale] if scale < 309 else v * np.inf
      if j < ncols:
        out[r, j] = sign * v
      j += 1
    got[r] = -1 if bad else j


def _compile():
  nb = _jit()
  if not nb:
    return None
  if "parse" not in _COMPILED:
    globals()["prange"] = nb.prange
    _COMPILED["parse"] = nb.njit(cache=True, parallel=True, nogil=True)(_parseLoop)
  return _COMPILED["parse"]


# the number of fields in the first row of numbers
def _firstWidth(block, comments):
  for line in block[:65536].split(b"\n"):
    s = line.strip()
    if s and not (comments is not None and s.startswith(comments)):
      return len(s.split())
  raise ValueError("the file holds no data rows")


def parseBlockNumba(block, comma, comments):
  loop = _compile()
  if loop is None:
    return None
  ncols = _firstWidth(block, comments)
  buf = np.frombuffer(block, dtype=np.uint8)
  nl = np.flatnonzero(buf == 10)
  starts = np.concatenate([[0], nl + 1]).astype(np.int64)
  ends = np.concatenate([nl, [len(buf)]]).astype(np.int64)
  out = np.empty((len(starts), ncols))
  got = np.empty(len(starts), dtype=np.int32)
  loop(buf, starts, ends, ncols, comma, comments[0] if comments else -1, out, got)
  keep = got != 0
  if not keep.all():
    n = int(np.flatnonzero(keep)[-1]) + 1 if keep.any() else 0
    if keep[:n].all():
      out, got = out[:n], got[:n]
    else:
      out, got = out[keep], got[keep]
  if out.shape[0] == 0:
    raise ValueError("the file holds no data rows")
  bad = np.flatnonzero(got != ncols)
  if bad.size:
    r = int(bad[0])
    row = int(np.flatnonzero(keep)[r]) + 1
    if got[r] < 0:
      raise ValueError(f"row {row} of the numeric block is not numeric")
    raise ValueError(f"row {row} has {got[r]} columns, expected {ncols}")
  return out


def parseBlockNumpy(block, comma, comments):
  if comma:
    block = block.replace(b",", b".")
  try:
    M = np.loadtxt(io.BytesIO(block), comments=comments.decode() if comments else None, ndmin=2)
  except ValueError as e:
    raise ValueError(f"the numeric block could not be read: {e}") from None
  if M.size == 0:
    raise ValueError("the file holds no data rows")
  return M


def parseBlock(block, comma=False, comments=None):
  if backend() == "numba":
    M = parseBlockNumba(block, comma, comments)
    if M is not None:
      return M
  return parseBlockNumpy(block, comma, comments)


__all__ = []


# ------------------------------------------------------------------ writing a numeric block

# the bytes of one row of numbers: every column to a fixed number of decimals, or as an
# integer where the decimals are negative; two passes, one to size every row and one to
# write them in parallel, since a row's width is what its digits make it
def _rowLengths(cols, decimals, lengths):
  n, m = cols.shape
  for r in prange(n):
    total = m - 1
    for j in range(m):
      total += _numberWidth(cols[r, j], decimals[j])
    lengths[r] = total


def _numberWidth(v, d):
  if v != v:
    return 3
  if d < 0:
    r = int(np.rint(v))
    w = 1 if r < 0 else 0
    a = -r if r < 0 else r
    while True:
      w += 1
      a //= 10
      if a == 0:
        return w
  r = int(np.rint(v * POW10[d]))
  w = 1 if np.signbit(v) else 0
  a = -r if r < 0 else r
  ip = a // int(POW10[d])
  while True:
    w += 1
    ip //= 10
    if ip == 0:
      break
  return w + (d + 1 if d > 0 else 0)


def _writeNumber(out, k, v, d):
  if v != v:
    out[k] = 110
    out[k + 1] = 97
    out[k + 2] = 110
    return k + 3
  if d < 0:
    r = int(np.rint(v))
    frac = 0
    scale = 1
  else:
    r = int(np.rint(v * POW10[d]))
    scale = int(POW10[d])
  # the sign of a negative that rounds to zero stays, as printf keeps it
  if r < 0 or (d >= 0 and np.signbit(v)):
    out[k] = 45
    k += 1
    r = -r
  ip = r // scale
  frac = r - ip * scale
  # the integer part, written backwards then reversed in place
  start = k
  while True:
    out[k] = 48 + ip % 10
    k += 1
    ip //= 10
    if ip == 0:
      break
  i, j = start, k - 1
  while i < j:
    t = out[i]
    out[i] = out[j]
    out[j] = t
    i += 1
    j -= 1
  if d > 0:
    out[k] = 46
    k += 1
    for p in range(d - 1, -1, -1):
      q = frac // int(POW10[p])
      out[k] = 48 + q
      frac -= q * int(POW10[p])
      k += 1
  return k


def _writeRows(cols, decimals, offsets, sep, eol, crlf, out):
  n, m = cols.shape
  for r in prange(n):
    k = offsets[r]
    for j in range(m):
      if j > 0:
        out[k] = sep
        k += 1
      k = _writeNumber(out, k, cols[r, j], decimals[j])
    if crlf:
      out[k] = 13
      k += 1
    out[k] = 10


def _compileWriter():
  nb = _jit()
  if not nb:
    return None
  if "writeRows" not in _COMPILED:
    globals()["prange"] = nb.prange
    for name in ("_numberWidth", "_writeNumber"):
      globals()[name] = nb.njit(cache=True)(globals()[name])
    _COMPILED["rowLengths"] = nb.njit(cache=True, parallel=True)(_rowLengths)
    _COMPILED["writeRows"] = nb.njit(cache=True, parallel=True)(_writeRows)
  return _COMPILED["writeRows"]


def formatTableNumba(cols, decimals, sep=b" ", crlf=False):
  writer = _compileWriter()
  if writer is None:
    return None
  cols = np.ascontiguousarray(cols, dtype=np.float64)
  decimals = np.asarray(decimals, dtype=np.int64)
  n = cols.shape[0]
  lengths = np.empty(n, dtype=np.int64)
  _COMPILED["rowLengths"](cols, decimals, lengths)
  lengths += 2 if crlf else 1
  offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
  out = np.empty(int(lengths.sum()), dtype=np.uint8)
  writer(cols, decimals, offsets, sep[0], 10, crlf, out)
  return out.tobytes()


def formatTableNumpy(cols, decimals, sep=b" ", crlf=False):
  cols = np.asarray(cols, dtype=np.float64)
  fmt = ["%d" if d < 0 else f"%.{d}f" for d in decimals]
  buf = io.BytesIO()
  rounded = cols.copy()
  for j, d in enumerate(decimals):
    if d < 0:
      rounded[:, j] = np.rint(rounded[:, j])
  np.savetxt(buf, rounded, fmt=fmt, delimiter=sep.decode(), newline="\r\n" if crlf else "\n")
  return buf.getvalue()


# the rows of a table as text, the columns to the decimals given, a negative count an integer
def formatTable(cols, decimals, sep=b" ", crlf=False):
  if backend() == "numba":
    b = formatTableNumba(cols, decimals, sep, crlf)
    if b is not None:
      return b
  return formatTableNumpy(cols, decimals, sep, crlf)
